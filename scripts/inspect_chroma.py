"""
scripts/inspect_chroma.py — Manual query tool for Checkpoint 1 verification.

Embeds a query string using nomic-embed-text, queries the ChromaDB collection,
and pretty-prints the top-k results with entity metadata and distance scores.

Usage:
    python scripts/inspect_chroma.py "Where was Albert Einstein born?"
    python scripts/inspect_chroma.py "What is the Eiffel Tower?" --top-k 3
    python scripts/inspect_chroma.py "famous scientist" --filter person
    python scripts/inspect_chroma.py --stats          # print collection stats only

Options:
    --top-k N       Number of results to return (default: 5)
    --filter TYPE   Filter by entity_type: 'person', 'place', or 'both' (default: both)
    --config PATH   Path to settings.yaml (default: config/settings.yaml)
    --stats         Print collection stats and manifest summary, then exit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.manifest import summary as manifest_summary  # noqa: E402
from src.rag.embeddings import embed_one  # noqa: E402
from src.rag.vector_store import (  # noqa: E402
    collection_count,
    get_client,
    get_collection,
    query,
)

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
DIM = "\033[2m"


def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: config not found: {p}", file=sys.stderr)
        sys.exit(1)
    with p.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _entity_colour(entity_type: str) -> str:
    return CYAN if entity_type == "person" else MAGENTA


def _print_stats(config: dict) -> None:
    chroma_path = config["chroma_path"]
    manifest_path = config["manifest_path"]

    client = get_client(chroma_path)
    col = get_collection(client)
    count = collection_count(col)

    print(f"\n{BOLD}Collection:{RESET} wiki_rag  |  {BOLD}Total chunks:{RESET} {count}")

    rows = manifest_summary(manifest_path)
    if not rows:
        print("Manifest is empty — run: python scripts/run_ingest.py --rebuild")
        return

    persons = [r for r in rows if r["entity_type"] == "person"]
    places = [r for r in rows if r["entity_type"] == "place"]

    print(
        f"\n{BOLD}Manifest:{RESET} {len(rows)} entities  "
        f"({CYAN}{len(persons)} persons{RESET} / {MAGENTA}{len(places)} places{RESET})\n"
    )

    col_w = max(len(r["name"]) for r in rows) + 2
    print(f"  {'Name':<{col_w}} {'Type':<8} {'Chunks':>6}  {'Ingested at'}")
    print("  " + "─" * (col_w + 30))
    for r in sorted(rows, key=lambda x: (x["entity_type"], x["name"])):
        colour = _entity_colour(r["entity_type"])
        ts = r["ingested_at"][:19].replace("T", " ")
        print(
            f"  {colour}{r['name']:<{col_w}}{RESET}"
            f"{r['entity_type']:<8} {r['chunk_count']:>6}  {DIM}{ts}{RESET}"
        )


def _do_query(
    query_str: str,
    config: dict,
    top_k: int,
    filter_type: str,
) -> None:
    embed_model = config.get("embedding_model", "nomic-embed-text")
    chroma_path = config["chroma_path"]

    print(f"\n{BOLD}Query:{RESET}  {query_str}")
    print(f"{BOLD}Filter:{RESET} {filter_type}  |  {BOLD}Top-k:{RESET} {top_k}")
    print(f"{DIM}Embedding query...{RESET}", end=" ", flush=True)

    try:
        q_vector = embed_one(query_str, model=embed_model)
    except RuntimeError as exc:
        print(f"\n{RESET}ERROR: Could not embed query — {exc}")
        print("Is the Ollama daemon running?  Run:  ollama serve")
        sys.exit(1)

    print("done.")

    client = get_client(chroma_path)
    col = get_collection(client)

    where = None
    if filter_type in ("person", "place"):
        where = {"entity_type": {"$eq": filter_type}}

    results = query(col, q_vector, n_results=top_k, where=where)

    ids = results["ids"][0]
    distances = results["distances"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not ids:
        print("\nNo results found.  Is the vector store populated?")
        print("Run: python scripts/run_ingest.py --rebuild")
        return

    print(f"\n{'─' * 60}")
    for rank, (doc_id, dist, doc_text, meta) in enumerate(
        zip(ids, distances, documents, metadatas), start=1
    ):
        etype = meta.get("entity_type", "?")
        ename = meta.get("entity_name", "?")
        cidx = meta.get("chunk_index", "?")
        url = meta.get("source_url", "")

        colour = _entity_colour(etype)
        preview = doc_text[:220].replace("\n", " ")
        if len(doc_text) > 220:
            preview += "…"

        print(
            f"\n{BOLD}#{rank}{RESET}  "
            f"{colour}[{etype}]{RESET}  "
            f"{BOLD}{ename}{RESET}  "
            f"{DIM}chunk {cidx}{RESET}  "
            f"distance={dist:.4f}"
        )
        print(f"  {preview}")
        if url:
            print(f"  {DIM}{url}{RESET}")

    print(f"\n{'─' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the ChromaDB vector store for Checkpoint 1 verification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("query", nargs="?", help="Natural-language query string.")
    parser.add_argument("--top-k", type=int, default=5, metavar="N")
    parser.add_argument(
        "--filter",
        choices=["person", "place", "both"],
        default="both",
        dest="filter_type",
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print collection and manifest stats, then exit.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)

    if args.stats:
        _print_stats(config)
        return

    if not args.query:
        parser.print_help()
        sys.exit(0)

    _do_query(args.query, config, top_k=args.top_k, filter_type=args.filter_type)


if __name__ == "__main__":
    main()
