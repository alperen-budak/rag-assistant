"""
scripts/run_ingest.py — One-shot CLI entry point for the ingestion pipeline.

Usage:
    python scripts/run_ingest.py [--rebuild] [--limit N] [--config PATH]

Options:
    --rebuild       Drop the existing ChromaDB collection before ingesting.
                    Use this for a clean re-index from scratch.
    --limit N       Process only the first N entities (useful for fast testing).
    --config PATH   Path to settings.yaml (default: config/settings.yaml).

Examples:
    # Full rebuild (Checkpoint 1 verification):
    python scripts/run_ingest.py --rebuild

    # Test with the first 3 entities only (fast iteration):
    python scripts/run_ingest.py --rebuild --limit 3

    # Incremental update (only re-embeds changed articles):
    python scripts/run_ingest.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

# Ensure the project root is on sys.path when running as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ingest.pipeline import run  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest Wikipedia articles into the local RAG vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop the ChromaDB collection and re-ingest from scratch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N entities (for fast testing).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        metavar="PATH",
        help="Path to settings.yaml (default: config/settings.yaml).",
    )
    return parser.parse_args()


def _load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with config_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)

    print("=" * 60)
    print("  Wikipedia RAG — Ingestion Pipeline")
    print("=" * 60)
    if args.rebuild:
        print("  Mode   : REBUILD (existing collection will be dropped)")
    else:
        print("  Mode   : INCREMENTAL (unchanged articles skipped)")
    if args.limit:
        print(f"  Limit  : first {args.limit} entities")
    print(f"  Model  : {config.get('embedding_model', 'nomic-embed-text')}")
    print(f"  Chroma : {config.get('chroma_path', './data/chroma')}")
    print("=" * 60)
    print()

    t0 = time.perf_counter()

    stats = run(config, rebuild=args.rebuild, limit=args.limit)

    elapsed = time.perf_counter() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    if stats.failed > 0 and stats.ingested == 0:
        print("\nAll entities failed — check Ollama daemon and Wikipedia connectivity.")
        sys.exit(1)


if __name__ == "__main__":
    main()
