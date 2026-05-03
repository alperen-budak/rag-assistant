"""
src/interface/cli.py — Interactive terminal RAG chat loop.

Run with:
    python -m src.interface.cli
    python -m src.interface.cli --verbose    # show retrieved chunks
    python -m src.interface.cli --no-stream  # wait for full answer (no streaming)

Commands inside the REPL:
    :quit / :q / Ctrl-D / Ctrl-C  — exit
    :sources                      — toggle source-chunk display (same as --verbose)
    :category                     — toggle router-decision display
    :help                         — print command list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root on sys.path when run as a module
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.rag.generator import generate_answer, stream_answer  # noqa: E402
from src.rag.prompts import REFUSAL_STRING, unique_sources  # noqa: E402
from src.rag.retriever import retrieve_with_category  # noqa: E402

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
MAGENTA= "\033[95m"
RED    = "\033[91m"
BLUE   = "\033[94m"

_BANNER = f"""{BOLD}{BLUE}
╔══════════════════════════════════════════════════════════════╗
║          Wikipedia RAG  —  Local Knowledge Assistant         ║
║    Model: llama3.2:3b  │  Embeddings: nomic-embed-text       ║
║    Corpus: 44 entities (22 persons + 22 places)              ║
╚══════════════════════════════════════════════════════════════╝{RESET}
{DIM}Type your question and press Enter.
Commands: :quit  :sources  :category  :help{RESET}
"""

_HELP = f"""{BOLD}Commands:{RESET}
  :quit / :q      Exit the program
  :sources        Toggle display of retrieved context chunks
  :category       Toggle display of router category decision
  :help           Show this help message
"""

_ENTITY_COLOUR = {"person": CYAN, "place": MAGENTA}


def _colour_category(cat: str) -> str:
    colour = _ENTITY_COLOUR.get(cat, YELLOW)
    return f"{colour}{cat}{RESET}"


def _print_sources(chunks, *, verbose: bool) -> None:
    sources = unique_sources(chunks)
    if not sources:
        return

    print(f"\n{DIM}{'─' * 60}{RESET}")
    print(f"{BOLD}Sources:{RESET}")
    for name, etype, url in sources:
        colour = _ENTITY_COLOUR.get(etype, YELLOW)
        print(f"  {colour}[{etype}]{RESET} {BOLD}{name}{RESET}  {DIM}{url}{RESET}")

    if verbose and chunks:
        print(f"\n{BOLD}Retrieved chunks ({len(chunks)}):{RESET}")
        for i, chunk in enumerate(chunks, 1):
            colour = _ENTITY_COLOUR.get(chunk.entity_type, YELLOW)
            preview = chunk.text[:200].replace("\n", " ")
            if len(chunk.text) > 200:
                preview += "…"
            print(
                f"  {DIM}#{i}{RESET} {colour}{chunk.entity_name}{RESET} "
                f"{DIM}(dist={chunk.distance:.3f}){RESET}\n  {preview}\n"
            )


def run_cli(*, verbose: bool = False, no_stream: bool = False, show_category: bool = False) -> None:
    print(_BANNER)

    show_sources = verbose
    show_cat = show_category

    while True:
        # ── Read input ────────────────────────────────────────────────────────
        try:
            raw = input(f"{BOLD}{GREEN}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye.{RESET}")
            break

        if not raw:
            continue

        # ── In-REPL commands ──────────────────────────────────────────────────
        if raw.lower() in (":quit", ":q", "quit", "exit"):
            print(f"{DIM}Goodbye.{RESET}")
            break
        if raw.lower() == ":help":
            print(_HELP)
            continue
        if raw.lower() == ":sources":
            show_sources = not show_sources
            state = "ON" if show_sources else "OFF"
            print(f"{DIM}Source display: {state}{RESET}")
            continue
        if raw.lower() == ":category":
            show_cat = not show_cat
            state = "ON" if show_cat else "OFF"
            print(f"{DIM}Category display: {state}{RESET}")
            continue

        # ── Retrieve ──────────────────────────────────────────────────────────
        print(f"{DIM}Searching…{RESET}", end="\r")
        chunks, category = retrieve_with_category(raw)

        if show_cat:
            print(
                f"{DIM}Router → {_colour_category(category)}  "
                f"({len(chunks)} chunks retrieved){RESET}   "
            )
        else:
            print(" " * 40, end="\r")  # clear "Searching…"

        # ── Generate ──────────────────────────────────────────────────────────
        print(f"\n{BOLD}Assistant:{RESET} ", end="", flush=True)

        if not chunks:
            print(f"{YELLOW}{REFUSAL_STRING}{RESET}")
        elif no_stream:
            answer = generate_answer(raw, chunks, stream=False)
            print(answer)
        else:
            # Stream tokens directly to stdout
            full_answer: list[str] = []
            for token in stream_answer(raw, chunks):
                print(token, end="", flush=True)
                full_answer.append(token)
            print()  # newline after stream ends

        # ── Sources footer ────────────────────────────────────────────────────
        if chunks:
            _print_sources(chunks, verbose=show_sources)

        print()  # blank line between turns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wikipedia RAG — local terminal chat assistant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show retrieved context chunks alongside each answer.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Wait for the full answer instead of streaming tokens.",
    )
    parser.add_argument(
        "--show-category",
        action="store_true",
        help="Always display the router category decision.",
    )
    args = parser.parse_args()

    run_cli(
        verbose=args.verbose,
        no_stream=args.no_stream,
        show_category=args.show_category,
    )


if __name__ == "__main__":
    main()
