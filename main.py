"""
main.py — Project entry point for the Wikipedia RAG system.

Usage:
    python main.py                   # interactive chat (streaming)
    python main.py --verbose         # show retrieved chunks
    python main.py --show-category   # show router decision per query
    python main.py --no-stream       # disable token streaming
    python main.py --help            # full option list

Equivalent to:
    python -m src.interface.cli [options]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.interface.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
