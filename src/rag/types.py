"""
src/rag/types.py — Shared dataclasses for the RAG pipeline.

Keeping types in a separate module breaks the circular import risk and
allows test files to import RetrievedChunk without pulling in chromadb.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    text: str
    entity_name: str
    entity_type: str        # "person" | "place"
    distance: float         # cosine distance (lower = more similar)
    source_url: str
    chunk_index: int
