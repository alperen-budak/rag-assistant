"""
src/rag/retriever.py — Category-aware top-k retrieval.

MODULE CONTRACT (PROJECT.md §6):
    retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]

Pipeline:
  1. detect_category(query)  → "person" | "place" | "both"
  2. embed_one(query)        → 768-dim query vector
  3. vector_store.query()    → top-k Chroma results (filtered by entity_type)
  4. Apply similarity threshold — drop chunks that are too far from the query
  5. Return list[RetrievedChunk]

If zero chunks survive the threshold, an empty list is returned.
The generator interprets an empty list as "no grounding available" and
short-circuits to the refusal string (PROJECT.md §3.4).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.rag.embeddings import embed_one
from src.rag.router import detect_category
from src.rag.types import RetrievedChunk
from src.rag.vector_store import get_client, get_collection, query as chroma_query

# ── Config ────────────────────────────────────────────────────────────────────

_SETTINGS_PATH = Path("config/settings.yaml")


def _load_settings() -> dict:
    try:
        with _SETTINGS_PATH.open() as fh:
            return yaml.safe_load(fh)
    except Exception:  # noqa: BLE001
        return {}


# RetrievedChunk is defined in src/rag/types.py and re-exported here for
# backwards compatibility so existing callers don't need to change their import.
__all__ = ["RetrievedChunk", "retrieve", "retrieve_with_category"]

# ── Chroma client — lazy singleton per process ────────────────────────────────

_client = None
_collection = None


def _get_collection():
    global _client, _collection  # noqa: PLW0603
    if _collection is None:
        cfg = _load_settings()
        chroma_path = cfg.get("chroma_path", "./data/chroma")
        _client = get_client(chroma_path)
        _collection = get_collection(_client)
    return _collection


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int | None = None,
    *,
    similarity_threshold: float | None = None,
    use_llm_fallback: bool = True,
) -> list[RetrievedChunk]:
    """
    Return the top-k most relevant chunks for `query`, filtered by category.

    Args:
        query:                Natural-language user query.
        top_k:                Number of results (default from settings.yaml).
        similarity_threshold: Maximum cosine distance to accept (default from settings).
                              Chunks beyond this threshold are dropped.
        use_llm_fallback:     Passed through to `detect_category()`.

    Returns:
        List of `RetrievedChunk` objects, sorted by ascending distance.
        Returns an empty list if no chunks pass the threshold.
    """
    cfg = _load_settings()
    if top_k is None:
        top_k = int(cfg.get("top_k", 5))
    if similarity_threshold is None:
        similarity_threshold = float(cfg.get("similarity_threshold", 1.5))

    # Step 1 — classify query
    category = detect_category(query, use_llm_fallback=use_llm_fallback)

    # Step 2 — build Chroma metadata filter
    if category == "person":
        where: dict | None = {"entity_type": {"$eq": "person"}}
    elif category == "place":
        where = {"entity_type": {"$eq": "place"}}
    else:
        where = None  # search all

    # Step 3 — embed query
    try:
        q_vector = embed_one(query)
    except RuntimeError:
        return []

    # Step 4 — query Chroma
    col = _get_collection()
    results = chroma_query(col, q_vector, n_results=top_k, where=where)

    # Step 5 — parse and threshold
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    chunks: list[RetrievedChunk] = []
    for doc_id, dist, doc, meta in zip(ids, distances, documents, metadatas):
        if dist > similarity_threshold:
            continue  # too distant — drop
        chunks.append(
            RetrievedChunk(
                text=doc,
                entity_name=meta.get("entity_name", ""),
                entity_type=meta.get("entity_type", ""),
                distance=dist,
                source_url=meta.get("source_url", ""),
                chunk_index=int(meta.get("chunk_index", 0)),
            )
        )

    return chunks


def retrieve_with_category(
    query: str,
    top_k: int | None = None,
    *,
    similarity_threshold: float | None = None,
    use_llm_fallback: bool = True,
) -> tuple[list[RetrievedChunk], str]:
    """
    Like `retrieve()` but also returns the detected category string.
    Used by the CLI to display the router decision.
    """
    cfg = _load_settings()
    if top_k is None:
        top_k = int(cfg.get("top_k", 5))

    category = detect_category(query, use_llm_fallback=use_llm_fallback)

    # Temporarily set category so retrieve() doesn't call detect_category again
    # (simpler: just inline the logic here)
    if similarity_threshold is None:
        similarity_threshold = float(cfg.get("similarity_threshold", 1.5))

    if category == "person":
        where: dict | None = {"entity_type": {"$eq": "person"}}
    elif category == "place":
        where = {"entity_type": {"$eq": "place"}}
    else:
        where = None

    try:
        q_vector = embed_one(query)
    except RuntimeError:
        return [], category

    col = _get_collection()
    results = chroma_query(col, q_vector, n_results=top_k, where=where)

    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    chunks: list[RetrievedChunk] = []
    for _, dist, doc, meta in zip(ids, distances, documents, metadatas):
        if dist > similarity_threshold:
            continue
        chunks.append(
            RetrievedChunk(
                text=doc,
                entity_name=meta.get("entity_name", ""),
                entity_type=meta.get("entity_type", ""),
                distance=dist,
                source_url=meta.get("source_url", ""),
                chunk_index=int(meta.get("chunk_index", 0)),
            )
        )

    return chunks, category
