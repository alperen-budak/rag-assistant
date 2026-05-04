"""
src/rag/retriever.py — Category-aware top-k retrieval.

MODULE CONTRACT (PROJECT.md §6):
    retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]

Pipeline:
  1. detect_category(query)   → "person" | "place" | "both"
  2. embed_one(query)         → 768-dim query vector
  3. vector_store.query()     → oversample (top_k × OVERSAMPLE_FACTOR) candidates
                                filtered by entity_type metadata
  4. Apply similarity threshold — drop chunks above the ceiling
  5. Sort by distance, truncate to top_k, return list[RetrievedChunk]

Oversampling rationale:
  Without oversampling, a similarity threshold applied *after* Chroma returns
  exactly `top_k` results can silently discard the most relevant chunks (the
  correct entity's chunks may rank just outside top_k before thresholding).
  By requesting `top_k × OVERSAMPLE_FACTOR` from Chroma and then thresholding,
  we ensure the best semantically relevant chunk always has a chance to appear.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml

import math

from src.core.manifest import get_entity_types
from src.rag.embeddings import embed_one
from src.rag.router import detect_category
from src.rag.types import RetrievedChunk
from src.rag.vector_store import collection_count, get_client, get_collection
from src.rag.vector_store import query as chroma_query

# RetrievedChunk re-exported for backwards-compatible imports from this module
__all__ = ["RetrievedChunk", "retrieve", "retrieve_with_category"]

# ── Config ────────────────────────────────────────────────────────────────────

_SETTINGS_PATH = Path("config/settings.yaml")

# Ask Chroma for this many times top_k before applying the threshold.
# Ensures the true best match is never pruned before we see it.
OVERSAMPLE_FACTOR = 4

# How many chunks to force-include from an explicitly mentioned entity.
ENTITY_INJECTION_K = 3


def _load_settings() -> dict:
    try:
        with _SETTINGS_PATH.open() as fh:
            return yaml.safe_load(fh)
    except Exception:  # noqa: BLE001
        return {}


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


# ── Entity-name injection helper ─────────────────────────────────────────────

def _find_mentioned_entities(query: str, category: str) -> list[str]:
    """
    Scan `query` for explicit entity-name mentions.

    Returns original-case entity names (as stored in the manifest and in
    ChromaDB's `entity_name` metadata field) that:
      - appear as a substring of the lowercased query, AND
      - match the given `category` ('person', 'place', or 'both').

    Example: query="Where is the Colosseum located?" → ['Colosseum']
    """
    cfg = _load_settings()
    db_path = cfg.get("manifest_path", "./data/manifest.db")
    try:
        entity_map: dict[str, str] = get_entity_types(db_path)  # {name: entity_type}
    except Exception:  # noqa: BLE001
        return []

    query_lower = query.lower()
    matched: list[str] = []
    for name, etype in entity_map.items():
        if category not in ("both", etype):
            continue
        if name.lower() in query_lower:
            matched.append(name)

    # Prefer longer matches first (avoids "Paris" overshadowing "Palace of Versailles")
    matched.sort(key=len, reverse=True)
    return matched


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine distance ∈ [0, 2].  Lower = more similar."""
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 1.0
    return 1.0 - dot / (mag_a * mag_b)


def _parse_chroma_rows(
    raw: dict,
    similarity_threshold: float,
) -> list[RetrievedChunk]:
    """Convert a raw Chroma query-result dict to RetrievedChunk objects."""
    ids       = raw.get("ids",       [[]])[0]
    distances = raw.get("distances", [[]])[0]
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    chunks: list[RetrievedChunk] = []
    for _, dist, doc, meta in zip(ids, distances, documents, metadatas):
        if dist > similarity_threshold:
            continue
        chunks.append(RetrievedChunk(
            text=doc,
            entity_name=meta.get("entity_name", ""),
            entity_type=meta.get("entity_type", ""),
            distance=dist,
            source_url=meta.get("source_url", ""),
            chunk_index=int(meta.get("chunk_index", 0)),
        ))
    return chunks


def _inject_entity_chunks(
    col,
    entity_name: str,
    q_vector: list[float],
    k: int,
    debug: bool,
) -> list[RetrievedChunk]:
    """
    Fetch up to `k` chunks for `entity_name` using a pure metadata lookup
    (``collection.get()``), then rank them by cosine distance to `q_vector`
    in pure Python.

    Why ``get()`` instead of ``query()``:
    ChromaDB's HNSW-based ``query()`` with a ``where`` filter can silently
    return zero results when the filtered subset is small relative to the full
    index — a known limitation in v0.5.x.  ``get()`` bypasses the vector index
    entirely and performs a direct metadata scan, which is always reliable.
    """
    try:
        result = col.get(
            where={"entity_name": {"$eq": entity_name}},
            include=["documents", "metadatas", "embeddings"],
            limit=100,
        )
    except Exception as exc:  # noqa: BLE001
        if debug:
            print(f"[retriever DEBUG] injection get() failed for '{entity_name}': {exc}")
        return []

    docs  = result.get("documents") or []
    metas = result.get("metadatas") or []
    raw_embs = result.get("embeddings")
    # raw_embs may be a numpy array — use explicit None check, not truthiness test
    embs = raw_embs if raw_embs is not None else []

    if debug:
        print(
            f"[retriever DEBUG] injection get() returned {len(docs)} chunks "
            f"for '{entity_name}'"
        )

    if not docs or len(embs) == 0:
        return []

    ranked: list[tuple[float, str, dict]] = []
    for doc, meta, emb in zip(docs, metas, embs):
        if emb is None or len(emb) == 0:
            continue
        dist = _cosine_distance(q_vector, list(emb))
        ranked.append((dist, doc, meta))

    ranked.sort(key=lambda x: x[0])

    chunks: list[RetrievedChunk] = []
    for dist, doc, meta in ranked[:k]:
        chunks.append(RetrievedChunk(
            text=doc,
            entity_name=meta.get("entity_name", ""),
            entity_type=meta.get("entity_type", ""),
            distance=dist,
            source_url=meta.get("source_url", ""),
            chunk_index=int(meta.get("chunk_index", 0)),
        ))

    if debug and chunks:
        print(
            f"[retriever DEBUG] injection top-{k} for '{entity_name}': "
            + ", ".join(f"chunk={c.chunk_index} dist={c.distance:.4f}" for c in chunks)
        )

    return chunks


# ── Core retrieval logic (shared by both public functions) ────────────────────

def _retrieve_core(
    query: str,
    top_k: int,
    similarity_threshold: float,
    use_llm_fallback: bool,
    debug: bool,
) -> tuple[list[RetrievedChunk], str]:
    """
    Internal implementation.  Returns (chunks, category).

    Oversample strategy:
    - Request `min(top_k × OVERSAMPLE_FACTOR, total_filtered_chunks)` from Chroma.
    - Apply similarity_threshold to drop noisy results.
    - Sort by ascending distance.
    - Truncate to top_k.
    """

    # Step 1 — classify query
    category: Literal["person", "place", "both"] = detect_category(
        query, use_llm_fallback=use_llm_fallback
    )

    # Step 2 — build category-level Chroma metadata filter (Option B)
    if category == "person":
        cat_where: dict | None = {"entity_type": {"$eq": "person"}}
    elif category == "place":
        cat_where = {"entity_type": {"$eq": "place"}}
    else:
        cat_where = None

    # Step 3 — embed query once; reuse for all sub-queries
    try:
        q_vector = embed_one(query)
    except RuntimeError as exc:
        if debug:
            print(f"[retriever] embed_one failed: {exc}")
        return [], category

    col = _get_collection()
    total = collection_count(col)

    if debug:
        print(
            f"\n[retriever DEBUG] query     = {query!r}\n"
            f"[retriever DEBUG] category  = {category}\n"
            f"[retriever DEBUG] where     = {cat_where}\n"
            f"[retriever DEBUG] top_k     = {top_k}, threshold = {similarity_threshold}, "
            f"total_chunks = {total}"
        )

    # ── Step 4a — Entity-name injection ───────────────────────────────────────
    # If the query explicitly names a known entity (e.g. "Colosseum"), its chunks
    # may not rank in the semantic top-k because the Wikipedia article uses a
    # different primary name (e.g. "Flavian Amphitheatre").  We force-include
    # the top ENTITY_INJECTION_K chunks for any entity whose name appears
    # literally in the query text.

    injected: list[RetrievedChunk] = []
    mentioned = _find_mentioned_entities(query, category)

    if debug and mentioned:
        print(f"[retriever DEBUG] entity-name injection: {mentioned}")

    for entity_name in mentioned:
        injected.extend(
            _inject_entity_chunks(col, entity_name, q_vector, ENTITY_INJECTION_K, debug)
        )

    # ── Step 4b — Semantic oversample across full category ─────────────────────
    n_request = min(top_k * OVERSAMPLE_FACTOR, max(total, 1))
    semantic_raw = chroma_query(col, q_vector, n_results=n_request, where=cat_where)
    semantic_chunks = _parse_chroma_rows(semantic_raw, similarity_threshold)

    if debug:
        ids_debug = semantic_raw.get("ids", [[]])[0]
        dist_debug = semantic_raw.get("distances", [[]])[0]
        meta_debug = semantic_raw.get("metadatas", [[]])[0]
        print(f"[retriever DEBUG] Semantic search returned {len(ids_debug)} candidates:")
        for i, (rid, dist, meta) in enumerate(zip(ids_debug, dist_debug, meta_debug)):
            entity = meta.get("entity_name", "?")
            etype  = meta.get("entity_type", "?")
            cidx   = meta.get("chunk_index", "?")
            flag   = "✓" if dist <= similarity_threshold else "✗ (over threshold)"
            injected_tag = " ← injected" if any(
                c.entity_name == entity and c.chunk_index == int(meta.get("chunk_index", -1))
                for c in injected
            ) else ""
            print(f"  [{i+1:02d}] {flag}  dist={dist:.4f}  [{etype}] {entity}  "
                  f"chunk={cidx}{injected_tag}")

    # ── Step 5 — Merge with reserved slots for injected chunks ───────────────
    #
    # Injected chunks are PINNED to the front of the result list — they represent
    # the user's explicit intent (they named the entity) and must appear regardless
    # of their cosine distance relative to the semantic results.
    #
    # Strategy:
    #   1. Sort injected chunks by their own distance (best first).
    #   2. Sort semantic chunks by their own distance (best first).
    #   3. Fill the list: injected chunks occupy the first N slots; semantic
    #      results fill the remaining (top_k - N) slots, skipping duplicates.
    #
    # This is a "pinned results" pattern: explicitly named entities always surface.

    injected_sorted  = sorted(injected,       key=lambda c: c.distance)
    semantic_sorted  = sorted(semantic_chunks, key=lambda c: c.distance)

    seen: set[tuple[str, int]] = set()
    chunks: list[RetrievedChunk] = []

    for chunk in injected_sorted:
        key = (chunk.entity_name, chunk.chunk_index)
        if key not in seen:
            seen.add(key)
            chunks.append(chunk)

    for chunk in semantic_sorted:
        if len(chunks) >= top_k:
            break
        key = (chunk.entity_name, chunk.chunk_index)
        if key not in seen:
            seen.add(key)
            chunks.append(chunk)

    if debug:
        print(
            f"[retriever DEBUG] After merge+truncate: {len(chunks)} chunks returned "
            f"(entities: {list(dict.fromkeys(c.entity_name for c in chunks))})\n"
        )

    return chunks, category


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int | None = None,
    *,
    similarity_threshold: float | None = None,
    use_llm_fallback: bool = True,
    debug: bool = False,
) -> list[RetrievedChunk]:
    """
    Return the top-k most relevant chunks for `query`, filtered by category.

    Args:
        query:                Natural-language user query.
        top_k:                Number of results (default from settings.yaml).
        similarity_threshold: Maximum cosine distance to accept.
        use_llm_fallback:     Whether to call the LLM when Tier-1 routing is
                              ambiguous.
        debug:                Print detailed retrieval diagnostics to stdout.

    Returns:
        Sorted list of RetrievedChunk objects (best match first).
        Empty list when no chunks pass the threshold.
    """
    cfg = _load_settings()
    if top_k is None:
        top_k = int(cfg.get("top_k", 5))
    if similarity_threshold is None:
        similarity_threshold = float(cfg.get("similarity_threshold", 1.5))

    chunks, _ = _retrieve_core(
        query, top_k, similarity_threshold, use_llm_fallback, debug
    )
    return chunks


def retrieve_with_category(
    query: str,
    top_k: int | None = None,
    *,
    similarity_threshold: float | None = None,
    use_llm_fallback: bool = True,
    debug: bool = False,
) -> tuple[list[RetrievedChunk], str]:
    """
    Like `retrieve()` but also returns the detected category string.
    Used by the CLI and Streamlit UI to display the router decision.
    """
    cfg = _load_settings()
    if top_k is None:
        top_k = int(cfg.get("top_k", 5))
    if similarity_threshold is None:
        similarity_threshold = float(cfg.get("similarity_threshold", 1.5))

    return _retrieve_core(
        query, top_k, similarity_threshold, use_llm_fallback, debug
    )
