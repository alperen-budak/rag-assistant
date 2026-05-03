"""
src/rag/vector_store.py — ChromaDB PersistentClient wrapper.

Implements the "Option B" single-collection + metadata-filtering strategy
from PROJECT.md §3.1.

Collection name  : wiki_rag
Metadata schema  : entity_name (str), entity_type ("person"|"place"),
                   source_url (str), chunk_index (int)
Chunk ID format  : "{entity_name}::{chunk_index:04d}"

Public API
----------
    get_client(chroma_path: str | Path) -> chromadb.PersistentClient
    get_collection(client)              -> chromadb.Collection
    upsert(collection, ids, embeddings, documents, metadatas) -> None
    query(collection, query_embedding, n_results, where)      -> QueryResult
    drop_collection(client)             -> None
    collection_count(collection)        -> int
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import chromadb
from chromadb import Collection
from chromadb.api import ClientAPI
from chromadb.config import Settings

_PathLike = Union[str, Path]

COLLECTION_NAME = "wiki_rag"

# Distance function: cosine (vectors from nomic-embed-text are L2-normalised)
_DISTANCE_FN = "cosine"


# ── Client / collection management ───────────────────────────────────────────

def get_client(chroma_path: _PathLike) -> ClientAPI:
    """
    Return a ChromaDB PersistentClient for the given directory.
    The directory is created if it doesn't exist.

    The client is intentionally NOT cached in a module-level variable here —
    call sites (pipeline, retriever, CLI) own the lifetime of the client
    object to guarantee the lazy-client contract from PROJECT.md §4.2.
    """
    path = Path(chroma_path)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(client: ClientAPI) -> Collection:
    """
    Return the ``wiki_rag`` collection, creating it if needed.
    Uses cosine distance to match nomic-embed-text normalisation.
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": _DISTANCE_FN},
    )


def drop_collection(client: ClientAPI) -> None:
    """Delete the ``wiki_rag`` collection (used by --rebuild flag)."""
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001
        pass  # Collection didn't exist — that's fine


def collection_count(collection: Collection) -> int:
    """Return the number of chunks stored in the collection."""
    return collection.count()


# ── Write ──────────────────────────────────────────────────────────────────────

def upsert(
    collection: Collection,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    """
    Upsert chunks into the collection.  Idempotent — re-running with the
    same IDs overwrites existing records (supports incremental re-ingestion).

    Args:
        ids:        Deterministic chunk IDs, format ``"{entity_name}::{chunk_index:04d}"``.
        embeddings: 768-dim float vectors (one per chunk).
        documents:  Raw chunk text strings.
        metadatas:  Dicts with keys: entity_name, entity_type, source_url, chunk_index.
    """
    if not ids:
        return
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


# ── Read ───────────────────────────────────────────────────────────────────────

def query(
    collection: Collection,
    query_embedding: list[float],
    n_results: int = 5,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Return the top-k nearest chunks to `query_embedding`.

    Args:
        collection:      The ``wiki_rag`` Chroma collection.
        query_embedding: A single 768-dim query vector.
        n_results:       Number of results to return.
        where:           Optional Chroma metadata filter, e.g.
                         ``{"entity_type": {"$eq": "person"}}``.
                         Pass ``None`` for no filter (searches both categories).

    Returns:
        ChromaDB query result dict with keys:
            ids, distances, documents, metadatas  (each a list-of-lists)
    """
    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        kwargs["where"] = where

    return collection.query(**kwargs)


# ── Chunk ID helper ───────────────────────────────────────────────────────────

def make_chunk_id(entity_name: str, chunk_index: int) -> str:
    """Return the deterministic chunk ID used across ingest and retrieval."""
    return f"{entity_name}::{chunk_index:04d}"
