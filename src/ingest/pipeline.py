"""
src/ingest/pipeline.py — End-to-end ingestion orchestrator.

Execution order per entity (PROJECT.md §2):
  fetch → clean → chunk (generator) → embed in batches → upsert to Chroma → record in manifest

Memory discipline (PROJECT.md §4.2):
  - Articles are processed ONE AT A TIME — the full corpus is never materialised.
  - Chunks are consumed from the generator in batches of `embed_batch_size`;
    each batch is embedded and upserted immediately, then the batch list is cleared.
  - The ChromaDB client and collection are opened once and reused for the entire run.

Public API
----------
    run(config: dict, *, rebuild: bool = False, limit: int | None = None) -> RunStats
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.core.chunking import chunk_text
from src.core.cleaning import clean
from src.core.manifest import hash_text, init_db, is_unchanged, upsert_entity
from src.ingest.wiki_fetcher import fetch
from src.rag.embeddings import embed
from src.rag.vector_store import (
    drop_collection,
    get_client,
    get_collection,
    make_chunk_id,
    upsert,
)

# ── Result container ──────────────────────────────────────────────────────────


@dataclass
class RunStats:
    total: int = 0
    ingested: int = 0
    skipped: int = 0
    failed: int = 0
    total_chunks: int = 0
    errors: list[str] = field(default_factory=list)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _load_entities(entities_yaml: str | Path) -> list[dict]:
    with open(entities_yaml, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    persons = data.get("persons", [])
    places = data.get("places", [])
    return persons + places


def _log(msg: str, indent: int = 0) -> None:
    prefix = "  " * indent
    print(f"{prefix}{msg}", flush=True)


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run(
    config: dict,
    *,
    rebuild: bool = False,
    limit: int | None = None,
) -> RunStats:
    """
    Run the full ingestion pipeline.

    Args:
        config:   Dict loaded from ``config/settings.yaml``.
        rebuild:  If True, drop the existing ChromaDB collection first.
        limit:    If set, process only the first `limit` entities (for fast dev iteration).

    Returns:
        RunStats dataclass summarising what happened.
    """
    stats = RunStats()

    # ── Paths from config ──────────────────────────────────────────────────────
    chroma_path = Path(config["chroma_path"])
    manifest_path = Path(config["manifest_path"])
    raw_cache_path = Path(config["raw_cache_path"])
    entities_yaml = Path("config/entities.yaml")

    chunk_size: int = int(config.get("chunk_size", 500))
    overlap: int = int(config.get("chunk_overlap", 50))
    embed_batch_size: int = int(config.get("embed_batch_size", 8))
    embed_model: str = config.get("embedding_model", "nomic-embed-text")

    # ── Initialise infrastructure ──────────────────────────────────────────────
    init_db(manifest_path)

    client = get_client(chroma_path)
    if rebuild:
        _log("--rebuild: dropping existing collection")
        drop_collection(client)
    collection = get_collection(client)

    # ── Load entity list ───────────────────────────────────────────────────────
    entities = _load_entities(entities_yaml)
    if limit is not None:
        entities = entities[:limit]

    stats.total = len(entities)
    _log(f"Entities to process: {stats.total}")

    # ── Per-entity loop ────────────────────────────────────────────────────────
    for idx, entity in enumerate(entities, start=1):
        name: str = entity["name"]
        wiki_title: str = entity["wiki_title"]
        entity_type: str = entity["entity_type"]

        _log(f"[{idx:02d}/{stats.total}] {name} ({entity_type})")

        # 1. Fetch ─────────────────────────────────────────────────────────────
        try:
            result = fetch(wiki_title, cache_dir=raw_cache_path)
        except Exception as exc:  # noqa: BLE001 — unexpected network error
            _log(f"SKIP — unexpected fetch error: {exc}", indent=1)
            stats.failed += 1
            stats.errors.append(f"{name}: fetch exception — {exc}")
            continue

        if result is None:
            _log(f"SKIP — could not fetch '{wiki_title}'", indent=1)
            stats.failed += 1
            stats.errors.append(f"{name}: fetch failed")
            continue

        # 2. Clean ─────────────────────────────────────────────────────────────
        cleaned = clean(result.text)
        if not cleaned:
            _log("SKIP — empty after cleaning", indent=1)
            stats.failed += 1
            stats.errors.append(f"{name}: empty after cleaning")
            continue

        # 3. Deduplication check via manifest ──────────────────────────────────
        content_hash = hash_text(cleaned)
        if not rebuild and is_unchanged(manifest_path, name, content_hash):
            _log("SKIP — content unchanged since last ingest", indent=1)
            stats.skipped += 1
            continue

        # 4. Chunk → embed → upsert (streaming, never builds a full list) ──────
        chunk_texts_batch: list[str] = []
        chunk_indices_batch: list[int] = []
        chunk_index = 0
        entity_chunk_count = 0

        def _flush_batch() -> None:
            nonlocal entity_chunk_count
            if not chunk_texts_batch:
                return

            try:
                vectors = embed(chunk_texts_batch, model=embed_model, batch_size=embed_batch_size)
            except RuntimeError as exc:
                _log(f"ERROR embedding batch: {exc}", indent=2)
                stats.errors.append(f"{name}: embed error — {exc}")
                chunk_texts_batch.clear()
                chunk_indices_batch.clear()
                return

            ids = [make_chunk_id(name, ci) for ci in chunk_indices_batch]
            metadatas = [
                {
                    "entity_name": name,
                    "entity_type": entity_type,
                    "source_url": result.url,
                    "chunk_index": ci,
                }
                for ci in chunk_indices_batch
            ]

            upsert(collection, ids, vectors, chunk_texts_batch, metadatas)
            entity_chunk_count += len(ids)

            chunk_texts_batch.clear()
            chunk_indices_batch.clear()

        for chunk in chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap):
            chunk_texts_batch.append(chunk)
            chunk_indices_batch.append(chunk_index)
            chunk_index += 1

            if len(chunk_texts_batch) >= embed_batch_size:
                _flush_batch()

        _flush_batch()  # flush the final partial batch

        if entity_chunk_count == 0:
            _log("WARN — zero chunks produced", indent=1)
            stats.failed += 1
            stats.errors.append(f"{name}: zero chunks")
            continue

        # 5. Record in manifest ────────────────────────────────────────────────
        upsert_entity(manifest_path, name, wiki_title, entity_type, content_hash, entity_chunk_count)

        _log(f"OK — {entity_chunk_count} chunks stored", indent=1)
        stats.ingested += 1
        stats.total_chunks += entity_chunk_count

    # ── Summary ────────────────────────────────────────────────────────────────
    _log("")
    _log("═" * 50)
    _log(f"Done.  ingested={stats.ingested}  skipped={stats.skipped}  failed={stats.failed}")
    _log(f"Total chunks in ChromaDB: {stats.total_chunks}")
    if stats.errors:
        _log("Errors:")
        for err in stats.errors:
            _log(f"  • {err}")

    return stats
