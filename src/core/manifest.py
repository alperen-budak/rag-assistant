"""
src/core/manifest.py — SQLite ingestion ledger.

Tracks which entities have been embedded and stored in ChromaDB so that
incremental re-runs skip unchanged articles (PROJECT.md §4.4).

Schema
------
Table: entities
    name          TEXT PRIMARY KEY   — display name from entities.yaml
    wiki_title    TEXT NOT NULL      — exact Wikipedia page title
    entity_type   TEXT NOT NULL      — 'person' | 'place'
    content_hash  TEXT NOT NULL      — SHA-256 of the cleaned article text
    chunk_count   INTEGER NOT NULL   — number of chunks stored in ChromaDB
    ingested_at   TEXT NOT NULL      — ISO-8601 UTC timestamp

Table: router_cache  (written by src/rag/router.py in Phase 2)
    query_hash    TEXT PRIMARY KEY   — SHA-256 of the raw query string
    category      TEXT NOT NULL      — 'person' | 'place' | 'both'
    cached_at     TEXT NOT NULL      — ISO-8601 UTC timestamp

Public API
----------
    init_db(db_path: str | Path) -> None
    upsert_entity(db_path, name, wiki_title, entity_type, content_hash, chunk_count) -> None
    is_unchanged(db_path, name, content_hash) -> bool
    all_entity_names(db_path) -> list[str]
    get_entity_types(db_path) -> dict[str, str]   # name -> entity_type
    summary(db_path) -> list[dict]
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

_PathLike = Union[str, Path]

# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS entities (
    name          TEXT PRIMARY KEY,
    wiki_title    TEXT NOT NULL,
    entity_type   TEXT NOT NULL CHECK(entity_type IN ('person', 'place')),
    content_hash  TEXT NOT NULL,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    ingested_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS router_cache (
    query_hash  TEXT PRIMARY KEY,
    category    TEXT NOT NULL CHECK(category IN ('person', 'place', 'both')),
    cached_at   TEXT NOT NULL
);
"""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _connect(db_path: _PathLike) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Public API ────────────────────────────────────────────────────────────────

def init_db(db_path: _PathLike) -> None:
    """Create tables if they don't exist yet.  Safe to call multiple times."""
    with _connect(db_path) as con:
        con.executescript(_DDL)


def upsert_entity(
    db_path: _PathLike,
    name: str,
    wiki_title: str,
    entity_type: str,
    content_hash: str,
    chunk_count: int,
) -> None:
    """Insert or update a single entity record."""
    sql = """
        INSERT INTO entities (name, wiki_title, entity_type, content_hash, chunk_count, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            wiki_title   = excluded.wiki_title,
            entity_type  = excluded.entity_type,
            content_hash = excluded.content_hash,
            chunk_count  = excluded.chunk_count,
            ingested_at  = excluded.ingested_at
    """
    with _connect(db_path) as con:
        con.execute(sql, (name, wiki_title, entity_type, content_hash, chunk_count, _now_utc()))


def is_unchanged(db_path: _PathLike, name: str, content_hash: str) -> bool:
    """
    Return True if `name` is already in the manifest with the same content hash.
    Used to skip re-embedding unchanged articles.
    """
    with _connect(db_path) as con:
        row = con.execute(
            "SELECT content_hash FROM entities WHERE name = ?", (name,)
        ).fetchone()
    return row is not None and row["content_hash"] == content_hash


def all_entity_names(db_path: _PathLike) -> list[str]:
    """Return a sorted list of all entity names in the manifest."""
    with _connect(db_path) as con:
        rows = con.execute("SELECT name FROM entities ORDER BY name").fetchall()
    return [r["name"] for r in rows]


def get_entity_types(db_path: _PathLike) -> dict[str, str]:
    """Return a mapping of entity name -> entity_type for router Tier 1."""
    with _connect(db_path) as con:
        rows = con.execute("SELECT name, entity_type FROM entities").fetchall()
    return {r["name"]: r["entity_type"] for r in rows}


def summary(db_path: _PathLike) -> list[dict]:
    """Return all rows as a list of dicts — used by check_env and run_ingest."""
    with _connect(db_path) as con:
        rows = con.execute(
            "SELECT name, entity_type, chunk_count, ingested_at FROM entities ORDER BY name"
        ).fetchall()
    return [dict(r) for r in rows]


# ── Utility: compute a content hash for a cleaned article string ──────────────

def hash_text(text: str) -> str:
    """Return a 16-hex-char truncated SHA-256 of `text` (collision-safe for our scale)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
