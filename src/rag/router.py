"""
src/rag/router.py — Query category detection (person / place / both).

MODULE CONTRACT (PROJECT.md §6):
    detect_category(query: str) -> Literal["person", "place", "both"]

Two-tier strategy (PROJECT.md §3.2):
  Tier 1 — Lexical/Heuristic (<1 ms):
    1a. Exact name match against known entity names from the manifest.
    1b. Keyword scoring: count PERSON_KEYWORDS vs PLACE_KEYWORDS in the query.
        If one side wins clearly (score > 0, other side = 0) → return that category.
  Tier 2 — LLM fallback (slow path):
    2.  Send a one-token classification prompt to llama3.2:3b.
        Result is cached in the manifest's `router_cache` SQLite table.
  Fallback:
    If Tier 2 fails or is unavailable → return "both" (safe default).
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import ollama
import yaml

# ── Keyword tables ─────────────────────────────────────────────────────────────

PERSON_KEYWORDS: frozenset[str] = frozenset({
    # Pronouns / references
    "he", "she", "his", "her", "hers", "him", "they", "their",
    # Life events — include verb forms (die/dies/died, born/birth, etc.)
    "born", "birth", "die", "dies", "died", "death", "childhood", "grew",
    "married", "wife", "husband", "son", "daughter", "family",
    "father", "mother", "parents", "sibling", "brother", "sister",
    # Career / identity
    "inventor", "scientist", "artist", "musician", "writer", "author",
    "philosopher", "politician", "president", "king", "queen", "emperor",
    "general", "doctor", "physicist", "mathematician", "poet", "painter",
    "actor", "actress", "director", "composer", "architect", "engineer",
    "soldier", "revolutionary", "activist", "leader", "ruler", "nobel",
    "genius", "pioneer",
    # Achievement language
    "discovered", "invented", "wrote", "composed", "painted", "fought",
    "founded", "created", "developed", "published", "won", "awarded",
    "biography", "autobiography", "legacy", "career", "achievement",
    "achievements", "nationality", "education", "personality",
})

PLACE_KEYWORDS: frozenset[str] = frozenset({
    # Geographical terms
    "city", "country", "nation", "continent", "island", "ocean", "sea",
    "river", "lake", "mountain", "desert", "forest", "park", "region",
    "valley", "canyon", "waterfall", "volcano", "peninsula", "bay",
    # Built structures
    "tower", "bridge", "palace", "castle", "temple", "mosque", "church",
    "cathedral", "monument", "museum", "stadium", "building", "landmark",
    "ruins", "site", "pyramid", "wall", "fort", "harbour", "harbour",
    # Location language
    "located", "situated", "built", "constructed", "established",
    "visit", "visited", "travel", "tourism", "tourists", "attraction",
    "geography", "area", "region", "capital", "population", "climate",
    "height", "altitude", "length", "size", "UNESCO", "heritage",
    "location", "landmark",
})

# ── Configuration ──────────────────────────────────────────────────────────────

_SETTINGS_PATH = Path("config/settings.yaml")
_MANIFEST_PATH_DEFAULT = Path("./data/manifest.db")
_LLM_MODEL = "llama3.2:3b"

_CLASSIFICATION_PROMPT = (
    "Classify the following question as PERSON (about a specific person) "
    "or PLACE (about a specific location, landmark, or geographical feature). "
    "Respond with exactly one word: PERSON or PLACE.\n\n"
    "Question: {query}"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation for keyword matching."""
    return re.sub(r"[^\w\s]", " ", text.lower())


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:20]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_manifest_path() -> Path:
    try:
        with _SETTINGS_PATH.open() as fh:
            cfg = yaml.safe_load(fh)
        return Path(cfg.get("manifest_path", str(_MANIFEST_PATH_DEFAULT)))
    except Exception:  # noqa: BLE001
        return _MANIFEST_PATH_DEFAULT


# ── Manifest-backed entity name lookup ────────────────────────────────────────

_ENTITY_TYPES: dict[str, str] | None = None  # lazy-loaded cache


def _get_entity_types() -> dict[str, str]:
    """Return {entity_name: entity_type} from the manifest (loaded once)."""
    global _ENTITY_TYPES  # noqa: PLW0603
    if _ENTITY_TYPES is not None:
        return _ENTITY_TYPES

    db_path = _load_manifest_path()
    if not db_path.exists():
        _ENTITY_TYPES = {}
        return _ENTITY_TYPES

    try:
        con = sqlite3.connect(str(db_path), check_same_thread=False)
        rows = con.execute("SELECT name, entity_type FROM entities").fetchall()
        con.close()
        _ENTITY_TYPES = {name.lower(): etype for name, etype in rows}
    except Exception:  # noqa: BLE001
        _ENTITY_TYPES = {}

    return _ENTITY_TYPES


# ── SQLite router cache ────────────────────────────────────────────────────────

def _cache_lookup(db_path: Path, qhash: str) -> Literal["person", "place", "both"] | None:
    if not db_path.exists():
        return None
    try:
        con = sqlite3.connect(str(db_path), check_same_thread=False)
        row = con.execute(
            "SELECT category FROM router_cache WHERE query_hash = ?", (qhash,)
        ).fetchone()
        con.close()
        return row[0] if row else None  # type: ignore[return-value]
    except Exception:  # noqa: BLE001
        return None


def _cache_store(
    db_path: Path,
    qhash: str,
    category: Literal["person", "place", "both"],
) -> None:
    if not db_path.exists():
        return
    try:
        con = sqlite3.connect(str(db_path), check_same_thread=False)
        con.execute(
            """
            INSERT INTO router_cache (query_hash, category, cached_at)
            VALUES (?, ?, ?)
            ON CONFLICT(query_hash) DO UPDATE SET
                category  = excluded.category,
                cached_at = excluded.cached_at
            """,
            (qhash, category, _now_utc()),
        )
        con.commit()
        con.close()
    except Exception:  # noqa: BLE001
        pass


# ── Tier 1 — Lexical/Heuristic ────────────────────────────────────────────────

def _tier1(query: str) -> Literal["person", "place", "both"] | None:
    """
    Return a category if the heuristic is confident, else None.

    Confidence rules:
    - Exact entity-name match → immediately return that category.
    - Keyword scoring: person_score vs place_score.
      If one is > 0 and the other is 0 → return the winner.
      If both > 0 or both 0 → None (ambiguous, escalate to Tier 2).
    """
    norm = _normalise(query)
    words = set(norm.split())

    # 1a. Exact name match against known entities (case-insensitive)
    entity_types = _get_entity_types()
    for name, etype in entity_types.items():
        # Check if all words of the entity name appear in the query
        name_words = set(name.split())
        if name_words and name_words.issubset(words):
            return etype  # type: ignore[return-value]

    # 1b. Keyword scoring
    person_score = len(words & PERSON_KEYWORDS)
    place_score = len(words & PLACE_KEYWORDS)

    if person_score > 0 and place_score == 0:
        return "person"
    if place_score > 0 and person_score == 0:
        return "place"

    # Ambiguous or no keywords — escalate
    return None


# ── Tier 2 — LLM fallback ────────────────────────────────────────────────────

def _tier2(query: str, db_path: Path) -> Literal["person", "place", "both"]:
    """
    Ask llama3.2:3b to classify the query.  Result is cached in SQLite.
    Falls back to 'both' on any error.
    """
    qhash = _query_hash(query)

    # Check cache first
    cached = _cache_lookup(db_path, qhash)
    if cached is not None:
        return cached

    try:
        response = ollama.chat(
            model=_LLM_MODEL,
            messages=[{"role": "user", "content": _CLASSIFICATION_PROMPT.format(query=query)}],
            options={"temperature": 0, "num_predict": 5},
        )
        if isinstance(response, dict):
            raw = response.get("message", {}).get("content", "")
        else:
            raw = response.message.content  # type: ignore[union-attr]

        raw = raw.strip().upper()
        if "PERSON" in raw:
            category: Literal["person", "place", "both"] = "person"
        elif "PLACE" in raw:
            category = "place"
        else:
            category = "both"

    except Exception:  # noqa: BLE001 — Ollama not running, model not loaded, etc.
        category = "both"

    _cache_store(db_path, qhash, category)
    return category


# ── Public API ────────────────────────────────────────────────────────────────

def detect_category(
    query: str,
    *,
    use_llm_fallback: bool = True,
) -> Literal["person", "place", "both"]:
    """
    Classify a natural-language query as 'person', 'place', or 'both'.

    Args:
        query:           The raw user query string.
        use_llm_fallback: If False, skip Tier 2 (useful in tests / batch mode).

    Returns:
        'person' — query is about a specific person.
        'place'  — query is about a location, landmark, or geographical feature.
        'both'   — ambiguous; retrieval will search across both categories.
    """
    if not query or not query.strip():
        return "both"

    # Tier 1 — fast path
    result = _tier1(query)
    if result is not None:
        return result

    # Tier 2 — LLM fallback
    if use_llm_fallback:
        db_path = _load_manifest_path()
        return _tier2(query, db_path)

    return "both"


def reset_entity_cache() -> None:
    """Force reload of entity names from the manifest (useful after ingest)."""
    global _ENTITY_TYPES  # noqa: PLW0603
    _ENTITY_TYPES = None
