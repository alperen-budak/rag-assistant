"""
src/ingest/wiki_fetcher.py — Wikipedia article fetcher with file-level caching.

Wraps `wikipedia-api` to:
  - Fetch the full plaintext summary + content for a given Wikipedia page title.
  - Cache raw results to `data/raw/<wiki_title>.json` so re-runs don't re-hit the API.
  - Handle disambiguation and missing pages gracefully (returns None).

Public API
----------
    fetch(wiki_title: str, cache_dir: str | Path, *, force: bool = False) -> FetchResult | None

FetchResult fields:
    title      str   — canonical Wikipedia page title
    url        str   — full Wikipedia URL
    text       str   — full article text (summary + sections concatenated)
    summary    str   — lead section only
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import wikipediaapi

# Retry config for transient network errors
_FETCH_MAX_RETRIES = 3
_FETCH_RETRY_DELAY_SECONDS = 5.0

_PathLike = Union[str, Path]

_WIKI = wikipediaapi.Wikipedia(
    language="en",
    user_agent="aided-hw3-rag/1.0 (BLG483E homework; contact: student)",
    extract_format=wikipediaapi.ExtractFormat.WIKI,
)

# Rate-limit courtesy: wait between API calls when fetching live
_FETCH_DELAY_SECONDS = 0.5


@dataclass
class FetchResult:
    title: str
    url: str
    text: str
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FetchResult":
        return cls(**d)


def _cache_path(wiki_title: str, cache_dir: Path) -> Path:
    safe_name = wiki_title.replace("/", "_").replace(" ", "_")
    return cache_dir / f"{safe_name}.json"


def _load_from_cache(cache_file: Path) -> FetchResult | None:
    if not cache_file.exists():
        return None
    try:
        with cache_file.open("r", encoding="utf-8") as fh:
            return FetchResult.from_dict(json.load(fh))
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _save_to_cache(result: FetchResult, cache_file: Path) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, ensure_ascii=False, indent=2)


def _build_full_text(page: wikipediaapi.WikipediaPage) -> str:
    """
    Concatenate summary + all section texts into a single string.
    Sections are separated by double newlines to preserve paragraph structure
    for the sentence-aware chunker.
    """
    parts: list[str] = []

    if page.summary:
        parts.append(page.summary.strip())

    def _walk(sections: list) -> None:
        for section in sections:
            if section.text.strip():
                parts.append(section.text.strip())
            _walk(section.sections)

    _walk(page.sections)
    return "\n\n".join(parts)


def fetch(
    wiki_title: str,
    cache_dir: _PathLike,
    *,
    force: bool = False,
) -> FetchResult | None:
    """
    Fetch a Wikipedia article, returning a `FetchResult` or `None` on failure.

    Args:
        wiki_title: Exact Wikipedia page title (from entities.yaml).
        cache_dir:  Directory for JSON cache files (e.g. ``data/raw``).
        force:      If True, bypass the cache and re-fetch from Wikipedia.

    Returns:
        FetchResult on success, None if the page is missing or a disambiguation page.
    """
    cache_dir = Path(cache_dir)
    cache_file = _cache_path(wiki_title, cache_dir)

    if not force:
        cached = _load_from_cache(cache_file)
        if cached is not None:
            return cached

    # Live fetch with retry loop for transient network errors
    last_exc: Exception | None = None
    for attempt in range(1, _FETCH_MAX_RETRIES + 1):
        try:
            time.sleep(_FETCH_DELAY_SECONDS)
            page = _WIKI.page(wiki_title)

            if not page.exists():
                return None

            # Disambiguation pages — detect by common category membership
            categories_lower = {c.lower() for c in page.categories}
            if any("disambiguation" in c for c in categories_lower):
                return None

            full_text = _build_full_text(page)
            if not full_text.strip():
                return None

            result = FetchResult(
                title=page.title,
                url=page.fullurl,
                text=full_text,
                summary=page.summary,
            )
            _save_to_cache(result, cache_file)
            return result

        except Exception as exc:  # noqa: BLE001 — network errors, timeouts, etc.
            last_exc = exc
            if attempt < _FETCH_MAX_RETRIES:
                wait = _FETCH_RETRY_DELAY_SECONDS * attempt
                print(
                    f"    WARN: fetch attempt {attempt}/{_FETCH_MAX_RETRIES} failed "
                    f"({type(exc).__name__}), retrying in {wait:.0f}s…",
                    flush=True,
                )
                time.sleep(wait)

    print(
        f"    ERROR: all {_FETCH_MAX_RETRIES} fetch attempts failed for '{wiki_title}': {last_exc}",
        flush=True,
    )
    return None
