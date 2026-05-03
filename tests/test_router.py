"""
tests/test_router.py — Unit tests for src/rag/router.py.

All tests run with `use_llm_fallback=False` so no Ollama daemon is required.
They exercise only the Tier 1 heuristic path.

Run with:
    pytest tests/test_router.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.router import (  # noqa: E402
    _normalise,
    _tier1,
    detect_category,
)

# ── Helper ────────────────────────────────────────────────────────────────────

def classify(query: str) -> str:
    """Run only Tier 1 (no LLM) and fall back to 'both' if ambiguous."""
    result = _tier1(query)
    return result if result is not None else "both"


# ── Person-biased queries (6) ─────────────────────────────────────────────────

class TestPersonQueries:
    def test_born_keyword(self):
        assert classify("Where was Einstein born?") == "person"

    def test_she_pronoun(self):
        assert classify("What did she discover?") == "person"

    def test_scientist_keyword(self):
        assert classify("Who was the greatest scientist of the 20th century?") == "person"

    def test_died_keyword(self):
        assert classify("When did Newton die?") == "person"

    def test_biography_keyword(self):
        assert classify("Tell me about the biography of Marie Curie.") == "person"

    def test_invented_keyword(self):
        assert classify("Who invented the telephone?") == "person"


# ── Place-biased queries (6) ──────────────────────────────────────────────────

class TestPlaceQueries:
    def test_tower_keyword(self):
        assert classify("How tall is the Eiffel Tower?") == "place"

    def test_located_keyword(self):
        assert classify("In which country is the Colosseum located?") == "place"

    def test_built_keyword(self):
        assert classify("When was the Great Wall of China built?") == "place"

    def test_tourists_keyword(self):
        assert classify("How many tourists visit Machu Picchu each year?") == "place"

    def test_city_keyword(self):
        assert classify("What is the population of this city?") == "place"

    def test_UNESCO_keyword(self):
        assert classify("Is Angkor Wat a UNESCO heritage site?") == "place"


# ── Ambiguous / both (2) ──────────────────────────────────────────────────────

class TestAmbiguousQueries:
    def test_no_keywords(self):
        """Pure content query with no person/place markers → 'both'."""
        result = classify("What happened in 1889?")
        assert result == "both"

    def test_mixed_keywords(self):
        """Query with both person and place markers → 'both'."""
        result = classify("Who built this famous city and why was she born there?")
        assert result == "both"


# ── Normalisation util ────────────────────────────────────────────────────────

class TestNormalise:
    def test_lowercases(self):
        assert _normalise("HELLO WORLD") == "hello world"

    def test_strips_punctuation(self):
        result = _normalise("Hello, world! What's happening?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_preserves_words(self):
        result = _normalise("Einstein was born in Ulm.")
        assert "einstein" in result
        assert "born" in result
        assert "ulm" in result


# ── detect_category without LLM ──────────────────────────────────────────────

class TestDetectCategoryNoLLM:
    def test_returns_valid_literal(self):
        result = detect_category("Who invented electricity?", use_llm_fallback=False)
        assert result in ("person", "place", "both")

    def test_empty_query_returns_both(self):
        assert detect_category("", use_llm_fallback=False) == "both"

    def test_whitespace_only_returns_both(self):
        assert detect_category("   ", use_llm_fallback=False) == "both"

    def test_person_query_without_llm(self):
        result = detect_category("When was she born?", use_llm_fallback=False)
        assert result == "person"

    def test_place_query_without_llm(self):
        result = detect_category("How tall is this tower?", use_llm_fallback=False)
        assert result == "place"
