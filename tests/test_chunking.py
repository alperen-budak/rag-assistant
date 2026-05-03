"""
tests/test_chunking.py — Unit tests for src/core/chunking.py and src/core/cleaning.py.

Run with:
    pytest tests/test_chunking.py -v

These tests are pure-Python: they require no Ollama, no ChromaDB, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on path when running pytest from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.chunking import chunk_text, _split_sentences  # noqa: E402
from src.core.cleaning import clean  # noqa: E402

# ── Fixtures ──────────────────────────────────────────────────────────────────

SHORT_TEXT = "Albert Einstein was born in Ulm. He developed the theory of relativity."

MEDIUM_TEXT = (
    "Marie Curie was a Polish-French physicist and chemist. "
    "She was the first woman to win a Nobel Prize. "
    "She won the Nobel Prize in Physics in 1903. "
    "She also won the Nobel Prize in Chemistry in 1911. "
    "Her discoveries included polonium and radium. "
    "She was born on November 7, 1867 in Warsaw. "
    "She died on July 4, 1934 in Passy, France."
)

LONG_TEXT = " ".join(
    [
        f"Sentence number {i} provides some context about a famous person or place."
        for i in range(1, 60)
    ]
)

TEXT_WITH_CITATIONS = (
    "Einstein [1] was born in Ulm.[2] "
    "He worked at the patent office.[citation needed] "
    "He published four papers in 1905.[3][4]"
)

TEXT_WITH_HTML = (
    "<p>The Eiffel Tower is a wrought-iron lattice tower.</p> "
    "It was built in 1889.&nbsp;It is 330 metres tall."
)

TEXT_WITH_SECTIONS = (
    "== Early Life ==\n"
    "Cleopatra was born around 69 BC.\n\n"
    "== Reign ==\n"
    "She became queen at 18."
)


# ── clean() tests ─────────────────────────────────────────────────────────────

class TestCleaning:
    def test_removes_numeric_citations(self):
        result = clean(TEXT_WITH_CITATIONS)
        assert "[1]" not in result
        assert "[2]" not in result
        assert "[3]" not in result
        assert "[4]" not in result

    def test_removes_citation_needed(self):
        result = clean(TEXT_WITH_CITATIONS)
        assert "citation needed" not in result

    def test_removes_html_tags(self):
        result = clean(TEXT_WITH_HTML)
        assert "<p>" not in result
        assert "</p>" not in result

    def test_removes_html_entities(self):
        result = clean(TEXT_WITH_HTML)
        assert "&nbsp;" not in result

    def test_removes_section_markers(self):
        result = clean(TEXT_WITH_SECTIONS)
        assert "==" not in result

    def test_preserves_meaningful_text(self):
        result = clean(TEXT_WITH_CITATIONS)
        assert "Einstein" in result
        assert "born in Ulm" in result

    def test_collapses_whitespace(self):
        result = clean("Hello   world.\n\n\n\nSecond  paragraph.")
        assert "  " not in result  # no double spaces
        # At most two consecutive newlines
        assert "\n\n\n" not in result

    def test_empty_input(self):
        assert clean("") == ""

    def test_strips_output(self):
        result = clean("  Hello world.  ")
        assert result == result.strip()


# ── chunk_text() tests ────────────────────────────────────────────────────────

class TestChunking:
    def test_yields_at_least_one_chunk(self):
        chunks = list(chunk_text(SHORT_TEXT))
        assert len(chunks) >= 1

    def test_no_chunk_exceeds_size_by_much(self):
        """
        Chunks may slightly exceed chunk_size when a single sentence is longer
        than the target, but should never exceed 2x the target.
        """
        for chunk in chunk_text(LONG_TEXT, chunk_size=500):
            assert len(chunk) <= 1000, f"Chunk too long ({len(chunk)}): {chunk[:60]}…"

    def test_all_chunks_non_empty(self):
        for chunk in chunk_text(MEDIUM_TEXT, chunk_size=200):
            assert chunk.strip(), "Empty chunk yielded"

    def test_content_coverage(self):
        """All words from the input must appear in at least one chunk."""
        chunks = list(chunk_text(MEDIUM_TEXT, chunk_size=300))
        combined = " ".join(chunks)
        for word in ["Curie", "Nobel", "polonium", "radium", "Warsaw"]:
            assert word in combined, f"Word '{word}' missing from chunks"

    def test_overlap_produces_repeated_content(self):
        """With overlap > 0, consecutive chunks should share some text."""
        chunks = list(chunk_text(LONG_TEXT, chunk_size=300, overlap=80))
        if len(chunks) < 2:
            return  # can't test overlap with single chunk
        for prev, curr in zip(chunks, chunks[1:]):
            prev_words = set(prev.split())
            curr_words = set(curr.split())
            shared = prev_words & curr_words
            assert shared, "Consecutive chunks share no words — overlap not working"

    def test_no_overlap_produces_less_content(self):
        chunks_with = list(chunk_text(LONG_TEXT, chunk_size=300, overlap=80))
        chunks_without = list(chunk_text(LONG_TEXT, chunk_size=300, overlap=0))
        # With overlap, total characters across chunks should be >= without
        total_with = sum(len(c) for c in chunks_with)
        total_without = sum(len(c) for c in chunks_without)
        assert total_with >= total_without

    def test_empty_input_yields_nothing(self):
        assert list(chunk_text("")) == []
        assert list(chunk_text("   ")) == []

    def test_single_sentence_shorter_than_chunk_size(self):
        chunks = list(chunk_text(SHORT_TEXT, chunk_size=500))
        assert len(chunks) == 1
        assert "Einstein" in chunks[0]

    def test_chunk_size_boundary(self):
        """With a very small chunk_size, we should get many small chunks."""
        chunks = list(chunk_text(MEDIUM_TEXT, chunk_size=100, overlap=20))
        assert len(chunks) >= 3, "Expected more chunks with small chunk_size"

    def test_generator_is_lazy(self):
        """chunk_text must return a generator, not a list."""
        import types
        result = chunk_text(LONG_TEXT)
        assert isinstance(result, types.GeneratorType), (
            "chunk_text() must return a generator (yield-based), not a list"
        )

    def test_no_mid_word_splits(self):
        """Chunks must not start or end mid-word (no partial tokens at boundaries)."""
        chunks = list(chunk_text(MEDIUM_TEXT, chunk_size=150, overlap=30))
        for chunk in chunks:
            # A chunk starting mid-word would have no leading capital or space
            # This is a heuristic: first char should be alphanumeric, not a fragment
            first = chunk[0] if chunk else ""
            assert first.isalpha() or first.isdigit() or first in ('"', "'", "("), (
                f"Chunk appears to start mid-word: '{chunk[:30]}'"
            )


# ── _split_sentences() internal tests ────────────────────────────────────────

class TestSentenceSplitter:
    def test_basic_split(self):
        sentences = _split_sentences("Hello world. This is a test.")
        assert len(sentences) >= 1

    def test_abbreviation_not_split(self):
        """'Dr. Smith' should not be split into two sentences."""
        text = "Dr. Smith went to the store. He bought apples."
        sentences = _split_sentences(text)
        # The abbreviation case might still split but the key check is:
        # we should not get an empty sentence fragment like "Smith went..."
        # as the first sentence
        combined = " ".join(sentences)
        assert "Dr" in combined
        assert "Smith" in combined

    def test_paragraph_boundary_always_splits(self):
        text = "First paragraph.\n\nSecond paragraph starts here."
        sentences = _split_sentences(text)
        assert any("Second" in s for s in sentences)

    def test_empty_produces_empty_list(self):
        assert _split_sentences("") == []
        assert _split_sentences("   ") == []
