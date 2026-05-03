"""
tests/test_prompts.py — Prompt structure snapshot and refusal-logic tests.

These tests verify that:
  1. The prompt template version is stable (PROMPT_VERSION == "v1").
  2. The rendered prompt contains all required structural elements.
  3. The refusal clause is present verbatim in the template.
  4. Prompt assembly from RetrievedChunk objects works correctly.
  5. unique_sources deduplication is correct.
  6. generator.generate_answer() short-circuits on empty contexts WITHOUT calling Ollama.

Run with:
    pytest tests/test_prompts.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.prompts import (  # noqa: E402
    PROMPT_VERSION,
    REFUSAL_STRING,
    SYSTEM_MESSAGE,
    build_answer_prompt,
    build_context_block,
    unique_sources,
)
from src.rag.types import RetrievedChunk  # noqa: E402 — avoids chromadb import

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_chunk(
    entity_name: str = "Albert Einstein",
    entity_type: str = "person",
    text: str = "Einstein was born in Ulm, Germany on 14 March 1879.",
    distance: float = 0.25,
    source_url: str = "https://en.wikipedia.org/wiki/Albert_Einstein",
    chunk_index: int = 0,
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        entity_name=entity_name,
        entity_type=entity_type,
        distance=distance,
        source_url=source_url,
        chunk_index=chunk_index,
    )


CHUNK_PERSON = _make_chunk()
CHUNK_PLACE = _make_chunk(
    entity_name="Eiffel Tower",
    entity_type="place",
    text="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
    source_url="https://en.wikipedia.org/wiki/Eiffel_Tower",
    chunk_index=0,
)
CHUNK_PERSON_2 = _make_chunk(chunk_index=1, text="He developed the theory of relativity.")


# ── Version snapshot ──────────────────────────────────────────────────────────

class TestPromptVersion:
    def test_version_is_v1(self):
        """Lock the prompt version to v1 — bump when template changes."""
        assert PROMPT_VERSION == "v1", (
            f"PROMPT_VERSION changed to '{PROMPT_VERSION}'. "
            "Update this snapshot test to acknowledge the change."
        )


# ── Refusal string ────────────────────────────────────────────────────────────

class TestRefusalString:
    def test_refusal_string_non_empty(self):
        assert REFUSAL_STRING.strip()

    def test_refusal_string_in_system_message(self):
        assert REFUSAL_STRING in SYSTEM_MESSAGE

    def test_refusal_string_in_answer_template(self):
        rendered = build_answer_prompt("anything?", [CHUNK_PERSON])
        assert REFUSAL_STRING in rendered

    def test_refusal_string_exact(self):
        """Exact wording lock — model is instructed to output this verbatim."""
        assert REFUSAL_STRING == "I don't know based on the available information."


# ── Context block assembly ────────────────────────────────────────────────────

class TestContextBlock:
    def test_single_chunk_contains_entity_name(self):
        block = build_context_block([CHUNK_PERSON])
        assert "Albert Einstein" in block

    def test_single_chunk_contains_text(self):
        block = build_context_block([CHUNK_PERSON])
        assert "born in Ulm" in block

    def test_single_chunk_contains_entity_type(self):
        block = build_context_block([CHUNK_PERSON])
        assert "person" in block

    def test_multiple_chunks_numbered(self):
        block = build_context_block([CHUNK_PERSON, CHUNK_PLACE])
        assert "[1]" in block
        assert "[2]" in block

    def test_empty_chunks_returns_empty(self):
        block = build_context_block([])
        assert block == ""


# ── Full prompt structure ─────────────────────────────────────────────────────

class TestBuildAnswerPrompt:
    def test_prompt_contains_question(self):
        q = "Where was Einstein born?"
        prompt = build_answer_prompt(q, [CHUNK_PERSON])
        assert q in prompt

    def test_prompt_contains_context(self):
        prompt = build_answer_prompt("test?", [CHUNK_PERSON])
        assert "born in Ulm" in prompt

    def test_prompt_contains_refusal_clause(self):
        prompt = build_answer_prompt("test?", [CHUNK_PERSON])
        assert REFUSAL_STRING in prompt

    def test_prompt_contains_answer_label(self):
        prompt = build_answer_prompt("test?", [CHUNK_PERSON])
        assert "Answer:" in prompt

    def test_prompt_contains_context_label(self):
        prompt = build_answer_prompt("test?", [CHUNK_PERSON])
        assert "Context:" in prompt

    def test_prompt_contains_question_label(self):
        prompt = build_answer_prompt("test?", [CHUNK_PERSON])
        assert "Question:" in prompt

    def test_system_message_contains_source_isolation(self):
        assert "ONLY" in SYSTEM_MESSAGE
        assert "prior knowledge" in SYSTEM_MESSAGE


# ── unique_sources deduplication ──────────────────────────────────────────────

class TestUniqueSources:
    def test_no_duplicates(self):
        """Two chunks from the same entity produce one source entry."""
        sources = unique_sources([CHUNK_PERSON, CHUNK_PERSON_2])
        names = [s[0] for s in sources]
        assert names.count("Albert Einstein") == 1

    def test_preserves_order(self):
        sources = unique_sources([CHUNK_PERSON, CHUNK_PLACE])
        assert sources[0][0] == "Albert Einstein"
        assert sources[1][0] == "Eiffel Tower"

    def test_empty_input(self):
        assert unique_sources([]) == []

    def test_returns_all_three_fields(self):
        sources = unique_sources([CHUNK_PERSON])
        assert len(sources[0]) == 3
        name, etype, url = sources[0]
        assert name == "Albert Einstein"
        assert etype == "person"
        assert "wikipedia" in url


# ── Generator short-circuit (no Ollama required) ──────────────────────────────

class TestGeneratorShortCircuit:
    def test_empty_contexts_returns_refusal(self):
        """generate_answer must NOT call ollama when contexts is empty."""
        from src.rag.generator import generate_answer  # noqa: PLC0415

        with patch("src.rag.generator.ollama") as mock_ollama:
            result = generate_answer("What is the speed of light?", contexts=[])
            mock_ollama.chat.assert_not_called()

        assert result == REFUSAL_STRING

    def test_stream_empty_contexts_yields_refusal(self):
        """stream_answer must yield only the refusal string on empty contexts."""
        from src.rag.generator import stream_answer  # noqa: PLC0415

        with patch("src.rag.generator.ollama") as mock_ollama:
            tokens = list(stream_answer("What is the capital of Mars?", contexts=[]))
            mock_ollama.chat.assert_not_called()

        assert "".join(tokens) == REFUSAL_STRING
