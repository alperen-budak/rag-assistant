"""
src/rag/prompts.py — Versioned prompt templates (PROJECT.md §3.4).

The ANSWER_PROMPT enforces three anti-hallucination rules:
  1. Source isolation  — "Use ONLY the context below."
  2. Refusal clause   — exact refusal string when answer not in context.
  3. Citation         — model must name the entity its answer is drawn from.

PROMPT_VERSION must be bumped whenever the template structure changes.
The snapshot test in tests/test_prompts.py is keyed to this version.
"""

from __future__ import annotations

from src.rag.types import RetrievedChunk

PROMPT_VERSION = "v1"

# The exact string the model must output (and the generator short-circuits to)
# when no grounded answer is available.
REFUSAL_STRING = "I don't know based on the available information."

# ── System message ─────────────────────────────────────────────────────────────

SYSTEM_MESSAGE = (
    "You are a precise question-answering assistant. "
    "You answer questions ONLY using the provided context passages. "
    "You do NOT use any prior knowledge, training data, or external sources. "
    "If the answer cannot be found in the context, you respond with exactly: "
    f'"{REFUSAL_STRING}" '
    "You always cite which person or place your answer is about."
)

# ── Context block template ─────────────────────────────────────────────────────

_CONTEXT_BLOCK_TEMPLATE = "[{idx}] ({entity_type}: {entity_name})\n{text}"

# ── Full answer prompt template ────────────────────────────────────────────────

_ANSWER_TEMPLATE = """\
Use ONLY the context passages below to answer the question.
Do NOT use any knowledge outside of these passages.
If the answer is not present in the passages, respond with exactly:
"{refusal}"

Context:
{context_block}

Question: {question}

Answer:"""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.
    De-duplicates by entity_name to avoid repeating the same source label.
    """
    parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        parts.append(
            _CONTEXT_BLOCK_TEMPLATE.format(
                idx=idx,
                entity_type=chunk.entity_type,
                entity_name=chunk.entity_name,
                text=chunk.text.strip(),
            )
        )
    return "\n\n".join(parts)


def build_answer_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """
    Return the complete user-turn prompt string for the answer generation call.

    Args:
        question: Raw user query.
        chunks:   Retrieved context chunks (already filtered and sorted).

    Returns:
        Fully formatted prompt string.  Feed this as the `user` message;
        pair it with SYSTEM_MESSAGE as the `system` message.
    """
    context_block = build_context_block(chunks)
    return _ANSWER_TEMPLATE.format(
        refusal=REFUSAL_STRING,
        context_block=context_block,
        question=question,
    )


def unique_sources(chunks: list[RetrievedChunk]) -> list[tuple[str, str, str]]:
    """
    Return deduplicated (entity_name, entity_type, source_url) triples
    from the chunk list, preserving first-occurrence order.
    Used by the CLI to print the Sources footer.
    """
    seen: set[str] = set()
    result: list[tuple[str, str, str]] = []
    for chunk in chunks:
        if chunk.entity_name not in seen:
            seen.add(chunk.entity_name)
            result.append((chunk.entity_name, chunk.entity_type, chunk.source_url))
    return result
