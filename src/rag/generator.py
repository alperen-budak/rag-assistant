"""
src/rag/generator.py — Llama 3.2 3B answer generator via Ollama.

MODULE CONTRACT (PROJECT.md §6):
    generate_answer(query: str, contexts: list[RetrievedChunk]) -> str

Key behaviours:
  - If `contexts` is empty → short-circuit and return REFUSAL_STRING immediately.
    This saves latency and prevents the model from hallucinating without grounding.
  - Streaming mode: tokens are printed to stdout as they arrive so the user
    sees output immediately on the M1 (warm 3B model ≈ 20–40 tok/s via Metal).
  - Non-streaming fallback available via `stream=False` for tests / batch use.
  - ollama==0.3.x returns dicts; newer versions return typed objects — both handled.
"""

from __future__ import annotations

from typing import Iterator

import ollama

from src.rag.prompts import (
    REFUSAL_STRING,
    SYSTEM_MESSAGE,
    build_answer_prompt,
)
from src.rag.types import RetrievedChunk

_MODEL = "llama3.2:3b"

# Generation parameters tuned for M1 (predictable memory, focused answers)
_OPTIONS = {
    "temperature": 0.1,       # low temperature → factual, less creative
    "top_p": 0.9,
    "num_predict": 512,        # max tokens in the answer
    "num_ctx": 4096,           # context window (fits ~5 chunks + question)
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_content(chunk) -> str:  # type: ignore[no-untyped-def]
    """Extract the text content from an Ollama streaming or final chunk."""
    if isinstance(chunk, dict):
        return chunk.get("message", {}).get("content", "")
    # Typed object (future versions)
    msg = getattr(chunk, "message", None)
    if msg is None:
        return ""
    return getattr(msg, "content", "") or ""


def _build_messages(query: str, chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user",   "content": build_answer_prompt(query, chunks)},
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def stream_answer(
    query: str,
    contexts: list[RetrievedChunk],
) -> Iterator[str]:
    """
    Yield answer tokens one by one as the model generates them.

    If `contexts` is empty, yields the refusal string as a single token
    without invoking the LLM (instant response, zero model overhead).

    Usage (in CLI):
        for token in stream_answer(query, chunks):
            print(token, end="", flush=True)
        print()
    """
    if not contexts:
        yield REFUSAL_STRING
        return

    messages = _build_messages(query, contexts)
    try:
        stream = ollama.chat(
            model=_MODEL,
            messages=messages,
            stream=True,
            options=_OPTIONS,
        )
        for part in stream:
            token = _extract_content(part)
            if token:
                yield token
    except Exception as exc:  # noqa: BLE001
        yield f"\n[Generation error: {exc}]"


def generate_answer(
    query: str,
    contexts: list[RetrievedChunk],
    *,
    stream: bool = False,
) -> str:
    """
    Return the complete answer string.

    Args:
        query:    Raw user query.
        contexts: Retrieved chunks from `retriever.retrieve()`.
        stream:   If True, stream tokens to stdout while accumulating the full
                  string (best for interactive CLI use).
                  If False, wait for the full response (used in tests).

    Returns:
        The model's answer string, or REFUSAL_STRING if contexts is empty.
    """
    if not contexts:
        return REFUSAL_STRING

    if stream:
        # Stream to stdout AND accumulate for return
        tokens: list[str] = []
        for token in stream_answer(query, contexts):
            print(token, end="", flush=True)
            tokens.append(token)
        print()  # newline after streaming ends
        return "".join(tokens)

    # Non-streaming (batch / test mode)
    messages = _build_messages(query, contexts)
    try:
        response = ollama.chat(
            model=_MODEL,
            messages=messages,
            stream=False,
            options=_OPTIONS,
        )
        if isinstance(response, dict):
            return response.get("message", {}).get("content", REFUSAL_STRING).strip()
        return (response.message.content or REFUSAL_STRING).strip()  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        return f"[Generation error: {exc}]"
