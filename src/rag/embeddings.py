"""
src/rag/embeddings.py — Batched Ollama embedding client.

MODULE CONTRACT (PROJECT.md §6):
    embed(texts: list[str], model: str = "nomic-embed-text") -> list[list[float]]

Design constraints (PROJECT.md §4.2):
  - Batch size capped at `EMBED_BATCH_SIZE` (default 8) to stay within M1 RAM budget.
  - Retry loop for transient Ollama daemon errors (model not yet loaded, timeout).
  - No in-process model caching — models live in the Ollama daemon.
"""

from __future__ import annotations

import time
from typing import Iterator

import ollama

# Maximum chunks per Ollama embed call (PROJECT.md §4.2)
EMBED_BATCH_SIZE = 8

# Retry configuration for transient daemon errors
_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 2.0


def _batches(items: list[str], size: int) -> Iterator[list[str]]:
    """Yield successive `size`-length sublists."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _embed_batch(texts: list[str], model: str) -> list[list[float]]:
    """
    Call Ollama to embed a single batch of texts.  Retries up to `_MAX_RETRIES`
    times on connection or model-loading errors.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = ollama.embed(model=model, input=texts)
            # ollama==0.3.x returns a plain dict; newer versions return a typed
            # object.  Support both to be forward-compatible.
            if isinstance(response, dict):
                return response["embeddings"]
            return response.embeddings  # type: ignore[return-value]
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_SECONDS * attempt)
    raise RuntimeError(
        f"Ollama embed failed after {_MAX_RETRIES} attempts: {last_exc}"
    ) from last_exc


def embed(
    texts: list[str],
    model: str = "nomic-embed-text",
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """
    Return a list of embedding vectors for the given texts.

    Texts are sent to the Ollama daemon in batches of `batch_size` to avoid
    spiking unified memory on the M1.

    Args:
        texts:      List of strings to embed.  Must be non-empty.
        model:      Ollama model name (default ``nomic-embed-text``).
        batch_size: Max texts per Ollama call (default 8).

    Returns:
        List of float vectors in the same order as ``texts``.
        Each vector has 768 dimensions (nomic-embed-text).

    Raises:
        ValueError:   If `texts` is empty.
        RuntimeError: If Ollama is unreachable after retries.
    """
    if not texts:
        raise ValueError("embed() received an empty list of texts")

    vectors: list[list[float]] = []
    for batch in _batches(texts, batch_size):
        batch_vectors = _embed_batch(batch, model)
        vectors.extend(batch_vectors)

    return vectors


def embed_one(text: str, model: str = "nomic-embed-text") -> list[float]:
    """Convenience wrapper to embed a single string and return its vector."""
    return embed([text], model=model)[0]
