"""
src/core/chunking.py — Sentence-aware text chunker.

Zero external dependencies: pure re + str only (PROJECT.md §3.3).

Public API (MODULE CONTRACT — do not change signatures without updating callers):
    chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> Iterator[str]

Design:
    1. Split the cleaned text into sentences using a compiled regex that handles
       common abbreviations (Mr., Dr., U.S., etc.) without false-splitting.
    2. Greedily accumulate sentences into a window until adding the next sentence
       would exceed `chunk_size` characters.
    3. When a window is full, yield it and start a new window that begins with
       the last `overlap` characters of the previous window (overlap is applied
       at the character level, aligning to the nearest sentence boundary within
       ±overlap chars to avoid mid-sentence cuts).
    4. The final window is always yielded even if it is shorter than chunk_size.
"""

from __future__ import annotations

import re
from typing import Iterator

# ── Sentence boundary detector ────────────────────────────────────────────────
#
# Strategy: tokenise on any [.!?] followed by optional quotes and whitespace
# then an uppercase letter.  Post-process each candidate boundary to reject
# splits that follow a known abbreviation or a bare initial (e.g. "Dr.", "A.").
#
# Python's `re` only supports fixed-width lookbehinds, so abbreviation
# checking is done imperatively after splitting.

_ABBREVS: frozenset[str] = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs",
    "etc", "est", "approx", "dept", "govt", "corp", "inc",
    "no", "vol", "fig", "ph", "st", "mt", "ft", "sq",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "oct", "nov", "dec",
})

# Matches a potential sentence boundary: punctuation + optional quote + spaces
# followed by an uppercase letter.  No lookbehind — handled in code below.
_POTENTIAL_BOUNDARY = re.compile(r'([.!?]["\']?)\s+(?=[A-Z])')

# Paragraph breaks are always safe split points
_PARA_BREAK = re.compile(r'\n{2,}')

# Last "word" before a period  (used to check abbreviations)
_LAST_WORD = re.compile(r'\b([A-Za-z]+)$')


def _is_abbreviation_boundary(left: str) -> bool:
    """
    Return True if the period at the end of `left` should NOT be treated
    as a sentence boundary (i.e. it follows an abbreviation or initial).
    """
    left = left.rstrip()
    if not left.endswith("."):
        return False
    # Single uppercase letter before period → initial (e.g. "A.")
    if len(left) >= 2 and left[-2].isupper():
        return True
    # Known abbreviation before period
    m = _LAST_WORD.search(left[:-1])  # strip the period before searching
    if m and m.group(1).lower() in _ABBREVS:
        return True
    return False


def _split_sentences(text: str) -> list[str]:
    """
    Return a list of sentence strings.  Preserves sentence-ending
    punctuation on the left side of each split.
    """
    paragraphs = _PARA_BREAK.split(text)
    sentences: list[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Find all candidate boundary spans
        parts: list[str] = []
        last = 0
        for m in _POTENTIAL_BOUNDARY.finditer(para):
            left = para[last:m.end(1)]  # text up to and including punctuation
            if _is_abbreviation_boundary(left):
                # Not a real boundary — keep accumulating
                continue
            parts.append(para[last:m.end(1)])
            last = m.end()  # skip the whitespace
        parts.append(para[last:])  # final segment

        for part in parts:
            stripped = part.strip()
            if stripped:
                sentences.append(stripped)

    return sentences


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> Iterator[str]:
    """
    Yield text chunks of approximately `chunk_size` characters with
    `overlap` character overlap between consecutive chunks.

    Guarantees:
    - Never splits mid-sentence when a sentence boundary is available.
    - Overlap is applied at sentence granularity: the new window starts
      with the last sentence(s) that fit within `overlap` characters.
    - Single sentences longer than `chunk_size` are yielded as-is (no
      hard truncation — oversized chunks are rare in Wikipedia prose).

    Args:
        text:       Cleaned plaintext (output of `cleaning.clean()`).
        chunk_size: Target maximum characters per chunk (default 500).
        overlap:    Approximate character overlap between chunks (default 50).

    Yields:
        Non-empty string chunks.
    """
    if not text or not text.strip():
        return

    sentences = _split_sentences(text)
    if not sentences:
        return

    window: list[str] = []      # sentences in the current chunk
    window_len: int = 0         # total characters in window (excl. spaces)

    def _render(sents: list[str]) -> str:
        return " ".join(sents)

    for sent in sentences:
        sent_len = len(sent)

        # If adding this sentence would exceed chunk_size AND we already have
        # content, flush the current window first.
        if window and window_len + 1 + sent_len > chunk_size:
            yield _render(window)

            # Build overlap: walk back from the end of the window accumulating
            # sentences until we would exceed `overlap` characters.
            overlap_sents: list[str] = []
            overlap_acc = 0
            for s in reversed(window):
                if overlap_acc + len(s) + 1 > overlap and overlap_sents:
                    break
                overlap_sents.insert(0, s)
                overlap_acc += len(s) + 1

            window = overlap_sents
            window_len = overlap_acc

        window.append(sent)
        window_len += sent_len + (1 if len(window) > 1 else 0)

    # Yield the final (possibly partial) window
    if window:
        yield _render(window)
