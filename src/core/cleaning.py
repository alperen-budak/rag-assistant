"""
src/core/cleaning.py — Wikipedia plaintext cleaner.

Zero external dependencies: pure re + str only (PROJECT.md §3.3).

Public API:
    clean(text: str) -> str
"""

from __future__ import annotations

import re

# ── Compiled patterns (compiled once at import, reused per call) ──────────────

# Numeric citations: [1], [12], [1][2], [citation needed], [note 3]
_CITATIONS = re.compile(r"\[[^\[\]]{0,40}\]")

# HTML / XML tags that wikipedia-api occasionally leaks through
_HTML_TAGS = re.compile(r"<[^>]{0,200}>", re.DOTALL)

# HTML entities (&amp; &lt; &nbsp; &#160; etc.)
_HTML_ENTITIES = re.compile(r"&(?:[a-z]{2,6}|#\d{1,5});", re.IGNORECASE)

# Wikipedia section edit markers e.g. "== Section ==" sometimes appear in raw
# summaries from the API as unicode separators or leftover markup
_SECTION_MARKERS = re.compile(r"={2,}\s*[^=\n]{0,80}\s*={2,}")

# Parenthetical pronunciation guides: (/ ˈaɪ ... /) or (/ˈn../)
_PRONUNCIATION = re.compile(r"\(\/[^)]{0,80}\/\)")

# Multiple whitespace / non-breaking spaces → single space
_MULTI_SPACE = re.compile(r"[ \t\u00a0\u2009\u202f]+")

# More than two consecutive newlines → two newlines (preserve paragraph breaks)
_MULTI_NEWLINE = re.compile(r"\n{3,}")

# Leading/trailing whitespace per line
_LINE_EDGES = re.compile(r"^[ \t]+|[ \t]+$", re.MULTILINE)

# Lone punctuation lines (common artefact: "." or "–" on their own line)
_LONE_PUNCT = re.compile(r"^\s*[^\w\s]{1,3}\s*$", re.MULTILINE)

# Unicode quotation marks → ASCII (keeps text consistent for tokenisation)
_FANCY_QUOTES = str.maketrans({
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-",
})


def clean(text: str) -> str:
    """
    Return a clean, plain-text version of a Wikipedia article string.

    Strips citations, HTML artefacts, pronunciation guides, section markers,
    and normalises whitespace.  The output is safe to feed directly into the
    sentence-aware chunker in `src/core/chunking.py`.
    """
    if not text:
        return ""

    # Remove HTML tags first (before entity decoding, avoids double-processing)
    text = _HTML_TAGS.sub(" ", text)

    # Decode common HTML entities
    text = _HTML_ENTITIES.sub(" ", text)

    # Drop citation brackets
    text = _CITATIONS.sub("", text)

    # Drop pronunciation guides
    text = _PRONUNCIATION.sub("", text)

    # Drop Wikipedia section markers (== Heading ==)
    text = _SECTION_MARKERS.sub("\n\n", text)

    # Normalise fancy punctuation
    text = text.translate(_FANCY_QUOTES)

    # Normalise line-internal whitespace
    text = _MULTI_SPACE.sub(" ", text)

    # Strip per-line leading/trailing spaces
    text = _LINE_EDGES.sub("", text)

    # Remove lone-punctuation lines
    text = _LONE_PUNCT.sub("", text)

    # Collapse excessive blank lines
    text = _MULTI_NEWLINE.sub("\n\n", text)

    return text.strip()
