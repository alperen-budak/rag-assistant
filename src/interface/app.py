"""
src/interface/app.py — Streamlit web UI for the Wikipedia RAG system.

Launch with:
    streamlit run src/interface/app.py

Features:
  - Streaming answer display (token-by-token via st.write_stream)
  - Detected category badge (Person / Place / Both)
  - Expandable "Retrieved Context" panel with distance scores
  - Deduplicated Sources list with Wikipedia links
  - Session-level chat history
  - Cached Chroma client and Ollama warm-up (@st.cache_resource)
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.rag.generator import stream_answer  # noqa: E402
from src.rag.prompts import REFUSAL_STRING, unique_sources  # noqa: E402
from src.rag.retriever import retrieve_with_category  # noqa: E402
from src.rag.types import RetrievedChunk  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Wikipedia RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached resources (init once per Streamlit process) ────────────────────────

@st.cache_resource(show_spinner="Loading models and vector store…")
def _warm_up() -> bool:
    """
    Trigger the lazy Chroma client and verify Ollama is reachable.
    Called once; result is cached for the lifetime of the Streamlit server.
    """
    try:
        from src.rag.retriever import _get_collection  # noqa: PLC0415
        _get_collection()
        return True
    except Exception:  # noqa: BLE001
        return False


# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Main layout */
    .block-container { padding-top: 1.5rem; }

    /* Category badges — always white text on solid colour, dark-mode safe */
    .badge-person {
        background: #1a6b9a; color: #ffffff !important;
        padding: 2px 10px; border-radius: 12px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em;
    }
    .badge-place {
        background: #6b3fa0; color: #ffffff !important;
        padding: 2px 10px; border-radius: 12px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em;
    }
    .badge-both {
        background: #4a7c59; color: #ffffff !important;
        padding: 2px 10px; border-radius: 12px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📚 Wikipedia RAG")
    st.caption("BLG483E — HW3 · Local-only · M1 optimised")
    st.divider()

    st.markdown("**Models**")
    st.code("llama3.2:3b\nnomic-embed-text", language=None)

    st.markdown("**Corpus**")
    st.markdown("44 entities · 22 persons · 22 places")

    st.divider()
    top_k = st.slider("Top-k chunks", min_value=1, max_value=10, value=5)
    show_chunks = st.toggle("Show retrieved chunks", value=False)
    st.divider()

    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    system_ok = _warm_up()
    if system_ok:
        st.success("System ready", icon="✅")
    else:
        st.error("Vector store unreachable.\nRun: `python scripts/run_ingest.py --rebuild`")


# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, meta}


# ── Header ────────────────────────────────────────────────────────────────────

st.title("📚 Wikipedia Local RAG")
st.caption(
    "Ask anything about the 44 famous people and places in the corpus. "
    "All answers are grounded in Wikipedia — no hallucination."
)
st.divider()


# ── Chat history ──────────────────────────────────────────────────────────────

def _category_badge(cat: str) -> str:
    css = f"badge-{cat}"
    label = cat.upper()
    return f'<span class="{css}">{label}</span>'


def _render_sources(chunks: list[RetrievedChunk]) -> None:
    sources = unique_sources(chunks)
    if not sources:
        return
    st.markdown("**Sources**")
    for name, etype, url in sources:
        icon = "🧑‍🔬" if etype == "person" else "🏛️"
        # Native st.markdown link — theme controls text colour automatically
        st.markdown(f"{icon} [{name}]({url})")


def _render_chunks(chunks: list[RetrievedChunk]) -> None:
    with st.expander(f"📄 Retrieved context ({len(chunks)} chunks)", expanded=False):
        for i, chunk in enumerate(chunks, 1):
            icon = "🧑‍🔬" if chunk.entity_type == "person" else "🏛️"
            preview = chunk.text[:300].replace("\n", " ")
            if len(chunk.text) > 300:
                preview += "…"
            # st.container(border=True) is theme-aware: background and text
            # colours are managed by Streamlit, so dark mode works out of the box.
            with st.container(border=True):
                st.markdown(f"**#{i} {icon} {chunk.entity_name}**")
                st.caption(f"dist={chunk.distance:.3f} · {chunk.entity_type}")
                st.markdown(preview)


# Replay existing messages
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])
        if role == "assistant" and msg.get("chunks"):
            chunks = msg["chunks"]
            category = msg.get("category", "both")

            col1, col2 = st.columns([3, 1])
            with col1:
                _render_sources(chunks)
            with col2:
                st.markdown(
                    f"Category: {_category_badge(category)}",
                    unsafe_allow_html=True,
                )
            if show_chunks:
                _render_chunks(chunks)


# ── Input ─────────────────────────────────────────────────────────────────────

query = st.chat_input("Ask about a famous person or place…")

if query:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ── Retrieve ──────────────────────────────────────────────────────────────
    with st.spinner("Searching knowledge base…"):
        chunks, category = retrieve_with_category(query, top_k=top_k)

    # ── Generate (streaming) ──────────────────────────────────────────────────
    with st.chat_message("assistant"):
        if not chunks:
            answer = REFUSAL_STRING
            st.warning(answer, icon="🤷")
        else:
            # Stream tokens into the chat bubble
            answer_placeholder = st.empty()
            full_tokens: list[str] = []

            with answer_placeholder:
                streamed = st.write_stream(stream_answer(query, chunks))
            answer = streamed if isinstance(streamed, str) else "".join(full_tokens)

        # Sources + category badge below the answer
        if chunks:
            st.divider()
            col1, col2 = st.columns([3, 1])
            with col1:
                _render_sources(chunks)
            with col2:
                st.markdown(
                    f"Category: {_category_badge(category)}",
                    unsafe_allow_html=True,
                )
            if show_chunks:
                _render_chunks(chunks)

    # Persist to session history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": chunks,
        "category": category,
    })
