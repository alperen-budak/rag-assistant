# Wikipedia RAG — Local Knowledge Assistant

Demo video: https://youtu.be/f8QPb9DZm-A
Repository: https://github.com/alperen-budak/rag-assistant

> **Course:** BLG483E — Homework 3  
> **Author:** Alperen Budak  
> **Runtime:** 100 % localhost — no external API calls, no internet required at inference time

A fully local Retrieval-Augmented Generation system that answers natural-language questions about 44 famous people and places by retrieving grounded passages from a Wikipedia corpus and generating answers with Llama 3.2 3B via Ollama.

---

## Architecture

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ 1. Data Ingest   │ -> │ 2. Native Chunk  │ -> │ 3. Embedding     │
│ wikipedia-api    │    │ pure-Python      │    │ nomic-embed-text │
│ (22 persons +    │    │ sentence-aware   │    │ via Ollama       │
│  22 places)      │    │ generator        │    │ (Metal GPU)      │
└──────────────────┘    └──────────────────┘    └──────────────────┘
                                                          │
                                                          ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ 6. Generation    │ <- │ 5. Retrieval     │ <- │ 4. Vector Store  │
│ Llama 3.2 3B     │    │ category-aware   │    │ ChromaDB         │
│ (Ollama, Metal)  │    │ metadata filter  │    │ (persistent)     │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

**Person / Place routing:** Queries are classified by a two-tier router (keyword heuristics → LLM fallback) so retrieval is filtered by `entity_type` metadata, keeping results precise.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| macOS (Apple Silicon) | Ventura 13+ | Intel Macs work but no Metal GPU |
| Python | 3.11+ | `brew install python@3.11` |
| Ollama | 0.3+ | [ollama.ai](https://ollama.ai) |
| Disk space | ~6 GB | Models (~3 GB) + ChromaDB (~200 MB) |
| RAM | 8 GB | Minimum; 16 GB recommended |

---

## Setup (Step by Step)

### 1 — Install Ollama

Download from [ollama.ai/download](https://ollama.ai/download) and install the macOS `.dmg`, or via Homebrew:

```bash
brew install ollama
```

Verify:

```bash
ollama --version
```

### 2 — Pull the required models

Open a terminal and start the Ollama daemon:

```bash
ollama serve
```

In a **second terminal**, pull both models (one-time, ~3 GB total):

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Verify Metal (GPU) acceleration — after pulling, run:

```bash
ollama run llama3.2:3b "ping"
ollama ps          # should show 100% GPU
```

### 3 — Clone / navigate to the project

```bash
cd ~/Desktop/aided-hw3
```

### 4 — Create and activate the virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of your prompt.

### 5 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 6 — Verify the environment

```bash
python scripts/check_env.py
```

All checks should show ✔ green. If any required package is missing, the script will tell you exactly what to install.

---

## Building the Knowledge Base (Ingest)

Run the ingestion pipeline once to fetch Wikipedia articles, chunk them, embed them, and store them in ChromaDB:

```bash
# Make sure ollama serve is running in another terminal
python scripts/run_ingest.py --rebuild
```

This will:
1. Fetch 44 Wikipedia articles (22 persons + 22 places) — takes ~5–10 min on first run
2. Cache raw articles to `data/raw/` (subsequent runs are fast)
3. Clean and chunk each article (~500 chars/chunk, 50-char overlap)
4. Embed each chunk with `nomic-embed-text` in batches of 8
5. Upsert all vectors into ChromaDB at `data/chroma/`
6. Record ingestion state in `data/manifest.db`

**Incremental re-ingest** (skips unchanged articles):

```bash
python scripts/run_ingest.py
```

**Test with a small subset first:**

```bash
python scripts/run_ingest.py --rebuild --limit 3
```

---

## Running the System

### Option A — Terminal CLI (recommended for development)

```bash
python main.py
```

Flags:

```bash
python main.py --show-category    # show person/place routing decision
python main.py --verbose          # show retrieved chunks
python main.py --no-stream        # disable token streaming
```

Inside the chat loop:

| Command | Action |
|---------|--------|
| `:sources` | Toggle source chunk display |
| `:category` | Toggle router category display |
| `:help` | Show command list |
| `:quit` / Ctrl-D | Exit |

### Option B — Streamlit Web UI

```bash
streamlit run src/interface/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

Features:
- Streaming answer display
- Category badge (PERSON / PLACE / BOTH)
- Expandable "Retrieved Context" panel with distance scores
- Deduplicated Wikipedia source links
- Session chat history
- Sidebar controls: top-k slider, chunk display toggle, clear history

### Option C — Manual vector store inspection (Checkpoint 1)

```bash
# Pretty-print top-5 results for a query
python scripts/inspect_chroma.py "Where was Albert Einstein born?"
python scripts/inspect_chroma.py "What is the Eiffel Tower?" --filter place

# Collection + manifest summary
python scripts/inspect_chroma.py --stats
```

---

## Checkpoint 2 Verification

Run the CLI and ask these four questions in order:

```
1. Who was Marie Curie and what did she discover?
2. In which country is the Colosseum located?
3. What is one famous landmark in Paris?
4. What is the capital of Mars?
```

Expected behaviour:
- Questions 1–3: factually correct, grounded answers with a "Sources:" footer
- Question 4: `"I don't know based on the available information."` — no hallucination

---

## Running Tests

```bash
pytest tests/ -v
```

Tests are pure-Python (no Ollama or ChromaDB required):

| File | Tests | What it covers |
|------|-------|----------------|
| `tests/test_chunking.py` | 24 | Cleaner + chunker + sentence splitter |
| `tests/test_router.py` | 22 | Keyword heuristics, detect_category API |
| `tests/test_prompts.py` | 23 | Prompt template, refusal clause, short-circuit |

---

## Project Structure

```
aided-hw3/
├── main.py                       # python main.py — CLI entry point
├── requirements.txt
├── config/
│   ├── entities.yaml             # 44 entities (22 persons + 22 places)
│   └── settings.yaml             # chunk size, models, paths
├── data/                         # gitignored; created at runtime
│   ├── raw/                      # cached Wikipedia JSON
│   ├── chroma/                   # ChromaDB vector store
│   └── manifest.db               # SQLite ingestion ledger
├── src/
│   ├── core/
│   │   ├── cleaning.py           # Wikipedia markup stripper
│   │   ├── chunking.py           # sentence-aware generator
│   │   └── manifest.py           # SQLite helpers
│   ├── ingest/
│   │   ├── wiki_fetcher.py       # Wikipedia API + file cache
│   │   └── pipeline.py           # end-to-end ingest orchestrator
│   ├── rag/
│   │   ├── types.py              # RetrievedChunk dataclass
│   │   ├── embeddings.py         # batched Ollama embed client
│   │   ├── vector_store.py       # ChromaDB wrapper
│   │   ├── router.py             # person/place classifier
│   │   ├── retriever.py          # category-aware top-k retrieval
│   │   ├── prompts.py            # versioned prompt templates
│   │   └── generator.py          # Llama 3.2 3B generation
│   └── interface/
│       ├── cli.py                # terminal REPL
│       └── app.py                # Streamlit web UI
├── scripts/
│   ├── check_env.py              # environment validation
│   ├── run_ingest.py             # ingest CLI
│   └── inspect_chroma.py        # manual vector store query
└── tests/
    ├── test_chunking.py
    ├── test_router.py
    └── test_prompts.py
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Connection refused` on ingest | Ollama not running | `ollama serve` in a separate terminal |
| `model not found` | Model not pulled | `ollama pull llama3.2:3b` |
| `No results found` in inspect | Ingest not run | `python scripts/run_ingest.py --rebuild` |
| Answer takes >30 s | Models not warm / CPU mode | Check `ollama ps` for `100% GPU`; restart Ollama |
| High memory usage | Both models loaded simultaneously | Close other heavy apps; `OLLAMA_NUM_PARALLEL=1` is set in settings |
| Wikipedia fetch timeout | Slow network | Re-run `python scripts/run_ingest.py` (resumes from cache) |

---

## Memory Budget (M1 8 GB)

| Component | Resident Memory |
|-----------|----------------|
| Ollama — `llama3.2:3b` (Q4_K_M) | ~2.0 GB |
| Ollama — `nomic-embed-text` | ~0.3 GB |
| ChromaDB in-process | ~0.2 GB |
| Python runtime | ~0.4 GB |
| **Headroom** | **~1.1 GB** |

Peak RSS stays comfortably under 4 GB, leaving the other 4 GB for macOS and other apps.
