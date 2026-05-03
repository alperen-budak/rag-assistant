# Recommendations & Post-Mortem

**Product:** Wikipedia Local RAG Assistant  
**Course:** BLG483E — Homework 3  
**Author:** Alperen Budak  
**Version:** 1.0  

---

## 1. What Worked Well

### 1.1 Option B — Single Collection with Metadata Filtering

Storing persons and places in one ChromaDB collection and filtering at query time (rather than maintaining two separate collections) was the right call. It simplified schema management, eliminated cross-collection score comparison issues, and naturally supported hybrid "both-category" queries (e.g., "What did Einstein do in Berlin?") without query federation.

### 1.2 Two-Tier Router

The keyword heuristic path (Tier 1) handles ~80 % of real queries in under 1 ms with zero model overhead. Reserving the LLM call (Tier 2) for genuinely ambiguous queries, combined with SQLite caching of its results, keeps average routing latency negligible. This design is easy to tune — adding new keywords or adjusting the PERSON/PLACE sets does not require a model re-load.

### 1.3 Generator Short-Circuit on Empty Retrieval

Refusing to call the LLM when the retriever returns zero chunks above the similarity threshold was the single most effective anti-hallucination measure. It made the "I don't know" behaviour instant, predictable, and test-friendly (mock-able without a running model).

### 1.4 Generator-Based Chunking

Implementing `chunk_text()` as a Python `yield`-based generator, rather than a function that returns a list, meant the ingestion pipeline never materialised the full corpus in RAM. On an 8 GB M1 this matters: peak RSS during ingest stayed well under 3 GB even when processing 400+ chunks for a single article (Sigmund Freud: 388 chunks).

### 1.5 Sentence-Aware Splitting

Avoiding mid-sentence cuts — even at the cost of chunks slightly exceeding the 500-character target — produced noticeably cleaner context passages for the LLM. Chunks that start and end at sentence boundaries read as coherent prose rather than truncated fragments, which reduces prompt confusion.

---

## 2. Observed Limitations

### 2.1 3B Model Factual Recall

`llama3.2:3b` at `temperature=0.1` is highly obedient to the prompt context but occasionally rephrases retrieved text in ways that introduce subtle inaccuracies (e.g., swapping dates by a year). A larger model (7B or 13B) would reduce this, but the M1 Air's 8 GB unified memory makes loading anything beyond the 3B Q4 model alongside ChromaDB and the embedding model impractical without quantisation trade-offs.

### 2.2 Chunking Strategy Is Length-Only

The current chunker uses character count as the primary boundary signal. It does not account for semantic coherence: a 500-character chunk that starts mid-biography and ends mid-discovery can confuse the model. A sentence-transformer–based semantic chunker (e.g., splitting when cosine similarity between consecutive sentences drops below a threshold) would produce more coherent passages, at the cost of an extra embedding call per sentence during ingest.

### 2.3 Keyword Router False Positives

Words like `"founded"` and `"built"` appear in both PERSON_KEYWORDS (e.g., "He founded the company") and PLACE_KEYWORDS (e.g., "The city was built in…"). The current mutual-exclusion logic (return category only when one side scores > 0 and the other scores = 0) handles many cases, but edge queries like "Who founded this famous city?" incorrectly route to `"both"` when they are clearly person-biased. A lightweight named-entity recognition (NER) step would resolve this.

### 2.4 Single-Turn Context Window

The system sends up to 5 chunks (~2,500 characters) per query into a 4,096-token context window. This leaves little room for multi-turn conversation where prior turns are included as context. The CLI and Streamlit UI currently treat every question independently — they show a history but do not feed it to the model.

### 2.5 No Evaluation Harness

There is no automated quality measurement. The checkpoint tests verify correctness mechanically (refusal string, chunk count, etc.), but there is no RAGAS-style precision/recall evaluation over a labelled question set. This makes it difficult to compare prompt variants or chunking strategies objectively.

### 2.6 Wikipedia Fetch Fragility

The `wikipedia-api` library's underlying HTTP session has no configurable timeout, and long articles (Sigmund Freud, Nelson Mandela) occasionally trigger a read timeout on slower connections. The retry logic added in Phase 1 mitigates this but does not solve it completely — a proper `requests.Session` with explicit `timeout=(5, 30)` would be more reliable.

---

## 3. Future Work & Recommendations

### 3.1 Larger Generation Model (Priority: High)

With more RAM (16 GB M1 Pro/Max) or by using `llama3.2:8b` at Q4 quantisation, the factual fidelity and reasoning depth improve significantly. The codebase is model-agnostic — swapping the model requires only a change to `generation_model` in `config/settings.yaml`.

### 3.2 Hybrid BM25 + Dense Retrieval (Priority: High)

Cosine similarity alone can miss exact-match queries (e.g., "What year was [entity] born?") where term frequency matters more than semantic similarity. Adding a BM25 sparse index (using `rank_bm25`, ~200 lines of native Python) and merging scores with Reciprocal Rank Fusion would substantially improve precision on factual queries.

### 3.3 Re-Ranking (Priority: Medium)

A cross-encoder re-ranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`) applied to the top-k candidates before generation would improve the quality of the context window without changing the retrieval index. This runs on CPU in ~50 ms per query.

### 3.4 Semantic Chunking (Priority: Medium)

Replace the character-count chunker with a semantic boundary detector: embed each sentence with `nomic-embed-text` during ingest, compute cosine similarity between consecutive sentences, and split when similarity drops below a threshold (e.g., 0.7). This produces semantically coherent chunks at the cost of ~2× ingest time.

### 3.5 Multi-Turn Conversation (Priority: Medium)

Include the last N turns of chat history in the generation context. The 4,096-token window comfortably holds 2–3 prior exchanges alongside 3 retrieved chunks. This would make the CLI and Streamlit UI feel genuinely conversational rather than stateless Q&A.

### 3.6 RAGAS-Style Evaluation (Priority: Medium)

Build a labelled evaluation set of 50 questions (25 person, 25 place) with known ground-truth answers drawn from the corpus. Use RAGAS metrics — **context precision**, **context recall**, **faithfulness**, **answer relevancy** — to benchmark routing, retrieval, and generation separately. Automate this as a `pytest` fixture so regressions are caught on every PR.

### 3.7 Automated Corpus Updates (Priority: Low)

Schedule a weekly cron job that re-runs `python scripts/run_ingest.py` (without `--rebuild`). The SQLite manifest's content-hash check ensures only changed or new articles are re-embedded, keeping the corpus current with minimal compute.

### 3.8 Streaming REST API (Priority: Low)

Wrap the pipeline in a FastAPI server (`/api/query` → Server-Sent Events for streaming) so the Streamlit UI (or any other client) communicates via HTTP instead of in-process function calls. This would decouple the model-serving layer from the UI layer and enable multi-user deployments on a local network.

### 3.9 Named-Entity Recognition for Routing (Priority: Low)

Replace part of the keyword heuristic with spaCy's `en_core_web_sm` model (18 MB, CPU-only). NER-detected `PERSON` entities in the query would eliminate false positives from generic verbs shared between the two keyword sets.

---

## 4. Lessons Learned

| Lesson | Implication |
|--------|-------------|
| **Empty-context short-circuit is the cheapest anti-hallucination measure.** | Always implement it before prompt engineering — it's faster, testable, and never fails. |
| **Cosine distance thresholding is more important than top-k.** | Returning 3 high-quality chunks is always better than 5 noisy ones. Tune the threshold before tuning the prompt. |
| **Python's `re` module is sufficient for Wikipedia cleaning.** | Eight targeted compiled patterns remove 95 % of noise without NLP libraries. |
| **Streaming UX matters for 3B models.** | Without streaming, the 3–8 second latency feels broken. With streaming, 30+ tokens/s on Metal feels fast. |
| **Test the refusal clause first.** | Verifying that the model says "I don't know" on out-of-corpus queries before optimising in-corpus answers prevents a class of silent failures. |
| **SQLite for metadata beats a separate key-value store.** | Using SQLite for both the manifest and the router cache kept the dependency list minimal and made the state inspectable with standard tooling. |
