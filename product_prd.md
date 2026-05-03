# Product Requirements Document (PRD)

**Product:** Wikipedia Local RAG Assistant  
**Course:** BLG483E — Homework 3  
**Author:** Alperen Budak  
**Version:** 1.0  
**Date:** May 2026  

---

## 1. Problem Statement

Large Language Models (LLMs) are prone to *hallucination* — generating plausible-sounding but factually incorrect answers. Public LLM APIs (OpenAI, Anthropic, etc.) exacerbate this by requiring constant internet access, incurring per-token costs, and raising privacy concerns about user queries.

There is a clear need for a **lightweight, fully offline** question-answering system that:

- Grounds every answer in a verifiable document corpus.
- Runs entirely on commodity consumer hardware (Apple Silicon laptop).
- Costs nothing at inference time.
- Refuses to answer when the answer is not in the corpus.

---

## 2. Target User

**Primary persona:** A student or researcher with an Apple Silicon MacBook who wants to query a curated knowledge base about famous historical figures and world landmarks without exposing queries to external services.

**Secondary persona:** A software engineering student learning RAG system design who needs a reference implementation that demonstrates the full ingest → embed → retrieve → generate pipeline in clean, readable Python.

---

## 3. Goals

| Goal | Success Metric |
|------|---------------|
| Grounded answers | ≥ 90 % of in-corpus questions answered with facts traceable to retrieved chunks |
| Hallucination prevention | Refusal string returned for all out-of-corpus questions in manual testing |
| Offline operation | Zero external network calls at inference time |
| Latency | End-to-end response (warm models) < 10 s on M1 MacBook Air 8 GB |
| Memory | Peak RSS < 4 GB during query |
| Corpus coverage | ≥ 40 entities (≥ 20 persons, ≥ 20 places) |

---

## 4. Functional Requirements

### 4.1 Ingestion

| ID | Requirement |
|----|-------------|
| F-01 | The system SHALL fetch article text from English Wikipedia using `wikipedia-api`. |
| F-02 | Fetched articles SHALL be cached locally so re-runs do not re-hit the network. |
| F-03 | The system SHALL clean raw Wikipedia text — stripping citations, HTML tags, section markers, and pronunciation guides — using only native Python (`re`, `str`). |
| F-04 | Cleaned text SHALL be split into chunks of approximately 500 characters with 50-character overlap. The chunker SHALL be a Python generator and SHALL NOT split mid-sentence. |
| F-05 | Each chunk SHALL be embedded using `nomic-embed-text` via the Ollama daemon. Embedding calls SHALL be batched (≤ 8 chunks/call) to respect M1 memory constraints. |
| F-06 | Embeddings SHALL be stored in a single ChromaDB collection (`wiki_rag`) using a cosine-distance HNSW index. |
| F-07 | Every stored chunk SHALL carry metadata: `entity_name`, `entity_type` (`person` or `place`), `source_url`, `chunk_index`. |
| F-08 | An SQLite manifest SHALL record each ingested entity's content hash, enabling incremental updates that skip unchanged articles. |

### 4.2 Query Routing

| ID | Requirement |
|----|-------------|
| F-09 | The system SHALL classify each query as `person`, `place`, or `both` before retrieval. |
| F-10 | Classification SHALL use a two-tier approach: (1) keyword heuristics + entity name matching (<1 ms); (2) LLM call to `llama3.2:3b` as fallback. |
| F-11 | LLM-based classification results SHALL be cached in SQLite to avoid repeated model calls for identical queries. |
| F-12 | If classification is ambiguous, the system SHALL default to `both` and retrieve across all entity types. |

### 4.3 Retrieval

| ID | Requirement |
|----|-------------|
| F-13 | The retriever SHALL embed the query using `nomic-embed-text` and query ChromaDB with a `where` filter matching the detected `entity_type`. |
| F-14 | The system SHALL return the top-k (default 5) most similar chunks by cosine distance. |
| F-15 | Chunks with cosine distance above a configurable threshold (default 1.5) SHALL be dropped from the result set. |
| F-16 | If zero chunks survive the threshold, the system SHALL short-circuit generation and return the refusal string. |

### 4.4 Generation

| ID | Requirement |
|----|-------------|
| F-17 | The system SHALL use `llama3.2:3b` via Ollama for answer generation. |
| F-18 | The generation prompt SHALL enforce source isolation: "Use ONLY the context below. Do not use prior knowledge." |
| F-19 | The generation prompt SHALL include a verbatim refusal clause: `"I don't know based on the available information."` |
| F-20 | The model SHALL cite which entity (person/place name) its answer is drawn from. |
| F-21 | If `contexts` is empty, the system SHALL return the refusal string WITHOUT invoking the LLM. |
| F-22 | Generation SHALL support a streaming mode that prints tokens as they are produced. |

### 4.5 Interface

| ID | Requirement |
|----|-------------|
| F-23 | The system SHALL provide a terminal REPL supporting multi-turn questions. |
| F-24 | Each answer SHALL be followed by a deduplicated "Sources" footer listing contributing entities and their Wikipedia URLs. |
| F-25 | The system SHALL provide a Streamlit web UI with streaming answers, a category badge, and an expandable context panel. |
| F-26 | The CLI SHALL exit gracefully on `:quit`, `:q`, Ctrl-D, and Ctrl-C. |

---

## 5. Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NF-01 | **Offline operation** | Zero external API calls during inference. Wikipedia fetches occur only during ingest; cached thereafter. |
| NF-02 | **Latency** | < 10 s end-to-end for a warm (loaded) model on M1 MacBook Air 8 GB. |
| NF-03 | **Memory** | Peak RSS < 4 GB during a single query cycle. Ollama daemon manages model weights separately from Python. |
| NF-04 | **Throughput** | Ingest pipeline processes all 44 entities in < 30 min on first run (network-bound); < 5 min on cached re-runs. |
| NF-05 | **Idempotency** | Re-running the ingest pipeline with unchanged articles produces identical ChromaDB state. |
| NF-06 | **Native-first** | Core text processing (cleaning, chunking, prompt assembly) implemented in pure Python. No LangChain, LlamaIndex, or Haystack. |
| NF-07 | **Portability** | Runs on any macOS machine with Apple Silicon and Python 3.11. |
| NF-08 | **Testability** | All text-processing and prompt modules are unit-testable without Ollama or ChromaDB. |

---

## 6. Success Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Grounding rate | % of in-corpus queries where the answer is traceable to a retrieved source | ≥ 90 % |
| Refusal precision | % of out-of-corpus queries that return the refusal string (no hallucination) | 100 % |
| Retrieval precision | Top-1 retrieved chunk's `entity_name` matches query subject | ≥ 80 % |
| Test pass rate | `pytest tests/` | 100 % (all 69 tests green) |
| Latency (p95) | End-to-end query on warm M1 | < 10 s |
| Memory headroom | Available RAM during query | > 1 GB |

---

## 7. Out of Scope

The following are explicitly **not** part of this release:

- Multi-user concurrency or REST API exposure.
- Re-ranking models (Cohere Rerank, BGE-reranker, etc.).
- Hybrid BM25 + dense vector retrieval.
- Multilingual Wikipedia support (English only).
- Fine-tuning of either the embedding or generation model.
- Automatic corpus update scheduling.
- User authentication or query logging.

---

## 8. Assumptions & Constraints

- The user has already installed Ollama and pulled `llama3.2:3b` and `nomic-embed-text`.
- The system is single-user and single-process; no concurrency safeguards are implemented beyond Ollama's `OLLAMA_NUM_PARALLEL=1` setting.
- Wikipedia article structure is stable enough that the cleaning heuristics remain effective without retraining.
- The 500-character chunk size is appropriate for the 4096-token context window of `llama3.2:3b` when providing 5 chunks.
