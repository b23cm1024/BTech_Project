# IIT Jodhpur Institutional Intelligence Advisor

> An evidence-grounded, hybrid-retrieval RAG system that answers questions about IIT Jodhpur from official institutional documents, with citations and multi-stage answer validation.


---

##  Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Repository Structure](#-repository-structure)
5. [Tech Stack](#-tech-stack)
6. [Getting Started](#-getting-started)
7. [Usage](#-usage)
8. [Data Pipeline](#-data-pipeline)
9. [Evaluation & Results](#-evaluation--results)
10. [Troubleshooting](#-troubleshooting)
11. [Limitations & Future Work](#-limitations--future-work)
12. [Acknowledgements](#-acknowledgements)

---

##  Overview

Institutional information at IIT Jodhpur (academic rules, regulations, curricula, fees, circulars) is scattered across many websites and PDFs, so reliable answers are hard to find.

This project is an **AI advisor** that:

- **Crawls** official sources and downloads documents
- **Parses** PDFs into text, tables, figures and equations (Docling + MinerU)
- **Indexes** **43,322 chunks** from **4,233 documents** using hybrid dense + sparse retrieval
- **Plans** retrieval from the user's query, then **retrieves** the best evidence
- **Generates** answers grounded *only* in retrieved evidence
- **Validates** every answer before showing it to the user

---

##  Key Features

| Feature | Description |
|---|---|
|  **Crawling** | `crawler.py`, `deep_crawl.py`, `crawl_all.py` with configurable `download_rules.py` |
|  **Dual PDF parsing** | `parsers/` combines **Docling** and **MinerU** adapters via `dual_parser` |
|  **Rich extraction** | Dedicated table, figure and equation extractors |
|  **Structured data** | Canonical structured tables and an academic database (`iitj_academic.db`) |
|  **Smart chunking** | Structure-aware chunker (`chunker_v3.py`) |
|  **Hybrid retrieval** | Dense search (ChromaDB + `all-MiniLM-L6-v2`) and sparse BM25 (`bm25s`) fused with **Reciprocal Rank Fusion** |
|  **Metadata filtering** | Detects programme (CSE, AIDE, ME, EE, CE, …) and semester from the query |
|  **Query intelligence** | Query understanding and planning before retrieval |
|  **Source authority** | Source registry and authority scoring prefer official sources |
|  **Evidence-grounded generation** | Evidence extraction → evidence pack → grounded generator |
|  **Multi-stage validation** | Retrieval validator, evidence integrity, topic-mismatch validator, deterministic verifier, answer validator |
|  **LLM fallback chain** | Groq `gpt-oss-120b` → `gpt-oss-20b` / `qwen3.8-27b`|
|  **Multimodal** | CLIP embeddings for figures/images |
|  **REST API** | FastAPI service in `api/` |
|  **Web UI** | Streamlit chat interface in `app.py` |
|  **Tracing & auditing** | `execution_tracer.py`, document inventory, corpus completeness auditor |

---

##  System Architecture

```
 ┌─────────────────── INGESTION (offline) ───────────────────┐
 │  IIT Jodhpur websites & PDFs                              │
 │        │  crawler · deep_crawl · crawl_all                │
 │        ▼                                                  │
 │  PDF manifest + document inventory                        │
 │        ▼                                                  │
 │  Parsers: Docling + MinerU (dual_parser)                  │
 │        │  tables · figures · equations                    │
 │        ▼                                                  │
 │  Chunker (v3) ──► data/index_chunks.json(.gz)             │
 │        ▼  embed.py                                        │
 │  ChromaDB (dense) + BM25 (sparse) + CLIP (images)         │
 └───────────────────────────────────────────────────────────┘

 ┌─────────────────── QUERY (online) ────────────────────────┐
 │  User question (Streamlit app.py / FastAPI api/server.py) │
 │        ▼                                                  │
 │  Query understanding ► Query planner                      │
 │        ▼                                                  │
 │  Retrieval engine: dense + BM25 ► RRF fusion              │
 │        │  + source authority + metadata filters           │
 │        ▼                                                  │
 │  Retrieval validator                                      │
 │        ▼                                                  │
 │  Evidence orchestrator ► Evidence pack                    │
 │        ▼                                                  │
 │  Evidence-grounded generator (LLM via provider_layer)     │
 │        ▼                                                  │
 │  Evidence integrity · Topic-mismatch · Deterministic      │
 │  verifier · Answer validator                              │
 │        ▼                                                  │
 │  Final answer with citations                              │
 └───────────────────────────────────────────────────────────┘
```

---

##  Repository Structure

```
BTech_Project/
├── api/                              # FastAPI service
│   ├── schemas.py                    # Request/response models
│   └── server.py                     # API server
├── assets/                           # UI images and logo
├── data/                             # Knowledge base
│   ├── canonical_structured_tables.json
│   ├── document_inventory.json
│   ├── iitj_academic.db              # Academic database
│   ├── index_chunks.json.gz          # Compressed chunk index
│   └── pdf_manifest.json
├── database/                         # connection.py · init_db.py · models.py
├── parsers/                          # docling_adapter · mineru_adapter · dual_parser
├── storage/                          # index_qdrant.py · qdrant_manager.py
├── eval/                             # Evaluation scripts
│   ├── questions.json                # 56 evaluation questions
│   ├── check_questions.py            # Validates keywords against the corpus
│   └── run_eval.py                   # Dense vs BM25 vs Hybrid evaluation
│
├── app.py                            # Streamlit app
├── rag.py                            # Core RAG pipeline
├── provider_layer.py                 # LLM provider abstraction (Groq / Gemini)
├── request_context.py · execution_tracer.py
│
├── crawler.py · deep_crawl.py · crawl_all.py · download_rules.py
├── ingest.py · ingest_v3.py · document_inventory.py
├── corpus_completeness_auditor.py · source_registry.py · source_authority.py
│
├── table_extractor.py · figure_extractor.py · equation_extractor.py
├── chunker_v3.py · embed.py · clip_multimodal.py
├── chromadb_store.py                 # Hybrid ChromaDB + BM25 store (RRF)
│
├── query_understanding.py · query_planner.py · query_planning.py
├── retrieval_engine_v3.py · universal_engine.py
├── universal_retriever.py · universal_manifest.py · retrieval_validator.py
│
├── evidence_models.py · evidence_extractor.py · evidence_orchestrator.py
├── evidence_pack.py · evidence_integrity.py · evidence_grounded_generator.py
│
├── answer_validator.py · deterministic_verifier.py · topic_mismatch_validator.py
├── requirements.txt · .env.example · .gitignore
```

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| UI | Streamlit (port 8501) |
| API | FastAPI + Uvicorn (port 8000) |
| LLM | Groq: `openai/gpt-oss-120b` (default), `openai/gpt-oss-20b`, `qwen/qwen3.8-27b` |
| Dense retrieval | ChromaDB (collection `iitj_chunks`) + `all-MiniLM-L6-v2` |
| Sparse retrieval | BM25 via `bm25s` |
| Fusion | Reciprocal Rank Fusion (k = 60) |
| Re-ranking | Cross-Encoder (optional, falls back if not cached) |
| PDF parsing | Docling, MinerU, PyMuPDF |
| Multimodal | CLIP |
| Data store | `iitj_academic.db`, Qdrant module in `storage/` |

---

##  Getting Started

### Prerequisites

- Python 3.10+
- A free **Groq API key** (https://console.groq.com → API Keys) and/or a Gemini key
- Internet for the first run (to download the embedding model)

### 1. Clone and install

```bash
git clone https://github.com/b23cm1024/BTech_Project.git
cd BTech_Project

python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # Linux / macOS

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
copy .env.example .env          # Windows
# cp .env.example .env          # Linux / macOS
```

Edit **`.env`** (never commit it):

```env
GROQ_API_KEY=gsk_your_key_here

HF_HUB_OFFLINE=0                # set to 0 for the first run, then 1
TRANSFORMERS_OFFLINE=0

STREAMLIT_SERVER_PORT=8501
FASTAPI_PORT=8000
```

>  Keep real keys only in `.env`. `.env.example` must contain placeholders only.

### 3. Prepare the data and build the indexes

```bash
# Decompress the chunk index (the repo ships it as .gz)
python -c "import gzip,shutil; shutil.copyfileobj(gzip.open('data/index_chunks.json.gz','rb'),open('data/index_chunks.json','wb'))"

# Download the embedding model once (needed if the first run is blocked by offline mode)
python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2')"

# Build ChromaDB + BM25 indexes (can take 20-60 min on CPU)
python embed.py
```

This creates `data/chromadb/` and `data/bm25_index/`. Afterwards you may set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

---

##  Usage

### Run the web app

```bash
streamlit run app.py
```
Open http://localhost:8501, select **Groq** as the provider, and ask a question.

### Run the API

```bash
uvicorn api.server:app --port 8000 --reload
```

### Example questions

- *"What is the minimum attendance required to appear in the end semester examination?"*
- *"List the courses offered in the 5th semester of the AIDE programme."*
- *"What are the rules for changing branch after the first year?"*
- *"What is the fee refund policy if a student withdraws?"*

---

##  Data Pipeline

The repo ships with a pre-built chunk index. To rebuild the knowledge base from scratch:

```bash
python crawl_all.py                     # 1. Crawl official sources
python ingest_v3.py                     # 2. Parse, chunk and ingest documents
python embed.py                         # 3. Build ChromaDB + BM25 indexes
python corpus_completeness_auditor.py   # 4. (Optional) Audit corpus coverage
```

**Corpus size:** 43,322 chunks · 4,233 document records.

---

##  Evaluation & Results

### Setup

Retrieval quality was evaluated on **48 questions** (`eval/questions.json`) covering academic rules, curriculum, admissions & programmes, fees & scholarships, hostel & campus, research, placement & internship, and short keyword/acronym queries. A retrieved chunk counts as relevant if it contains all `expected_keywords` of the question.

Two systems were compared on the same questions:

| System | Retrieval |
|---|---|
| **Basic RAG** (baseline) | Dense search only (ChromaDB + `all-MiniLM-L6-v2`) |
| **Hybrid** (this project) | Dense + BM25 sparse search, fused with Reciprocal Rank Fusion |

**Metrics.** *Recall@k* is the share of questions with at least one relevant chunk in the top *k* results (a hit rate). *MRR* is the mean of 1/rank of the first relevant chunk. *Latency* is the mean retrieval time per query.

```bash
python eval/run_eval.py
```

### Results

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | Latency (s) |
|---|---|---|---|---|---|
| Basic RAG (dense only) | 0.500 | 0.729 | 0.854 | 0.603 | **0.012** |
| **Hybrid (ours: dense + BM25, RRF)** | **0.583** | **0.792** | **0.896** | **0.670** | 0.079 |

**Improvement of hybrid over basic RAG**

| Metric | Absolute gain | Relative gain | Questions (of 48) |
|---|---|---|---|
| Recall@1 | +0.083 | +16.6% | 24 → 28 |
| Recall@5 | +0.063 | +8.6% | 35 → 38 |
| Recall@10 | +0.042 | +4.9% | 41 → 43 |
| MRR | +0.067 | +11.1% | – |

### Analysis

- **Hybrid retrieval is better on every quality metric.** The gain is largest at rank 1, where the correct chunk is placed first for 4 more questions out of 48.
- **Why it helps:** dense search understands paraphrases but can miss exact terms such as programme names, acronyms and regulation keywords, while BM25 finds exact terms but not paraphrases. Fusing both with RRF recovers evidence that either method alone would rank too low.
- **The cost is small:** hybrid takes about 79 ms per query (vs. 12 ms for dense only), which is negligible next to LLM generation time.
- The relevant chunk is in the top 10 for **89.6%** of questions with hybrid retrieval, in a corpus of 43,322 chunks.

### Limitations of the evaluation

- Relevance is judged by **keyword matching**, which can favour lexical (BM25) matching over purely semantic matches.
- The test set has 48 questions, so the gains (2–4 questions per metric) are indicative rather than statistically significant.
- These metrics cover **retrieval only**. End-to-end answer correctness, faithfulness and the effect of the validators and evidence pack are not measured here.
- The Cross-Encoder re-ranker was not active during these runs.

---


##  Limitations & Future Work

**Limitations**
- Answers are only as current as the last crawl of the source documents.
- Quality depends on PDF parsing accuracy and the chosen LLM.
- Free-tier LLM APIs are rate-limited.

**Future work**
- End-to-end evaluation (answer correctness, faithfulness, citation accuracy)
- Evaluation with human-labelled relevant chunks, and per-category analysis
- Scheduled re-crawling to keep data fresh
- Hindi / multilingual support
- Public deployment for the IIT Jodhpur community

---

##  Acknowledgements

- **Supervisor:** <!-- Prof. Sumit Kalra -->
- Department of <!-- CSE -->, **IIT Jodhpur**
- Open-source projects: ChromaDB, Qdrant, Docling, MinerU, bm25s, Sentence-Transformers, Streamlit, FastAPI, CLIP

---

