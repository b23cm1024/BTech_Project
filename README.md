# 🎓 IIT Jodhpur Institutional Intelligence Advisor

> An evidence-grounded, hybrid-retrieval RAG system that answers questions about IIT Jodhpur from official institutional documents, with citations and multi-stage answer validation.


---

## 📌 Table of Contents

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

## 🔍 Overview

Institutional information at IIT Jodhpur (academic rules, regulations, curricula, fees, circulars) is scattered across many websites and PDFs, so reliable answers are hard to find.

This project is an **AI advisor** that:

- **Crawls** official sources and downloads documents
- **Parses** PDFs into text, tables, figures and equations (Docling + MinerU)
- **Indexes** **43,322 chunks** from **4,233 documents** using hybrid dense + sparse retrieval
- **Plans** retrieval from the user's query, then **retrieves** the best evidence
- **Generates** answers grounded *only* in retrieved evidence
- **Validates** every answer before showing it to the user

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🌐 **Crawling** | `crawler.py`, `deep_crawl.py`, `crawl_all.py` with configurable `download_rules.py` |
| 📄 **Dual PDF parsing** | `parsers/` combines **Docling** and **MinerU** adapters via `dual_parser` |
| 🧮 **Rich extraction** | Dedicated table, figure and equation extractors |
| 🗂️ **Structured data** | Canonical structured tables and an academic database (`iitj_academic.db`) |
| 🧩 **Smart chunking** | Structure-aware chunker (`chunker_v3.py`) |
| 🔀 **Hybrid retrieval** | Dense search (ChromaDB + `all-MiniLM-L6-v2`) and sparse BM25 (`bm25s`) fused with **Reciprocal Rank Fusion** |
| 🎯 **Metadata filtering** | Detects programme (CSE, AIDE, ME, EE, CE, …) and semester from the query |
| 🔎 **Query intelligence** | Query understanding and planning before retrieval |
| 🏛️ **Source authority** | Source registry and authority scoring prefer official sources |
| 🧠 **Evidence-grounded generation** | Evidence extraction → evidence pack → grounded generator |
| ✅ **Multi-stage validation** | Retrieval validator, evidence integrity, topic-mismatch validator, deterministic verifier, answer validator |
| 🔁 **LLM fallback chain** | Groq `gpt-oss-120b` → `gpt-oss-20b` / `qwen3.8-27b`; optional Google Gemini |
| 🖼️ **Multimodal** | CLIP embeddings for figures/images |
| 🔌 **REST API** | FastAPI service in `api/` |
| 💬 **Web UI** | Streamlit chat interface in `app.py` |
| 🕵️ **Tracing & auditing** | `execution_tracer.py`, document inventory, corpus completeness auditor |

---

## 🏗️ System Architecture

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

## 📁 Repository Structure

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

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| UI | Streamlit (port 8501) |
| API | FastAPI + Uvicorn (port 8000) |
| LLM | Groq: `openai/gpt-oss-120b` (default), `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`; optional Google Gemini |
| Dense retrieval | ChromaDB (collection `iitj_chunks`) + `all-MiniLM-L6-v2` |
| Sparse retrieval | BM25 via `bm25s` |
| Fusion | Reciprocal Rank Fusion (k = 60) |
| Re-ranking | Cross-Encoder (optional, falls back if not cached) |
| PDF parsing | Docling, MinerU, PyMuPDF |
| Multimodal | CLIP |
| Data store | `iitj_academic.db`, Qdrant module in `storage/` |

---

## 🚀 Getting Started

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

> ⚠️ Keep real keys only in `.env`. `.env.example` must contain placeholders only.

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

## ▶️ Usage

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

## 🔄 Data Pipeline

The repo ships with a pre-built chunk index. To rebuild the knowledge base from scratch:

```bash
python crawl_all.py                     # 1. Crawl official sources
python ingest_v3.py                     # 2. Parse, chunk and ingest documents
python embed.py                         # 3. Build ChromaDB + BM25 indexes
python corpus_completeness_auditor.py   # 4. (Optional) Audit corpus coverage
```

**Corpus size:** 43,322 chunks · 4,233 document records.

---

## 📈 Evaluation & Results

### Setup

Retrieval was evaluated on **56 questions** (`eval/questions.json`) across 7 categories (academic rules, curriculum, admissions, fees & scholarships, hostel & campus, research & faculty, placement & internship) plus keyword/acronym queries. A retrieved chunk counts as relevant if it contains all `expected_keywords` of the question. The metric reported as Recall@k is a **hit rate**: the share of questions with at least one relevant chunk in the top *k*.

```bash
python eval/check_questions.py   # verify keywords exist in the corpus
python eval/run_eval.py          # run the comparison -> eval/results.json
```

### Retrieval results

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | Avg latency (s) |
|---|---|---|---|---|---|
| Dense only (ChromaDB) | 0.732 | 0.911 | 0.964 | 0.817 | 0.012 |
| BM25 only | **0.821** | **0.929** | **0.982** | **0.867** | 0.001 |
| Hybrid (RRF) | **0.821** | **0.929** | 0.946 | 0.861 | 0.064 |

### Analysis

- Retrieval is strong overall: the relevant chunk is in the **top 5 for 91–93%** of questions and in the **top 10 for 95–98%** across a 43,322-chunk corpus.
- **Dense-only retrieval ranks the correct chunk first less often** (Recall@1 = 0.732) than BM25 or hybrid (0.821).
- **Hybrid matches the best single method** at Recall@1 and Recall@5 and is far more robust than dense alone. Its small Recall@10 gap to BM25 (about 2 questions out of 56) is within the noise of a small test set.
- Retrieval latency is negligible (≤ 64 ms) compared with LLM generation time.

### Limitations of this evaluation

- Relevance is judged by **keyword matching**, which tends to favour BM25 over dense retrieval. Dense search may return semantically correct chunks that lack the exact keyword.
- The test set has 56 questions, so differences of 1–2 questions are not statistically meaningful.
- Cross-Encoder re-ranking was not active in these runs (model not cached in offline mode).
- These metrics cover **retrieval only**. End-to-end answer correctness and faithfulness are not yet measured.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: data/index_chunks.json` | Decompress `index_chunks.json.gz` (see step 3) |
| `We couldn't connect to 'https://huggingface.co'` | Set `HF_HUB_OFFLINE=0` and `TRANSFORMERS_OFFLINE=0`, then download the model with `snapshot_download` |
| `does not appear to have a file named model.safetensors` | The cache has only `config.json`. Run the `snapshot_download` command above |
| Groq `401 AuthenticationError` | Wrong or placeholder API key in `.env` |
| Groq `404 model_not_found` | Use a current model from `client.models.list()`, e.g. `openai/gpt-oss-120b` |
| Gemini auth error | Add a real `GEMINI_API_KEY`, or choose **Groq** in the app |
| `CrossEncoder fallback` notice | Harmless; allow a one-time online run to download the re-ranker |

---

## ⚠️ Limitations & Future Work

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

## 🙏 Acknowledgements

- **Supervisor:** <!-- Prof. ______ -->
- Department of <!-- ______ -->, **IIT Jodhpur**
- Open-source projects: ChromaDB, Qdrant, Docling, MinerU, bm25s, Sentence-Transformers, Streamlit, FastAPI, CLIP

---

<p align="center">Made with ❤️ by <a href="https://github.com/b23cm1024">b23cm1024</a></p>
