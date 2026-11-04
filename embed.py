"""
embed.py — V2 Embedding Pipeline for IIT Jodhpur Academic RAG
==============================================================
Builds:
  1. ChromaDB vector store  (data/chromadb/)   — dense ANN via all-MiniLM-L6-v2
  2. BM25 index             (data/bm25_index/) — sparse keyword index via bm25s

Run once (or after re-ingestion):
    python embed.py

The script is idempotent: existing ChromaDB collections are wiped and rebuilt.
It processes ALL 92k+ chunks from data/index_chunks.json in batches of 512.
"""

import os, sys, json, re, pickle, time
import numpy as np

# Force offline mode for HuggingFace
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_PATH = os.path.join(DATA_DIR, "index_chunks.json")
CHROMA_DIR = os.path.join(DATA_DIR, "chromadb")
BM25_DIR   = os.path.join(DATA_DIR, "bm25_index")
COLLECTION_NAME = "iitj_chunks"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE  = 512

# ────────────────────────────────────────────────────────────────────────────
# Metadata extraction helpers
# ────────────────────────────────────────────────────────────────────────────

PROGRAMME_PATTERNS = [
    (r'\b(aide|ai\s*&?\s*d[sa]|artificial\s+intelligence\s+.*data)', 'AIDE'),
    (r'\bcse\b|computer\s+science', 'CSE'),
    (r'\bme\b|mechanical\s+engineering', 'ME'),
    (r'\bee\b|electrical\s+engineering', 'EE'),
    (r'\bce\b|civil\s+engineering', 'CE'),
    (r'\bchem\b|chemical\s+engineering', 'CHEM'),
    (r'\bci\b|computational\s+intelligence', 'CI'),
    (r'\bbsbe\b|bioscience|biomedical', 'BSBE'),
]

SEM_PATTERNS = [
    (r'\b(?:semester|sem)[_\s\-]*([1-8])\b', None),
    (r'\b([1-8])(?:st|nd|rd|th)\s+(?:semester|sem)\b', None),
    (r'\b(i{1,3}|iv|v|vi|vii|viii)\s+semester\b', None),
]
ROMAN_MAP = {'i':1,'ii':2,'iii':3,'iv':4,'v':5,'vi':6,'vii':7,'viii':8}

BATCH_YEAR_PATTERNS = [
    (r'2025\s+onwards|ay\s+2025|ug\s+2025', '2025_onwards'),
    (r'2019.{0,5}2025|2019.{0,5}2024|165.{0,10}credit', '2019_2024'),
]

def extract_programme(text: str, source: str) -> str:
    combined = (source + " " + text[:400]).lower()
    for pat, label in PROGRAMME_PATTERNS:
        if re.search(pat, combined):
            return label
    return "GENERAL"

def extract_semester(text: str) -> int:
    lower = text.lower()
    for pat, _ in SEM_PATTERNS:
        m = re.search(pat, lower)
        if m:
            val = m.group(1).strip()
            if val.isdigit():
                return int(val)
            return ROMAN_MAP.get(val, 0)
    return 0

def extract_batch_year(text: str, source: str) -> str:
    combined = (source + " " + text[:400]).lower()
    for pat, label in BATCH_YEAR_PATTERNS:
        if re.search(pat, combined):
            return label
    return "unknown"

def safe_meta_value(v) -> str:
    """ChromaDB metadata values must be str/int/float/bool — coerce to str."""
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v) if v is not None else ""

# ────────────────────────────────────────────────────────────────────────────
# Token normaliser for BM25 (simple whitespace + lowercase)
# ────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    return re.sub(r'[^a-z0-9\s]', ' ', text.lower()).split()

# ────────────────────────────────────────────────────────────────────────────
# Main build function
# ────────────────────────────────────────────────────────────────────────────

def build_indexes():
    print(f"[embed.py] Loading chunks from {INDEX_PATH} ...")
    with open(INDEX_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"[embed.py] Loaded {len(chunks):,} chunks.")

    # ── Dense embeddings with all-MiniLM-L6-v2 ──────────────────────────────
    print(f"[embed.py] Loading sentence-transformer model: {EMBED_MODEL} ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    import chromadb
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Wipe existing collection to ensure a clean rebuild
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[embed.py] Deleted existing ChromaDB collection '{COLLECTION_NAME}'.")
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"[embed.py] Created fresh ChromaDB collection '{COLLECTION_NAME}'.")

    # ── BM25 corpus accumulation ──────────────────────────────────────────────
    bm25_corpus_tokens = []   # list[list[str]]
    chunk_ids          = []   # parallel list of chunk IDs (str)
    seen_ids           = set()

    total = len(chunks)
    t0 = time.time()

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        texts = []
        metadatas = []
        ids = []

        for j, c in enumerate(batch):
            global_idx = batch_start + j
            raw_id = str(c.get("id") or global_idx)
            if raw_id in seen_ids:
                chunk_id = f"{raw_id}_{global_idx}"
            else:
                chunk_id = raw_id
            seen_ids.add(chunk_id)
            c["id"] = chunk_id
            text     = c.get("text", "")
            source   = c.get("source", "")
            page     = str(c.get("page", ""))
            modality = c.get("modality", "prose")

            programme = extract_programme(text, source)
            semester  = extract_semester(text)
            batch_year = extract_batch_year(text, source)

            texts.append(text)
            metadatas.append({
                "source":     safe_meta_value(source),
                "page":       safe_meta_value(page),
                "modality":   safe_meta_value(modality),
                "programme":  safe_meta_value(programme),
                "semester":   safe_meta_value(semester),
                "batch_year": safe_meta_value(batch_year),
            })
            ids.append(str(chunk_id))
            bm25_corpus_tokens.append(tokenize(text))
            chunk_ids.append(str(chunk_id))

        # Compute dense embeddings for this batch
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )

        elapsed = time.time() - t0
        done = batch_start + len(batch)
        pct = done / total * 100
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(f"  [{done:>6}/{total}] {pct:5.1f}%  |  {rate:6.0f} chunks/s  |  ETA {eta:5.0f}s", end='\r')

    print(f"\n[embed.py] ChromaDB: {total:,} chunks upserted in {time.time()-t0:.1f}s.")

    # ── Build BM25 index ──────────────────────────────────────────────────────
    print("[embed.py] Building BM25 sparse index ...")
    import bm25s

    retriever = bm25s.BM25()
    retriever.index(bm25_corpus_tokens)

    os.makedirs(BM25_DIR, exist_ok=True)
    retriever.save(BM25_DIR)

    # Also save the chunk ID mapping so retrieval can map BM25 rank → chunk id
    id_map_path = os.path.join(BM25_DIR, "chunk_ids.json")
    with open(id_map_path, 'w', encoding='utf-8') as f:
        json.dump(chunk_ids, f)
    print(f"[embed.py] BM25 index saved to {BM25_DIR}/  ({len(chunk_ids):,} documents).")

    # Update index_chunks.json with unique IDs
    with open(INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"[embed.py] Saved updated chunks to {INDEX_PATH}.")
    print("[embed.py] V2 index build COMPLETE.")


if __name__ == "__main__":
    build_indexes()
