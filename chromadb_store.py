"""
chromadb_store.py — V2 Hybrid Retrieval Store for IIT Jodhpur RAG
=================================================================
Provides IITJVectorStore: a unified retrieval class combining:
  - Dense ANN search via ChromaDB (all-MiniLM-L6-v2 cosine similarity)
  - Sparse keyword search via BM25 (bm25s)
  - Reciprocal Rank Fusion (RRF) for score combination
  - Optional metadata pre-filtering (programme, semester, batch_year)
  - Lazy loading: ChromaDB + BM25 index are loaded only on first use

Usage:
    from chromadb_store import IITJVectorStore
    store = IITJVectorStore()
    results = store.search("5th semester AIDE courses", top_k=10)
    # Each result: {"chunk_id": str, "score": float, "text": str, "metadata": dict}
"""

import os, json, re, time
import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

DATA_DIR       = os.path.join(os.path.dirname(__file__), "data")
INDEX_PATH     = os.path.join(DATA_DIR, "index_chunks.json")
CHROMA_DIR     = os.path.join(DATA_DIR, "chromadb")
BM25_DIR       = os.path.join(DATA_DIR, "bm25_index")
COLLECTION_NAME = "iitj_chunks"
EMBED_MODEL    = "all-MiniLM-L6-v2"

# RRF constant (standard value — higher k means less extreme penalisation of lower ranks)
RRF_K = 60

# ────────────────────────────────────────────────────────────────────────────
# Tokeniser (must match embed.py)
# ────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list:
    return re.sub(r'[^a-z0-9\s]', ' ', text.lower()).split()

# ────────────────────────────────────────────────────────────────────────────
# Programme / semester extraction (mirrors embed.py; kept self-contained)
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
    (r'\b(?:semester|sem)[_\s\-]*([1-8])\b',),
    (r'\b([1-8])(?:st|nd|rd|th)\s+(?:semester|sem)\b',),
]
ROMAN_MAP = {'i':1,'ii':2,'iii':3,'iv':4,'v':5,'vi':6,'vii':7,'viii':8}

def _detect_programme_in_query(query: str):
    lower = query.lower()
    for pat, label in PROGRAMME_PATTERNS:
        if re.search(pat, lower):
            return label
    return None

def _detect_semester_in_query(query: str) -> int:
    lower = query.lower()
    for (pat,) in SEM_PATTERNS:
        m = re.search(pat, lower)
        if m:
            v = m.group(1).strip()
            if v.isdigit():
                return int(v)
    # Roman numerals
    m = re.search(r'\b(i{1,3}|iv|v|vi|vii|viii)\s+(?:semester|sem)\b', lower)
    if m:
        return ROMAN_MAP.get(m.group(1), 0)
    m = re.search(r'\b(?:semester|sem)\s+(i{1,3}|iv|v|vi|vii|viii)\b', lower)
    if m:
        return ROMAN_MAP.get(m.group(1), 0)
    return 0

# ────────────────────────────────────────────────────────────────────────────
# IITJVectorStore
# ────────────────────────────────────────────────────────────────────────────

class IITJVectorStore:
    """
    Lazy-loaded hybrid retrieval store.
    Thread-safe for read access (no state mutation after initialisation).
    """

    def __init__(self):
        self._ready  = False
        self._chunks = None   # list[dict] — the full JSON index (for text lookup)
        self._id_to_chunk = None  # dict[str, dict]
        self._chroma_col  = None
        self._bm25_retriever = None
        self._bm25_chunk_ids = None  # list[str] — BM25 row index → chunk id
        self._embed_model = None

    # ── Lazy init ────────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._ready:
            return
        self._load_chunks()
        self._load_chroma()
        self._load_bm25()
        self._load_embed_model()
        self._ready = True
        print(f"[VectorStore] Ready: {len(self._chunks):,} chunks, ChromaDB + BM25 loaded.")

    def _load_chunks(self):
        print(f"[VectorStore] Loading chunk JSON from {INDEX_PATH} ...")
        with open(INDEX_PATH, 'r', encoding='utf-8') as f:
            self._chunks = json.load(f)
        self._id_to_chunk = {}
        for i, c in enumerate(self._chunks):
            cid = c.get("id", str(i))
            self._id_to_chunk[str(cid)] = c

    def _load_chroma(self):
        import chromadb
        if not os.path.isdir(CHROMA_DIR):
            raise RuntimeError(
                f"ChromaDB directory not found at {CHROMA_DIR}.\n"
                "Please run 'python embed.py' first to build the V2 index."
            )
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self._chroma_col = client.get_collection(COLLECTION_NAME)
        print(f"[VectorStore] ChromaDB collection '{COLLECTION_NAME}' loaded "
              f"({self._chroma_col.count():,} items).")

    def _load_bm25(self):
        import bm25s
        id_map_path = os.path.join(BM25_DIR, "chunk_ids.json")
        if not os.path.isdir(BM25_DIR) or not os.path.exists(id_map_path):
            raise RuntimeError(
                f"BM25 index not found at {BM25_DIR}.\n"
                "Please run 'python embed.py' first to build the V2 index."
            )
        self._bm25_retriever = bm25s.BM25.load(BM25_DIR, load_corpus=False)
        with open(id_map_path, 'r', encoding='utf-8') as f:
            self._bm25_chunk_ids = json.load(f)
        print(f"[VectorStore] BM25 index loaded ({len(self._bm25_chunk_ids):,} documents).")

    def _load_embed_model(self):
        from sentence_transformers import SentenceTransformer
        self._embed_model = SentenceTransformer(EMBED_MODEL)
        print(f"[VectorStore] Embedding model '{EMBED_MODEL}' loaded.")

    # ── Direct Channel Retrieval ─────────────────────────────────────────────

    def dense_search(self, query: str, top_k: int = 80, where: dict = None) -> list:
        """Direct dense ANN search returning list of {'id', 'distance', 'chunk'}."""
        self._ensure_loaded()
        q_emb = self._embed_model.encode([query], normalize_embeddings=True)[0].tolist()
        kwargs = dict(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        if where:
            kwargs["where"] = where
        try:
            res = self._chroma_col.query(**kwargs)
        except Exception:
            kwargs.pop("where", None)
            res = self._chroma_col.query(**kwargs)
        results = []
        for cid, dist in zip(res["ids"][0], res["distances"][0]):
            chunk = self._id_to_chunk.get(cid)
            if chunk:
                results.append({"id": cid, "distance": float(dist), "chunk": chunk})
        return results

    def bm25_search(self, query: str, top_k: int = 80) -> list:
        """Direct sparse BM25 search returning list of {'id', 'score', 'chunk'}."""
        self._ensure_loaded()
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        bm25_results, bm25_scores = self._bm25_retriever.retrieve(
            [q_tokens], k=top_k, return_as="tuple"
        )
        results = []
        for doc_idx, score in zip(bm25_results[0], bm25_scores[0]):
            cid = self._bm25_chunk_ids[int(doc_idx)]
            chunk = self._id_to_chunk.get(cid)
            if chunk:
                results.append({"id": cid, "score": float(score), "chunk": chunk})
        return results

    # ── Core retrieval ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        dense_candidates: int = 80,
        sparse_candidates: int = 80,
        programme_filter: str = None,
        semester_filter: int = None,
    ) -> list:
        """
        Hybrid RRF retrieval.

        Returns list of dicts (ranked best first):
            {"score": float, "chunk": dict}
        where chunk is the original JSON dict from index_chunks.json.

        Auto-detects programme and semester from query if not supplied.
        """
        self._ensure_loaded()

        # Auto-detect filters from query
        if programme_filter is None:
            programme_filter = _detect_programme_in_query(query)
        if semester_filter is None or semester_filter == 0:
            semester_filter = _detect_semester_in_query(query)

        # ── Build ChromaDB where clause ──────────────────────────────────────
        q_low = query.lower()
        is_curriculum_query = any(k in q_low for k in [
            "course", "courses", "credit", "credits", "curriculum", "syllabus", "pe", "oe", "pc", "elective", "electives", "semester", "sem", "ltp"
        ])

        where_clause = None
        conditions = []
        # Only constrain programme metadata when query specifically asks about curriculum/courses
        if is_curriculum_query and programme_filter and programme_filter != "GENERAL":
            conditions.append({"$or": [{"programme": {"$eq": programme_filter}}, {"programme": {"$eq": "GENERAL"}}]})
        if semester_filter and semester_filter > 0:
            conditions.append({"semester": {"$eq": semester_filter}})

        if len(conditions) == 1:
            where_clause = conditions[0]
        elif len(conditions) > 1:
            where_clause = {"$and": conditions}

        # ── Dense (ChromaDB ANN) ──────────────────────────────────────────────
        q_emb = self._embed_model.encode([query], normalize_embeddings=True)[0].tolist()

        dense_kwargs = dict(
            query_embeddings=[q_emb],
            n_results=dense_candidates,
            include=["documents", "metadatas", "distances"]
        )
        if where_clause is not None:
            dense_kwargs["where"] = where_clause

        try:
            dense_res = self._chroma_col.query(**dense_kwargs)
        except Exception:
            # Fallback: ignore metadata filter if it fails (e.g. no items match)
            dense_kwargs.pop("where", None)
            dense_res = self._chroma_col.query(**dense_kwargs)

        dense_ids = dense_res["ids"][0]           # list[str]
        dense_dists = dense_res["distances"][0]   # list[float] — cosine distance (lower=better)

        # ── Sparse (BM25) ─────────────────────────────────────────────────────
        q_tokens = _tokenize(query)
        if q_tokens:
            bm25_results, bm25_scores = self._bm25_retriever.retrieve(
                [q_tokens], k=sparse_candidates, return_as="tuple"
            )
            bm25_doc_indices = bm25_results[0]  # 1D array of row indices
            bm25_score_vals  = bm25_scores[0]
        else:
            bm25_doc_indices = []
            bm25_score_vals  = []

        # ── RRF Fusion ────────────────────────────────────────────────────────
        rrf_scores = {}   # chunk_id → rrf score

        for rank, cid in enumerate(dense_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        for rank, doc_idx in enumerate(bm25_doc_indices):
            if rank >= sparse_candidates:
                break
            cid = self._bm25_chunk_ids[int(doc_idx)]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        # ── Materialise results ───────────────────────────────────────────────
        results = []
        for cid in sorted_ids[:top_k * 4]:   # over-retrieve for dedup below
            chunk = self._id_to_chunk.get(cid)
            if chunk is None:
                continue
            results.append({
                "score": rrf_scores[cid],
                "chunk": chunk,
            })

        return results[:top_k * 2]   # caller (rag.py) does final dedup/boost

    # ── Convenience: check if V2 index exists ────────────────────────────────

    @staticmethod
    def is_available() -> bool:
        chroma_ok = os.path.isdir(CHROMA_DIR) and os.path.exists(
            os.path.join(CHROMA_DIR, "chroma.sqlite3")
        )
        bm25_ok = os.path.isdir(BM25_DIR) and os.path.exists(
            os.path.join(BM25_DIR, "chunk_ids.json")
        )
        return chroma_ok and bm25_ok
