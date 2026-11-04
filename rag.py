import os
import re
import json
import pickle
import uuid
import numpy as np
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
INDEX_PATH = os.path.join(DATA_DIR, "index_chunks.json")
TFIDF_PATH = os.path.join(DATA_DIR, "tfidf.pkl")

# ── V2 hybrid store (ChromaDB + BM25). Falls back to TF-IDF when not built. ──
_v2_store = None
_v2_store_attempted = False

def _get_v2_store():
    global _v2_store, _v2_store_attempted
    if _v2_store_attempted:
        return _v2_store
    _v2_store_attempted = True
    try:
        from chromadb_store import IITJVectorStore
        if IITJVectorStore.is_available():
            _v2_store = IITJVectorStore()
            print("[RAG Engine] V2 hybrid store (ChromaDB + BM25) active.")
        else:
            print("[RAG Engine] V2 index not yet built — using TF-IDF fallback. Run embed.py to upgrade.")
    except Exception as e:
        print(f"[RAG Engine] V2 store load failed ({e}) — using TF-IDF fallback.")
    return _v2_store

# Dictionary of frequent phonetic / typing mistakes
TYPO_MAPPING = {
    "atatendence": "attendance",
    "attendence": "attendance",
    "attandance": "attendance",
    "attandence": "attendance",
    "atendance": "attendance",
    "curriculam": "curriculum",
    "curriculm": "curriculum",
    "curiculum": "curriculum",
    "syllubus": "syllabus",
    "syllbus": "syllabus",
    "admissoin": "admission",
    "admisson": "admission",
    "hostal": "hostel",
    "facuty": "faculty",
    "faculti": "faculty",
    "profesor": "professor",
    "libary": "library",
    "examnation": "examination",
    "exam": "examination",
    "regualtion": "regulation",
    "regualtions": "regulations",
    "ordinace": "ordinance",
    "calender": "calendar",
    "eligiblity": "eligibility",
    "grad": "grade",
    "gradeing": "grading",
    "semster": "semester",
    "sem": "semester",
    "sems": "semesters",
}

# Full standard English stopwords + institute generic filler words
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", 
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", 
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", 
    "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", 
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", "should", 
    "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them", "themselves", 
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", 
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", 
    "why", "will", "with", "you", "your", "yours", "yourself", "yourselves",
    "iitj", "iit", "jodhpur", "institute", "indian", "technology", "tell", "show", "give", "please",
    "rule", "rules", "policy", "detail", "details"
}

CRITICAL_ACRONYMS = {
    "ai", "ds", "cs", "ee", "me", "ce", "pe", "oe", "pc", "pp", "ie", "is", "ih", "ne", "nh", "nd",
    "phc", "opd", "hod",
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"
}

def preprocess_query(query: str) -> tuple[str, list[str]]:
    """Clean typos, normalize keywords, preserve domain acronyms, and expand numerical equivalents."""
    q_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', query.lower())
    raw_tokens = q_clean.split()
    tokens = [TYPO_MAPPING.get(t, t) for t in raw_tokens]
    
    # Preserve words >2 chars OR critical domain/roman acronyms (ai, ds, pe, oe, etc.)
    topic_keywords = [t for t in tokens if (len(t) > 2 or t in CRITICAL_ACRONYMS) and t not in STOPWORDS]
    
    # Bidirectional numerical equivalents: roman, serial, ordinal, words
    num_expansions = {
        "8th": ["viii", "8", "eighth"],
        "8": ["viii", "8th", "eighth"],
        "eighth": ["viii", "8", "8th"],
        "viii": ["8", "8th", "eighth"],
        "7th": ["vii", "7", "seventh"],
        "7": ["vii", "7th", "seventh"],
        "seventh": ["vii", "7", "7th"],
        "vii": ["7", "7th", "seventh"],
        "6th": ["vi", "6", "sixth"],
        "6": ["vi", "6th", "sixth"],
        "sixth": ["vi", "6", "6th"],
        "vi": ["6", "6th", "sixth"],
        "5th": ["v", "5", "fifth"],
        "5": ["v", "5th", "fifth"],
        "fifth": ["v", "5", "5th"],
        "v": ["5", "5th", "fifth"],
        "4th": ["iv", "4", "fourth"],
        "4": ["iv", "4th", "fourth"],
        "fourth": ["iv", "4", "4th"],
        "iv": ["4", "4th", "fourth"],
        "3rd": ["iii", "3", "third"],
        "3": ["iii", "3rd", "third"],
        "third": ["iii", "3", "3rd"],
        "iii": ["3", "3rd", "third"],
        "2nd": ["ii", "2", "second"],
        "2": ["ii", "2nd", "second"],
        "second": ["ii", "2", "2nd"],
        "ii": ["2", "2nd", "second"],
        "1st": ["i", "1", "first"],
        "1": ["i", "1st", "first"],
        "first": ["i", "1", "1st"],
        "i": ["1", "1st", "first"],
    }
    
    # Domain concept expansions
    concept_expansions = {
        "ai": ["aide", "artificial", "intelligence"],
        "ds": ["data", "science"],
        "aide": ["ai", "artificial", "intelligence", "data"],
        "aids": ["ai", "artificial", "intelligence", "data"],
        "pe": ["programme", "elective", "electives"],
        "oe": ["open", "elective", "electives"],
        "pc": ["programme", "compulsory"],
        "pp": ["project", "btech", "capstone"],
        "phc": ["primary", "health", "centre", "center", "hospital", "clinic", "medical"],
        "health": ["primary", "centre", "center", "hospital", "medical", "clinic", "phc"],
        "medical": ["health", "centre", "center", "hospital", "phc", "opd", "doctor"],
        "hospital": ["health", "centre", "primary", "phc", "goyal", "medical"],
        "doctor": ["health", "centre", "medical", "phc", "opd", "physician"],
        "doctors": ["health", "centre", "medical", "phc", "opd", "physicians"],
        "hod": ["head", "heads", "chairperson", "leadership", "department"],
        "head": ["hod", "heads", "chairperson", "leadership"],
        "mechanical": ["me", "mech"],
        "me": ["mechanical"],
        "cse": ["computer", "science"],
        "ee": ["electrical", "engineering"]
    }
    
    for t in list(tokens):
        if t in num_expansions:
            topic_keywords.extend(num_expansions[t])
        if t in concept_expansions:
            topic_keywords.extend(concept_expansions[t])

    # If all tokens were filtered as stopwords, retain non-typo tokens
    if not topic_keywords:
        topic_keywords = [t for t in tokens if len(t) > 2 or t in CRITICAL_ACRONYMS]
        
    normalized_query = " ".join(tokens)
    return normalized_query, topic_keywords

def is_personal_financial_query(query: str) -> bool:
    """
    Firewall to prevent disclosure of confidential personal salary, remuneration,
    individual earnings, payslips, or immovable property returns of specific faculty members or employees.
    NEVER blocks academic questions about earning credits, minors, degrees, paying semester/hostel fees, etc.
    """
    q_lower = query.lower()
    
    # Academic safe words that immediately disqualify a query from being personal financial
    # unless it explicitly specifies a named professor's salary/earnings
    academic_safe_terms = [
        r'\bcredits?\b', r'\bminor\b', r'\bspecialization\b', r'\bdegree\b',
        r'\bcourse\b', r'\bcourses\b', r'\bgrade\b', r'\bgrades\b', r'\bcgpa\b',
        r'\bsgpa\b', r'\bcurriculum\b', r'\bsyllabus\b', r'\bsemester\b', r'\bsem\b',
        r'\bbranch\s+change\b', r'\bbranch\b', r'\battendance\b', r'\bhostel\b',
        r'\btuition\b', r'\bfee\s+structure\b', r'\badmission\b'
    ]
    
    # Financial keywords specifically associated with employee compensation/wealth
    fin_patterns = [
        r'\bsalar(?:y|ies)\b',
        r'\bpay(?:slip|slips|check|checks|out|outs)?\b',
        r'\bgross\s+pay\b',
        r'\bnet\s+pay\b',
        r'\btake\s*home\b',
        r'\bremunerat(?:ion|ions)?\b',
        r'\bcompensation\b',
        r'\bctc\b',
        r'\bimmovable\s+propert(?:y|ies)\b',
        r'\bpropert(?:y|ies)\s+return\b',
        r'\bhow\s+much\b.*\b(?:make|makes|made|earn|earns|get|gets|paid)\b'
    ]
    
    has_fin = any(re.search(pat, q_lower) for pat in fin_patterns)
    if not has_fin:
        return False

    is_academic = any(re.search(ast, q_lower) for ast in academic_safe_terms)
    
    # Specific individual person or named faculty indicators
    individual_target_patterns = [
        r'\bdr\.?\s+[a-z]+',
        r'\bprof(?:essor)?\s+[a-z]+',
        r'\bfaculty\b',
        r'\bteacher\b',
        r'\bstaff\b',
        r'\bemployee\b',
        r'\bdirector\b',
        r'\bregistrar\b',
        r'\bdean\b',
        r'\bwho\b',
        r'\bhis\b',
        r'\bher\b',
        r'\bhim\b',
        r'\bindividual\b',
        r'\bperson\b',
        r'\bkalra\b',
        r'\bparanjothy\b',
        r'\bmanikandan\b',
        r'\bname\b'
    ]
    has_specific_individual = any(re.search(ip, q_lower) for ip in individual_target_patterns)
    
    # Check for general institutional scale of pay without any personal or faculty inquiry
    scale_patterns = [
        r'\bscale\s+of\s+pay\b',
        r'\bpay\s+scale\b',
        r'\bpay\s+band\b',
        r'\bpay\s+level\b',
        r'\b7th\s+cpc\b',
        r'\bpay\s+matrix\b'
    ]
    is_general_scale_syntax = any(re.search(sp, q_lower) for sp in scale_patterns)
    
    # If general scale question (e.g. "what is the general scale of pay for assistant professor" or "7th cpc pay level")
    # and NOT asking about an individual's personal earnings/salary/financial records
    personal_record_terms = [r'\bfinancial\s+record', r'\bpayslip', r'\bpackage\b', r'\bctc\b', r'\bproperty\s+return']
    has_personal_record_demand = any(re.search(pr, q_lower) for pr in personal_record_terms)
    
    if is_general_scale_syntax and not has_personal_record_demand:
        return False

    # If it is an academic question (e.g. earning credits/minor) and NOT asking about an individual faculty's salary/earnings
    if is_academic and not has_specific_individual:
        return False
        
    # Block if targeting an individual or explicit employee salary/remuneration/property returns
    if has_specific_individual or re.search(r'\b(?:immovable\s+property|property\s+return|remuneration|salary|payslip)\b', q_lower):
        return True
        
    return False

def canonicalize_source_name(src: str) -> str:
    """Normalize file names, intranet duplicate prefixes, url encoded strings, and aliases to a single canonical source cluster."""
    s = src.lower()
    # Strip URL prefixes and download wrappers
    s = re.sub(r'^view_\(page_opens_in_intranet_only\)_', '', s)
    s = re.sub(r'^\(download_file(?:_\d+_[a-z]+)?\)_', '', s)
    s = re.sub(r'^(?:english|hindi)format_type__pdf,_', '', s)
    s = re.sub(r'^format_type__pdf,_', '', s)
    s = s.replace('%20', '_').replace('-', '_').replace(' ', '_')
    s = re.sub(r'(\.pdf)+$', '.pdf', s)
    s = re.sub(r'_\d{10,}\.pdf$', '.pdf', s)
    
    # Semantic document clustering for identical/superseded uploads
    if any(k in s for k in ["revised_ug_curriculum_aide", "btech_ai_and_data_science_curriculum_2025", "detailed_curriculum_for_b.tech._aide_(ay_2026_onwards)"]):
        return "curriculum_aide_2025_onwards"
    if any(k in s for k in ["btech_ai_and_data_science_curriculum_2019", "curriculum_btech_aid"]):
        return "curriculum_aide_2019_2025"
    if "cse" in s and ("2020" in s or "program_structure" in s):
        return "curriculum_cse_2020"
    if "cse" in s and ("curriculum" in s or "revised" in s or "btech" in s or "ug" in s):
        return "curriculum_cse_revised"
    if any(k in s for k in ["ug_ee_curriculum", "btech_ee"]):
        return "curriculum_ee"
    if "me" in s and ("2020" in s or "program_structure" in s):
        return "curriculum_me_2020"
    if any(k in s for k in ["concept_note_btech_me", "curriculum_of_the_b.tech._(me)", "btech_me"]):
        return "curriculum_me_revised"
    if "circular_51" in s or "attendance_policy" in s:
        return "circular_51_attendance"
    if "circular_52" in s or "examination_policy" in s:
        return "circular_52_examination"
    if "academic_regulations" in s:
        return "academic_regulations"
    if "fee_structure" in s:
        return "fee_structure"
    if "constitution_of_the_senate" in s or "senate" in s:
        return "senate_constitution"
    return s

def clean_html_tags(text: str) -> str:
    """Removes raw HTML tags like <br>, <p>, <span> while preserving clean markdown formatting."""
    if not text:
        return ""
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            # Inside markdown table rows, replace <br> with a clean slash or semicolon
            line = re.sub(r'<\s*br\s*/?\s*>', ' / ', line, flags=re.IGNORECASE)
        else:
            # Outside tables, replace <br> with newline
            line = re.sub(r'<\s*br\s*/?\s*>', '\n', line, flags=re.IGNORECASE)
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r'<\s*br\s*/?\s*>', '\n', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'</?(?:p|span|div)[^>]*>', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace('\u202f', ' ').replace('\u00a0', ' ')
    return cleaned

class IITJRagEngine:
    def __init__(self):
        self.chunks = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.cross_encoder = None
        self.load_index()

    def load_index(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(TFIDF_PATH):
            with open(INDEX_PATH, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            with open(TFIDF_PATH, 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data["vectorizer"]
                self.tfidf_matrix = data["matrix"]
            print(f"[RAG Engine] Loaded {len(self.chunks)} chunks into search index.")
            try:
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("[RAG Engine] Local Neural Cross-Encoder loaded successfully (offline mode).")
            except Exception as e:
                print(f"[RAG Engine] Notice: CrossEncoder fallback ({e})")
        else:
            print("[RAG Engine] Warning: Index not found. Please run ingest.py first.")

    def search(self, query: str, top_k: int = 8):
        """
        Multi-modal semantic retrieval — V3 Scope-Aware Multi-Channel Path
        with automatic fallback to V2 and TF-IDF.
        """
        if is_personal_financial_query(query):
            return []

        # ────────────────────────────────────────────────────────────────────
        # V3 SCOPE-AWARE RETRIEVAL PATH
        # ────────────────────────────────────────────────────────────────────
        try:
            from retrieval_engine_v3 import retrieval_engine_v3
            candidates, _ = retrieval_engine_v3.retrieve_and_assess(query, top_k=top_k)
            if candidates:
                return [
                    {
                        "score": c.scores.cross_encoder_score if c.scores.cross_encoder_score is not None else (c.scores.rrf_score or 0.0),
                        "raw_score": c.scores.rrf_score or 0.0,
                        "chunk": c.chunk
                    }
                    for c in candidates
                ]
        except Exception as e:
            pass

        q_lower = query.lower()
        is_tabular_query = any(k in q_lower for k in [
            "credits", "credit", "how many", "pe", "oe", "pc", "fee", "fees", "amount",
            "courses", "table", "breakdown", "structure", "audit", "graded", "ltp", "l-t-p",
            "semester", "sem", "semster", "plan", "schedule", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"
        ])
        is_policy_query = any(k in q_lower for k in [
            "attendance", "policy", "circular", "rule", "rules", "regulations",
            "amendment", "de-registration", "withdrawal", "exam", "examination", "passing", "grade", "probation"
        ])
        is_governance_query = any(k in q_lower for k in [
            "hod", "head", "heads", "dean", "deans", "director", "senate", "chairman", "chairperson", "leadership"
        ])

        # ────────────────────────────────────────────────────────────────────
        # V2 HYBRID RETRIEVAL PATH
        # ────────────────────────────────────────────────────────────────────
        v2 = _get_v2_store()
        if v2 is not None:
            try:
                raw_results = v2.search(
                    query,
                    top_k=top_k,
                    dense_candidates=100,
                    sparse_candidates=100,
                )
                cand_chunks = [r["chunk"] for r in raw_results]
                # Use RRF scores as initial scores for boost/rerank
                rrf_scores  = [r["score"] for r in raw_results]

                # Privacy filter
                cand_chunks_filtered = []
                rrf_scores_filtered  = []
                for c, s in zip(cand_chunks, rrf_scores):
                    if not any(bad in c.get('source', '').lower() for bad in [
                        'gross%20salary', 'gross_salary', 'salary%20jan', 'salary%20july', 'monthly remuneration'
                    ]):
                        cand_chunks_filtered.append(c)
                        rrf_scores_filtered.append(s)
                cand_chunks = cand_chunks_filtered
                initial_scores = rrf_scores_filtered

                # ── Modality channel boosting: inject high-priority structured tables
                #    and circulars that may have been missed (manual channel injection)
                if is_tabular_query:
                    extra_tables = [
                        c for c in self.chunks
                        if c.get("modality") == "structured_table"
                        and c not in cand_chunks
                    ]
                    # Score them with a small base RRF so they participate in ranking
                    for c in extra_tables[:60]:
                        cand_chunks.append(c)
                        initial_scores.append(0.002)

                if is_policy_query:
                    extra_circulars = [
                        c for c in self.chunks
                        if c.get("modality") == "official_circular"
                        and c not in cand_chunks
                    ]
                    for c in extra_circulars[:40]:
                        cand_chunks.append(c)
                        initial_scores.append(0.002)

                if is_governance_query:
                    extra_gov = [
                        c for c in self.chunks
                        if any(k in c.get("text", "").lower() for k in ["heads of the departments", "heads of departments", "constitution of senate", "key functionaries"])
                        and c not in cand_chunks
                    ]
                    for c in extra_gov[:40]:
                        cand_chunks.append(c)
                        initial_scores.append(0.002)

                scores = initial_scores
                using_v2 = True

            except Exception as e:
                print(f"[RAG Engine] V2 search error ({e}); falling back to TF-IDF.")
                v2 = None

        # ────────────────────────────────────────────────────────────────────
        # TF-IDF FALLBACK PATH
        # ────────────────────────────────────────────────────────────────────
        if v2 is None:
            using_v2 = False
            if not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
                return []

            norm_query, topic_keywords = preprocess_query(query)
            search_query = norm_query + " " + " ".join(topic_keywords)

            query_vec = self.vectorizer.transform([search_query])
            sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

            general_candidates = list(np.argsort(sims)[::-1][:150])
            table_candidates = []
            if is_tabular_query:
                table_indices = [i for i, c in enumerate(self.chunks) if c.get("modality") == "structured_table"]
                table_scored = [(i, sims[i]) for i in table_indices if sims[i] > 0.005]
                table_scored.sort(key=lambda x: x[1], reverse=True)
                table_candidates = [idx for idx, _ in table_scored[:60]]
            policy_candidates = []
            if is_policy_query:
                policy_indices = [i for i, c in enumerate(self.chunks) if c.get("modality") == "official_circular"]
                policy_scored = [(i, sims[i]) for i in policy_indices if sims[i] > 0.005]
                policy_scored.sort(key=lambda x: x[1], reverse=True)
                policy_candidates = [idx for idx, _ in policy_scored[:40]]

            seen_cand = set()
            candidate_indices = []
            for idx in table_candidates + policy_candidates + general_candidates:
                if idx not in seen_cand and sims[idx] > 0.005:
                    seen_cand.add(idx)
                    candidate_indices.append(idx)

            cand_chunks = [self.chunks[i] for i in candidate_indices]
            cand_chunks = [
                c for c in cand_chunks
                if not any(bad in c.get('source', '').lower() for bad in [
                    'gross%20salary', 'gross_salary', 'salary%20jan', 'salary%20july', 'monthly remuneration'
                ])
            ]
            scores = [sims[candidate_indices[j]] for j in range(len(cand_chunks))]

        if not cand_chunks:
            return []

        # ── Neural Cross-Encoder Re-Ranking (1,500-char window) ──────────────
        if self.cross_encoder is not None:
            pairs = [(query, c['text'][:1500]) for c in cand_chunks]
            ce_scores = self.cross_encoder.predict(pairs)
            # Blend: 70% Cross-Encoder + 30% initial retrieval score
            ce_arr = np.array(ce_scores, dtype=float)
            init_arr = np.array(scores[:len(cand_chunks)], dtype=float)
            # Normalise initial scores to same scale as CE (roughly -10..10)
            init_min, init_max = init_arr.min(), init_arr.max()
            if init_max > init_min:
                init_norm = (init_arr - init_min) / (init_max - init_min) * 20 - 10
            else:
                init_norm = init_arr * 0
            scores = list(0.70 * ce_arr + 0.30 * init_norm)
        # (else: keep initial retrieval scores as-is)

        # ── Modality-Targeted Semantic Boost ──────────────────────────────────
        sem_match = re.search(r'\b(1st|2nd|3rd|4th|5th|6th|7th|8th|first|second|third|fourth|fifth|sixth|seventh|eighth|[1-8]|i|ii|iii|iv|v|vi|vii|viii)\s*(?:st|nd|rd|th)?\s+(?:sem|semester|semster)\b|\b(?:sem|semester|semster)\s+([1-8]|i|ii|iii|iv|v|vi|vii|viii)\b', q_lower)
        sem_val = None
        sem_equivs = []
        if sem_match:
            sem_val = sem_match.group(1) or sem_match.group(2)
            num_expansions_local = {
                "8th": ["viii", "8", "eighth"], "8": ["viii", "8th", "eighth"], "eighth": ["viii", "8", "8th"], "viii": ["8", "8th", "eighth"],
                "7th": ["vii", "7", "seventh"], "7": ["vii", "7th", "seventh"], "seventh": ["vii", "7", "7th"], "vii": ["7", "7th", "seventh"],
                "6th": ["vi", "6", "sixth"], "6": ["vi", "6th", "sixth"], "sixth": ["vi", "6", "6th"], "vi": ["6", "6th", "sixth"],
                "5th": ["v", "5", "fifth"], "5": ["v", "5th", "fifth"], "fifth": ["v", "5", "5th"], "v": ["5", "5th", "fifth"],
                "4th": ["iv", "4", "fourth"], "4": ["iv", "4th", "fourth"], "fourth": ["iv", "4", "4th"], "iv": ["4", "4th", "fourth"],
                "3rd": ["iii", "3", "third"], "3": ["iii", "3rd", "third"], "third": ["iii", "3", "3rd"], "iii": ["3", "3rd", "third"],
                "2nd": ["ii", "2", "second"], "2": ["ii", "2nd", "second"], "second": ["ii", "2", "2nd"], "ii": ["2", "2nd", "second"],
                "1st": ["i", "1", "first"], "1": ["i", "1st", "first"], "first": ["i", "1", "1st"], "i": ["1", "1st", "first"]
            }
            sem_equivs = num_expansions_local.get(sem_val, [sem_val]) + [sem_val]

        adjusted_scores = []
        for i, score in enumerate(scores):
            adj = float(score)
            c = cand_chunks[i]
            modality = c.get("modality", "")
            c_text = c.get("text", "")

            # Boost structured tables when query asks quantitative / credit / fee questions
            if is_tabular_query:
                if modality == "structured_table" or "| --- |" in c_text or "|:---" in c_text:
                    adj += 2.6
                elif "table" in c.get("page", "").lower():
                    adj += 1.5

            # Target semester boost: Give a strong priority boost when chunk contains the exact requested semester plan/table
            if sem_equivs:
                c_lower = c_text.lower()
                for eq in sem_equivs:
                    if f"{eq} semester" in c_lower or f"semester {eq}" in c_lower or f"semester ({eq})" in c_lower or f"sem {eq}" in c_lower:
                        adj += 3.2
                        break

            # Boost statutory circulars and regulations when query asks about attendance, examinations, or rules
            if is_policy_query:
                if modality == "official_circular" or "circular no. 51" in c_text.lower() or "circular no. 52" in c_text.lower():
                    adj += 2.2
                elif "academic_regulations" in c.get("source", "").lower() or "section 4.1.9" in c_text.lower():
                    adj += 1.8

            # Boost institutional governance records when query asks about HOD, Dean, Director, Senate
            if is_governance_query:
                c_low = c_text.lower()
                if "heads of the departments" in c_low or "heads of departments" in c_low or "key functionaries" in c_low:
                    adj += 3.2
                elif "constitution of senate" in c_low or "constitution of the senate" in c_low:
                    adj += 2.2

            adjusted_scores.append(adj)

        ranked_indices = np.argsort(adjusted_scores)[::-1]

        # 5. Canonical Multi-Document De-Duplication & Near-Duplicate Content Suppression
        # Dynamic cluster allocation: Allow up to 4 chunks for the top-scoring primary document cluster, 2 for secondary clusters
        results = []
        source_counts = {}
        seen_text_fingerprints = set()
        primary_cluster = None
        if len(ranked_indices) > 0:
            primary_cluster = canonicalize_source_name(cand_chunks[ranked_indices[0]].get("source", ""))

        for idx in ranked_indices:
            c = cand_chunks[idx]
            src = c.get("source", "")
            canon_key = canonicalize_source_name(src)

            # Suppress identical text paragraphs uploaded under duplicate intranet filenames
            raw_txt = re.sub(r'\[.*?\]', '', c.get("text", "")).strip()
            txt_fp = re.sub(r'\s+', ' ', raw_txt[:85]).lower().strip()
            if len(txt_fp) > 20 and txt_fp in seen_text_fingerprints:
                continue

            max_allowed = 4 if (canon_key == primary_cluster) else 2
            if source_counts.get(canon_key, 0) >= max_allowed:
                continue

            seen_text_fingerprints.add(txt_fp)
            source_counts[canon_key] = source_counts.get(canon_key, 0) + 1
            results.append({
                "score": float(adjusted_scores[idx]),
                "raw_score": float(scores[idx]),
                "chunk": c
            })
            if len(results) >= top_k:
                break

        return results

    def generate_answer(
        self,
        query: str,
        api_provider: str = None,
        api_key: str = None,
        model_name: str = None,
        chat_history: list = None
    ):
        """Generates answer using retrieved IITJ context and chosen LLM with strict request isolation and topic validation."""
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
        if api_provider is None:
            if os.getenv("GROQ_API_KEY"):
                api_provider = "Groq"
            elif os.getenv("GEMINI_API_KEY"):
                api_provider = "Google Gemini"
            else:
                api_provider = "Deterministic"
        if model_name is None:
            model_name = "openai/gpt-oss-120b" if api_provider == "Groq" else ("gemini-1.5-flash" if api_provider == "Google Gemini" else None)

        request_id = uuid.uuid4().hex[:12]
        from execution_tracer import tracer
        from query_understanding import analyze_query
        trace = tracer.create_trace(request_id, query)
        q_profile = analyze_query(query)
        trace.domain = q_profile.domain.value
        trace.intent = q_profile.intent.value
        trace.normalized_query = q_profile.normalized_query
        trace.subject_entity = q_profile.subject_entity
        trace.target_attributes = q_profile.target_attributes
        trace.search_rewrites = q_profile.search_rewrites

        # Absolute privacy guardrail against disclosing individual financial/salary records
        if is_personal_financial_query(query):
            ans = (
                "In accordance with institutional privacy policies, data protection regulations, and administrative ethics, "
                "personal financial records, individual salary payouts, remuneration figures, and property returns of specific faculty members or employees "
                "are confidential and strictly protected from disclosure.\n\n"
                "For official information on faculty compensation scales, please refer to the general 7th Central Pay Commission (7th CPC) pay matrices (e.g. Academic Pay Level 10 for Assistant Professor to Level 14A for Professor) published in official Institute recruitment notifications: [Scale of Pay | Faculty Positions](http://iitj.ac.in/faculty-positions/en/scale-of-pay)."
            )
            trace.final_response_preview = ans[:200]
            tracer.save_trace(trace)
            return {
                "answer": ans,
                "sources": [],
                "_request_id": request_id
            }

        # ── Master Institutional Intelligence Pipeline (Directive Section 20) ──
        try:
            from evidence_orchestrator import institutional_orchestrator
            orch_resp = institutional_orchestrator.generate_answer(
                query=query,
                api_provider=api_provider,
                api_key=api_key,
                model_name=model_name,
                chat_history=chat_history
            )
            orch_resp["_request_id"] = request_id
            return orch_resp
        except Exception as e_orch:
            print(f"[RAG Engine] Institutional orchestrator notice ({e_orch}); continuing with V3 engine.")

        # ── V3 Scope-Aware Evidence Pipeline ──────────────────────────────────
        try:
            from retrieval_engine_v3 import retrieval_engine_v3
            from evidence_grounded_generator import generate_grounded_answer, synthesize_direct_grounded_fallback
            candidates, assessment = retrieval_engine_v3.retrieve_and_assess(query, top_k=8)
            if assessment:
                if api_key:
                    return generate_grounded_answer(
                        query=query,
                        candidates=candidates,
                        assessment=assessment,
                        api_provider=api_provider,
                        api_key=api_key,
                        model_name=model_name or "openai/gpt-oss-120b",
                        chat_history=chat_history,
                        request_id=request_id
                    )
                else:
                    fb_ans = synthesize_direct_grounded_fallback(q_profile, assessment, candidates)
                    srcs = []
                    for c in candidates[:6]:
                        c_src = c.chunk.get("source", "")
                        if any(w in query.lower() for w in ["withdrawal", "withdraw", "leave", "sabbatical", "grade", "attendance", "probation", "fellowship", "hod", "head"]):
                            if any(k in c_src.lower() for k in ["curricul", "structure", "04_ug", "03_chemical", "10_curriculum", "07_ug", "08_concept", "05_revised", "02_ug", "01_ug", "bsbe"]):
                                continue
                        srcs.append({
                            "source": c_src,
                            "page": c.chunk.get("page", 1),
                            "text": c.chunk.get("text", "")[:300]
                        })
                    return {
                        "answer": fb_ans,
                        "sources": srcs,
                        "_request_id": request_id
                    }
        except Exception as e:
            print(f"[RAG Engine] V3 Engine notice ({e}); continuing with legacy path.")

        if not api_key:
            ans = "Please configure your API key in System Settings to enable AI answers."
            trace.final_response_preview = ans
            tracer.save_trace(trace)
            return {
                "answer": ans,
                "sources": [],
                "_request_id": request_id
            }

        retrieved = self.search(query, top_k=8)
        # Defense-in-depth: Filter curriculum documents if query is strictly regulatory or governance
        non_curr_kw = [
            "withdrawal", "withdraw", "leave", "sabbatical", "grade", "grading", "incomplete",
            "probation", "termination", "appeal", "correction", "attendance", "absent",
            "fellowship", "assistantship", "overhead", "consultancy", "health", "phc", "ambulance",
            "hospital", "hod", "head of department"
        ]
        if any(w in query.lower() for w in non_curr_kw) and not any(k in query.lower() for k in ["curriculum", "syllabus", "b.tech in", "btech in", "course structure", "credit structure"]):
            filtered_ret = [
                item for item in retrieved
                if not any(k in item["chunk"].get("source", "").lower() for k in [
                    "curricul", "structure", "04_ug", "03_chemical", "10_curriculum", "07_ug",
                    "08_concept", "05_revised", "02_ug", "01_ug", "b.tech", "btech", "bsbe"
                ])
            ]
            if filtered_ret:
                retrieved = filtered_ret

        if not retrieved:
            return {
                "answer": "I could not find relevant rules in the indexed IIT Jodhpur documents for your question.",
                "sources": []
            }

        # ── Retrieval Validation (V3) ────────────────────────────────────────
        try:
            from retrieval_validator import validate_retrieval, insufficient_evidence_response
            ok, val_report, confidence = validate_retrieval(query, retrieved)
            if not ok:
                # On validation failure: escalate by widening search before giving up
                extended = self.search(query, top_k=16)
                ok2, val_report2, confidence2 = validate_retrieval(query, extended)
                if ok2:
                    retrieved = extended
                else:
                    # Still weak — return transparent insufficient-evidence message
                    return insufficient_evidence_response(val_report2)
        except Exception:
            pass  # Validator import failure is non-fatal; continue with retrieved results

        # Dynamic context budgeting to keep prompts strictly within free-tier provider rate limits (e.g. Groq 7k ITPM)
        effective_retrieved = retrieved[:5] if api_provider == "Groq" else retrieved
        context_blocks = []
        for i, item in enumerate(effective_retrieved):
            c = item["chunk"]
            loc = f"Document: {c['source']}, Page/Section: {c['page']}"
            txt = c['text']
            if api_provider == "Groq" and len(txt) > 1100:
                txt = txt[:1100] + "..."
            context_blocks.append(f"--- [EXCERPT {i+1} | {loc}] ---\n{txt}")

        context_text = "\n\n".join(context_blocks)

        system_instruction = (
            "You are the official IIT Jodhpur Academic Advisor AI assistant.\n"
            "Your task is to answer user questions accurately, completely, and strictly according to the official IIT Jodhpur documents, circulars, and portals provided in the context excerpts below.\n\n"
            "STRICT GROUNDING & ZERO-SPECULATION CONSTRAINTS:\n"
            "0. CLEAN FORMATTING MANDATE: NEVER output raw HTML tags like <br>, <p>, <span>, or <div> anywhere in your response or in tables. Always format line breaks and multi-line items using standard Markdown newlines and clean bullet points.\n"
            "1. CHAIN-OF-EVIDENCE MANDATE: For every quantitative value (credits, fees, course counts, percentages), passing grade requirement, attendance threshold, and policy rule you state, you MUST explicitly cite the exact document name, page number, and table or section from the provided excerpts.\n"
            "2. ABSOLUTE ZERO-SPECULATION DIRECTIVE: NEVER guess, extrapolate, estimate, or fill in gaps from general university conventions or 'standard frameworks' (e.g. NEVER speculate that PE is 12 or OE is 6 based on general assumptions). If a specific number, fee, or rule is not explicitly written in the provided excerpts, state: 'The official retrieved documents do not explicitly state this specific figure.'\n"
            "3. NO SPECULATIVE DISCLAIMERS: Strictly avoid formulating speculative disclaimers (e.g. 'Disclaimer: The exact number of PE credits is based on standard UG frameworks...'). State only facts that are visibly verifiable in the excerpts.\n\n"
            "Canonical Interpretation & Equivalences:\n"
            "1. Numerical & Word Equivalence: Treat Roman numerals (I, II, III, IV, V, VI, VII, VIII), Arabic numbers (1, 2, 3, 4, 5, 6, 7, 8), ordinal words ('first', 'second', ... 'eighth'), and shorthand ('1st', '2nd', ... '8th', 'Sem 8', 'Semester VIII') as COMPLETELY IDENTICAL and interchangeable designations for semesters, regulations, and course levels.\n"
            "2. Curriculum Abbreviations: Interpret curriculum acronyms accurately: 'PE' = Programme Elective, 'OE' = Open Elective, 'PC' = Programme Compulsory, 'PP' = Programme/B.Tech Project, 'IH/IE/IS' = Institute Humanities/Engineering/Science Core, 'NE/NH/ND' = Essential Audit (Non-graded).\n"
            "3. MANDATORY DUAL-COHORT REPORTING DIRECTIVE FOR CURRICULUM & ELECTIVES:\n"
            "   Whenever asked about elective credits (PE and OE), credit distributions, or degree requirements for any B.Tech programme (e.g., CSE, AIDE, EE, ME, Civil) without the user specifying a single batch, you MUST ALWAYS PROACTIVELY REPORT BOTH CURRICULUM EDITIONS:\n"
            "   (1) Earlier Scheme (2019–2024 Enrolled Batches, 165-Credit Framework):\n"
            "       - Programme Electives (PE): EXACTLY 17 CREDITS (or 18 for Double B.Tech).\n"
            "       - Open Electives (OE): EXACTLY 10 CREDITS (0 for Double B.Tech).\n"
            "       - Combined Electives Total: 27 CREDITS (scheduled across Sem VI: 6 credits, Sem VII: 6 credits, Sem VIII: 15 credits).\n"
            "       - Total Graded: 150 Credits | Non-Graded / Audit: 15 Credits (9 non-graded + 6 design credits) | Grand Total Degree Requirement: 165 CREDITS.\n"
            "   (2) Revised Scheme (2025 Onwards Batches, 153 Credits for CSE / 154 Credits for AIDE):\n"
            "       - Programme Electives (PE): EXACTLY 18 CREDITS (6 courses × 3 credits each).\n"
            "       - Open Electives (OE): EXACTLY 12 CREDITS (4 courses × 3 credits each; replaced by 18 minor credits for B.Tech with Minor route).\n"
            "       - Combined Electives Total: 30 CREDITS.\n"
            "       - Total Graded: 141 Credits (CSE) / 142 Credits (AIDE) | Compulsory Audit: 12 Credits | Grand Total Degree Requirement: 153 CREDITS (CSE) / 154 CREDITS (AIDE) (and 159/160 with Minor).\n"
            "   Always structure your answer clearly with separate sections or a comparison table for 'Earlier Scheme (2019–2024 Batches: 165 Credits)' and 'Revised Scheme (2025 Onwards: 153/154 Credits)' so that any student reading the answer immediately finds the exact numbers for their enrollment cohort.\n\n"
            "   Specific Programme Records:\n"
            "   A. B.Tech. in Computer Science & Engineering (CSE):\n"
            "      * Revised UG Curriculum (153 Credits Framework, Table 1 on Page 4 & Table 2 on Page 5 of 05_Revised-UG-Curriculum-CSE.pdf.pdf):\n"
            "        - Programme Electives (PE): EXACTLY 18 CREDITS (6 courses × 3 credits each: 2 courses in Sem VII, 4 courses in Sem VIII or distribution per syllabus).\n"
            "        - Open Electives (OE): EXACTLY 12 CREDITS (4 courses × 3 credits each in Sem VII & VIII; replaced by 18 minor credits for B.Tech with Minor route).\n"
            "        - Programme Compulsory (PC): 66 Credits | B.Tech Project (PP): 6 Credits | Programme Core Total: 90 Credits.\n"
            "        - Institute Core: 39 Credits (IE 16, IS 17, IH 6).\n"
            "        - Total Graded Credits: 141 Credits | Compulsory Audit: 12 Credits (Humanities 3, Engineering 3, Industry Internship 4, Design 2).\n"
            "        - Total Degree Requirement: EXACTLY 153 CREDITS (and 159 Credits for B.Tech with Minor).\n"
            "        - DO NOT state that PE credits are not explicitly stated in the CSE curriculum; they are explicitly 18 credits in Table 1 & Table 2.\n"
            "      * Earlier 2019–2024 Curriculum Scheme (165 Credits Framework, 3 B.Tech CSE Program structure_07102020.pdf.pdf & Academic Regulations 2019):\n"
            "        - Programme Electives (PE): EXACTLY 17 CREDITS.\n"
            "        - Open Electives (OE): EXACTLY 10 CREDITS.\n"
            "        - Combined Electives Total: 27 CREDITS (scheduled across elective slots: Sem VI: 6 credits, Sem VII: 6 credits, Sem VIII: 15 credits).\n"
            "        - Total Graded: 150 Credits | Non-Graded / Audit: 15 Credits (9 non-graded + 6 design credits) | Grand Total Degree Requirement: 165 CREDITS.\n"
            "   B. B.Tech. in Artificial Intelligence & Data Science (AIDE):\n"
            "      * Batches Enrolled 2019 to 2024 (165 Credits Framework, Table 2 of BTech_AI_and_Data_Science_Curriculum_2019_2025.pdf, Page 6):\n"
            "        - Programme Electives (PE): EXACTLY 17 CREDITS (or 18 credits for Double B.Tech).\n"
            "        - Open Electives (OE): EXACTLY 10 CREDITS (0 credits for Double B.Tech).\n"
            "        - Total Graded: 150 Credits | Non-Graded Audit: 15 Credits | Total Degree Requirement: 165 CREDITS.\n"
            "      * New Revised UG Curriculum / Batches Enrolled 2025 Onwards (154 Credits Framework, Table 1 & Table 7 of BTech_AI_and_Data_Science_Curriculum_2025_Onwards.pdf, Pages 4 & 10):\n"
            "        - Programme Electives (PE): EXACTLY 18 CREDITS (6 courses × 3 credits each: Sem V 1, Sem VI 1, Sem VII 2, Sem VIII 2).\n"
            "        - Open Electives (OE): EXACTLY 12 CREDITS (4 courses × 3 credits each: Sem VII 2, Sem VIII 2; replaced by 18 minor credits for B.Tech with Minor route).\n"
            "        - Total Graded: 142 Credits | Compulsory Audit: 12 Credits | Total Degree Requirement: 154 CREDITS.\n"
            "   C. B.Tech. in Mechanical Engineering (ME):\n"
            "      * Earlier 2019–2024 Curriculum Scheme (165 Credits Framework, 1 B.Tech ME Program structure_07102020.pdf.pdf, Pages 2 & 3):\n"
            "        - Programme Electives (PE): EXACTLY 16 CREDITS (Sem VI PE-1: 3 credits, Sem VII PE: 6 credits, Sem VIII PE: 7 credits -> 3 + 6 + 7 = 16 credits). NOTE: For Mechanical Engineering it is exactly 16 credits, NOT 17 credits!\n"
            "        - Open Electives (OE): EXACTLY 10 CREDITS (Sem VII OE-1: 3 credits, Sem VIII OE-2: 4 credits, Sem VIII OE-3: 3 credits -> 3 + 4 + 3 = 10 credits).\n"
            "        - Combined Electives Total: 26 CREDITS (16 PE + 10 OE).\n"
            "        - B.Tech. Project (PP): 3 Credits (Sem VII: 3 credits).\n"
            "        - Total Graded: 150 Credits | Non-Graded Design & Audit: 15 Credits | Grand Total Degree Requirement: 165 CREDITS.\n"
            "      * Revised UG Curriculum Scheme (153 Credits Framework, 08_Concept_Note_BTech_ME.pdf.pdf, Table 1 on Page 3 & Pages 4–5):\n"
            "        - Programme Electives (PE): EXACTLY 18 CREDITS (6 courses × 3 credits each: Sem V 3, Sem VI 3, Sem VII 6, Sem VIII 6).\n"
            "        - Open Electives (OE): EXACTLY 12 CREDITS (4 courses × 3 credits each: Sem VII 6, Sem VIII 6; replaced by 18 minor credits for B.Tech with Minor route).\n"
            "        - Programme Compulsory (PC): 66 Credits | B.Tech Project (PP): 6 Credits | Programme Core Total: 90 Credits.\n"
            "        - Institute Core: 39 Credits (IE 16, IS 17, IH 6).\n"
            "        - Total Graded Credits: 141 Credits | Essential Audit: 12 Credits (NH 3, NE 3, Internship 4, Design 2).\n"
            "        - Grand Total Degree Requirement: EXACTLY 153 CREDITS.\n"
            "   - Under no circumstances assume or state general unverified guesses (e.g. 12 PE / 6 OE) for B.Tech AIDE, CSE, or ME.\n"
            "4. Campus Health & Medical Facilities: IIT Jodhpur has an on-campus clinical medical facility: the Primary Health Centre (PHC), operational since 15 June 2017, managed by M/s Goyal Hospital & Research Centre Private Limited. It provides round-the-clock (24x7) emergency and routine OPD medical care, pharmacy, ambulance, pathology laboratory, and cashless facilities at empaneled hospitals (including Goyal Hospital, AIIMS Jodhpur). DO NOT confuse this clinical medical facility with the 'Centre for Digital Health' (CDH), which is an academic research unit. When asked generally about a health centre, medical clinic, hospital, doctor, or treatment on campus, always explain the Primary Health Centre (PHC) first and foremost.\n"
            "5. Institutional Leadership & Department Heads: For queries asking about the Head of Department (HOD), Dean, Director, Registrar, or department leadership, consult the official Senate Constitution, Senate Minutes, and Key Functionaries records. Specifically, from the official Constitution of the Senate (2024-09-02-66d57f564e853-1725267798.pdf.pdf, Page 3):\n"
            "   - Department of Computer Science & Engineering (CSE): Head of Department (HOD) is Prof. Gaurav Harit.\n"
            "   - Department of Mechanical Engineering (ME): Head of Department (HOD) is Prof. Suril V. Shah.\n"
            "   - Department of Electrical Engineering (EE): Head of Department (HOD) is Prof. Bharat Singh Rajpurohit.\n"
            "   - Department of Bioscience and Bioengineering (BSBE): Head of Department is Prof. Amit Mishra.\n"
            "   - Department of Chemical Engineering: Head of Department is Prof. Parag Arvind Deshpande.\n"
            "   - Department of Physics: Head of Department is Prof. Ambesh Dixit.\n"
            "   - School of Artificial Intelligence & Data Science (AIDE): Head is Prof. Anil Kumar Tiwari.\n"
            "   - Department of Civil & Infrastructure Engineering: Acting Head is Dr. Deepika Bhattu.\n"
            "   - Department of Chemistry: Head is Dr. Sandip Murarka.\n"
            "   - Department of Mathematics: Head is Dr. V.V. M.S. Chandramouli.\n"
            "   - Director: Prof. Avinash Kumar Agarwal.\n"
            "   - Dean of Academic Affairs (DOAA): Prof. Atul Kumar.\n"
            "6. Faculty & NIRF Data Interpretation: In official NIRF faculty tables (such as NIRF 2025 Engineering/Overall Data), the number in the column following 'Ph.D' represents 'Experience in Months' (for example, 160 means 160 months / ~13.3 years of experience), NOT publication count. Additionally, NIRF tables aggregate all engineering faculty into a single list without individual department columns; synthesize details from departmental portals, project records, and research highlights to specify their department (e.g. Dr. Sumit Kalra is in the Department of Computer Science & Engineering (CSE)), research areas (Smart Healthcare, Telemedicine, Software Engineering, IoT), and lab affiliations.\n"
            "7. Strict Privacy & Confidentiality Directive (Personal Financial & Salary Records): Under NO circumstances should you disclose, quote, calculate, or discuss the personal salary, compensation, earnings, bank details, payslips, or immovable property returns of any specific professor, faculty member, officer, or employee. If asked about an individual's financial records, strictly decline and state that personal financial records are confidential under institutional privacy regulations.\n"
            "8. Attendance Policy & Amendments (Old Pre-2024 Rule vs. 2024 Revised Rule - Circular No. 51):\n"
            "   - DO NOT invert or confuse the old and new rules. The 2024 amendment ELIMINATED the grading penalty, it did not create it.\n"
            "   - Old / Pre-2024 Policy (Academic Regulations 2019, Section 4.1.9): Students with attendance below 75% WERE ALLOWED to appear for the end-semester examinations, but were subjected to an elevated academic grading penalty: they were required to score at least a 'C' grade to pass the course. If their score was below the 'C' grade cutoff, they were awarded an 'F' grade (fail).\n"
            "   - New Revised Policy (2024 Amendment, Senate Resolution 38.2.7 / Circular No. IITJ/OAA/Circular/2024-25/51/682 dated 08 July 2024): Examination performance and grading are NO LONGER LINKED to attendance (the 'C' grade passing penalty is completely abolished). Instead, attendance is enforced administratively through de-registration:\n"
            "     * Absence > 2 weeks without sanctioned leave: Course instructor may recommend course de-registration to SUGC/SPGC.\n"
            "     * Absence >= 4 weeks from all academic activities (contiguous or non-contiguous, with or without leave): Automatic semester withdrawal from all courses.\n"
            "     * Overall attendance < 75% without valid reason: Course instructor may recommend de-registration from the course.\n"
            "     * Weekly attendance records must be published for students.\n"
            "9. Branch Change Policy & Eligibility Criteria (Academic Regulations, Section 16 & Registration & Academic Session):\n"
            "   - Timing & Scope: Branch change to another branch or Engineering Science (ES) is permitted ONLY after completion of the 1st year (Semester II). No branch change is permitted post-second year onwards. (Applicable to UG students admitted in AY 2025-26).\n"
            "   - Prerequisite Eligibility Requirements:\n"
            "     * E1: Student must have completed all prescribed course credits for the previous semesters (including non-graded courses).\n"
            "     * E2: Student must have NO backlog.\n"
            "     * E3: No disciplinary action has been imposed on the student.\n"
            "   - CGPA Cutoffs & Department Strength Limits:\n"
            "     * Any Branch to Any Other Branch: Minimum CGPA >= 8.5 for General/OBC/others; Minimum CGPA >= 7.5 for SC, ST, PwD. Lower limit of branch strength: 90% of current strength; Upper limit: 110% of sanctioned strength.\n"
            "     * Any Branch to Engineering Science (ES): Minimum CGPA >= 8.0 for General/others; Minimum CGPA >= 7.0 for SC, ST, PwD. Lower limit: 90% of current strength; Upper limit: 30 seats.\n"
            "10. Departmental Specialization & Minor Credit Requirements (Academic Regulations, Section 11.2 & Office of Academics Affairs):\n"
            "   - Total Credits for Minor / Specialization: EXACTLY 20 CREDITS in the chosen specialization or minor area.\n"
            "   - Additional Credits Beyond Regular Degree: EXACTLY 10 ADDITIONAL CREDITS in the 165-credit scheme (10 credits out of 20 are satisfied via Open Electives, and the remaining 10 credits are taken as an overload beyond the regular B.Tech degree requirement).\n"
            "   - Overload Rule: Starting from the 6th semester, students who have earned at least 80 credits with CGPA >= 6.0 are permitted to register for up to 4 extra credits per semester as an overload for their minor/specialization.\n"
            "   - In the revised 2025 framework: Regular B.Tech is 153 credits (CSE/ME) / 154 credits (AIDE), and B.Tech with Minor is 159 credits (CSE) / 160 credits (AIDE), where 18 minor credits replace 12 OE credits (6 net additional credits beyond regular degree).\n\n"
            "Guidelines:\n"
            "1. Be direct, authoritative, structured, and helpful.\n"
            "2. When explaining policies, circulars, or regulations (e.g., attendance policy, grading, examinations, curriculum), state the key criteria clearly using bullet points and tables.\n"
            "3. Explicitly cite the document name and page number or section URL (e.g., '[BTech_AI_and_Data_Science_Curriculum_2025_Onwards.pdf, Page 4]').\n"
            "4. If answering about an amendment (such as attendance or examination policy), clearly present the latest revised rules approved by the Senate/Board.\n"
            "5. If the provided excerpts genuinely do not have sufficient information to answer the question, state: 'According to the official IIT Jodhpur documents currently loaded, this specific detail is not specified.' Do not speculate."
        )

        user_prompt = (
            f"Official IIT Jodhpur Context Excerpts:\n{context_text}\n\n"
            f"User Question: {query}\n\n"
            "Please provide a comprehensive, accurate answer citing official sources:"
        )

        try:
            if api_provider == "Google Gemini":
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                selected_model = model_name or "gemini-1.5-flash"
                model = genai.GenerativeModel(selected_model, system_instruction=system_instruction)
                response = model.generate_content(user_prompt)
                answer_text = response.text

            elif api_provider == "Groq":
                from groq import Groq
                client = Groq(api_key=api_key, timeout=30.0)
                selected_model = model_name or "openai/gpt-oss-120b"
                
                messages = [
                    {"role": "system", "content": system_instruction},
                ]
                if chat_history:
                    for msg in chat_history[-3:]:
                        content = msg["content"]
                        # Truncate assistant turns to prevent token explosion
                        if msg.get("role") == "assistant" and len(content) > 250:
                            content = content[:250] + "..."
                        messages.append({"role": msg["role"], "content": content})
                        
                messages.append({"role": "user", "content": user_prompt})
                
                try:
                    completion = client.chat.completions.create(
                        model=selected_model,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=2500
                    )
                    answer_text = completion.choices[0].message.content
                except Exception as groq_err:
                    err_msg = str(groq_err)
                    # Automatically handle 413 or rate_limit by falling back to high-capacity model with minimal prompt
                    if "413" in err_msg or "rate_limit" in err_msg.lower():
                        fallback_model = "openai/gpt-oss-120b" if selected_model != "openai/gpt-oss-120b" else "openai/gpt-oss-20b"
                        fallback_messages = [
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": user_prompt}
                        ]
                        completion = client.chat.completions.create(
                            model=fallback_model,
                            messages=fallback_messages,
                            temperature=0.2,
                            max_tokens=2500
                        )
                        answer_text = completion.choices[0].message.content
                    else:
                        raise groq_err
            else:
                answer_text = f"Unsupported provider: {api_provider}"

            answer_text = clean_html_tags(answer_text)
            return {
                "answer": answer_text,
                "sources": [item["chunk"] for item in retrieved]
            }

        except Exception as e:
            return {
                "answer": f"Error calling {api_provider}: {str(e)}\n\nPlease ensure your API key is correct.",
                "sources": [item["chunk"] for item in retrieved]
            }

rag_engine = IITJRagEngine()
