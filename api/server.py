"""
api/server.py
=============
Headless FastAPI backend service for IIT Jodhpur Academic RAG system.
Exposes RESTful endpoints for query processing, document provenance,
atomic claim verification, and execution tracing.
"""

import os
import time
import uuid
import logging
from typing import List, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from dotenv import load_dotenv

load_dotenv()

from database.connection import get_db, SessionLocal
from database.models import Document, AtomicClaim, QueryExecutionTraceRecord
from storage.qdrant_manager import qdrant_manager
from retrieval_engine_v3 import retrieval_engine_v3
from query_understanding import analyze_query, QueryDomain, QueryIntent
from query_planning import create_query_plan
from evidence_grounded_generator import generate_grounded_answer, synthesize_direct_grounded_fallback
from api.schemas import (
    QueryRequest,
    QueryResponse,
    SourceCitation,
    AtomicClaimOut,
    DocumentOut,
    DocumentListResponse,
    TraceOut,
)

logger = logging.getLogger("AcademicAPI")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="IIT Jodhpur Academic Intelligence API",
    description="Statutory Academic Regulations & Curriculum Intelligence Service",
    version="3.2.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", summary="Health Check")
def health_check(db: Session = Depends(get_db)):
    """Reports system health, database counts, and vector store indexing status."""
    try:
        doc_count = db.query(Document).count()
        claim_count = db.query(AtomicClaim).count()
    except Exception as e:
        doc_count, claim_count = -1, -1

    try:
        qdrant_count = qdrant_manager.count()
    except Exception:
        qdrant_count = -1

    return {
        "status": "healthy",
        "version": "3.2.0",
        "documents_indexed": doc_count,
        "verified_atomic_claims": claim_count,
        "qdrant_vector_points": qdrant_count,
        "active_llm_provider": "Groq" if os.getenv("GROQ_API_KEY") else "Deterministic Grounded Synthesizer"
    }


@app.post("/api/v1/query", response_model=QueryResponse, summary="Execute Grounded Academic Query")
def execute_query(req: QueryRequest, db: Session = Depends(get_db)):
    """
    Executes a grounded query against statutory regulations and academic curricula:
    1. Dynamic Query Understanding & Domain Intent Classification.
    2. Atomic Claim & Provenance Retrieval from Relational Store.
    3. Multi-Channel Vector & Lexical Retrieval with Neural Reranking.
    4. Scope & Topic Validation.
    5. Zero-speculation answer synthesis with full provenance traceability.
    """
    start_time = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]

    # 1. Query Understanding & Structured Attribute Plan
    profile = analyze_query(req.query, request_id=request_id)
    plan = create_query_plan(req.query, profile)

    # 2. Relational Atomic Claims Lookup
    matched_db_claims = []
    try:
        import re
        stopwords = {"what", "is", "the", "for", "a", "an", "and", "or", "to", "in", "of", "from", "at", "by", "with", "how", "many", "does", "are"}
        query_words = [w for w in re.findall(r"\b[a-zA-Z0-9]+\b", profile.normalized_query) if len(w) > 2 and w not in stopwords]
        all_db_claims = db.query(AtomicClaim).all()
        scored_claims = []
        for cl in all_db_claims:
            comb = f"{cl.entity} {cl.attribute} {cl.raw_excerpt}".lower()
            score = sum(1 for w in query_words if w in comb)
            for attr in profile.target_attributes:
                for part in attr.split("_"):
                    if len(part) > 2 and part in comb:
                        score += 3
            if score >= 2:
                scored_claims.append((score, cl))
        scored_claims.sort(key=lambda x: x[0], reverse=True)
        matched_db_claims = [sc[1] for sc in scored_claims[:8]]
    except Exception as e:
        logger.warning(f"Error querying relational atomic claims: {e}")

    # 3. Candidate Retrieval & Evidence Assessment
    top_k = min(req.max_candidates, 16)
    candidates, assessment = retrieval_engine_v3.retrieve_and_assess(req.query, top_k=top_k)

    # Inject matched relational atomic claims into assessment context
    from evidence_models import AtomicClaim as EvClaim, ApplicabilityScope as EvScope, Provenance as EvProv
    for db_cl in reversed(matched_db_claims):
        doc_filename = db_cl.document.filename if db_cl.document else "Official Academic Regulations"
        ev_claim = EvClaim(
            entity=db_cl.entity,
            attribute=db_cl.attribute,
            value=db_cl.value,
            unit=db_cl.unit or "",
            scope=EvScope(
                programme="All Programmes" if not db_cl.scope_cohort else db_cl.scope_cohort,
                cohort=db_cl.scope_cohort or "General",
                semester=db_cl.scope_semester,
                effective_date="Current",
                degree_level="Statutory Regulations"
            ),
            provenance=EvProv(
                doc_id=db_cl.document_id,
                doc_name=doc_filename,
                page_number=str(db_cl.page_number),
                section_heading=db_cl.section.heading_title if db_cl.section else ""
            ),
            confidence=db_cl.confidence_score or 0.98
        )
        if not any(c.attribute == ev_claim.attribute and c.entity == ev_claim.entity for c in assessment.claims):
            assessment.claims.insert(0, ev_claim)

    # 4. Answer Generation (using Groq LLM if configured, else Deterministic Grounded Synthesis)
    force_det = os.getenv("FORCE_DETERMINISTIC_EVAL") == "1"
    api_key = "" if force_det else os.getenv("GROQ_API_KEY", "")
    api_provider = "Groq" if api_key else "Deterministic"

    if api_key:
        gen_result = generate_grounded_answer(
            query=req.query,
            candidates=candidates,
            assessment=assessment,
            api_provider=api_provider,
            api_key=api_key,
            model_name="qwen/qwen3.8-27b",
            request_id=request_id,
            profile=profile,
            plan=plan
        )
        answer_text = gen_result.get("answer", "")
        raw_sources = gen_result.get("sources", [])
    else:
        answer_text = synthesize_direct_grounded_fallback(profile, assessment, candidates)
        raw_sources = []
        for c in candidates[:top_k]:
            raw_sources.append({
                "source": c.chunk.get("source", ""),
                "page": c.chunk.get("page", 1),
                "text": c.chunk.get("text", "")[:300],
                "score": c.scores.cross_encoder_score or c.scores.rrf_score or 0.0
            })

    # 5. Format Output Citations
    citations: List[SourceCitation] = []
    for s in raw_sources:
        citations.append(SourceCitation(
            doc_name=str(s.get("source", "Official Document")),
            page=s.get("page", 1),
            section=s.get("section"),
            similarity_score=float(s.get("score", 0.0)),
            excerpt=str(s.get("text", ""))[:400]
        ))

    # Format Output Claims
    claims_out: List[AtomicClaimOut] = []
    for c in matched_db_claims:
        claims_out.append(AtomicClaimOut(
            id=c.id,
            entity=c.entity,
            attribute=c.attribute,
            value=c.value,
            unit=c.unit,
            scope_cohort=c.scope_cohort,
            scope_semester=c.scope_semester,
            page_number=c.page_number,
            raw_excerpt=c.raw_excerpt,
            confidence_score=c.confidence_score
        ))

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # 6. Audit Logging into Relational Database
    try:
        trace_record = QueryExecutionTraceRecord(
            request_id=request_id,
            original_query=req.query,
            normalized_query=profile.normalized_query,
            domain=profile.domain.value,
            intent=profile.intent.value,
            subject_entity=profile.subject_entity,
            retrieved_chunk_count=len(candidates),
            evidence_state=assessment.state.value if assessment else "UNKNOWN",
            evidence_confidence=float(getattr(assessment, "overall_confidence", 0.95)),
            llm_response=answer_text[:1000],
            latency_breakdown_ms={"total": elapsed_ms}
        )
        db.add(trace_record)
        db.commit()
    except Exception as e:
        logger.warning(f"Failed to record trace in database: {e}")
        db.rollback()

    return QueryResponse(
        request_id=request_id,
        query=req.query,
        domain=profile.domain.value,
        intent=profile.intent.value,
        evidence_status=assessment.state.value if assessment else "SUFFICIENT",
        answer=answer_text,
        citations=citations,
        atomic_claims=claims_out,
        processing_time_ms=round(elapsed_ms, 2),
        disclaimer="Information synthesized strictly from indexed official IIT Jodhpur academic records."
    )


@app.get("/api/v1/documents", response_model=DocumentListResponse, summary="List Authoritative Documents")
def list_documents(
    category: Optional[str] = Query(None, description="Filter by category"),
    programme: Optional[str] = Query(None, description="Filter by programme"),
    cohort: Optional[str] = Query(None, description="Filter by cohort"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """Returns paginated authoritative document inventory with metadata."""
    q = db.query(Document)
    if category:
        q = q.filter(Document.category.ilike(f"%{category}%"))
    if programme:
        q = q.filter(Document.programme.ilike(f"%{programme}%"))
    if cohort:
        q = q.filter(Document.cohort.ilike(f"%{cohort}%"))

    total = q.count()
    docs = q.offset(offset).limit(limit).all()

    doc_list = [
        DocumentOut(
            id=d.id,
            filename=d.filename,
            canonical_title=d.canonical_title,
            category=d.category,
            programme=d.programme,
            cohort=d.cohort,
            page_count=d.page_count,
            is_indexed=d.is_indexed
        )
        for d in docs
    ]

    return DocumentListResponse(
        total=total,
        limit=limit,
        offset=offset,
        documents=doc_list
    )


@app.get("/api/v1/claims", response_model=List[AtomicClaimOut], summary="List Verified Atomic Claims")
def list_atomic_claims(
    entity: Optional[str] = Query(None, description="Filter by target entity"),
    cohort: Optional[str] = Query(None, description="Filter by cohort"),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """Returns verified institutional atomic claims with document and page provenance."""
    q = db.query(AtomicClaim)
    if entity:
        q = q.filter(AtomicClaim.entity.ilike(f"%{entity}%"))
    if cohort:
        q = q.filter(AtomicClaim.scope_cohort.ilike(f"%{cohort}%"))

    claims = q.limit(limit).all()
    return [
        AtomicClaimOut(
            id=c.id,
            entity=c.entity,
            attribute=c.attribute,
            value=c.value,
            unit=c.unit,
            scope_cohort=c.scope_cohort,
            scope_semester=c.scope_semester,
            page_number=c.page_number,
            raw_excerpt=c.raw_excerpt,
            confidence_score=c.confidence_score
        )
        for c in claims
    ]


@app.get("/api/v1/traces/{request_id}", response_model=TraceOut, summary="Get Query Execution Trace")
def get_trace(request_id: str, db: Session = Depends(get_db)):
    """Retrieves full audit log and timing metadata for a specific query execution."""
    trace = db.query(QueryExecutionTraceRecord).filter(
        QueryExecutionTraceRecord.request_id == request_id
    ).first()
    if not trace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace with request_id '{request_id}' not found."
        )

    return TraceOut(
        request_id=trace.request_id,
        timestamp=trace.timestamp,
        original_query=trace.original_query,
        normalized_query=trace.normalized_query,
        domain=trace.domain,
        intent=trace.intent,
        subject_entity=trace.subject_entity,
        retrieved_chunk_count=trace.retrieved_chunk_count,
        evidence_state=trace.evidence_state,
        evidence_confidence=trace.evidence_confidence,
        latency_breakdown_ms=trace.latency_breakdown_ms
    )


@app.get("/api/v1/traces", response_model=List[TraceOut], summary="List Recent Execution Traces")
def list_traces(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    """Lists recent execution traces ordered by timestamp descending."""
    traces = db.query(QueryExecutionTraceRecord).order_by(
        QueryExecutionTraceRecord.timestamp.desc()
    ).limit(limit).all()

    return [
        TraceOut(
            request_id=t.request_id,
            timestamp=t.timestamp,
            original_query=t.original_query,
            normalized_query=t.normalized_query,
            domain=t.domain,
            intent=t.intent,
            subject_entity=t.subject_entity,
            retrieved_chunk_count=t.retrieved_chunk_count,
            evidence_state=t.evidence_state,
            evidence_confidence=t.evidence_confidence,
            latency_breakdown_ms=t.latency_breakdown_ms
        )
        for t in traces
    ]
