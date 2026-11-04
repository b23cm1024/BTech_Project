"""
api/schemas.py
==============
Pydantic v2 schemas for API requests, responses, provenance citations,
atomic claims, and query execution trace logs.
"""

from typing import List, Optional, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question regarding academic rules or curricula")
    target_programme: Optional[str] = Field(None, description="Optional target programme (e.g. Mechanical Engineering)")
    target_cohort: Optional[str] = Field(None, description="Target cohort (e.g. 2025_onwards, 2019_2024, General)")
    max_candidates: int = Field(8, description="Maximum number of candidates to retrieve and evaluate")


class SourceCitation(BaseModel):
    doc_name: str = Field(..., description="Source document filename")
    page: Any = Field(..., description="Page number or structured sheet identifier")
    section: Optional[str] = Field(None, description="Section heading path")
    similarity_score: float = Field(..., description="Calculated relevance score")
    excerpt: Optional[str] = Field(None, description="Verbatim source excerpt")


class AtomicClaimOut(BaseModel):
    id: str = Field(..., description="Claim identifier")
    entity: str = Field(..., description="Target entity (e.g. Summer Term, Branch Change)")
    attribute: str = Field(..., description="Attribute name (e.g. Maximum Credits, Minimum CGPA)")
    value: str = Field(..., description="Fact value (e.g. 8 credits, 8.5 CGPA)")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    scope_cohort: Optional[str] = Field(None, description="Cohort scope")
    scope_semester: Optional[int] = Field(None, description="Semester scope")
    page_number: int = Field(..., description="Page number in statutory document")
    raw_excerpt: str = Field(..., description="Authoritative statutory text excerpt")
    confidence_score: float = Field(0.95, description="Extraction confidence score")


class QueryResponse(BaseModel):
    request_id: str = Field(..., description="Unique trace identifier")
    query: str = Field(..., description="Input user query")
    domain: str = Field(..., description="Classified query domain")
    intent: str = Field(..., description="Classified intent")
    evidence_status: str = Field(..., description="Evidence assessment status (SUFFICIENT, INSUFFICIENT, CONFLICT)")
    answer: str = Field(..., description="Synthesized verified answer with markdown tables and citations")
    citations: List[SourceCitation] = Field(default_factory=list, description="Verified source citations")
    atomic_claims: List[AtomicClaimOut] = Field(default_factory=list, description="Verified atomic claims linked to response")
    processing_time_ms: float = Field(..., description="Total execution time in milliseconds")
    disclaimer: Optional[str] = Field(None, description="Authoritative institutional disclaimer")


class DocumentOut(BaseModel):
    id: str
    filename: str
    canonical_title: str
    category: str
    programme: Optional[str] = None
    cohort: Optional[str] = None
    page_count: int
    is_indexed: bool


class DocumentListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    documents: List[DocumentOut]


class TraceOut(BaseModel):
    request_id: str
    timestamp: datetime
    original_query: str
    normalized_query: Optional[str] = None
    domain: str
    intent: str
    subject_entity: Optional[str] = None
    retrieved_chunk_count: int
    evidence_state: Optional[str] = None
    evidence_confidence: Optional[float] = 0.0
    latency_breakdown_ms: Optional[Dict[str, Any]] = None
