"""
evidence_models.py
==================
Structured models for multi-stage evidence recovery, score tracking,
atomic fact extraction, scope-aware claims, provenance verification,
and corpus coverage verification.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union


class EvidenceState(str, Enum):
    # Canonical Requirement 7 States
    FULLY_SUPPORTED = "FULLY_SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    SCOPE_UNRESOLVED = "SCOPE_UNRESOLVED"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    SCOPED_ALTERNATIVES = "SCOPED_ALTERNATIVES"
    INCOMPLETE_CORPUS = "INCOMPLETE_CORPUS"
    EXTRACTION_FAILURE = "EXTRACTION_FAILURE"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    VERIFIED_NO_EVIDENCE = "VERIFIED_NO_EVIDENCE"

    # Backward compatibility aliases
    FULL_SUPPORT = "FULLY_SUPPORTED"
    PARTIAL_SUPPORT = "PARTIALLY_SUPPORTED"
    GENUINE_CONFLICT = "CONFLICTING_EVIDENCE"
    EXTRACTION_UNCERTAINTY = "EXTRACTION_FAILURE"
    NO_EVIDENCE = "VERIFIED_NO_EVIDENCE"


@dataclass
class Provenance:
    doc_id: str = ""
    doc_name: str = ""
    page_number: str = ""
    section_heading: str = ""
    table_id: Optional[str] = None
    row_idx: Optional[int] = None
    col_name: Optional[str] = None
    raw_text: str = ""
    clean_text: str = ""

    def citation(self) -> str:
        loc = f"Page: {self.page_number}" if self.page_number else "Institute Record"
        if self.section_heading:
            loc += f", Section: {self.section_heading}"
        if self.table_id:
            loc += f", Table: {self.table_id}"
        return f"[{self.doc_name} | {loc}]"


@dataclass
class ApplicabilityScope:
    programme: str = "Institute-Wide"
    cohort: str = "General"
    semester: Optional[int] = None
    effective_date: Optional[str] = None
    degree_level: str = "UG"
    measurement_category: str = "General"

    def is_compatible(self, other: 'ApplicabilityScope') -> bool:
        """Checks if two scopes overlap or refer to the exact same temporal/cohort segment."""
        # Check programme overlap
        if self.programme != "Institute-Wide" and other.programme != "Institute-Wide":
            if self.programme.lower() != other.programme.lower():
                return False
        # Check cohort overlap
        if self.cohort != "General" and other.cohort != "General":
            if self.cohort.lower() != other.cohort.lower():
                return False
        # Check semester overlap
        if self.semester is not None and other.semester is not None:
            if self.semester != other.semester:
                return False
        return True


@dataclass
class AtomicClaim:
    entity: str
    attribute: str
    value: str
    unit: str
    scope: ApplicabilityScope
    provenance: Provenance
    confidence: float = 1.0


@dataclass
class ScoreProfile:
    """Maintains independent scores across all retrieval representations."""
    dense_distance: Optional[float] = None
    dense_similarity: Optional[float] = None
    sparse_bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    cross_encoder_score: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dense_distance": round(self.dense_distance, 4) if self.dense_distance is not None else None,
            "dense_similarity": round(self.dense_similarity, 4) if self.dense_similarity is not None else None,
            "sparse_bm25_score": round(self.sparse_bm25_score, 4) if self.sparse_bm25_score is not None else None,
            "rrf_score": round(self.rrf_score, 5) if self.rrf_score is not None else None,
            "cross_encoder_score": round(self.cross_encoder_score, 4) if self.cross_encoder_score is not None else None,
        }


@dataclass
class RetrievalCandidate:
    chunk_id: str
    chunk: Dict[str, Any]
    scores: ScoreProfile
    expanded_context: str
    provenance: Provenance


@dataclass
class CorpusCoverageAssessment:
    """Evaluates corpus discovery, indexing, and completeness independently from retrieval."""
    target_entity: Optional[str] = None
    target_category: Optional[str] = None
    discovered_documents: List[str] = field(default_factory=list)
    searched_documents: List[str] = field(default_factory=list)
    corpus_complete: bool = True
    coverage_condition: str = "FULLY_INDEXED"  # FULLY_INDEXED, DOCUMENT_UNPROCESSED, NO_RELEVANT_DOCUMENT, EXTRACTION_INCOMPLETE
    unindexed_candidates: List[str] = field(default_factory=list)
    retrieval_relevance: float = 0.0
    corpus_coverage_score: float = 0.0
    extraction_reliability: float = 0.0
    evidence_completeness: float = 0.0
    citation_validity: float = 0.0
    failure_diagnosis: Optional[str] = None


@dataclass
class EvidenceAssessment:
    state: EvidenceState
    query: str
    claims: List[AtomicClaim] = field(default_factory=list)
    scoped_groups: Dict[str, List[AtomicClaim]] = field(default_factory=dict)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    supporting_candidates: List[RetrievalCandidate] = field(default_factory=list)
    recovery_stage: int = 1
    overall_confidence: float = 0.0
    reasoning: str = ""
    coverage: Optional[CorpusCoverageAssessment] = None

    def as_dict(self) -> Dict[str, Any]:
        cov_dict = None
        if self.coverage:
            cov_dict = {
                "target_entity": self.coverage.target_entity,
                "coverage_condition": self.coverage.coverage_condition,
                "discovered_documents": self.coverage.discovered_documents,
                "corpus_complete": self.coverage.corpus_complete,
                "coverage_score": round(self.coverage.corpus_coverage_score, 3),
                "failure_diagnosis": self.coverage.failure_diagnosis
            }
        return {
            "state": self.state.value,
            "query": self.query,
            "overall_confidence": round(self.overall_confidence, 3),
            "recovery_stage": self.recovery_stage,
            "claims_count": len(self.claims),
            "scoped_groups": {k: [c.value + " " + c.unit for c in v] for k, v in self.scoped_groups.items()},
            "conflicts_count": len(self.conflicts),
            "candidates_count": len(self.supporting_candidates),
            "reasoning": self.reasoning,
            "coverage": cov_dict
        }
