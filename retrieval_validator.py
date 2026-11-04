"""
retrieval_validator.py — Scope-Aware Evidence Verification Layer (V3 Rebuild)
=============================================================================
Replaces brittle raw-score thresholds with Scope-Aware Evidence Assessment.
Eliminates false refusals, preserves multi-cohort alternatives, and prevents
ungrounded hallucinations.

Maintains backward-compatible public API:
    validate_retrieval(query, results, query_type) -> (ok: bool, report: dict, confidence: float)
    insufficient_evidence_response(report) -> dict
"""

import os
from typing import List, Dict, Tuple, Any
from evidence_models import (
    EvidenceState,
    RetrievalCandidate,
    ScoreProfile,
    Provenance,
    ApplicabilityScope,
)
from evidence_extractor import assess_evidence


def _convert_dict_results_to_candidates(results: List[Dict[str, Any]]) -> List[RetrievalCandidate]:
    """Converts legacy dict results into structured RetrievalCandidate models."""
    candidates = []
    for r in results:
        chunk = r.get("chunk", r)
        score = r.get("score", 0.0)
        raw_score = r.get("raw_score", score)

        prov = Provenance(
            doc_id=chunk.get("doc_id", ""),
            doc_name=chunk.get("source", ""),
            page_number=str(chunk.get("page", "")),
            section_heading=chunk.get("heading_path", ""),
            table_id=chunk.get("table_id"),
            raw_text=chunk.get("text", "")[:400],
            clean_text=chunk.get("text", "")[:400]
        )

        cand = RetrievalCandidate(
            chunk_id=chunk.get("id") or chunk.get("chunk_id", ""),
            chunk=chunk,
            scores=ScoreProfile(
                dense_distance=r.get("dense_distance"),
                sparse_bm25_score=r.get("bm25_score"),
                rrf_score=r.get("rrf_score"),
                cross_encoder_score=score
            ),
            expanded_context=chunk.get("parent_context", {}).get("parent_text", ""),
            provenance=prov
        )
        candidates.append(cand)
    return candidates


def validate_retrieval(
    query: str,
    results: List[Dict],
    query_type: str = None,
) -> Tuple[bool, Dict, float]:
    """
    Scope-aware evidence assessment replacing raw-score rejection gates.

    Returns:
        (ok: bool, report: dict, confidence: float 0-1)
    """
    if not results:
        return False, {"error": "no_results", "state": EvidenceState.NO_EVIDENCE.value}, 0.0

    candidates = _convert_dict_results_to_candidates(results)
    assessment = assess_evidence(query, candidates)

    report = assessment.as_dict()

    # Never refuse if full support or scoped alternatives exist
    if assessment.state in [EvidenceState.FULL_SUPPORT, EvidenceState.SCOPED_ALTERNATIVES]:
        return True, report, assessment.overall_confidence

    # If genuine conflict exists, allow generation so LLM discloses both viewpoints
    if assessment.state == EvidenceState.GENUINE_CONFLICT:
        return True, report, assessment.overall_confidence

    # If partial support exists with substantive text, allow generation with stated limitations
    if assessment.state == EvidenceState.PARTIAL_SUPPORT:
        return True, report, assessment.overall_confidence

    # Only refuse when truly NO_EVIDENCE is present across all candidates
    ok = assessment.overall_confidence >= 0.50
    return ok, report, assessment.overall_confidence


def insufficient_evidence_response(report: Dict) -> Dict:
    """Standard response when retrieval validation genuinely finds no evidence."""
    state = report.get("state", "NO_EVIDENCE")
    reasoning = report.get("reasoning", "No evidence found in loaded documents.")
    return {
        "answer": (
            "According to the official IIT Jodhpur documents currently loaded, "
            "the retrieved excerpts do not contain explicit evidence to answer "
            "this question with certainty.\n\n"
            f"Evidence State: {state}. Context Evaluation: {reasoning}\n\n"
            "Please refer to the official IIT Jodhpur academic documents or contact "
            "the Office of Academic Affairs for authoritative information."
        ),
        "sources": [],
        "_validation_report": report,
    }
