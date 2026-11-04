"""
evidence_pack.py — Strict Immutable Evidence Pack & Provenance Object
====================================================================
Implements Section 8 of the Master Engineering Directive:
  - Assembles structured, immutable evidence items with verifiable identifiers.
  - Retains source authority, applicable cohort, exact page numbers, section paths,
    table row/column labels, and semantic relationship (SUPPORTS, AMENDS, CONTRADICTS,
    HISTORICAL).
  - Ensures every factual assertion can be traced directly to an exact EvidenceID.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any, Set


class EvidenceRelationship(str, Enum):
    SUPPORTS = "SUPPORTS"
    AMENDS = "AMENDS"
    SUPERSEDES = "SUPERSEDES"
    CONTRADICTS = "CONTRADICTS"
    HISTORICAL = "HISTORICAL"
    RELATED_CONTEXT = "RELATED_CONTEXT"


@dataclass(frozen=True)
class StructuredEvidenceItem:
    evidence_id: str
    source_id: str
    document_title: str
    authority_tier: int
    programme: str
    degree_level: str
    department: str
    applicable_cohort: str
    academic_year: str
    regulation_version: str
    page_number: int
    section_number: str
    exact_source_text: str
    evidence_relationship: EvidenceRelationship
    table_id: Optional[str] = None
    row_label: Optional[str] = None
    column_label: Optional[str] = None
    structured_value: Optional[str] = None
    extraction_confidence: float = 1.0

    def to_citation_str(self) -> str:
        loc = f"Page {self.page_number}"
        if self.section_number and self.section_number != "General":
            loc += f", {self.section_number}"
        if self.table_id:
            loc += f", {self.table_id}"
        return f"[{self.document_title}, {loc}]"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_relationship"] = self.evidence_relationship.value
        return d


@dataclass
class EvidencePack:
    request_id: str
    query: str
    items: List[StructuredEvidenceItem] = field(default_factory=list)
    sub_requirement_coverage: Dict[str, List[str]] = field(default_factory=dict) # req_id -> list of evidence_ids

    def add_item(self, item: StructuredEvidenceItem, requirement_id: Optional[str] = None):
        self.items.append(item)
        if requirement_id:
            self.sub_requirement_coverage.setdefault(requirement_id, []).append(item.evidence_id)

    def get_by_id(self, evidence_id: str) -> Optional[StructuredEvidenceItem]:
        for it in self.items:
            if it.evidence_id == evidence_id:
                return it
        return None

    def format_for_prompt(self) -> str:
        """Formats the evidence pack cleanly for LLM input with explicit evidence IDs."""
        lines = []
        for it in self.items:
            lines.append(f"--- Evidence ID: {it.evidence_id} ---")
            lines.append(f"Document: {it.document_title} (Source ID: {it.source_id})")
            lines.append(f"Location: Page {it.page_number} | Section: {it.section_number}")
            if it.table_id:
                lines.append(f"Table: {it.table_id} | Row: {it.row_label or 'N/A'} | Col: {it.column_label or 'N/A'}")
            lines.append(f"Applicability: Dept={it.department}, Prog={it.programme}, Cohort={it.applicable_cohort}, Version={it.regulation_version}")
            lines.append(f"Relationship: {it.evidence_relationship.value}")
            src_text = (it.exact_source_text or "").strip()
            lines.append(f"Exact Statutory Excerpt:\n\"{src_text}\"\n")
        return "\n".join(lines)

    def get_sources_summary(self) -> List[Dict[str, Any]]:
        seen = set()
        summary = []
        for it in self.items:
            key = (it.document_title, it.page_number)
            if key not in seen:
                seen.add(key)
                summary.append({
                    "source": it.document_title,
                    "page": it.page_number,
                    "section": it.section_number,
                    "text": (it.exact_source_text or "")[:350],
                    "evidence_id": it.evidence_id
                })
        return summary
