"""
source_authority.py — Authority Hierarchy and Policy Version Resolution Engine
==============================================================================
Implements V4.2 Requirement 3 (Source Authority) and Requirement 4 (Policy Relationships).
Distinguishes formal statutory regulations, approved amendments, recruitment advertisements,
curriculum specifications, institutional portals, and general informational brochures.
"""

from enum import IntEnum
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import re


class SourceAuthorityLevel(IntEnum):
    """
    Source Authority Hierarchy (higher number = higher legal/binding authority).
    """
    APPROVED_AMENDMENT = 6       # Senate-approved notifications/circulars overriding base rules
    STATUTORY_REGULATION = 5     # Formal Senate-approved Academic Regulations (UG/PG)
    OFFICIAL_ADVERTISEMENT = 4   # Formal Statutory Recruitment Advertisements (e.g. Adv-1)
    CURRICULUM_SPECIFICATION = 3 # Formal Departmental Curriculum Booklets
    INSTITUTIONAL_PORTAL = 3     # Official IIT Jodhpur Webpages (OAA, ERP, Portal)
    GUIDEBOOK_BROCHURE = 2       # General Informational Handbooks/Brochures (advisory only)
    UNOFFICIAL_SECONDARY = 1     # Unverified secondary notes or tertiary snippets


class PolicyRelationType:
    AMENDS = "AMENDS"
    REPLACES = "REPLACES"
    SUPERSEDES = "SUPERSEDES"
    APPLIES_TO = "APPLIES_TO"
    EXCEPTS_FROM = "EXCEPTS_FROM"
    REFERENCES = "REFERENCES"
    DERIVED_FROM = "DERIVED_FROM"


@dataclass
class DocumentMetadata:
    doc_name: str
    authority_level: SourceAuthorityLevel
    authority_title: str
    effective_date: str
    binding_status: str  # "BINDING_STATUTORY", "BINDING_RECRUITMENT", "ACADEMIC_CURRICULUM", "GENERAL_GUIDANCE"
    amendment_target: Optional[str] = None
    notes: str = ""


# Authority classification registry
KNOWN_DOCUMENT_AUTHORITY: Dict[str, DocumentMetadata] = {
    # 1. Circulars & Amendments (Level 6)
    "Circular-51_Notification_Deregistration_Attendance-07012022.pdf.pdf": DocumentMetadata(
        doc_name="Circular-51_Notification_Deregistration_Attendance-07012022.pdf.pdf",
        authority_level=SourceAuthorityLevel.APPROVED_AMENDMENT,
        authority_title="Senate Circular 51 (Attendance & De-registration Amendment)",
        effective_date="2022-01-07",
        binding_status="BINDING_STATUTORY",
        amendment_target="Academic Regulations Section 4.1.9",
        notes="Mandates 75% attendance cutoff; supersedes older discretionary attendance provisions."
    ),

    # 2. Statutory Academic Regulations (Level 5)
    "1_Academic_Regulations_Final_03_09_2019.pdf.pdf": DocumentMetadata(
        doc_name="1_Academic_Regulations_Final_03_09_2019.pdf.pdf",
        authority_level=SourceAuthorityLevel.STATUTORY_REGULATION,
        authority_title="UG Academic Regulations (2019 Onwards)",
        effective_date="2019-09-03",
        binding_status="BINDING_STATUTORY",
        notes="Defines 6-semester minimum residence (Sec 17.1), 14-semester maximum duration (Sec 11.4), Branch Change (Sec 16), Specialization/Minor (Sec 11.2)."
    ),
    "BTech_Old_REV_09092014.pdf.pdf": DocumentMetadata(
        doc_name="BTech_Old_REV_09092014.pdf.pdf",
        authority_level=SourceAuthorityLevel.STATUTORY_REGULATION,
        authority_title="UG Academic Regulations (Earlier / 2014 Scheme)",
        effective_date="2014-09-09",
        binding_status="BINDING_STATUTORY",
        notes="Defines earlier graduation criteria: 5.5 CGPA (Sec 10.2) and 16-semester maximum duration (Sec 10.1)."
    ),
    "4_Regulation_PG_2022-onwards_20022023.pdf.pdf": DocumentMetadata(
        doc_name="4_Regulation_PG_2022-onwards_20022023.pdf.pdf",
        authority_level=SourceAuthorityLevel.STATUTORY_REGULATION,
        authority_title="Postgraduate Academic Regulations (2022 Onwards)",
        effective_date="2023-02-20",
        binding_status="BINDING_STATUTORY",
        notes="Defines Table 12.6 Ph.D. coursework/thesis credit requirements and Table 12.8 / 22.8 assistantship continuation rules."
    ),
    "For_students_who_joined_in_July_2014_and_later_MTech_New_REVISED_13Feb2017.pdf.pdf": DocumentMetadata(
        doc_name="For_students_who_joined_in_July_2014_and_later_MTech_New_REVISED_13Feb2017.pdf.pdf",
        authority_level=SourceAuthorityLevel.STATUTORY_REGULATION,
        authority_title="M.Tech Academic Regulations (2014–2017 Scheme)",
        effective_date="2017-02-13",
        binding_status="BINDING_STATUTORY",
        notes="Defines grading turnaround deadlines (72 hours after exams)."
    ),

    # 3. Formal Statutory Advertisements (Level 4 - Binding for Recruitment)
    "Adv-1-638701145587801456.pdf.pdf": DocumentMetadata(
        doc_name="Adv-1-638701145587801456.pdf.pdf",
        authority_level=SourceAuthorityLevel.OFFICIAL_ADVERTISEMENT,
        authority_title="Faculty Recruitment Rolling Advertisement (Adv-1/2025)",
        effective_date="2025-03-21",
        binding_status="BINDING_RECRUITMENT",
        notes="Statutory recruitment advertisement defining relocation reimbursements (SR-116/TTA for Govt domestic, Rs 1.5L for other domestic, Rs 2.5L for abroad) and CPDA (Rs 3 Lakhs per 3-year block)."
    ),

    # 4. Curriculum Booklets (Level 3)
    "06_Revised-UG-Curriculum-AIDE.pdf.pdf": DocumentMetadata(
        doc_name="06_Revised-UG-Curriculum-AIDE.pdf.pdf",
        authority_level=SourceAuthorityLevel.CURRICULUM_SPECIFICATION,
        authority_title="Revised UG Curriculum — AI & Data Science (2025+)",
        effective_date="2025-01-01",
        binding_status="ACADEMIC_CURRICULUM",
        notes="Outlines 4-year (8-semester) normal curriculum layout and credit breakdowns."
    ),

    # 5. Informational Brochures & Handbooks (Level 2 - General Advisory Guidance)
    "Brochure-1-638785941111940671.pdf.pdf": DocumentMetadata(
        doc_name="Brochure-1-638785941111940671.pdf.pdf",
        authority_level=SourceAuthorityLevel.GUIDEBOOK_BROCHURE,
        authority_title="Faculty Information Brochure & FAQ",
        effective_date="2021-04-01",
        binding_status="GENERAL_GUIDANCE",
        notes="General informational guidance brochure; explicitly notes it is advisory and not a formal rulebook."
    ),
    "Information-manual-handbook-for-faculty-members.pdf.pdf": DocumentMetadata(
        doc_name="Information-manual-handbook-for-faculty-members.pdf.pdf",
        authority_level=SourceAuthorityLevel.GUIDEBOOK_BROCHURE,
        authority_title="Information Manual Handbook for Faculty Members",
        effective_date="2020-01-01",
        binding_status="GENERAL_GUIDANCE",
        notes="Advisory handbook for faculty onboarding."
    ),
}


def classify_document_authority(doc_name: str) -> DocumentMetadata:
    """
    Classifies a document's legal authority, effective date, and binding hierarchy.
    """
    clean_name = doc_name.strip()
    if clean_name in KNOWN_DOCUMENT_AUTHORITY:
        return KNOWN_DOCUMENT_AUTHORITY[clean_name]

    # Heuristic classification for any unmapped documents
    lower = clean_name.lower()
    if "circular" in lower or "notification" in lower or "amendment" in lower:
        return DocumentMetadata(
            doc_name=clean_name,
            authority_level=SourceAuthorityLevel.APPROVED_AMENDMENT,
            authority_title="Official Institute Notification / Circular",
            effective_date="Recent",
            binding_status="BINDING_STATUTORY"
        )
    if "regulation" in lower or "regulations" in lower:
        return DocumentMetadata(
            doc_name=clean_name,
            authority_level=SourceAuthorityLevel.STATUTORY_REGULATION,
            authority_title="Statutory Academic Regulations",
            effective_date="Statutory",
            binding_status="BINDING_STATUTORY"
        )
    if "adv-" in lower or "advertisement" in lower or "recruitment" in lower:
        return DocumentMetadata(
            doc_name=clean_name,
            authority_level=SourceAuthorityLevel.OFFICIAL_ADVERTISEMENT,
            authority_title="Statutory Recruitment Notification",
            effective_date="Statutory",
            binding_status="BINDING_RECRUITMENT"
        )
    if "curriculum" in lower or "booklet" in lower or "syllabus" in lower:
        return DocumentMetadata(
            doc_name=clean_name,
            authority_level=SourceAuthorityLevel.CURRICULUM_SPECIFICATION,
            authority_title="Curriculum Specification",
            effective_date="Curricular",
            binding_status="ACADEMIC_CURRICULUM"
        )
    if "brochure" in lower or "guidebook" in lower or "handbook" in lower:
        return DocumentMetadata(
            doc_name=clean_name,
            authority_level=SourceAuthorityLevel.GUIDEBOOK_BROCHURE,
            authority_title="Informational Guidebook / Brochure",
            effective_date="Advisory",
            binding_status="GENERAL_GUIDANCE"
        )

    return DocumentMetadata(
        doc_name=clean_name,
        authority_level=SourceAuthorityLevel.INSTITUTIONAL_PORTAL,
        authority_title="Institutional Document",
        effective_date="General",
        binding_status="GENERAL_STATUTORY"
    )


def compare_source_precedence(doc_a: str, doc_b: str) -> int:
    """
    Compares two sources by authority and recency.
    Returns > 0 if doc_a has higher precedence than doc_b, < 0 if doc_b has higher, 0 if equal.
    """
    meta_a = classify_document_authority(doc_a)
    meta_b = classify_document_authority(doc_b)

    if meta_a.authority_level != meta_b.authority_level:
        return int(meta_a.authority_level) - int(meta_b.authority_level)

    # If authority is equal, compare effective dates if available
    if meta_a.effective_date and meta_b.effective_date:
        if meta_a.effective_date > meta_b.effective_date:
            return 1
        elif meta_a.effective_date < meta_b.effective_date:
            return -1

    return 0
