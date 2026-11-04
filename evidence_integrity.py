"""
evidence_integrity.py
=====================
V4.4 Evidence Integrity and Retrieval Consistency Layer for IIT Jodhpur Document Intelligence.

Responsibilities:
1. Strict Role-to-Entity Alignment (preventing designation confusion in administrative roles).
2. Semantic Query Canonicalization (ensuring paraphrased and equivalent queries retrieve identical evidence).
3. Degree-Level and Scope Isolation (preventing UG/PG/Doctoral context bleeding).
4. Evidence Quarantine & Integrity Filtering (filtering out semantically noisy but irrelevant chunks before LLM context assembly).
5. Attribute-Level Answer Completeness Tracking.
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from source_authority import classify_document_authority, SourceAuthorityLevel, compare_source_precedence
from query_planning import QueryPlan, AttributeEvidenceRequirement


# ── 1. Strict Administrative Role Registry ────────────────────────────────────

@dataclass
class AdministrativeRoleDefinition:
    canonical_title: str
    office_holder: str
    aliases: List[str]
    confusable_roles: List[str]
    primary_source: str


ADMINISTRATIVE_ROLES: Dict[str, AdministrativeRoleDefinition] = {
    "director": AdministrativeRoleDefinition(
        canonical_title="Director, IIT Jodhpur",
        office_holder="Prof. Avinash Kumar Agarwal",
        aliases=["director", "head of institute", "director of iit jodhpur"],
        confusable_roles=["deputy director", "director jckif"],
        primary_source="Board of Governors Notification & Office of the Director"
    ),
    "registrar": AdministrativeRoleDefinition(
        canonical_title="Registrar, IIT Jodhpur",
        office_holder="Dr. Hari Om Yadav (Officiating / Statutory)",
        aliases=["registrar", "institute registrar"],
        confusable_roles=["assistant registrar", "deputy registrar"],
        primary_source="Office of the Registrar & BoG Minutes"
    ),
    "bog_chairman": AdministrativeRoleDefinition(
        canonical_title="Chairman, Board of Governors (BoG)",
        office_holder="Sh. A. S. Kiran Kumar",
        aliases=["chairman bog", "chairman board of governors", "bog chairman", "chairman of the board"],
        confusable_roles=["director", "bog members"],
        primary_source="BoG Resolution No. 37.1 / Ministry of Education"
    ),
    "dean_academics": AdministrativeRoleDefinition(
        canonical_title="Dean of Academic Affairs (DOAA)",
        office_holder="Dean (Academic Affairs)",
        aliases=["dean of academic affairs", "dean academics", "doaa", "dean of academics", "dean academic"],
        confusable_roles=["dean of faculty affairs", "dean of administration", "dean of students", "dean of r&d"],
        primary_source="Office of Academic Affairs Portal & Administration Records"
    ),
    "dean_rnd": AdministrativeRoleDefinition(
        canonical_title="Dean of Research and Development (DORD)",
        office_holder="Dean (Research & Development)",
        aliases=["dean of research and development", "dean rnd", "dean r&d", "dord", "dean research"],
        confusable_roles=["dean of faculty affairs", "dean of administration", "dean academics"],
        primary_source="Office of R&D Portal & Administration Records"
    ),
    "dean_students": AdministrativeRoleDefinition(
        canonical_title="Dean of Students Affairs (DOSA)",
        office_holder="Dean (Students Affairs)",
        aliases=["dean of students affairs", "dean of students", "dosa", "dean student affairs"],
        confusable_roles=["chief warden", "dean of administration", "dean of faculty affairs"],
        primary_source="Office of Students Affairs Portal & Administration Records"
    ),
    "dean_faculty": AdministrativeRoleDefinition(
        canonical_title="Dean of Faculty Affairs (DOFA)",
        office_holder="Prof. Neeraj Jain",
        aliases=["dean of faculty affairs", "dean faculty", "dofa"],
        confusable_roles=["dean of academic affairs", "dean of administration"],
        primary_source="Office of Faculty Affairs Portal"
    ),
    "dean_admin": AdministrativeRoleDefinition(
        canonical_title="Dean of Administration (DOAD)",
        office_holder="Prof. Shree Prakash Tiwari",
        aliases=["dean of administration", "dean administration", "doad"],
        confusable_roles=["dean of faculty affairs", "dean of academic affairs"],
        primary_source="Office of Administration Records & Joint Consultative Committee"
    )
}


# ── 2. Query Paraphrase Canonicalizer ─────────────────────────────────────────

CANONICAL_INTENT_MAP = {
    # Ph.D. Coursework & Thesis credits
    r"\b(?:ph\.?d|phd|doctoral)\b.*\b(?:coursework|thesis|credits?)\b": "phd_coursework_thesis_credits",
    r"\b(?:coursework\s+and\s+thesis|thesis\s+and\s+coursework|direct\s+ph\.?d|direct\s+entry|admitted\s+directly)\b": "phd_coursework_thesis_credits",

    # UG Semester Registration Limits
    r"\b(?:credit\s+registration\s+limit|semester\s+credit\s+limit|normal\s+semester.*(?:undergraduate|ug|btech)|maximum.*credit.*normal\s+semester.*(?:undergraduate|ug|btech)|undergraduate.*register.*normal\s+semester)\b": "ug_semester_registration_limits",

    # M.Tech Registration Limits
    r"\b(?:mtech|m\.tech|postgraduate|pg).*(?:registration\s+limit|credit\s+limit|normal\s+semester|regular\s+semester)\b": "mtech_registration_limits",

    # Branch Change
    r"\b(?:branch\s+change|change\s+of\s+branch|switch\s+branch|transfer\s+branch)\b": "branch_change_criteria",

    # Add / Drop / Audit Courses
    r"\b(?:add\s+(?:and|&)\s+drop|drop\s+(?:a\s+)?course|dropping\s+(?:a\s+)?course|convert(?:ing)?.*audit)\b": "course_add_drop_audit",

    # Course Repetition & F Grade
    r"\b(?:repeat(?:ing)?\s+(?:a\s+)?course|clear(?:ing)?\s+an?\s*['\"]?f['\"]?|clear.*f\s+grade|f\s+grade.*repeat|re-?taking)\b": "course_repeat_f_grade",

    # Academic Probation Termination
    r"\b(?:academic\s+probation.*termination|termination.*academic\s+probation|probation.*terminate|consecutive\s+semesters.*probation)\b": "academic_probation_termination",

    # Course Overload
    r"\b(?:course\s+overload|additional\s+overload\s+credits?|overload.*6th\s+semester|overload.*per\s+semester)\b": "course_overload_credits_limit",

    # CPDA Allowance
    r"\b(?:cpda|cumulative\s+professional\s+development\s+allowance|block\s+period.*cpda)\b": "cpda_financial_grant",

    # Faculty Relocation Reimbursement
    r"\b(?:relocation|relocation\s+reimbursement|relocation\s+expenses?)\b": "relocation_reimbursement",

    # Research Initiation Seed Grant
    r"\b(?:seed\s+grant|research\s+initiation\s+grant|research\s+seed\s+grant)\b": "research_seed_grant",

    # B.Tech Programme Duration & Residence
    r"\b(?:permissible\s+duration|programme\s+duration.*b\.?tech|minimum\s+residence|residential\s+requirement)\b": "btech_programme_duration",

    # E-Grade & Re-examination
    r"\b(?:['\"]?e['\"]?\s+grade|grade\s+['\"]?e['\"]?|e\s+grade.*re-?examination|e\s+grade.*points?)\b": "grade_e_additional_exam",

    # Late Registration
    r"\b(?:late\s+registration|fine\s+for\s+late|late\s+fine|penalty\s+for\s+late)\b": "late_registration_procedure",

    # Summer Term Credits
    r"\b(?:summer\s+term|summer\s+semester|summer\s+courses?)\b": "summer_term_credits",

    # Assistantship Continuation CGPA
    r"\b(?:assistantship|fellowship).*(?:continuation|cgpa|minimum\s+cgpa)\b": "assistantship_continuation_cgpa",

    # Ph.D. Comprehensive Exam
    r"\b(?:comprehensive\s+exam(?:ination)?|candidacy.*deadline|comprehensive.*attempts?)\b": "phd_comprehensive_exam",

    # Non-Graded Graduation Requirements
    r"\b(?:ncc|nss|nso|sports|extra[- ]curricular|non[- ]graded\s+requirements?)\b": "non_graded_graduation_requirements",

    # Sabbatical & Supervision Continuity
    r"\b(?:sabbatical|lien|caretaker\s+supervisor|supervisor.*sabbatical)\b": "sabbatical_supervision_continuity",

    # Consultancy & Project Overhead
    r"\b(?:overhead.*sponsored|overhead.*consultancy|consultancy\s+overhead|sponsored\s+project\s+overhead)\b": "consultancy_overhead"
}


def canonicalize_query_intent(query: str) -> Optional[str]:
    """Resolves arbitrary paraphrased questions into their canonical policy/attribute key."""
    q_low = query.lower()
    for pattern, canonical_key in CANONICAL_INTENT_MAP.items():
        if re.search(pattern, q_low):
            return canonical_key
    return None


# ── 3. Evidence Integrity Filter ──────────────────────────────────────────────

@dataclass
class EvidenceIntegrityResult:
    accepted_candidates: List[Any]
    quarantined_candidates: List[Tuple[Any, str]]  # (candidate, quarantine_reason)
    detected_role: Optional[AdministrativeRoleDefinition]
    canonical_intent: Optional[str]
    is_role_query: bool
    strict_evidence_available: bool


class EvidenceIntegrityLayer:
    """
    Enforces strict role-to-entity matching, domain isolation, and semantic relevance
    before candidate chunks are passed to the context builder and generator.
    """

    @staticmethod
    def identify_requested_role(query: str) -> Optional[AdministrativeRoleDefinition]:
        q_low = query.lower()
        for role_key, role_def in ADMINISTRATIVE_ROLES.items():
            for alias in role_def.aliases:
                if re.search(rf"\b{re.escape(alias)}\b", q_low):
                    return role_def
        return None

    @staticmethod
    def filter_and_validate_evidence(
        query: str,
        candidates: List[Any],
        plan: Optional[QueryPlan] = None
    ) -> EvidenceIntegrityResult:
        canonical_intent = canonicalize_query_intent(query)
        requested_role = EvidenceIntegrityLayer.identify_requested_role(query)
        q_low = query.lower()

        accepted: List[Any] = []
        quarantined: List[Tuple[Any, str]] = []

        is_role_q = requested_role is not None

        for c in candidates:
            c_text = c.chunk.get("text", "")
            c_src = c.chunk.get("source", "")
            comb = f"{c_src} {c_text}".lower()

            # 1. Role-to-Entity Isolation Check
            if is_role_q and requested_role:
                # Check if this chunk is discussing a confusable role without mentioning the target role
                has_target_role = any(re.search(rf"\b{re.escape(a)}\b", comb) for a in requested_role.aliases)
                has_confusable_role = any(re.search(rf"\b{re.escape(cr)}\b", comb) for cr in requested_role.confusable_roles)

                if has_confusable_role and not has_target_role:
                    quarantined.append((c, f"ROLE_CONFUSION: Contains confusable role but missing '{requested_role.canonical_title}'"))
                    continue

                if not has_target_role:
                    # If this is a general committee list that does not mention the specific role
                    quarantined.append((c, f"ROLE_MISMATCH: Missing target role '{requested_role.canonical_title}'"))
                    continue

            # 2. Degree-Level Scope Isolation Check
            is_ug_query = bool(re.search(r"\b(?:btech|b\.tech|undergraduate|ug)\b", q_low)) and not bool(re.search(r"\b(?:phd|ph\.d|mtech|m\.tech|pg)\b", q_low))
            is_pg_query = bool(re.search(r"\b(?:mtech|m\.tech|postgraduate|pg)\b", q_low)) and not bool(re.search(r"\b(?:btech|b\.tech|ug)\b", q_low))
            is_phd_query = bool(re.search(r"\b(?:phd|ph\.d|doctoral)\b", q_low))

            # Disallow pure UG regulations when querying PG M.Tech credit limits
            if canonical_intent == "mtech_registration_limits":
                if "1_academic_regulations_final_03_09_2019" in c_src.lower() and "undergraduate student" in c_text.lower() and "table 11.1" not in c_text.lower():
                    quarantined.append((c, "SCOPE_MISMATCH: UG registration clause supplied for M.Tech query"))
                    continue

            # Disallow PG regulations when querying UG registration limits
            if canonical_intent == "ug_semester_registration_limits":
                if "4_regulation_pg" in c_src.lower() and "m.tech." in c_text.lower() and "undergraduate" not in c_text.lower():
                    quarantined.append((c, "SCOPE_MISMATCH: PG registration clause supplied for UG query"))
                    continue

            # 3. Privacy Guardrail: Faculty Salary vs Academic Curriculum
            if any(k in comb for k in ["gross_salary", "monthly remuneration", "immovable property return"]):
                quarantined.append((c, "PRIVACY_FIREWALL: Contains individual compensation/salary record"))
                continue

            accepted.append(c)

        strict_evidence_available = len(accepted) > 0
        return EvidenceIntegrityResult(
            accepted_candidates=accepted,
            quarantined_candidates=quarantined,
            detected_role=requested_role,
            canonical_intent=canonical_intent,
            is_role_query=is_role_q,
            strict_evidence_available=strict_evidence_available
        )
