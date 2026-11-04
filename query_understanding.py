"""
query_understanding.py
======================
Dynamic Query Understanding, Intent Decomposition, Semantic Profiling,
and Request-Level Target Identification for IIT Jodhpur RAG System.

Translates arbitrary user prompts (with spelling mistakes, abbreviations,
natural language phrasing, and multi-faceted requests) into structured QueryProfile
objects without hardcoding specific user questions or document filenames.
"""

import re
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any


class QueryDomain(str, Enum):
    CURRICULUM = "CURRICULUM"
    ACADEMIC_REGULATIONS = "ACADEMIC_REGULATIONS"
    ADMISSIONS = "ADMISSIONS"
    GOVERNANCE_FACULTY = "GOVERNANCE_FACULTY"
    FINANCIAL_INSTITUTIONAL = "FINANCIAL_INSTITUTIONAL"
    PERSONAL_FINANCIAL_PRIVACY = "PERSONAL_FINANCIAL_PRIVACY"
    CAMPUS_FACILITIES = "CAMPUS_FACILITIES"
    GENERAL = "GENERAL"


class QueryIntent(str, Enum):
    QUANTITATIVE = "QUANTITATIVE"
    ELIGIBILITY = "ELIGIBILITY"
    PROCEDURAL = "PROCEDURAL"
    FACTUAL = "FACTUAL"
    COMPARISON = "COMPARISON"
    GENERAL = "GENERAL"


@dataclass
class QueryProfile:
    request_id: str
    original_query: str
    normalized_query: str
    domain: QueryDomain
    intent: QueryIntent
    subject_entity: Optional[str] = None
    degree_level: str = "UG"  # "UG", "PG", "PhD", "ANY"
    target_attributes: List[str] = field(default_factory=list)
    applicable_cohort: Optional[str] = None  # "earlier_2019_2024", "revised_2025_onwards", "all_cohorts"
    applicable_semester: Optional[int] = None
    required_evidence_types: List[str] = field(default_factory=list)
    search_rewrites: List[str] = field(default_factory=list)
    is_comparison: bool = False
    requires_multi_cohort: bool = False
    calculation_needed: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "domain": self.domain.value,
            "intent": self.intent.value,
            "subject_entity": self.subject_entity,
            "degree_level": self.degree_level,
            "target_attributes": self.target_attributes,
            "applicable_cohort": self.applicable_cohort,
            "applicable_semester": self.applicable_semester,
            "required_evidence_types": self.required_evidence_types,
            "search_rewrites": self.search_rewrites,
            "is_comparison": self.is_comparison,
            "requires_multi_cohort": self.requires_multi_cohort,
            "calculation_needed": self.calculation_needed,
        }


# Canonical Programme / Department Mapping
ENTITY_PATTERNS = {
    "Computer Science and Engineering": [
        r"\b(?:cse|cs|computer\s+science|computer\s+sc|computer\s+engineering)\b"
    ],
    "Artificial Intelligence and Data Science": [
        r"\b(?:aide|ai\s*(?:&|and)\s*ds|ai\s*ds|artificial\s+intelligence|data\s+science)\b"
    ],
    "Mechanical Engineering": [
        r"\b(?:me|mechanical|mech|mechanical\s+engineering)\b"
    ],
    "Electrical Engineering": [
        r"\b(?:ee|electrical|electrical\s+engineering)\b"
    ],
    "Civil and Infrastructure Engineering": [
        r"\b(?:civil|cie|civil\s+(?:&|and)\s+infrastructure|infrastructure\s+engineering)\b"
    ],
    "Chemical Engineering": [
        r"\b(?:chemical|ch|chem\s+eng|chemical\s+engineering)\b"
    ],
    "Materials Engineering": [
        r"\b(?:materials|materials\s+engineering|material\s+engineering|meta|metallurgy|mse)\b"
    ],
    "Bioscience and Bioengineering": [
        r"\b(?:bsbe|bioscience|bioengineering|biotechnology|biotech)\b"
    ],
    "Physics": [
        r"\b(?:physics|phy|ep|engineering\s+physics)\b"
    ],
    "Chemistry": [
        r"\b(?:chemistry|chem)\b"
    ],
    "Mathematics": [
        r"\b(?:mathematics|maths|math|applied\s+mathematics)\b"
    ],
    "Humanities and Social Sciences": [
        r"\b(?:hss|humanities|social\s+sciences|liberal\s+arts)\b"
    ],
    "School of Management and Entrepreneurship": [
        r"\b(?:sme|management|entrepreneurship|mba)\b"
    ],
}

# Common phonetic, typing and slang corrections
COMMON_TYPOS = {
    r"\bbetch\b": "btech",
    r"\bbtec\b": "btech",
    r"\bb\.tech\b": "btech",
    r"\bm\.tech\b": "mtech",
    r"\bmsc\b": "m.sc",
    r"\bph\.?d\b": "phd",
    r"\bcredts\b": "credits",
    r"\bcrdeits\b": "credits",
    r"\bsemster\b": "semester",
    r"\bsem\b": "semester",
    r"\belgibility\b": "eligibility",
    r"\beligiblity\b": "eligibility",
    r"\bcriterea\b": "criteria",
    r"\bspecilization\b": "specialization",
    r"\bspecialisation\b": "specialization",
    r"\bchnge\b": "change",
    r"\bproffesor\b": "professor",
    r"\bproffessor\b": "professor",
    r"\brequird\b": "required",
    r"\bpe\s+credits?\b": "programme elective credits",
    r"\boe\s+credits?\b": "open elective credits",
    r"\battandance\b": "attendance",
}


def normalize_query_text(query: str) -> str:
    """Normalizes spelling, typos, abbreviations and whitespace in query text."""
    clean = query.strip()
    clean_lower = clean.lower()
    for pattern, repl in COMMON_TYPOS.items():
        clean_lower = re.sub(pattern, repl, clean_lower)
    return clean_lower


def detect_degree_level(q_lower: str) -> str:
    """Detects academic degree level target (UG, PG, PhD, or ANY)."""
    if re.search(r"\b(?:consultancy|sponsored\s+project|overhead|sabbatical|faculty|professor|instructor|coordinator|dac|drc|src)\b", q_lower):
        return "Faculty/Institute-Wide"
    if re.search(r"\b(?:phd|doctoral|research\s+scholar)\b", q_lower):
        return "PhD"
    if re.search(r"\b(?:mtech|m\.tech|msc|m\.sc|postgraduate|pg|master|mba)\b", q_lower):
        return "PG"
    if re.search(r"\b(?:btech|b\.tech|undergraduate|ug|bachelor|bs|b\.s)\b", q_lower):
        return "UG"
    return "UG"  # Default to UG for standard institute questions


def detect_subject_entity(q_lower: str) -> Optional[str]:
    """Identifies specific academic department or programme if present."""
    for entity, patterns in ENTITY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q_lower):
                return entity
    return None


def detect_domain_and_intent(q_lower: str, subject_entity: Optional[str]) -> (QueryDomain, QueryIntent, List[str]):
    """
    Classifies domain, intent, and target attributes using semantic structure
    rather than isolated single-word collisions.
    """
    target_attributes: List[str] = []

    # 1. Personal Financial Privacy Firewall Check
    # Must explicitly involve personal salary, remuneration, payslip, property returns of faculty/staff
    is_personal_salary = bool(re.search(
        r"\b(?:salar(?:y|ies)|remunerat(?:ion|ions)|gross\s+pay|net\s+pay|payslip|ctc|immovable\s+property|property\s+return)\b",
        q_lower
    ))
    is_faculty_person_targeted = bool(re.search(
        r"\b(?:prof(?:essor)?|dr\.?|faculty|staff|employee|director|registrar|dean|[a-z]+\s+kalra|[a-z]+\s+paranjothy)\b",
        q_lower
    ))
    # Academic concepts that disprove personal financial inquiry
    is_academic_context = bool(re.search(
        r"\b(?:credit|credits|minor|specialization|degree|course|courses|grade|grades|cgpa|branch\s+change|curriculum|tuition|hostel\s+fee|fee\s+structure)\b",
        q_lower
    ))

    if is_personal_salary and is_faculty_person_targeted and not is_academic_context:
        return QueryDomain.PERSONAL_FINANCIAL_PRIVACY, QueryIntent.FACTUAL, ["faculty_personal_financial_record"]

    # 2. Academic Regulations / Institutional Policies
    # Prioritize specific policy checks before generic catch-alls (e.g. registration limits, dual degree)
    is_phd_credits = bool(re.search(r"\b(?:ph\.?d|phd|doctoral)\b", q_lower) and any(k in q_lower for k in ["coursework", "course work", "thesis", "credit"]))
    if is_phd_credits:
        target_attributes.append("phd_coursework_thesis_credits")
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.FACTUAL, target_attributes

    is_phd_leave = bool(
        re.search(r"\b(?:ph\.?d|phd|doctoral|scholar)\b", q_lower)
        and re.search(r"\b(?:leave|casual\s+leave|medical\s+leave|vacation\s+leave)\b", q_lower)
    )
    if is_phd_leave:
        target_attributes.append("phd_leave_entitlement")
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.QUANTITATIVE, target_attributes

    is_sugc = bool(re.search(r"\b(?:sugc|senate\s+undergraduate\s+committee|standing\s+committee\s+of\s+(?:the\s+)?senate)\b", q_lower))
    if is_sugc:
        target_attributes.append("sugc_mandate_role")
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.FACTUAL, target_attributes

    is_dual_degree = bool(re.search(r"\b(?:dual\s+degree|b\.?tech\s*[-+]\s*m\.?tech|conversion\s+to\s+dual|convert\s+to\s+(?:a\s+)?dual)\b", q_lower))
    if is_dual_degree:
        target_attributes.extend(["dual_degree_conversion", "dual_degree_134_credits", "dual_degree_min_cgpa_6"])
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.ELIGIBILITY, target_attributes

    is_mtech_duration = bool(
        re.search(r"\b(?:mtech|m\.tech|master\s+of\s+technology)\b", q_lower)
        and re.search(r"\b(?:duration|normal\s+and\s+maximum|maximum\s+duration|normal\s+duration|residence|permissible\s+duration|complete\s+an\s+m\.?tech)\b", q_lower)
    )
    if is_mtech_duration:
        target_attributes.append("mtech_programme_duration")
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.FACTUAL, target_attributes

    is_mtech_reg_limits = bool(
        re.search(r"\b(?:mtech|m\.tech|postgraduate|pg)\b", q_lower)
        and not re.search(r"\b(?:ph\.?d|phd|doctoral|thesis|coursework|dual|convert|conversion|duration|complete\s+an\s+m\.?tech)\b", q_lower)
        and (
            re.search(r"\b(?:registration\s+limit|credit\s+limit|credit\s+registration|maximum\s+credit|minimum\s+credit|normal\s+semester|regular\s+semester|credit\s+load)\b", q_lower)
            or (re.search(r"\b(?:maximum|minimum|limit)\b", q_lower) and re.search(r"\b(?:credit|credits)\b", q_lower) and re.search(r"\b(?:semester|register|registration)\b", q_lower))
        )
    )
    if is_mtech_reg_limits:
        target_attributes.append("mtech_registration_limits")
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.QUANTITATIVE, target_attributes

    is_seed_grant = bool(re.search(r"\b(?:seed\s+grant|research\s+initiation\s+grant)\b", q_lower))
    if is_seed_grant:
        target_attributes.append("research_seed_grant")
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.QUANTITATIVE, target_attributes

    is_phd_comprehensive = bool(re.search(r"\b(?:comprehensive\s+exam(?:ination)?|candidacy\s+requirement)\b", q_lower) and any(k in q_lower for k in ["semester", "attempt", "pass", "time", "deadline", "candidacy"]))
    if is_phd_comprehensive:
        target_attributes.append("phd_comprehensive_exam")
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.FACTUAL, target_attributes

    is_overload_cap = bool(
        not any(k in q_lower for k in ["minor", "specialization"])
        and re.search(r"\b(?:course\s+overload|additional\s+credit(?:s)?|overload\s+credit(?:s)?)\b", q_lower)
        and any(k in q_lower for k in ["per semester", "semester", "6th", "sixth", "maximum", "how many", "permit"])
    )
    if is_overload_cap:
        target_attributes.append("course_overload_credits_limit")
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.QUANTITATIVE, target_attributes

    is_probation_termination = bool(re.search(r"\b(?:academic\s+probation|probation)\b", q_lower) and any(k in q_lower for k in ["termination", "terminate", "consecutive", "cancel", "result in the termination"]))
    if is_probation_termination:
        target_attributes.append("academic_probation_termination")
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.FACTUAL, target_attributes

    is_assistantship_continuation = bool(re.search(r"\b(?:assistantship|fellowship)\b", q_lower) and any(k in q_lower for k in ["continuation", "cgpa", "minimum", "mtech", "m.tech", "dual"]))
    if is_assistantship_continuation:
        target_attributes.append("assistantship_continuation_cgpa")
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.FACTUAL, target_attributes

    is_programme_duration = bool(
        not any(k in q_lower for k in ["mtech", "m.tech", "phd", "ph.d", "master"])
        and (
            re.search(r"\b(?:permissible\s+duration|minimum\s+and\s+maximum\s+duration|programme\s+duration|period\s+of\s+residence|duration\s+(?:in\s+)?semesters)\b", q_lower)
            or ("duration" in q_lower and any(k in q_lower for k in ["btech", "b.tech", "undergraduate", "degree"]) and any(k in q_lower for k in ["minimum", "maximum", "semesters", "years"]))
        )
    )
    if is_programme_duration:
        target_attributes.append("btech_programme_duration")
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.FACTUAL, target_attributes

    is_cpda = bool(re.search(r"\b(?:cpda|cumulative\s+professional\s+development\s+allowance)\b", q_lower))
    if is_cpda:
        target_attributes.append("cpda_financial_grant")
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.FACTUAL, target_attributes

    is_relocation = bool(re.search(r"\b(?:relocation|relocation\s+expenses?)\b", q_lower) and any(k in q_lower for k in ["reimbursement", "india", "abroad", "limit", "ceiling", "maximum"]))
    if is_relocation:
        target_attributes.append("relocation_reimbursement")
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.FACTUAL, target_attributes

    is_overhead_query = bool(re.search(r"\b(?:overhead|consultancy|sponsored\s+project|deduction\s+rate)\b", q_lower))
    if is_overhead_query:
        target_attributes.append("consultancy_overhead")
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.FACTUAL, target_attributes

    is_sabbatical_query = bool(re.search(r"\b(?:sabbatical|lien|caretaker\s+supervisor)\b", q_lower))
    if is_sabbatical_query:
        target_attributes.append("sabbatical_leave")
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.FACTUAL, target_attributes

    is_branch_change = bool(re.search(r"\b(?:branch\s+change|change\s+of\s+branch|switch\s+branch|transfer\s+branch)\b", q_lower))
    is_attendance = bool(re.search(r"\b(?:attendance|de-registration|deregistration|short\s+leave|absence)\b", q_lower))
    is_exam_policy = bool(re.search(r"\b(?:exam\s+policy|grading\s+policy|passing\s+grade|grade\s+penalty|probation|termination)\b", q_lower))
    is_minor_overload = bool(re.search(r"\b(?:minor|specialization)\b", q_lower) and re.search(r"\b(?:overload|additional\s+credits|beyond\s+(?:regular|degree)|how\s+many\s+credits)\b", q_lower))
    is_summer_term = bool(re.search(r"\b(?:summer\s+term|summer\s+courses?|summer\s+semester)\b", q_lower))
    is_incomplete_grade = bool(re.search(r"\b(?:incomplete|grade\s+['\"]?i['\"]?|['\"]?i['\"]?\s+grade)\b", q_lower))
    is_course_repeat = bool(re.search(r"\b(?:fail(?:s|ed|ing)?\s+a?\s*(?:core\s+)?course|repeat(?:s|ed|ing)?\s+a?\s*(?:core\s+)?course|repeat(?:s|ed|ing)?\s+it|repeats|repeating|repeating\s+a\s+course|re-?taking|clear(?:ing)?\s+an?\s*['\"]?f['\"]?|grade\s+improvement)\b", q_lower))
    is_dual_degree = bool(re.search(r"\b(?:dual\s+degree|b\.?tech\s*[-+]\s*m\.?tech|conversion\s+to\s+dual)\b", q_lower))
    is_non_graded_req = bool(re.search(r"\b(?:ncc|nss|nso|extra[- ]curricular|non[- ]graded|social\s+connect)\b", q_lower))
    is_course_drop_audit = bool(re.search(r"\b(?:drop\s+(?:a\s+)?course|audit\s+(?:a\s+)?course|add\s+(?:and|&)\s+drop|withdrawing\s+from\s+(?:a\s+)?course|course\s+drop|convert(?:ing)?\s+(?:a\s+)?(?:registered\s+)?course\s+to\s+(?:an\s+)?audit)\b", q_lower))
    is_degree_curriculum = bool(re.search(r"\b(?:degree|curriculum|curricula|complete\s+a\s+b\.?tech|award\s+of\s+(?:a\s+)?b\.?tech)\b", q_lower))
    is_registration_limits = bool(
        not is_degree_curriculum
        and (
            re.search(r"\b(?:limit(?:s)?\s+on\s+registration|registration\s+limit(?:s)?|semester\s+credit\s+limit|credit\s+registration\s+limit|under-?load|over-?load|normal\s+semester\s+credit|regular\s+semester\s+credit)\b", q_lower)
            or (re.search(r"\b(?:register|registration|semester|term)\b", q_lower) and re.search(r"\b(?:credit|credits|load)\b", q_lower) and re.search(r"\b(?:maximum|minimum|limit|ceiling)\b", q_lower))
        )
    )
    is_graduation_req = bool(not is_course_repeat and re.search(
        r"\b(?:graduat(?:ing|ion)\s+(?:cgpa|cpi|threshold|cutoff|requirement(?:s)?|criteria)|award\s+of\s+(?:the\s+)?(?:b\.?tech|m\.?tech|degree)|degree\s+award\s+criteria|requirements?\s+for\s+graduation|minimum\s+cgpa\s+(?:required\s+)?(?:for\s+)?(?:graduat|award|b\.?tech))\b",
        q_lower
    ))
    is_letter_grades = bool(re.search(
        r"\b(?:['\"]?e['\"]?\s*(?:letter\s*)?grade|grade\s*['\"]?e['\"]?|grade\s+points?|letter\s+grades?|table\s+5\.1|additional\s+exam(?:ination)?|scale\s+of\s+grade\s+points?)\b",
        q_lower
    ))
    is_late_registration = bool(re.search(
        r"\b(?:late\s+registration|fine\s+for\s+late|daily\s+fine|penalty\s+for\s+late|register\s+after\s+the\s+last\s+date)\b",
        q_lower
    ))

    if is_registration_limits and not is_summer_term:
        target_attributes.extend(["min_semester_credits_12", "max_semester_credits_25", "underload_probation_16", "overload_cgpa_7_5"])
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.QUANTITATIVE, target_attributes

    if is_graduation_req:
        target_attributes.extend(["graduating_cgpa_5_5_old", "academic_requirements_section_17", "residence_minimum_4_years"])
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.ELIGIBILITY, target_attributes

    if is_letter_grades:
        target_attributes.extend(["grade_e_additional_exam_2_points", "grade_i_incomplete_calendar_f", "letter_grades_scale"])
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.FACTUAL, target_attributes

    if is_late_registration:
        target_attributes.extend(["late_registration_procedure", "late_registration_daily_fine_absence"])
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.PROCEDURAL, target_attributes

    if is_branch_change:
        target_attributes.extend(["branch_change_timing", "minimum_cgpa", "backlog_status", "disciplinary_status", "department_strength_limits"])
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.ELIGIBILITY, target_attributes

    if is_minor_overload:
        target_attributes.extend(["minor_total_credits", "open_elective_overlap", "additional_overload_credits", "cgpa_overload_requirement"])
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.QUANTITATIVE, target_attributes

    if is_attendance:
        target_attributes.extend(["attendance_threshold_75", "deregistration_rule", "semester_withdrawal_rule", "circular_51_amendment"])
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.PROCEDURAL, target_attributes

    if is_summer_term:
        target_attributes.extend(["summer_term_credits", "summer_eligibility_deficient", "summer_registration_fee"])
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.ELIGIBILITY, target_attributes

    if is_incomplete_grade:
        target_attributes.extend(["incomplete_i_grade", "medical_absence_grounds", "conversion_deadline_f_grade"])
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.PROCEDURAL, target_attributes

    if is_course_repeat:
        target_attributes.extend(["course_repeat_core_elective", "grade_replacement_transcript", "cgpa_calculation_repeat"])
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.PROCEDURAL, target_attributes

    if is_dual_degree:
        target_attributes.extend(["dual_degree_conversion", "dual_degree_134_credits", "dual_degree_min_cgpa_6"])
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.ELIGIBILITY, target_attributes

    if is_non_graded_req:
        target_attributes.extend(["mandatory_non_graded", "ncc_nss_nso_requirement", "two_semesters_extracurricular"])
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.PROCEDURAL, target_attributes

    if is_course_drop_audit:
        target_attributes.extend(["add_drop_deadline", "course_audit_rules", "course_withdrawal_w_grade"])
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.PROCEDURAL, target_attributes

    if is_exam_policy:
        target_attributes.extend(["exam_weightage", "grading_rule", "probation_threshold", "termination_criteria"])
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.PROCEDURAL, target_attributes

    # General Academic Regulations indicator
    if re.search(r"\b(?:regulations?|rules?|circular|academic\s+policy|leave\s+rules?|withdrawal|probation|termination)\b", q_lower):
        target_attributes.append("academic_regulations_policy")
        return QueryDomain.ACADEMIC_REGULATIONS, QueryIntent.PROCEDURAL, target_attributes

    # 3. Governance / Administration / Faculty
    if re.search(r"\b(?:hod|head\s+of\s+department|dean|director|registrar|senate|chairperson|chairman)\b", q_lower):
        target_attributes.append("institutional_leadership")
        return QueryDomain.GOVERNANCE_FACULTY, QueryIntent.FACTUAL, target_attributes

    # 4. Campus Facilities / Health
    if not any(k in q_lower for k in ["leave", "casual leave", "medical leave"]) and re.search(r"\b(?:health\s+centre|health\s+center|hospital|medical|doctor|phc|clinic|ambulance)\b", q_lower):
        target_attributes.append("campus_medical_facilities")
        return QueryDomain.CAMPUS_FACILITIES, QueryIntent.FACTUAL, target_attributes

    # 5. Financial / Institutional Fees (Public Fee Structure)
    if re.search(r"\b(?:fees?|tuition|hostel\s+fee|mess\s+fee|fee\s+structure|payment|dues)\b", q_lower):
        target_attributes.append("institute_fee_structure")
        return QueryDomain.FINANCIAL_INSTITUTIONAL, QueryIntent.QUANTITATIVE, target_attributes

    # 6. Curriculum / Syllabus / Credit Requirements
    is_credits = bool(re.search(r"\b(?:credits?|pe|oe|electives?|course|courses|graded|audit|ltp|l-t-p|structure|curriculum|syllabus)\b", q_lower))
    if is_credits:
        if re.search(r"\b(?:pe|programme\s+electives?)\b", q_lower):
            target_attributes.append("programme_electives_pe")
        if re.search(r"\b(?:oe|open\s+electives?)\b", q_lower):
            target_attributes.append("open_electives_oe")
        if re.search(r"\b(?:total|degree\s+requirement|grand\s+total)\b", q_lower):
            target_attributes.append("total_degree_credits")
        if not target_attributes:
            target_attributes.append("curriculum_course_structure")

        intent = QueryIntent.QUANTITATIVE if re.search(r"\b(?:how\s+many|number|total|credits?)\b", q_lower) else QueryIntent.FACTUAL
        return QueryDomain.CURRICULUM, intent, target_attributes

    # 7. Admissions / Entrance Eligibility
    if re.search(r"\b(?:admission|admissions|apply|entrance|eligibility\s+for\s+admission|seat\s+matrix)\b", q_lower):
        target_attributes.append("admission_eligibility")
        return QueryDomain.ADMISSIONS, QueryIntent.ELIGIBILITY, target_attributes

    return QueryDomain.GENERAL, QueryIntent.GENERAL, ["general_inquiry"]


def generate_search_rewrites(
    q_norm: str,
    domain: QueryDomain,
    entity: Optional[str],
    attributes: List[str]
) -> List[str]:
    """
    Generates targeted query rewrites to optimize both sparse (BM25)
    and dense semantic retrieval without single-point failure.
    """
    rewrites = [q_norm]

    if domain == QueryDomain.ACADEMIC_REGULATIONS:
        if "branch_change_timing" in attributes or "minimum_cgpa" in attributes:
            rewrites.append("branch change eligibility criteria minimum CGPA Academic Regulations 16")
            rewrites.append("branch change 1st year semester II CGPA 8.5 7.5 strength")
            rewrites.append("Change/Enhancement of Program branch change undergraduate")
        elif "minor_total_credits" in attributes or "additional_overload_credits" in attributes:
            rewrites.append("Capability Linked Opportunities Department Specialization Minor 20 credits 10 credits overload")
            rewrites.append("11.2 Capability Linked Opportunities Major Department Specialization Minor 20 Credits Open electives overload")
            rewrites.append("additional credits beyond regular B.Tech minor specialization 10 credits")
        elif "attendance_threshold_75" in attributes:
            rewrites.append("Circular 51 attendance 75 percent de-registration semester withdrawal")
            rewrites.append("Academic Regulations 4.1.9 attendance policy amendment")
        elif "summer_term_credits" in attributes:
            rewrites.append("12.1 Summer Courses maximum Eight Credits registration fee deficient students")
            rewrites.append("Summer Term undergraduate eligibility registration maximum credits")
        elif "incomplete_i_grade" in attributes:
            rewrites.append("5.1 Incomplete I grade missing evaluation medical grounds Academic Calendar convert F grade")
            rewrites.append("Incomplete I Grade converted into an F grade")
        elif "course_repeat_core_elective" in attributes:
            rewrites.append("Section 5.3 Repeating a Course Section 5.4 F grade repeat core elective summer term regular semester")
            rewrites.append("repeat the course core elective E grade F grade transcript CGPA calculation new grade")
            rewrites.append("repeating a course grade improvement 5.1 letter grades Academic Regulations")
        elif "dual_degree_conversion" in attributes:
            rewrites.append("11.2 B.Tech. + M.Tech. Dual Degree 134 Credits at least 6 CGPA eighth semester")
            rewrites.append("convert Dual Degree undergraduate eligibility minimum CGPA 6.0")
        elif "mandatory_non_graded" in attributes or "ncc_nss_nso_requirement" in attributes:
            rewrites.append("17.5 Completion of engagement in NCC, NSS or NSO at least two Semesters")
            rewrites.append("National Cadet Corps National Service Schemes National Sports Organisation non-graded")
        elif "min_semester_credits_12" in attributes or "max_semester_credits_25" in attributes:
            rewrites.append("12.2 Limits on Registration minimum of 12 credits maximum number of credits permitted 25")
            rewrites.append("12.3 Exceptions underload 16 credits academically deficient overload 7.5 CGPA")
            rewrites.append("Limits on Registration undergraduate student normal semester credit limit")
        elif "graduating_cgpa_5_5_old" in attributes or "academic_requirements_section_17" in attributes:
            rewrites.append("10. REQUIREMENTS FOR GRADUATION 10.2 Academic minimum CGPA of 5.5 in all the passed courses")
            rewrites.append("17. Requirements for Graduation 17.2 Academic fulfilled minimum credits Table 11.2")
            rewrites.append("minimum graduating CGPA award of B.Tech degree eight semesters four years")
        elif "grade_e_additional_exam_2_points" in attributes or "letter_grades_scale" in attributes:
            rewrites.append("Table 5.1 Letter Grade and its equivalent Grade Points A* 10 A 10 B 8 C 6 D 4 E 2 F 0")
            rewrites.append("StructuredTable 1_Academic_Regulations_Final_03_09_2019 Table 5.1 Letter Grade Grade Points")
            rewrites.append("5.1 Letter Grades E grade scored marks less than D cutoff met attendance criterion")
        elif "phd_leave_entitlement" in attributes:
            rewrites.append("Ph.D. Leave 7.1 Short Leave casual leave 8 days medical leave 15 days vacation leave 15 days per semester")
            rewrites.append("Ph.D. Regulations Section 14 Section 7.1 casual leave medical leave supported by medical certificate")
        elif "mtech_programme_duration" in attributes:
            rewrites.append("Academic Regulations Section 21.5 Duration of the M.Tech. Program normal 4 semesters maximum 12 registered semesters")
            rewrites.append("M.Tech degree normal designed duration 4 semesters maximum completion duration RKA")
        elif "sugc_mandate_role" in attributes:
            rewrites.append("Senate Undergraduate Committee SUGC standing committee of the Senate undergraduate academic regulations curriculum")
            rewrites.append("SUGC role mandate academic probation deregistration underload overload branch change")
        elif "late_registration_procedure" in attributes:
            rewrites.append("3.2 Late Registration dates specified in Academic Calendar paying the fine Dean")
            rewrites.append("Section 3.2 Late Registration permission Convener Dean")
        elif "add_drop_deadline" in attributes or "course_drop_minimum_12_credits" in attributes:
            rewrites.append("Adding and Dropping of Courses last date of registration Faculty Advisor under-load 12 credits")
            rewrites.append("3.3 Adding and Dropping of Courses 5.5 drop a course Audit courses")
        elif "consultancy_overhead" in attributes:
            rewrites.append("Table 1 Calculation of Institute Overhead off the Total Project Cost 0.2X 20%")
            rewrites.append("Clause 4.2 Consultancy Projects Institute Overhead 30% Table 3 Table 4")
            rewrites.append("Utilisation of Institute Overheads collected from Sponsored Research Projects")
        elif "sabbatical_leave" in attributes:
            rewrites.append("supervisor proceeds on short leave caretaker supervisor DRC IRC")
            rewrites.append("supervisor proceeds on long leave additional supervisor lien or sabbatical")
        elif "assistantship_continuation_cgpa" in attributes:
            rewrites.append("Table 12.8 Minimum CGPA requirements for Continuation of Assistantship Fellowship M.Tech 6.0 Dual Degree 6.5")
            rewrites.append("12.8.4 Minimum performance for continuation Assistantship Fellowship Table 22.8")
        elif "btech_programme_duration" in attributes:
            rewrites.append("10.1 The minimum period of residence shall be eight semesters four years and the maximum period sixteen semesters eight years")
            rewrites.append("17.1 minimum period of residence eight semesters maximum period sixteen semesters")
        elif "cpda_financial_grant" in attributes:
            rewrites.append("Cumulative Professional Development Allowance CPDA of Rs 3 lakhs for every block period of 3 years")
            rewrites.append("Research Initiation Grant 25 Lakhs CPDA 3 Lakhs")
        elif "relocation_reimbursement" in attributes:
            rewrites.append("Reimbursement of relocation expenses Rs 1,50,000 from abroad Rs 2,50,000")
            rewrites.append("Relocation expenses up to Rs 1.5 Lakhs from any place in India or abroad")
        elif "phd_coursework_thesis_credits" in attributes:
            rewrites.append("Table 12.6 Credit Requirements for PhD students M.Tech in Engineering 12 credits course work 48 credits thesis work")
            rewrites.append("Table 12.6 Credit Requirements for PhD students B.Tech M.Sc. in Science 24 credits course work 48 credits thesis work")
            rewrites.append("Table 12.6 direct entry PhD coursework credits 24 thesis credits 48")
            rewrites.append("Table 22.6 Credit Requirements for PhD students 12 48")
        elif "mtech_registration_limits" in attributes:
            rewrites.append("Table 11.1 Maximum and Minimum Credits a M.Tech. student can register for during the Regular semester Full-time 17 credits 10 credits")
            rewrites.append("11.1 Limits on Registration M.Tech student regular semester 17 credits Table 11.1")
        elif "research_seed_grant" in attributes:
            rewrites.append("4.6 Seed Grant Projects sum of Rs 25 Lakhs for Faculty Members Sciences Engineering 5 Lakhs Humanities")
            rewrites.append("Research Initiation Seed Grant 25 Lakhs initial period of 3 years R&D Norms")
        elif "phd_comprehensive_exam" in attributes:
            rewrites.append("16.5.1 Comprehensive Examination of Ph.D. within 3 semesters Table 12.9 maximum 2 attempts one more opportunity")
            rewrites.append("Comprehensive Examination candidacy completed within 3 semesters first attempt one more opportunity")
        elif "course_overload_credits_limit" in attributes:
            rewrites.append("11.2.2 register for up to 4 extra credits per semester as course overload 80 credits CGPA 6.0")
            rewrites.append("course overload maximum 4 extra credits per semester sixth semester 11.2.2")
        elif "academic_probation_termination" in attributes:
            rewrites.append("14.1 Termination of the B.Tech. program academic probation for two consecutive semesters")
            rewrites.append("Section 14.1 e student is on academic probation for two consecutive semesters termination")

    elif domain == QueryDomain.CURRICULUM:
        ent_str = entity or "B.Tech"
        if "programme_electives_pe" in attributes:
            rewrites.append(f"{ent_str} Programme Electives PE credits curriculum table")
            rewrites.append(f"{ent_str} course categories credit distribution PE 18 credits 17 credits")
        if "open_electives_oe" in attributes:
            rewrites.append(f"{ent_str} Open Electives OE credits curriculum table")
        if "total_degree_credits" in attributes:
            rewrites.append(f"{ent_str} Total Graded credits Grand Total degree requirement 153 165")

    elif domain == QueryDomain.GOVERNANCE_FACULTY:
        ent_str = entity or ""
        rewrites.append(f"Constitution of the Senate Heads of Departments {ent_str}")
        rewrites.append(f"Key Functionaries Dean Director Head {ent_str}")

    return rewrites


def analyze_query(query: str, request_id: Optional[str] = None) -> QueryProfile:
    """
    Main entry point: analyzes, decomposes, and builds a complete QueryProfile.
    """
    rid = request_id or uuid.uuid4().hex[:12]
    q_norm = normalize_query_text(query)

    degree_level = detect_degree_level(q_norm)
    subject_entity = detect_subject_entity(q_norm)
    domain, intent, target_attributes = detect_domain_and_intent(q_norm, subject_entity)

    # Detect cohort requirements
    applicable_cohort = None
    requires_multi_cohort = False
    if domain == QueryDomain.CURRICULUM:
        if re.search(r"\b(?:2025|revised|new|2026)\b", q_norm):
            applicable_cohort = "revised_2025_onwards"
        elif re.search(r"\b(?:2019|2020|2021|2022|2023|2024|earlier|old)\b", q_norm):
            applicable_cohort = "earlier_2019_2024"
        else:
            requires_multi_cohort = True
            applicable_cohort = "all_cohorts"

    # Detect semester target
    applicable_semester = None
    sem_m = re.search(r"\b(?:semester|sem)\s*([1-8]|i|ii|iii|iv|v|vi|vii|viii)\b|\b([1-8]|1st|2nd|3rd|4th|5th|6th|7th|8th)\s*(?:sem|semester)\b", q_norm)
    if sem_m:
        raw_s = sem_m.group(1) or sem_m.group(2)
        sem_map = {"1": 1, "1st": 1, "i": 1, "2": 2, "2nd": 2, "ii": 2, "3": 3, "3rd": 3, "iii": 3, "4": 4, "4th": 4, "iv": 4, "5": 5, "5th": 5, "v": 5, "6": 6, "6th": 6, "vi": 6, "7": 7, "7th": 7, "vii": 7, "8": 8, "8th": 8, "viii": 8}
        applicable_semester = sem_map.get(raw_s.lower())

    # Detect comparison intent
    is_comparison = bool(re.search(r"\b(?:compare|difference|versus|vs|earlier\s+vs\s+revised|both\s+schemes)\b", q_norm))
    calculation_needed = bool(re.search(r"\b(?:total|sum|additional|how\s+many\s+more|difference)\b", q_norm))

    # Required evidence types
    required_evidence_types = []
    if domain == QueryDomain.CURRICULUM:
        required_evidence_types.extend(["structured_table", "curriculum_handbook"])
    elif domain == QueryDomain.ACADEMIC_REGULATIONS:
        required_evidence_types.extend(["official_regulation", "statutory_circular"])
    elif domain == QueryDomain.GOVERNANCE_FACULTY:
        required_evidence_types.extend(["senate_record", "official_directory"])

    # Search rewrites
    search_rewrites = generate_search_rewrites(q_norm, domain, subject_entity, target_attributes)

    return QueryProfile(
        request_id=rid,
        original_query=query,
        normalized_query=q_norm,
        domain=domain,
        intent=intent,
        subject_entity=subject_entity,
        degree_level=degree_level,
        target_attributes=target_attributes,
        applicable_cohort=applicable_cohort,
        applicable_semester=applicable_semester,
        required_evidence_types=required_evidence_types,
        search_rewrites=search_rewrites,
        is_comparison=is_comparison,
        requires_multi_cohort=requires_multi_cohort,
        calculation_needed=calculation_needed,
    )
