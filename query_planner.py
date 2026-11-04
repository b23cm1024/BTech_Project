"""
query_planner.py — Intelligent Query-Planning and Decomposition Engine
=====================================================================
Implements Section 7 of the Master Engineering Directive:
  - Parses each institutional query into a structured multi-part execution plan.
  - Extracts department, degree level, programme, academic year, admission cohort,
    regulation version, numerical attributes, and comparison dimensions.
  - Decomposes composite questions into independent sub-requirements so that
    no sub-question is silently omitted.
  - Strictly prevents intent hijacking (e.g. 'undergraduate' substring collision
    with graduation requirements, course repeat colliding with degree award).
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Set


class PlanDomain(str, Enum):
    ACADEMIC_REGULATIONS = "ACADEMIC_REGULATIONS"
    CURRICULUM = "CURRICULUM"
    RESEARCH_CONSULTANCY = "RESEARCH_CONSULTANCY"
    CAMPUS_FACILITIES = "CAMPUS_FACILITIES"
    GOVERNANCE_FACULTY = "GOVERNANCE_FACULTY"
    FEES_FINANCIAL = "FEES_FINANCIAL"
    GENERAL = "GENERAL"


@dataclass
class SubRequirement:
    requirement_id: str
    target_attribute: str
    description: str
    is_numerical: bool = False
    expected_unit: Optional[str] = None
    statutory_source_hint: Optional[str] = None


@dataclass
class QueryPlan:
    original_query: str
    normalized_query: str
    domain: PlanDomain
    primary_intent: str
    target_department: Optional[str] = None      # e.g. "CSE", "ME", "EE", "AIDE", "CE", "CHE"
    degree_level: str = "UG"                     # "UG", "PG", "DOCTORAL", "FACULTY", "ALL"
    programme: str = "B.Tech"                    # "B.Tech", "M.Tech", "Ph.D.", "ALL"
    admission_cohort: Optional[str] = None       # "2019-2024", "2025+", "ALL"
    regulation_version: Optional[str] = None     # "165_CREDIT", "153_CREDIT", "CIRCULAR_51"
    sub_requirements: List[SubRequirement] = field(default_factory=list)
    requires_multi_document: bool = False
    requires_historical_comparison: bool = False
    requested_attributes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "domain": self.domain.value,
            "primary_intent": self.primary_intent,
            "target_department": self.target_department,
            "degree_level": self.degree_level,
            "programme": self.programme,
            "sub_requirements": [r.__dict__ for r in self.sub_requirements],
            "requires_multi_document": self.requires_multi_document,
            "requires_historical_comparison": self.requires_historical_comparison
        }


class QueryPlanner:
    """Decomposes institutional questions into comprehensive verifiable plans."""

    DEPARTMENT_MAP = {
        "cse": "CSE",
        "computer science": "CSE",
        "computer science and engineering": "CSE",
        "computer science & engineering": "CSE",
        "me": "ME",
        "mechanical": "ME",
        "mechanical engineering": "ME",
        "ee": "EE",
        "electrical": "EE",
        "electrical engineering": "EE",
        "ece": "EE",
        "aide": "AIDE",
        "ai & ds": "AIDE",
        "ai and ds": "AIDE",
        "artificial intelligence": "AIDE",
        "data science": "AIDE",
        "civil": "CE",
        "civil engineering": "CE",
        "civil and infrastructure": "CE",
        "civil & infrastructure": "CE",
        "cie": "CE",
        "chemical": "CHE",
        "chemical engineering": "CHE",
        "che": "CHE",
        "bsbe": "BSBE",
        "bioengineering": "BSBE",
        "bioscience": "BSBE"
    }

    def plan_query(self, query: str) -> QueryPlan:
        q_clean = query.strip()
        q_lower = q_clean.lower()

        # 1. Detect target department strictly (prevent false triggers like 'me' pronoun)
        target_dept = self._detect_department(q_lower)

        # 2. Detect Degree Level & Programme
        degree_level = "UG"
        programme = "B.Tech"
        if re.search(r"\b(?:m\.?tech|postgraduate|pg)\b", q_lower):
            degree_level = "PG"
            programme = "M.Tech"
        elif re.search(r"\b(?:ph\.?d|phd|doctoral|doctorate)\b", q_lower):
            degree_level = "DOCTORAL"
            programme = "Ph.D."
        elif re.search(r"\b(?:faculty|professor|pi|co-pi|cpda|seed grant|relocation|honorarium|sabbatical)\b", q_lower):
            degree_level = "FACULTY"
            programme = "Faculty"
        elif re.search(r"\b(?:dual\s+degree|b\.?tech\s*[-+]\s*m\.?tech)\b", q_lower):
            degree_level = "UG_PG_DUAL"
            programme = "Dual Degree"

        # 3. Detect Admission Cohort (handling en-dashes, hyphens, and slashes)
        cohort = None
        if re.search(r"\b(?:2026[-–—/]?27|2026\s*cohort|admitted\s+in\s+2026)\b", q_lower):
            cohort = "2026-27"
        elif re.search(r"\b(?:2025[-–—/]?26|2025\s*cohort|admitted\s+in\s+2025)\b", q_lower):
            cohort = "2025-26"
        elif re.search(r"\b(?:earlier|2019[-–—/]?24|2019|2020|2021|2022|2023|2024)\b", q_lower):
            cohort = "2019-2024"

        # 4. Detect Domain & Sub-requirements
        plan = self._decompose_intent_and_requirements(q_clean, q_lower, target_dept, degree_level, programme)
        plan.admission_cohort = cohort
        return plan

    def _detect_department(self, q_lower: str) -> Optional[str]:
        # Check pronoun 'me' guard
        is_pronoun_me = bool(re.search(
            r"\b(?:tell|show|give|help|contact|send|email|between|let|ask|guide|inform|teach|advise|notify|call|to|with|from|for|by|about|of|in|at)\s+me\b",
            q_lower
        ))
        
        # Test specific multi-word and single-word department phrases
        for phrase, code in [
            ("artificial intelligence & data science", "AIDE"),
            ("artificial intelligence and data science", "AIDE"),
            ("artificial intelligence", "AIDE"),
            ("data science", "AIDE"),
            ("civil and infrastructure", "CE"),
            ("civil & infrastructure", "CE"),
            ("civil engineering", "CE"),
            ("computer science & engineering", "CSE"),
            ("computer science and engineering", "CSE"),
            ("computer science", "CSE"),
            ("mechanical engineering", "ME"),
            ("electrical engineering", "EE"),
            ("chemical engineering", "CHE"),
            ("bioscience and bioengineering", "BSBE"),
            ("bioscience & bioengineering", "BSBE"),
            ("bioengineering", "BSBE"),
            ("bioscience", "BSBE"),
            ("chemical", "CHE"),
            ("civil", "CE"),
            ("electrical", "EE"),
            ("mechanical", "ME")
        ]:
            if re.search(rf"\b{re.escape(phrase)}\b", q_lower):
                return code

        # Word boundary search for abbreviations
        for token, code in [
            ("aide", "AIDE"),
            ("cse", "CSE"),
            ("cie", "CE"),
            ("che", "CHE"),
            ("bsbe", "BSBE"),
            ("ee", "EE"),
            ("me", "ME")
        ]:
            if re.search(rf"\b{token}\b", q_lower):
                if token == "me" and is_pronoun_me:
                    continue
                return code

        return None

    def _decompose_intent_and_requirements(
        self,
        q_raw: str,
        q_lower: str,
        target_dept: Optional[str],
        degree_level: str,
        programme: str
    ) -> QueryPlan:
        sub_reqs: List[SubRequirement] = []
        domain = PlanDomain.GENERAL
        primary_intent = "FACTUAL"
        multi_doc = False
        hist_comp = False

        # --- A. ATTENDANCE POLICY DECOMPOSITION ---
        if (
            re.search(r"\b(?:attendance|deregistr\w*|de-registr\w*|75\s*%|minimum\s+attendance|mandatory\s+attendance|attendance\s+shortfall|shortfall\s+in\s+attendance|circular\s*51)\b", q_lower) and
            not any(k in q_lower for k in ["short leave", "maternity leave", "leave rule", "leave policy", "annual leave", "vacation entitlement"])
        ):
            domain = PlanDomain.ACADEMIC_REGULATIONS
            primary_intent = "ATTENDANCE_REGULATION"
            multi_doc = True
            hist_comp = True
            sub_reqs.append(SubRequirement(
                requirement_id="ATT_1",
                target_attribute="minimum_attendance_threshold",
                description="Minimum mandatory percentage required for examinations (75%)",
                is_numerical=True,
                expected_unit="%",
                statutory_source_hint="Academic Regulations Section 4.1.9 & Circular No. 51"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="ATT_2",
                target_attribute="consequence_shortfall",
                description="Consequence of attendance shortfall: Section 4.1.9 C-grade requirement vs Circular 51 administrative de-registration and semester withdrawal",
                statutory_source_hint="Academic Regulations Section 4.1.9 & Senate Circular No. 51 (Resolution 38.2.7)"
            ))
            if any(k in q_lower for k in ["medical", "leave", "sanction", "exempt", "grounds"]):
                sub_reqs.append(SubRequirement(
                    requirement_id="ATT_3",
                    target_attribute="medical_sanctioned_absence",
                    description="Treatment of approved medical and sanctioned absences",
                    statutory_source_hint="Academic Regulations Section 4.1.9 & Circular No. 51"
                ))

        # --- B. SUMMER TERM DECOMPOSITION ---
        elif re.search(r"\b(?:summer\s+term|summer\s+courses?|summer\s+semester)\b", q_lower):
            domain = PlanDomain.ACADEMIC_REGULATIONS
            primary_intent = "SUMMER_TERM_REGISTRATION"
            sub_reqs.append(SubRequirement(
                requirement_id="SUM_1",
                target_attribute="maximum_summer_credits",
                description="Maximum credit limit permitted during Summer Term (8 credits for UG under Section 12.1; M.Tech category-specific limits under Table 21.1: Executive accelerated 16, Normal/Sponsored/Regular 9, Slow-paced/Part-time 6)",
                is_numerical=True,
                expected_unit="credits",
                statutory_source_hint="Academic Regulations Section 12.1 (UG) / Table 21.1 (M.Tech)"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="SUM_2",
                target_attribute="summer_eligibility_categories",
                description="Eligible categories: academically deficient students, Major students, Minor students, students short of <= 8 credits for graduation, backlog students, and vacancies for advance completion",
                statutory_source_hint="Academic Regulations Section 12.1"
            ))

        # --- C. INCOMPLETE ('I') GRADE DECOMPOSITION ---
        elif "incomplete" in q_lower or re.search(r"\bgrade\s*['\"(]*i['\")]*\b|\b['\"(]*i['\")]*\s*grade\b", q_lower):
            domain = PlanDomain.ACADEMIC_REGULATIONS
            primary_intent = "INCOMPLETE_GRADE_POLICY"
            sub_reqs.append(SubRequirement(
                requirement_id="INC_1",
                target_attribute="incomplete_award_conditions",
                description="Conditions for awarding 'I' grade: student request and instructor approval due to absence on medical grounds or special extenuating circumstances; requested not later than last day of major tests",
                statutory_source_hint="Academic Regulations Section 5.1 & Table 1 row 3"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="INC_2",
                target_attribute="conversion_deadline",
                description="Conversion deadline: completed before the end of the first week of the next semester for conversion to regular grade (no automatic conversion to F)",
                statutory_source_hint="Academic Regulations Section 5.1 & Table 1 row 3"
            ))

        # --- D. COURSE REPETITION & GRADE TREATMENT (PRIORITY OVER GRADUATION) ---
        elif re.search(r"\b(?:fail(?:s|ed|ing)?\s+a?\s*(?:core\s+)?course|repeat(?:s|ed|ing)?\s+a?\s*(?:core\s+)?course|repeat(?:s|ed|ing)?\s+it|repeats|repeating|repeating\s+a\s+course|re-?taking|clear(?:ing)?\s+an?\s*['\"]?f['\"]?)\b", q_lower):
            domain = PlanDomain.ACADEMIC_REGULATIONS
            primary_intent = "COURSE_REPETITION_GRADE_TREATMENT"
            sub_reqs.append(SubRequirement(
                requirement_id="REP_1",
                target_attribute="sgpa_vs_cgpa_treatment",
                description="'E'/'F' grades are counted in SGPA but excluded from CGPA; CGPA calculation includes only pass grades (A* to D)",
                statutory_source_hint="Academic Regulations Section 5.3 & Table 1 row 2"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="REP_2",
                target_attribute="transcript_display_and_grade_replacement",
                description="Upon passing the repeated course, the new pass grade is included in CGPA; both original attempt and repeat attempt stay on the transcript/grade card",
                statutory_source_hint="Academic Regulations Section 5.3 & Section 5.4"
            ))

        # --- E. R&D OVERHEADS & CONSULTANCY DECOMPOSITION ---
        elif re.search(r"\b(?:overhead|consultancy|sponsored\s+project|deduction\s+rate|honorarium\s+ceiling)\b", q_lower):
            domain = PlanDomain.RESEARCH_CONSULTANCY
            primary_intent = "RD_OVERHEADS_AND_CONSULTANCY"
            sub_reqs.append(SubRequirement(
                requirement_id="RD_1",
                target_attribute="sponsored_research_overhead",
                description="Sponsored Research Projects overhead: 20% of Total Project Cost (0.2X)",
                is_numerical=True,
                expected_unit="%",
                statutory_source_hint="R&D Norms June 2018 Table 1"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="RD_2",
                target_attribute="consultancy_overhead",
                description="Consultancy Projects overhead: 30% of Consultancy Fee charged (excluding GST)",
                is_numerical=True,
                expected_unit="%",
                statutory_source_hint="R&D Norms June 2018 Clause 4.2(1) & Table 3"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="RD_3",
                target_attribute="faculty_honorarium_ceiling",
                description="Faculty honorarium ceiling: capped at maximum of two-thirds (66.7%) of net consultancy fee (excluding GST)",
                is_numerical=True,
                expected_unit="%",
                statutory_source_hint="R&D Norms June 2018 Clause 4.2(4)(a)"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="RD_4",
                target_attribute="internal_overhead_distribution",
                description="Internal distribution of overheads: 50% Institute Overhead Account, 35% R&D Overhead Account, 5% IPR Fund, 6% PI PDA, 4% Co-PI PDA",
                statutory_source_hint="R&D Norms June 2018 Table 2 & Table 4"
            ))

        # --- F. PRIMARY HEALTH CENTRE & EMERGENCY SERVICES ---
        elif re.search(r"\b(?:health\s+centre|health\s+center|phc|ambulance|medical\s+services|clinic|hospital)\b", q_lower) and not any(k in q_lower for k in ["casual leave", "vacation leave"]):
            domain = PlanDomain.CAMPUS_FACILITIES
            primary_intent = "CAMPUS_HEALTH_FACILITIES"
            sub_reqs.append(SubRequirement(
                requirement_id="HC_1",
                target_attribute="emergency_clinical_services",
                description="Primary Health Centre (PHC) provides 24x7 Medical Officers, 24x7 nursing, 24x7 emergency room, and 24x7 pharmacy",
                statutory_source_hint="Annual Report 2023-24 Section 10.2.4"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="HC_2",
                target_attribute="ambulance_equipment",
                description="Stationed ambulances: 01 Basic Life Support (BLS) + 01 Advanced Cardiac Life Support (ACLS) ambulance stationed 24x7",
                statutory_source_hint="Annual Report 2023-24 Section 10.2.4"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="HC_3",
                target_attribute="diagnostic_operating_hours",
                description="Diagnostic facilities: Pathology laboratory and Physiotherapy operate during scheduled clinic hours (not 24x7)",
                statutory_source_hint="Annual Report 2023-24 Section 10.2.4"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="HC_4",
                target_attribute="empanelled_hospitals",
                description="Empanelled tertiary care hospitals: Goyal Hospital & Research Centre, MediPulse Hospital, Kamla Nagar Hospital, ASG Eye Hospital, Vasundhara Hospital, AIIMS Jodhpur",
                statutory_source_hint="Annual Report 2023-24 Section 10.2.4"
            ))

        # --- G. CENTRES OF EXCELLENCE VS RESEARCH CENTRES ---
        elif re.search(r"\b(?:center\s+of\s+excellence|centers\s+of\s+excellence|centre\s+of\s+excellence|centres\s+of\s+excellence|coe|ayurtech|clean\s+energy)\b", q_lower):
            domain = PlanDomain.RESEARCH_CONSULTANCY
            primary_intent = "CENTRES_OF_EXCELLENCE"
            sub_reqs.append(SubRequirement(
                requirement_id="COE_1",
                target_attribute="designated_centres_of_excellence",
                description="Official Centres of Excellence (CoEs): AyurTech CoE (Integrative Medicine & Ayurvedic Tech), Center of Excellence in Clean Energy, Center of Excellence on Arts and Digital Immersion (ADI), Center of Excellence in Intelligent Infrastructure",
                statutory_source_hint="IIT Jodhpur Annual Report & Official Research Records"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="COE_2",
                target_attribute="interdisciplinary_platforms_distinction",
                description="Distinction from Interdisciplinary Research Platforms: Center for Space Technology, Center for Technology Foresight and Policy (CTFP), and Center for Digital Humanities operate as interdisciplinary research platforms, not standalone CoEs",
                statutory_source_hint="IIT Jodhpur Academic and Research Administration Records"
            ))

        # --- H. BRANCH CHANGE RULES (PRIORITIZED OVER GENERAL CGPA QUERIES) ---
        elif re.search(r"\b(?:branch[- ]+change|change\s+of\s+branch|switch\s+branch)\b", q_lower):
            domain = PlanDomain.ACADEMIC_REGULATIONS
            primary_intent = "BRANCH_CHANGE_CRITERIA"
            sub_reqs.append(SubRequirement(
                requirement_id="BC_1",
                target_attribute="branch_change_eligibility",
                description="Timing: after 1st year (Semester II); all prescribed credits completed; no backlog; no disciplinary action",
                statutory_source_hint="Academic Regulations Section 16 & AY 2025–26 Branch Change Circular"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="BC_2",
                target_attribute="branch_change_cgpa_cutoffs",
                description="AY 2025–26 CGPA cutoffs: >= 8.5 for other applicants (>= 7.5 for SC/ST/PwD); for Engineering Science >= 8.0 for other (>= 7.0 for SC/ST/PwD)",
                is_numerical=True,
                statutory_source_hint="Academic Regulations Section 16.2 & AY 2025–26 Circular"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="BC_3",
                target_attribute="branch_change_intake_limits",
                description="Intake limits: Upper limit of 110% of sanctioned strength and lower limit of 90% of current strength for the branch-change exercise (Engineering Science has separate 30 seats upper limit); discontinued from AY 2026–27",
                statutory_source_hint="Academic Regulations Section 16 & Senate Decisions"
            ))

        # --- I. GRADUATION REQUIREMENTS & MINIMUM CGPA (EXCLUDING BRANCH CHANGE & REPEAT) ---
        elif (
            (re.search(r"\bgraduat\w*\b", q_lower) and re.search(r"\b(?:cgpa|cpi|threshold|cutoff|passing\s+mark)\b", q_lower))
            or re.search(r"\b(?:minimum\s+cgpa|cgpa\s+required\s+(?:to|for)\s+graduat\w*|degree\s+award\s+criteria|award\s+of\s+(?:the\s+)?(?:b\.?tech|m\.?tech|degree))\b", q_lower)
        ) and not re.search(r"\b(?:repeat|repeating|fail(?:s|ed|ing)?\s+a?\s*core|branch\s+change|change\s+of\s+branch)\b", q_lower):
            domain = PlanDomain.ACADEMIC_REGULATIONS
            primary_intent = "GRADUATION_REQUIREMENTS"
            sub_reqs.append(SubRequirement(
                requirement_id="GRAD_1",
                target_attribute="minimum_graduating_cgpa",
                description="Minimum CGPA of 5.5 in passed courses to be eligible for award of degree (Section 10.2)",
                is_numerical=True,
                statutory_source_hint="Academic Regulations Section 10.2 & Section 17.1"
            ))

        # --- J. FACULTY RELOCATION REIMBURSEMENT ---
        elif re.search(r"\b(?:relocation|relocation\s+expenses?)\b", q_lower):
            domain = PlanDomain.RESEARCH_CONSULTANCY
            primary_intent = "RELOCATION_REIMBURSEMENT"
            sub_reqs.append(SubRequirement(
                requirement_id="RELOC_1",
                target_attribute="maximum_relocation_reimbursement",
                description="Up to Rs. 2,50,000/- for relocation from abroad (joining within 3 months, 3-year bond) and up to Rs. 1,50,000/- from other domestic institutions",
                is_numerical=True,
                expected_unit="Rs.",
                statutory_source_hint="Faculty Recruitment Advertisement & Relocation Guidelines"
            ))

        # --- K. ACADEMIC PROBATION & TERMINATION (SECTIONS 11.5, 11.5.1, 12.3 & 14.1) ---
        elif re.search(r"\b(?:academic\s+probation|probation|terminat(?:ed|ion)|termination\s+condition|academically\s+deficient|continuation\s+requirements?|restart)\b", q_lower) and not any(k in q_lower for k in ["summer"]):
            domain = PlanDomain.ACADEMIC_REGULATIONS
            primary_intent = "ACADEMIC_PROBATION_AND_TERMINATION"
            sub_reqs.append(SubRequirement(
                requirement_id="PROB_1",
                target_attribute="first_year_continuation_thresholds",
                description="First-two-semester continuation: 18 earned credits for GEN/OBC/GEN-EWS and 14 earned credits for SC/ST/PwD by end of second registered semester (Table 11.3). Failure permits one-time restart; otherwise registration is terminated.",
                statutory_source_hint="Academic Regulations Section 11.5 & Table 11.3"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="PROB_2",
                target_attribute="third_semester_onward_probation_triggers",
                description="Third semester onwards (Section 11.5.1): Earning fewer than 12 credits in a registered semester places student on Academic Probation. Earning 15 or fewer credits triggers an academic warning and parental intimation. Two consecutive probation semesters lead to termination.",
                statutory_source_hint="Academic Regulations Section 11.5.1"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="PROB_3",
                target_attribute="probation_registration_limit",
                description="Academic Probation credit cap: Maximum 16 credits per semester under Section 12.3",
                is_numerical=True,
                expected_unit="credits",
                statutory_source_hint="Academic Regulations Section 12.3"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="PROB_4",
                target_attribute="termination_triggers",
                description="Five Termination triggers under Section 14.1: 1. Unauthorised absence >= 8 weeks, 2. Failure to register by deadline without reason, 3. Conduct violation, 4. Failure to earn minimum credits in maximum duration, 5. Academic probation for two consecutive semesters",
                statutory_source_hint="Academic Regulations Section 14.1"
            ))

        # --- L. MINOR & SPECIALIZATION CREDIT REQUIREMENTS (SECTION 11.2) ---
        elif re.search(r"\b(?:minor|specialization)\b", q_lower) and re.search(r"\b(?:credit|credits|additional|overload|beyond|course\s+overload)\b", q_lower):
            domain = PlanDomain.ACADEMIC_REGULATIONS
            primary_intent = "MINOR_SPECIALIZATION_CREDITS"
            sub_reqs.append(SubRequirement(
                requirement_id="MIN_1",
                target_attribute="minor_total_credits",
                description="Total credits in Minor discipline: 20 credits required under Section 11.2",
                is_numerical=True,
                expected_unit="credits",
                statutory_source_hint="Academic Regulations Section 11.2"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="MIN_2",
                target_attribute="minor_credit_breakdown",
                description="Credit breakdown: 10 credits completed through Open Electives (OE) + 10 additional credits earned through course overload beyond normal curriculum",
                is_numerical=True,
                expected_unit="credits",
                statutory_source_hint="Academic Regulations Section 11.2"
            ))

        # --- M. M.TECH TEACHING ASSISTANTSHIP (TA) HOURS ---
        elif re.search(r"\b(?:teaching\s+assistantship|assistantship\s+work|research\s+assistantship|ta\s+duties?|assistantship\s+hours?|fellowship\s+hours?|hours\s+(?:per|a)\s+week)\b", q_lower) or ("m.tech" in q_lower and "assistantship" in q_lower):
            domain = PlanDomain.ACADEMIC_REGULATIONS
            primary_intent = "MTECH_TA_ASSISTANTSHIP_HOURS"
            sub_reqs.append(SubRequirement(
                requirement_id="TA_1",
                target_attribute="mandatory_ta_hours",
                description="Mandatory workload: 8 hours per week of Teaching or Research Assistantship duties for regular full-time M.Tech students receiving Institute fellowship / monthly stipend",
                is_numerical=True,
                expected_unit="hours/week",
                statutory_source_hint="IIT Jodhpur M.Tech Admission Advertisement & Assistantship Guidelines"
            ))

        # --- N. PH.D. COMPREHENSIVE EXAMINATION & CONSEQUENCES ---
        elif re.search(r"\b(?:comprehensive\s+exam(?:ination)?|ph\.?d\s+comprehensive|second\s+attempt|qualifier\s+requirement)\b", q_lower):
            domain = PlanDomain.ACADEMIC_REGULATIONS
            primary_intent = "PHD_COMPREHENSIVE_EXAMINATION"
            sub_reqs.append(SubRequirement(
                requirement_id="PHD_1",
                target_attribute="comprehensive_exam_timeline",
                description="Timeline: Full-time Ph.D. scholars must pass comprehensive examination within 3 semesters; first unsatisfactory attempt permits one additional attempt within 6 months; student must perform satisfactorily in all papers",
                statutory_source_hint="Postgraduate Academic Regulations Section 26.5"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="PHD_2",
                target_attribute="qualifier_and_cancellation_clauses",
                description="Related consequences: Qualifier requirements normally completed before 5th semester (no later than 6th semester); Section 26.5.1 does not prescribe automatic termination solely for failing second attempt, but under Section 26.8 registration is liable for cancellation if SRC finds student's progress unsatisfactory",
                statutory_source_hint="Postgraduate Academic Regulations Section 26.8"
            ))

        # --- O. FACULTY SABBATICAL LEAVE (FIRST STATUTES CLAUSE 27) ---
        elif re.search(r"\b(?:sabbatical|sabbatical\s+leave)\b", q_lower):
            domain = PlanDomain.RESEARCH_CONSULTANCY
            primary_intent = "FACULTY_SABBATICAL_LEAVE"
            sub_reqs.append(SubRequirement(
                requirement_id="SABB_1",
                target_attribute="sabbatical_continuous_service",
                description="Eligibility: Six or more years of continuous regular service with the Institute (Clause 27(1))",
                is_numerical=True,
                expected_unit="years",
                statutory_source_hint="IIT Jodhpur First Statutes Clause 27"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="SABB_2",
                target_attribute="deputation_reset_and_lifetime_cap",
                description="Deputation condition: If special leave or deputation of one year or more is taken, six years of service must be completed after returning; sabbatical leave cannot exceed three occasions during entire service (Clause 27(2) & 27(3))",
                statutory_source_hint="IIT Jodhpur First Statutes Clause 27"
            ))

        # --- P. AUDIT COURSE REGULATIONS & NOMENCLATURE (SECTION 4.1.8, 5.1 & 5.3) ---
        elif re.search(r"\b(?:audit\s+course|audit\s+pass|audit\s+grade|['\"]?ap['\"]?\s+grade|auditing\s+of\s+courses)\b", q_lower) or ("audit" in q_lower and any(k in q_lower for k in ["pass", "grade", "sgpa", "cgpa", "satisfactory"])):
            domain = PlanDomain.ACADEMIC_REGULATIONS
            primary_intent = "AUDIT_COURSE_REGULATIONS"
            sub_reqs.append(SubRequirement(
                requirement_id="AUD_1",
                target_attribute="official_audit_nomenclature",
                description="IIT Jodhpur official nomenclature uses letter grade 'S' (Satisfactory), NOT 'AP' (Audit Pass), for satisfactory performance in audit courses; 'U' is awarded if requirements are not fulfilled",
                statutory_source_hint="Academic Regulations Section 5.1 & Table 1"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="AUD_2",
                target_attribute="sgpa_cgpa_exclusion",
                description="Grades obtained in audit courses ('S' or 'U') are completely excluded from both SGPA and CGPA calculations (Section 5.3)",
                statutory_source_hint="Academic Regulations Section 5.3"
            ))

        # --- Q. FORMAL GRADE APPEAL & CORRECTION PROCEDURE (SECTION 5.2) ---
        elif (
            re.search(r"\b(?:appeal|reconsideration|correct(?:ion)?|change|revision|revise|alter)\b.*\bgrades?\b", q_lower) or
            re.search(r"\bgrades?\b.*\b(?:appeal|reconsideration|correct(?:ion)?|change|revision|revise|alter)\b", q_lower) or
            re.search(r"\b(?:grade\s+appeal|grade\s+correction|declared\s+(?:course\s+)?grade|approval\s+from\s+(?:the\s+)?head\s+of\s+department|48\s*hours?)\b", q_lower)
        ):
            domain = PlanDomain.ACADEMIC_REGULATIONS
            primary_intent = "GRADE_APPEAL_PROCEDURE"
            sub_reqs.append(SubRequirement(
                requirement_id="GAP_1",
                target_attribute="grade_correction_48_hours",
                description="Grade correction: Allowed within 48 hours of display of grades after approval of the Head of the Department (Step 7)",
                is_numerical=True,
                expected_unit="hours",
                statutory_source_hint="Academic Regulations Section 5.2 Step 7"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="GAP_2",
                target_attribute="exceptional_senate_appeal",
                description="Finality and Exceptional Appeal: Results once approved cannot ordinarily be changed; in exceptional cases, student may appeal to Chairman, Senate for reconsideration with formal justification (Step 11; no separate explicit deadline)",
                statutory_source_hint="Academic Regulations Section 5.2 Step 11"
            ))

        # --- R1. SEMESTER WITHDRAWAL & COMPARISON WITH COURSE WITHDRAWAL ---
        elif re.search(r"\b(?:semester\s+withdrawal|withdraw\s+(?:from\s+)?(?:a\s+)?semester|withdrawal\s+from\s+semester)\b", q_lower) or ("withdrawal" in q_lower and "semester" in q_lower):
            domain = PlanDomain.ACADEMIC_REGULATIONS
            primary_intent = "SEMESTER_WITHDRAWAL_POLICY"
            sub_reqs.append(SubRequirement(
                requirement_id="SW_1",
                target_attribute="ug_semester_withdrawal",
                description="Undergraduate Semester Withdrawal: Section 15.3 unforeseen circumstances / medical grounds; absence >= 4 weeks results in automatic semester withdrawal; DUGC approval; extension beyond one semester requires Academic Committee approval; medical withdrawal not counted towards max programme duration",
                statutory_source_hint="Academic Regulations Section 15.3"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="SW_2",
                target_attribute="pg_semester_withdrawal_fellowship_stopped",
                description="Postgraduate Semester Withdrawal: Section 23.3 medical/unforeseen grounds; absence >= 4 weeks causes automatic withdrawal; DRC approval; fellowship STOPPED immediately and resumed only after rejoining; extension requires Academic Committee; medical withdrawal not counted towards max duration",
                statutory_source_hint="Academic Regulations Section 23.3"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="SW_3",
                target_attribute="course_vs_semester_withdrawal_distinction",
                description="Contrast with Course Withdrawal: Course withdrawal (Section 3.e) applies to individual courses within 4 weeks resulting in 'W' grade with no CGPA effect; Semester withdrawal (Section 15.3/23.3) withdraws entire semester, stops PG fellowship, and exempts medical duration",
                statutory_source_hint="Academic Regulations Section 3.e vs Section 15.3 & 23.3"
            ))

        # --- R2. ACADEMIC LEAVE REGULATIONS (UG SECTION 15.4 & PG SECTION 23.4) ---
        elif re.search(r"\b(?:academic\s+leave|leave\s+for\s+fieldwork|leave\s+for\s+library\s+work|exchange\s+leave)\b", q_lower) or ("academic" in q_lower and "leave" in q_lower and not any(k in q_lower for k in ["sabbatical", "casual leave", "vacation"])):
            domain = PlanDomain.ACADEMIC_REGULATIONS
            primary_intent = "ACADEMIC_LEAVE_POLICY"
            sub_reqs.append(SubRequirement(
                requirement_id="AL_1",
                target_attribute="academic_leave_eligibility_and_purposes",
                description="Academic Leave: Permitted for fieldwork, library work at specialized external libraries, experimental work at external laboratories, or exchange coursework/research; requires DUGC/DRC recommendation and Office of Academics approval",
                statutory_source_hint="Academic Regulations Section 15.4 & Section 23.4"
            ))

        # --- R2b. STUDENT LEAVE RULES & POLICY (UG SECTION 15 & PG SECTION 23) ---
        elif (
            re.search(r"\b(?:leaves?\s+rules?|leaves?\s+policy|short\s+leaves?|medical\s+leaves?|maternity\s+leaves?|annual\s+leaves?|leaves?\s+quota|leaves?\s+of\s+absence|mid[- ]semester\s+recess|vacation\s+(?:entitlement|policy|rules?)|apply\s+(?:for\s+)?leaves?|days?\s+(?:of\s+)?leaves?|tak(?:e|ing)\s+leaves?|avail(?:ing)?\s+leaves?|sanction(?:ed)?\s+leaves?)\b", q_lower) or
            (("leave" in q_lower or "leaves" in q_lower or "vacation" in q_lower) and any(k in q_lower for k in ["student", "phd", "mtech", "btech", "doctorate", "scholar", "undergraduate", "postgraduate", "medical", "dugc", "drc", "rules", "quota", "days", "stretch", "avail", "take", "permitted", "allowed"]))
        ) and not any(k in q_lower for k in ["sabbatical", "semester withdrawal", "withdraw from semester"]):
            domain = PlanDomain.ACADEMIC_REGULATIONS
            primary_intent = "STUDENT_LEAVE_POLICY"
            sub_reqs.append(SubRequirement(
                requirement_id="LEAVE_UG",
                target_attribute="undergraduate_leave_rules",
                description="Undergraduate Leave Rules (Section 15): Short leave up to 10 days on medical grounds and 5 days for other valid reasons in a semester (Section 15.2); entitlement to mid-semester recess and vacation as per Academic Calendar (Section 15.1); application addressed to Convener DUGC with medical certificate.",
                statutory_source_hint="Academic Regulations Section 15 (UG Programmes)"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="LEAVE_PG",
                target_attribute="postgraduate_leave_rules",
                description="Postgraduate Leave Rules (Section 23): Maximum 30 days in an academic year including medical leave and mid-semester break (Section 23.1); Ph.D. max 15 days at a stretch; M.Tech/M.Sc. max 10 days at a stretch; leave beyond 30 days granted without assistantship/fellowship; application to Convener DRC.",
                statutory_source_hint="Academic Regulations Section 23 (PG Programmes)"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="LEAVE_MATERNITY",
                target_attribute="maternity_leave_policy",
                description="Maternity Leave (Section 23.2): Sanctioned as per MHRD/MoE directives for Ph.D. students and is NOT counted towards the maximum permissible duration of the programme.",
                statutory_source_hint="Academic Regulations Section 23.2"
            ))

        # --- R3. COURSE DROP VS COURSE WITHDRAWAL ('W' GRADE) ---
        elif re.search(r"\b(?:course\s+drop|drop\s+a\s+course|drop\s+courses?|withdrawing\s+from\s+a\s+course|course\s+withdrawal|withdraw\s+a\s+course|['\"]?w['\"]?\s+grade|grade\s+['\"]?w['\"]?)\b", q_lower) or ("drop" in q_lower and "withdrawal" in q_lower):
            domain = PlanDomain.ACADEMIC_REGULATIONS
            primary_intent = "COURSE_DROP_AND_WITHDRAWAL"
            sub_reqs.append(SubRequirement(
                requirement_id="DRP_1",
                target_attribute="course_drop_rules",
                description="Course Drop: Allowed till last date of registration (add/drop window) with Faculty Advisor approval, min 12 credits maintained; does NOT appear on grade card/transcript",
                statutory_source_hint="Academic Regulations Section 3.d"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="DRP_2",
                target_attribute="course_withdrawal_w_grade",
                description="Course Withdrawal: Permitted within 4 weeks from beginning of semester; recorded on grade card with letter grade 'W'; carries no grade points and no CGPA impact",
                statutory_source_hint="Academic Regulations Section 3.e"
            ))

        # --- S. NON-GRADED EXTRACURRICULAR REQUIREMENTS ---
        elif re.search(r"\b(?:ncc|nss|nso|extra[- ]curricular|non[- ]graded|social\s+connect)\b", q_lower):
            domain = PlanDomain.ACADEMIC_REGULATIONS
            primary_intent = "EXTRACURRICULAR_NON_GRADED"
            sub_reqs.append(SubRequirement(
                requirement_id="NG_1",
                target_attribute="mandatory_extracurricular_semesters",
                description="Mandatory completion of at least two semesters of approved extra-curricular activities (NCC/NSS/NSO)",
                is_numerical=True,
                expected_unit="semesters",
                statutory_source_hint="Academic Regulations Section 17.5"
            ))

        # --- T. CAMPUS VEHICLE & PARKING PERMITS ---
        elif re.search(r"\b(?:parking|vehicle|car|four[- ]wheeler|two[- ]wheeler)\b", q_lower):
            domain = PlanDomain.GENERAL
            primary_intent = "CAMPUS_PARKING_PERMIT"
            sub_reqs.append(SubRequirement(
                requirement_id="PARK_1",
                target_attribute="parking_permit_regulations",
                description="Campus vehicle parking permits, quotas, and fees: managed by Campus Security & Estate Section; not indexed in academic regulations corpus",
                statutory_source_hint="Campus Security & Estate Administration Guidelines"
            ))

        # --- U. HOSTEL ROOM ALLOTMENT & INTRANET NOTICES ---
        elif any(k in q_lower for k in ["room allotment", "allotment list", "intranet"]) or ("hostel" in q_lower and any(k in q_lower for k in ["room", "allotment", "intranet", "warden"])):
            domain = PlanDomain.GENERAL
            primary_intent = "HOSTEL_INTRANET_QUERY"
            sub_reqs.append(SubRequirement(
                requirement_id="HOSTEL_1",
                target_attribute="internal_hostel_records_status",
                description="Internal hostel room allotment records and intranet circulars are administered by Office of Student Affairs and not present in academic corpus",
                statutory_source_hint="Office of Student Affairs"
            ))

        # --- V. HEADS OF DEPARTMENTS, SCHOOLS & DEANS (SENATE CONSTITUTION) ---
        elif (
            re.search(r"\b(?:head\s+of\s+(?:the\s+)?department|hod|department\s+head|school\s+head|head\s+of\s+(?:the\s+)?school|head\s+of\s+dept|who\s+is\s+(?:the\s+)?head|dean\s+of|director\s+of|who\s+is\s+(?:the\s+)?dean|who\s+is\s+(?:the\s+)?director)\b", q_lower) and
            not re.search(r"\b(?:approval\s+from\s+(?:the\s+)?head\s+of\s+department\s+to\s+correct|grade\s+appeal|grade\s+correction)\b", q_lower)
        ):
            domain = PlanDomain.GOVERNANCE_FACULTY
            primary_intent = "DEPARTMENT_HEAD_DIRECTORY"
            sub_reqs.append(SubRequirement(
                requirement_id="HOD_1",
                target_attribute="appointed_head_or_dean",
                description="Authoritative appointed Head of Department, School Head, or Dean designated in the Constitution of the Senate, IIT Jodhpur (Updated 05.01.2026)",
                statutory_source_hint="Constitution of the Senate (IIT Jodhpur Academic Governance) Updated 05.01.2026"
            ))

        # --- M. CURRICULUM CREDITS & DEGREE REQUIREMENTS ---
        elif (target_dept and re.search(r"\b(?:curriculum|\bpe\b|\boe\b|\bpc\b|programme\s+elective|open\s+elective|programme\s+compulsory|graded\s+credits|total\s+credits|degree\s+credits|b\.?tech\s+credits|credits?|courses?|requirements?|structure|audit|core)\b", q_lower)) or re.search(r"\b(?:curriculum|\bpe\b|\boe\b|programme\s+elective|open\s+elective|graded\s+credits|total\s+credits|degree\s+credits|b\.?tech\s+credits)\b", q_lower):
            domain = PlanDomain.CURRICULUM
            primary_intent = "CURRICULUM_REQUIREMENTS"
            multi_doc = True
            hist_comp = True
            dept_name = target_dept or "All Engineering"
            sub_reqs.append(SubRequirement(
                requirement_id="CURR_1",
                target_attribute="earlier_scheme_credits",
                description=f"Earlier 165-credit scheme for {dept_name}: Graded credits, PE credits, OE credits, non-graded/audit credits",
                is_numerical=True,
                expected_unit="credits",
                statutory_source_hint=f"B.Tech {dept_name} Earlier Curriculum (2019-2024)"
            ))
            sub_reqs.append(SubRequirement(
                requirement_id="CURR_2",
                target_attribute="revised_scheme_credits",
                description=f"Revised 153/154-credit scheme for {dept_name} (2025 onwards): Graded credits, PE, OE, essential audit credits",
                is_numerical=True,
                expected_unit="credits",
                statutory_source_hint=f"B.Tech {dept_name} Revised Curriculum (2025 Onwards)"
            ))

        # Default fallback sub-requirement
        if not sub_reqs:
            sub_reqs.append(SubRequirement(
                requirement_id="GEN_1",
                target_attribute="general_inquiry",
                description="Extract factual institutional answer with supporting evidence citations",
                statutory_source_hint="Relevant IIT Jodhpur Official Documents"
            ))

        return QueryPlan(
            original_query=q_raw,
            normalized_query=q_lower,
            domain=domain,
            primary_intent=primary_intent,
            target_department=target_dept,
            degree_level=degree_level,
            programme=programme,
            sub_requirements=sub_reqs,
            requires_multi_document=multi_doc,
            requires_historical_comparison=hist_comp,
            requested_attributes=[r.target_attribute for r in sub_reqs]
        )


query_planner = QueryPlanner()
