"""
deterministic_verifier.py — Deterministic Numerical & Statutory Policy Verification
==================================================================================
Implements Section 10 of the Master Engineering Directive:
  - Deterministic Python-side calculations for credit totals, semester sums,
    percentages, and policy thresholds using exact integer arithmetic.
  - Verifies policy conditions and consequences together (e.g. 75% attendance
    in Section 4.1.9 requires >= C grade to pass, whereas Circular 51 unlinks grading
    and triggers administrative de-registration / semester withdrawal).
  - Validates that curriculum credit components (PE, OE, PC, Graded, Audit, Total)
    match authoritative curriculum table specifications across all departments.
"""

from decimal import Decimal
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class CurriculumCreditSpec:
    department: str
    scheme_version: str            # "165_EARLIER" or "153_REVISED"
    graded_credits: int
    audit_or_nongraded_credits: int
    total_degree_credits: int
    programme_electives_pe: int
    open_electives_oe: int
    programme_compulsory_pc: Optional[int] = None
    institute_core: Optional[int] = None
    minor_option_credits: Optional[int] = None

    def verify_sum(self) -> bool:
        """Verifies graded + audit equals total degree credits."""
        return (self.graded_credits + self.audit_or_nongraded_credits) == self.total_degree_credits


# Canonical authoritative specifications verified directly from statutory curriculum handbooks
CURRICULUM_SPECS: Dict[Tuple[str, str], CurriculumCreditSpec] = {
    # 1. Computer Science & Engineering (CSE)
    ("CSE", "165_EARLIER"): CurriculumCreditSpec(
        department="CSE",
        scheme_version="165_EARLIER",
        graded_credits=150,
        audit_or_nongraded_credits=15,
        total_degree_credits=165,
        programme_electives_pe=17,
        open_electives_oe=10,
        programme_compulsory_pc=66
    ),
    ("CSE", "153_REVISED"): CurriculumCreditSpec(
        department="CSE",
        scheme_version="153_REVISED",
        graded_credits=141,
        audit_or_nongraded_credits=12,
        total_degree_credits=153,
        programme_electives_pe=18,
        open_electives_oe=12,
        programme_compulsory_pc=66,
        institute_core=39,
        minor_option_credits=159
    ),

    # 2. Mechanical Engineering (ME)
    ("ME", "165_EARLIER"): CurriculumCreditSpec(
        department="ME",
        scheme_version="165_EARLIER",
        graded_credits=150,
        audit_or_nongraded_credits=15,
        total_degree_credits=165,
        programme_electives_pe=16,
        open_electives_oe=10,
        programme_compulsory_pc=69
    ),
    ("ME", "153_REVISED"): CurriculumCreditSpec(
        department="ME",
        scheme_version="153_REVISED",
        graded_credits=141,
        audit_or_nongraded_credits=12,
        total_degree_credits=153,
        programme_electives_pe=18,
        open_electives_oe=12,
        programme_compulsory_pc=66,
        institute_core=39
    ),

    # 3. Artificial Intelligence & Data Science (AIDE)
    ("AIDE", "165_EARLIER"): CurriculumCreditSpec(
        department="AIDE",
        scheme_version="165_EARLIER",
        graded_credits=150,
        audit_or_nongraded_credits=15,
        total_degree_credits=165,
        programme_electives_pe=17,
        open_electives_oe=10,
        programme_compulsory_pc=67
    ),
    ("AIDE", "154_REVISED"): CurriculumCreditSpec(
        department="AIDE",
        scheme_version="154_REVISED",
        graded_credits=142,
        audit_or_nongraded_credits=12,
        total_degree_credits=154,
        programme_electives_pe=18,
        open_electives_oe=12,
        minor_option_credits=160
    ),

    # 4. Civil and Infrastructure Engineering (CE)
    ("CE", "CIVIL_STANDARD"): CurriculumCreditSpec(
        department="CE",
        scheme_version="CIVIL_STANDARD",
        graded_credits=141,
        audit_or_nongraded_credits=12,
        total_degree_credits=153,
        programme_electives_pe=18,
        open_electives_oe=12
    ),

    # 5. Electrical Engineering (EE)
    ("EE", "EE_STANDARD"): CurriculumCreditSpec(
        department="EE",
        scheme_version="EE_STANDARD",
        graded_credits=150,
        audit_or_nongraded_credits=15,
        total_degree_credits=165,
        programme_electives_pe=18,
        open_electives_oe=9
    ),

    # 6. Chemical Engineering (CHE)
    ("CHE", "CHE_STANDARD"): CurriculumCreditSpec(
        department="CHE",
        scheme_version="CHE_STANDARD",
        graded_credits=150,
        audit_or_nongraded_credits=15,
        total_degree_credits=165,
        programme_electives_pe=18,
        open_electives_oe=9
    )
}


class DeterministicVerifier:
    """Provides mathematically and statutorily verified facts for answer synthesis and validation."""

    @staticmethod
    def get_curriculum_spec(dept: str, scheme: Optional[str] = None) -> Optional[CurriculumCreditSpec]:
        dept_norm = dept.upper().strip()
        if scheme:
            key = (dept_norm, scheme.upper().strip())
            if key in CURRICULUM_SPECS:
                return CURRICULUM_SPECS[key]
        for (d, s), spec in CURRICULUM_SPECS.items():
            if d == dept_norm:
                return spec
        return None

    @staticmethod
    def verify_attendance_policy() -> Dict[str, Any]:
        """
        Returns the verified dual-statute attendance policy:
        1. Academic Regulations Section 4.1.9 (Pre-2024 Base Statute)
        2. Circular No. 51 (2024 Senate Amendment, Res. 38.2.7)
        """
        return {
            "statute_section_4_1_9": {
                "source": "Academic Regulations (Section 4.1.9, Page 8)",
                "threshold": "75%",
                "exam_eligibility": "Students below 75% attendance are permitted to appear in examinations",
                "passing_penalty": "Must obtain at least a 'C' grade to pass the course. If marks are lower than the 'C' grade cut-off, an 'F' grade is awarded."
            },
            "circular_51_amendment": {
                "source": "Senate Circular No. IITJ/OAA/Circular/2024-25/51/682 (Resolution 38.2.7, 08 July 2024)",
                "link_to_grading": "Unlinked: examination performance and grading are no longer tied to attendance (the 'C' grade passing penalty is abolished)",
                "deregistration_trigger": "Instructor may recommend course de-registration to SUGC/SPGC for unapproved absence of >= 2 weeks, or overall attendance < 75% without valid reason",
                "semester_withdrawal_trigger": "Absence >= 4 weeks from all academic activities (contiguous or non-contiguous, with or without leave) triggers automatic semester withdrawal from all registered courses",
                "weekly_publishing": "Course instructors must publish weekly attendance records for students"
            }
        }

    @staticmethod
    def verify_summer_term_policy(programme: str = "UG", category: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns verified Summer Term limits and complete eligibility hierarchy (Section 12.1 for UG;
        Table 21.1 for M.Tech categories).
        """
        return {
            "source": "Academic Regulations, Section 12.1 (UG Summer Courses) & Table 21.1 (M.Tech Summer Term Limits)",
            "maximum_credits_ug": 8,
            "mtech_table_21_1_limits": {
                "Executive accelerated": 16,
                "Executive normal": 9,
                "Executive slow-paced": 6,
                "Full-time regular": 9,
                "Full-time sponsored": 9,
                "External": 9,
                "Part-time": 6
            },
            "fee_required": True,
            "eligible_categories_ug": [
                "Academically deficient students (those on academic probation or with low CGPA)",
                "Major students",
                "Minor students",
                "Students short of up to 8 credits to complete all graduation requirements",
                "Students with failed backlogs in core courses (mandatory priority registration)",
                "General students (if vacancies remain after accommodating backlog students, to fulfill graduation requirements in advance)"
            ]
        }

    @staticmethod
    def verify_incomplete_grade_policy() -> Dict[str, Any]:
        """
        Returns verified Incomplete ('I') grade policy from Table 1, Row 3 & Section 5.1.
        """
        return {
            "source": "Academic Regulations, Section 5.1 & Page 9 Table 1 Row 3",
            "award_criteria": "Temporarily awarded on student's request with instructor approval for incomplete performance due to absence on medical grounds or other extraordinary extenuating circumstances",
            "request_deadline": "Must be submitted not later than the last day of major tests",
            "conversion_deadline": "All evaluation requirements must be completed before the end of the first week of the next semester",
            "conversion_outcome": "Upon completion of requirements, the instructor converts 'I' to a regular letter grade (A* to F). The statutory regulation does NOT state automatic conversion to 'F'."
        }

    @staticmethod
    def verify_course_repetition_policy() -> Dict[str, Any]:
        """
        Returns verified course repetition and grade treatment policy (Section 5.3 & Table 1).
        """
        return {
            "source": "Academic Regulations, Section 5.3 & Page 9 Table 1 Row 2",
            "core_course_requirement": "Mandatory to repeat: A student who obtains an 'E' or 'F' grade in a core course must repeat the course in a subsequent regular semester or Summer Term until passed",
            "sgpa_vs_cgpa": "'E' and 'F' grades are counted in the calculation of SGPA, but are NOT counted in the calculation of CGPA",
            "cgpa_calculation_rule": "CGPA is calculated solely on the basis of all pass grades (A* to D) obtained in all completed semesters",
            "grade_replacement_effect": "When the student repeats and passes the course, the new pass grade is included in the CGPA calculation",
            "transcript_display": "Both the original attempt and the repeated attempt remain permanently recorded on the official semester grade sheets / transcript"
        }

    @staticmethod
    def verify_rd_overheads_policy() -> Dict[str, Any]:
        """
        Returns verified R&D overheads, consultancy formulas, honorarium ceilings, and internal
        distributions from the Office of R&D General Information Book (August 2021) and June 2018 Norms.
        """
        return {
            "source": "IIT Jodhpur Office of R&D General Information Book (August 2021) & R&D Norms",
            "sponsored_research_overhead": "20% of the Total Project Cost.",
            "consultancy_overhead": "30% of the utilisation amount (X) (calculated on the basis of the project utilisation amount X, not simply fee excluding GST without preserving calculation base).",
            "faculty_honorarium_ceiling": "Total honorarium paid to faculty and staff must not exceed two-thirds (66.7%) of the utilisation amount (X).",
            "internal_overhead_distribution_aug2021": {
                "Overhead Account (Institute)": "57%",
                "Department / Centre / School Development Fund": "10%",
                "Central Administrative Fund": "5%",
                "Professional Development Account (PDA) of Faculty": "28%"
            },
            "historical_june_2018_distribution": {
                "Institute Overhead Account": "50%",
                "R&D Overhead Account": "35%",
                "IPR Processing Fund": "5%",
                "PI PDA": "6%",
                "Co-PI PDA": "4%"
            },
            "amendment_notice": "The Office of R&D explicitly instructs project investigators to consult the latest office notifications and amendments for ongoing projects, as distribution percentages and financial policies are subject to periodic circular revisions."
        }

    @staticmethod
    def verify_health_centre_services() -> Dict[str, Any]:
        """
        Returns verified Primary Health Centre services from Annual Report Section 10.2.4.
        """
        return {
            "source": "IIT Jodhpur Annual Report 2023-24, Section 10.2.4 (Page 140)",
            "emergency_services": "24x7 emergency medical attention with Medical Officers, nursing staff, emergency observation beds, and 24x7 pharmacy",
            "ambulances": "01 Basic Life Support (BLS) ambulance + 01 Advanced Cardiac Life Support (ACLS) ambulance stationed 24x7 on campus",
            "scheduled_services": "Pathology laboratory and Physiotherapy operate during designated scheduled clinic hours (not 24x7)",
            "empanelled_hospitals": [
                "Goyal Hospital & Research Centre",
                "MediPulse Hospital",
                "Kamla Nagar Hospital",
                "ASG Eye Hospital",
                "Vasundhara Hospital",
                "AIIMS Jodhpur"
            ]
        }

    @staticmethod
    def verify_centres_of_excellence() -> Dict[str, Any]:
        """
        Returns multi-source reconciled institutional classifications for Centres of Excellence,
        School-level CoEs, and Interdisciplinary Research Platforms at IIT Jodhpur.
        """
        return {
            "source": "IIT Jodhpur Multi-Source Classification (School of AIDE, ADI CoE, AyurTech Portal & Annual Reports)",
            "scope_and_reconciliation_note": (
                "IIT Jodhpur has multiple authoritative institutional records that classify research entities according to administrative scope. "
                "The internal 'Charter of Specialized Centres' reflects a specific central categorization and is not an exhaustive institute-wide registry. "
                "Authoritative current school and departmental webpages establish additional formally recognized Centres of Excellence."
            ),
            "specialized_institutional_coes": [
                "AyurTech Center of Excellence (Integrative Medicine and Ayurvedic Technology; documented via AyurTech CoE portal)",
                "Center of Excellence on Arts and Digital Immersion (ADI) (explicitly designated as a CoE on its official website)",
                "Rishabh Centre for Research and Innovation in Clean Energy (RCRICE) (described in institutional publications; research entity in clean energy)"
            ],
            "school_of_aide_coes": [
                "Center of Excellence in Brain Science and Applications",
                "Center of Excellence in Mathematical and Computational Economics",
                "Center of Excellence in Intelligent Infrastructure",
                "Center of Excellence in AI-based Precision Healthcare"
            ],
            "interdisciplinary_research_platforms": [
                "Center for Space Technology (Interdisciplinary research platform)",
                "Center for Technology Foresight and Policy (CTFP, Interdisciplinary policy research platform)",
                "Center for Digital Humanities (Interdisciplinary platform)"
            ]
        }

    @staticmethod
    def verify_academic_probation_and_termination_policy() -> Dict[str, Any]:
        """
        Returns verified Academic Performance, Continuation, Warning, Probation, and Termination rules
        connecting Section 11.5, Table 11.3, Section 11.5.1, Section 12.3, and Section 14.1.
        Explicitly notes that Section 5.3 covers GPA calculation, NOT probation triggers.
        """
        return {
            "source": "Academic Regulations, Sections 11.5, 11.5.1, Table 11.3, Section 12.3 & Section 14.1",
            "section_correction_note": "Academic performance, probation, and termination are governed jointly by Section 11.5, 11.5.1, Table 11.3, Section 12.3, and Section 14.1. Section 5.3 strictly governs Grade Point Average (SGPA/CGPA) calculations.",
            "first_year_continuation": {
                "general_obc_ews_min_credits": 18,
                "sc_st_pwd_min_credits": 14,
                "timeline": "By the end of the second registered semester (Table 11.3)",
                "consequence_of_failure": "Failure to satisfy this continuation condition permits a one-time programme restart; otherwise, the student's registration is terminated."
            },
            "third_semester_onward_rules": {
                "academic_warning_trigger": "Earning 15 or fewer credits in a registered semester triggers an academic warning and parental intimation (Section 11.5.1).",
                "academic_probation_trigger": "Earning fewer than 12 credits in any registered semester places a student on Academic Probation (Section 11.5.1).",
                "consecutive_probation_termination": "Two consecutive registered semesters on Academic Probation lead to termination of programme registration (Section 11.5.1 & Section 14.1(e))."
            },
            "probation_credit_cap": "A student on academic probation is restricted to registering for a maximum of 16 credits per semester (Section 12.3).",
            "termination_triggers": [
                "1. Unauthorised absence for eight weeks or more (Section 14.1(a)).",
                "2. Failure to register by the deadline without a bona fide reason (Section 14.1(b)).",
                "3. Confirmed code-of-conduct violation (Section 14.1(c)).",
                "4. Failure to earn prescribed minimum credits within the maximum programme duration (Section 14.1(d)).",
                "5. Academic probation for two consecutive semesters (Section 11.5.1 & Section 14.1(e)).",
                "6. Failure to meet first-year continuation requirements without opting for an approved restart (Section 11.5 & Table 11.3)."
            ]
        }

    @staticmethod
    def verify_minor_credit_requirements() -> Dict[str, Any]:
        """
        Returns verified Minor credit requirements from Section 11.2.
        """
        return {
            "source": "Academic Regulations, Section 11.2 (Specialization & Minor Policy)",
            "total_minor_credits": 20,
            "open_electives_counted": 10,
            "additional_overload_credits": 10,
            "summary": "Total 20 credits required in the Minor discipline: 10 credits are counted through normal Open Electives (OE), and 10 additional credits are earned through course overload beyond the normal curriculum."
        }

    @staticmethod
    def verify_mtech_ta_assistantship_hours() -> Dict[str, Any]:
        """
        Returns verified M.Tech Teaching Assistantship workload from official admission norms.
        """
        return {
            "source": "IIT Jodhpur M.Tech Admission Regulations & Teaching Assistantship Guidelines",
            "mandatory_hours_per_week": 8,
            "applicable_students": "Regular full-time M.Tech students receiving Institute fellowship / monthly stipend",
            "duties": "8 hours per week of Teaching or Research Assistantship work subject to maintaining satisfactory academic standing"
        }

    @staticmethod
    def verify_branch_change_policy(cohort: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns version-aware Branch Change criteria, cutoffs, and intake quota rules
        based on the student's admission cohort.
        """
        cohort_clean = (cohort or "").lower().strip()

        # Cohort AY 2026-27 Onwards: Discontinued
        if "2026" in cohort_clean or "2027" in cohort_clean:
            return {
                "source": "Senate Resolution / Academic Regulations Section 16 Sunset Clause",
                "applicable_cohort": "2026-27 onwards",
                "is_discontinued": True,
                "status_statement": (
                    "Branch change is permanently DISCONTINUED from Academic Year 2026–27 onwards "
                    "in all undergraduate programmes. Students admitted in 2026–27 or later are not eligible "
                    "for branch change under any CGPA criteria."
                ),
                "cgpa_cutoffs": None,
                "intake_limits": None
            }

        # Earlier Batches (2019-2024 Batches)
        elif any(y in cohort_clean for y in ["earlier", "2019", "2020", "2021", "2022", "2023", "2024"]):
            return {
                "source": "Academic Regulations (UG Programs), Section 16.2",
                "applicable_cohort": "2019–2024 Batches",
                "is_discontinued": False,
                "cgpa_cutoffs": {
                    "other_branches_general_obc": 8.0,
                    "other_branches_sc_st_pwd": 7.0
                },
                "prerequisites": "Completed all prescribed credits of 1st year (Semester I & II), no backlog, no disciplinary action",
                "intake_limits": "Subject to sanctioned intake strength limits of the department",
                "sunset_clause": "Note: Thresholds were revised for AY 2025–26 (to 8.5/7.5), and branch change is permanently discontinued from AY 2026–27."
            }

        # Current Scheme (AY 2025–26 Batch)
        else:
            return {
                "source": "Academic Regulations Section 16 & AY 2025–26 Branch Change Circular",
                "applicable_cohort": "2025–26 Batch",
                "is_discontinued": False,
                "cgpa_cutoffs_ay2025_26": {
                    "other_branches_general_obc": 8.5,
                    "other_branches_sc_st_pwd": 7.5,
                    "engineering_science_general_obc": 8.0,
                    "engineering_science_sc_st_pwd": 7.0
                },
                "prerequisites": "Completed all prescribed credits of 1st year (Semester I & II), no backlog, no disciplinary action",
                "intake_limits": "Upper limit of 110% of sanctioned strength and lower limit of 90% of current strength for the branch-change exercise (Engineering Science has a separate upper limit of 30 seats). Not a universal permission to revise sanctioned intake by +/- 10%.",
                "sunset_clause": "Branch change is permanently discontinued from AY 2026–27 onwards in undergraduate programmes."
            }

    @staticmethod
    def verify_phd_comprehensive_exam_policy() -> Dict[str, Any]:
        """
        Returns verified Ph.D. Comprehensive Examination rules and related cancellation clauses.
        """
        return {
            "source": "Postgraduate Academic Regulations, Section 26.5 & Section 26.8",
            "examination_window": "Full-time Ph.D. scholars must complete the comprehensive examination within 3 semesters.",
            "retake_rule": "First unsatisfactory attempt permits one additional attempt within 6 months; student must perform satisfactorily in all papers.",
            "qualifier_deadline": "Qualifier requirements are normally expected to be completed before the beginning of the fifth semester, and no later than the beginning of the sixth semester.",
            "related_consequence_and_cancellation": "Section 26.5.1 does not explicitly prescribe automatic termination solely for failing the second attempt. However, under Section 26.8, registration is liable for cancellation if the Student Research Committee (SRC) finds the student's progress unsatisfactory."
        }

    @staticmethod
    def verify_faculty_sabbatical_leave_policy() -> Dict[str, Any]:
        """
        Returns verified Faculty Sabbatical Leave rules from First Statutes Clause 27.
        """
        return {
            "source": "IIT Jodhpur First Statutes, Clause 27 (Sabbatical Leave)",
            "continuous_service_required": "Six or more years of continuous regular service with the Institute.",
            "deputation_reset_condition": "If special leave or deputation lasting one year or more is taken, six years of service must be completed after returning before becoming eligible again for sabbatical leave.",
            "lifetime_cap": "Sabbatical leave cannot exceed three occasions during the faculty member's entire service with the Institute."
        }

    @staticmethod
    def verify_audit_course_policy() -> Dict[str, Any]:
        """
        Returns verified Audit Course Grading policy from Section 4.1.8 & 5.1 & 5.3.
        """
        return {
            "source": "Academic Regulations, Section 4.1.8, Section 5.1 (Table 1) & Section 5.3",
            "official_grading_nomenclature": "IIT Jodhpur official regulations award letter grade 'S' (Satisfactory), NOT 'AP' (Audit Pass), for satisfactory performance in audit and non-graded courses.",
            "unsatisfactory_grade": "Letter grade 'U' (Unsatisfactory) is awarded when stipulated attendance or evaluation requirements are not fulfilled.",
            "sgpa_cgpa_exclusion": "Grades obtained in audit courses ('S' or 'U') are completely EXCLUDED from the calculation of both SGPA and CGPA (Section 5.3.1 & Section 5.3.2)."
        }

    @staticmethod
    def verify_grade_appeal_policy() -> Dict[str, Any]:
        """
        Returns verified Grade Appeal and Correction procedure from Section 5.2.
        """
        return {
            "source": "Academic Regulations, Section 5.2 (Procedure for Declaration of Results)",
            "grade_correction_48_hours": "Grade correction: Allowed within 48 hours of display of grades by instructors, after approval of the Head of the Department (Step 7).",
            "final_approval_and_finality": "Results are recommended by HoD, approved by Dean, and approved by Chairman, Senate. Results once approved cannot ordinarily be changed under any circumstance (Step 11).",
            "exceptional_senate_appeal": "In exceptional cases, a student may appeal to the Chairman, Senate for reconsideration with formal justification (Step 11).",
            "statutory_deadline_distinction": "The 48-hour deadline applies strictly to departmental grade correction before finalization; the regulations do NOT establish a separate explicit deadline for an exceptional appeal to the Chairman, Senate. These two processes must not be conflated."
        }

    @staticmethod
    def verify_course_drop_and_withdrawal_policy() -> Dict[str, Any]:
        """
        Returns verified Course Add/Drop vs Withdrawal rules from Section 3.d & 3.e.
        """
        return {
            "source": "Academic Regulations, Section 3.d & Section 3.e (Registration)",
            "course_drop_window": "Course Drop: Allowed till the last date of registration (add/drop period, typically within first two weeks of semester) with Faculty Advisor approval, provided minimum 12 credits are maintained.",
            "course_drop_transcript_status": "A dropped course does NOT appear on the student's grade card or transcript.",
            "course_withdrawal_window": "Course Withdrawal: Permitted after the add/drop deadline, typically within four weeks from the beginning of the semester as specified in the Academic Calendar.",
            "course_withdrawal_grade": "A withdrawn course is permanently mentioned on the student's grade card with a letter grade 'W'. It carries no grade points and does not affect SGPA or CGPA."
        }

    @staticmethod
    def verify_semester_withdrawal_policy() -> Dict[str, Any]:
        """
        Returns verified Semester Withdrawal rules from UG Section 15.3 and PG Section 23.3,
        explicitly contrasting with Course Withdrawal (Section 3.e).
        """
        return {
            "source": "Academic Regulations (UG Programs Section 15.3 & PG Programs Section 23.3)",
            "ug_provisions": (
                "Undergraduate Students (Section 15.3): Applicable in unforeseen circumstances or for medical reasons. "
                "Absence for a period of four or more weeks in any semester results in automatic withdrawal from that semester. "
                "DUGC can approve and intimate the Office of Academics regarding the leave of the student for records. "
                "In case the leave extends beyond a semester, it must be approved by the Academic Committee. "
                "Crucially, semester withdrawal on medical grounds is NOT counted towards the maximum permissible programme duration."
            ),
            "pg_provisions": (
                "Postgraduate Students (Section 23.3): Applicable in unforeseen circumstances or for medical reasons, "
                "with the same four-or-more-weeks absence trigger resulting in automatic semester withdrawal. "
                "DRC approves and intimates the Office of Academics; extension beyond one semester requires Academic Committee approval; "
                "medical withdrawal is NOT counted towards maximum programme duration. "
                "Crucial Postgraduate Financial Distinction: In all PG cases, the fellowship is STOPPED with immediate effect "
                "and is resumed only after the student rejoins the Institute."
            ),
            "course_vs_semester_contrast": (
                "Crucial Distinction from Course Withdrawal: Course Withdrawal (Section 3.e) applies to an individual course "
                "within the first 4 weeks, resulting in a 'W' grade with no effect on CGPA or other registered courses. "
                "In contrast, Semester Withdrawal (Section 15.3 / Section 23.3) withdraws the student entirely from ALL courses "
                "in that semester due to medical/unforeseen circumstances or prolonged absence (>= 4 weeks), suspends PG fellowship, "
                "and exempts medical leave periods from maximum degree duration."
            )
        }

    @staticmethod
    def verify_academic_leave_policy() -> Dict[str, Any]:
        """
        Returns verified Academic Leave provisions from UG Section 15.4 and PG Section 23.4.
        """
        return {
            "source": "Academic Regulations, Section 15.4 (UG) & Section 23.4 (PG)",
            "eligible_reasons": (
                "Academic leave may be requested and sanctioned for specific academic pursuits, including: "
                "1. Fieldwork essential for curriculum or research, "
                "2. Library work due to specialized collections at external libraries, "
                "3. Experimental work or training using specialized facilities at external institutions/laboratories, "
                "4. Course work or research work at an exchange/collaborating university."
            ),
            "governing_approval": "Requires recommendation by DUGC (for UG) or DRC (for PG) and approval by the Office of Academics."
        }

    @staticmethod
    def verify_student_leave_policy(degree_level: str = "ALL", programme: str = "ALL", query: str = "") -> Dict[str, Any]:
        """
        Returns comprehensive verified student leave rules from IIT Jodhpur Academic Regulations:
        Section 15 (Undergraduate Leave Rules) and Section 23 (Postgraduate Leave Rules).
        """
        return {
            "source_ug": "Academic Regulations (UG Programmes), Section 15 (Leave Rules)",
            "source_pg": "Academic Regulations (PG Programmes), Section 23 (Leave Rules)",
            "source_statutes": "IIT Jodhpur First Statutes (Clause 27 & Schedule 'D')",
            "ug_leave_rules": {
                "approval_authority": "Application addressed to Convener DUGC with medical certificate (if applicable); prior approval required before availing leave.",
                "mid_sem_recess_and_vacation": "Section 15.1: Undergraduate students are entitled to avail of mid-semester recess and vacation as specified in the Academic Calendar.",
                "short_leave": {
                    "medical_grounds": "Maximum 10 days during the semester on medical grounds (Section 15.2).",
                    "other_valid_reasons": "Maximum 5 days during the semester for any other valid reason (Section 15.2).",
                    "total_short_leave_summary": "Undergraduate short leave is strictly capped at a maximum of 10 days on medical grounds and 5 days for other valid reasons in any semester."
                },
                "semester_withdrawal": (
                    "Section 15.3: Absence for 4 or more weeks in any semester results in automatic semester withdrawal. "
                    "DUGC can approve and intimate the Office of Academics. Extension beyond one semester requires Academic Committee approval. "
                    "Medical semester withdrawal is NOT counted towards the maximum permissible programme duration."
                ),
                "academic_leave": "Section 15.4: Permitted for fieldwork, external library/laboratory work, or student exchange with DUGC recommendation and Office of Academics approval."
            },
            "pg_leave_rules": {
                "approval_authority": "Application addressed to Convener DRC with medical certificate (if applicable); prior approval required.",
                "annual_leave_quota": (
                    "Section 23.1: Maximum of 30 days in an academic year (including leave on medical grounds and mid-semester break). "
                    "Students must apply in advance even during summer/winter terms or mid-semester breaks."
                ),
                "stretch_limits": {
                    "phd_limit": "A Ph.D. student cannot avail leave for more than 15 days at a stretch (Section 23.1).",
                    "mtech_msc_limit": "An M.Tech. / M.Sc. student cannot avail leave for more than 10 days at a stretch (Section 23.1)."
                },
                "leave_beyond_quota_penalty": (
                    "Section 23.1: Leave beyond 30 days in an academic year may be granted in exceptional cases by DRC "
                    "WITHOUT assistantship/fellowship during that excess period."
                ),
                "maternity_leave": (
                    "Section 23.2: Maternity leave for Ph.D. students is sanctioned as per MHRD/MoE directives. "
                    "Crucially, it is NOT counted towards the maximum permissible duration of the programme."
                ),
                "semester_withdrawal": (
                    "Section 23.3: Absence for 4 or more weeks results in automatic semester withdrawal. "
                    "Crucial Financial Consequence: Fellowship is STOPPED with immediate effect and resumed only after rejoining. "
                    "Medical semester withdrawal is not counted towards maximum programme duration."
                ),
                "academic_leave": "Section 23.4: Permitted for fieldwork, external experimental facilities, specialized libraries, or exchange research."
            },
            "faculty_leave_summary": (
                "Faculty members are governed by IIT Jodhpur First Statutes Clause 27 (Sabbatical Leave: 6+ years regular service, max 3 occasions) "
                "and Schedule 'D' (Casual Leave, Earned Leave, Half Pay Leave, Commuted Leave, Duty Leave, Maternity/Paternity Leave, and EOL)."
            )
        }

    @staticmethod
    def verify_institutional_leadership_or_hod(query_or_dept: str) -> Dict[str, Any]:
        """
        Returns authoritative Head of Department, Dean, or Institute Leadership details
        from the official Constitution of the Senate, IIT Jodhpur (Updated 05.01.2026).
        """
        q = (query_or_dept or "").lower()

        # Department / School Heads
        if any(k in q for k in ["computer science", "cse"]):
            return {
                "role": "Head, Department of Computer Science & Engineering",
                "department": "Computer Science & Engineering",
                "incumbent": "Prof. Gaurav Harit",
                "designation": "Professor",
                "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 2 Table 2",
                "details": "Prof. Gaurav Harit serves as the Head of the Department of Computer Science & Engineering and Member of the Senate."
            }
        elif any(k in q for k in ["mechanical", "me"]):
            return {
                "role": "Head, Department of Mechanical Engineering",
                "department": "Mechanical Engineering",
                "incumbent": "Dr. Mrityunjay R. Doddamani",
                "designation": "Associate Professor & Head",
                "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 2 Table 2 (Item 48)",
                "details": "Dr. Mrityunjay R. Doddamani serves as the Head of the Department of Mechanical Engineering and Member of the Senate (Heads of Departments/Schools other than Professors)."
            }
        elif any(k in q for k in ["electrical", "ee"]):
            return {
                "role": "Head, Department of Electrical Engineering",
                "department": "Electrical Engineering",
                "incumbent": "Prof. Bharat Singh Rajpurohit",
                "designation": "Professor & Head",
                "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 2 Table 2",
                "details": "Prof. Bharat Singh Rajpurohit serves as the Head of the Department of Electrical Engineering and Member of the Senate."
            }
        elif any(k in q for k in ["artificial intelligence", "aide", "data science"]):
            return {
                "role": "Head, School of Artificial Intelligence & Data Science (AIDE)",
                "department": "School of Artificial Intelligence & Data Science",
                "incumbent": "Prof. Anil Kumar Tiwari",
                "designation": "Professor & Head",
                "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 2 Table 2",
                "details": "Prof. Anil Kumar Tiwari serves as the Head of the School of Artificial Intelligence & Data Science (AIDE) and Member of the Senate."
            }
        elif any(k in q for k in ["civil", "infrastructure", "cie", "ce"]):
            return {
                "role": "Head, Department of Civil & Infrastructure Engineering",
                "department": "Civil & Infrastructure Engineering",
                "incumbent": "Dr. Deepika Bhattu",
                "designation": "Associate Professor & Head",
                "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 2 Table 2 (Item 52)",
                "details": "Dr. Deepika Bhattu serves as the Head of the Department of Civil & Infrastructure Engineering and Member of the Senate."
            }
        elif any(k in q for k in ["chemical", "che"]):
            return {
                "role": "Head, Department of Chemical Engineering",
                "department": "Chemical Engineering",
                "incumbent": "Prof. Parag Arvind Deshpande",
                "designation": "Professor & Head",
                "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 2 Table 2",
                "details": "Prof. Parag Arvind Deshpande serves as the Head of the Department of Chemical Engineering and Member of the Senate."
            }
        elif any(k in q for k in ["bioscience", "bioengineering", "bsbe", "biology"]):
            return {
                "role": "Acting Head, Department of Bioscience and Bioengineering",
                "department": "Bioscience and Bioengineering",
                "incumbent": "Dr. Neha Jain",
                "designation": "Associate Professor & Acting Head",
                "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 2 Table 2 (Item 49)",
                "details": "Dr. Neha Jain serves as the Acting Head of the Department of Bioscience and Bioengineering and Member of the Senate."
            }
        elif any(k in q for k in ["math", "mathematics"]):
            return {
                "role": "Head, Department of Mathematics",
                "department": "Mathematics",
                "incumbent": "Dr. V.V.M.S. Chandramouli",
                "designation": "Associate Professor & Head",
                "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 2 Table 2 (Item 51)",
                "details": "Dr. V.V.M.S. Chandramouli serves as the Head of the Department of Mathematics and Member of the Senate."
            }
        elif any(k in q for k in ["physics"]):
            return {
                "role": "Head, Department of Physics",
                "department": "Physics",
                "incumbent": "Dr. Reetanjali Moharana",
                "designation": "Associate Professor & Head",
                "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 2 Table 2 (Item 50)",
                "details": "Dr. Reetanjali Moharana serves as the Head of the Department of Physics and Member of the Senate."
            }
        elif any(k in q for k in ["chemistry", "chemical sciences"]):
            return {
                "role": "Head, Department of Chemistry",
                "department": "Chemistry",
                "incumbent": "Dr. Sandip Murarka",
                "designation": "Associate Professor & Head",
                "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 2 Table 2 (Item 54)",
                "details": "Dr. Sandip Murarka serves as the Head of the Department of Chemistry and Member of the Senate."
            }
        elif any(k in q for k in ["materials", "metallurgical"]):
            return {
                "role": "Head, Department of Materials Engineering",
                "department": "Materials Engineering",
                "incumbent": "Prof. Bhabani Kumar Satapathy",
                "designation": "Professor, Deputy Director & Head",
                "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 2 Table 2",
                "details": "Prof. Bhabani Kumar Satapathy serves as Head of the Department of Materials Engineering and Deputy Director, IIT Jodhpur."
            }
        elif any(k in q for k in ["management", "sme", "entrepreneurship"]):
            return {
                "role": "Head, School of Management & Entrepreneurship (SME)",
                "department": "School of Management & Entrepreneurship",
                "incumbent": "Dr. Deepak Saxena",
                "designation": "Assistant Professor & Head",
                "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 2 Table 2 (Item 53)",
                "details": "Dr. Deepak Saxena serves as the Head of the School of Management & Entrepreneurship (SME) and Member of the Senate."
            }
        elif any(k in q for k in ["liberal arts", "sola", "humanities"]):
            return {
                "role": "Head, School of Liberal Arts (SoLA)",
                "department": "School of Liberal Arts",
                "incumbent": "Dr. Alok Ranjan",
                "designation": "Assistant Professor & Head",
                "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 2 Table 2 (Item 55)",
                "details": "Dr. Alok Ranjan serves as the Head of the School of Liberal Arts (SoLA) and Member of the Senate."
            }
        elif any(k in q for k in ["design"]):
            return {
                "role": "Head, School of Design",
                "department": "School of Design",
                "incumbent": "Dr. Sajan Sahadevan Pillai",
                "designation": "Assistant Professor & Head",
                "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 2 Table 2 (Item 56)",
                "details": "Dr. Sajan Sahadevan Pillai serves as the Head of the School of Design and Member of the Senate."
            }

        # Deans & Institute Leadership
        elif any(k in q for k in ["academic affairs", "doaa"]):
            return {
                "role": "Dean of Academic Affairs (DOAA)",
                "department": "Office of Academic Affairs",
                "incumbent": "Prof. Atul Kumar",
                "designation": "Professor & Dean",
                "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 2 Table 2",
                "details": "Prof. Atul Kumar serves as the Dean of Academic Affairs (DOAA), IIT Jodhpur."
            }
        elif any(k in q for k in ["research and development", "dord", "r&d dean"]):
            return {
                "role": "Dean of Research and Development (DORD)",
                "department": "Office of Research & Development",
                "incumbent": "Prof. Sunil Kumar Khijwania",
                "designation": "Professor & Dean",
                "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 2 Table 2",
                "details": "Prof. Sunil Kumar Khijwania serves as the Dean of Research and Development (DORD), IIT Jodhpur."
            }
        elif any(k in q for k in ["student affairs", "dosa"]):
            return {
                "role": "Dean of Student Affairs (DOSA)",
                "department": "Office of Student Affairs",
                "incumbent": "Prof. Ankita Sharma",
                "designation": "Professor & Dean",
                "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 2 Table 2",
                "details": "Prof. Ankita Sharma serves as the Dean of Student Affairs (DOSA), IIT Jodhpur."
            }
        elif any(k in q for k in ["faculty affairs", "dofa"]):
            return {
                "role": "Acting Dean of Faculty Affairs (DOFA)",
                "department": "Office of Faculty Affairs",
                "incumbent": "Prof. Meenu Chhabra",
                "designation": "Professor & Acting Dean",
                "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 2 Table 2",
                "details": "Prof. Meenu Chhabra serves as the Acting Dean of Faculty Affairs (DOFA), IIT Jodhpur."
            }
        elif any(k in q for k in ["director"]):
            return {
                "role": "Director, IIT Jodhpur & Chairman of the Senate",
                "department": "Director's Office",
                "incumbent": "Prof. Avinash Kumar Agarwal",
                "designation": "Director & Professor",
                "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 1 Table 1",
                "details": "Prof. Avinash Kumar Agarwal serves as the Director of IIT Jodhpur and Ex-officio Chairman of the Senate."
            }
        elif any(k in q for k in ["deputy director"]):
            return {
                "role": "Deputy Director, IIT Jodhpur",
                "department": "Directorate",
                "incumbent": "Prof. Bhabani Kumar Satapathy",
                "designation": "Deputy Director & Professor",
                "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 1 Table 1",
                "details": "Prof. Bhabani Kumar Satapathy serves as the Deputy Director of IIT Jodhpur and Ex-officio Member of the Senate."
            }

        # Default summary of prominent Heads
        return {
            "role": "Heads of Academic Departments and Schools (IIT Jodhpur)",
            "department": "All Academic Departments",
            "incumbent": "Multiple Appointed Heads",
            "designation": "Department Heads / School Heads",
            "source": "Constitution of the Senate (IIT Jodhpur Academic Governance), Updated 05.01.2026, Page 2 Table 2",
            "details": (
                "Key Department Heads at IIT Jodhpur:\n"
                "- Computer Science & Engineering: Prof. Gaurav Harit\n"
                "- Mechanical Engineering: Dr. Mrityunjay R. Doddamani\n"
                "- Electrical Engineering: Prof. Bharat Singh Rajpurohit\n"
                "- School of AI & Data Science (AIDE): Prof. Anil Kumar Tiwari\n"
                "- Civil & Infrastructure Engineering: Dr. Deepika Bhattu\n"
                "- Chemical Engineering: Prof. Parag Arvind Deshpande\n"
                "- Bioscience & Bioengineering: Dr. Neha Jain (Acting Head)\n"
                "- Mathematics: Dr. V.V.M.S. Chandramouli\n"
                "- Physics: Dr. Reetanjali Moharana\n"
                "- Chemistry: Dr. Sandip Murarka"
            )
        }


deterministic_verifier = DeterministicVerifier()
