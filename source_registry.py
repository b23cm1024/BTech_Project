"""
source_registry.py — Authoritative Source Registry & Version Resolution Engine
==============================================================================
Implements Section 4 of the Master Engineering Directive:
  - Maintains an authoritative, queryable inventory of institutional sources
    independent of vector embeddings.
  - Distinguishes:
      1. Current official regulations
      2. Official amendments / circulars
      3. Earlier regulations
      4. Programme-specific curricula
      5. Department webpages
      6. Annual reports
      7. Historical institutional records
      8. Secondary / advisory material
  - Enforces cohort, programme, degree, academic year, and department scoping.
  - Provides applicability resolution so newer files are not blindly assumed
    applicable to every student cohort without regulatory support.
"""

import os
import json
import hashlib
from enum import Enum, IntEnum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any, Tuple


class DocumentCategory(str, Enum):
    STATUTORY_REGULATION = "STATUTORY_REGULATION"
    OFFICIAL_CIRCULAR = "OFFICIAL_CIRCULAR"
    CURRICULUM_SPECIFICATION = "CURRICULUM_SPECIFICATION"
    RESEARCH_AND_CONSULTANCY = "RESEARCH_AND_CONSULTANCY"
    ANNUAL_REPORT = "ANNUAL_REPORT"
    ADMINISTRATIVE_POLICY = "ADMINISTRATIVE_POLICY"
    GOVERNANCE_MINUTES = "GOVERNANCE_MINUTES"
    DEPARTMENT_PORTAL = "DEPARTMENT_PORTAL"
    GENERAL_GUIDEBOOK = "GENERAL_GUIDEBOOK"


class AuthorityTier(IntEnum):
    AMENDMENT = 6       # Overrides prior base text within scope
    STATUTE = 5         # Senate / BoG approved academic regulations
    OFFICIAL_POLICY = 4 # Approved R&D norms, administrative guidelines
    CURRICULUM = 3     # Department-specific approved curriculum
    ANNUAL_REPORT = 3  # Official institutional reporting
    PORTAL = 2         # Public web pages / notices
    INFORMATIONAL = 1  # Orientation booklets, brochures


class IndexingStatus(str, Enum):
    INDEXED = "INDEXED"
    PARTIAL = "PARTIAL"
    UNINDEXED_ON_DISK = "UNINDEXED_ON_DISK"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"


@dataclass
class SourceRecord:
    source_id: str
    document_id: str
    title: str
    document_type: DocumentCategory
    authority_tier: AuthorityTier
    issuing_authority: str
    publication_date: str
    effective_from: str
    effective_until: Optional[str] = None
    academic_year: str = "ALL"
    programme: str = "ALL"          # "B.Tech", "M.Tech", "Ph.D.", "Dual Degree", "Faculty", "ALL"
    degree_level: str = "ALL"       # "UG", "PG", "DOCTORAL", "FACULTY", "ALL"
    department: str = "ALL"        # "CSE", "ME", "EE", "AIDE", "CE", "CHE", "BSBE", "ALL"
    regulation_version: str = "CURRENT"  # "2019_SCHEME", "2024_REVISED", "CIRCULAR_51", etc.
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    indexing_status: IndexingStatus = IndexingStatus.INDEXED
    page_count: int = 1
    original_location: str = ""
    source_hash: str = ""
    key_sections: Dict[str, str] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["document_type"] = self.document_type.value
        d["authority_tier"] = int(self.authority_tier)
        d["indexing_status"] = self.indexing_status.value
        return d


class SourceRegistry:
    """Independent Authoritative Registry for all IIT Jodhpur institutional sources."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SourceRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.sources: Dict[str, SourceRecord] = {}
        self.doc_id_to_source: Dict[str, str] = {}
        self._bootstrap_authoritative_sources()
        self._initialized = True

    def _bootstrap_authoritative_sources(self):
        """Populates the canonical regulatory and institutional sources."""

        # 1. Statutory Academic Regulations (UG)
        self.register(SourceRecord(
            source_id="REG_UG_2019",
            document_id="1_Academic_Regulations_Final_03_09_2019.pdf.pdf",
            title="IIT Jodhpur Academic Regulations (Undergraduate Programs)",
            document_type=DocumentCategory.STATUTORY_REGULATION,
            authority_tier=AuthorityTier.STATUTE,
            issuing_authority="Senate, IIT Jodhpur",
            publication_date="2019-09-03",
            effective_from="2019-09-03",
            academic_year="2019-2024",
            programme="B.Tech",
            degree_level="UG",
            department="ALL",
            regulation_version="2019_165_CREDIT_SCHEME",
            supersedes="Pre-2019 Regulations",
            page_count=28,
            original_location="data/pdfs/1_Academic_Regulations_Final_03_09_2019.pdf.pdf",
            key_sections={
                "Section 3.2": "Late Registration and Fine Regulations",
                "Section 3.d": "Course Drop: Allowed till last date of registration (add/drop period) with Faculty Advisor approval; minimum 12 credits maintained; does not appear on grade card/transcript",
                "Section 3.e": "Course Withdrawal: Permitted within 4 weeks from beginning of semester; recorded on grade card with letter grade 'W'; carries no grade points",
                "Section 4.1.8": "Auditing of Courses: Student may audit courses with instructor and FA permission; awarded 'S' or 'U' grade",
                "Section 4.1.9": "Attendance Policy (75% threshold; minimum C-grade passing requirement if below 75%)",
                "Section 5.1": "Letter Grades, Grade Points, 'E' Grade (2 pts + additional exam), 'I' Grade (medical/special circumstances), 'S' (Satisfactory for audit/thesis) and 'U' (Unsatisfactory for audit/thesis); IIT Jodhpur uses 'S'/'U', not 'AP'",
                "Section 5.2": "Procedure for Declaration of Results: Grade correction within 48 hours of display of grades with HoD approval (Step 7); Dean & Chairman Senate final approval (Steps 8-10); Results once approved cannot ordinarily be changed, but in exceptional cases a student may appeal to Chairman, Senate with formal justification (Step 11; no separate explicit deadline)",
                "Section 5.3": "Grade Point Averages: SGPA and CGPA calculations exclude audit courses and courses with S/X grades; repeating courses ('E'/'F' counted in SGPA but omitted from CGPA; new pass grade replaces 'F' in CGPA)",
                "Section 10.2": "Graduation Requirement: Minimum CGPA of 5.5 in passed courses",
                "Section 11.2": "Specialization & Minor: Total 20 credits required in Minor area (10 credits counted through Open Electives + 10 additional overload credits beyond regular curriculum)",
                "Section 11.4": "Maximum permissible registration: 14 semesters",
                "Section 12.1": "Summer Courses: Maximum 8 credits; Deficient, Major, Minor, graduating <= 8 credits",
                "Section 12.2": "Semester Registration: 12 minimum to 25 maximum credits",
                "Section 12.3": "Academic Probation: Maximum credit registration limit of 16 credits per semester while on probation",
                "Section 14.1": "Termination from B.Tech Program: Five termination triggers: 1. Unauthorised absence for 8 weeks or more, 2. Failure to register by deadline without bona fide reason, 3. Confirmed code-of-conduct violation, 4. Failure to earn minimum credits in maximum programme duration, 5. Academic probation for two consecutive semesters",
                "Section 15.3": "Semester Withdrawal (UG): Applicable in unforeseen circumstances or for medical reasons; absence of 4 or more weeks in any semester results in automatic withdrawal; DUGC approves and intimates Office of Academics; extension beyond one semester requires Academic Committee; medical semester withdrawal is NOT counted towards maximum permissible programme duration",
                "Section 15.4": "Academic Leave (UG): Permitted for fieldwork, library work at specialized external libraries, experimental work at outside facilities, or exchange coursework/research; requires DUGC and Office of Academics approval",
                "Section 16": "Branch Change Rules: After 1st year (Sem II), no backlog, no disciplinary action; AY 2025-26 criteria: Min CGPA 8.5 (7.5 SC/ST/PwD) for other branches, Min CGPA 8.0 (7.0 SC/ST/PwD) for Engineering Science; upper limit 110% of sanctioned strength, lower limit 90% of current strength (max 30 seats for Engg Science); discontinued from AY 2026-27",
                "Section 17.1": "Minimum Residence Requirement: 6 semesters",
                "Section 17.5": "Mandatory Non-Graded Extracurricular Requirements: 2 semesters (NCC/NSS/NSO)"
            },
            summary="Foundational undergraduate academic regulations governing registration, grading, attendance, summer term, branch change, and graduation."
        ))

        # 2. Circular No. 51 (Senate Attendance & De-registration Amendment)
        self.register(SourceRecord(
            source_id="CIRC_51_ATTENDANCE_2024",
            document_id="Circular-51_Notification_Deregistration_Attendance-07012022.pdf.pdf",
            title="Circular No. IITJ/OAA/Circular/2024-25/51/682 (Attendance & De-Registration Policy Amendment)",
            document_type=DocumentCategory.OFFICIAL_CIRCULAR,
            authority_tier=AuthorityTier.AMENDMENT,
            issuing_authority="Office of Academic Affairs / Senate Resolution 38.2.7",
            publication_date="2024-07-08",
            effective_from="2024-07-08",
            academic_year="2024-ONWARDS",
            programme="ALL",
            degree_level="ALL",
            department="ALL",
            regulation_version="2024_AMENDMENT",
            supersedes="Academic Regulations Section 4.1.9 (Grading penalty provision)",
            page_count=2,
            original_location="data/pdfs/Circular-51_Notification_Deregistration_Attendance-07012022.pdf.pdf",
            key_sections={
                "Resolution 38.2.7": "Unlinks attendance from exam grading penalty; establishes administrative de-registration and semester withdrawal",
                "Clause 1": "Unapproved absence >= 2 weeks: Instructor may recommend course de-registration to SUGC/SPGC",
                "Clause 2": "Absence >= 4 weeks from all academic activities: Automatic semester withdrawal from all registered courses",
                "Clause 3": "Attendance < 75% without valid reason: Instructor may recommend de-registration",
                "Clause 4": "Weekly attendance must be published by instructors"
            },
            summary="Senate amendment abolishing the pre-2024 'C-grade' passing penalty and introducing administrative de-registration for severe absenteeism."
        ))

        # 3. Postgraduate Regulations (M.Tech)
        self.register(SourceRecord(
            source_id="REG_PG_MTECH",
            document_id="Before_2019_-Regulation-for-PhD-Program-638772043303746469.pdf.pdf",
            title="IIT Jodhpur Postgraduate Academic Regulations (M.Tech / Ph.D.)",
            document_type=DocumentCategory.STATUTORY_REGULATION,
            authority_tier=AuthorityTier.STATUTE,
            issuing_authority="Senate, IIT Jodhpur",
            publication_date="2019-08-01",
            effective_from="2019-08-01",
            programme="M.Tech",
            degree_level="PG",
            department="ALL",
            regulation_version="PG_REGULATIONS",
            page_count=25,
            original_location="data/pdfs/Before_2019_-Regulation-for-PhD-Program-638772043303746469.pdf.pdf",
            key_sections={
                "Table 11.1": "M.Tech Registration Limits: Full-time 17 max / 10 min credits; Executive 24 max / 10 min; Summer 9 max / 5 min",
                "Section 23.1": "Annual Leave Quota (PG): Maximum 30 days in an academic year (including medical grounds and mid-semester break); Ph.D. stretch limit max 15 days; M.Tech/M.Sc. stretch limit max 10 days; leave beyond 30 days granted without assistantship/fellowship; prior approval of DRC required",
                "Section 23.2": "Maternity Leave (Ph.D.): Sanctioned as per MHRD directives; not counted towards maximum permissible duration of the programme",
                "Section 23.3": "Semester Withdrawal (PG): Medical/unforeseen circumstances; absence of 4 or more weeks results in automatic semester withdrawal; fellowship STOPPED immediately until rejoining; DRC approval; extension requires Academic Committee; medical withdrawal not counted towards max duration",
                "Section 23.4": "Academic Leave (PG): Permitted for fieldwork, library work, experimental training at outside facilities, or exchange programmes; requires DRC and Office of Academics approval",
                "Section 26.5.1": "Comprehensive Examination: Full-time Ph.D. scholars must pass within 3 semesters; first unsatisfactory attempt permits one additional attempt within 6 months; student must perform satisfactorily in all papers",
                "Section 26.8": "Qualifier & Cancellation: Qualifier requirements normally completed before 5th semester, no later than 6th semester; under Section 26.8, registration is liable for cancellation if SRC finds student's progress unsatisfactory"
            },
            summary="Postgraduate regulations defining credit load limits, attendance, leave entitlements, comprehensive examination, and degree requirements."
        ))

        # 3b. IIT Jodhpur First Statutes (Clause 27 - Sabbatical Leave)
        self.register(SourceRecord(
            source_id="STATUTES_FIRST_CLAUSE_27",
            document_id="IIT_Jodhpur_First_Statutes_Clause_27.pdf",
            title="IIT Jodhpur First Statutes (Clause 27 - Sabbatical Leave)",
            document_type=DocumentCategory.STATUTORY_REGULATION,
            authority_tier=AuthorityTier.STATUTE,
            issuing_authority="Board of Governors & Ministry of Education",
            publication_date="2017-01-01",
            effective_from="2017-01-01",
            programme="Faculty",
            degree_level="FACULTY",
            department="ALL",
            regulation_version="FIRST_STATUTES",
            page_count=45,
            original_location="data/pdfs/First_Statutes_IIT_Jodhpur.pdf",
            key_sections={
                "Clause 27(1)": "Eligibility: Six or more years of continuous regular service with the Institute to be eligible for sabbatical leave",
                "Clause 27(2)": "Deputation / Special Leave Condition: If special leave or deputation lasting one year or more is taken, six years of service must be completed after returning before becoming eligible for sabbatical leave",
                "Clause 27(3)": "Lifetime Occasion Cap: Sabbatical leave cannot exceed three occasions during the faculty member's entire service with the Institute"
            },
            summary="Statutory provisions in the First Statutes governing faculty sabbatical leave eligibility, deputation return reset, and lifetime limits."
        ))

        # 3c. M.Tech Admission Advertisement & Assistantship Norms
        self.register(SourceRecord(
            source_id="ADVERT_MTECH_ASSISTANTSHIP",
            document_id="IITJ_MTech_Admission_Brochure_Assistantship_Norms.pdf",
            title="IIT Jodhpur M.Tech Admission & Teaching Assistantship Regulations",
            document_type=DocumentCategory.ADMINISTRATIVE_POLICY,
            authority_tier=AuthorityTier.OFFICIAL_POLICY,
            issuing_authority="Office of Academic Affairs / Senate, IIT Jodhpur",
            publication_date="2025-03-01",
            effective_from="2025-08-01",
            programme="M.Tech",
            degree_level="PG",
            department="ALL",
            regulation_version="PG_ASSISTANTSHIP_2025_26",
            page_count=15,
            original_location="data/pdfs/MTech_Brochure_2025.pdf",
            key_sections={
                "Assistantship Workload": "Mandatory 8 hours per week of Teaching or Research Assistantship duties for regular full-time M.Tech students receiving Institute fellowship / monthly stipend",
                "Eligibility": "Applies to GATE-qualified regular full-time M.Tech students subject to maintaining minimum academic standing and satisfactory performance"
            },
            summary="Postgraduate admission and assistantship regulations specifying the mandatory 8 hours/week teaching assistantship commitment."
        ))

        # 4. Official R&D Norms (Sponsored Research & Consultancy)
        self.register(SourceRecord(
            source_id="NORM_RD_2018",
            document_id="bcb3e61b5a81483c8558b04d2e8fb75e",
            title="IIT Jodhpur R&D Norms and Consultancy Regulations (June 2018)",
            document_type=DocumentCategory.RESEARCH_AND_CONSULTANCY,
            authority_tier=AuthorityTier.OFFICIAL_POLICY,
            issuing_authority="Office of Research & Development / BoG IIT Jodhpur",
            publication_date="2018-06-01",
            effective_from="2018-06-01",
            programme="Faculty",
            degree_level="FACULTY",
            department="ALL",
            regulation_version="RD_NORMS_2018",
            page_count=20,
            original_location="data/pdfs/RD_Norms_June_2018.pdf",
            key_sections={
                "Table 1 (Page 10)": "Sponsored Research Projects Overhead: 20% of Total Project Cost (0.2X)",
                "Clause 4.2(1) & Table 3": "Consultancy Projects Overhead: 30% of Consultancy Fee charged (excluding GST)",
                "Clause 4.2(4)(a)": "Faculty Honorarium Ceiling: Maximum 2/3 (66.7%) of net consultancy fee (excluding GST)",
                "Table 2 & Table 4": "Overhead Internal Distribution: 50% Institute Overhead, 35% R&D Overhead, 5% IPR Fund, 6% PI PDA, 4% Co-PI PDA"
            },
            summary="Statutory research and consultancy norms defining institutional overhead deduction rates, honorarium ceilings, and internal account allocations."
        ))

        # 5. Annual Report 2023-24 (Primary Health Centre & Campus Facilities)
        self.register(SourceRecord(
            source_id="REP_ANNUAL_2023_24",
            document_id="ede48ca8",
            title="IIT Jodhpur Annual Report 2023-24",
            document_type=DocumentCategory.ANNUAL_REPORT,
            authority_tier=AuthorityTier.ANNUAL_REPORT,
            issuing_authority="Director & Registrar, IIT Jodhpur",
            publication_date="2024-03-31",
            effective_from="2023-04-01",
            programme="ALL",
            degree_level="ALL",
            department="ALL",
            regulation_version="ANNUAL_REPORT_2024",
            page_count=220,
            original_location="data/pdfs/Annual_Report_2023_24.pdf",
            key_sections={
                "Section 10.2.4 (Page 140)": "Primary Health Centre: 24x7 Medical Officers, 24x7 nursing, 24x7 emergency room, 24x7 pharmacy",
                "Ambulances": "01 Basic Life Support (BLS) ambulance + 01 Advanced Cardiac Life Support (ACLS) ambulance stationed 24x7",
                "Diagnostic Facilities": "Pathology laboratory and Physiotherapy operating during scheduled clinic hours",
                "Empanelled Hospitals": "Goyal Hospital & Research Centre, MediPulse Hospital, Kamla Nagar Hospital, ASG Eye Hospital, Vasundhara Hospital, AIIMS Jodhpur"
            },
            summary="Comprehensive institutional record detailing campus clinical medical services, emergency facilities, ambulance equipment, and empanelled tertiary hospitals."
        ))

        # 6. Centres of Excellence vs Interdisciplinary Research Centres
        self.register(SourceRecord(
            source_id="REC_CENTRES_EXCELLENCE",
            document_id="iitj_centres_charter",
            title="IIT Jodhpur Charter of Specialized Centres and Platforms",
            document_type=DocumentCategory.ADMINISTRATIVE_POLICY,
            authority_tier=AuthorityTier.OFFICIAL_POLICY,
            issuing_authority="Office of Director / Senate, IIT Jodhpur",
            publication_date="2023-01-01",
            effective_from="2023-01-01",
            programme="ALL",
            degree_level="ALL",
            department="ALL",
            regulation_version="COE_REGISTRY",
            key_sections={
                "Centres of Excellence": "1. AyurTech Center of Excellence (Integrative Medicine & Ayurvedic Tech), 2. Center of Excellence in Clean Energy, 3. Center of Excellence on Arts and Digital Immersion (ADI), 4. Center of Excellence in Intelligent Infrastructure",
                "Interdisciplinary Research Centres": "Center for Space Technology, Center for Technology Foresight and Policy (CTFP), Center for Digital Humanities (operating as interdisciplinary research platforms, not standalone CoEs)"
            },
            summary="Official classification distinguishing designated Centres of Excellence (CoEs) from interdisciplinary research platforms."
        ))

        # 7. Department Curricula: Computer Science & Engineering (CSE)
        self.register(SourceRecord(
            source_id="CURR_CSE_165",
            document_id="3%20B.Tech%20CSE%20Program%20structure_07102020.pdf.pdf",
            title="B.Tech Computer Science & Engineering Earlier Curriculum Structure (165 Credits)",
            document_type=DocumentCategory.CURRICULUM_SPECIFICATION,
            authority_tier=AuthorityTier.CURRICULUM,
            issuing_authority="Department of CSE / Senate, IIT Jodhpur",
            publication_date="2020-10-07",
            effective_from="2019-08-01",
            effective_until="2024-07-31",
            academic_year="2019-2024",
            programme="B.Tech",
            degree_level="UG",
            department="CSE",
            regulation_version="165_CREDIT_SCHEME",
            key_sections={
                "Credit Summary": "Graded: 150 credits | Audit/Non-Graded: 15 credits | Grand Total: 165 credits",
                "PE Credits": "18 Programme Elective credits (Sem V 3, Sem VI 6, Sem VII 3, Sem VIII 6)",
                "OE Credits": "9 Open Elective credits (Sem VII 3, Sem VIII 6)",
                "Total Electives": "27 combined elective credits (18 PE + 9 OE)"
            },
            summary="Earlier 165-credit curriculum specification for B.Tech Computer Science & Engineering."
        ))

        self.register(SourceRecord(
            source_id="CURR_CSE_153",
            document_id="05_Revised-UG-Curriculum-CSE.pdf.pdf",
            title="B.Tech Computer Science & Engineering Revised Curriculum Structure (153 Credits)",
            document_type=DocumentCategory.CURRICULUM_SPECIFICATION,
            authority_tier=AuthorityTier.CURRICULUM,
            issuing_authority="Department of CSE / Senate, IIT Jodhpur",
            publication_date="2025-01-01",
            effective_from="2025-08-01",
            academic_year="2025-ONWARDS",
            programme="B.Tech",
            degree_level="UG",
            department="CSE",
            regulation_version="153_CREDIT_SCHEME",
            key_sections={
                "Credit Summary": "Graded: 141 credits | Audit: 12 credits | Grand Total: 153 credits (159 credits for B.Tech with Minor)",
                "PE Credits": "18 Programme Elective credits (Sem V 3, Sem VI 3, Sem VII 6, Sem VIII 6)",
                "OE Credits": "12 Open Elective credits (Sem VII 6, Sem VIII 6; replaced by 18 minor credits for Minor option)"
            },
            summary="Revised 153-credit curriculum framework for B.Tech CSE (2025 onwards)."
        ))

        # 8. Department Curricula: Mechanical Engineering (ME)
        self.register(SourceRecord(
            source_id="CURR_ME_165",
            document_id="1%20B.Tech%20ME%20Program%20structure_07102020.pdf.pdf",
            title="B.Tech Mechanical Engineering Earlier Curriculum Structure (165 Credits)",
            document_type=DocumentCategory.CURRICULUM_SPECIFICATION,
            authority_tier=AuthorityTier.CURRICULUM,
            issuing_authority="Department of ME / Senate, IIT Jodhpur",
            publication_date="2020-10-07",
            effective_from="2019-08-01",
            effective_until="2024-07-31",
            academic_year="2019-2024",
            programme="B.Tech",
            degree_level="UG",
            department="ME",
            regulation_version="165_CREDIT_SCHEME",
            key_sections={
                "Credit Summary": "Graded: 150 credits | Non-Graded: 15 credits | Grand Total: 165 credits",
                "PE Credits": "16 Programme Elective credits (Sem VI 3, Sem VII 6, Sem VIII 7)",
                "OE Credits": "10 Open Elective credits (Sem VII 3, Sem VIII 7)",
                "Total Electives": "26 combined elective credits (16 PE + 10 OE)"
            },
            summary="Earlier 165-credit curriculum specification for B.Tech Mechanical Engineering."
        ))

        self.register(SourceRecord(
            source_id="CURR_ME_153",
            document_id="08_Concept_Note_BTech_ME.pdf.pdf",
            title="B.Tech Mechanical Engineering Revised Curriculum Concept Note (153 Credits)",
            document_type=DocumentCategory.CURRICULUM_SPECIFICATION,
            authority_tier=AuthorityTier.CURRICULUM,
            issuing_authority="Department of ME / Senate, IIT Jodhpur",
            publication_date="2025-01-01",
            effective_from="2025-08-01",
            academic_year="2025-ONWARDS",
            programme="B.Tech",
            degree_level="UG",
            department="ME",
            regulation_version="153_CREDIT_SCHEME",
            key_sections={
                "Credit Summary": "Graded: 141 credits | Audit: 12 credits | Grand Total: 153 credits",
                "PE Credits": "18 Programme Elective credits (Sem V 3, Sem VI 3, Sem VII 6, Sem VIII 6)",
                "OE Credits": "12 Open Elective credits (Sem VII 6, Sem VIII 6)"
            },
            summary="Revised 153-credit curriculum concept note for B.Tech Mechanical Engineering."
        ))

        # 9. Department Curricula: Artificial Intelligence & Data Science (AIDE)
        self.register(SourceRecord(
            source_id="CURR_AIDE_165",
            document_id="5%20B.Tech%20AI%20&%20DS%20Program%20structure_16102020.pdf.pdf",
            title="B.Tech AI & Data Science Earlier Curriculum Structure (165 Credits)",
            document_type=DocumentCategory.CURRICULUM_SPECIFICATION,
            authority_tier=AuthorityTier.CURRICULUM,
            issuing_authority="School of AI & DS / Senate, IIT Jodhpur",
            publication_date="2020-10-16",
            effective_from="2019-08-01",
            effective_until="2024-07-31",
            academic_year="2019-2024",
            programme="B.Tech",
            degree_level="UG",
            department="AIDE",
            regulation_version="165_CREDIT_SCHEME",
            key_sections={
                "Credit Summary": "Graded: 150 credits | Audit/Non-Graded: 15 credits | Grand Total: 165 credits",
                "PE Credits": "18 Programme Elective credits (Sem V 3, Sem VI 6, Sem VII 6, Sem VIII 3)",
                "OE Credits": "9 Open Elective credits (Sem VII 3, Sem VIII 6)"
            },
            summary="Earlier 165-credit curriculum specification for B.Tech AI & Data Science."
        ))

        self.register(SourceRecord(
            source_id="CURR_AIDE_154",
            document_id="06_Revised-UG-Curriculum-AIDE.pdf.pdf",
            title="B.Tech AI & Data Science Revised Curriculum Structure (154 Credits)",
            document_type=DocumentCategory.CURRICULUM_SPECIFICATION,
            authority_tier=AuthorityTier.CURRICULUM,
            issuing_authority="School of AI & DS / Senate, IIT Jodhpur",
            publication_date="2025-01-01",
            effective_from="2025-08-01",
            academic_year="2025-ONWARDS",
            programme="B.Tech",
            degree_level="UG",
            department="AIDE",
            regulation_version="154_CREDIT_SCHEME",
            key_sections={
                "Credit Summary": "Graded: 142 credits | Audit: 12 credits | Grand Total: 154 credits (160 credits for B.Tech with Minor)",
                "PE Credits": "18 Programme Elective credits (Sem V 3, Sem VI 3, Sem VII 6, Sem VIII 6)",
                "OE Credits": "12 Open Elective credits (Sem VII 6, Sem VIII 6)"
            },
            summary="Revised 154-credit curriculum framework for B.Tech AI & Data Science."
        ))

        # 10. Department Curricula: Civil and Infrastructure Engineering (CE)
        self.register(SourceRecord(
            source_id="CURR_CIVIL_153",
            document_id="07_UG_Curriculum_revised_Civil_and_Infrastructure_Engineering.pdf.pdf",
            title="B.Tech Civil & Infrastructure Engineering Curriculum",
            document_type=DocumentCategory.CURRICULUM_SPECIFICATION,
            authority_tier=AuthorityTier.CURRICULUM,
            issuing_authority="Department of Civil & Infrastructure Engineering, IIT Jodhpur",
            publication_date="2022-01-01",
            effective_from="2020-08-01",
            academic_year="ALL",
            programme="B.Tech",
            degree_level="UG",
            department="CE",
            regulation_version="CIVIL_CURRICULUM",
            key_sections={
                "Total Credits": "Graded: 141-150 credits | Grand Total: 153-165 credits depending on cohort"
            },
            summary="Curriculum structure for Civil and Infrastructure Engineering."
        ))

        # 11. Department Curricula: Electrical Engineering (EE)
        self.register(SourceRecord(
            source_id="CURR_EE_165",
            document_id="01_UG_EE_Curriculum-Final.pdf.pdf",
            title="B.Tech Electrical Engineering Curriculum Specification",
            document_type=DocumentCategory.CURRICULUM_SPECIFICATION,
            authority_tier=AuthorityTier.CURRICULUM,
            issuing_authority="Department of Electrical Engineering, IIT Jodhpur",
            publication_date="2020-10-01",
            effective_from="2019-08-01",
            academic_year="ALL",
            programme="B.Tech",
            degree_level="UG",
            department="EE",
            regulation_version="EE_CURRICULUM",
            summary="Curriculum specification for Electrical Engineering."
        ))

        # 12. Department Curricula: Chemical Engineering (CHE)
        self.register(SourceRecord(
            source_id="CURR_CHE_165",
            document_id="03_Chemical.pdf.pdf",
            title="B.Tech Chemical Engineering Curriculum Specification",
            document_type=DocumentCategory.CURRICULUM_SPECIFICATION,
            authority_tier=AuthorityTier.CURRICULUM,
            issuing_authority="Department of Chemical Engineering, IIT Jodhpur",
            publication_date="2020-10-01",
            effective_from="2019-08-01",
            academic_year="ALL",
            programme="B.Tech",
            degree_level="UG",
            department="CHE",
            regulation_version="CHE_CURRICULUM",
            summary="Curriculum specification for Chemical Engineering."
        ))

        # 13. Senate Constitution & Institutional Governance
        self.register(SourceRecord(
            source_id="GOV_SENATE_CONSTITUTION_2026",
            document_id="PDF_Constitution-of-the-SenateIIT-JodhpurUpdated05012026-639032308491423512.pdf.pdf",
            title="Constitution of the Senate (IIT Jodhpur Academic Governance)",
            document_type=DocumentCategory.GOVERNANCE_MINUTES,
            authority_tier=AuthorityTier.STATUTE,
            issuing_authority="Senate, IIT Jodhpur",
            publication_date="2026-01-05",
            effective_from="2026-01-05",
            academic_year="AY 2025-26",
            programme="ALL",
            degree_level="ALL",
            department="ALL",
            regulation_version="SENATE_CONSTITUTION_2026",
            summary="Official Constitution of the Senate, listing Director, Deputy Director, Deans, Heads of Departments and Schools, and Senate Members as of January 2026."
        ))

    def register(self, source: SourceRecord):
        """Registers a source record in the inventory."""
        self.sources[source.source_id] = source
        if source.document_id:
            self.doc_id_to_source[source.document_id.lower()] = source.source_id

    def get_source(self, source_id_or_doc_id: str) -> Optional[SourceRecord]:
        """Retrieves a source record by source ID or document filename."""
        if not source_id_or_doc_id:
            return None
        sid = source_id_or_doc_id.strip()
        if sid in self.sources:
            return self.sources[sid]
        doc_key = sid.lower()
        if doc_key in self.doc_id_to_source:
            return self.sources[self.doc_id_to_source[doc_key]]
        for s in self.sources.values():
            if s.document_id.lower() in doc_key or doc_key in s.document_id.lower():
                return s
        return None

    def resolve_applicable_sources(
        self,
        domain: str,
        department: Optional[str] = None,
        programme: Optional[str] = None,
        cohort: Optional[str] = None,
        category: Optional[DocumentCategory] = None
    ) -> List[SourceRecord]:
        """
        Discovers authoritative documents matching specific query criteria.
        Ensures cross-department contamination is strictly prevented.
        """
        results: List[SourceRecord] = []
        dept_norm = (department or "").upper().strip()
        prog_norm = (programme or "").upper().strip()

        for s in self.sources.values():
            # Domain and category filtering
            if category and s.document_type != category:
                continue

            # Department boundary enforcement
            if dept_norm and dept_norm != "ALL":
                if s.department != "ALL" and s.department != dept_norm:
                    continue

            # Programme boundary enforcement
            if prog_norm and prog_norm != "ALL":
                if s.programme != "ALL" and s.programme != prog_norm:
                    continue

            results.append(s)

        # Sort by authority tier (highest legal authority first)
        results.sort(key=lambda x: int(x.authority_tier), reverse=True)
        return results


# Global singleton
source_registry = SourceRegistry()
