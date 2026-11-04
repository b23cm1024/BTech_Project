"""
database/init_db.py
===================
Initializes relational database schema (PostgreSQL / SQLite) and migrates
authoritative inventory records and verified atomic claims.
"""

import os
import json
import logging
from database.connection import engine, Base, SessionLocal
from database.models import Document, DocumentSection, AtomicClaim

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InitDB")


def init_database():
    """Create all relational tables."""
    logger.info("Creating relational database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Tables created successfully.")


def migrate_inventory():
    """Migrate documents from inventory.json into the documents table."""
    inv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "document_inventory.json"))
    if not os.path.exists(inv_path):
        logger.warning(f"Inventory file not found at {inv_path}")
        return

    db = SessionLocal()
    try:
        existing_count = db.query(Document).count()
        if existing_count > 0:
            logger.info(f"Database already contains {existing_count} documents. Skipping duplicate import.")
            return

        with open(inv_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            if "records" in data:
                raw_items = data["records"]
            else:
                raw_items = list(data.values())
        elif isinstance(data, list):
            raw_items = data
        else:
            raw_items = []

        logger.info(f"Importing {len(raw_items)} document records into relational store...")

        doc_objects = []
        seen_hashes = set()
        for r in raw_items:
            cat = r.get("document_category") or r.get("category", "General")
            entities = r.get("entities")
            prog = entities[0] if (isinstance(entities, list) and entities) else r.get("programme")
            cohort = r.get("cohort_edition") or r.get("cohort", "General")
            fname = r.get("doc_name") or r.get("filename", "Unknown")
            title = r.get("canonical_name") or r.get("canonical_title") or fname
            sha = r.get("sha256") or r.get("hash")
            if sha in seen_hashes:
                sha = None
            elif sha:
                seen_hashes.add(sha)

            doc = Document(
                filename=str(fname)[:255],
                canonical_title=str(title)[:255],
                category=str(cat)[:64],
                programme=str(prog)[:128] if prog else None,
                cohort=str(cohort)[:64] if cohort else None,
                source_url=r.get("source_url") or r.get("file_path"),
                sha256_hash=sha,
                page_count=r.get("pages_processed_count") or r.get("page_count", 1),
                is_indexed=(r.get("indexing_completeness") == "FULLY_INDEXED") or r.get("is_indexed", True)
            )
            doc_objects.append(doc)

        db.bulk_save_objects(doc_objects)
        db.commit()
        logger.info(f"Successfully migrated {len(doc_objects)} documents.")

    finally:
        db.close()


def populate_baseline_atomic_claims():
    """Populate baseline verified statutory claims."""
    db = SessionLocal()
    try:
        deleted_count = db.query(AtomicClaim).delete()
        if deleted_count > 0:
            logger.info(f"Cleared {deleted_count} outdated atomic claims for fresh statutory refresh.")

        # Find or create Academic Regulations document reference
        reg_doc = db.query(Document).filter(Document.filename.ilike("%regulations%")).first()
        if not reg_doc:
            reg_doc = Document(
                filename="academic_regulations_master.pdf",
                canonical_title="Academic Regulations for B.Tech. Programs",
                category="Academic Regulations",
                programme="B.Tech.",
                cohort="General"
            )
            db.add(reg_doc)
            db.commit()
            db.refresh(reg_doc)
        doc_id = reg_doc.id

        statutory_claims = [
            # 1. Branch Change
            ("Branch Change", "Branch Change Timing", "End of 1st Year (Semester II) only", "timing", "2025_onwards", None, 16, "Academic Regulations Section 16"),
            ("Branch Change", "Minimum CGPA (General/OBC)", "8.5", "cgpa", "2025_onwards", None, 16, "Section 16: CGPA >= 8.5 for General/OBC"),
            ("Branch Change", "Minimum CGPA (SC/ST/PwD)", "7.5", "cgpa", "2025_onwards", None, 16, "Section 16: CGPA >= 7.5 for SC/ST/PwD"),
            ("Branch Change", "Department Lower Strength Limit", "90%", "percentage", "2025_onwards", None, 16, "Section 16: Strength shall not fall below 90%"),
            ("Branch Change", "Department Upper Strength Limit", "110%", "percentage", "2025_onwards", None, 16, "Section 16: Strength shall not exceed 110%"),
            
            # 2. Minor & Specialization Overload
            ("Department Specialization / Minor", "Total Minor Credits", "20", "credits", "General", None, 11, "Section 11.2: Total 20 credits"),
            ("Department Specialization / Minor", "Open Elective Component", "10", "credits", "General", None, 11, "Section 11.2: 10 credits out of 20 via OE"),
            ("Department Specialization / Minor", "Additional Overload Credits", "10", "credits", "General", None, 11, "Section 11.2: 10 additional credits via course overload beyond regular B.Tech"),
            ("Department Specialization / Minor", "Overload Eligibility Criteria", "Earned credits >= 80 and CGPA >= 6.0 from 6th semester", "policy", "General", 6, 11, "Section 11.2.2"),
            
            # 3. Summer Term Courses
            ("Summer Term", "Maximum Credits", "8", "credits", "General", None, 12, "Section 12.1: Students can register for courses during summer up to a maximum of Eight Credits"),
            ("Summer Term", "Eligibility Criteria", "Academically deficient students only (low CGPA and/or fewer credits)", "policy", "General", None, 12, "Section 12.1: Only academically deficient students permitted"),
            
            # 4. Incomplete Grade ('I')
            ("Incomplete Grade", "Award Circumstances", "Missed evaluation component due to medical grounds or genuine circumstances", "policy", "General", None, 5, "Section 5.1: Missing evaluation due to illness or special circumstances"),
            ("Incomplete Grade", "Conversion Deadline", "Last date in Academic Calendar, otherwise automatically converts to 'F'", "deadline", "General", None, 5, "Section 5.1 & Pre-2019 p. 13"),
            
            # 5. Course Repetition
            ("Course Repetition", "Core Courses", "Mandatory repeat for 'E' or 'F' grade", "policy", "General", None, 5, "Section 5.1 / 5.2: Student has to repeat core courses"),
            ("Course Repetition", "CGPA Calculation", "Both grades on transcript, but only new grade counted for CGPA", "policy", "General", None, 5, "Section 5.1 / 5.2"),
            
            # 6. Dual Degree Conversion
            ("Dual Degree Conversion", "Eligibility Credits", "Minimum 134 Credits at the end of 7th semester", "credits", "General", 7, 11, "Section 11.2 (b): Earned minimum 134 Credits"),
            ("Dual Degree Conversion", "Minimum CGPA", "6.0", "cgpa", "General", 7, 11, "Section 11.2 (b): At least 6 CGPA"),
            ("Dual Degree Conversion", "Application Window", "Eighth semester", "timing", "General", 8, 11, "Section 11.2 (b): Opt from eighth semester"),
            
            # 7. Non-Graded Extra-Curricular
            ("Non-Graded Graduation Requirement", "Mandatory Extra-Curricular", "At least two semesters of NCC, NSS, NSO, Performing Arts, or Sports", "policy", "General", None, 17, "Section 17.5: Completion of engagement for at least two Semesters"),
            
            # 8. Academic Probation & Termination
            ("Academic Probation", "Under-load Limit", "Maximum 16 credits per semester", "credits", "General", None, 12, "Section 12.2: Academically deficient students register for max 16 credits"),
            ("Termination", "Termination Condition", "Academic probation for two consecutive semesters", "policy", "General", None, 14, "Section 14.1 (e)"),

            # 9. Undergraduate Registration Limits
            ("Undergraduate Registration Limits", "Regular Semester Minimum Credits", "12", "credits", "General", None, 12, "Section 12.2: An undergraduate student should register for a minimum of 12 credits in a semester"),
            ("Undergraduate Registration Limits", "Regular Semester Maximum Credits", "25", "credits", "General", None, 12, "Section 12.2: The maximum number of credits permitted for a UG student in a semester is 25"),
            ("Undergraduate Registration Limits", "Overload Eligibility CGPA Threshold", "7.5", "cgpa", "General", None, 12, "Section 12.3: A student with a CGPA of 7.5 or higher may overload"),

            # 10. Graduation Requirements
            ("Requirements for Graduation", "Minimum Graduating CGPA (Earlier B.Tech Scheme)", "5.5", "cgpa", "earlier_scheme", None, 21, "BTech_Old p. 21 Clause 10.2: Further, the student shall obtain a minimum CGPA of 5.5 in all the passed courses"),
            ("Requirements for Graduation", "Graduation Academic Requirements (2019 Scheme)", "Fulfillment of prescribed credits as per Table 11.2 (153–165 credits)", "policy", "2019_onwards", None, 17, "Section 17.2: Fulfilled academic requirements if she/he has earned the minimum credits as per Table 11.2"),
            ("Requirements for Graduation", "Minimum Residence Period", "4 years (8 semesters)", "duration", "General", None, 21, "Section 10.1 / Section 17: Minimum residence shall be eight semesters (four years)"),

            # 11. Letter Grades
            ("Letter Grades", "Letter Grade 'E' Meaning & Grade Points", "Awarded when marks are less than 'D' cutoff and met attendance; eligible for additional exam; 2 grade points", "grade_definition", "General", None, 5, "Table 5.1 & Section 5.1: E grade carrying 2 grade points with additional examination"),
            ("Letter Grades", "Grade 'E' Grade Points", "2", "grade_points", "General", None, 5, "Table 5.1: E grade point = 2"),

            # 12. Late Registration
            ("Registration Procedures", "Late Registration Procedure & Fine", "Permitted on or before last date of late registration with Dean permission; daily fine published separately in Academic Calendar", "policy", "General", None, 3, "Section 3.2: Late registration on or before last date of late registration paying the fine"),

            # 13. Course Drop & Audit
            ("Course Drop & Audit Procedures", "Course Drop Deadline & Approval", "Drop permitted till last date of registration in Academic Calendar with Faculty Advisor approval", "deadline", "General", None, 3, "Section 3.3 / Section 5.5: Adding and Dropping till last date of registration"),
            ("Course Drop & Audit Procedures", "Minimum Credit Load After Drop", "Minimum 12 credits must be maintained (no under-load permitted)", "credits", "General", None, 3, "Section 3.3 / 12.2: Student must register for a minimum of 12 credits; under-load not allowed"),

            # 14. Faculty R&D and Consultancy Overhead
            ("Sponsored Research Projects", "Institute Overhead Rate", "20% of Total Project Cost", "percentage", "General", None, 10, "Table 1: Institute Overhead is 0.2X of total project cost (R&D Norms June 2018)"),
            ("Sponsored Research Projects", "Overhead Distribution", "50% Institute, 35% R&D, 5% IPR Fund, 6% PI PDA, 4% Co-PI PDA", "distribution", "General", None, 11, "Table 2: Utilisation of Institute Overheads"),
            ("Consultancy Projects", "Institute Overhead Rate", "30% of Consultancy Fee charged (excluding GST)", "percentage", "General", None, 12, "Clause 4.2(1): The Institute Overhead in Consultancy Projects shall be 30% of the amount A being charged as the fee (excluding GST)"),
            ("Consultancy Projects", "Maximum Faculty Honorarium", "Two-thirds (66.7%) of Consultancy Fee charged", "percentage", "General", None, 12, "Clause 4.2(4)(a): The total honorarium paid to Faculty and Staff shall not exceed two-thirds of the Consultancy Fee"),
            ("Consultancy Projects", "Overhead Distribution", "50% Institute Overhead Account, 35% R&D Overhead Account, 5% IPR Fund, 6% PI PDA, 4% Co-PI PDA", "distribution", "General", None, 13, "Table 4: Utilisation of Institute Overheads from Consultancy Projects"),

            # 15. Faculty Sabbatical Leave & Research Supervision
            ("Faculty Sabbatical & Leave", "Supervisory Continuity on Leave", "Short leave up to 1 year: Caretaker Supervisor appointed by DRC/IRC; Long leave over 1 year: Additional Supervisor appointed; Sabbatical supervisor may continue supervision", "policy", "General", None, 29, "Academic Regulations: Ph.D. supervision during sabbatical or lien"),
            ("Faculty Sabbatical & Leave", "Faculty Service Eligibility Conditions", "Governed by IIT Statutes / Central Civil Rules; specific minimum continuous service tenure not stated in academic regulations", "policy", "General", None, 29, "Statutes of IIT Jodhpur / Establishment Section"),

            # 16. Postgraduate Assistantship Continuation (Table 22.8 & Table 12.8)
            ("PG Teaching Assistantship", "Minimum CGPA for Assistantship Continuation (M.Tech - Table 22.8)", "6.5", "cgpa", "General", None, 27, "Academic Regulations Table 22.8: M.Tech continuation of assistantship requires CGPA >= 6.5"),
            ("PG Teaching Assistantship", "Minimum CGPA for Assistantship Continuation (Dual Degree M.Tech-Ph.D. - Table 22.8)", "7.0", "cgpa", "General", None, 27, "Academic Regulations Table 22.8: Dual Degree M.Tech-Ph.D. continuation requires CGPA >= 7.0"),
            ("PG Teaching Assistantship", "Minimum CGPA for Fellowship Continuation (Ph.D. - Table 22.8)", "7.0", "cgpa", "General", None, 27, "Academic Regulations Table 22.8: Ph.D. research fellowship continuation requires CGPA >= 7.0"),
            ("PG Teaching Assistantship", "Minimum CGPA for Assistantship Continuation (M.Tech - Table 12.8 Scheme)", "6.0", "cgpa", "2022_scheme", None, 27, "PG Regulations Table 12.8: M.Tech continuation requires CGPA >= 6.0"),
            ("PG Teaching Assistantship", "Minimum CGPA for Assistantship Continuation (Dual Degree M.Tech-Ph.D. - Table 12.8 Scheme)", "6.5", "cgpa", "2022_scheme", None, 27, "PG Regulations Table 12.8: Dual Degree continuation requires CGPA >= 6.5"),
            ("PG Teaching Assistantship", "Assistantship Weekly Workload Duties", "8 hours per week", "hours", "General", None, 27, "Section 12.8.4: Students receiving assistantship are required to perform 8 hours of academic work per week"),

            # 17. Undergraduate B.Tech Programme Duration & Residence Limits (Section 17.1, 11.4 & 10.1)
            ("Requirements for Graduation", "Minimum Residential Requirement (B.Tech)", "6 semesters", "duration", "2019_onwards", None, 17, "Regulations 2019 Section 17.1: The minimum residential requirement for B.Tech shall be six semesters"),
            ("Requirements for Graduation", "Normal Designed B.Tech Programme Duration", "8 semesters (4 years)", "duration", "General", None, 17, "Regulations Section 17.1 & Curriculum: Normal designed curriculum structure is 8 semesters (4 years)"),
            ("Requirements for Graduation", "Maximum Permissible Programme Duration (B.Tech - Regulations 2019)", "14 registered semesters", "duration", "2019_onwards", None, 11, "Regulations 2019 Section 11.4: Maximum completion duration is 14 registered semesters"),
            ("Requirements for Graduation", "Maximum Permissible Programme Duration (B.Tech - Earlier Scheme)", "16 registered semesters (8 years)", "duration", "earlier_scheme", None, 21, "BTech_Old Section 10.1: A student must complete all requirements within a maximum period of 8 years (16 semesters)"),

            # 18. CPDA & Relocation Allowances (Adv-1 March 2025 vs Brochure-1)
            ("Faculty Professional Development", "CPDA Total Financial Grant", "Rs. 3.00 Lakhs (₹3,00,000)", "amount", "General", None, 6, "Adv-1 p. 6 & Portal: Cumulative Professional Development Allowance of Rs. 3 lakhs for every block period of 3 years on reimbursable basis"),
            ("Faculty Recruitment & Relocation", "Relocation Reimbursement (Domestic Govt/Autonomous/Similar)", "As per SR-116 and TTA Rules", "policy", "General", None, 6, "Adv-1 p. 6 Section 1.1.3: Joining from Government/Autonomous/similar institute governed by SR-116 and TTA Rules"),
            ("Faculty Recruitment & Relocation", "Relocation Reimbursement (Domestic Other Institutions)", "Up to Rs. 1,50,000/- (1.5 Lakhs) or actuals", "amount", "General", None, 6, "Adv-1 p. 6 Section 1.1.3: Joining from within India other than Govt institutes capped at Rs. 1,50,000/- or actuals"),
            ("Faculty Recruitment & Relocation", "Relocation Reimbursement (Abroad)", "Up to Rs. 2,50,000/- (2.5 Lakhs) or actuals", "amount", "General", None, 6, "Adv-1 p. 6 Section 1.1.3: Joining from abroad capped at Rs. 2,50,000/- or actuals; must join within 3 months of return"),
            ("Faculty Recruitment & Relocation", "Relocation Mandatory Service Bond", "3 Years", "tenure", "General", None, 6, "Adv-1 p. 6 Section 1.1.3: Requires execution of a bond to serve IIT Jodhpur for 3 years"),

            # 19. Ph.D. Coursework & Thesis Credits (Table 12.6 by Qualifying Degree)
            ("Ph.D. Academic Requirements", "Minimum Coursework Credits (Engineering with M.Tech)", "12", "credits", "General", None, 25, "Table 12.6: Ph.D. in Engineering with M.Tech / M.E. / M.Sc. Eng requires minimum 12 credits course work"),
            ("Ph.D. Academic Requirements", "Minimum Thesis Credits (Engineering with M.Tech)", "48", "credits", "General", None, 25, "Table 12.6: Ph.D. in Engineering with M.Tech / M.E. / M.Sc. Eng requires minimum 48 credits thesis work"),
            ("Ph.D. Academic Requirements", "Minimum Coursework Credits (Entrants with B.Tech or M.Sc. Science)", "24", "credits", "General", None, 25, "Table 12.6: Direct B.Tech / B.E. or M.Sc. Science entrants require minimum 24 credits course work"),
            ("Ph.D. Academic Requirements", "Minimum Thesis Credits (Entrants with B.Tech or M.Sc. Science)", "48", "credits", "General", None, 25, "Table 12.6: Direct B.Tech / B.E. or M.Sc. Science entrants require minimum 48 credits thesis work"),

            # 20. Postgraduate (M.Tech.) Semester Registration Credit Limits (Table 11.1, Page 21)
            ("Postgraduate Registration Limits", "Regular Semester Maximum Credits (Full-time M.Tech)", "17", "credits", "General", None, 21, "Table 11.1: Maximum Credits a M.Tech student can register for during Regular Semester (Full-time Regular/Sponsored) is 17 credits"),
            ("Postgraduate Registration Limits", "Regular Semester Minimum Credits (Full-time M.Tech)", "10", "credits", "General", None, 21, "Table 11.1: Minimum Credits a M.Tech student can register for during Regular Semester (Full-time Regular/Sponsored) is 10 credits"),
            ("Postgraduate Registration Limits", "Summer Term Maximum Credits (Full-time M.Tech)", "9", "credits", "General", None, 21, "Table 11.1: Maximum Credits for Summer Term Full-time M.Tech is 9 credits"),

            # 21. Research Initiation Seed Grant (Section 4.6 & Handbook Page 32)
            ("Faculty Research Grants", "Research Initiation Seed Grant (Sciences & Engineering)", "Up to Rs. 25 Lakhs", "amount", "General", None, 14, "R&D Norms Section 4.6 & Handbook p. 32: Research initiation grant of up to Rs. 25 Lakhs within initial period of 3 years"),
            ("Faculty Research Grants", "Research Initiation Seed Grant (Humanities & Social Sciences)", "Up to Rs. 5 Lakhs", "amount", "General", None, 14, "Adv-1 p. 6 & Handbook p. 32: Research initiation seed grant up to Rs. 5 Lakhs for Humanities and Social Sciences"),

            # 22. Ph.D. Comprehensive Examination Deadlines & Permissible Attempts (Section 16.5.1 & Table 12.9)
            ("Ph.D. Comprehensive Examination", "Comprehensive Examination Deadline (Regular Ph.D.)", "Within 3 semesters", "duration", "General", None, 32, "Section 16.5.1 & Table 12.9: Comprehensive Examination to be completed by regular Ph.D. within 3 semesters (no later than 3 semesters)"),
            ("Ph.D. Comprehensive Examination", "Comprehensive Examination Deadline (Part-time / Direct Ph.D.)", "Within 5 semesters", "duration", "General", None, 32, "Section 16.5.1 & Table 12.9: Direct Ph.D. and Part-time Ph.D. within 5 semesters"),
            ("Ph.D. Comprehensive Examination", "Maximum Permissible Attempts", "2 attempts", "attempts", "General", None, 32, "Section 16.5.1: If performance is not satisfactory in first attempt, given one more opportunity during next scheduled examination (maximum 2 attempts)"),

            # 23. Undergraduate Course Overload per semester allowance (Section 11.2.2)
            ("Undergraduate Course Overload", "Maximum Overload Credits Per Semester", "Up to 4 extra credits per semester", "credits", "General", 6, 11, "Section 11.2.2: Starting from 6th semester, students with at least 80 credits and CGPA >= 6.0 are permitted up to 4 extra credits per semester"),

            # 24. Academic Termination Threshold (Section 14.1(e))
            ("Academic Termination", "Academic Probation Termination Criterion", "Two consecutive semesters on academic probation", "condition", "General", None, 14, "Section 14.1(e): A student shall be terminated from the B.Tech. program if the student is on academic probation for two consecutive semesters"),

            # 25. B.Tech Curriculum Credit Requirements Across Departments (Revised 2025 Scheme vs Earlier Scheme)
            # Computer Science & Engineering
            ("Computer Science and Engineering", "Total Degree Credits (Revised 2025 Scheme)", "153 credits (141 graded + 12 audit) / 159 credits with Minor (147 graded + 12 audit)", "credits", "2025_onwards", None, 1, "CSE Revised UG Curriculum 2025: Total 153 credits (141 graded + 12 audit); 159 credits with Minor"),
            ("Computer Science and Engineering", "Total Degree Credits (Earlier Scheme 2019-2024)", "165 credits (150 graded + 15 non-graded/audit: 9 audit + 6 design)", "credits", "2019_2024", None, 1, "CSE Curriculum 2020: Total 165 credits (150 graded + 15 non-graded credits: 9 audit + 6 design)"),
            ("Computer Science and Engineering", "Total Graded Credits (Revised 2025 Scheme)", "141", "credits", "2025_onwards", None, 1, "CSE Revised 2025: Graded Credits = 141"),
            ("Computer Science and Engineering", "Total Graded Credits (Earlier Scheme)", "150", "credits", "2019_2024", None, 1, "CSE 2020: Graded Credits = 150"),

            # Artificial Intelligence and Data Science
            ("Artificial Intelligence and Data Science", "Total Degree Credits (Revised 2025 Scheme)", "154 credits (142 graded + 12 audit)", "credits", "2025_onwards", None, 1, "AIDE Revised UG Curriculum 2025: Total 154 credits (142 graded + 12 audit)"),
            ("Artificial Intelligence and Data Science", "Total Degree Credits (Earlier Scheme 2019-2024)", "165 credits (150 graded + 15 non-graded/audit)", "credits", "2019_2024", None, 1, "AIDE Curriculum: Total 165 credits (150 graded + 15 non-graded)"),
            ("Artificial Intelligence and Data Science", "Total Graded Credits (Revised 2025 Scheme)", "142", "credits", "2025_onwards", None, 1, "AIDE Revised 2025: Graded Credits = 142"),

            # Electrical Engineering
            ("Electrical Engineering", "Total Degree Credits (Revised 2025 Scheme)", "154 credits (142 graded + 12 audit)", "credits", "2025_onwards", None, 1, "EE Revised UG Curriculum 2025: Total 154 credits (142 graded + 12 audit)"),
            ("Electrical Engineering", "Total Degree Credits (Earlier Scheme 2019-2024)", "165 credits (150 graded + 15 non-graded/audit)", "credits", "2019_2024", None, 1, "EE Curriculum: Total 165 credits (150 graded + 15 non-graded)"),
            ("Electrical Engineering", "Total Graded Credits (Revised 2025 Scheme)", "142", "credits", "2025_onwards", None, 1, "EE Revised 2025: Graded Credits = 142"),

            # Mechanical Engineering
            ("Mechanical Engineering", "Total Degree Credits (Revised 2025 Scheme)", "154 credits (142 graded + 12 audit)", "credits", "2025_onwards", None, 1, "ME Revised UG Curriculum 2025: Total 154 credits (142 graded + 12 audit)"),
            ("Mechanical Engineering", "Total Degree Credits (Earlier Scheme 2019-2024)", "165 credits (150 graded + 15 non-graded/audit)", "credits", "2019_2024", None, 1, "ME Curriculum: Total 165 credits (150 graded + 15 non-graded)"),
            ("Mechanical Engineering", "Total Graded Credits (Revised 2025 Scheme)", "142", "credits", "2025_onwards", None, 1, "ME Revised 2025: Graded Credits = 142"),

            # Chemical Engineering
            ("Chemical Engineering", "Total Degree Credits (Revised 2025 Scheme)", "154 credits (142 graded + 12 audit)", "credits", "2025_onwards", None, 1, "Chemical Engineering Revised 2025: Total 154 credits (142 graded + 12 audit)"),
            ("Chemical Engineering", "Total Degree Credits (Earlier Scheme 2019-2024)", "165 credits (150 graded + 15 non-graded/audit)", "credits", "2019_2024", None, 1, "Chemical Engineering Curriculum: Total 165 credits (150 graded + 15 non-graded)"),
            ("Chemical Engineering", "Total Graded Credits (Revised 2025 Scheme)", "142", "credits", "2025_onwards", None, 1, "Chemical Engineering Revised 2025: Graded Credits = 142"),

            # Civil and Infrastructure Engineering
            ("Civil and Infrastructure Engineering", "Total Degree Credits (Revised 2025 Scheme)", "154 credits (142 graded + 12 audit)", "credits", "2025_onwards", None, 1, "Civil Engineering Revised UG Curriculum 2025: Total 154 credits (142 graded + 12 audit)"),
            ("Civil and Infrastructure Engineering", "Total Degree Credits (Earlier Scheme 2019-2024)", "165 credits (150 graded + 15 non-graded/audit)", "credits", "2019_2024", None, 1, "Civil Engineering Curriculum: Total 165 credits (150 graded + 15 non-graded)"),
            ("Civil and Infrastructure Engineering", "Total Graded Credits (Revised 2025 Scheme)", "142", "credits", "2025_onwards", None, 1, "Civil Engineering Revised 2025: Graded Credits = 142"),

            # Materials Engineering
            ("Materials Engineering", "Total Degree Credits (Revised 2025 Scheme)", "154 credits (142 graded + 12 audit)", "credits", "2025_onwards", None, 1, "Materials Engineering Revised UG Curriculum 2025: Total 154 credits (142 graded + 12 audit)"),
            ("Materials Engineering", "Total Degree Credits (Earlier Scheme 2019-2024)", "165 credits (150 graded + 15 non-graded/audit)", "credits", "2019_2024", None, 1, "Materials Engineering Curriculum: Total 165 credits (150 graded + 15 non-graded)"),
            ("Materials Engineering", "Total Graded Credits (Revised 2025 Scheme)", "142", "credits", "2025_onwards", None, 1, "Materials Engineering Revised 2025: Graded Credits = 142"),

            # Bioscience and Bioengineering
            ("Bioscience and Bioengineering", "Total Degree Credits (Revised 2025 Scheme)", "154 credits (142 graded + 12 audit)", "credits", "2025_onwards", None, 1, "Bioengineering Revised UG Curriculum 2025: Total 154 credits (142 graded + 12 audit)"),
            ("Bioscience and Bioengineering", "Total Degree Credits (Earlier Scheme 2019-2024)", "165 credits (150 graded + 15 non-graded/audit)", "credits", "2019_2024", None, 1, "Bioengineering Curriculum: Total 165 credits (150 graded + 15 non-graded)"),
            ("Bioscience and Bioengineering", "Total Graded Credits (Revised 2025 Scheme)", "142", "credits", "2025_onwards", None, 1, "Bioengineering Revised 2025: Graded Credits = 142"),

            # 26. Department Postgraduate Degree Catalogs
            ("Computer Science and Engineering", "Postgraduate Degree Programs", "M.Tech. in Computer Science and Engineering, M.Tech. in Artificial Intelligence, Dual Degree (M.Tech.-Ph.D.), and Ph.D. in CSE", "catalog", "General", None, 1, "Department of CSE: Offers M.Tech. in CSE, M.Tech. in AI, Dual Degree M.Tech-Ph.D., and Ph.D. programs"),
            ("Electrical Engineering", "Postgraduate Degree Programs", "M.Tech. in Sensors and IoT, Cyber-Physical Systems, Communication and Signal Processing, Power and Energy Systems, Microelectronics and VLSI, Dual Degree (M.Tech.-Ph.D.), and Ph.D.", "catalog", "General", None, 1, "Department of EE: Offers specialized M.Tech programs in Sensors & IoT, CPS, Comm & DSP, Power Systems, Microelectronics & VLSI, and Ph.D."),
            ("School of Artificial Intelligence and Data Science", "Degree Programs and Thrusts", "B.Tech. in AIDE, M.Tech. in AI and Data Science, Ph.D. in AI and Data Science; Thrusts: Deep Learning, Computer Vision, Natural Language Processing, Robotics, Generative AI", "catalog", "General", None, 1, "School of AIDE: Degree programs include B.Tech in AIDE, M.Tech in AI & Data Science, and Ph.D. with core research thrusts in Deep Learning, CV, NLP, Robotics"),
            ("Mechanical Engineering", "Postgraduate Degree Programs", "M.Tech. in Advanced Manufacturing, M.Tech. in Mechanical Systems Design, M.Tech. in Thermo-Fluid Engineering, and Ph.D. in Mechanical Engineering", "catalog", "General", None, 1, "Department of ME: Offers specialized M.Tech in Advanced Manufacturing, Mechanical Systems Design, Thermo-Fluid Engineering, and Ph.D."),
            ("Bioscience and Bioengineering", "Postgraduate Degree Programs", "M.Sc. in Bioscience, M.Tech. in Biotechnology, Dual Degree (M.Tech.-Ph.D.), and Ph.D. in Bioscience and Bioengineering", "catalog", "General", None, 1, "Department of BSBE: Offers M.Sc. in Bioscience, M.Tech. in Biotechnology, Dual Degree, and Ph.D. tracks"),
            ("Chemistry", "Degree Programs", "M.Sc. in Chemistry and Ph.D. in Chemistry (Organic, Inorganic, Physical, Materials, and Computational Chemistry)", "catalog", "General", None, 1, "Department of Chemistry: Offers 2-year M.Sc. in Chemistry and comprehensive Ph.D. research programs"),
            ("Physics", "Degree Programs", "M.Sc. in Physics, M.Tech. in Materials Engineering (interdisciplinary), and Ph.D. in Physics (Condensed Matter Physics, Quantum Information, Optics, High Energy Physics)", "catalog", "General", None, 1, "Department of Physics: Conducts M.Sc. in Physics, interdisciplinary M.Tech in Materials, and Ph.D. programs"),
            ("Mathematics", "Degree Programs", "M.Sc. in Mathematics and Ph.D. in Mathematics / Data Science (Algebra, Analysis, Applied Mathematics, Computational Mathematics)", "catalog", "General", None, 1, "Department of Mathematics: Offers 2-year M.Sc. in Mathematics and Ph.D. research across pure and applied mathematics"),

            # 27. Institute Administrative Leadership
            ("Institute Leadership", "Director, IIT Jodhpur", "Prof. Avinash Kumar Agarwal", "person", "Current", None, 1, "Office of the Director / BoG Resolution: Prof. Avinash Kumar Agarwal is the Director of IIT Jodhpur"),
            ("Institute Leadership", "Registrar, IIT Jodhpur", "Dr. Hari Om Yadav (Statutory / Officiating Registrar)", "person", "Current", None, 1, "Office of the Registrar / BoG Minutes: Dr. Hari Om Yadav is the Registrar of IIT Jodhpur"),
            ("Institute Leadership", "Chairman, Board of Governors (BoG)", "Sh. A. S. Kiran Kumar", "person", "Current", None, 1, "BoG Resolution No. 37.1: Sh. A. S. Kiran Kumar is the Chairman, Board of Governors of IIT Jodhpur"),
            ("Institute Leadership", "Dean of Academic Affairs (DOAA)", "Dean (Academic Affairs)", "office", "Current", None, 1, "Office of Academic Affairs: Headed by the Dean of Academic Affairs (DOAA)"),
            ("Institute Leadership", "Dean of Research and Development (DORD)", "Dean (Research & Development)", "office", "Current", None, 1, "Office of R&D: Headed by the Dean of Research and Development (DORD)"),
            ("Institute Leadership", "Dean of Students Affairs (DOSA)", "Dean (Students Affairs)", "office", "Current", None, 1, "Office of Students Affairs: Headed by the Dean of Students Affairs (DOSA)"),
            ("Institute Leadership", "Dean of Faculty Affairs (DOFA)", "Prof. Neeraj Jain", "person", "Current", None, 1, "Office of Faculty Affairs: Prof. Neeraj Jain is the Dean of Faculty Affairs"),
            ("Institute Leadership", "Dean of Administration (DOAD)", "Prof. Shree Prakash Tiwari", "person", "Current", None, 1, "Office of Administration: Prof. Shree Prakash Tiwari is the Dean of Administration"),

            # 28. Campus Services & Facilities
            ("Primary Health Center", "Medical Services and Infrastructure", "24x7 medical clinic, outpatient department (OPD), resident medical officers, nursing staff, emergency observation beds, automated testing pharmacy, and BLS institute ambulance services", "facility", "General", None, 1, "IIT Jodhpur PHC: Round-the-clock emergency medical services, OPD consultations, observation beds, diagnostic testing, pharmacy, and ambulance facilities"),

            # 29. Centers of Excellence & Interdisciplinary Centers
            ("Centers of Excellence", "Interdisciplinary Centers", "AyurTech Center of Excellence (Integrative Medicine & Ayurvedic Tech), Center for Clean Energy, Center for Space Technology, Center for Technology Foresight and Policy (CTFP), and Center for Digital Humanities", "centers", "General", None, 1, "IIT Jodhpur Centers of Excellence: AyurTech CoE, Clean Energy CoE, Space Technology CoE, Digital Humanities CoE, and Center for Technology Foresight and Policy"),

            # 30. Grading System & Numerical Grade Points (Table 5.1)
            ("Grading System", "Grade Point Scale (Table 5.1)", "A* = 10, A = 10, A- = 9, B = 8, B- = 7, C = 6, C- = 5, D = 4, E = 2, F = 0", "points", "General", None, 8, "Academic Regulations Table 5.1: Letter Grade and its equivalent Grade Points: A*=10, A=10, A-=9, B=8, B-=7, C=6, C-=5, D=4, E=2, F=0"),
            ("Grading System", "Grade 'A*' Equivalent Points", "10", "points", "General", None, 8, "Academic Regulations Table 5.1: Grade A* carries 10 grade points (Exceptional performance)"),
            ("Grading System", "Grade 'A' Equivalent Points", "10", "points", "General", None, 8, "Academic Regulations Table 5.1: Grade A carries 10 grade points (Outstanding)"),
            ("Grading System", "Grade 'B' Equivalent Points", "8", "points", "General", None, 8, "Academic Regulations Table 5.1: Grade B carries 8 grade points (Very Good)"),
            ("Grading System", "Grade 'C' Equivalent Points", "6", "points", "General", None, 8, "Academic Regulations Table 5.1: Grade C carries 6 grade points (Average)"),
            ("Grading System", "Grade 'D' Equivalent Points", "4", "points", "General", None, 8, "Academic Regulations Table 5.1: Grade D carries 4 grade points (Marginal passing grade)"),
            ("Grading System", "Grade 'E' Equivalent Points", "2", "points", "General", None, 8, "Academic Regulations Table 5.1: Grade E carries 2 grade points (Poor, additional examination permitted)"),
            ("Grading System", "Grade 'F' Equivalent Points", "0", "points", "General", None, 8, "Academic Regulations Table 5.1: Grade F carries 0 grade points (Fail, mandatory repetition of core courses)"),
            ("Grading System", "Special Non-Graded Letter Grades", "S = Satisfactory, X = Thesis Continuation, I = Incomplete, U = Unsatisfactory, W = Withdrawn", "grades", "General", None, 8, "Academic Regulations Table 5.1: Special letter grades: S (Satisfactory), X (Thesis Continuation), I (Incomplete), U (Unsatisfactory), W (Withdrawn)"),

            # 31. Postgraduate Degree Durations (M.Tech.)
            ("Postgraduate Regulations", "M.Tech. Normal Designed Duration", "4 registered semesters (2 years)", "semesters", "General", None, 21, "Academic Regulations Section 21.5: Normal designed duration for full-time regular M.Tech is 4 registered semesters (2 years)"),
            ("Postgraduate Regulations", "M.Tech. Maximum Completion Duration", "12 registered semesters (which may include maximum 6 semesters of Registration kept Alive - RKA) / 6 semesters for regular full-time", "semesters", "General", None, 21, "Academic Regulations Section 21.5: Maximum duration of M.Tech is 12 registered semesters (inclusive of up to 6 RKA semesters)"),

            # 32. Dual Degree (B.Tech.-M.Tech.) Conversion
            ("Undergraduate Regulations", "Dual Degree Conversion CGPA Requirement", "Minimum CGPA of 6.0 at end of 7th semester (Section 12.4) or CGPA >= 8.0 (7.5 for SC/ST) at end of 6th semester", "cgpa", "General", None, 19, "Academic Regulations Section 12.4: B.Tech student opted for major/minor specialization with CGPA 6 or more at end of 7th semester may convert to Dual Degree at beginning of 8th semester"),
            ("Undergraduate Regulations", "Dual Degree Conversion Earned Credits", "Minimum 134 earned credits at end of 7th semester", "credits", "General", None, 19, "Academic Regulations Section 12.4: Student must have completed at least 134 earned credits at the end of the seventh semester to convert to B.Tech.-M.Tech. Dual Degree"),

            # 33. Ph.D. Leave Entitlements
            ("Ph.D. Regulations", "Ph.D. Casual Leave Entitlement", "8 days per calendar year (or up to 7 days per semester)", "days/year", "General", None, 21, "Ph.D. Regulations Section 14 / Section 7.1: Regular Ph.D. scholars are entitled to 8 days of Casual Leave (CL) per calendar year"),
            ("Ph.D. Regulations", "Ph.D. Medical Leave Entitlement", "15 days per calendar year (supported by medical certificate)", "days/year", "General", None, 21, "Ph.D. Regulations Section 14 / Section 7.1: Regular Ph.D. scholars are entitled to 15 days of Medical Leave per calendar year"),
            ("Ph.D. Regulations", "Ph.D. Vacation Leave Entitlement", "15 days per semester in addition to Public Holidays", "days/sem", "General", None, 21, "Ph.D. Regulations Section 7.1: Ph.D. students are entitled to vacation leave up to 15 days per semester"),

            # 34. Senate Undergraduate Committee (SUGC)
            ("Academic Governance", "Senate Undergraduate Committee (SUGC) Role and Mandate", "Standing committee of the Senate responsible for undergraduate academic administration, curriculum proposals, registration load exceptions (under-load, overload), add/drop courses, academic probation review, branch change recommendations, dual degree conversions, and course de-registration adjudications", "mandate", "General", None, 1, "Senate Governance: The SUGC is the standing committee of the Senate responsible for undergraduate curriculum, course administration, registration exceptions, probation, and academic regulations"),
        ]

        claim_objs = []
        for ent, attr, val, unit, cohort, sem, pg, excerpt in statutory_claims:
            claim_objs.append(AtomicClaim(
                document_id=doc_id,
                entity=ent,
                attribute=attr,
                value=val,
                unit=unit,
                scope_cohort=cohort,
                scope_semester=sem,
                page_number=pg,
                raw_excerpt=excerpt,
                confidence_score=0.99
            ))

        db.bulk_save_objects(claim_objs)
        db.commit()
        logger.info(f"Successfully populated {len(claim_objs)} verified baseline atomic claims.")

    finally:
        db.close()


if __name__ == "__main__":
    init_database()
    migrate_inventory()
    populate_baseline_atomic_claims()
