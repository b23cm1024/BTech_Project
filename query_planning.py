"""
query_planning.py — Structured Attribute-Level Query Planning & Request Isolation
=================================================================================
Implements V4.2 Requirement 1 (Request-Level Evidence Isolation), Requirement 2
(Structured Attribute-Level Query Planning), Requirement 5 (Numerical Fact Integrity),
and Requirement 6 (Multi-Attribute Completeness).
"""

import uuid
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from query_understanding import QueryProfile, QueryDomain, QueryIntent
from evidence_models import AtomicClaim, RetrievalCandidate, EvidenceAssessment, ApplicabilityScope, Provenance
from source_authority import classify_document_authority, SourceAuthorityLevel, compare_source_precedence


@dataclass
class AttributeEvidenceRequirement:
    """Represents a discrete required attribute in a composite query."""
    attribute_key: str
    description: str
    expected_unit: Optional[str] = None
    supporting_claims: List[AtomicClaim] = field(default_factory=list)
    is_satisfied: bool = False
    resolved_value: Optional[str] = None
    resolved_citation: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class QueryPlan:
    """Structured decomposition plan for an incoming information request."""
    request_id: str
    original_query: str
    primary_subject: str
    applicable_population: str  # "UG", "PG", "PhD", "Faculty", "All"
    requested_attributes: List[str] = field(default_factory=list)
    comparison_dimensions: List[str] = field(default_factory=list)
    expected_answer_structure: str = "tabular"  # "tabular", "comparison_matrix", "prose"
    attribute_requirements: Dict[str, AttributeEvidenceRequirement] = field(default_factory=dict)

    @property
    def is_fully_satisfied(self) -> bool:
        if not self.attribute_requirements:
            return False
        return all(req.is_satisfied for req in self.attribute_requirements.values())

    @property
    def completeness_ratio(self) -> float:
        if not self.attribute_requirements:
            return 0.0
        sat = sum(1 for req in self.attribute_requirements.values() if req.is_satisfied)
        return sat / len(self.attribute_requirements)


@dataclass
class RequestEvidencePackage:
    """
    Immutable, request-scoped snapshot of evidence gathered for this specific request.
    Strictly prevents cross-query state leakage and shared mutable contamination.
    """
    request_id: str
    query_plan: QueryPlan
    candidates: List[RetrievalCandidate] = field(default_factory=list)
    assessment: Optional[EvidenceAssessment] = None
    verified_claims: List[AtomicClaim] = field(default_factory=list)
    policy_resolutions: List[Dict[str, Any]] = field(default_factory=list)


def create_query_plan(query: str, profile: QueryProfile) -> QueryPlan:
    """
    Deconstructs a user query into structured attributes and comparison dimensions.
    """
    req_id = uuid.uuid4().hex[:12]
    q_low = profile.normalized_query.lower()

    # 1. B.Tech Duration & Residence
    if "btech_programme_duration" in profile.target_attributes or (
        any(k in q_low for k in ["duration", "residence", "semesters", "years"]) and
        any(k in q_low for k in ["btech", "b.tech", "undergraduate"]) and
        any(k in q_low for k in ["minimum", "maximum", "permissible", "complete", "requirements"])
    ):
        attrs = {
            "min_residential_requirement": AttributeEvidenceRequirement(
                attribute_key="min_residential_requirement",
                description="Minimum Residential Requirement",
                expected_unit="semesters"
            ),
            "normal_programme_duration": AttributeEvidenceRequirement(
                attribute_key="normal_programme_duration",
                description="Normal Designed B.Tech Programme Duration",
                expected_unit="semesters / years"
            ),
            "max_permissible_duration": AttributeEvidenceRequirement(
                attribute_key="max_permissible_duration",
                description="Maximum Permissible Completion Duration",
                expected_unit="registered semesters / years"
            )
        }
        return QueryPlan(
            request_id=req_id,
            original_query=query,
            primary_subject="Undergraduate B.Tech Programme Duration & Residence Limits",
            applicable_population="UG",
            requested_attributes=list(attrs.keys()),
            comparison_dimensions=["residential_minimum", "normal_duration", "maximum_permissible"],
            expected_answer_structure="tabular",
            attribute_requirements=attrs
        )

    # 2. Assistantship / Fellowship Continuation CGPA
    if "assistantship_continuation_cgpa" in profile.target_attributes or (
        any(k in q_low for k in ["assistantship", "fellowship"]) and
        any(k in q_low for k in ["continuation", "cgpa", "minimum", "threshold", "m.tech", "mtech", "dual degree"])
    ):
        attrs = {
            "mtech_assistantship_cgpa": AttributeEvidenceRequirement(
                attribute_key="mtech_assistantship_cgpa",
                description="M.Tech. Teaching Assistantship Continuation Minimum CGPA",
                expected_unit="CGPA"
            ),
            "dual_degree_assistantship_cgpa": AttributeEvidenceRequirement(
                attribute_key="dual_degree_assistantship_cgpa",
                description="Dual Degree (M.Tech.-Ph.D.) Assistantship Continuation Minimum CGPA",
                expected_unit="CGPA"
            ),
            "phd_fellowship_cgpa": AttributeEvidenceRequirement(
                attribute_key="phd_fellowship_cgpa",
                description="Ph.D. Research Fellowship Continuation Minimum CGPA",
                expected_unit="CGPA"
            ),
            "assistantship_workload": AttributeEvidenceRequirement(
                attribute_key="assistantship_workload",
                description="Weekly Academic Duty Hours for Assistantship",
                expected_unit="hours/week"
            )
        }
        return QueryPlan(
            request_id=req_id,
            original_query=query,
            primary_subject="Postgraduate Assistantship / Fellowship Continuation CGPA Thresholds",
            applicable_population="PG / PhD",
            requested_attributes=list(attrs.keys()),
            comparison_dimensions=["M.Tech.", "Dual Degree (M.Tech.-Ph.D.)", "Ph.D."],
            expected_answer_structure="comparison_matrix",
            attribute_requirements=attrs
        )

    # 3. Faculty Relocation Reimbursement Ceilings
    if "relocation_reimbursement" in profile.target_attributes or (
        "relocation" in q_low and any(k in q_low for k in ["reimbursement", "india", "abroad", "limit", "ceiling", "maximum"])
    ):
        attrs = {
            "relocation_domestic_govt": AttributeEvidenceRequirement(
                attribute_key="relocation_domestic_govt",
                description="Joining from Govt / Autonomous / Similar Institute in India",
                expected_unit="statutory rules (SR-116 / TTA)"
            ),
            "relocation_domestic_other": AttributeEvidenceRequirement(
                attribute_key="relocation_domestic_other",
                description="Joining from Other Institutes / Industry within India",
                expected_unit="INR"
            ),
            "relocation_abroad": AttributeEvidenceRequirement(
                attribute_key="relocation_abroad",
                description="Joining from Abroad",
                expected_unit="INR"
            ),
            "relocation_bond_conditions": AttributeEvidenceRequirement(
                attribute_key="relocation_bond_conditions",
                description="Mandatory Service Bond and Return Timeline Conditions",
                expected_unit="tenure / timeline"
            )
        }
        return QueryPlan(
            request_id=req_id,
            original_query=query,
            primary_subject="Faculty Relocation Expense Reimbursement Ceilings & Rules",
            applicable_population="Faculty",
            requested_attributes=list(attrs.keys()),
            comparison_dimensions=["domestic_govt", "domestic_other", "abroad"],
            expected_answer_structure="comparison_matrix",
            attribute_requirements=attrs
        )

    # 4. CPDA Financial Grant
    if "cpda_financial_grant" in profile.target_attributes or "cpda" in q_low or "cumulative professional development" in q_low:
        attrs = {
            "cpda_total_amount": AttributeEvidenceRequirement(
                attribute_key="cpda_total_amount",
                description="Total Financial Grant Amount",
                expected_unit="INR"
            ),
            "cpda_block_tenure": AttributeEvidenceRequirement(
                attribute_key="cpda_block_tenure",
                description="Block Period Duration",
                expected_unit="years"
            ),
            "cpda_disbursement_type": AttributeEvidenceRequirement(
                attribute_key="cpda_disbursement_type",
                description="Disbursement Mechanism (Reimbursable Basis)",
                expected_unit="policy"
            ),
            "cpda_eligible_activities": AttributeEvidenceRequirement(
                attribute_key="cpda_eligible_activities",
                description="Eligible Professional Development Activities",
                expected_unit="scope"
            )
        }
        return QueryPlan(
            request_id=req_id,
            original_query=query,
            primary_subject="Cumulative Professional Development Allowance (CPDA) Provisions",
            applicable_population="Faculty",
            requested_attributes=list(attrs.keys()),
            comparison_dimensions=["amount", "cycle", "disbursement", "expenses"],
            expected_answer_structure="tabular",
            attribute_requirements=attrs
        )

    # 5. Ph.D. Coursework and Thesis Requirements
    if "phd_coursework_thesis_credits" in profile.target_attributes or (
        any(k in q_low for k in ["phd", "ph.d", "doctoral"]) and
        any(k in q_low for k in ["coursework", "course work", "thesis"]) and
        any(k in q_low for k in ["credit", "credits", "minimum", "requirement"])
    ):
        attrs = {
            "phd_coursework_mtech_entry": AttributeEvidenceRequirement(
                attribute_key="phd_coursework_mtech_entry",
                description="Minimum Coursework Credits (Entrants with M.Tech / M.E. / M.Sc. Eng)",
                expected_unit="credits"
            ),
            "phd_thesis_mtech_entry": AttributeEvidenceRequirement(
                attribute_key="phd_thesis_mtech_entry",
                description="Minimum Thesis Credits (Entrants with M.Tech / M.E. / M.Sc. Eng)",
                expected_unit="credits"
            ),
            "phd_coursework_btech_entry": AttributeEvidenceRequirement(
                attribute_key="phd_coursework_btech_entry",
                description="Minimum Coursework Credits (Direct Entrants with B.Tech / B.E. / M.Sc. Science)",
                expected_unit="credits"
            ),
            "phd_thesis_btech_entry": AttributeEvidenceRequirement(
                attribute_key="phd_thesis_btech_entry",
                description="Minimum Thesis Credits (Direct Entrants with B.Tech / B.E. / M.Sc. Science)",
                expected_unit="credits"
            )
        }
        return QueryPlan(
            request_id=req_id,
            original_query=query,
            primary_subject="Ph.D. Academic Credit Requirements by Prior Qualifying Degree",
            applicable_population="PhD",
            requested_attributes=list(attrs.keys()),
            comparison_dimensions=["M.Tech_Entrants", "Direct_B.Tech_Science_Entrants"],
            expected_answer_structure="comparison_matrix",
            attribute_requirements=attrs
        )

    # 6. Postgraduate (M.Tech.) Registration Limits
    if "mtech_registration_limits" in profile.target_attributes or (
        any(k in q_low for k in ["mtech", "m.tech", "postgraduate", "pg"]) and
        any(k in q_low for k in ["credit", "registration limit", "load", "maximum credit", "normal semester"])
    ):
        attrs = {
            "mtech_regular_max_credits": AttributeEvidenceRequirement(
                attribute_key="mtech_regular_max_credits",
                description="Regular Semester Maximum Credits (Full-time Regular/Sponsored M.Tech)",
                expected_unit="credits"
            ),
            "mtech_regular_min_credits": AttributeEvidenceRequirement(
                attribute_key="mtech_regular_min_credits",
                description="Regular Semester Minimum Credits (Full-time Regular/Sponsored M.Tech)",
                expected_unit="credits"
            ),
            "mtech_summer_max_credits": AttributeEvidenceRequirement(
                attribute_key="mtech_summer_max_credits",
                description="Summer Term Maximum Credits (Full-time M.Tech)",
                expected_unit="credits"
            )
        }
        return QueryPlan(
            request_id=req_id,
            original_query=query,
            primary_subject="Postgraduate (M.Tech.) Semester Registration Limits",
            applicable_population="PG",
            requested_attributes=list(attrs.keys()),
            comparison_dimensions=["Regular Semester", "Summer Term", "Other Categories"],
            expected_answer_structure="tabular",
            attribute_requirements=attrs
        )

    # 7. Research Initiation Seed Grant
    if "research_seed_grant" in profile.target_attributes or (
        any(k in q_low for k in ["seed grant", "research initiation grant"])
    ):
        attrs = {
            "seed_grant_sciences_eng": AttributeEvidenceRequirement(
                attribute_key="seed_grant_sciences_eng",
                description="Maximum Seed Grant for Sciences and Engineering",
                expected_unit="INR"
            ),
            "seed_grant_humanities": AttributeEvidenceRequirement(
                attribute_key="seed_grant_humanities",
                description="Maximum Seed Grant for Humanities & Social Sciences",
                expected_unit="INR"
            ),
            "seed_grant_tenure": AttributeEvidenceRequirement(
                attribute_key="seed_grant_tenure",
                description="Tenure / Eligibility Window",
                expected_unit="years"
            )
        }
        return QueryPlan(
            request_id=req_id,
            original_query=query,
            primary_subject="Faculty Research Initiation Seed Grant Ceilings",
            applicable_population="Faculty",
            requested_attributes=list(attrs.keys()),
            comparison_dimensions=["Sciences/Engineering", "Humanities & Social Sciences"],
            expected_answer_structure="comparison_matrix",
            attribute_requirements=attrs
        )

    # 8. Ph.D. Comprehensive Examination Requirements & Attempts
    if "phd_comprehensive_exam" in profile.target_attributes or (
        any(k in q_low for k in ["comprehensive exam", "comprehensive examination", "candidacy requirement"])
    ):
        attrs = {
            "comprehensive_deadline_regular": AttributeEvidenceRequirement(
                attribute_key="comprehensive_deadline_regular",
                description="Completion Deadline for Regular Ph.D. Students",
                expected_unit="semesters"
            ),
            "comprehensive_max_attempts": AttributeEvidenceRequirement(
                attribute_key="comprehensive_max_attempts",
                description="Maximum Permissible Attempts for Comprehensive Examination",
                expected_unit="attempts"
            ),
            "comprehensive_deadline_part_time": AttributeEvidenceRequirement(
                attribute_key="comprehensive_deadline_part_time",
                description="Completion Deadline for Part-Time / Direct Ph.D. Students",
                expected_unit="semesters"
            )
        }
        return QueryPlan(
            request_id=req_id,
            original_query=query,
            primary_subject="Ph.D. Comprehensive Examination Statutory Deadlines & Permissible Attempts",
            applicable_population="PhD",
            requested_attributes=list(attrs.keys()),
            comparison_dimensions=["deadline", "attempts"],
            expected_answer_structure="tabular",
            attribute_requirements=attrs
        )

    # 9. Undergraduate Course Overload per-semester limit
    if "course_overload_credits_limit" in profile.target_attributes or (
        any(k in q_low for k in ["course overload", "overload credit", "additional credit"]) and any(k in q_low for k in ["6th", "sixth", "per semester", "semester"])
    ):
        attrs = {
            "overload_max_credits_per_sem": AttributeEvidenceRequirement(
                attribute_key="overload_max_credits_per_sem",
                description="Maximum Overload Credits Permitted Per Semester",
                expected_unit="credits/semester"
            ),
            "overload_eligibility_criteria": AttributeEvidenceRequirement(
                attribute_key="overload_eligibility_criteria",
                description="Minimum Earned Credits and CGPA Prerequisite",
                expected_unit="thresholds"
            )
        }
        return QueryPlan(
            request_id=req_id,
            original_query=query,
            primary_subject="Undergraduate Course Overload Allowance Starting From 6th Semester",
            applicable_population="UG",
            requested_attributes=list(attrs.keys()),
            comparison_dimensions=["credits", "eligibility"],
            expected_answer_structure="tabular",
            attribute_requirements=attrs
        )

    # 10. Academic Probation Termination
    if "academic_probation_termination" in profile.target_attributes or (
        any(k in q_low for k in ["academic probation", "probation"]) and any(k in q_low for k in ["termination", "terminate", "consecutive"])
    ):
        attrs = {
            "probation_termination_consecutive_semesters": AttributeEvidenceRequirement(
                attribute_key="probation_termination_consecutive_semesters",
                description="Consecutive Semesters on Academic Probation Resulting in Termination",
                expected_unit="consecutive semesters"
            )
        }
        return QueryPlan(
            request_id=req_id,
            original_query=query,
            primary_subject="Academic Probation Statutory Termination Threshold",
            applicable_population="UG",
            requested_attributes=list(attrs.keys()),
            comparison_dimensions=["consecutive_semesters"],
            expected_answer_structure="tabular",
            attribute_requirements=attrs
        )

    # Default Single-Attribute Query Plan
    attr_key = profile.target_attributes[0] if profile.target_attributes else "statutory_provision"
    attrs = {
        attr_key: AttributeEvidenceRequirement(
            attribute_key=attr_key,
            description=f"Authoritative statutory requirement for {attr_key.replace('_', ' ')}",
            expected_unit="statutory policy"
        )
    }
    return QueryPlan(
        request_id=req_id,
        original_query=query,
        primary_subject=profile.subject_entity or "IIT Jodhpur Institutional Policy",
        applicable_population=profile.degree_level,
        requested_attributes=[attr_key],
        comparison_dimensions=[],
        expected_answer_structure="tabular",
        attribute_requirements=attrs
    )
