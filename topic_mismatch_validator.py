"""
topic_mismatch_validator.py
===========================
Validates that generated assistant responses align with original user query intent,
domain, and retrieved evidence, rejecting and intercepting cross-domain hallucinations
or privacy disclaimers generated for academic questions.
"""

import re
from typing import Tuple, List, Optional
from query_understanding import QueryProfile, QueryDomain, QueryIntent
from evidence_models import EvidenceAssessment, AtomicClaim


class TopicMismatchValidator:
    """
    Evaluates whether the generated text addresses the user's information need
    or whether a severe topic shift/hallucination/context leakage has occurred.
    """

    @staticmethod
    def validate(
        profile: QueryProfile,
        generated_text: str,
        assessment: Optional[EvidenceAssessment] = None
    ) -> Tuple[bool, str]:
        """
        Returns (is_valid, reason).
        If is_valid is False, the response must be rejected or substituted with
        strictly grounded fallback synthesis.
        """
        gen_lower = generated_text.lower()

        # 1. Critical Failure Mode B check:
        # If the user asked an academic question (e.g. minor, specialization, credits, branch change, attendance),
        # but the generated text talks about faculty salary, personal remuneration, or privacy policies.
        if profile.domain in [QueryDomain.CURRICULUM, QueryDomain.ACADEMIC_REGULATIONS, QueryDomain.ADMISSIONS]:
            privacy_indicators = [
                "personal financial records",
                "individual salary payouts",
                "remuneration figures",
                "property returns",
                "strictly protected from disclosure",
                "scale of pay | faculty positions",
                "7th central pay commission",
                "academic pay level 10"
            ]
            if any(pi in gen_lower for pi in privacy_indicators):
                return False, f"Severe Topic Mismatch: Academic query in domain '{profile.domain.value}' received a personal financial privacy disclaimer."

        # 2. Critical Failure Mode A check:
        # If the user asked about branch change eligibility, but the answer claims no information
        # when authoritative Section 16 chunks were retrieved.
        if "branch_change_timing" in profile.target_attributes:
            if "do not contain authoritative information" in gen_lower or "no matching claims" in gen_lower:
                if assessment and assessment.supporting_candidates:
                    has_sec16 = any("branch change" in c.chunk.get("text", "").lower() for c in assessment.supporting_candidates)
                    if has_sec16:
                        return False, "Evidence Inversion: Branch change regulations exist in retrieved context but response generated a no-evidence refusal."

        # 3. Subject Entity Alignment check:
        # If query targeted a specific engineering programme (e.g. Materials Engineering),
        # verify that the answer mentions that programme and doesn't substitute another programme exclusively.
        if profile.subject_entity and profile.domain == QueryDomain.CURRICULUM:
            ent_key = profile.subject_entity.lower()
            short_ent = ent_key.split()[0]
            # Check if answer discusses the requested department
            if short_ent not in gen_lower:
                # Check if it exclusively discusses a foreign department
                foreign_departments = ["mechanical", "electrical", "civil", "chemical", "materials", "bioscience"]
                foreign_mentions = [fd for fd in foreign_departments if fd in gen_lower and fd != short_ent]
                if foreign_mentions and len(foreign_mentions) == 1:
                    return False, f"Entity Drift: Query asked for '{profile.subject_entity}' but generated response exclusively discusses '{foreign_mentions[0]}'."

        # 4. Attribute Relevance check:
        # If query asked for specific quantitative values (e.g. credits or CGPA), check if numerical figures or requirements appear.
        if profile.intent == QueryIntent.QUANTITATIVE and profile.domain == QueryDomain.CURRICULUM:
            has_numbers = bool(re.search(r"\b\d+\b", gen_lower))
            if not has_numbers and "not explicitly" not in gen_lower:
                return False, "Quantitative Omission: Query requested numerical credits but no numbers were returned."

        return True, "Response aligns with query domain, subject entity, and requested attributes."


topic_validator = TopicMismatchValidator()
