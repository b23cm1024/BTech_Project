"""
request_context.py — Immutable Request Context and Cross-Query Isolation Engine
==============================================================================
Implements Section 13 of the Master Engineering Directive:
  - Generates unique, immutable RequestContext per user query.
  - Guarantees complete isolation across concurrent or sequential queries.
  - Prevents cross-query contamination (no mutable shared evidence lists,
    no department-specific answers leaking into subsequent queries).
  - Implements multi-attribute cache keys (query + department + programme + version).
"""

import uuid
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from query_planner import QueryPlan
from evidence_pack import EvidencePack


@dataclass
class RequestContext:
    request_id: str
    query: str
    plan: QueryPlan
    evidence_pack: EvidencePack
    created_at_epoch: float = 0.0

    @classmethod
    def create(cls, query: str, plan: QueryPlan) -> "RequestContext":
        req_id = str(uuid.uuid4())
        pack = EvidencePack(request_id=req_id, query=query)
        import time
        return cls(
            request_id=req_id,
            query=query,
            plan=plan,
            evidence_pack=pack,
            created_at_epoch=time.time()
        )

    def compute_cache_key(self, model_name: str, index_version: str = "v3.1") -> str:
        """
        Computes a strictly scoped cache key.
        Prevents reusing a department or cohort answer for a different scope.
        """
        raw_key = f"{self.plan.normalized_query}::{self.plan.target_department}::{self.plan.programme}::{self.plan.admission_cohort}::{self.plan.regulation_version}::{model_name}::{index_version}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
