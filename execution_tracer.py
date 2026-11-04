"""
execution_tracer.py
===================
Request-Level Context Isolation and End-to-End Execution Tracing for IIT Jodhpur RAG System.

Tracks every stage of request lifecycle:
Query Profile -> Discovered Docs -> Candidate Retrieval -> Reranking -> 
Evidence Validation -> Assembled Context -> LLM Generation -> Topic-Mismatch Validation -> Rendered Output.
"""

import os
import json
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class StageTiming:
    query_understanding_ms: float = 0.0
    document_discovery_ms: float = 0.0
    passage_retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    evidence_validation_ms: float = 0.0
    llm_generation_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class ExecutionTrace:
    request_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    original_query: str = ""
    normalized_query: str = ""
    domain: str = ""
    intent: str = ""
    subject_entity: Optional[str] = None
    target_attributes: List[str] = field(default_factory=list)
    search_rewrites: List[str] = field(default_factory=list)
    discovered_documents: List[str] = field(default_factory=list)
    retrieved_candidate_count: int = 0
    candidate_summary: List[Dict[str, Any]] = field(default_factory=list)
    rejected_candidates: List[Dict[str, Any]] = field(default_factory=list)
    reranked_top_candidates: List[Dict[str, Any]] = field(default_factory=list)
    evidence_state: str = ""
    evidence_confidence: float = 0.0
    evidence_reasoning: str = ""
    satisfied_attributes: List[str] = field(default_factory=list)
    missing_attributes: List[str] = field(default_factory=list)
    context_chunk_count: int = 0
    generation_prompt_length: int = 0
    raw_llm_response: str = ""
    topic_mismatch_detected: bool = False
    topic_mismatch_reason: str = ""
    final_response_preview: str = ""
    timings: StageTiming = field(default_factory=StageTiming)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class ExecutionTracer:
    """Singleton tracer that records and persists request-level execution logs."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ExecutionTracer, cls).__new__(cls)
            cls._instance._traces = {}
            cls._instance._trace_dir = os.path.join(os.path.dirname(__file__), "data", "traces")
            os.makedirs(cls._instance._trace_dir, exist_ok=True)
        return cls._instance

    def create_trace(self, request_id: str, original_query: str) -> ExecutionTrace:
        trace = ExecutionTrace(request_id=request_id, original_query=original_query)
        self._traces[request_id] = trace
        return trace

    def get_trace(self, request_id: str) -> Optional[ExecutionTrace]:
        return self._traces.get(request_id)

    def save_trace(self, trace: ExecutionTrace) -> str:
        """Persists execution trace to disk without exposing it to the UI."""
        file_path = os.path.join(self._trace_dir, f"trace_{trace.request_id}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(trace.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ExecutionTracer] Could not save trace file: {e}")
        return file_path


tracer = ExecutionTracer()
