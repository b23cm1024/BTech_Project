"""
database/models.py
==================
Relational models for documents, sections, atomic claims, and execution traces.
Compatible with PostgreSQL (production) and SQLite (development).
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Text,
    DateTime,
    ForeignKey,
    JSON,
    Boolean,
    Index
)
from sqlalchemy.orm import relationship
from database.connection import Base


def generate_uuid_str():
    return uuid.uuid4().hex


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=generate_uuid_str)
    filename = Column(String(255), nullable=False, index=True)
    canonical_title = Column(String(255), nullable=False, index=True)
    category = Column(String(64), nullable=False, index=True)
    programme = Column(String(128), nullable=True, index=True)
    cohort = Column(String(64), nullable=True, index=True)
    effective_year = Column(Integer, nullable=True)
    source_url = Column(Text, nullable=True)
    sha256_hash = Column(String(64), unique=True, nullable=True, index=True)
    page_count = Column(Integer, default=1)
    is_indexed = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    sections = relationship("DocumentSection", back_populates="document", cascade="all, delete-orphan")
    claims = relationship("AtomicClaim", back_populates="document", cascade="all, delete-orphan")


class DocumentSection(Base):
    __tablename__ = "document_sections"

    id = Column(String(36), primary_key=True, default=generate_uuid_str)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    section_number = Column(String(32), nullable=True, index=True)
    heading_title = Column(Text, nullable=False)
    heading_path = Column(Text, nullable=False)
    page_start = Column(Integer, nullable=False)
    page_end = Column(Integer, nullable=False)
    raw_markdown = Column(Text, nullable=False)
    modality = Column(String(32), default="text")

    document = relationship("Document", back_populates="sections")
    claims = relationship("AtomicClaim", back_populates="section", cascade="all, delete-orphan")


class AtomicClaim(Base):
    __tablename__ = "atomic_claims"

    id = Column(String(36), primary_key=True, default=generate_uuid_str)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    section_id = Column(String(36), ForeignKey("document_sections.id", ondelete="CASCADE"), nullable=True, index=True)
    entity = Column(String(128), nullable=False, index=True)
    attribute = Column(String(128), nullable=False, index=True)
    value = Column(Text, nullable=False)
    unit = Column(String(32), nullable=True)
    scope_cohort = Column(String(64), nullable=True, index=True)
    scope_semester = Column(Integer, nullable=True)
    page_number = Column(Integer, nullable=False)
    raw_excerpt = Column(Text, nullable=False)
    confidence_score = Column(Float, default=0.95)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    document = relationship("Document", back_populates="claims")
    section = relationship("DocumentSection", back_populates="claims")


class QueryExecutionTraceRecord(Base):
    __tablename__ = "query_execution_traces"

    id = Column(String(36), primary_key=True, default=generate_uuid_str)
    request_id = Column(String(32), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    original_query = Column(Text, nullable=False)
    normalized_query = Column(Text, nullable=True)
    domain = Column(String(64), nullable=False, index=True)
    intent = Column(String(64), nullable=False, index=True)
    subject_entity = Column(String(128), nullable=True)
    retrieved_chunk_count = Column(Integer, default=0)
    evidence_state = Column(String(64), nullable=True)
    evidence_confidence = Column(Float, default=0.0)
    llm_response = Column(Text, nullable=True)
    latency_breakdown_ms = Column(JSON, nullable=True)
    topic_mismatch_detected = Column(Boolean, default=False)
