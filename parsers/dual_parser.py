"""
parsers/dual_parser.py
======================
Dual-path document parsing pipeline combining Docling and MinerU.
Parses PDFs, decomposes markdown into semantic sections with header paths,
and stores sections in the relational store (DocumentSection).
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional
from parsers.docling_adapter import DoclingAdapter
from parsers.mineru_adapter import MinerUAdapter
from database.connection import SessionLocal
from database.models import Document, DocumentSection

logger = logging.getLogger("DualParser")


class DualParser:
    """Orchestrates Docling and MinerU for complete multimodal extraction."""

    def __init__(self):
        self.docling = DoclingAdapter()
        self.mineru = MinerUAdapter()

    def parse_and_index_document(self, pdf_path: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Parses PDF using Docling/MinerU dual path and persists sections to relational store.
        """
        filename = os.path.basename(pdf_path)
        logger.info(f"Processing PDF via Dual Parser: {filename}")

        # 1. Primary parsing via Docling
        docling_result = self.docling.parse_document(pdf_path)
        full_md = docling_result["full_markdown"]
        tables = docling_result["tables"]

        # 2. Extract sections based on markdown headings
        sections_data = self._split_markdown_sections(full_md)

        # 3. Save sections to database if document_id is provided
        if document_id:
            db = SessionLocal()
            try:
                # Remove existing sections for this document
                db.query(DocumentSection).filter(DocumentSection.document_id == document_id).delete()

                section_objs = []
                for s in sections_data:
                    sec = DocumentSection(
                        document_id=document_id,
                        section_number=s.get("section_number"),
                        heading_title=s.get("title", "Untitled Section")[:255],
                        heading_path=s.get("heading_path", "")[:500],
                        page_start=s.get("page", 1),
                        page_end=s.get("page", 1),
                        raw_markdown=s.get("content", ""),
                        modality="table" if s.get("is_table") else "text"
                    )
                    section_objs.append(sec)

                db.bulk_save_objects(section_objs)
                db.commit()
                logger.info(f"Saved {len(section_objs)} sections for document {document_id}")
            except Exception as e:
                db.rollback()
                logger.error(f"Error persisting document sections: {e}")
            finally:
                db.close()

        return {
            "filename": filename,
            "section_count": len(sections_data),
            "table_count": len(tables),
            "full_markdown_length": len(full_md)
        }

    def _split_markdown_sections(self, markdown_text: str) -> List[Dict[str, Any]]:
        """Decomposes markdown into structured sections tracking heading hierarchy."""
        lines = markdown_text.split("\n")
        sections = []
        current_title = "Document Overview"
        current_heading_path = ["Document Overview"]
        current_lines = []
        sec_num = None

        heading_re = re.compile(r'^(#{1,6})\s+(.*)$')
        sec_num_re = re.compile(r'^(?:Section\s+)?([0-9]+(?:\.[0-9]+)*)\s*(.*)', re.IGNORECASE)

        for line in lines:
            h_match = heading_re.match(line)
            if h_match:
                # Save previous section
                if current_lines:
                    content = "\n".join(current_lines).strip()
                    if content:
                        sections.append({
                            "title": current_title,
                            "heading_path": " > ".join(current_heading_path),
                            "section_number": sec_num,
                            "content": content,
                            "is_table": "|" in content and "---" in content
                        })
                    current_lines = []

                level = len(h_match.group(1))
                heading_raw = h_match.group(2).strip()

                # Detect section number if present
                sn_match = sec_num_re.match(heading_raw)
                if sn_match:
                    sec_num = sn_match.group(1)
                    current_title = heading_raw
                else:
                    sec_num = None
                    current_title = heading_raw

                # Maintain heading path
                while len(current_heading_path) >= level:
                    current_heading_path.pop()
                current_heading_path.append(current_title)
            else:
                current_lines.append(line)

        # Append last section
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append({
                    "title": current_title,
                    "heading_path": " > ".join(current_heading_path),
                    "section_number": sec_num,
                    "content": content,
                    "is_table": "|" in content and "---" in content
                })

        return sections
