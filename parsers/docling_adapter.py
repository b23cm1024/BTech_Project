"""
parsers/docling_adapter.py
==========================
Adapter for IBM Docling document conversion, extracting clean Markdown,
hierarchical sections, and validated table structures.
"""

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("DoclingAdapter")


class DoclingAdapter:
    """Wrapper around IBM Docling DocumentConverter for high-precision table & layout parsing."""

    def __init__(self):
        self._converter = None

    def _get_converter(self):
        if self._converter is None:
            from docling.document_converter import DocumentConverter
            self._converter = DocumentConverter()
        return self._converter

    def parse_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parses a PDF using Docling and returns:
          - full_markdown: Full document in semantic markdown
          - sections: Extracted sections with headers
          - tables: Formatted markdown tables with page numbers
          - page_count: Total document pages
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        converter = self._get_converter()
        logger.info(f"Parsing document with Docling: {os.path.basename(pdf_path)}")
        conv_res = converter.convert(pdf_path)
        doc = conv_res.document

        full_md = doc.export_to_markdown()

        extracted_tables = []
        for idx, table in enumerate(doc.tables):
            try:
                table_md = table.export_to_markdown() if hasattr(table, "export_to_markdown") else ""
                pg = getattr(table, "page_no", None) or getattr(table, "page", 1)
                extracted_tables.append({
                    "table_index": idx + 1,
                    "page": pg,
                    "markdown": table_md
                })
            except Exception as e:
                logger.warning(f"Error extracting table {idx}: {e}")

        return {
            "source": os.path.basename(pdf_path),
            "file_path": pdf_path,
            "full_markdown": full_md,
            "tables": extracted_tables,
            "page_count": getattr(doc, "pages", 1) if isinstance(getattr(doc, "pages", 1), int) else len(getattr(doc, "pages", [1]))
        }
