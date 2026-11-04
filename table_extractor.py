"""
table_extractor.py — Structured Table Extraction with Cross-Page Stitcher (V3)
===============================================================================
Replaces the fragmented table extraction in ingest.py.

Key improvements over V1:
  1. Uses pdfplumber for accurate cell-level table extraction (not PyMuPDF blocks)
  2. Cross-page stitcher: detects when a table continues across a page boundary
     and merges the fragments into a single complete table
  3. Preserves column headers in every chunk derived from large tables
  4. Outputs both structured JSON and Markdown for dual indexing

Usage:
    from table_extractor import extract_tables_from_pdf
    tables = extract_tables_from_pdf("path/to/file.pdf")
    # Returns list of TableResult objects with .to_json(), .to_markdown(), .to_chunk()
"""

import re
import json
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Any

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("[table_extractor] WARNING: pdfplumber not installed. Run: pip install pdfplumber")


@dataclass
class TableResult:
    """Represents a fully extracted (possibly cross-page merged) table."""
    source: str                    # PDF filename
    page_start: int                # First page (1-indexed)
    page_end: int                  # Last page (after stitching)
    headers: List[str]             # Column header row (cleaned)
    rows: List[List[str]]          # Data rows
    caption: str = ""              # Table caption if detected
    table_index: int = 0           # Nth table in document
    section_context: str = ""      # Text immediately before the table

    @property
    def page_range(self) -> str:
        if self.page_start == self.page_end:
            return str(self.page_start)
        return f"{self.page_start}-{self.page_end}"

    @property
    def chunk_id(self) -> str:
        key = f"{self.source}:p{self.page_range}:t{self.table_index}"
        return "tbl_" + hashlib.md5(key.encode()).hexdigest()[:12]

    def to_markdown(self, max_rows: int = None) -> str:
        """Render table as Markdown."""
        if not self.headers and not self.rows:
            return ""
        lines = []
        if self.caption:
            lines.append(f"**{self.caption}**\n")
        header_row = self.headers or ([f"Col{i+1}" for i in range(len(self.rows[0]))] if self.rows else [])
        lines.append("| " + " | ".join(str(h) for h in header_row) + " |")
        lines.append("|" + "|".join([":---"] * len(header_row)) + "|")
        data_rows = self.rows[:max_rows] if max_rows else self.rows
        for row in data_rows:
            # Pad/trim row to match header width
            padded = list(row) + [""] * max(0, len(header_row) - len(row))
            padded = padded[:len(header_row)]
            lines.append("| " + " | ".join(str(c).replace("\n", " ") for c in padded) + " |")
        if max_rows and len(self.rows) > max_rows:
            lines.append(f"*[{len(self.rows) - max_rows} more rows...]*")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "id":        self.chunk_id,
            "source":    self.source,
            "page":      self.page_range,
            "caption":   self.caption,
            "headers":   self.headers,
            "rows":      self.rows,
            "n_rows":    len(self.rows),
            "n_cols":    len(self.headers) if self.headers else (len(self.rows[0]) if self.rows else 0),
        }

    def to_chunks(self, rows_per_chunk: int = 15) -> List[dict]:
        """
        Split large tables into retrievable chunks, each with the header row
        preserved at the top (prevents the V1 bug where headers were separated
        from data rows across page boundaries).
        """
        md_header = ""
        if self.headers:
            md_header = (
                "| " + " | ".join(str(h) for h in self.headers) + " |\n"
                + "|" + "|".join([":---"] * len(self.headers)) + "|\n"
            )

        ctx = f"Source: {self.source} | Pages: {self.page_range}"
        if self.caption:
            ctx = f"{self.caption} | {ctx}"
        if self.section_context:
            ctx = f"{self.section_context[:200]} | {ctx}"

        prefix = f"[STRUCTURED DATA TABLE | {ctx}]\n\n"

        if len(self.rows) <= rows_per_chunk:
            # Single chunk — entire table
            text = prefix + self.to_markdown()
            return [{
                "id":       self.chunk_id,
                "source":   self.source,
                "page":     self.page_range,
                "modality": "structured_table",
                "text":     text,
                "caption":  self.caption,
                "_json":    json.dumps(self.to_json()),
            }]

        # Multiple chunks for large tables
        chunks = []
        for i, start in enumerate(range(0, len(self.rows), rows_per_chunk)):
            group = self.rows[start : start + rows_per_chunk]
            row_lines = [
                "| " + " | ".join(str(c).replace("\n", " ") for c in row) + " |"
                for row in group
            ]
            text = prefix + md_header + "\n".join(row_lines)
            cid = self.chunk_id + f"_g{i}"
            chunks.append({
                "id":        cid,
                "parent_id": self.chunk_id,   # V3 parent-child linkage
                "source":    self.source,
                "page":      self.page_range,
                "modality":  "structured_table",
                "text":      text,
                "caption":   self.caption,
                "_part":     f"{i+1}/{((len(self.rows)-1)//rows_per_chunk)+1}",
            })
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_cell(cell: Any) -> str:
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell)).strip()


def _clean_row(row: list) -> List[str]:
    return [_clean_cell(c) for c in row]


def _is_header_row(row: List[str]) -> bool:
    """Heuristic: a header row has short, title-cased or all-caps cells."""
    if not row:
        return False
    non_empty = [c for c in row if c]
    if not non_empty:
        return False
    title_count = sum(1 for c in non_empty if c.istitle() or c.isupper() or len(c) < 30)
    return title_count / len(non_empty) >= 0.6


def _detect_caption_near(page_text: str, table_bbox, direction: str = "above") -> str:
    """Extract caption text appearing just above or below a table bounding box."""
    caption_pat = re.compile(
        r"(Table\s+\d[\.\d]*[\.:]\s*.{0,120}|"
        r"Tab\.\s*\d[\.\d]*[\.:]\s*.{0,120})",
        re.IGNORECASE,
    )
    m = caption_pat.search(page_text)
    if m:
        return m.group(0).strip()
    return ""


def _rows_match_column_count(rows: List[List[str]], expected_cols: int, tolerance: int = 1) -> bool:
    """Check if rows are consistent with a given column count."""
    if not rows:
        return False
    return all(abs(len(r) - expected_cols) <= tolerance for r in rows[:5])


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Page Stitcher
# ─────────────────────────────────────────────────────────────────────────────

def _stitch_cross_page_tables(
    page_tables: List[dict],   # list of {"page": int, "headers": [...], "rows": [[...]], "raw": [...]}
) -> List[TableResult]:
    """
    Merge table fragments that span across page boundaries.

    Detection rule:
      If page N ends with a table and page N+1 has a table whose first row
      does NOT look like a header AND has the same column count → they are one table.
    """
    if not page_tables:
        return []

    merged: List[TableResult] = []
    i = 0
    while i < len(page_tables):
        current = page_tables[i]
        result = TableResult(
            source=current["source"],
            page_start=current["page"],
            page_end=current["page"],
            headers=current["headers"],
            rows=list(current["rows"]),
            caption=current.get("caption", ""),
            table_index=current.get("table_index", 0),
            section_context=current.get("section_context", ""),
        )
        # Look ahead for continuation pages
        j = i + 1
        while j < len(page_tables):
            nxt = page_tables[j]
            # Must be the very next page
            if nxt["page"] != result.page_end + 1:
                break
            nxt_rows = nxt["rows"]
            # First row of next page should NOT look like a new header
            # AND column count should match
            expected_cols = len(result.headers) if result.headers else (len(result.rows[0]) if result.rows else 0)
            if (nxt_rows
                    and not _is_header_row(nxt_rows[0])
                    and _rows_match_column_count(nxt_rows, expected_cols)):
                result.rows.extend(nxt_rows)
                result.page_end = nxt["page"]
                j += 1
            else:
                break
        merged.append(result)
        i = j if j > i + 1 else i + 1

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction function
# ─────────────────────────────────────────────────────────────────────────────

def extract_tables_from_pdf(
    pdf_path: str,
    source_name: str = None,
    section_context_chars: int = 300,
    target_pages: Optional[List[int]] = None,
) -> List[TableResult]:
    """
    Extract all tables from a PDF, with cross-page stitching.

    Args:
        pdf_path:    Path to the PDF file.
        source_name: Override for the source label (defaults to basename).
        section_context_chars: How many chars of preceding page text to store as context.
        target_pages: Optional list of 1-indexed page numbers to inspect. If None, inspects all pages.

    Returns:
        List of TableResult objects, each representing one complete (possibly multi-page) table.
    """
    import os
    if source_name is None:
        source_name = os.path.basename(pdf_path)

    if not PDFPLUMBER_AVAILABLE:
        raise RuntimeError("pdfplumber is required: pip install pdfplumber")

    page_tables = []
    table_global_idx = 0

    with pdfplumber.open(pdf_path) as pdf:
        prev_page_text = ""
        for page_num, page in enumerate(pdf.pages, start=1):
            if target_pages is not None and page_num not in target_pages:
                continue
            page_text = page.extract_text() or ""
            raw_tables = page.extract_tables()

            for t_idx, raw_table in enumerate(raw_tables or []):
                if not raw_table or len(raw_table) < 2:
                    continue

                cleaned = [_clean_row(row) for row in raw_table]
                # Detect header: first row if it looks like headers
                if _is_header_row(cleaned[0]):
                    headers = cleaned[0]
                    rows = cleaned[1:]
                else:
                    headers = []
                    rows = cleaned

                # Try to extract caption from page text
                caption = _detect_caption_near(page_text, None)

                # Section context: last N chars of previous page text
                ctx = (prev_page_text + " " + page_text)[-section_context_chars:]

                page_tables.append({
                    "source":          source_name,
                    "page":            page_num,
                    "headers":         headers,
                    "rows":            rows,
                    "caption":         caption,
                    "table_index":     table_global_idx,
                    "section_context": ctx.strip(),
                    "raw":             raw_table,
                })
                table_global_idx += 1

            prev_page_text = page_text

    # Stitch cross-page fragments
    stitched = _stitch_cross_page_tables(page_tables)
    return stitched


def tables_to_index_chunks(tables: List[TableResult]) -> List[dict]:
    """Convert extracted tables to index chunks ready for ChromaDB + BM25."""
    chunks = []
    for t in tables:
        chunks.extend(t.to_chunks())
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# CLI: test on a single PDF
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python table_extractor.py <path/to/file.pdf>")
        sys.exit(1)
    tables = extract_tables_from_pdf(sys.argv[1])
    print(f"Extracted {len(tables)} table(s):\n")
    for t in tables:
        print(f"  Pages {t.page_range}: {len(t.rows)} rows x {len(t.headers)} cols  caption='{t.caption[:60]}'")
        print(t.to_markdown(max_rows=5))
        print()
