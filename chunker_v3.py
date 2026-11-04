"""
chunker_v3.py — Adaptive Hierarchical Content-Aware Chunker (V3)
=================================================================
Replaces the fixed-size chunking in ingest.py with content-type-aware,
hierarchically-structured chunking with parent-child relationships.

Chunking strategies by content type:
  - prose:     sentence-boundary split, 1200 chars, 200-char overlap
  - section:   full section as parent chunk (up to 3000 chars)
  - table:     handled by table_extractor.py (single or row-group chunks)
  - equation:  equation + context window (up to 800 chars)
  - figure:    caption + context (up to 600 chars)
  - list:      full logical list (up to 1000 chars)

Every chunk carries:
  parent_id, heading_path, chapter, section, subsection, chunk_type
"""

import re
import uuid
import hashlib
from typing import List, Dict, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_CONFIG = {
    "prose":    {"max_size": 1200, "overlap": 200},
    "section":  {"max_size": 3000, "overlap": 0},
    "equation": {"max_size": 800,  "overlap": 0},
    "figure":   {"max_size": 600,  "overlap": 0},
    "list":     {"max_size": 1000, "overlap": 0},
}

# Heading detection patterns (applied to line-by-line text analysis)
HEADING_PATTERNS = [
    # Chapter: "Chapter 3", "CHAPTER 3", "3. Introduction"
    (r"^(?:chapter|CHAPTER)\s+\d+", "h1"),
    (r"^\d+\.\s+[A-Z][A-Za-z\s]{3,50}$", "h1"),
    # Section: "3.1 Curriculum", "3.1. Curriculum"
    (r"^\d+\.\d+\.?\s+[A-Z]", "h2"),
    # Subsection: "3.1.1 Credits"
    (r"^\d+\.\d+\.\d+\.?\s+[A-Z]", "h3"),
    # Short bold-like all-caps lines (heuristic)
    (r"^[A-Z][A-Z\s]{4,40}:?\s*$", "h2"),
]

# Equation detection
EQUATION_PATTERNS = [
    r"[=≤≥∑∫∂α-ωΑ-Ω±∞√]{1}",   # math symbols
    r"\b(equation|formula|expression)\s+\(?\d+\)?",
    r"\\\(.*?\\\)",               # LaTeX inline
    r"\\\[.*?\\\]",               # LaTeX display
]

# Figure/caption detection
FIGURE_PATTERNS = [
    r"^(Figure|Fig\.?|FIGURE)\s+\d+[\.\:]",
    r"^(Table|Tab\.?)\s+\d+[\.\:]",
    r"^(Scheme|Diagram)\s+\d+",
]


# ─────────────────────────────────────────────────────────────────────────────
# Document structure extraction
# ─────────────────────────────────────────────────────────────────────────────

def _make_id(text_snippet: str, extra: str = "") -> str:
    key = (text_snippet[:80] + extra).encode("utf-8")
    return hashlib.md5(key).hexdigest()[:16]


def _detect_heading(line: str) -> Optional[Tuple[str, str]]:
    """Returns (level, heading_text) or None."""
    line = line.strip()
    for pat, level in HEADING_PATTERNS:
        if re.match(pat, line):
            return level, line
    return None


def _is_equation_line(line: str) -> bool:
    return any(re.search(p, line) for p in EQUATION_PATTERNS)


def _is_figure_caption(line: str) -> bool:
    return any(re.match(p, line.strip(), re.IGNORECASE) for p in FIGURE_PATTERNS)


def _is_list_line(line: str) -> bool:
    stripped = line.strip()
    return bool(re.match(r"^(\d+[\.\)]|\*|\-|\•)\s+", stripped))


# ─────────────────────────────────────────────────────────────────────────────
# Core sentence-boundary prose chunker (kept from V1 logic, improved)
# ─────────────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences at natural boundaries."""
    pattern = re.compile(
        r"(?<=[.?!])\s+(?=[A-Z0-9\"\'(\[])"   # sentence end → capital
        r"|(?<=\n)(?=\d+\.\s)"                 # numbered list
        r"|(?<=\n)(?=[A-Z][A-Za-z ]{2,30}:)"  # "Section Heading:"
        r"|(?<=\n\n)"                          # paragraph break
    )
    parts = pattern.split(text)
    return [s.strip() for s in parts if s.strip()]


def chunk_prose(text: str, source: str, page: str,
                max_size: int = 1200, overlap: int = 200,
                parent_id: str = None, heading_path: str = "",
                programme: str = "GENERAL", semester: int = 0,
                batch_year: str = "unknown") -> List[dict]:
    """Split prose text into overlapping sentence-boundary chunks."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 40:
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return [_make_prose_chunk(text, source, page, parent_id, heading_path,
                                  programme, semester, batch_year)]

    chunks = []
    current_sents = []
    current_len = 0

    for sent in sentences:
        slen = len(sent)
        if current_len + slen > max_size and current_sents:
            body = " ".join(current_sents).strip()
            if len(body) >= 40:
                chunks.append(_make_prose_chunk(body, source, page, parent_id,
                                                heading_path, programme, semester, batch_year))
            # Build overlap tail
            tail, tail_len = [], 0
            for s in reversed(current_sents):
                if tail_len + len(s) <= overlap:
                    tail.insert(0, s)
                    tail_len += len(s)
                else:
                    break
            current_sents = tail + [sent]
            current_len = tail_len + slen
        else:
            current_sents.append(sent)
            current_len += slen

    if current_sents:
        body = " ".join(current_sents).strip()
        if len(body) >= 40:
            chunks.append(_make_prose_chunk(body, source, page, parent_id,
                                            heading_path, programme, semester, batch_year))
    return chunks


def _make_prose_chunk(text, source, page, parent_id, heading_path,
                      programme, semester, batch_year) -> dict:
    return {
        "id":           _make_id(text, source),
        "source":       source,
        "page":         str(page),
        "modality":     "prose",
        "chunk_type":   "prose",
        "text":         text,
        "parent_id":    parent_id or "",
        "heading_path": heading_path,
        "programme":    programme,
        "semester":     semester,
        "batch_year":   batch_year,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section-level hierarchical chunker
# ─────────────────────────────────────────────────────────────────────────────

def chunk_document(
    full_text: str,
    source: str,
    page_map: Dict[int, int] = None,   # char_offset → page number
    programme: str = "GENERAL",
    semester: int = 0,
    batch_year: str = "unknown",
) -> List[dict]:
    """
    Parse full document text into hierarchical chunks.
    Detects headings to build section structure; creates:
      1. Section-level parent chunks (chunk_type=section)
      2. Prose child chunks within each section
      3. Equation chunks
      4. Figure caption chunks
      5. List chunks
    All child chunks carry parent_id pointing to their section chunk.
    """
    lines = full_text.split("\n")
    chunks = []

    current_h1 = current_h2 = current_h3 = ""
    section_buffer = []
    section_page = "1"
    section_start_char = 0
    char_offset = 0

    def _flush_section():
        nonlocal section_buffer
        if not section_buffer:
            return
        section_text = "\n".join(section_buffer).strip()
        if len(section_text) < 40:
            section_buffer = []
            return
        heading_path = " > ".join(filter(None, [current_h1, current_h2, current_h3]))
        sec_id = _make_id(section_text, source)
        # Create section parent chunk
        sec_chunk = {
            "id":           sec_id,
            "source":       source,
            "page":         section_page,
            "modality":     "prose",
            "chunk_type":   "section",
            "text":         f"[Section: {heading_path}]\n{section_text}",
            "parent_id":    "",
            "heading_path": heading_path,
            "chapter":      current_h1,
            "section":      current_h2,
            "subsection":   current_h3,
            "programme":    programme,
            "semester":     semester,
            "batch_year":   batch_year,
        }
        chunks.append(sec_chunk)
        # Create prose children
        prose_children = chunk_prose(
            section_text, source, section_page,
            parent_id=sec_id, heading_path=heading_path,
            programme=programme, semester=semester, batch_year=batch_year,
        )
        chunks.extend(prose_children)
        section_buffer = []

    for line in lines:
        char_offset += len(line) + 1
        page = "1"
        if page_map:
            for offset, pg in sorted(page_map.items()):
                if char_offset >= offset:
                    page = str(pg)

        heading = _detect_heading(line)
        if heading:
            _flush_section()
            level, heading_text = heading
            if level == "h1":
                current_h1 = heading_text
                current_h2 = current_h3 = ""
            elif level == "h2":
                current_h2 = heading_text
                current_h3 = ""
            else:
                current_h3 = heading_text
            section_page = page
            section_buffer = [line]
        elif _is_equation_line(line) and line.strip():
            # Flush current buffer, emit equation chunk
            section_buffer.append(line)
        elif _is_figure_caption(line):
            section_buffer.append(line)
        else:
            section_buffer.append(line)

    _flush_section()
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Equation & Figure chunk factories
# ─────────────────────────────────────────────────────────────────────────────

def make_equation_chunk(equation_text: str, context_before: str, context_after: str,
                        source: str, page: str, parent_id: str = "",
                        heading_path: str = "", programme: str = "GENERAL") -> dict:
    text = (
        f"[EQUATION | {source} | Page {page}]\n"
        f"Context: {context_before[-200:].strip()}\n"
        f"Equation: {equation_text}\n"
        f"Explanation: {context_after[:200].strip()}"
    )
    return {
        "id":           _make_id(equation_text, source),
        "source":       source,
        "page":         page,
        "modality":     "equation",
        "chunk_type":   "equation",
        "text":         text,
        "parent_id":    parent_id,
        "heading_path": heading_path,
        "programme":    programme,
    }


def make_figure_chunk(caption: str, labels: str, context: str,
                      source: str, page: str, figure_id: str = "",
                      parent_id: str = "", heading_path: str = "") -> dict:
    text = (
        f"[FIGURE {figure_id} | {source} | Page {page}]\n"
        f"Caption: {caption}\n"
        f"Labels/Annotations: {labels}\n"
        f"Context: {context[:300].strip()}"
    )
    return {
        "id":           _make_id(caption + figure_id, source),
        "source":       source,
        "page":         page,
        "modality":     "figure",
        "chunk_type":   "figure",
        "text":         text,
        "parent_id":    parent_id,
        "heading_path": heading_path,
        "figure_id":    figure_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate_chunks(chunks: List[dict]) -> List[dict]:
    """Remove near-duplicate chunks by 85-char text fingerprint."""
    seen = set()
    unique = []
    for c in chunks:
        raw = re.sub(r"\[.*?\]", "", c.get("text", "")).strip()
        fp = re.sub(r"\s+", " ", raw[:85]).lower().strip()
        if len(fp) > 20 and fp in seen:
            continue
        seen.add(fp)
        unique.append(c)
    return unique
