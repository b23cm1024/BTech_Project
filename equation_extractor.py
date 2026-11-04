"""
equation_extractor.py — Mathematical Formula & Context Extractor (V3)
=====================================================================
Detects mathematical equations, formulas, and expressions from PDF documents,
preserving LaTeX/symbolic notation alongside surrounding contextual explanations.

Features:
  1. Detects display equations and numbered equations (e.g., "(1)", "(2.3)")
  2. Identifies mathematical expressions containing integrals, sums, Greek symbols, matrices
  3. Binds adjacent explanatory context (variable definitions, derivations)
  4. Generates equation chunks with full V3 metadata schema
"""

import re
import os
import hashlib
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF


# Math symbol patterns
MATH_SYMBOLS = r"[=≤≥∑∫∂α-ωΑ-Ω±∞√×÷≠≈∈∉⊂⊃∪∩∇∆λμσθπτ]"
NUMBERED_EQ_PAT = r"^\s*(\([0-9]+(?:\.[0-9]+)*\))\s*$"
STANDALONE_EQ_PAT = r"^\s*([A-Za-z0-9_+\-*/^=()\s≤≥∑∫∂α-ωΑ-Ω±∞√×÷≠≈∈∉⊂⊃∪∩∇∆λμσθπτ]{3,80})\s*(\([0-9]+(?:\.[0-9]+)*\))?\s*$"


def _is_math_line(line: str) -> bool:
    """Checks if a single line strongly looks like an equation."""
    line_clean = line.strip()
    if not line_clean or len(line_clean) > 120:
        return False

    # Numbered equation label at the end, e.g. "E = mc^2  (1)"
    if re.search(r"\(\d+(?:\.\d+)*\)\s*$", line_clean):
        return True

    # High concentration of math operators/symbols
    math_count = len(re.findall(MATH_SYMBOLS, line_clean))
    if math_count >= 2 and len(line_clean) < 80:
        return True

    # LaTeX patterns
    if re.search(r"\\(?:frac|sum|int|sqrt|alpha|beta|gamma|lambda|sigma)", line_clean):
        return True

    return False


def extract_equations_from_pdf(pdf_path: str, doc_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extract mathematical equations and formulas with surrounding explanation.
    Returns a list of equation chunk dictionaries.
    """
    if not os.path.exists(pdf_path):
        return []

    if doc_name is None:
        doc_name = os.path.basename(pdf_path)

    equation_chunks = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[equation_extractor] Error opening {pdf_path}: {e}")
        return []

    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1

            blocks = page.get_text("blocks")
            text_blocks = [b for b in blocks if b[6] == 0]

            for i, block in enumerate(text_blocks):
                block_text = block[4].strip()
                lines = block_text.splitlines()

                for j, line in enumerate(lines):
                    if _is_math_line(line):
                        formula = line.strip()

                        # Equation number if any
                        eq_num_match = re.search(r"\((\d+(?:\.\d+)*)\)\s*$", formula)
                        eq_num = eq_num_match.group(1) if eq_num_match else ""

                        # Extract surrounding context (preceding and succeeding lines)
                        preceding = []
                        if j > 0:
                            preceding.append(lines[j - 1].strip())
                        elif i > 0:
                            prev_block_lines = text_blocks[i - 1][4].strip().splitlines()
                            if prev_block_lines:
                                preceding.append(prev_block_lines[-1].strip())

                        succeeding = []
                        if j + 1 < len(lines):
                            succeeding.append(lines[j + 1].strip())
                        elif i + 1 < len(text_blocks):
                            next_block_lines = text_blocks[i + 1][4].strip().splitlines()
                            if next_block_lines:
                                succeeding.append(next_block_lines[0].strip())

                        context_str = " ".join(preceding + succeeding)

                        # Skip trivial false positives (e.g. single numbers or purely short labels)
                        if len(formula) < 4 or (formula.isdigit()):
                            continue

                        label_prefix = f"[EQUATION {eq_num}]" if eq_num else "[EQUATION]"
                        chunk_text = (
                            f"{label_prefix} Formula: {formula}\n"
                            f"Context / Variable Definitions: {context_str}"
                        )

                        cid_key = f"{doc_name}:p{page_num}:eq:{formula[:30]}"
                        chunk_id = "eq_" + hashlib.md5(cid_key.encode("utf-8")).hexdigest()[:12]

                        chunk = {
                            "chunk_id": chunk_id,
                            "id": chunk_id,
                            "source": doc_name,
                            "page": page_num,
                            "page_number": page_num,
                            "text": chunk_text,
                            "content_type": "equation",
                            "modality": "equation",
                            "equation_metadata": {
                                "formula": formula,
                                "equation_number": eq_num,
                                "has_context": len(context_str) > 0,
                            },
                            "extraction_confidence": 0.80 if eq_num else 0.65,
                        }
                        equation_chunks.append(chunk)

    except Exception as e:
        print(f"[equation_extractor] Error processing {doc_name}: {e}")
    finally:
        doc.close()

    return equation_chunks
