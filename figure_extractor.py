"""
figure_extractor.py — Content-Aware Figure & Caption Extractor (V3)
===================================================================
Extracts embedded figures, diagrams, and illustrations along with their
captions and surrounding contextual text from PDF pages.

Features:
  1. Inspects image blocks using PyMuPDF (fitz)
  2. Detects caption lines ("Figure 1: ...", "Fig. 2.1 ...", "Diagram 3 ...")
  3. Binds adjacent contextual text (1-2 sentences before and after)
  4. Generates retrievable chunks with full V3 metadata schema
"""

import re
import os
import hashlib
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF


FIGURE_CAPTION_PATTERNS = [
    r"^(?:Figure|Fig\.?|FIGURE)\s+(\d+[\.\d*]*)\s*[:\.\-—]\s*(.*)",
    r"^(?:Diagram|Scheme|Illustration)\s+(\d+[\.\d*]*)\s*[:\.\-—]\s*(.*)",
    r"^(?:Graph|Chart)\s+(\d+[\.\d*]*)\s*[:\.\-—]\s*(.*)",
]


def _is_caption(text: str) -> Optional[tuple]:
    """Returns (label, caption_body) or None."""
    text_clean = text.strip()
    for pat in FIGURE_CAPTION_PATTERNS:
        m = re.match(pat, text_clean, re.IGNORECASE)
        if m:
            label = m.group(1).strip()
            body = m.group(2).strip()
            return label, body
    return None


def extract_figures_from_pdf(pdf_path: str, doc_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extract figures, diagrams, and their captions from a PDF document.
    Returns a list of figure chunk dictionaries ready for indexing.
    """
    if not os.path.exists(pdf_path):
        return []

    if doc_name is None:
        doc_name = os.path.basename(pdf_path)

    figure_chunks = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[figure_extractor] Error opening {pdf_path}: {e}")
        return []

    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1

            # Get image list on the page
            image_list = page.get_images(full=True)
            if not image_list:
                # Also check if text has explicit figure captions even if vector graphics
                pass

            # Extract page blocks with bounding boxes: (x0, y0, x1, y1, text, block_no, block_type)
            # block_type 0 = text, 1 = image
            blocks = page.get_text("blocks")
            text_blocks = [b for b in blocks if b[6] == 0]

            for i, block in enumerate(text_blocks):
                block_text = block[4].strip()
                lines = block_text.splitlines()

                for j, line in enumerate(lines):
                    cap_match = _is_caption(line)
                    if cap_match:
                        fig_num, cap_text = cap_match
                        full_caption = line.strip()

                        # Context: preceding line/block and following lines
                        context_parts = []
                        if j > 0:
                            context_parts.append(lines[j - 1])
                        elif i > 0:
                            prev_lines = text_blocks[i - 1][4].strip().splitlines()
                            if prev_lines:
                                context_parts.append(prev_lines[-1])

                        context_parts.append(full_caption)

                        if j + 1 < len(lines):
                            context_parts.append(lines[j + 1])
                        elif i + 1 < len(text_blocks):
                            next_lines = text_blocks[i + 1][4].strip().splitlines()
                            if next_lines:
                                context_parts.append(next_lines[0])

                        combined_context = " ".join(context_parts)
                        chunk_text = (
                            f"[FIGURE {fig_num}] {full_caption}\n"
                            f"Description / Context: {combined_context}"
                        )

                        cid_key = f"{doc_name}:p{page_num}:fig{fig_num}:{full_caption[:30]}"
                        chunk_id = "fig_" + hashlib.md5(cid_key.encode("utf-8")).hexdigest()[:12]

                        chunk = {
                            "chunk_id": chunk_id,
                            "id": chunk_id,
                            "source": doc_name,
                            "page": page_num,
                            "page_number": page_num,
                            "text": chunk_text,
                            "content_type": "figure",
                            "modality": "figure",
                            "figure_metadata": {
                                "figure_number": fig_num,
                                "caption": full_caption,
                                "has_embedded_image": len(image_list) > 0,
                                "bbox": [round(coord, 2) for coord in block[:4]],
                            },
                            "extraction_confidence": 0.85 if len(image_list) > 0 else 0.70,
                        }
                        figure_chunks.append(chunk)

    except Exception as e:
        print(f"[figure_extractor] Error processing {doc_name}: {e}")
    finally:
        doc.close()

    return figure_chunks
