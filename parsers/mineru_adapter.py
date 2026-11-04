"""
parsers/mineru_adapter.py
=========================
Adapter interface for MinerU 4.1.x (magic-pdf) multimodal document parser.
Provides structured layout extraction, formula preservation, and prose analysis.
"""

import os
import shutil
import logging
import subprocess
from typing import Dict, Any, Optional

logger = logging.getLogger("MinerUAdapter")


class MinerUAdapter:
    """Adapter for MinerU 4.1.x multimodal document parser."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), "..", "data", "mineru_output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.cli_available = shutil.which("magic-pdf") is not None

    def is_available(self) -> bool:
        """Checks if MinerU CLI or package is installed."""
        if self.cli_available:
            return True
        try:
            try:
    import magic_pdf
except ImportError:
    magic_pdf = None
            return True
        except ImportError:
            return False

    def parse_document(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """Parses document using MinerU if available."""
        if not self.is_available():
            logger.info("MinerU (magic-pdf) is not installed on this system. Docling path is used.")
            return None

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            logger.info(f"Executing MinerU extraction for: {os.path.basename(pdf_path)}")
            cmd = ["magic-pdf", "-p", pdf_path, "-o", self.output_dir, "-m", "auto"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if res.returncode == 0:
                logger.info(f"MinerU extraction succeeded for {os.path.basename(pdf_path)}")
                # Locate markdown file in output dir
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                target_md = os.path.join(self.output_dir, base_name, "auto", f"{base_name}.md")
                if os.path.exists(target_md):
                    with open(target_md, "r", encoding="utf-8") as f:
                        md_content = f.read()
                    return {
                        "source": os.path.basename(pdf_path),
                        "markdown": md_content,
                        "engine": "mineru"
                    }
        except Exception as e:
            logger.warning(f"MinerU execution notice: {e}")

        return None
