"""
clip_multimodal.py — Multimodal Zero-Shot Vision & Image Understanding via CLIP
=================================================================================
Integrates OpenAI's CLIP (Contrastive Language-Image Pre-training) model
(clip-ViT-B-32 via sentence-transformers) to enable:
  1. Multimodal document & visual asset indexing (campus infrastructure, flowcharts, figures).
  2. Natural language cross-modal search: text queries retrieved against image features.
  3. Seamless hybrid integration with dense text vector retrieval in RetrievalEngineV3.
"""

import os
import glob
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

_clip_model = None


def get_clip_model():
    """Lazily load the CLIP model."""
    global _clip_model
    if _clip_model is not None:
        return _clip_model
    try:
        from sentence_transformers import SentenceTransformer
        _clip_model = SentenceTransformer("clip-ViT-B-32")
        print("[CLIP Multimodal] clip-ViT-B-32 loaded successfully.")
    except Exception as e:
        print(f"[CLIP Multimodal] Notice: Could not load CLIP model: {e}")
        _clip_model = None
    return _clip_model


class MultimodalImageStore:
    def __init__(self, asset_dirs: Optional[List[str]] = None):
        if asset_dirs is None:
            base_dir = os.path.dirname(__file__)
            asset_dirs = [
                os.path.join(base_dir, "assets"),
                os.path.join(base_dir, "data", "figures"),
                os.path.join(base_dir, "data", "images"),
            ]
        self.asset_dirs = asset_dirs
        self.images: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self._index_visual_assets()

    def _index_visual_assets(self):
        """Scans image directories and indexes visual assets."""
        model = get_clip_model()
        valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
        found_paths = []
        for d in self.asset_dirs:
            if os.path.exists(d):
                for ext in valid_exts:
                    found_paths.extend(glob.glob(os.path.join(d, "**", ext), recursive=True))

        self.images = []
        embs = []
        for p in found_paths:
            try:
                img = Image.open(p).convert("RGB")
                fname = os.path.basename(p)
                title = fname.rsplit(".", 1)[0].replace("_", " ").title()
                self.images.append({
                    "path": p,
                    "filename": fname,
                    "title": title,
                    "rel_path": os.path.relpath(p, os.path.dirname(__file__))
                })
                if model:
                    emb = model.encode(img)
                    embs.append(emb)
            except Exception:
                continue

        if embs and model:
            self.embeddings = np.vstack(embs)
            print(f"[MultimodalImageStore] Indexed {len(self.images)} visual assets with CLIP embeddings.")
        else:
            self.embeddings = None

    def search_images(self, query: str, top_k: int = 3, threshold: float = 0.12) -> List[Dict[str, Any]]:
        """Searches visual assets using text-image cosine similarity via CLIP."""
        model = get_clip_model()
        if model is None or self.embeddings is None or not self.images:
            return []

        try:
            q_emb = model.encode(query)
            # Cosine similarity
            norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb)
            sims = np.dot(self.embeddings, q_emb) / np.maximum(norms, 1e-9)

            results = []
            for idx in np.argsort(sims)[::-1][:top_k]:
                score = float(sims[idx])
                if score >= threshold:
                    res = dict(self.images[idx])
                    res["similarity"] = score
                    results.append(res)
            return results
        except Exception as e:
            print(f"[MultimodalImageStore] Search error: {e}")
            return []


# Singleton
multimodal_store = MultimodalImageStore()
