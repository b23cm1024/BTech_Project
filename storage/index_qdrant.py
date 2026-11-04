"""
storage/index_qdrant.py
=======================
Migrates and batch indexes vector representations and payloads from ChromaDB
into local/remote Qdrant collection (iitj_academic_corpus).
"""

import os
import uuid
import logging
import chromadb
from qdrant_client.http import models
from storage.qdrant_manager import qdrant_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IndexQdrant")

CHROMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chromadb"))
CHROMA_COLLECTION = "iitj_chunks"
BATCH_SIZE = 1000


def migrate_to_qdrant(limit: int = None):
    """Batch transfers existing embedded chunks from ChromaDB to Qdrant."""
    if not os.path.isdir(CHROMA_DIR):
        logger.error(f"ChromaDB directory not found at {CHROMA_DIR}")
        return

    logger.info("Opening ChromaDB persistent client...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_col = chroma_client.get_collection(CHROMA_COLLECTION)
    total_count = chroma_col.count()
    logger.info(f"ChromaDB collection contains {total_count} chunks.")

    target_total = min(limit, total_count) if limit else total_count
    logger.info(f"Beginning Qdrant indexing for {target_total} chunks (batch_size={BATCH_SIZE})...")

    offset = 0
    indexed_count = 0

    while offset < target_total:
        current_batch_size = min(BATCH_SIZE, target_total - offset)
        logger.info(f"Fetching batch offset {offset} to {offset + current_batch_size}...")

        batch = chroma_col.get(
            limit=current_batch_size,
            offset=offset,
            include=["embeddings", "metadatas", "documents"]
        )

        ids = batch.get("ids", [])
        embeddings = batch.get("embeddings", [])
        metadatas = batch.get("metadatas", [])
        documents = batch.get("documents", [])

        if not ids:
            break

        points = []
        for i, orig_id in enumerate(ids):
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(orig_id)))
            emb = embeddings[i]
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            doc_text = documents[i] if documents and i < len(documents) else ""

            payload = {
                "chunk_id": orig_id,
                "source": meta.get("source", ""),
                "page": meta.get("page", 1),
                "programme": meta.get("programme"),
                "semester": meta.get("semester"),
                "modality": meta.get("modality", "text"),
                "batch_year": meta.get("batch_year"),
                "text": doc_text
            }

            points.append(
                models.PointStruct(
                    id=point_uuid,
                    vector={"dense": emb.tolist() if hasattr(emb, "tolist") else list(emb)},
                    payload=payload
                )
            )

        qdrant_manager.upsert_chunks(points)
        indexed_count += len(points)
        offset += current_batch_size
        logger.info(f"Progress: {indexed_count}/{target_total} indexed in Qdrant.")

    logger.info(f"Migration completed. Total in Qdrant: {qdrant_manager.count()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max records to index")
    args = parser.parse_args()
    migrate_to_qdrant(limit=args.limit)
