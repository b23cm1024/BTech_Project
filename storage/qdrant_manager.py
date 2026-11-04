"""
storage/qdrant_manager.py
=========================
Production Qdrant integration supporting hybrid search (dense vectors + sparse tokens)
with rich payload filtering for IIT Jodhpur academic corpus.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger("QdrantManager")


class QdrantManager:
    """Manages collection lifecycle, payload schemas, and hybrid retrieval in Qdrant."""

    COLLECTION_NAME = "iitj_academic_corpus"

    def __init__(self, vector_size: int = 384):
        self.vector_size = vector_size
        qdrant_url = os.getenv("QDRANT_URL")

        if qdrant_url:
            api_key = os.getenv("QDRANT_API_KEY")
            self.client = QdrantClient(url=qdrant_url, api_key=api_key)
            logger.info(f"Connected to remote Qdrant cluster at {qdrant_url}")
        else:
            local_storage_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "data", "qdrant_storage")
            )
            os.makedirs(local_storage_path, exist_ok=True)
            self.client = QdrantClient(path=local_storage_path)
            logger.info(f"Initialized local embedded Qdrant store at {local_storage_path}")

        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Creates the collection and payload indexes if they do not exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.COLLECTION_NAME for c in collections)

        if not exists:
            logger.info(f"Creating Qdrant collection: {self.COLLECTION_NAME} (dim={self.vector_size})")
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False
                        )
                    )
                } if hasattr(models, "SparseVectorParams") else None
            )

            # Create payload indexes for sub-millisecond metadata filtering
            index_fields = [
                ("category", models.PayloadSchemaType.KEYWORD),
                ("programme", models.PayloadSchemaType.KEYWORD),
                ("cohort", models.PayloadSchemaType.KEYWORD),
                ("degree_level", models.PayloadSchemaType.KEYWORD),
                ("page_number", models.PayloadSchemaType.INTEGER),
                ("is_table", models.PayloadSchemaType.BOOL),
                ("doc_id", models.PayloadSchemaType.KEYWORD),
            ]
            for field_name, field_type in index_fields:
                try:
                    self.client.create_payload_index(
                        collection_name=self.COLLECTION_NAME,
                        field_name=field_name,
                        field_schema=field_type
                    )
                except Exception as e:
                    logger.debug(f"Payload index on {field_name} note: {e}")

    def upsert_chunks(self, points: List[models.PointStruct]):
        """Batch upsert chunks into Qdrant."""
        if not points:
            return
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points
        )

    def search_dense(
        self,
        query_vector: List[float],
        filter_dict: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> List[Any]:
        """Performs filtered dense vector search."""
        q_filter = None
        if filter_dict:
            must_conditions = []
            for k, v in filter_dict.items():
                if v is not None:
                    must_conditions.append(
                        models.FieldCondition(
                            key=k,
                            match=models.MatchValue(value=v)
                        )
                    )
            if must_conditions:
                q_filter = models.Filter(must=must_conditions)

        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=query_vector,
                using="dense",
                query_filter=q_filter,
                limit=limit
            )
            return res.points
        elif hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=("dense", query_vector),
                query_filter=q_filter,
                limit=limit
            )
        else:
            raise NotImplementedError("QdrantClient does not have query_points or search")

    def count(self) -> int:
        """Returns total records indexed in collection."""
        res = self.client.count(collection_name=self.COLLECTION_NAME)
        return res.count


# Singleton instance
qdrant_manager = QdrantManager(vector_size=384)
