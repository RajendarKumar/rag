import logging
from typing import Any

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from pymilvus.milvus_client.index import IndexParam, IndexParams

logger = logging.getLogger(__name__)


class MilvusStorageProvider:
    """Handles Milvus interaction patterns, schemas, and structural ingestion."""

    def __init__(self, host: str = "localhost", port: str = "19530") -> None:
        self.client = MilvusClient(uri=f"http://{host}:{port}")

    def initialize_hybrid_collection(
        self, collection_name: str, dense_dim: int = 768
    ) -> None:
        """Enforce the collection schema and build its search indexes."""
        if self.client.has_collection(collection_name):
            logger.info(
                "Collection '%s' exists. Recreating it with dim=%s.",
                collection_name,
                dense_dim,
            )
            try:
                self.client.drop_collection(collection_name)
            except Exception:
                logger.exception(
                    "Failed to drop existing collection %s; aborting initialization.",
                    collection_name,
                )
                raise

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]
        schema = CollectionSchema(fields=fields, description="Production Hybrid RAG Store")
        self.client.create_collection(collection_name=collection_name, schema=schema)
        self._build_indices(collection_name)

    def _build_indices(self, collection_name: str) -> None:
        dense_index = IndexParam(
            field_name="dense_vector",
            index_type="HNSW",
            index_name="dense_vector_hnsw",
            metric_type="COSINE",
            M=16,
            efConstruction=200,
        )
        self.client.create_index(collection_name, index_params=IndexParams([dense_index]))

        sparse_index = IndexParam(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            index_name="sparse_vector_inverted",
            metric_type="IP",
            drop_ratio_build=0.0,
        )
        self.client.create_index(
            collection_name, index_params=IndexParams([sparse_index])
        )
        self.client.load_collection(collection_name)

    def batch_insert(
        self, collection_name: str, entities: list[dict[str, Any]]
    ) -> int:
        """Insert a batch and return the number of inserted entities."""
        result = self.client.insert(collection_name=collection_name, data=entities)
        return result.get("insert_count", 0)
