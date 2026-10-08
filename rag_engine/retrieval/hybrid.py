from typing import Any

from pymilvus import AnnSearchRequest, RRFRanker

from ..embeddings import HybridEmbeddingEngine
from ..storage.milvus import MilvusStorageProvider


class HybridSearchOrchestrator:
    """Coordinates dense/sparse queries and server-side RRF result fusion."""

    def __init__(
        self,
        embedding_engine: HybridEmbeddingEngine,
        storage_provider: MilvusStorageProvider,
    ) -> None:
        self.embedder = embedding_engine
        self.db = storage_provider.client

    def execute_hybrid_query(
        self,
        collection_name: str,
        query: str,
        top_k: int = 3,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        """Run hybrid search and return normalized hit dictionaries."""
        dense_q, sparse_q = self.embedder.generate_query_vectors(query)
        dense_req = AnnSearchRequest(
            data=[dense_q],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k * 2,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_q],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.0}},
            limit=top_k * 2,
        )
        raw_hits = self.db.hybrid_search(
            collection_name=collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=rrf_k),
            limit=top_k,
            output_fields=["text", "source"],
        )
        return self._hydrate_results(raw_hits)

    def _hydrate_results(self, raw_hits: list[Any]) -> list[dict[str, Any]]:
        if not raw_hits:
            return []

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "text": hit.entity.get("text"),
                "source": hit.entity.get("source"),
            }
            for hit in raw_hits[0]
        ]
