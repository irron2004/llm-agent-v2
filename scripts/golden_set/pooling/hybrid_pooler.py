"""Hybrid (Dense + Sparse) pooling strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import PoolingStrategy
from scripts.golden_set.config import PooledDocument, PoolingConfig

if TYPE_CHECKING:
    from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchEngine
    from backend.llm_infrastructure.embedding.base import BaseEmbedder


class HybridPooler(PoolingStrategy):
    """Hybrid search (RRF) based pooling."""

    def __init__(
        self,
        es_engine: "EsSearchEngine",
        config: PoolingConfig,
        embedder: "BaseEmbedder",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        use_rrf: bool = True,
        rrf_k: int = 60,
    ) -> None:
        super().__init__(es_engine, config)
        self.embedder = embedder
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.use_rrf = use_rrf
        self.rrf_k = rrf_k

    @property
    def method_name(self) -> str:
        return "hybrid"

    def pool(self, query: str, top_k: int) -> list[PooledDocument]:
        """Retrieve documents using hybrid search.

        Args:
            query: Search query text.
            top_k: Number of documents to retrieve.

        Returns:
            List of PooledDocument objects.
        """
        query_vec = self._embed_query(query)

        hits = self.es_engine.hybrid_search(
            query_vector=query_vec,
            query_text=query,
            top_k=top_k,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
            use_rrf=self.use_rrf,
            rrf_k=self.rrf_k,
        )
        return [self._hit_to_pooled_doc(h, self.method_name) for h in hits]

    def _embed_query(self, query: str) -> list[float]:
        """Embed query and L2 normalize."""
        vec = self.embedder.embed_batch([query])[0]
        arr = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()


__all__ = ["HybridPooler"]
