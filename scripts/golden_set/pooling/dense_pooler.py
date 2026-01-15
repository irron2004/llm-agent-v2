"""Dense (Vector) pooling strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import PoolingStrategy
from scripts.golden_set.config import PooledDocument, PoolingConfig

if TYPE_CHECKING:
    from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchEngine
    from backend.llm_infrastructure.embedding.base import BaseEmbedder


class DensePooler(PoolingStrategy):
    """Dense vector search based pooling."""

    def __init__(
        self,
        es_engine: "EsSearchEngine",
        config: PoolingConfig,
        embedder: "BaseEmbedder",
    ) -> None:
        super().__init__(es_engine, config)
        self.embedder = embedder

    @property
    def method_name(self) -> str:
        return "dense"

    def pool(self, query: str, top_k: int) -> list[PooledDocument]:
        """Retrieve documents using dense vector search.

        Args:
            query: Search query text.
            top_k: Number of documents to retrieve.

        Returns:
            List of PooledDocument objects.
        """
        query_vec = self._embed_query(query)

        hits = self.es_engine.dense_search(
            query_vector=query_vec,
            top_k=top_k,
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


__all__ = ["DensePooler"]
