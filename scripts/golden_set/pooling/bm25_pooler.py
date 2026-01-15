"""BM25 (Sparse) pooling strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import PoolingStrategy
from scripts.golden_set.config import PooledDocument, PoolingConfig

if TYPE_CHECKING:
    from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchEngine


class BM25Pooler(PoolingStrategy):
    """BM25 (Sparse) search based pooling."""

    def __init__(self, es_engine: "EsSearchEngine", config: PoolingConfig) -> None:
        super().__init__(es_engine, config)

    @property
    def method_name(self) -> str:
        return "bm25"

    def pool(self, query: str, top_k: int) -> list[PooledDocument]:
        """Retrieve documents using BM25 search.

        Args:
            query: Search query text.
            top_k: Number of documents to retrieve.

        Returns:
            List of PooledDocument objects.
        """
        hits = self.es_engine.sparse_search(
            query_text=query,
            top_k=top_k,
        )
        return [self._hit_to_pooled_doc(h, self.method_name) for h in hits]


__all__ = ["BM25Pooler"]
