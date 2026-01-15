"""Base class for pooling strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from scripts.golden_set.config import PooledDocument, PoolingConfig

if TYPE_CHECKING:
    from backend.llm_infrastructure.retrieval.engines.es_search import (
        EsSearchEngine,
        EsSearchHit,
    )


class PoolingStrategy(ABC):
    """Abstract base class for pooling strategies."""

    def __init__(self, es_engine: "EsSearchEngine", config: PoolingConfig) -> None:
        self.es_engine = es_engine
        self.config = config

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Method name used in source_method field."""
        pass

    @abstractmethod
    def pool(self, query: str, top_k: int) -> list[PooledDocument]:
        """Retrieve top_k documents for the query.

        Args:
            query: Search query text.
            top_k: Number of documents to retrieve.

        Returns:
            List of PooledDocument objects.
        """
        pass

    def _hit_to_pooled_doc(self, hit: "EsSearchHit", method: str) -> PooledDocument:
        """Convert EsSearchHit to PooledDocument."""
        return PooledDocument(
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            content=hit.content,
            score=hit.score,
            source_method=method,
            doc_type=hit.metadata.get("doc_type", "unknown"),
            metadata=hit.metadata,
        )


__all__ = ["PoolingStrategy"]
