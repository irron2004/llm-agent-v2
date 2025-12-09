"""Base classes for reranking methods."""

from abc import ABC, abstractmethod
from typing import Any

from backend.llm_infrastructure.retrieval.base import RetrievalResult


class BaseReranker(ABC):
    """Base class for all reranking methods.

    Each reranking method should:
    1. Inherit from this class
    2. Implement the rerank() method
    3. Register itself using @register_reranker decorator

    Example:
        ```python
        from .registry import register_reranker

        @register_reranker("cross_encoder", version="v1")
        class CrossEncoderReranker(BaseReranker):
            def rerank(self, query, results, top_k=5):
                # Reranking logic here
                return reranked_results
        ```
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize reranker with optional config."""
        self.config = kwargs

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results for a query.

        Args:
            query: Original search query
            results: List of retrieval results to rerank
            top_k: Number of results to return after reranking (None = return all)
            **kwargs: Additional reranking parameters

        Returns:
            List of reranked results, sorted by new score (descending)
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"


__all__ = ["BaseReranker"]
