"""Base classes for retrieval methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    doc_id: str
    content: str
    score: float
    metadata: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return f"RetrievalResult(doc_id={self.doc_id}, score={self.score:.4f})"


class BaseRetriever(ABC):
    """Base class for all retrieval methods.

    Each retrieval method should:
    1. Inherit from this class
    2. Implement the retrieve() method
    3. Register itself using @register_retriever decorator

    Example:
        ```python
        from .registry import register_retriever

        @register_retriever("my_retriever", version="v1")
        class MyRetriever(BaseRetriever):
            def retrieve(self, query, top_k=10):
                # Retrieval logic here
                return [
                    RetrievalResult(
                        doc_id="doc1",
                        content="...",
                        score=0.95
                    )
                ]
        ```
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize retriever with optional config."""
        self.config = kwargs

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of results to return
            **kwargs: Additional retrieval parameters

        Returns:
            List of retrieval results, sorted by score (descending)
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
