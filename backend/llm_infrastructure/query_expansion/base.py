"""Base classes for query expansion methods."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ExpandedQueries:
    """Result of query expansion."""

    original_query: str
    expanded_queries: list[str]
    include_original: bool = True

    def get_all_queries(self) -> list[str]:
        """Get all queries (original + expanded if include_original is True)."""
        if self.include_original:
            # Original query first, then expanded
            all_queries = [self.original_query] + [
                q for q in self.expanded_queries if q != self.original_query
            ]
            return all_queries
        return self.expanded_queries

    def __len__(self) -> int:
        return len(self.get_all_queries())


class BaseQueryExpander(ABC):
    """Base class for all query expansion methods.

    Each query expansion method should:
    1. Inherit from this class
    2. Implement the expand() method
    3. Register itself using @register_query_expander decorator

    Example:
        ```python
        from .registry import register_query_expander

        @register_query_expander("llm", version="v1")
        class LLMQueryExpander(BaseQueryExpander):
            def expand(self, query, n=3):
                # Expansion logic here
                return ExpandedQueries(...)
        ```
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize query expander with optional config."""
        self.config = kwargs

    @abstractmethod
    def expand(
        self,
        query: str,
        n: int = 3,
        include_original: bool = True,
        **kwargs: Any,
    ) -> ExpandedQueries:
        """Expand a query into multiple related queries.

        Args:
            query: Original search query
            n: Number of expanded queries to generate
            include_original: Whether to include original query in results
            **kwargs: Additional expansion parameters

        Returns:
            ExpandedQueries with original and expanded queries
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"


__all__ = ["BaseQueryExpander", "ExpandedQueries"]