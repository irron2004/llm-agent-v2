"""Base classes for vector database abstraction.

Defines the common interface that all vector DB implementations must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchHit:
    """Single search result from vector database."""

    id: str
    score: float
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"SearchHit(id={self.id}, score={self.score:.4f})"


@dataclass
class UpsertResult:
    """Result of upsert operation."""

    succeeded: int
    failed: int
    errors: list[str] = field(default_factory=list)


@dataclass
class DeleteResult:
    """Result of delete operation."""

    deleted: int
    not_found: int = 0
    errors: list[str] = field(default_factory=list)


class VectorDBClient(ABC):
    """Abstract base class for vector database clients.

    All vector DB implementations must inherit from this class and implement
    the required methods. This allows switching between different backends
    (Elasticsearch, Pinecone, Milvus, etc.) without changing application code.

    Example implementation:
        @register_vectordb("elasticsearch")
        class EsVectorDB(VectorDBClient):
            def __init__(self, es_client, index_name, ...):
                ...

            def upsert(self, documents, batch_size=100):
                ...

            def search(self, query_vector, top_k=10, filters=None):
                ...
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the vector DB client.

        Args:
            **kwargs: Implementation-specific configuration.
        """
        self.config = kwargs

    @abstractmethod
    def upsert(
        self,
        documents: list[dict[str, Any]],
        *,
        batch_size: int = 100,
    ) -> UpsertResult:
        """Insert or update documents.

        Args:
            documents: List of documents to upsert. Each document should have:
                - id: Unique identifier
                - embedding: Vector embedding (list of floats)
                - content: Text content
                - metadata: Optional metadata dict
            batch_size: Number of documents per batch.

        Returns:
            UpsertResult with success/failure counts.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> list[SearchHit]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            filters: Optional filter conditions.
            include_metadata: Whether to include metadata in results.

        Returns:
            List of SearchHit sorted by score (descending).
        """
        raise NotImplementedError

    @abstractmethod
    def delete(
        self,
        ids: list[str],
    ) -> DeleteResult:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete.

        Returns:
            DeleteResult with deletion counts.
        """
        raise NotImplementedError

    def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        *,
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Hybrid search combining dense and sparse retrieval.

        Not all backends support hybrid search. Default implementation
        falls back to dense-only search.

        Args:
            query_vector: Query embedding vector.
            query_text: Query text for sparse search.
            top_k: Number of results to return.
            dense_weight: Weight for dense scores.
            sparse_weight: Weight for sparse scores.
            filters: Optional filter conditions.

        Returns:
            List of SearchHit sorted by combined score.
        """
        # Default: fall back to dense-only search
        return self.search(query_vector, top_k=top_k, filters=filters)

    def health_check(self) -> bool:
        """Check if the database is reachable and healthy.

        Returns:
            True if healthy, False otherwise.
        """
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Dict with implementation-specific stats.
        """
        return {}


__all__ = [
    "VectorDBClient",
    "SearchHit",
    "UpsertResult",
    "DeleteResult",
]
