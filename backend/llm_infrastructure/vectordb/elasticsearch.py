"""Elasticsearch implementation of VectorDBClient.

Provides a VectorDBClient implementation using Elasticsearch as the backend.

Usage:
    from elasticsearch import Elasticsearch
    from backend.llm_infrastructure.vectordb import get_vectordb

    es_client = Elasticsearch(["http://localhost:9200"])
    db = get_vectordb(
        "elasticsearch",
        es_client=es_client,
        index_name="rag_chunks_dev_current",
    )

    # Upsert documents
    result = db.upsert([
        {"id": "1", "embedding": [...], "content": "...", "metadata": {...}}
    ])

    # Search
    hits = db.search(query_vector=[...], top_k=10)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from backend.llm_infrastructure.vectordb.base import (
    VectorDBClient,
    SearchHit,
    UpsertResult,
    DeleteResult,
)
from backend.llm_infrastructure.vectordb.registry import register_vectordb

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


@register_vectordb("elasticsearch")
class EsVectorDB(VectorDBClient):
    """Elasticsearch vector database implementation.

    Supports dense vector search, hybrid search, and filtering.
    """

    def __init__(
        self,
        es_client: "Elasticsearch",
        index_name: str,
        vector_field: str = "embedding",
        content_field: str = "content",
        text_field: str = "search_text",
        id_field: str = "chunk_id",
        **kwargs: Any,
    ) -> None:
        """Initialize ES vector DB.

        Args:
            es_client: Elasticsearch client instance.
            index_name: Index name or alias.
            vector_field: Field name for dense vectors.
            content_field: Field name for content.
            text_field: Field name for BM25 text search.
            id_field: Field name for document ID.
            **kwargs: Additional config.
        """
        super().__init__(**kwargs)
        self.es = es_client
        self.index_name = index_name
        self.vector_field = vector_field
        self.content_field = content_field
        self.text_field = text_field
        self.id_field = id_field

    def upsert(
        self,
        documents: list[dict[str, Any]],
        *,
        batch_size: int = 100,
    ) -> UpsertResult:
        """Upsert documents to Elasticsearch.

        Args:
            documents: List of documents with id, embedding, content, metadata.
            batch_size: Bulk batch size.

        Returns:
            UpsertResult with counts.
        """
        from elasticsearch import helpers

        actions = []
        for doc in documents:
            doc_id = doc.get("id", doc.get(self.id_field))
            if not doc_id:
                continue

            source = {
                self.id_field: doc_id,
                self.vector_field: doc.get("embedding", []),
                self.content_field: doc.get("content", ""),
            }

            # Add metadata fields
            metadata = doc.get("metadata", {})
            for key, value in metadata.items():
                if key not in (self.id_field, self.vector_field, self.content_field):
                    source[key] = value

            # Build search_text if not provided
            if self.text_field not in source:
                source[self.text_field] = doc.get("content", "")

            actions.append({
                "_index": self.index_name,
                "_id": doc_id,
                "_source": source,
            })

        if not actions:
            return UpsertResult(succeeded=0, failed=0)

        try:
            success, failed = helpers.bulk(
                self.es,
                actions,
                chunk_size=batch_size,
                raise_on_error=False,
            )
            errors = []
            if isinstance(failed, list):
                errors = [str(f) for f in failed[:10]]  # Limit error messages
            return UpsertResult(
                succeeded=success,
                failed=len(failed) if isinstance(failed, list) else failed,
                errors=errors,
            )
        except Exception as e:
            logger.error("Bulk upsert failed: %s", e)
            return UpsertResult(
                succeeded=0,
                failed=len(actions),
                errors=[str(e)],
            )

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> list[SearchHit]:
        """Dense vector search using kNN.

        Args:
            query_vector: Query embedding.
            top_k: Number of results.
            filters: ES filter clause.
            include_metadata: Include metadata in results.

        Returns:
            List of SearchHit.
        """
        knn_query: dict[str, Any] = {
            "field": self.vector_field,
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": top_k * 2,
        }
        if filters:
            knn_query["filter"] = filters

        source_fields = [self.id_field, self.content_field]
        if include_metadata:
            source_fields.append("*")

        body: dict[str, Any] = {
            "knn": knn_query,
            "size": top_k,
            "_source": source_fields if include_metadata else source_fields[:2],
        }

        try:
            resp = self.es.search(index=self.index_name, body=body)
            return self._parse_hits(resp)
        except Exception as e:
            logger.error("Search failed: %s", e)
            return []

    def delete(
        self,
        ids: list[str],
    ) -> DeleteResult:
        """Delete documents by ID.

        Args:
            ids: Document IDs to delete.

        Returns:
            DeleteResult with counts.
        """
        from elasticsearch import helpers

        actions = [
            {"_op_type": "delete", "_index": self.index_name, "_id": doc_id}
            for doc_id in ids
        ]

        if not actions:
            return DeleteResult(deleted=0)

        try:
            success, failed = helpers.bulk(
                self.es,
                actions,
                raise_on_error=False,
            )
            return DeleteResult(
                deleted=success,
                not_found=len(failed) if isinstance(failed, list) else failed,
            )
        except Exception as e:
            logger.error("Bulk delete failed: %s", e)
            return DeleteResult(deleted=0, errors=[str(e)])

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
        """Hybrid search using dense + BM25.

        Args:
            query_vector: Query embedding.
            query_text: Query text for BM25.
            top_k: Number of results.
            dense_weight: Dense score weight.
            sparse_weight: Sparse score weight.
            filters: ES filter clause.

        Returns:
            List of SearchHit.
        """
        script_source = (
            f"params.dense_weight * cosineSimilarity(params.query_vector, '{self.vector_field}') "
            f"+ params.sparse_weight * _score + 1.0"
        )

        match_query: dict[str, Any] = {
            "match": {
                self.text_field: query_text,
            }
        }

        query: dict[str, Any] = {
            "script_score": {
                "query": match_query,
                "script": {
                    "source": script_source,
                    "params": {
                        "query_vector": query_vector,
                        "dense_weight": float(dense_weight),
                        "sparse_weight": float(sparse_weight),
                    },
                },
            }
        }

        if filters:
            query = {
                "bool": {
                    "must": query,
                    "filter": filters,
                }
            }

        body: dict[str, Any] = {
            "query": query,
            "size": top_k,
            "_source": True,
            "track_total_hits": False,
        }

        try:
            resp = self.es.search(index=self.index_name, body=body)
            return self._parse_hits(resp)
        except Exception as e:
            logger.error("Hybrid search failed: %s", e)
            return []

    def health_check(self) -> bool:
        """Check if ES is reachable."""
        try:
            return self.es.ping()
        except Exception:
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get index stats."""
        try:
            stats = self.es.indices.stats(index=self.index_name)
            return {
                "index": self.index_name,
                "doc_count": stats["_all"]["primaries"]["docs"]["count"],
                "size_bytes": stats["_all"]["primaries"]["store"]["size_in_bytes"],
            }
        except Exception as e:
            logger.warning("Failed to get stats: %s", e)
            return {"index": self.index_name, "error": str(e)}

    def _parse_hits(self, resp: dict[str, Any]) -> list[SearchHit]:
        """Parse ES response to SearchHit list."""
        hits = resp.get("hits", {}).get("hits", [])
        results: list[SearchHit] = []

        for hit in hits:
            source = hit.get("_source", {})
            results.append(
                SearchHit(
                    id=source.get(self.id_field, hit.get("_id", "")),
                    score=float(hit.get("_score", 0.0)),
                    content=source.get(self.content_field, ""),
                    metadata={
                        k: v
                        for k, v in source.items()
                        if k not in (self.vector_field,)
                    },
                )
            )

        return results


__all__ = ["EsVectorDB"]
