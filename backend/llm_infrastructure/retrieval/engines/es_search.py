"""Elasticsearch search engine for dense and sparse retrieval.

Provides low-level ES query building for:
- Dense search (kNN on dense_vector)
- Sparse search (BM25 on text fields)
- Hybrid search (combined dense + sparse with RRF or weighted scoring)

Usage:
    from elasticsearch import Elasticsearch
    from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchEngine

    es_client = Elasticsearch(["http://localhost:9200"])
    engine = EsSearchEngine(es_client, "rag_chunks_dev_current")

    # Dense search
    results = engine.dense_search(query_vector, top_k=10)

    # Hybrid search
    results = engine.hybrid_search(query_vector, "search query", top_k=10)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


@dataclass
class EsSearchHit:
    """Single Elasticsearch search hit."""

    doc_id: str
    chunk_id: str
    content: str
    score: float
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_text: str | None = None

    def to_retrieval_result(self) -> "RetrievalResult":
        """Convert to RetrievalResult."""
        from backend.llm_infrastructure.retrieval.base import RetrievalResult

        return RetrievalResult(
            doc_id=self.doc_id,
            content=self.content,
            score=self.score,
            metadata=self.metadata,
            raw_text=self.raw_text or self.content,
        )


class EsSearchEngine:
    """Elasticsearch search engine for RAG retrieval.

    Supports dense (vector) search, sparse (BM25) search, and hybrid search.
    """

    def __init__(
        self,
        es_client: "Elasticsearch",
        index_name: str,
        vector_field: str = "embedding",
        text_field: str = "search_text",
        text_fields: list[str] | None = None,
        content_field: str = "content",
    ) -> None:
        """Initialize ES search engine.

        Args:
            es_client: Elasticsearch client instance.
            index_name: Index name or alias to search.
            vector_field: Field name for dense vectors.
            text_field: Field name for BM25 text search (legacy single-field).
            text_fields: Field names for BM25 text search (multi-field).
            content_field: Field name for content retrieval.
        """
        self.es = es_client
        self.index_name = index_name
        self.vector_field = vector_field
        self.text_field = text_field
        if text_fields is None:
            text_fields = [text_field]
        self.text_fields = list(text_fields)
        self.content_field = content_field

    def dense_search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[EsSearchHit]:
        """Perform dense vector search using kNN.

        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            filters: Optional ES filter clause.

        Returns:
            List of search hits.
        """
        knn_query: dict[str, Any] = {
            "field": self.vector_field,
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": top_k * 2,
        }
        if filters:
            knn_query["filter"] = filters

        body: dict[str, Any] = {
            "knn": knn_query,
            "size": top_k,
            "_source": self._source_fields(),
        }

        try:
            resp = self.es.search(index=self.index_name, body=body)
            return self._parse_hits(resp)
        except Exception as e:
            logger.error("Dense search failed: %s", e)
            raise

    def sparse_search(
        self,
        query_text: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[EsSearchHit]:
        """Perform sparse BM25 text search.

        Args:
            query_text: Search query text.
            top_k: Number of results to return.
            filters: Optional ES filter clause.

        Returns:
            List of search hits.
        """
        query: dict[str, Any] = {
            "bool": {
                "must": self._build_text_query(query_text),
            }
        }
        if filters:
            query["bool"]["filter"] = filters

        body: dict[str, Any] = {
            "query": query,
            "size": top_k,
            "_source": self._source_fields(),
        }

        try:
            resp = self.es.search(index=self.index_name, body=body)
            return self._parse_hits(resp)
        except Exception as e:
            logger.error("Sparse search failed: %s", e)
            raise

    def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        filters: dict[str, Any] | None = None,
        use_rrf: bool = False,
        rrf_k: int = 60,
    ) -> list[EsSearchHit]:
        """Perform hybrid search combining dense and sparse.

        Two modes:
        1. use_rrf=True: Use ES 8.x RRF (Reciprocal Rank Fusion)
        2. use_rrf=False: Use script_score with weighted combination

        Args:
            query_vector: Query embedding vector.
            query_text: Search query text.
            top_k: Number of results to return.
            dense_weight: Weight for dense (vector) score.
            sparse_weight: Weight for sparse (BM25) score.
            filters: Optional ES filter clause.
            use_rrf: Whether to use RRF (requires ES 8.x+).
            rrf_k: RRF constant (only used if use_rrf=True).

        Returns:
            List of search hits.
        """
        if use_rrf:
            return self._hybrid_search_rrf(
                query_vector, query_text, top_k, filters, rrf_k
            )
        return self._hybrid_search_script_score(
            query_vector, query_text, top_k, dense_weight, sparse_weight, filters
        )

    def _hybrid_search_rrf(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int,
        filters: dict[str, Any] | None,
        rrf_k: int,
    ) -> list[EsSearchHit]:
        """Hybrid search using ES 8.x RRF."""
        knn_query: dict[str, Any] = {
            "field": self.vector_field,
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": top_k * 2,
        }
        if filters:
            knn_query["filter"] = filters

        text_query = self._build_text_query(query_text)
        if filters:
            text_query = {
                "bool": {
                    "must": text_query,
                    "filter": filters,
                }
            }

        body: dict[str, Any] = {
            "sub_searches": [
                {"query": {"knn": knn_query}},
                {"query": text_query},
            ],
            "rank": {
                "rrf": {
                    "window_size": top_k * 2,
                    "rank_constant": rrf_k,
                }
            },
            "size": top_k,
            "_source": self._source_fields(),
        }

        try:
            resp = self.es.search(index=self.index_name, body=body)
            return self._parse_hits(resp)
        except Exception as e:
            # RRF may not be supported in all ES versions
            logger.warning("RRF search failed, falling back to script_score: %s", e)
            return self._hybrid_search_script_score(
                query_vector, query_text, top_k, 0.7, 0.3, filters
            )

    def _hybrid_search_script_score(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
        filters: dict[str, Any] | None,
    ) -> list[EsSearchHit]:
        """Hybrid search using script_score with weighted combination."""
        # Combine BM25 and cosine similarity using script_score
        script_source = (
            f"params.dense_weight * cosineSimilarity(params.query_vector, '{self.vector_field}') "
            f"+ params.sparse_weight * _score + 1.0"  # +1.0 to avoid negative scores
        )

        match_query = self._build_text_query(query_text)

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

        # Add filter if provided
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
            "_source": self._source_fields(),
            "track_total_hits": False,
        }

        try:
            resp = self.es.search(index=self.index_name, body=body)
            return self._parse_hits(resp)
        except Exception as e:
            logger.error("Script score hybrid search failed: %s", e)
            raise

    def _source_fields(self) -> list[str]:
        """Get fields to include in search results."""
        return [
            "doc_id",
            "chunk_id",
            self.content_field,
            "search_text",
            "chunk_summary",
            "chunk_keywords",
            "page",
            "doc_type",
            "tenant_id",
            "project_id",
            "lang",
            "tags",
        ]

    def _build_text_query(self, query_text: str) -> dict[str, Any]:
        """Build text query for single or multi-field BM25."""
        if len(self.text_fields) > 1 or any("^" in f for f in self.text_fields):
            return {
                "multi_match": {
                    "query": query_text,
                    "fields": self.text_fields,
                }
            }
        return {"match": {self.text_fields[0]: query_text}}

    def _parse_hits(self, resp: dict[str, Any]) -> list[EsSearchHit]:
        """Parse ES response into EsSearchHit objects."""
        hits = resp.get("hits", {}).get("hits", [])
        results: list[EsSearchHit] = []

        for hit in hits:
            source = hit.get("_source", {})
            results.append(
                EsSearchHit(
                    doc_id=source.get("doc_id", hit.get("_id", "")),
                    chunk_id=source.get("chunk_id", hit.get("_id", "")),
                    content=source.get(self.content_field, ""),
                    score=float(hit.get("_score", 0.0)),
                    page=source.get("page"),
                    metadata={
                        k: v
                        for k, v in source.items()
                        if k not in ("embedding", self.content_field)
                    },
                    raw_text=source.get(self.content_field),
                )
            )

        return results

    def build_filter(
        self,
        *,
        tenant_id: str | None = None,
        project_id: str | None = None,
        doc_type: str | None = None,
        lang: str | None = None,
    ) -> dict[str, Any] | None:
        """Build ES filter clause from common filter parameters.

        Args:
            tenant_id: Filter by tenant.
            project_id: Filter by project.
            doc_type: Filter by document type.
            lang: Filter by language.

        Returns:
            ES filter clause or None if no filters.
        """
        terms: list[dict[str, Any]] = []

        if tenant_id:
            terms.append({"term": {"tenant_id": tenant_id}})
        if project_id:
            terms.append({"term": {"project_id": project_id}})
        if doc_type:
            terms.append({"term": {"doc_type": doc_type}})
        if lang:
            terms.append({"term": {"lang": lang}})

        if not terms:
            return None
        if len(terms) == 1:
            return terms[0]
        return {"bool": {"must": terms}}


__all__ = ["EsSearchEngine", "EsSearchHit"]
