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

from ..rrf import merge_retrieval_result_lists_rrf

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

    from ..base import RetrievalResult

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
        from ..base import RetrievalResult

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
        device_boost: str | None = None,
        device_boost_weight: float = 2.0,
    ) -> list[EsSearchHit]:
        """Perform hybrid search combining dense and sparse.

        Two modes:
        1. use_rrf=True: Use app-level RRF (Reciprocal Rank Fusion)
        2. use_rrf=False: Use script_score with weighted combination

        Args:
            query_vector: Query embedding vector.
            query_text: Search query text.
            top_k: Number of results to return.
            dense_weight: Weight for dense (vector) score.
            sparse_weight: Weight for sparse (BM25) score.
            filters: Optional ES filter clause.
            use_rrf: Whether to use app-level RRF fusion.
            rrf_k: RRF constant (only used if use_rrf=True).
            device_boost: Optional device_name to boost.
            device_boost_weight: Boost weight for matching device (default: 2.0).

        Returns:
            List of search hits.
        """
        if use_rrf:
            return self._hybrid_search_rrf(
                query_vector,
                query_text,
                top_k,
                filters,
                rrf_k,
                device_boost,
                device_boost_weight,
            )
        return self._hybrid_search_script_score(
            query_vector,
            query_text,
            top_k,
            dense_weight,
            sparse_weight,
            filters,
            device_boost,
            device_boost_weight,
        )

    def _hybrid_search_rrf(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int,
        filters: dict[str, Any] | None,
        rrf_k: int,
        device_boost: str | None = None,
        device_boost_weight: float = 2.0,
    ) -> list[EsSearchHit]:
        """Hybrid search using app-level RRF over dense + sparse candidates."""

        def _has_value(value: object | None) -> bool:
            return value not in (None, "")

        def _dedupe_key(
            doc_id: str, metadata: dict[str, Any] | None
        ) -> tuple[str] | tuple[str, str]:
            base_metadata = metadata or {}

            chunk_id = base_metadata.get("chunk_id")
            if _has_value(chunk_id):
                return (doc_id, str(chunk_id))

            page = base_metadata.get("page")
            if _has_value(page):
                return (doc_id, str(page))

            return (doc_id,)

        def _rank_by_key(hits: list[EsSearchHit]) -> dict[tuple[str] | tuple[str, str], int]:
            rank_map: dict[tuple[str] | tuple[str, str], int] = {}
            for rank, hit in enumerate(hits, start=1):
                key = _dedupe_key(hit.doc_id, hit.metadata)
                if key not in rank_map:
                    rank_map[key] = rank
            return rank_map

        top_n = top_k * 2

        dense_hits = self.dense_search(query_vector=query_vector, top_k=top_n, filters=filters)

        sparse_query: dict[str, Any] = {
            "bool": {
                "must": self._build_text_query(query_text, device_boost, device_boost_weight),
            }
        }
        if filters:
            sparse_query["bool"]["filter"] = filters

        sparse_body: dict[str, Any] = {
            "query": sparse_query,
            "size": top_n,
            "_source": self._source_fields(),
        }
        sparse_resp = self.es.search(index=self.index_name, body=sparse_body)
        sparse_hits = self._parse_hits(sparse_resp)

        dense_rank_by_key = _rank_by_key(dense_hits)
        sparse_rank_by_key = _rank_by_key(sparse_hits)

        dense_results = [hit.to_retrieval_result() for hit in dense_hits]
        sparse_results = [hit.to_retrieval_result() for hit in sparse_hits]
        fused_results = merge_retrieval_result_lists_rrf([dense_results, sparse_results], k=rrf_k)

        fused_hits: list[EsSearchHit] = []
        for fused in fused_results[:top_k]:
            metadata = dict(fused.metadata or {})
            fused_key = _dedupe_key(fused.doc_id, metadata)
            metadata["rrf_dense_rank"] = dense_rank_by_key.get(fused_key)
            metadata["rrf_sparse_rank"] = sparse_rank_by_key.get(fused_key)
            metadata["rrf_score"] = fused.score
            metadata["rrf_k"] = rrf_k
            chunk_id = str(metadata.get("chunk_id") or fused.doc_id)
            page_value = metadata.get("page")
            page: int | None = None
            if isinstance(page_value, int):
                page = page_value
            elif isinstance(page_value, str):
                try:
                    page = int(page_value)
                except ValueError:
                    page = None
            fused_hits.append(
                EsSearchHit(
                    doc_id=fused.doc_id,
                    chunk_id=chunk_id,
                    content=fused.content,
                    score=fused.score,
                    page=page,
                    metadata=metadata,
                    raw_text=fused.raw_text,
                )
            )
        return fused_hits

    def _hybrid_search_script_score(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
        filters: dict[str, Any] | None,
        device_boost: str | None = None,
        device_boost_weight: float = 2.0,
    ) -> list[EsSearchHit]:
        """Hybrid search using script_score with weighted combination."""
        # Combine BM25 and cosine similarity using script_score
        script_source = (
            f"params.dense_weight * cosineSimilarity(params.query_vector, '{self.vector_field}') "
            f"+ params.sparse_weight * _score + 1.0"  # +1.0 to avoid negative scores
        )

        match_query = self._build_text_query(query_text, device_boost, device_boost_weight)

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
            "device_name",
            "equip_id",
            "title",
            "section_chapter",
            "section_number",
            "chapter_source",
            "chapter_ok",
        ]

    def fetch_section_chunks(
        self,
        doc_id: str,
        section_chapter: str,
        max_pages: int = 8,
        content_index: str | None = None,
        min_page: int | None = None,
        max_page: int | None = None,
    ) -> list[EsSearchHit]:
        """Fetch all chunks for a doc_id + section_chapter, sorted by page.

        Args:
            min_page: If set, only return chunks with page >= min_page.
            max_page: If set, only return chunks with page <= max_page.
                      Useful to scope results to the correct SOP in multi-SOP documents.
        """
        if not doc_id or not section_chapter:
            return []
        filters: list[dict[str, Any]] = [
            {"term": {"doc_id": doc_id}},
            {"term": {"section_chapter": section_chapter}},
        ]
        if min_page is not None:
            filters.append({"range": {"page": {"gte": min_page}}})
        if max_page is not None:
            filters.append({"range": {"page": {"lte": max_page}}})
        query: dict[str, Any] = {
            "bool": {
                "filter": filters,
            }
        }
        body: dict[str, Any] = {
            "query": query,
            "size": max_pages,
            "sort": [{"page": {"order": "asc"}}],
            "_source": self._source_fields(),
        }
        index = content_index or self.index_name
        try:
            resp = self.es.search(index=index, body=body)
            return self._parse_hits(resp)
        except Exception as e:
            logger.warning("Section chunk fetch failed: %s", e)
            return []

    def fetch_section_chunks_by_keyword(
        self,
        doc_id: str,
        keyword: str,
        max_pages: int = 8,
        content_index: str | None = None,
        min_page: int | None = None,
    ) -> list[EsSearchHit]:
        """Fetch chunks whose section_chapter contains *keyword*, sorted by page.

        Unlike ``fetch_section_chunks`` (exact match), this uses a wildcard
        query so that e.g. keyword="Work Procedure" matches
        "6. Work Procedure" or "Work Procedure / 작업 절차".

        Args:
            min_page: If set, only return chunks with page >= min_page.
                      Useful to skip earlier sections in multi-SOP documents.
        """
        if not doc_id or not keyword:
            return []
        filters: list[dict[str, Any]] = [
            {"term": {"doc_id": doc_id}},
        ]
        if min_page is not None:
            filters.append({"range": {"page": {"gte": min_page}}})
        query: dict[str, Any] = {
            "bool": {
                "filter": filters,
                "must": [
                    {"wildcard": {"section_chapter": f"*{keyword}*"}},
                ],
            }
        }
        body: dict[str, Any] = {
            "query": query,
            "size": max_pages,
            "sort": [{"page": {"order": "asc"}}],
            "_source": self._source_fields(),
        }
        index = content_index or self.index_name
        try:
            resp = self.es.search(index=index, body=body)
            return self._parse_hits(resp)
        except Exception as e:
            logger.warning("Section keyword fetch failed (doc=%s, kw=%s): %s", doc_id, keyword, e)
            return []

    def _build_text_query(
        self,
        query_text: str,
        device_boost: str | None = None,
        device_boost_weight: float = 2.0,
    ) -> dict[str, Any]:
        """Build text query for single or multi-field BM25 with optional device boost.

        Args:
            query_text: Search query text.
            device_boost: Optional device_name to boost.
            device_boost_weight: Boost weight for matching device.

        Returns:
            ES query clause.
        """
        # Base text query
        if len(self.text_fields) > 1 or any("^" in f for f in self.text_fields):
            base_query: dict[str, Any] = {
                "multi_match": {
                    "query": query_text,
                    "fields": self.text_fields,
                }
            }
        else:
            base_query = {"match": {self.text_fields[0]: query_text}}

        # If no device boost, return base query
        if not device_boost:
            return base_query

        # Wrap in bool query with should clause for device boost
        # This boosts documents with matching device_name without filtering others
        return {
            "bool": {
                "must": base_query,
                "should": [
                    {
                        "bool": {
                            "should": [
                                {
                                    "term": {
                                        "device_name": {
                                            "value": device_boost,
                                            "boost": device_boost_weight,
                                        }
                                    }
                                },
                                {
                                    "term": {
                                        "device_name.keyword": {
                                            "value": device_boost,
                                            "boost": device_boost_weight,
                                        }
                                    }
                                },
                            ],
                            "minimum_should_match": 1,
                        }
                    }
                ],
            }
        }

    def _parse_hits(self, resp: Any) -> list[EsSearchHit]:
        """Parse ES response into EsSearchHit objects."""
        hits = resp.get("hits", {}).get("hits", [])
        results: list[EsSearchHit] = []

        for hit in hits:
            source = hit.get("_source", {})
            # Compute chunk_id: use source.chunk_id, fallback to _id
            chunk_id = source.get("chunk_id", hit.get("_id", ""))
            # Parse score: handle None, missing, or non-numeric values
            raw_score = hit.get("_score")
            try:
                score = float(raw_score) if raw_score is not None else 0.0
            except (TypeError, ValueError):
                score = 0.0
            # Build metadata, ensuring chunk_id is always present
            metadata = {
                k: v for k, v in source.items() if k not in ("embedding", self.content_field)
            }
            if "chunk_id" not in metadata:
                metadata["chunk_id"] = chunk_id

            results.append(
                EsSearchHit(
                    doc_id=source.get("doc_id", hit.get("_id", "")),
                    chunk_id=chunk_id,
                    content=source.get(self.content_field, ""),
                    score=score,
                    page=source.get("page"),
                    metadata=metadata,
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
        doc_types: list[str] | None = None,
        doc_ids: list[str] | None = None,
        equip_ids: list[str] | None = None,
        lang: str | None = None,
        device_names: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Build ES filter clause from common filter parameters.

        Args:
            tenant_id: Filter by tenant.
            project_id: Filter by project.
            doc_type: Filter by document type.
            doc_ids: Filter by document IDs (OR logic).
            equip_ids: Filter by equip_id values (OR logic).
            lang: Filter by language.
            device_names: Filter by device names (OR logic - match any).

        Returns:
            ES filter clause or None if no filters.
        """

        def _term_or_keyword(field: str, value: str) -> dict[str, Any]:
            """Build a robust exact-match filter for both keyword and dynamic text mappings.

            Many indices map strings as `text` + `.keyword` multi-field via dynamic mapping.
            Some indices may map them as pure `keyword`. This helper matches either.
            """
            return {
                "bool": {
                    "should": [
                        {"term": {field: value}},
                        {"term": {f"{field}.keyword": value}},
                    ],
                    "minimum_should_match": 1,
                }
            }

        def _terms_or_keyword(field: str, values: list[str]) -> dict[str, Any]:
            """Build OR filter for multiple values."""
            should_clauses = []
            for value in values:
                should_clauses.append({"term": {field: value}})
                should_clauses.append({"term": {f"{field}.keyword": value}})
            return {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1,
                }
            }

        terms: list[dict[str, Any]] = []

        if tenant_id:
            terms.append(_term_or_keyword("tenant_id", tenant_id))
        if project_id:
            terms.append(_term_or_keyword("project_id", project_id))
        if doc_types:
            normalized = [dt for dt in doc_types if dt]
            if doc_type and doc_type not in normalized:
                normalized.append(doc_type)
            if normalized:
                terms.append(_terms_or_keyword("doc_type", normalized))
        elif doc_type:
            terms.append(_term_or_keyword("doc_type", doc_type))
        if lang:
            terms.append(_term_or_keyword("lang", lang))
        if device_names:
            terms.append(_terms_or_keyword("device_name", device_names))
        if doc_ids:
            normalized_doc_ids = [str(doc_id).strip() for doc_id in doc_ids if str(doc_id).strip()]
            if normalized_doc_ids:
                terms.append(_terms_or_keyword("doc_id", normalized_doc_ids))
        if equip_ids:
            normalized_eids = [str(eid).strip().upper() for eid in equip_ids if str(eid).strip()]
            if normalized_eids:
                terms.append(_terms_or_keyword("equip_id", normalized_eids))

        if not terms:
            return None
        if len(terms) == 1:
            return terms[0]
        return {"bool": {"must": terms}}


__all__ = ["EsSearchEngine", "EsSearchHit"]
