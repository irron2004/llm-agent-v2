"""Elasticsearch-backed search service (dense + BM25 hybrid).

Provides a high-level search interface compatible with SearchService,
using EsSearchEngine and EsHybridRetriever under the hood.

Usage:
    # Create from settings
    svc = EsSearchService.from_settings()
    results = svc.search("query text", top_k=10)

    # With filters
    results = svc.search(
        "query text",
        top_k=10,
        tenant_id="tenant_001",
        doc_type="sop",
    )
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from elasticsearch import Elasticsearch

from backend.config.settings import rag_settings, search_settings
from backend.llm_infrastructure.preprocessing.registry import get_preprocessor
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchEngine
from backend.llm_infrastructure.retrieval.adapters.es_hybrid import EsHybridRetriever
from backend.llm_infrastructure.reranking import get_reranker
from backend.llm_infrastructure.reranking.base import BaseReranker
from backend.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class EsSearchService:
    """Elasticsearch search service for RAG retrieval.

    Provides a unified interface for searching ES indices with support for:
    - Hybrid search (dense + sparse)
    - Dense-only search
    - Tenant/project filtering
    - Query preprocessing
    """

    def __init__(
        self,
        *,
        retriever: EsHybridRetriever,
        es_engine: EsSearchEngine | None = None,
        top_k: int = 10,
        method: str = "hybrid",
        reranker: Optional[BaseReranker] = None,
    ) -> None:
        """Initialize ES search service.

        Args:
            retriever: EsHybridRetriever instance.
            es_engine: Optional EsSearchEngine (for direct access).
            top_k: Default number of results.
            method: Search method (hybrid, dense).
            reranker: Optional reranker instance for reranking results.
        """
        self.retriever = retriever
        self.es_engine = es_engine
        self.top_k = top_k
        self.method = method
        self.reranker = reranker

    @classmethod
    def from_settings(
        cls,
        *,
        index: Optional[str] = None,
        es_client: Elasticsearch | None = None,
    ) -> "EsSearchService":
        """Create EsSearchService from global settings.

        Args:
            index: ES index/alias name. Defaults to current alias.
            es_client: Pre-configured ES client. Created from settings if None.

        Returns:
            Configured EsSearchService instance.
        """
        # Elasticsearch client
        if es_client is None:
            client_kwargs: dict[str, Any] = {
                "hosts": [search_settings.es_host],
                "verify_certs": True,
            }
            if search_settings.es_user and search_settings.es_password:
                client_kwargs["basic_auth"] = (
                    search_settings.es_user,
                    search_settings.es_password,
                )
            es_client = Elasticsearch(**client_kwargs)

        # Index name
        if index is None:
            index = f"{search_settings.es_index_prefix}_{search_settings.es_env}_current"

        # Embedder
        embed_svc = EmbeddingService(
            method=rag_settings.embedding_method,
            version=rag_settings.embedding_version,
            device=rag_settings.embedding_device,
            use_cache=rag_settings.embedding_use_cache,
            cache_dir=rag_settings.embedding_cache_dir,
        )

        # Preprocessor
        preprocessor = get_preprocessor(
            rag_settings.preprocess_method,
            version=rag_settings.preprocess_version,
            level=rag_settings.preprocess_level,
        )

        # ES Search Engine
        es_engine = EsSearchEngine(
            es_client=es_client,
            index_name=index,
            text_fields=[
                "search_text^1.0",
                "chunk_summary^0.7",
                "chunk_keywords^0.8",
            ],
        )

        # Dimension validation (guardrail)
        embedder_instance = embed_svc.get_raw_embedder()
        embedder_dims = embedder_instance.get_dimension()
        config_dims = search_settings.es_embedding_dims

        logger.info(
            f"Dimension check: embedder={embedder_dims}, config={config_dims}"
        )

        if embedder_dims != config_dims:
            raise ValueError(
                f"Embedding dimension mismatch detected!\n"
                f"  Embedder dimension: {embedder_dims}\n"
                f"  Config (SEARCH_ES_EMBEDDING_DIMS): {config_dims}\n"
                f"  Embedder method: {rag_settings.embedding_method}\n"
                f"  Action: Update SEARCH_ES_EMBEDDING_DIMS to {embedder_dims} "
                f"or use a different embedder"
            )

        # Optional: Validate against actual ES index if it exists
        try:
            from backend.llm_infrastructure.elasticsearch import EsIndexManager
            manager = EsIndexManager(
                es_client=es_client,
                env=search_settings.es_env,
                index_prefix=search_settings.es_index_prefix,
            )
            es_dims = manager.get_index_dims(use_alias=True)
            if es_dims is not None and es_dims != embedder_dims:
                raise ValueError(
                    f"ES index dimension mismatch detected!\n"
                    f"  Embedder dimension: {embedder_dims}\n"
                    f"  ES index dimension: {es_dims}\n"
                    f"  Index: {index}\n"
                    f"  Action: Reindex with correct dimensions or use matching embedder"
                )
        except ImportError:
            logger.warning("Could not import EsIndexManager for dimension validation")
        except Exception as e:
            logger.warning(f"Could not validate ES index dimensions: {e}")

        # ES Hybrid Retriever
        # use_rrf defaults to True (RRF mode), can be overridden per-request via API
        # When use_rrf=False, dense_weight/sparse_weight control the score combination
        retriever = EsHybridRetriever(
            es_engine=es_engine,
            embedder=embedder_instance,
            dense_weight=rag_settings.hybrid_dense_weight,
            sparse_weight=rag_settings.hybrid_sparse_weight,
            top_k=rag_settings.retrieval_top_k,
            normalize_vectors=rag_settings.vector_normalize,
            preprocessor=preprocessor,
        )

        # Reranker (optional)
        reranker: Optional[BaseReranker] = None
        if rag_settings.rerank_enabled:
            logger.info(
                f"Building reranker: method={rag_settings.rerank_method}, "
                f"model={rag_settings.rerank_model}"
            )
            reranker = get_reranker(
                rag_settings.rerank_method,
                version="v1",
                model_name=rag_settings.rerank_model,
                device=rag_settings.embedding_device,
            )

        return cls(
            retriever=retriever,
            es_engine=es_engine,
            top_k=rag_settings.retrieval_top_k,
            method=rag_settings.retrieval_method,
            reranker=reranker,
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        *,
        tenant_id: str | None = None,
        project_id: str | None = None,
        doc_type: str | None = None,
        lang: str | None = None,
        text_fields: list[str] | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        use_rrf: bool | None = None,
        rrf_k: int | None = None,
        device_name: str | None = None,
        device_boost_weight: float = 2.0,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Search for relevant documents.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            tenant_id: Optional tenant filter.
            project_id: Optional project filter.
            doc_type: Optional document type filter.
            lang: Optional language filter.
            text_fields: Optional list of text fields with weights (e.g. ["search_text^1.0", "chunk_summary^0.7"])
            dense_weight: Optional dense (vector) weight for hybrid search (overrides default)
            sparse_weight: Optional sparse (BM25) weight for hybrid search (overrides default)
            use_rrf: Whether to use RRF for score combination (True=RRF, False=weights)
            rrf_k: RRF rank constant (only used if use_rrf=True)
            device_name: Optional device_name to boost (not filter).
            device_boost_weight: Boost weight for matching device (default: 2.0).
            **kwargs: Additional parameters.

        Returns:
            List of RetrievalResult sorted by score (descending).
        """
        k = top_k or self.top_k

        try:
            # If custom text_fields are provided, temporarily override the engine's text_fields
            original_text_fields = None
            if text_fields is not None and self.es_engine is not None:
                original_text_fields = self.es_engine.text_fields
                self.es_engine.text_fields = text_fields

            # If custom hybrid weights are provided, temporarily override retriever's weights
            original_dense_weight = None
            original_sparse_weight = None
            original_use_rrf = None
            original_rrf_k = None

            if dense_weight is not None:
                original_dense_weight = self.retriever.dense_weight
                self.retriever.dense_weight = dense_weight
            if sparse_weight is not None:
                original_sparse_weight = self.retriever.sparse_weight
                self.retriever.sparse_weight = sparse_weight
            if use_rrf is not None:
                original_use_rrf = self.retriever.use_rrf
                self.retriever.use_rrf = use_rrf
            if rrf_k is not None:
                original_rrf_k = self.retriever.rrf_k
                self.retriever.rrf_k = rrf_k

            try:
                return self.retriever.retrieve(
                    query=query,
                    top_k=k,
                    tenant_id=tenant_id,
                    project_id=project_id,
                    doc_type=doc_type,
                    lang=lang,
                    device_name=device_name,
                    device_boost_weight=device_boost_weight,
                    **kwargs,
                )
            finally:
                # Restore original text_fields
                if original_text_fields is not None and self.es_engine is not None:
                    self.es_engine.text_fields = original_text_fields

                # Restore original hybrid weights
                if original_dense_weight is not None:
                    self.retriever.dense_weight = original_dense_weight
                if original_sparse_weight is not None:
                    self.retriever.sparse_weight = original_sparse_weight
                if original_use_rrf is not None:
                    self.retriever.use_rrf = original_use_rrf
                if original_rrf_k is not None:
                    self.retriever.rrf_k = original_rrf_k

        except Exception as exc:
            index = self.es_engine.index_name if self.es_engine is not None else "<unknown>"
            raise RuntimeError(
                f"Elasticsearch search failed: host={search_settings.es_host}, index={index}, error={exc}"
            ) from exc

    def fetch_doc_pages(
        self,
        doc_id: str,
        pages: list[int],
        *,
        max_docs: int | None = None,
    ) -> list[RetrievalResult]:
        """Fetch specific pages for a document by doc_id."""
        if self.es_engine is None:
            return []

        page_values = sorted({p for p in pages if isinstance(p, int) and p > 0})
        if not doc_id or not page_values:
            return []

        size = max_docs or len(page_values)

        doc_filter = {
            "bool": {
                "should": [
                    {"term": {"doc_id": doc_id}},
                    {"term": {"doc_id.keyword": doc_id}},
                ],
                "minimum_should_match": 1,
            }
        }

        body: dict[str, Any] = {
            "query": {
                "bool": {
                    "filter": [
                        doc_filter,
                        {"terms": {"page": page_values}},
                    ]
                }
            },
            "size": size,
            "sort": [{"page": "asc"}],
            "_source": self.es_engine._source_fields(),
            "track_total_hits": False,
        }

        try:
            resp = self.es_engine.es.search(index=self.es_engine.index_name, body=body)
            hits = self.es_engine._parse_hits(resp)
            return [hit.to_retrieval_result() for hit in hits]
        except Exception as exc:
            index = self.es_engine.index_name if self.es_engine is not None else "<unknown>"
            logger.warning(
                "fetch_doc_pages failed: doc_id=%s pages=%s index=%s err=%s",
                doc_id,
                page_values,
                index,
                exc,
            )
            return []

    def fetch_doc_chunks(
        self,
        doc_id: str,
        *,
        max_chunks: int = 50,
    ) -> list[RetrievalResult]:
        """Fetch all chunks for a document by doc_id."""
        if self.es_engine is None or not doc_id:
            return []

        doc_filter = {
            "bool": {
                "should": [
                    {"term": {"doc_id": doc_id}},
                    {"term": {"doc_id.keyword": doc_id}},
                ],
                "minimum_should_match": 1,
            }
        }

        body: dict[str, Any] = {
            "query": {"bool": {"filter": [doc_filter]}},
            "size": max_chunks,
            "sort": [{"chunk_id": "asc"}],
            "_source": self.es_engine._source_fields(),
            "track_total_hits": False,
        }

        try:
            resp = self.es_engine.es.search(index=self.es_engine.index_name, body=body)
            hits = self.es_engine._parse_hits(resp)
            return [hit.to_retrieval_result() for hit in hits]
        except Exception as exc:
            index = self.es_engine.index_name if self.es_engine is not None else "<unknown>"
            logger.warning(
                "fetch_doc_chunks failed: doc_id=%s index=%s err=%s",
                doc_id,
                index,
                exc,
            )
            return []

    def health_check(self) -> bool:
        """Check if ES is reachable.

        Returns:
            True if ES is healthy, False otherwise.
        """
        if self.es_engine is None:
            return False
        try:
            return self.es_engine.es.ping()
        except Exception as e:
            logger.warning("ES health check failed: %s", e)
            return False


__all__ = ["EsSearchService"]
