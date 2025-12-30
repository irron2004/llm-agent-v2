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
    ) -> None:
        """Initialize ES search service.

        Args:
            retriever: EsHybridRetriever instance.
            es_engine: Optional EsSearchEngine (for direct access).
            top_k: Default number of results.
            method: Search method (hybrid, dense).
        """
        self.retriever = retriever
        self.es_engine = es_engine
        self.top_k = top_k
        self.method = method

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
                "chunk_keywords.text^0.8",
            ],
        )

        # ES Hybrid Retriever
        retriever = EsHybridRetriever(
            es_engine=es_engine,
            embedder=embed_svc.get_raw_embedder(),
            dense_weight=rag_settings.hybrid_dense_weight,
            sparse_weight=rag_settings.hybrid_sparse_weight,
            top_k=rag_settings.retrieval_top_k,
            normalize_vectors=rag_settings.vector_normalize,
            preprocessor=preprocessor,
        )

        return cls(
            retriever=retriever,
            es_engine=es_engine,
            top_k=rag_settings.retrieval_top_k,
            method=rag_settings.retrieval_method,
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

            try:
                return self.retriever.retrieve(
                    query=query,
                    top_k=k,
                    tenant_id=tenant_id,
                    project_id=project_id,
                    doc_type=doc_type,
                    lang=lang,
                    **kwargs,
                )
            finally:
                # Restore original text_fields
                if original_text_fields is not None and self.es_engine is not None:
                    self.es_engine.text_fields = original_text_fields

        except Exception as exc:
            index = self.es_engine.index_name if self.es_engine is not None else "<unknown>"
            raise RuntimeError(
                f"Elasticsearch search failed: host={search_settings.es_host}, index={index}, error={exc}"
            ) from exc

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
