"""High-level search orchestration service."""

from __future__ import annotations

import logging
from typing import Any, Optional

from backend.config.settings import rag_settings
from backend.llm_infrastructure.retrieval import get_retriever
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.llm_infrastructure.reranking import get_reranker
from backend.llm_infrastructure.reranking.base import BaseReranker
from backend.services.embedding_service import EmbeddingService
from backend.services.document_service import IndexedCorpus

logger = logging.getLogger(__name__)


class SearchService:
    """Compose retrievers (dense/bm25/hybrid) over a prepared corpus with optional reranking."""

    def __init__(
        self,
        corpus: IndexedCorpus,
        *,
        method: Optional[str] = None,
        version: Optional[str] = None,
        top_k: Optional[int] = None,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
        rrf_k: Optional[int] = None,
        # Reranking options
        rerank_enabled: Optional[bool] = None,
        rerank_method: Optional[str] = None,
        rerank_model: Optional[str] = None,
        rerank_top_k: Optional[int] = None,
        rerank_device: Optional[str] = None,
    ) -> None:
        self.corpus = corpus
        self.method = (method or rag_settings.retrieval_method).lower()
        self.version = version or rag_settings.retrieval_version
        self.top_k = top_k or rag_settings.retrieval_top_k
        self.dense_weight = dense_weight if dense_weight is not None else rag_settings.hybrid_dense_weight
        self.sparse_weight = sparse_weight if sparse_weight is not None else rag_settings.hybrid_sparse_weight
        self.rrf_k = rrf_k if rrf_k is not None else rag_settings.hybrid_rrf_k

        # Reranking settings
        self.rerank_enabled = rerank_enabled if rerank_enabled is not None else rag_settings.rerank_enabled
        self.rerank_method = rerank_method or rag_settings.rerank_method
        self.rerank_model = rerank_model or rag_settings.rerank_model
        self.rerank_top_k = rerank_top_k or rag_settings.rerank_top_k
        self.rerank_device = rerank_device or rag_settings.embedding_device

        # Use corpus embedder if available (ensures consistency), otherwise create from settings
        if corpus.embedder is not None:
            self._embedder = corpus.embedder
        else:
            self._embedding_service = EmbeddingService(
                method=rag_settings.embedding_method,
                version=rag_settings.embedding_version,
                device=rag_settings.embedding_device,
                use_cache=rag_settings.embedding_use_cache,
                cache_dir=rag_settings.embedding_cache_dir,
            )
            self._embedder = self._embedding_service.get_raw_embedder()

        self.retriever = self._build_retriever()
        self.reranker: Optional[BaseReranker] = self._build_reranker() if self.rerank_enabled else None

    def _build_dense(self, **kwargs: Any):
        if self.corpus.vector_store is None:
            raise ValueError("vector_store is required for dense retrieval")
        return get_retriever(
            "dense",
            version=self.version,
            vector_store=self.corpus.vector_store,
            embedder=self._embedder,
            top_k=kwargs.get("top_k", self.top_k),
            similarity_threshold=kwargs.get("similarity_threshold", 0.0),
        )

    def _build_sparse(self, **kwargs: Any):
        if self.corpus.bm25_index is None:
            raise ValueError("bm25_index is required for BM25 retrieval")
        return get_retriever(
            "bm25",
            version=self.version,
            bm25_index=self.corpus.bm25_index,
            top_k=kwargs.get("top_k", self.top_k),
        )

    def _build_retriever(self):
        if self.method == "dense":
            return self._build_dense()
        if self.method == "bm25":
            return self._build_sparse()
        if self.method == "hybrid":
            dense = self._build_dense()
            sparse = self._build_sparse() if self.corpus.bm25_index is not None else None
            return get_retriever(
                "hybrid",
                version=self.version,
                dense_retriever=dense,
                sparse_retriever=sparse,
                dense_weight=self.dense_weight,
                sparse_weight=self.sparse_weight,
                rrf_k=self.rrf_k,
                top_k=self.top_k,
            )
        raise ValueError(f"Unknown retrieval method: {self.method}")

    def _build_reranker(self) -> BaseReranker:
        """Build reranker based on settings."""
        logger.info(
            f"Building reranker: method={self.rerank_method}, "
            f"model={self.rerank_model}"
        )
        return get_reranker(
            self.rerank_method,
            version="v1",
            model_name=self.rerank_model,
            device=self.rerank_device,
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank: Optional[bool] = None,
        rerank_top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """Search for relevant documents with optional reranking.

        Args:
            query: Search query
            top_k: Number of results to retrieve (before reranking)
            rerank: Override reranking setting (None = use service setting)
            rerank_top_k: Number of results after reranking (None = use service setting)

        Returns:
            List of retrieval results (reranked if enabled)
        """
        # Determine effective top_k for retrieval
        retrieval_top_k = top_k or self.top_k

        # If reranking is enabled, retrieve more candidates for better reranking
        should_rerank = rerank if rerank is not None else self.rerank_enabled
        if should_rerank and self.reranker is not None:
            # Retrieve more candidates for reranking (e.g., 2x or at least 20)
            retrieval_top_k = max(retrieval_top_k * 2, 20, retrieval_top_k)

        # Retrieve documents
        results = self.retriever.retrieve(query, top_k=retrieval_top_k)

        # Apply reranking if enabled
        if should_rerank and self.reranker is not None and results:
            final_top_k = rerank_top_k or self.rerank_top_k or (top_k or self.top_k)
            logger.debug(
                f"Reranking {len(results)} results to top_k={final_top_k}"
            )
            results = self.reranker.rerank(query, results, top_k=final_top_k)

        return results


__all__ = ["SearchService"]
