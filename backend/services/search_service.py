"""High-level search orchestration service."""

from __future__ import annotations

from typing import Any, Optional

from backend.config.settings import rag_settings
from backend.llm_infrastructure.retrieval import get_retriever
from backend.services.embedding_service import EmbeddingService
from backend.services.document_service import IndexedCorpus


class SearchService:
    """Compose retrievers (dense/bm25/hybrid) over a prepared corpus."""

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
    ) -> None:
        self.corpus = corpus
        self.method = (method or rag_settings.retrieval_method).lower()
        self.version = version or rag_settings.retrieval_version
        self.top_k = top_k or rag_settings.retrieval_top_k
        self.dense_weight = dense_weight if dense_weight is not None else rag_settings.hybrid_dense_weight
        self.sparse_weight = sparse_weight if sparse_weight is not None else rag_settings.hybrid_sparse_weight
        self.rrf_k = rrf_k if rrf_k is not None else rag_settings.hybrid_rrf_k

        self._embedding_service = EmbeddingService(
            method=rag_settings.embedding_method,
            version=rag_settings.embedding_version,
            device=rag_settings.embedding_device,
            use_cache=rag_settings.embedding_use_cache,
            cache_dir=rag_settings.embedding_cache_dir,
        )
        self.retriever = self._build_retriever()

    def _build_dense(self, **kwargs: Any):
        if self.corpus.vector_store is None:
            raise ValueError("vector_store is required for dense retrieval")
        return get_retriever(
            "dense",
            version=self.version,
            vector_store=self.corpus.vector_store,
            embedder=self._embedding_service.get_raw_embedder(),
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

    def search(self, query: str, top_k: Optional[int] = None):
        return self.retriever.retrieve(query, top_k=top_k or self.top_k)


__all__ = ["SearchService"]
