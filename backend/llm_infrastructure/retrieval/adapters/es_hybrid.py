"""Elasticsearch hybrid retriever adapter.

Provides a BaseRetriever implementation that uses EsSearchEngine for hybrid
(dense + sparse) retrieval from Elasticsearch.

Usage:
    from elasticsearch import Elasticsearch
    from backend.llm_infrastructure.retrieval.adapters.es_hybrid import EsHybridRetriever
    from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchEngine
    from backend.llm_infrastructure.embedding import get_embedder

    es_client = Elasticsearch(["http://localhost:9200"])
    engine = EsSearchEngine(es_client, "rag_chunks_dev_current")
    embedder = get_embedder("koe5")

    retriever = EsHybridRetriever(
        es_engine=engine,
        embedder=embedder,
        dense_weight=0.7,
        sparse_weight=0.3,
    )
    results = retriever.retrieve("search query", top_k=10)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.llm_infrastructure.retrieval.base import BaseRetriever, RetrievalResult
from backend.llm_infrastructure.retrieval.registry import register_retriever

if TYPE_CHECKING:
    from backend.llm_infrastructure.embedding.base import BaseEmbedder
    from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchEngine

logger = logging.getLogger(__name__)


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize a vector."""
    norm = np.linalg.norm(vec)
    norm = max(norm, eps)
    return vec / norm


@register_retriever("es_hybrid", version="v1")
class EsHybridRetriever(BaseRetriever):
    """Elasticsearch hybrid retriever using dense + sparse search.

    Combines vector similarity (cosine) with BM25 text matching.
    """

    def __init__(
        self,
        es_engine: "EsSearchEngine",
        embedder: "BaseEmbedder",
        *,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        top_k: int = 10,
        normalize_vectors: bool = True,
        use_rrf: bool = True,
        rrf_k: int = 60,
        preprocessor: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ES hybrid retriever.

        Args:
            es_engine: EsSearchEngine instance.
            embedder: Embedding model for query vectorization.
            dense_weight: Weight for dense (vector) scores (only used if use_rrf=False).
            sparse_weight: Weight for sparse (BM25) scores (only used if use_rrf=False).
            top_k: Default number of results to return.
            normalize_vectors: Whether to L2 normalize query vectors.
            use_rrf: Whether to use RRF for score combination (default: True to avoid candidate limiting).
            rrf_k: RRF constant (only used if use_rrf=True).
            preprocessor: Optional query preprocessor.
            **kwargs: Additional config.
        """
        super().__init__(**kwargs)
        self.es_engine = es_engine
        self.embedder = embedder
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.top_k = top_k
        self.normalize_vectors = normalize_vectors
        self.use_rrf = use_rrf
        self.rrf_k = rrf_k
        self.preprocessor = preprocessor

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        tenant_id: str | None = None,
        project_id: str | None = None,
        doc_type: str | None = None,
        lang: str | None = None,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            tenant_id: Optional tenant filter.
            project_id: Optional project filter.
            doc_type: Optional document type filter.
            lang: Optional language filter.
            **kwargs: Additional parameters.

        Returns:
            List of RetrievalResult sorted by score (descending).
        """
        k = top_k or self.top_k

        # Preprocess query if preprocessor is available
        processed_query = query
        if self.preprocessor is not None:
            try:
                processed = list(self.preprocessor.preprocess([query]))
                if processed:
                    processed_query = processed[0]
            except Exception as e:
                logger.warning("Query preprocessing failed: %s", e)

        # Embed query
        query_vec = self._embed_query(processed_query)

        # Build filter
        filters = self.es_engine.build_filter(
            tenant_id=tenant_id,
            project_id=project_id,
            doc_type=doc_type,
            lang=lang,
        )

        # Perform hybrid search
        hits = self.es_engine.hybrid_search(
            query_vector=query_vec,
            query_text=processed_query,
            top_k=k,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
            filters=filters,
            use_rrf=self.use_rrf,
            rrf_k=self.rrf_k,
        )

        # Convert to RetrievalResult
        return [hit.to_retrieval_result() for hit in hits]

    def _embed_query(self, query: str) -> list[float]:
        """Embed query text to vector."""
        if hasattr(self.embedder, "embed_batch"):
            vec = self.embedder.embed_batch([query])[0]
        elif hasattr(self.embedder, "embed_texts"):
            vec = self.embedder.embed_texts([query])[0]
        elif hasattr(self.embedder, "embed"):
            vec = self.embedder.embed(query)
        else:
            raise TypeError("embedder must implement embed(), embed_batch(), or embed_texts()")

        arr = np.asarray(vec, dtype=np.float32)
        if self.normalize_vectors:
            arr = _l2_normalize(arr)

        return arr.tolist()


@register_retriever("es_dense", version="v1")
class EsDenseRetriever(BaseRetriever):
    """Elasticsearch dense-only retriever using kNN vector search."""

    def __init__(
        self,
        es_engine: "EsSearchEngine",
        embedder: "BaseEmbedder",
        *,
        top_k: int = 10,
        normalize_vectors: bool = True,
        preprocessor: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ES dense retriever.

        Args:
            es_engine: EsSearchEngine instance.
            embedder: Embedding model for query vectorization.
            top_k: Default number of results to return.
            normalize_vectors: Whether to L2 normalize query vectors.
            preprocessor: Optional query preprocessor.
            **kwargs: Additional config.
        """
        super().__init__(**kwargs)
        self.es_engine = es_engine
        self.embedder = embedder
        self.top_k = top_k
        self.normalize_vectors = normalize_vectors
        self.preprocessor = preprocessor

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        tenant_id: str | None = None,
        project_id: str | None = None,
        doc_type: str | None = None,
        lang: str | None = None,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Retrieve using dense vector search only."""
        k = top_k or self.top_k

        # Preprocess query if preprocessor is available
        processed_query = query
        if self.preprocessor is not None:
            try:
                processed = list(self.preprocessor.preprocess([query]))
                if processed:
                    processed_query = processed[0]
            except Exception as e:
                logger.warning("Query preprocessing failed: %s", e)

        # Embed query
        query_vec = self._embed_query(processed_query)

        # Build filter
        filters = self.es_engine.build_filter(
            tenant_id=tenant_id,
            project_id=project_id,
            doc_type=doc_type,
            lang=lang,
        )

        # Perform dense search
        hits = self.es_engine.dense_search(
            query_vector=query_vec,
            top_k=k,
            filters=filters,
        )

        return [hit.to_retrieval_result() for hit in hits]

    def _embed_query(self, query: str) -> list[float]:
        """Embed query text to vector."""
        if hasattr(self.embedder, "embed_batch"):
            vec = self.embedder.embed_batch([query])[0]
        elif hasattr(self.embedder, "embed_texts"):
            vec = self.embedder.embed_texts([query])[0]
        elif hasattr(self.embedder, "embed"):
            vec = self.embedder.embed(query)
        else:
            raise TypeError("embedder must implement embed(), embed_batch(), or embed_texts()")

        arr = np.asarray(vec, dtype=np.float32)
        if self.normalize_vectors:
            arr = _l2_normalize(arr)

        return arr.tolist()


__all__ = ["EsHybridRetriever", "EsDenseRetriever"]
