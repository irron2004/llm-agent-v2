"""High-level search orchestration service."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

from backend.config.settings import rag_settings
from backend.llm_infrastructure.retrieval import get_retriever
from backend.llm_infrastructure.retrieval.base import RetrievalResult
from backend.llm_infrastructure.reranking import get_reranker
from backend.llm_infrastructure.reranking.base import BaseReranker
from backend.llm_infrastructure.query_expansion import get_query_expander
from backend.llm_infrastructure.query_expansion.base import BaseQueryExpander
from backend.services.embedding_service import EmbeddingService
from backend.services.document_service import IndexedCorpus

logger = logging.getLogger(__name__)


class SearchService:
    """Compose retrievers (dense/bm25/hybrid) over a prepared corpus with optional multi-query and reranking."""

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
        # Multi-query expansion options
        multi_query_enabled: Optional[bool] = None,
        multi_query_method: Optional[str] = None,
        multi_query_n: Optional[int] = None,
        multi_query_include_original: Optional[bool] = None,
        multi_query_prompt: Optional[str] = None,
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

        # Multi-query expansion settings
        self.multi_query_enabled = (
            multi_query_enabled if multi_query_enabled is not None
            else rag_settings.multi_query_enabled
        )
        self.multi_query_method = multi_query_method or rag_settings.multi_query_method
        self.multi_query_n = multi_query_n or rag_settings.multi_query_n
        self.multi_query_include_original = (
            multi_query_include_original if multi_query_include_original is not None
            else rag_settings.multi_query_include_original
        )
        self.multi_query_prompt = multi_query_prompt or rag_settings.multi_query_prompt

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
        self.query_expander: Optional[BaseQueryExpander] = (
            self._build_query_expander() if self.multi_query_enabled else None
        )
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

    def _build_query_expander(self) -> BaseQueryExpander:
        """Build query expander based on settings."""
        logger.info(
            f"Building query expander: method={self.multi_query_method}, "
            f"n={self.multi_query_n}, prompt={self.multi_query_prompt}"
        )
        return get_query_expander(
            self.multi_query_method,
            version="v1",
            prompt_template=self.multi_query_prompt,
        )

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

    def _merge_results_rrf(
        self,
        result_lists: list[list[RetrievalResult]],
        k: int = 60,
    ) -> list[RetrievalResult]:
        """Merge multiple result lists using Reciprocal Rank Fusion.

        Args:
            result_lists: List of retrieval result lists
            k: RRF parameter (default: 60)

        Returns:
            Merged and deduplicated results sorted by RRF score
        """
        score_map: dict[str, float] = defaultdict(float)
        doc_payload: dict[str, RetrievalResult] = {}
        original_scores: dict[str, list[float]] = defaultdict(list)
        query_hits: dict[str, int] = defaultdict(int)

        for query_idx, results in enumerate(result_lists):
            for rank, result in enumerate(results):
                score_map[result.doc_id] += 1.0 / (k + rank + 1)
                original_scores[result.doc_id].append(result.score)
                query_hits[result.doc_id] += 1
                if result.doc_id not in doc_payload:
                    doc_payload[result.doc_id] = result

        merged: list[RetrievalResult] = []
        for doc_id, rrf_score in score_map.items():
            base = doc_payload[doc_id]
            # Preserve original metadata and add RRF info
            merged_metadata = dict(base.metadata) if base.metadata else {}
            merged_metadata.update({
                "rrf_score": rrf_score,
                "original_scores": original_scores[doc_id],
                "query_hits": query_hits[doc_id],
                "total_queries": len(result_lists),
            })
            merged.append(
                RetrievalResult(
                    doc_id=doc_id,
                    content=base.content,
                    score=rrf_score,
                    metadata=merged_metadata,
                    raw_text=base.raw_text,
                )
            )

        merged.sort(key=lambda r: r.score, reverse=True)
        return merged

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        multi_query: Optional[bool] = None,
        multi_query_n: Optional[int] = None,
        rerank: Optional[bool] = None,
        rerank_top_k: Optional[int] = None,
        device_name: Optional[str] = None,
        device_names: Optional[list[str]] = None,
    ) -> list[RetrievalResult]:
        """Search for relevant documents with optional multi-query expansion and reranking.

        Pipeline: Query → [Multi-Query Expansion] → Retrieve → [Merge RRF] → [Rerank] → Results

        Args:
            query: Search query
            top_k: Number of results to retrieve (before reranking)
            multi_query: Override multi-query setting (None = use service setting)
            multi_query_n: Override number of expanded queries (None = use service setting)
            rerank: Override reranking setting (None = use service setting)
            rerank_top_k: Number of results after reranking (None = use service setting)
            device_name: Optional single device_name to boost (legacy)
            device_names: Optional list of device names to filter (OR logic)

        Returns:
            List of retrieval results
        """
        final_top_k = top_k or self.top_k

        # Determine if multi-query expansion should be used
        should_expand = multi_query if multi_query is not None else self.multi_query_enabled
        should_rerank = rerank if rerank is not None else self.rerank_enabled

        # Calculate retrieval top_k (retrieve more if reranking)
        retrieval_top_k = final_top_k
        if should_rerank and self.reranker is not None:
            retrieval_top_k = max(final_top_k * 2, 20)

        # Build retriever kwargs for device filtering
        retriever_kwargs: dict = {}
        if device_names:
            retriever_kwargs["device_names"] = device_names
        elif device_name:
            retriever_kwargs["device_name"] = device_name

        # Step 1: Multi-Query Expansion (if enabled)
        if should_expand and self.query_expander is not None:
            n = multi_query_n or self.multi_query_n
            expanded = self.query_expander.expand(
                query,
                n=n,
                include_original=self.multi_query_include_original,
            )
            queries = expanded.get_all_queries()
            logger.debug(f"Expanded query into {len(queries)} queries: {queries}")

            # Step 2: Retrieve for each query
            all_results: list[list[RetrievalResult]] = []
            for q in queries:
                results = self.retriever.retrieve(q, top_k=retrieval_top_k, **retriever_kwargs)
                all_results.append(results)

            # Step 3: Merge results using RRF
            results = self._merge_results_rrf(all_results, k=self.rrf_k)
            logger.debug(f"Merged {sum(len(r) for r in all_results)} results into {len(results)}")

        else:
            # Single query retrieval
            results = self.retriever.retrieve(query, top_k=retrieval_top_k, **retriever_kwargs)

        # Step 4: Reranking (if enabled)
        if should_rerank and self.reranker is not None and results:
            rerank_k = rerank_top_k or self.rerank_top_k or final_top_k
            logger.debug(f"Reranking {len(results)} results to top_k={rerank_k}")
            results = self.reranker.rerank(query, results, top_k=rerank_k)

        # Limit to final top_k
        return results[:final_top_k]


__all__ = ["SearchService"]
