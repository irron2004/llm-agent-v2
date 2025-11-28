"""Hybrid retriever: dense + sparse with RRF/weights."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from ..base import BaseRetriever, RetrievalResult
from ..registry import register_retriever


def _rrf_merge(
    result_lists: Iterable[list[RetrievalResult]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion with optional list-wise weights."""
    score_map: dict[str, float] = defaultdict(float)
    doc_payload: dict[str, RetrievalResult] = {}

    for list_idx, results in enumerate(result_lists):
        weight = 1.0 if weights is None else weights[list_idx]
        for rank, res in enumerate(results):
            score_map[res.doc_id] += weight * (1.0 / (k + rank + 1))
            if res.doc_id not in doc_payload:
                doc_payload[res.doc_id] = res

    fused: list[RetrievalResult] = []
    for doc_id, score in score_map.items():
        base = doc_payload[doc_id]
        fused.append(
            RetrievalResult(
                doc_id=doc_id,
                content=base.content,
                score=score,
                metadata=base.metadata,
            )
        )
    fused.sort(key=lambda r: r.score, reverse=True)
    return fused


@register_retriever("hybrid", version="v1")
class HybridRetriever(BaseRetriever):
    """Combine dense and sparse retrievers via RRF."""

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever: BaseRetriever | None = None,
        *,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        k = top_k or self.top_k
        dense_results = self.dense_retriever.retrieve(query, top_k=k, **kwargs)
        sparse_results: list[RetrievalResult] = []
        if self.sparse_retriever is not None:
            sparse_results = self.sparse_retriever.retrieve(query, top_k=k, **kwargs)

        # If only dense available, return as-is.
        if not sparse_results:
            results = dense_results
        else:
            results = _rrf_merge(
                [dense_results, sparse_results],
                k=self.rrf_k,
                weights=[self.dense_weight, self.sparse_weight],
            )

        if self.similarity_threshold > 0:
            results = [
                r for r in results if r.score >= self.similarity_threshold
            ]
        return results[:k]


__all__ = ["HybridRetriever"]
