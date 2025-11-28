"""Dense retriever using a VectorStore and an embedder."""

from __future__ import annotations

from typing import Any

from ..base import BaseRetriever, RetrievalResult
from ..registry import register_retriever
from ...embedding.base import BaseEmbedder
from ..engines import VectorStore


@register_retriever("dense", version="v1")
class DenseRetriever(BaseRetriever):
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: BaseEmbedder,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        **_: Any,
    ) -> list[RetrievalResult]:
        k = top_k or self.top_k
        query_vec = self.embedder.embed(query)
        results = self.vector_store.search(query_vec, top_k=k)
        if self.similarity_threshold > 0:
            results = [
                r for r in results if r.score >= self.similarity_threshold
            ]
        return results


__all__ = ["DenseRetriever"]
