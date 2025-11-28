"""Sparse BM25 retriever."""

from __future__ import annotations

from typing import Any

from ..base import BaseRetriever, RetrievalResult
from ..registry import register_retriever
from ..engines import BM25Index


@register_retriever("bm25", version="v1")
class BM25Retriever(BaseRetriever):
    def __init__(
        self,
        bm25_index: BM25Index,
        top_k: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.bm25_index = bm25_index
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        **_: Any,
    ) -> list[RetrievalResult]:
        k = top_k or self.top_k
        return self.bm25_index.search(query, top_k=k)


__all__ = ["BM25Retriever"]
