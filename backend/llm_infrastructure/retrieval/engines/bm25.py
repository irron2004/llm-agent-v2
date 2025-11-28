"""BM25 sparse index wrapper."""

from __future__ import annotations

import re
from typing import Callable, Sequence

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - optional dependency
    BM25Okapi = None

from ..base import RetrievalResult
from .vector_store import StoredDocument


def default_tokenizer(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    """Simple BM25 wrapper to pair with VectorStore for hybrid search."""

    def __init__(
        self,
        documents: Sequence[StoredDocument],
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        if BM25Okapi is None:
            raise ImportError(
                "rank-bm25가 설치되어 있지 않습니다. `pip install rank-bm25` 후 다시 시도하세요."
            )
        self.tokenizer = tokenizer or default_tokenizer
        self._docs = list(documents)
        tokenized = [self.tokenizer(doc.content) for doc in self._docs]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if not self._docs:
            return []
        tokens = self.tokenizer(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]
        results: list[RetrievalResult] = []
        for idx, score in ranked:
            doc = self._docs[idx]
            results.append(
                RetrievalResult(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    score=float(score),
                    metadata=doc.metadata,
                )
            )
        return results

    def iter_documents(self) -> list[StoredDocument]:
        return list(self._docs)


__all__ = ["BM25Index", "default_tokenizer"]
