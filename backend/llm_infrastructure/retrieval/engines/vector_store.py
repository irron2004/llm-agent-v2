"""Lightweight vector store for dense retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..base import RetrievalResult


def _l2_normalize(vecs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return vecs / norm


@dataclass
class StoredDocument:
    """Document payload saved alongside embeddings."""

    doc_id: str
    content: str  # Preprocessed text for embedding/search
    metadata: dict[str, Any] | None = None
    raw_text: str | None = None  # Original text for display/LLM context


class VectorStore:
    """Minimal in-memory vector store with optional disk persistence."""

    def __init__(self, dimension: int, normalize: bool = True) -> None:
        self.dimension = dimension
        self.normalize = normalize
        self._vectors = np.zeros((0, dimension), dtype=np.float32)
        self._docs: list[StoredDocument] = []

    @property
    def size(self) -> int:
        return len(self._docs)

    def add(self, doc: StoredDocument, embedding: np.ndarray) -> None:
        vec = np.asarray(embedding, dtype=np.float32)
        if vec.ndim != 1 or vec.shape[0] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {vec.shape}"
            )
        if self.normalize:
            vec = _l2_normalize(vec[None, :])[0]
        self._vectors = np.vstack([self._vectors, vec])
        self._docs.append(doc)

    def add_batch(
        self,
        docs: Sequence[StoredDocument],
        embeddings: np.ndarray,
    ) -> None:
        mat = np.asarray(embeddings, dtype=np.float32)
        if mat.ndim != 2 or mat.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding matrix shape mismatch: expected (*, {self.dimension}), got {mat.shape}"
            )
        if len(docs) != mat.shape[0]:
            raise ValueError(
                f"Document count {len(docs)} must match embedding rows {mat.shape[0]}"
            )
        if self.normalize:
            mat = _l2_normalize(mat)
        self._vectors = np.vstack([self._vectors, mat])
        self._docs.extend(docs)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[RetrievalResult]:
        if self.size == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, got {q.shape[1]}"
            )
        if self.normalize:
            q = _l2_normalize(q)
        scores = (self._vectors @ q.T).reshape(-1)
        ranked_idx = np.argsort(-scores)[:top_k]
        results = []
        for idx in ranked_idx:
            doc = self._docs[int(idx)]
            results.append(
                RetrievalResult(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    score=float(scores[idx]),
                    metadata=doc.metadata,
                    raw_text=doc.raw_text,
                )
            )
        return results

    def save(self, path: str | Path) -> None:
        """Persist vectors + docs to disk for reuse."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self._vectors)
        meta = {
            "dimension": self.dimension,
            "normalize": self.normalize,
            "docs": [asdict(doc) for doc in self._docs],
        }
        (path / "docs.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        path = Path(path)
        meta = json.loads((path / "docs.json").read_text(encoding="utf-8"))
        vectors = np.load(path / "vectors.npy")
        store = cls(dimension=int(meta["dimension"]), normalize=bool(meta.get("normalize", True)))
        docs = [StoredDocument(**d) for d in meta["docs"]]
        store.add_batch(docs, vectors)
        return store

    def iter_documents(self) -> Iterable[StoredDocument]:
        return iter(self._docs)


__all__ = ["VectorStore", "StoredDocument"]
