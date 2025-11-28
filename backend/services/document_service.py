"""Document indexing service: preprocess → embed → index (dense + sparse)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from backend.llm_infrastructure.preprocessing.base import BasePreprocessor
from backend.llm_infrastructure.retrieval.engines import (
    BM25Index,
    StoredDocument,
    VectorStore,
)


@dataclass
class SourceDocument:
    """Raw document payload to ingest."""

    doc_id: str
    text: str
    metadata: dict[str, Any] | None = None


@dataclass
class IndexedCorpus:
    """Indexed artifacts for retrieval."""

    vector_store: VectorStore
    bm25_index: BM25Index | None
    documents: list[StoredDocument]


class DocumentIndexService:
    """Build dense/sparse indexes from raw documents."""

    def __init__(
        self,
        embedder: Any,
        preprocessor: BasePreprocessor | None = None,
        bm25_tokenizer=None,
        normalize_vectors: bool = True,
    ) -> None:
        self.embedder = embedder
        self.preprocessor = preprocessor
        self.bm25_tokenizer = bm25_tokenizer
        self.normalize_vectors = normalize_vectors

    def _preprocess(self, texts: Iterable[str]) -> list[str]:
        if self.preprocessor is None:
            return list(texts)
        return list(self.preprocessor.preprocess(texts))

    def _embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if hasattr(self.embedder, "embed_batch"):
            return self.embedder.embed_batch(list(texts))
        if hasattr(self.embedder, "embed_texts"):
            return self.embedder.embed_texts(list(texts))
        raise TypeError("embedder must implement embed_batch() or embed_texts()")

    def index(
        self,
        documents: Sequence[SourceDocument],
        *,
        preprocess: bool = True,
        build_sparse: bool = True,
        persist_dir: str | Path | None = None,
    ) -> IndexedCorpus:
        if not documents:
            raise ValueError("No documents provided for indexing")

        raw_texts = [doc.text for doc in documents]
        processed_texts = self._preprocess(raw_texts) if preprocess else raw_texts
        embeddings = self._embed_batch(processed_texts)
        if embeddings.ndim != 2:
            raise ValueError("embedder must return a 2D array for batch embedding")

        vector_store = VectorStore(
            dimension=embeddings.shape[1],
            normalize=self.normalize_vectors,
        )
        stored_docs: list[StoredDocument] = []
        for doc, text, emb in zip(documents, processed_texts, embeddings):
            stored = StoredDocument(
                doc_id=doc.doc_id,
                content=text,
                metadata=doc.metadata,
            )
            vector_store.add(stored, emb)
            stored_docs.append(stored)

        bm25_index = BM25Index(stored_docs, tokenizer=self.bm25_tokenizer) if build_sparse else None

        if persist_dir:
            vector_store.save(persist_dir)

        return IndexedCorpus(
            vector_store=vector_store,
            bm25_index=bm25_index,
            documents=stored_docs,
        )

    @staticmethod
    def load(
        path: str | Path,
        *,
        build_sparse: bool = True,
        bm25_tokenizer=None,
    ) -> IndexedCorpus:
        """Reload persisted vector store and rebuild sparse index if needed."""
        vector_store = VectorStore.load(path)
        docs = list(vector_store.iter_documents())
        bm25_index = BM25Index(docs, tokenizer=bm25_tokenizer) if build_sparse else None
        return IndexedCorpus(
            vector_store=vector_store,
            bm25_index=bm25_index,
            documents=docs,
        )

    @classmethod
    def from_settings(cls):
        """Initialize index service using global settings (preprocess + embedder)."""
        from backend.config.settings import rag_settings
        from backend.llm_infrastructure.preprocessing.registry import get_preprocessor
        from backend.services.embedding_service import EmbeddingService

        preprocessor = get_preprocessor(
            rag_settings.preprocess_method,
            version=rag_settings.preprocess_version,
        )
        embed_svc = EmbeddingService(
            method=rag_settings.embedding_method,
            version=rag_settings.embedding_version,
            device=rag_settings.embedding_device,
            use_cache=rag_settings.embedding_use_cache,
            cache_dir=rag_settings.embedding_cache_dir,
        )
        return cls(
            embedder=embed_svc.get_raw_embedder(),
            preprocessor=preprocessor,
            normalize_vectors=rag_settings.vector_normalize,
        )


__all__ = ["SourceDocument", "IndexedCorpus", "DocumentIndexService"]
