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
    embedder: Any | None = None  # Embedder used for indexing (for search consistency)
    preprocessor: BasePreprocessor | None = None  # Preprocessor used for indexing


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

    @staticmethod
    def _save_index_metadata(
        persist_dir: Path,
        embedder: Any,
        preprocessor: Optional[BasePreprocessor],
    ) -> None:
        """Save indexing metadata for reload consistency."""
        import json

        metadata = {
            "embedder": {
                "type": type(embedder).__name__,
                "module": type(embedder).__module__,
            },
            "preprocessor": None,
        }

        # Try to extract preprocessor info
        if preprocessor is not None:
            metadata["preprocessor"] = {
                "type": type(preprocessor).__name__,
                "module": type(preprocessor).__module__,
            }

        metadata_path = persist_dir / "index_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _load_index_metadata(persist_dir: Path) -> dict | None:
        """Load indexing metadata if exists."""
        import json

        metadata_path = persist_dir / "index_metadata.json"
        if not metadata_path.exists():
            return None

        return json.loads(metadata_path.read_text(encoding="utf-8"))

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
                content=text,  # Preprocessed text for search
                metadata=doc.metadata,
                raw_text=doc.text,  # Original text for display/LLM
            )
            vector_store.add(stored, emb)
            stored_docs.append(stored)

        bm25_index = BM25Index(stored_docs, tokenizer=self.bm25_tokenizer) if build_sparse else None

        if persist_dir:
            persist_path = Path(persist_dir)
            vector_store.save(persist_path)

            # Save index metadata for reload consistency
            self._save_index_metadata(
                persist_path,
                embedder=self.embedder,
                preprocessor=self.preprocessor,
            )

        return IndexedCorpus(
            vector_store=vector_store,
            bm25_index=bm25_index,
            documents=stored_docs,
            embedder=self.embedder,  # Store embedder for search consistency
            preprocessor=self.preprocessor,  # Store preprocessor for query consistency
        )

    @staticmethod
    def load(
        path: str | Path,
        *,
        build_sparse: bool = True,
        bm25_tokenizer=None,
        embedder: Any | None = None,
        preprocessor: BasePreprocessor | None = None,
        validate_metadata: bool = True,
    ) -> IndexedCorpus:
        """Reload persisted vector store and rebuild sparse index if needed.

        Args:
            path: Path to persisted index
            build_sparse: Whether to rebuild BM25 index
            bm25_tokenizer: Tokenizer for BM25
            embedder: Embedder to use (CRITICAL: must match indexing embedder)
            validate_metadata: Whether to validate against saved metadata

        Returns:
            IndexedCorpus with loaded data

        Raises:
            ValueError: If metadata validation fails and embedder not provided

        Warning:
            If embedder is not provided or doesn't match saved metadata,
            search results will be incorrect due to dimension/model mismatch.
        """
        import warnings

        persist_path = Path(path)
        vector_store = VectorStore.load(persist_path)
        docs = list(vector_store.iter_documents())

        # Load and validate metadata
        metadata = DocumentIndexService._load_index_metadata(persist_path)
        if validate_metadata and metadata:
            embedder_info = metadata.get("embedder", {})
            if embedder is None:
                warnings.warn(
                    f"No embedder provided for reloaded index. "
                    f"Original index used: {embedder_info.get('type', 'unknown')}. "
                    f"You MUST provide the same embedder to SearchService/RAGService "
                    f"or search results will be incorrect.",
                    UserWarning,
                    stacklevel=2,
                )
            elif embedder_info:
                # Basic type check
                provided_type = type(embedder).__name__
                expected_type = embedder_info.get("type")
                if provided_type != expected_type:
                    warnings.warn(
                        f"Embedder type mismatch: provided '{provided_type}' "
                        f"but index was created with '{expected_type}'. "
                        f"This may cause dimension mismatch or incorrect search.",
                        UserWarning,
                        stacklevel=2,
                    )

        bm25_index = BM25Index(docs, tokenizer=bm25_tokenizer) if build_sparse else None
        return IndexedCorpus(
            vector_store=vector_store,
            bm25_index=bm25_index,
            documents=docs,
            embedder=embedder,  # Pass through for consistency
            preprocessor=preprocessor,  # Pass through for query preprocessing
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
