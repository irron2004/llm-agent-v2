"""Document indexing service: preprocess → chunk → embed → index (dense + sparse)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from backend.llm_infrastructure.preprocessing.base import BasePreprocessor
from backend.llm_infrastructure.preprocessing.chunking import (
    BaseChunker,
    ChunkedDocument,
    get_chunker,
)
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
    chunker: BaseChunker | None = None  # Chunker used for indexing


class DocumentIndexService:
    """Build dense/sparse indexes from raw documents.

    Pipeline: preprocess → chunk → embed → index (dense + sparse)
    """

    def __init__(
        self,
        embedder: Any,
        preprocessor: BasePreprocessor | None = None,
        chunker: BaseChunker | None = None,
        bm25_tokenizer=None,
        normalize_vectors: bool = True,
    ) -> None:
        self.embedder = embedder
        self.preprocessor = preprocessor
        self.chunker = chunker
        self.bm25_tokenizer = bm25_tokenizer
        self.normalize_vectors = normalize_vectors

    def _preprocess(self, texts: Iterable[str]) -> list[str]:
        if self.preprocessor is None:
            return list(texts)
        return list(self.preprocessor.preprocess(texts))

    def _chunk_documents(
        self,
        documents: Sequence[SourceDocument],
        processed_texts: Sequence[str],
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Chunk documents into smaller pieces.

        Args:
            documents: Original source documents.
            processed_texts: Preprocessed text for each document.

        Returns:
            List of tuples: (chunk_id, chunk_text, raw_text, metadata)
        """
        if self.chunker is None:
            # No chunking - return documents as-is
            return [
                (
                    doc.doc_id,
                    text,
                    doc.text,
                    doc.metadata or {},
                )
                for doc, text in zip(documents, processed_texts)
            ]

        chunks = []
        for doc, text in zip(documents, processed_texts):
            doc_metadata = doc.metadata.copy() if doc.metadata else {}

            # Chunk the preprocessed text
            chunked = self.chunker.chunk(
                text=text,
                doc_id=doc.doc_id,
                metadata=doc_metadata,
            )

            # Also get raw text chunks for display (approximate by offsets)
            for chunk in chunked:
                # Extract approximate raw text using offsets
                raw_chunk_text = doc.text[chunk.start_offset:chunk.end_offset]

                chunk_metadata = {
                    **doc_metadata,
                    **chunk.metadata,
                    "source_doc_id": doc.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                }

                chunks.append((
                    chunk.chunk_id or f"{doc.doc_id}:{chunk.chunk_index}",
                    chunk.text,  # Preprocessed chunk text
                    raw_chunk_text,  # Raw chunk text for display
                    chunk_metadata,
                ))

        return chunks

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
        chunker: Optional[BaseChunker] = None,
    ) -> None:
        """Save indexing metadata for reload consistency."""
        import json

        metadata = {
            "embedder": {
                "type": type(embedder).__name__,
                "module": type(embedder).__module__,
            },
            "preprocessor": None,
            "chunker": None,
        }

        # Try to extract preprocessor info
        if preprocessor is not None:
            metadata["preprocessor"] = {
                "type": type(preprocessor).__name__,
                "module": type(preprocessor).__module__,
            }

        # Try to extract chunker info
        if chunker is not None:
            chunker_info = {
                "type": type(chunker).__name__,
                "module": type(chunker).__module__,
            }
            # Include chunker params if available
            if hasattr(chunker, "params"):
                params = chunker.params
                chunker_info["params"] = {
                    "chunk_size": params.chunk_size,
                    "chunk_overlap": params.chunk_overlap,
                    "split_by": params.split_by,
                    "min_chunk_size": params.min_chunk_size,
                }
            metadata["chunker"] = chunker_info

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
        chunk: bool = True,
        build_sparse: bool = True,
        persist_dir: str | Path | None = None,
        embed_batch_size: int = 32,
    ) -> IndexedCorpus:
        """Index documents: preprocess → chunk → embed → index.

        Args:
            documents: Source documents to index.
            preprocess: Whether to preprocess text (default: True).
            chunk: Whether to chunk documents (default: True, requires chunker).
            build_sparse: Whether to build BM25 index (default: True).
            persist_dir: Directory to persist the index.
            embed_batch_size: Batch size for embedding (default: 32).

        Returns:
            IndexedCorpus with vector store, BM25 index, and documents.
        """
        if not documents:
            raise ValueError("No documents provided for indexing")

        # 1. Preprocess
        raw_texts = [doc.text for doc in documents]
        processed_texts = self._preprocess(raw_texts) if preprocess else raw_texts

        # 2. Chunk (if enabled and chunker is provided)
        if chunk and self.chunker is not None:
            chunks = self._chunk_documents(documents, processed_texts)
        else:
            # No chunking - treat each document as a single chunk
            chunks = [
                (
                    doc.doc_id,
                    text,
                    doc.text,
                    doc.metadata or {},
                )
                for doc, text in zip(documents, processed_texts)
            ]

        # 3. Embed in batches
        chunk_texts = [c[1] for c in chunks]  # Preprocessed chunk texts
        all_embeddings = []

        for i in range(0, len(chunk_texts), embed_batch_size):
            batch = chunk_texts[i:i + embed_batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
        if embeddings.size > 0 and embeddings.ndim != 2:
            raise ValueError("embedder must return a 2D array for batch embedding")

        # 4. Build vector store and stored documents
        dimension = embeddings.shape[1] if embeddings.size > 0 else 768  # Default dimension
        vector_store = VectorStore(
            dimension=dimension,
            normalize=self.normalize_vectors,
        )
        stored_docs: list[StoredDocument] = []

        for (chunk_id, chunk_text, raw_text, metadata), emb in zip(chunks, embeddings):
            stored = StoredDocument(
                doc_id=chunk_id,  # Chunk ID (e.g., "doc1:0", "doc1:1")
                content=chunk_text,  # Preprocessed text for search
                metadata=metadata,
                raw_text=raw_text,  # Raw text for display/LLM
            )
            vector_store.add(stored, emb)
            stored_docs.append(stored)

        # 5. Build BM25 index
        bm25_index = BM25Index(stored_docs, tokenizer=self.bm25_tokenizer) if build_sparse else None

        # 6. Persist if requested
        if persist_dir:
            persist_path = Path(persist_dir)
            vector_store.save(persist_path)

            # Save index metadata for reload consistency
            self._save_index_metadata(
                persist_path,
                embedder=self.embedder,
                preprocessor=self.preprocessor,
                chunker=self.chunker,
            )

        return IndexedCorpus(
            vector_store=vector_store,
            bm25_index=bm25_index,
            documents=stored_docs,
            embedder=self.embedder,
            preprocessor=self.preprocessor,
            chunker=self.chunker,
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
        """Initialize index service using global settings (preprocess + embedder + chunker)."""
        from backend.config.settings import rag_settings
        from backend.llm_infrastructure.preprocessing.registry import get_preprocessor
        from backend.llm_infrastructure.preprocessing.chunking import (
            ChunkParams,
            get_chunker,
        )
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

        # Create chunker if enabled
        chunker = None
        if rag_settings.chunking_enabled:
            chunk_params = ChunkParams(
                chunk_size=rag_settings.chunk_size,
                chunk_overlap=rag_settings.chunk_overlap,
                split_by=rag_settings.chunk_split_by,
                min_chunk_size=rag_settings.chunk_min_size,
            )
            chunker = get_chunker(
                name=rag_settings.chunking_method,
                version=rag_settings.chunking_version,
                params=chunk_params,
            )

        return cls(
            embedder=embed_svc.get_raw_embedder(),
            preprocessor=preprocessor,
            chunker=chunker,
            normalize_vectors=rag_settings.vector_normalize,
        )


__all__ = ["SourceDocument", "IndexedCorpus", "DocumentIndexService"]
