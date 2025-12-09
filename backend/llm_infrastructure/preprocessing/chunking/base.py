"""Base classes for text chunking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class ChunkParams:
    """Parameters for chunking configuration.

    Attributes:
        chunk_size: Maximum size of each chunk (in characters or tokens).
        chunk_overlap: Number of characters/tokens to overlap between chunks.
        split_by: Unit for splitting - "char" (characters) or "token" (tokens).
        min_chunk_size: Minimum chunk size (chunks smaller than this are merged).
        separator: Preferred separator for splitting (e.g., "\n\n", "\n", " ").
        keep_separator: Whether to keep the separator in chunks.
    """

    chunk_size: int = 512
    chunk_overlap: int = 50
    split_by: str = "char"  # "char" or "token"
    min_chunk_size: int = 50
    separator: str | None = None
    keep_separator: bool = False


@dataclass
class ChunkedDocument:
    """A single chunk from a document.

    Attributes:
        text: The chunk text content.
        chunk_index: Index of this chunk within the source document (0-based).
        start_offset: Start character offset in the original document.
        end_offset: End character offset in the original document.
        source_doc_id: ID of the source document (for reference).
        chunk_id: Unique ID for this chunk (usually "{source_doc_id}_{chunk_index}").
        metadata: Additional metadata (preserved from source or added during chunking).
    """

    text: str
    chunk_index: int
    start_offset: int
    end_offset: int
    source_doc_id: str = ""
    chunk_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-generate chunk_id if not provided."""
        if not self.chunk_id and self.source_doc_id:
            self.chunk_id = f"{self.source_doc_id}_{self.chunk_index}"

    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return (
            f"ChunkedDocument(chunk_id={self.chunk_id!r}, "
            f"index={self.chunk_index}, "
            f"offset={self.start_offset}-{self.end_offset}, "
            f"text={preview!r})"
        )


class BaseChunker(ABC):
    """Base class for all text chunking methods.

    Each chunking method should:
    1. Inherit from this class
    2. Implement the chunk() method
    3. Register itself using @register_chunker decorator

    Example:
        ```python
        from .registry import register_chunker

        @register_chunker("my_chunker", version="v1")
        class MyChunker(BaseChunker):
            def chunk(self, text: str, doc_id: str = "") -> list[ChunkedDocument]:
                # Chunking logic here
                return [ChunkedDocument(text=text, chunk_index=0, ...)]
        ```
    """

    def __init__(self, params: ChunkParams | None = None, **kwargs: Any) -> None:
        """Initialize chunker with parameters.

        Args:
            params: ChunkParams instance with chunking configuration.
            **kwargs: Additional parameters (can override params fields).
        """
        if params is None:
            params = ChunkParams()

        # Allow kwargs to override params
        self.params = ChunkParams(
            chunk_size=kwargs.get("chunk_size", params.chunk_size),
            chunk_overlap=kwargs.get("chunk_overlap", params.chunk_overlap),
            split_by=kwargs.get("split_by", params.split_by),
            min_chunk_size=kwargs.get("min_chunk_size", params.min_chunk_size),
            separator=kwargs.get("separator", params.separator),
            keep_separator=kwargs.get("keep_separator", params.keep_separator),
        )
        self.config = kwargs

    @abstractmethod
    def chunk(
        self,
        text: str,
        doc_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkedDocument]:
        """Split text into chunks.

        Args:
            text: Input text to chunk.
            doc_id: Source document ID (for reference in chunks).
            metadata: Additional metadata to include in each chunk.

        Returns:
            List of ChunkedDocument instances.
        """
        raise NotImplementedError

    def chunk_batch(
        self,
        texts: Sequence[tuple[str, str]],
        metadata_list: Sequence[dict[str, Any]] | None = None,
    ) -> list[list[ChunkedDocument]]:
        """Chunk multiple documents.

        Args:
            texts: Sequence of (text, doc_id) tuples.
            metadata_list: Optional metadata for each document.

        Returns:
            List of chunk lists (one per input document).
        """
        if metadata_list is None:
            metadata_list = [{}] * len(texts)

        results = []
        for (text, doc_id), meta in zip(texts, metadata_list):
            chunks = self.chunk(text, doc_id=doc_id, metadata=meta)
            results.append(chunks)
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.params})"
