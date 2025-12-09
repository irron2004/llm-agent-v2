"""Fixed-size text chunker implementation."""

from typing import Any, Callable

from ..base import BaseChunker, ChunkedDocument, ChunkParams
from ..registry import register_chunker


@register_chunker("fixed_size", version="v1")
class FixedSizeChunker(BaseChunker):
    """Fixed-size text chunker with overlap support.

    Splits text into chunks of approximately equal size with configurable overlap.
    Supports both character-based and token-based splitting.

    For token-based splitting, a tokenizer must be provided. If not provided,
    falls back to character-based splitting with a warning.

    Example:
        ```python
        # Character-based chunking
        chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk("Long text here...", doc_id="doc1")

        # Token-based chunking (requires tokenizer)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        chunker = FixedSizeChunker(
            chunk_size=256,
            chunk_overlap=20,
            split_by="token",
            tokenizer=tokenizer,
        )
        chunks = chunker.chunk("Long text here...", doc_id="doc1")
        ```
    """

    # Default separators for finding good split points (priority order)
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

    def __init__(
        self,
        params: ChunkParams | None = None,
        tokenizer: Any | None = None,
        tokenizer_fn: Callable[[str], list[str]] | None = None,
        detokenizer_fn: Callable[[list[str]], str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize FixedSizeChunker.

        Args:
            params: ChunkParams configuration.
            tokenizer: HuggingFace tokenizer instance (for token-based splitting).
            tokenizer_fn: Custom tokenizer function (text -> token list).
            detokenizer_fn: Custom detokenizer function (token list -> text).
            **kwargs: Additional parameters (can override params).
        """
        super().__init__(params=params, **kwargs)

        self.tokenizer = tokenizer
        self.tokenizer_fn = tokenizer_fn
        self.detokenizer_fn = detokenizer_fn

        # Validate token-based splitting setup
        if self.params.split_by == "token":
            if not self._has_tokenizer():
                import warnings
                warnings.warn(
                    "Token-based splitting requested but no tokenizer provided. "
                    "Falling back to character-based splitting.",
                    UserWarning,
                    stacklevel=2,
                )
                self.params.split_by = "char"

    def _has_tokenizer(self) -> bool:
        """Check if tokenizer is available."""
        return (
            self.tokenizer is not None
            or self.tokenizer_fn is not None
        )

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into tokens."""
        if self.tokenizer_fn is not None:
            return self.tokenizer_fn(text)

        if self.tokenizer is not None:
            # HuggingFace tokenizer
            tokens = self.tokenizer.tokenize(text)
            return tokens

        # Fallback: simple whitespace tokenization
        return text.split()

    def _detokenize(self, tokens: list[str]) -> str:
        """Convert tokens back to text."""
        if self.detokenizer_fn is not None:
            return self.detokenizer_fn(tokens)

        if self.tokenizer is not None:
            # HuggingFace tokenizer
            return self.tokenizer.convert_tokens_to_string(tokens)

        # Fallback: join with space
        return " ".join(tokens)

    def _find_split_point(
        self,
        text: str,
        target_pos: int,
        search_range: int = 100,
    ) -> int:
        """Find a good split point near target position.

        Tries to split at sentence/paragraph boundaries if possible.

        Args:
            text: Text to split.
            target_pos: Target split position.
            search_range: Range to search for good split points.

        Returns:
            Actual split position.
        """
        if target_pos >= len(text):
            return len(text)

        # Define search window
        start = max(0, target_pos - search_range)
        end = min(len(text), target_pos + search_range)

        # Use custom separator if specified
        separators = (
            [self.params.separator]
            if self.params.separator
            else self.DEFAULT_SEPARATORS
        )

        # Find best split point (prefer earlier separators in list)
        best_pos = target_pos
        best_priority = len(separators)  # Lower is better

        for priority, sep in enumerate(separators):
            # Search backwards from target
            search_text = text[start:target_pos]
            sep_pos = search_text.rfind(sep)
            if sep_pos != -1:
                actual_pos = start + sep_pos + len(sep)
                if priority < best_priority:
                    best_pos = actual_pos
                    best_priority = priority
                    break  # Found good separator, stop searching

        return best_pos

    def _chunk_by_chars(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any],
    ) -> list[ChunkedDocument]:
        """Chunk text by character count."""
        chunks = []
        text_len = len(text)
        chunk_size = self.params.chunk_size
        overlap = self.params.chunk_overlap
        min_size = self.params.min_chunk_size

        if text_len == 0:
            return []

        # Handle small text
        if text_len <= chunk_size:
            return [
                ChunkedDocument(
                    text=text,
                    chunk_index=0,
                    start_offset=0,
                    end_offset=text_len,
                    source_doc_id=doc_id,
                    metadata=metadata.copy(),
                )
            ]

        pos = 0
        chunk_index = 0

        while pos < text_len:
            # Calculate end position
            end_pos = min(pos + chunk_size, text_len)

            # Find good split point if not at end
            if end_pos < text_len:
                end_pos = self._find_split_point(text, end_pos)

            # Ensure we always make progress
            if end_pos <= pos:
                end_pos = min(pos + chunk_size, text_len)

            # Extract chunk text
            chunk_text = text[pos:end_pos]

            # Skip if chunk is too small (unless it's the last one)
            if len(chunk_text) >= min_size or end_pos >= text_len:
                chunks.append(
                    ChunkedDocument(
                        text=chunk_text,
                        chunk_index=chunk_index,
                        start_offset=pos,
                        end_offset=end_pos,
                        source_doc_id=doc_id,
                        metadata=metadata.copy(),
                    )
                )
                chunk_index += 1

            # Move position with overlap
            prev_pos = pos
            pos = end_pos - overlap
            # Prevent infinite loop: ensure we always move forward
            if pos <= prev_pos:
                pos = end_pos

        return chunks

    def _chunk_by_tokens(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any],
    ) -> list[ChunkedDocument]:
        """Chunk text by token count."""
        chunks = []
        tokens = self._tokenize(text)
        total_tokens = len(tokens)
        chunk_size = self.params.chunk_size
        overlap = self.params.chunk_overlap
        min_size = self.params.min_chunk_size

        if total_tokens == 0:
            return []

        # Handle small text
        if total_tokens <= chunk_size:
            return [
                ChunkedDocument(
                    text=text,
                    chunk_index=0,
                    start_offset=0,
                    end_offset=len(text),
                    source_doc_id=doc_id,
                    metadata=metadata.copy(),
                )
            ]

        pos = 0
        chunk_index = 0

        while pos < total_tokens:
            # Calculate end position
            end_pos = min(pos + chunk_size, total_tokens)

            # Extract chunk tokens and convert back to text
            chunk_tokens = tokens[pos:end_pos]
            chunk_text = self._detokenize(chunk_tokens)

            # Calculate approximate character offsets
            # (This is approximate since tokenization may not preserve exact positions)
            start_offset = len(self._detokenize(tokens[:pos])) if pos > 0 else 0
            end_offset = start_offset + len(chunk_text)

            # Skip if chunk is too small (unless it's the last one)
            if len(chunk_tokens) >= min_size or end_pos >= total_tokens:
                chunks.append(
                    ChunkedDocument(
                        text=chunk_text,
                        chunk_index=chunk_index,
                        start_offset=start_offset,
                        end_offset=end_offset,
                        source_doc_id=doc_id,
                        metadata={
                            **metadata,
                            "token_count": len(chunk_tokens),
                        },
                    )
                )
                chunk_index += 1

            # Move position with overlap
            pos = end_pos - overlap
            if pos <= 0 or pos <= (end_pos - chunk_size):
                # Prevent infinite loop
                pos = end_pos

        return chunks

    def chunk(
        self,
        text: str,
        doc_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkedDocument]:
        """Split text into fixed-size chunks.

        Args:
            text: Input text to chunk.
            doc_id: Source document ID.
            metadata: Additional metadata for chunks.

        Returns:
            List of ChunkedDocument instances.
        """
        if metadata is None:
            metadata = {}

        # Add chunker info to metadata
        chunk_metadata = {
            **metadata,
            "chunker": "fixed_size",
            "chunk_size": self.params.chunk_size,
            "chunk_overlap": self.params.chunk_overlap,
            "split_by": self.params.split_by,
        }

        if self.params.split_by == "token":
            return self._chunk_by_tokens(text, doc_id, chunk_metadata)
        else:
            return self._chunk_by_chars(text, doc_id, chunk_metadata)
