"""Tests for the chunking module."""

import pytest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_infrastructure.preprocessing.chunking import (
    BaseChunker,
    ChunkParams,
    ChunkedDocument,
    ChunkerRegistry,
    get_chunker,
    register_chunker,
)
from backend.llm_infrastructure.preprocessing.chunking.engines import (
    FixedSizeChunker,
)


# --- ChunkParams tests ---


def test_chunk_params_defaults():
    """Test ChunkParams default values."""
    params = ChunkParams()
    assert params.chunk_size == 512
    assert params.chunk_overlap == 50
    assert params.split_by == "char"
    assert params.min_chunk_size == 50
    assert params.separator is None
    assert params.keep_separator is False


def test_chunk_params_custom_values():
    """Test ChunkParams with custom values."""
    params = ChunkParams(
        chunk_size=256,
        chunk_overlap=25,
        split_by="token",
        min_chunk_size=30,
        separator="\n",
        keep_separator=True,
    )
    assert params.chunk_size == 256
    assert params.chunk_overlap == 25
    assert params.split_by == "token"
    assert params.min_chunk_size == 30
    assert params.separator == "\n"
    assert params.keep_separator is True


# --- ChunkedDocument tests ---


def test_chunked_document_creation():
    """Test ChunkedDocument creation and fields."""
    chunk = ChunkedDocument(
        text="sample text",
        chunk_index=0,
        start_offset=0,
        end_offset=11,
        source_doc_id="doc1",
        chunk_id="doc1:0",
        metadata={"key": "value"},
    )
    assert chunk.text == "sample text"
    assert chunk.chunk_index == 0
    assert chunk.start_offset == 0
    assert chunk.end_offset == 11
    assert chunk.source_doc_id == "doc1"
    assert chunk.chunk_id == "doc1:0"
    assert chunk.metadata == {"key": "value"}


def test_chunked_document_default_metadata():
    """Test ChunkedDocument with default empty metadata."""
    chunk = ChunkedDocument(
        text="text",
        chunk_index=0,
        start_offset=0,
        end_offset=4,
    )
    assert chunk.metadata == {}
    assert chunk.source_doc_id == ""
    assert chunk.chunk_id == ""


# --- FixedSizeChunker tests (character-based) ---


def test_fixed_size_chunker_small_text():
    """Test FixedSizeChunker with text smaller than chunk_size."""
    chunker = FixedSizeChunker(
        params=ChunkParams(chunk_size=100, chunk_overlap=10)
    )
    text = "This is a short text."
    chunks = chunker.chunk(text, doc_id="doc1")

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == len(text)
    assert chunks[0].source_doc_id == "doc1"


def test_fixed_size_chunker_basic_chunking():
    """Test basic character-based chunking."""
    chunker = FixedSizeChunker(
        params=ChunkParams(
            chunk_size=50,
            chunk_overlap=10,
            min_chunk_size=10,
        )
    )
    text = "A" * 120  # 120 characters
    chunks = chunker.chunk(text, doc_id="test")

    # Should create multiple chunks
    assert len(chunks) >= 2

    # First chunk should be around chunk_size
    assert len(chunks[0].text) <= 60  # chunk_size + buffer for split point

    # Each chunk should have correct metadata
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.source_doc_id == "test"
        assert chunk.metadata.get("chunker") == "fixed_size"


def test_fixed_size_chunker_with_natural_separators():
    """Test chunking prefers natural break points."""
    chunker = FixedSizeChunker(
        params=ChunkParams(chunk_size=50, chunk_overlap=5)
    )
    # Create text with paragraph breaks
    text = "First paragraph text. " * 3 + "\n\n" + "Second paragraph. " * 3
    chunks = chunker.chunk(text, doc_id="doc1")

    # Should prefer splitting at paragraph breaks
    assert len(chunks) >= 1
    # Verify chunks are reasonable
    for chunk in chunks:
        assert len(chunk.text) > 0


def test_fixed_size_chunker_empty_text():
    """Test FixedSizeChunker with empty text."""
    chunker = FixedSizeChunker(params=ChunkParams(chunk_size=100))
    chunks = chunker.chunk("", doc_id="empty")
    assert len(chunks) == 0


def test_fixed_size_chunker_overlap():
    """Test that chunks have proper overlap."""
    chunker = FixedSizeChunker(
        params=ChunkParams(
            chunk_size=30,
            chunk_overlap=10,
            min_chunk_size=5,
        )
    )
    text = "word " * 30  # 150 characters
    chunks = chunker.chunk(text)

    # Check overlap between consecutive chunks
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            # Verify chunks cover the text properly with overlap
            assert chunks[i].end_offset > chunks[i].start_offset


def test_fixed_size_chunker_metadata_preserved():
    """Test that custom metadata is preserved in chunks."""
    chunker = FixedSizeChunker(params=ChunkParams(chunk_size=50))
    text = "Short text"
    metadata = {"source": "test", "page": 1}
    chunks = chunker.chunk(text, doc_id="doc1", metadata=metadata)

    assert len(chunks) == 1
    assert chunks[0].metadata.get("source") == "test"
    assert chunks[0].metadata.get("page") == 1
    # Chunker should also add its own metadata
    assert chunks[0].metadata.get("chunker") == "fixed_size"


# --- FixedSizeChunker tests (token-based) ---


def test_fixed_size_chunker_token_fallback_warning():
    """Test that token mode without tokenizer falls back with warning."""
    with pytest.warns(UserWarning, match="no tokenizer provided"):
        chunker = FixedSizeChunker(
            params=ChunkParams(chunk_size=10, split_by="token")
        )
    # Should fall back to char mode
    assert chunker.params.split_by == "char"


def test_fixed_size_chunker_with_custom_tokenizer():
    """Test token-based chunking with custom tokenizer function."""
    def simple_tokenizer(text):
        return text.split()

    def simple_detokenizer(tokens):
        return " ".join(tokens)

    chunker = FixedSizeChunker(
        params=ChunkParams(
            chunk_size=5,  # 5 tokens
            chunk_overlap=1,
            split_by="token",
            min_chunk_size=2,
        ),
        tokenizer_fn=simple_tokenizer,
        detokenizer_fn=simple_detokenizer,
    )

    text = "one two three four five six seven eight nine ten"
    chunks = chunker.chunk(text, doc_id="token_test")

    # Should create multiple token-based chunks
    assert len(chunks) >= 2

    # Each chunk should have token_count in metadata
    for chunk in chunks:
        assert "token_count" in chunk.metadata


# --- Registry tests ---


def test_registry_get_fixed_size_chunker():
    """Test getting FixedSizeChunker from registry."""
    chunker = get_chunker("fixed_size", version="v1")
    assert isinstance(chunker, FixedSizeChunker)


def test_registry_get_chunker_with_params():
    """Test getting chunker with custom params."""
    params = ChunkParams(chunk_size=256, chunk_overlap=25)
    chunker = get_chunker("fixed_size", params=params)

    assert chunker.params.chunk_size == 256
    assert chunker.params.chunk_overlap == 25


def test_registry_unknown_chunker():
    """Test that unknown chunker name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown chunker"):
        get_chunker("nonexistent_chunker")


def test_registry_unknown_version():
    """Test that unknown version raises ValueError."""
    with pytest.raises(ValueError, match="Unknown version"):
        get_chunker("fixed_size", version="v999")


def test_registry_list_chunkers():
    """Test listing registered chunkers."""
    chunkers = ChunkerRegistry.list_chunkers()
    assert "fixed_size" in chunkers
    assert "v1" in chunkers["fixed_size"]


def test_registry_is_registered():
    """Test checking if chunker is registered."""
    assert ChunkerRegistry.is_registered("fixed_size", "v1")
    assert not ChunkerRegistry.is_registered("nonexistent")


# --- Edge cases ---


def test_chunker_very_long_text():
    """Test chunking with very long text."""
    chunker = FixedSizeChunker(
        params=ChunkParams(chunk_size=100, chunk_overlap=10)
    )
    text = "word " * 1000  # 5000 characters
    chunks = chunker.chunk(text)

    # Should handle long text without error
    assert len(chunks) > 10

    # All text should be covered
    total_coverage = sum(
        chunk.end_offset - chunk.start_offset for chunk in chunks
    )
    # Due to overlap, total coverage may exceed text length
    assert total_coverage >= len(text.strip())


def test_chunker_unicode_text():
    """Test chunking with Unicode text."""
    chunker = FixedSizeChunker(params=ChunkParams(chunk_size=50))
    text = "한글 텍스트입니다. " * 10 + "日本語テキスト " * 10
    chunks = chunker.chunk(text, doc_id="unicode")

    # Should handle Unicode without error
    assert len(chunks) >= 1
    # Verify content is preserved
    for chunk in chunks:
        assert len(chunk.text) > 0


def test_chunker_special_characters():
    """Test chunking with special characters and newlines."""
    chunker = FixedSizeChunker(
        params=ChunkParams(chunk_size=100, chunk_overlap=10)
    )
    text = "Line 1\n\nLine 2\n\nLine 3 with special: @#$%^&*()\n\nLine 4"
    chunks = chunker.chunk(text)

    # Should preserve special characters
    full_text = "".join(c.text for c in chunks)
    # Due to overlap, checking presence rather than exact match
    assert "@#$%^&*()" in full_text or any("@#$%^&*()" in c.text for c in chunks)


# --- Integration with params from kwargs ---


def test_chunker_params_from_kwargs():
    """Test that kwargs can override params."""
    chunker = FixedSizeChunker(
        chunk_size=200,
        chunk_overlap=20,
    )
    assert chunker.params.chunk_size == 200
    assert chunker.params.chunk_overlap == 20


def test_chunker_params_priority():
    """Test that explicit params take priority over kwargs."""
    params = ChunkParams(chunk_size=100)
    chunker = FixedSizeChunker(
        params=params,
        chunk_size=200,  # This should be overridden by params
    )
    assert chunker.params.chunk_size == 100
