"""Chunking module for text splitting.

This module provides text chunking (splitting) functionality for preprocessing
long documents into smaller, manageable chunks suitable for embedding and retrieval.

Usage:
    from backend.llm_infrastructure.preprocessing.chunking import get_chunker

    chunker = get_chunker("fixed_size", chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk("Long document text here...")

    for chunk in chunks:
        print(f"Chunk {chunk.chunk_index}: {chunk.text[:50]}...")
"""

from .base import BaseChunker, ChunkParams, ChunkedDocument
from .registry import ChunkerRegistry, register_chunker, get_chunker

# Import engines to register them
from . import engines

__all__ = [
    "BaseChunker",
    "ChunkParams",
    "ChunkedDocument",
    "ChunkerRegistry",
    "register_chunker",
    "get_chunker",
]
