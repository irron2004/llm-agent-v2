"""Chunking engine implementations.

Available engines:
- fixed_size: Fixed character/token size chunking
- recursive: (TODO) Recursive text splitting by separators
- semantic: (TODO) Semantic-based chunking using embeddings
"""

from .fixed_size import FixedSizeChunker

__all__ = [
    "FixedSizeChunker",
]
