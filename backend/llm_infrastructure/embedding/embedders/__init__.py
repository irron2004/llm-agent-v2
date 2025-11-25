"""Embedding method implementations.

Import all embedders here to ensure they are registered.
"""

from .sentence_transformer import SentenceTransformerEmbedder
from .tei_client import TEIEmbedder

__all__ = [
    "SentenceTransformerEmbedder",
    "TEIEmbedder",
]
