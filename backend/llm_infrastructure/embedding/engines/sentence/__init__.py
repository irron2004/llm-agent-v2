"""SentenceTransformer embedding engine."""

from .factory import create_embedder
from .embedder import SentenceTransformerEmbedder

__all__ = ["create_embedder", "SentenceTransformerEmbedder"]
