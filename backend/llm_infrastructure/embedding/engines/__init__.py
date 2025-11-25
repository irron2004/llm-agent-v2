"""Embedding engines."""

# Re-export engine factories here if needed
from .sentence import create_embedder

__all__ = ["create_embedder"]
