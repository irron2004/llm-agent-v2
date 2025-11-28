"""Retrieval and RAG pipeline."""

from .registry import (
    RetrieverRegistry,
    get_retriever,
    register_retriever,
)

# Trigger adapter registration side effects
from . import adapters  # noqa: F401

__all__ = [
    "RetrieverRegistry",
    "get_retriever",
    "register_retriever",
]
