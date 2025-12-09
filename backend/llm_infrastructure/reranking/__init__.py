"""Reranking module for post-retrieval result refinement."""

from .base import BaseReranker
from .registry import (
    RerankerRegistry,
    register_reranker,
    get_reranker,
)

# Import adapters to trigger registration
from . import adapters  # noqa: F401

__all__ = [
    "BaseReranker",
    "RerankerRegistry",
    "register_reranker",
    "get_reranker",
]
