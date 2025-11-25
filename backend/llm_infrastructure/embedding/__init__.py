"""Embedding infrastructure."""

# 어댑터를 import하여 레지스트리에 등록
from . import adapters  # noqa: F401
from .registry import get_embedder, register_embedder, EmbedderRegistry  # noqa: F401
from .base import BaseEmbedder  # noqa: F401

__all__ = ["adapters", "get_embedder", "register_embedder", "EmbedderRegistry", "BaseEmbedder"]
