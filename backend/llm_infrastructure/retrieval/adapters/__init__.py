"""Retriever adapters registered to the registry."""

# Import adapters to trigger @register_retriever side effects
from .dense import DenseRetriever
from .bm25 import BM25Retriever
from .hybrid import HybridRetriever

__all__ = ["DenseRetriever", "BM25Retriever", "HybridRetriever"]
