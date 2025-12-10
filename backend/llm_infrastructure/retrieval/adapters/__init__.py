"""Retriever adapters registered to the registry."""

# Import adapters to trigger @register_retriever side effects
from .dense import DenseRetriever
from .bm25 import BM25Retriever
from .hybrid import HybridRetriever
from .es_hybrid import EsHybridRetriever, EsDenseRetriever

__all__ = [
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "EsHybridRetriever",
    "EsDenseRetriever",
]
