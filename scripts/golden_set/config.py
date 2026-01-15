"""Configuration and data models for Golden Set pooling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PoolingConfig:
    """Pooling configuration."""

    top_k_per_method: int = 50
    total_pool_size: int = 150
    min_per_doc_type: int = 10

    # Doc types to ensure diversity
    doc_types: list[str] = field(
        default_factory=lambda: ["sop", "ts_guide", "maintenance_log", "setup"]
    )

    # Near-duplicate removal thresholds
    similarity_threshold: float = 0.85
    doc_type_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "maintenance_log": 0.90,  # Logs need stricter dedup
            "sop": 0.75,
            "ts_guide": 0.75,
            "setup": 0.70,
        }
    )

    # Multi-query expansion (optional)
    enable_multi_query: bool = False
    multi_query_n: int = 2


@dataclass
class PooledDocument:
    """A document in the pool."""

    chunk_id: str
    doc_id: str
    content: str
    score: float
    source_method: str  # bm25, dense, hybrid, stratified
    doc_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "score": self.score,
            "source_method": self.source_method,
            "doc_type": self.doc_type,
            "metadata": self.metadata,
        }


@dataclass
class QueryPool:
    """Pool for a single query."""

    query_id: str
    query_text: str
    category: str
    difficulty: str
    documents: list[PooledDocument]
    pool_stats: dict[str, Any]


__all__ = ["PoolingConfig", "PooledDocument", "QueryPool"]
