"""Retrieval presets - predefined retrieval configurations.

This module provides a way to define and manage retrieval presets,
which combine multiple retrieval strategies and their parameters.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalPreset:
    """Retrieval preset configuration.

    A preset combines:
    - Search method (dense, sparse, hybrid)
    - Parameters (top_k, threshold, etc.)
    - Multi-query settings
    - Reranking settings
    """

    name: str
    description: str = ""

    # Core retrieval
    retrieval_method: str = "dense"
    retrieval_version: str = "v1"
    top_k: int = 10
    similarity_threshold: float = 0.0

    # Hybrid-specific
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rrf_k: int = 60  # Reciprocal Rank Fusion parameter

    # Multi-query
    multi_query_enabled: bool = False
    multi_query_n: int = 4
    multi_query_include_original: bool = True

    # Reranking
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 3

    # Additional parameters
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert preset to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "retrieval_method": self.retrieval_method,
            "retrieval_version": self.retrieval_version,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "rrf_k": self.rrf_k,
            "multi_query_enabled": self.multi_query_enabled,
            "multi_query_n": self.multi_query_n,
            "multi_query_include_original": self.multi_query_include_original,
            "rerank_enabled": self.rerank_enabled,
            "rerank_model": self.rerank_model,
            "rerank_top_k": self.rerank_top_k,
            "extra": self.extra,
        }


class PresetRegistry:
    """Registry for retrieval presets."""

    _presets: dict[str, RetrievalPreset] = {}

    @classmethod
    def register(cls, preset: RetrievalPreset) -> None:
        """Register a preset."""
        if preset.name in cls._presets:
            raise ValueError(f"Preset '{preset.name}' already registered")
        cls._presets[preset.name] = preset

    @classmethod
    def get(cls, name: str) -> RetrievalPreset:
        """Get a preset by name."""
        if name not in cls._presets:
            available = ", ".join(cls._presets.keys())
            raise ValueError(
                f"Unknown preset: '{name}'. Available: {available}"
            )
        return cls._presets[name]

    @classmethod
    def list_presets(cls) -> list[str]:
        """List all registered preset names."""
        return list(cls._presets.keys())


# Define default presets
DEFAULT_PRESETS = [
    RetrievalPreset(
        name="dense_only",
        description="Dense retrieval only (semantic search)",
        retrieval_method="dense",
        top_k=10,
        similarity_threshold=0.7,
    ),
    RetrievalPreset(
        name="hybrid_rrf_v1",
        description="Hybrid retrieval with RRF fusion",
        retrieval_method="hybrid",
        top_k=10,
        dense_weight=0.7,
        sparse_weight=0.3,
        rrf_k=60,
    ),
    RetrievalPreset(
        name="hybrid_multi_query",
        description="Hybrid with multi-query expansion",
        retrieval_method="hybrid",
        top_k=50,
        dense_weight=0.7,
        sparse_weight=0.3,
        rrf_k=60,
        multi_query_enabled=True,
        multi_query_n=4,
        multi_query_include_original=True,
    ),
    RetrievalPreset(
        name="hybrid_rerank",
        description="Hybrid with reranking",
        retrieval_method="hybrid",
        top_k=50,
        dense_weight=0.7,
        sparse_weight=0.3,
        rrf_k=60,
        rerank_enabled=True,
        rerank_top_k=10,
    ),
    RetrievalPreset(
        name="full_pipeline",
        description="Full pipeline: hybrid + multi-query + rerank",
        retrieval_method="hybrid",
        top_k=100,
        dense_weight=0.7,
        sparse_weight=0.3,
        rrf_k=60,
        multi_query_enabled=True,
        multi_query_n=4,
        rerank_enabled=True,
        rerank_top_k=10,
    ),
]

# Register default presets
for preset in DEFAULT_PRESETS:
    PresetRegistry.register(preset)


def get_preset(name: str) -> RetrievalPreset:
    """Get a preset by name (convenience function)."""
    return PresetRegistry.get(name)
