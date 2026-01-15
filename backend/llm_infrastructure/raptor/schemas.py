"""RAPTOR data schemas and data classes.

This module defines the core data structures for Meta-guided Hierarchical RAG:
- RaptorNode: Tree node (leaf or summary)
- GroupEdge: Leaf-to-group membership edge
- Partition: Meta-group partition
- GroupStats: Group statistics for adaptive thresholds
- RoutingResult: Soft routing result
- ValidationResult: Summary validation result
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EdgeType(str, Enum):
    """Type of group edge."""

    PRIMARY = "primary"  # Main assignment from metadata
    SECONDARY = "secondary"  # Escape hatch assignment
    SOFT = "soft"  # Soft routing for missing metadata


class NodeLevel(int, Enum):
    """RAPTOR tree node levels."""

    LEAF = 0
    SUMMARY_L1 = 1
    SUMMARY_L2 = 2
    SUMMARY_L3 = 3


@dataclass
class RaptorNode:
    """RAPTOR tree node representing either a leaf chunk or a summary node.

    Attributes:
        node_id: Unique identifier for this node
        level: Tree level (0=leaf, 1,2,3=summary levels)
        content: Text content (original chunk or generated summary)
        embedding: Dense vector embedding
        children: List of child node IDs (empty for leaves)
        parent_id: Parent node ID (None for root nodes)
        partition_key: Meta-group partition key (e.g., "SUPRA_XP_sop")
        cluster_id: GMM cluster ID within the partition
        evidence_links: Sentence-to-source mapping for summary validation
        validation_score: Summary quality score from NLI validation
        metadata: Additional metadata (doc_id, page, device_name, etc.)
        created_at: Creation timestamp
    """

    node_id: str
    level: int
    content: str
    embedding: list[float]
    children: list[str] = field(default_factory=list)
    parent_id: str | None = None
    partition_key: str = ""
    cluster_id: str = ""
    evidence_links: dict[str, list[str]] | None = None
    validation_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        """Set defaults after init."""
        if not self.node_id:
            self.node_id = f"node_{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.level == NodeLevel.LEAF

    @property
    def is_summary(self) -> bool:
        """Check if this is a summary node."""
        return self.level > NodeLevel.LEAF

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for ES storage."""
        doc: dict[str, Any] = {
            "node_id": self.node_id,
            "raptor_level": self.level,
            "content": self.content,
            "embedding": self.embedding,
            "raptor_children_ids": self.children,
            "raptor_parent_id": self.parent_id,
            "partition_key": self.partition_key,
            "cluster_id": self.cluster_id,
            "is_summary_node": self.is_summary,
            "created_at": self.created_at,
        }
        if self.evidence_links:
            doc["evidence_links"] = self.evidence_links
        if self.validation_score is not None:
            doc["validation_score"] = self.validation_score
        if self.metadata:
            doc.update(self.metadata)
        return doc

    @classmethod
    def from_chunk(
        cls,
        chunk_id: str,
        content: str,
        embedding: list[float],
        partition_key: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "RaptorNode":
        """Create a leaf node from a chunk."""
        return cls(
            node_id=chunk_id,
            level=NodeLevel.LEAF,
            content=content,
            embedding=embedding,
            partition_key=partition_key,
            metadata=metadata or {},
        )

    @classmethod
    def create_summary(
        cls,
        content: str,
        embedding: list[float],
        children: list[str],
        level: int,
        partition_key: str = "",
        cluster_id: str = "",
        evidence_links: dict[str, list[str]] | None = None,
        validation_score: float | None = None,
    ) -> "RaptorNode":
        """Create a summary node from children."""
        return cls(
            node_id=f"summary_{level}_{uuid.uuid4().hex[:12]}",
            level=level,
            content=content,
            embedding=embedding,
            children=children,
            partition_key=partition_key,
            cluster_id=cluster_id,
            evidence_links=evidence_links,
            validation_score=validation_score,
        )


@dataclass
class GroupEdge:
    """Edge representing leaf-to-group membership.

    This replaces data duplication with edge-based multi-membership.
    A single leaf can belong to multiple groups with different weights.

    Attributes:
        leaf_id: Source leaf node ID
        group_id: Target group (partition) ID
        weight: Membership weight (0-1, sum across groups may exceed 1)
        edge_type: Type of assignment (primary, secondary, soft)
        score: Raw routing score before normalization
        created_at: Edge creation timestamp
    """

    leaf_id: str
    group_id: str
    weight: float
    edge_type: EdgeType = EdgeType.SOFT
    score: float = 0.0
    created_at: str = ""

    def __post_init__(self) -> None:
        """Set defaults after init."""
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if isinstance(self.edge_type, str):
            self.edge_type = EdgeType(self.edge_type)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for ES nested storage."""
        return {
            "leaf_id": self.leaf_id,
            "group_id": self.group_id,
            "weight": self.weight,
            "edge_type": self.edge_type.value,
            "score": self.score,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GroupEdge":
        """Create from dictionary."""
        return cls(
            leaf_id=data["leaf_id"],
            group_id=data["group_id"],
            weight=data["weight"],
            edge_type=EdgeType(data.get("edge_type", "soft")),
            score=data.get("score", 0.0),
            created_at=data.get("created_at", ""),
        )


@dataclass
class GroupStats:
    """Statistics for a meta-group, used for adaptive thresholds.

    Attributes:
        group_id: Group identifier
        centroid: Group centroid embedding
        mean_similarity: Mean similarity of members to centroid
        std_similarity: Standard deviation of similarities
        threshold: Adaptive novelty threshold (5th percentile)
        member_count: Number of leaf members
        updated_at: Last update timestamp
    """

    group_id: str
    centroid: list[float]
    mean_similarity: float = 0.0
    std_similarity: float = 1.0
    threshold: float = 0.3
    member_count: int = 0
    updated_at: str = ""

    def __post_init__(self) -> None:
        """Set defaults after init."""
        if not self.updated_at:
            self.updated_at = datetime.now(timezone.utc).isoformat()

    def compute_z_score(self, similarity: float) -> float:
        """Compute z-score for a similarity value."""
        if self.std_similarity < 1e-8:
            return 0.0
        return (similarity - self.mean_similarity) / self.std_similarity

    def is_outlier(self, similarity: float, k: float = 2.0) -> bool:
        """Check if similarity indicates an outlier (z-score < -k)."""
        return self.compute_z_score(similarity) < -k

    def is_novel(self, similarity: float) -> bool:
        """Check if similarity is below the adaptive threshold."""
        return similarity < self.threshold


@dataclass
class Partition:
    """Meta-group partition containing chunks with the same metadata.

    Attributes:
        key: Composite key (e.g., "SUPRA_XP_sop")
        device_name: Device/equipment name
        doc_type: Document type (sop, ts, guide, etc.)
        chunk_ids: List of leaf chunk IDs in this partition
        tree_root_ids: Root node IDs of the local RAPTOR tree (may have multiple)
        stats: Group statistics for adaptive routing
        local_raptor_built: Whether local RAPTOR tree is built
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """

    key: str
    device_name: str | None = None
    doc_type: str | None = None
    chunk_ids: list[str] = field(default_factory=list)
    tree_root_ids: list[str] = field(default_factory=list)
    stats: GroupStats | None = None
    local_raptor_built: bool = False
    created_at: str = ""
    updated_at: str = ""

    @property
    def tree_root_id(self) -> str | None:
        """Get first root ID for backward compatibility."""
        return self.tree_root_ids[0] if self.tree_root_ids else None

    @tree_root_id.setter
    def tree_root_id(self, value: str | None) -> None:
        """Set root ID (replaces all roots, for backward compatibility)."""
        if value is None:
            self.tree_root_ids = []
        else:
            self.tree_root_ids = [value]

    def __post_init__(self) -> None:
        """Set defaults after init."""
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    @classmethod
    def create_key(
        cls,
        device_name: str | None = None,
        doc_type: str | None = None,
    ) -> str:
        """Create composite partition key from metadata."""
        parts = []
        if device_name:
            parts.append(device_name.strip().replace(" ", "_"))
        if doc_type:
            parts.append(doc_type.strip().lower())
        return "_".join(parts) if parts else "none"

    @classmethod
    def from_metadata(
        cls,
        device_name: str | None = None,
        doc_type: str | None = None,
    ) -> "Partition":
        """Create partition from metadata."""
        key = cls.create_key(device_name, doc_type)
        return cls(
            key=key,
            device_name=device_name,
            doc_type=doc_type,
        )

    @property
    def is_none_pool(self) -> bool:
        """Check if this is the None pool (missing metadata)."""
        return self.key == "none" or (
            self.device_name is None and self.doc_type is None
        )

    def add_chunk(self, chunk_id: str) -> None:
        """Add a chunk to this partition."""
        if chunk_id not in self.chunk_ids:
            self.chunk_ids.append(chunk_id)
            self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "device_name": self.device_name,
            "doc_type": self.doc_type,
            "chunk_count": len(self.chunk_ids),
            "tree_root_ids": self.tree_root_ids,
            "tree_root_id": self.tree_root_id,  # Backward compat (first root)
            "local_raptor_built": self.local_raptor_built,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class RoutingResult:
    """Result of soft routing for a leaf chunk.

    Attributes:
        leaf_id: Leaf chunk ID
        edges: List of group edges (primary + secondary/soft)
        is_novel: Whether this leaf is considered novel
        novelty_score: Novelty detection score
        metadata: Original metadata from the chunk
    """

    leaf_id: str
    edges: list[GroupEdge]
    is_novel: bool = False
    novelty_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_group(self) -> str | None:
        """Get primary group ID."""
        for edge in self.edges:
            if edge.edge_type == EdgeType.PRIMARY:
                return edge.group_id
        return None

    @property
    def secondary_groups(self) -> list[str]:
        """Get secondary/soft group IDs."""
        return [
            e.group_id
            for e in self.edges
            if e.edge_type in (EdgeType.SECONDARY, EdgeType.SOFT)
        ]

    def get_top_k_edges(self, k: int) -> list[GroupEdge]:
        """Get top-k edges by weight."""
        sorted_edges = sorted(self.edges, key=lambda e: e.weight, reverse=True)
        return sorted_edges[:k]


@dataclass
class ValidationResult:
    """Result of summary validation using NLI.

    Attributes:
        score: Overall validation score (0-1)
        supported_ratio: Ratio of sentences supported by evidence
        unsupported_sentences: List of sentences not supported
        evidence_links: Mapping of sentences to supporting leaf IDs
        details: Additional validation details
    """

    score: float
    supported_ratio: float
    unsupported_sentences: list[str] = field(default_factory=list)
    evidence_links: dict[str, list[str]] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if summary passes validation threshold."""
        return self.supported_ratio >= 0.7

    def get_filtered_summary(self, original_summary: str) -> str:
        """Return summary with unsupported sentences removed."""
        if not self.unsupported_sentences:
            return original_summary
        lines = original_summary.split(". ")
        filtered = [
            line
            for line in lines
            if line.strip() not in self.unsupported_sentences
        ]
        return ". ".join(filtered)


@dataclass
class ClusterQualityMetrics:
    """Metrics for cluster quality diagnosis.

    Used for the repair loop to detect problematic clusters.

    Attributes:
        cluster_id: Cluster identifier
        cohesion: Mean similarity of members to centroid
        outlier_rate: Ratio of outlier members
        meta_conflict_rate: Ratio of conflicting metadata values
        step_discontinuity: Measure of step sequence disruption
        summary_support_rate: Ratio of summary sentences with evidence
        self_retrieval_rate: Ratio of children retrieved by summary query
    """

    cluster_id: str
    cohesion: float = 0.0
    outlier_rate: float = 0.0
    meta_conflict_rate: float = 0.0
    step_discontinuity: float = 0.0
    summary_support_rate: float = 1.0
    self_retrieval_rate: float = 1.0

    @property
    def is_suspicious(self) -> bool:
        """Check if this cluster needs review."""
        return (
            self.cohesion < 0.3
            or self.outlier_rate > 0.2
            or self.meta_conflict_rate > 0.3
            or self.summary_support_rate < 0.7
            or self.self_retrieval_rate < 0.5
        )

    @property
    def quality_score(self) -> float:
        """Compute overall quality score (0-1)."""
        return (
            self.cohesion * 0.3
            + (1 - self.outlier_rate) * 0.2
            + (1 - self.meta_conflict_rate) * 0.2
            + self.summary_support_rate * 0.15
            + self.self_retrieval_rate * 0.15
        )


@dataclass
class StructurePrior:
    """Structure-based prior scores for procedural documents.

    Captures adjacency, document scope, and step coherence.

    Attributes:
        adjacency_score: Score based on adjacent chunk assignments
        doc_scope_score: Score based on same document/section membership
        step_coherence_score: Score based on step number patterns
        total_score: Weighted sum of all scores
    """

    adjacency_score: float = 0.0
    doc_scope_score: float = 0.0
    step_coherence_score: float = 0.0
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "adjacency": 0.3,
            "doc_scope": 0.4,
            "step_coherence": 0.3,
        }
    )

    @property
    def total_score(self) -> float:
        """Compute weighted total score."""
        return (
            self.weights["adjacency"] * self.adjacency_score
            + self.weights["doc_scope"] * self.doc_scope_score
            + self.weights["step_coherence"] * self.step_coherence_score
        )


__all__ = [
    "EdgeType",
    "NodeLevel",
    "RaptorNode",
    "GroupEdge",
    "GroupStats",
    "Partition",
    "RoutingResult",
    "ValidationResult",
    "ClusterQualityMetrics",
    "StructurePrior",
]
