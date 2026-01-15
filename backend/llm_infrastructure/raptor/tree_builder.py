"""Local RAPTOR tree construction for hierarchical RAG.

This module implements the RaptorTreeBuilder class that:
1. Builds local RAPTOR trees within each meta-group partition
2. Recursively clusters and summarizes chunks
3. Validates summaries using NLI
4. Builds cross-group similarity links

Reference: RAPTOR paper (arXiv:2401.18059)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from backend.llm_infrastructure.raptor.clustering import (
    ClusteringResult,
    RaptorClusterer,
)
from backend.llm_infrastructure.raptor.schemas import (
    NodeLevel,
    Partition,
    RaptorNode,
    ValidationResult,
)
from backend.llm_infrastructure.raptor.summary_validator import (
    SummaryValidator,
    ValidatorConfig,
)

if TYPE_CHECKING:
    from backend.llm_infrastructure.embedding.base import BaseEmbedder

logger = logging.getLogger(__name__)


@dataclass
class TreeBuilderConfig:
    """Configuration for RAPTOR tree building.

    Attributes:
        max_levels: Maximum tree depth (default: 3)
        min_cluster_size: Minimum nodes per cluster
        max_summary_tokens: Maximum tokens in summary
        validate_summaries: Whether to validate summaries with NLI
        validation_threshold: Minimum validation score to accept
        reduction_dim: UMAP reduction dimension
        cross_group_threshold: Similarity threshold for cross-group links
    """

    max_levels: int = 3
    min_cluster_size: int = 3
    max_summary_tokens: int = 500
    validate_summaries: bool = True
    validation_threshold: float = 0.5
    reduction_dim: int = 10
    cross_group_threshold: float = 0.6


@dataclass
class TreeBuildResult:
    """Result of tree building for a partition.

    Attributes:
        partition_key: Partition identifier
        nodes: All nodes in the tree (leaves + summaries)
        root_ids: Root node IDs (top level)
        levels: Nodes organized by level
        stats: Build statistics
    """

    partition_key: str
    nodes: list[RaptorNode]
    root_ids: list[str]
    levels: dict[int, list[RaptorNode]] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def total_nodes(self) -> int:
        """Total number of nodes."""
        return len(self.nodes)

    @property
    def leaf_count(self) -> int:
        """Number of leaf nodes."""
        return len(self.levels.get(0, []))

    @property
    def summary_count(self) -> int:
        """Number of summary nodes."""
        return self.total_nodes - self.leaf_count


@dataclass
class CrossGroupLink:
    """Similarity link between summary nodes across groups.

    Enables cross-partition retrieval when queries span groups.

    Attributes:
        source_id: Source summary node ID
        target_id: Target summary node ID
        source_partition: Source partition key
        target_partition: Target partition key
        similarity: Cosine similarity score
    """

    source_id: str
    target_id: str
    source_partition: str
    target_partition: str
    similarity: float


SummarizerFn = Callable[[list[str]], str]


class RaptorTreeBuilder:
    """Builds local RAPTOR trees within partitions.

    Implements the recursive clustering and summarization algorithm:
    1. Cluster leaf nodes using GMM
    2. Generate summary for each cluster
    3. Embed summaries and repeat until max_levels

    Args:
        clusterer: GMM clusterer for node grouping
        embedder: Embedder for summary embeddings
        summarizer: Function to generate summaries from texts
        validator: Summary validator (optional)
        config: Builder configuration
    """

    def __init__(
        self,
        clusterer: RaptorClusterer | None = None,
        embedder: "BaseEmbedder | None" = None,
        summarizer: SummarizerFn | None = None,
        validator: SummaryValidator | None = None,
        config: TreeBuilderConfig | None = None,
    ) -> None:
        self.config = config or TreeBuilderConfig()
        self.clusterer = clusterer or RaptorClusterer(
            reduction_dim=self.config.reduction_dim,
            min_cluster_size=self.config.min_cluster_size,
        )
        self.embedder = embedder
        self.summarizer = summarizer or self._default_summarizer
        self.validator = validator or SummaryValidator(ValidatorConfig())

    def _default_summarizer(self, texts: list[str]) -> str:
        """Default summarizer: concatenate and truncate."""
        combined = " ".join(texts)
        # Simple truncation
        max_chars = self.config.max_summary_tokens * 4  # Rough char estimate
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "..."
        return combined

    def build_local_tree(
        self,
        partition: Partition,
        leaf_nodes: list[RaptorNode],
    ) -> TreeBuildResult:
        """Build RAPTOR tree for a partition.

        Args:
            partition: Target partition
            leaf_nodes: Leaf nodes (chunks) in the partition

        Returns:
            TreeBuildResult with all nodes
        """
        if not leaf_nodes:
            return TreeBuildResult(
                partition_key=partition.key,
                nodes=[],
                root_ids=[],
                stats={"error": "No leaf nodes"},
            )

        logger.info(
            f"Building RAPTOR tree for partition {partition.key} "
            f"with {len(leaf_nodes)} leaves"
        )

        all_nodes: list[RaptorNode] = list(leaf_nodes)
        levels: dict[int, list[RaptorNode]] = {0: list(leaf_nodes)}
        current_level_nodes = leaf_nodes

        for level in range(1, self.config.max_levels + 1):
            if len(current_level_nodes) < self.config.min_cluster_size:
                logger.info(
                    f"Level {level}: Too few nodes ({len(current_level_nodes)}), "
                    "stopping tree construction"
                )
                break

            # Cluster current level
            summary_nodes = self._build_level(
                current_level_nodes,
                level,
                partition.key,
            )

            if not summary_nodes:
                logger.info(f"Level {level}: No clusters formed, stopping")
                break

            all_nodes.extend(summary_nodes)
            levels[level] = summary_nodes
            current_level_nodes = summary_nodes

            logger.info(
                f"Level {level}: Created {len(summary_nodes)} summary nodes"
            )

        # Set root IDs
        top_level = max(levels.keys())
        root_ids = [n.node_id for n in levels[top_level]]

        # Update parent references
        self._link_parents(levels)

        # Build stats
        stats = {
            "total_nodes": len(all_nodes),
            "leaf_count": len(levels.get(0, [])),
            "levels": len(levels),
            "nodes_per_level": {k: len(v) for k, v in levels.items()},
        }

        result = TreeBuildResult(
            partition_key=partition.key,
            nodes=all_nodes,
            root_ids=root_ids,
            levels=levels,
            stats=stats,
        )

        # Mark partition as built
        partition.local_raptor_built = True
        partition.tree_root_ids = root_ids  # Store all root IDs

        return result

    def _build_level(
        self,
        nodes: list[RaptorNode],
        level: int,
        partition_key: str,
    ) -> list[RaptorNode]:
        """Build one level of the RAPTOR tree.

        Args:
            nodes: Nodes from previous level
            level: Target level number
            partition_key: Partition identifier

        Returns:
            List of summary nodes for this level
        """
        # Get embeddings
        embeddings = [n.embedding for n in nodes]
        node_ids = [n.node_id for n in nodes]

        # Cluster
        clustering_result = self.clusterer.cluster_chunks(
            node_ids,
            embeddings,
            partition_key=f"{partition_key}_L{level}",
        )

        # Get clusters
        clusters = self.clusterer.get_clusters_by_id(clustering_result)

        # Create summary for each cluster
        summary_nodes: list[RaptorNode] = []
        node_map = {n.node_id: n for n in nodes}

        for cluster_id, member_ids in clusters.items():
            if len(member_ids) < 2:
                continue

            # Get member nodes
            members = [node_map[mid] for mid in member_ids if mid in node_map]
            if len(members) < 2:
                continue

            # Generate summary
            member_texts = [m.content for m in members]
            summary_text = self.summarizer(member_texts)

            # Validate summary
            validation_result: ValidationResult | None = None
            if self.config.validate_summaries:
                validation_result = self.validator.validate_summary(
                    summary_text,
                    member_texts,
                    member_ids,
                )

                # Filter if below threshold
                if validation_result.score < self.config.validation_threshold:
                    logger.debug(
                        f"Summary validation failed for cluster {cluster_id}: "
                        f"score={validation_result.score:.2f}"
                    )
                    # Use filtered summary
                    summary_text = self.validator.filter_unsupported(
                        summary_text, member_texts
                    )

            # Embed summary
            if self.embedder:
                summary_embedding = self.embedder.embed(summary_text)
            else:
                # Fallback: average of member embeddings
                summary_embedding = np.mean(
                    [m.embedding for m in members], axis=0
                ).tolist()

            # Create summary node
            summary_node = RaptorNode.create_summary(
                content=summary_text,
                embedding=summary_embedding,
                children=member_ids,
                level=level,
                partition_key=partition_key,
                cluster_id=cluster_id,
                evidence_links=validation_result.evidence_links if validation_result else None,
                validation_score=validation_result.score if validation_result else None,
            )

            summary_nodes.append(summary_node)

        return summary_nodes

    def _link_parents(self, levels: dict[int, list[RaptorNode]]) -> None:
        """Set parent references in the tree.

        Args:
            levels: Nodes organized by level
        """
        # Build child -> parent mapping
        child_to_parent: dict[str, str] = {}

        for level_nodes in levels.values():
            for node in level_nodes:
                for child_id in node.children:
                    child_to_parent[child_id] = node.node_id

        # Update parent references
        for level_nodes in levels.values():
            for node in level_nodes:
                if node.node_id in child_to_parent:
                    node.parent_id = child_to_parent[node.node_id]

    def build_cross_group_links(
        self,
        tree_results: list[TreeBuildResult],
        level: int = 1,
    ) -> list[CrossGroupLink]:
        """Build similarity links between summary nodes across partitions.

        Enables retrieval to traverse partition boundaries when needed.

        Args:
            tree_results: Tree build results for all partitions
            level: Summary level to link (default: 1 = first summary level)

        Returns:
            List of cross-group links
        """
        links: list[CrossGroupLink] = []

        # Collect summary nodes at target level from all partitions
        all_summaries: list[tuple[str, RaptorNode]] = []

        for result in tree_results:
            level_nodes = result.levels.get(level, [])
            for node in level_nodes:
                all_summaries.append((result.partition_key, node))

        if len(all_summaries) < 2:
            return links

        # Compare all pairs across partitions
        for i, (partition_i, node_i) in enumerate(all_summaries):
            for j, (partition_j, node_j) in enumerate(all_summaries[i + 1 :], i + 1):
                # Skip same partition
                if partition_i == partition_j:
                    continue

                # Compute similarity
                similarity = self._cosine_similarity(
                    node_i.embedding, node_j.embedding
                )

                if similarity >= self.config.cross_group_threshold:
                    links.append(
                        CrossGroupLink(
                            source_id=node_i.node_id,
                            target_id=node_j.node_id,
                            source_partition=partition_i,
                            target_partition=partition_j,
                            similarity=similarity,
                        )
                    )

        logger.info(
            f"Built {len(links)} cross-group links from "
            f"{len(all_summaries)} summary nodes"
        )

        return links

    def _cosine_similarity(
        self,
        vec1: list[float],
        vec2: list[float],
    ) -> float:
        """Compute cosine similarity."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))


class IncrementalTreeBuilder(RaptorTreeBuilder):
    """Tree builder that supports incremental updates.

    Extends RaptorTreeBuilder to handle:
    - Adding new chunks to existing tree
    - Recomputing affected branches only
    - Maintaining tree consistency

    Args:
        **kwargs: Arguments passed to RaptorTreeBuilder
    """

    def add_to_tree(
        self,
        existing_result: TreeBuildResult,
        new_nodes: list[RaptorNode],
    ) -> TreeBuildResult:
        """Add new nodes to existing tree.

        Recomputes affected clusters and summaries.

        Args:
            existing_result: Current tree state
            new_nodes: New leaf nodes to add

        Returns:
            Updated TreeBuildResult
        """
        if not new_nodes:
            return existing_result

        # Add new leaves
        existing_leaves = existing_result.levels.get(0, [])
        all_leaves = existing_leaves + new_nodes

        # Find affected clusters (simplified: rebuild from level 1)
        # In practice, would only recompute clusters containing new nodes

        # Create dummy partition
        partition = Partition(key=existing_result.partition_key)

        # Rebuild tree
        return self.build_local_tree(partition, all_leaves)

    def remove_from_tree(
        self,
        existing_result: TreeBuildResult,
        node_ids_to_remove: list[str],
    ) -> TreeBuildResult:
        """Remove nodes from existing tree.

        Args:
            existing_result: Current tree state
            node_ids_to_remove: Node IDs to remove

        Returns:
            Updated TreeBuildResult
        """
        remove_set = set(node_ids_to_remove)

        # Filter leaves
        remaining_leaves = [
            n for n in existing_result.levels.get(0, [])
            if n.node_id not in remove_set
        ]

        if not remaining_leaves:
            return TreeBuildResult(
                partition_key=existing_result.partition_key,
                nodes=[],
                root_ids=[],
            )

        # Rebuild tree
        partition = Partition(key=existing_result.partition_key)
        return self.build_local_tree(partition, remaining_leaves)


class EnsembleTreeBuilder:
    """Builds multiple RAPTOR trees for ensemble retrieval.

    Creates trees with different random seeds/configurations
    and combines results using RRF.

    Args:
        n_trees: Number of trees in ensemble
        base_config: Base configuration
    """

    def __init__(
        self,
        n_trees: int = 3,
        base_config: TreeBuilderConfig | None = None,
        embedder: "BaseEmbedder | None" = None,
        summarizer: SummarizerFn | None = None,
    ) -> None:
        self.n_trees = n_trees
        self.base_config = base_config or TreeBuilderConfig()
        self.embedder = embedder
        self.summarizer = summarizer

        # Create builders with different random seeds
        self.builders: list[RaptorTreeBuilder] = []
        for i in range(n_trees):
            clusterer = RaptorClusterer(
                reduction_dim=self.base_config.reduction_dim,
                min_cluster_size=self.base_config.min_cluster_size,
                random_state=42 + i,  # Different seed per tree
            )
            builder = RaptorTreeBuilder(
                clusterer=clusterer,
                embedder=embedder,
                summarizer=summarizer,
                config=base_config,
            )
            self.builders.append(builder)

    def build_ensemble(
        self,
        partition: Partition,
        leaf_nodes: list[RaptorNode],
    ) -> list[TreeBuildResult]:
        """Build ensemble of trees.

        Args:
            partition: Target partition
            leaf_nodes: Leaf nodes

        Returns:
            List of TreeBuildResults (one per tree)
        """
        results = []
        for i, builder in enumerate(self.builders):
            # Create partition copy with unique key for each tree
            tree_partition = Partition(
                key=f"{partition.key}_tree{i}",
                device_name=partition.device_name,
                doc_type=partition.doc_type,
                chunk_ids=partition.chunk_ids,
            )
            result = builder.build_local_tree(tree_partition, leaf_nodes)
            results.append(result)

        return results


__all__ = [
    "RaptorTreeBuilder",
    "IncrementalTreeBuilder",
    "EnsembleTreeBuilder",
    "TreeBuilderConfig",
    "TreeBuildResult",
    "CrossGroupLink",
]
