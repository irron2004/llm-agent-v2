"""Soft membership routing with novelty detection for RAPTOR.

This module implements the core probabilistic routing logic:
1. Meta-prior routing with soft membership
2. Novelty detection using adaptive thresholds (DP mixture inspired)
3. Soft escape hatch for metadata misclassification
4. Structure-based priors for procedural documents

Key formulation:
    p(z | x, m_obs) ∝ p(x | z) · p(m_obs | z) · p(z)

    Log-linear approximation:
    score(g | leaf_i) = β · sim(e_i, c_g)
                      + Σ_j α_j · match(meta_j)
                      + γ · adjacency_prior
                      + δ · doc_scope_prior
                      + κ · step_coherence
                      + b_g
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from backend.llm_infrastructure.raptor.schemas import (
    EdgeType,
    GroupEdge,
    GroupStats,
    Partition,
    RoutingResult,
    StructurePrior,
)

if TYPE_CHECKING:
    from backend.llm_infrastructure.elasticsearch.document import EsChunkDocument

logger = logging.getLogger(__name__)


@dataclass
class SoftRouterConfig:
    """Configuration for soft routing.

    Attributes:
        beta: Semantic similarity weight
        alpha_device: Device name match weight
        alpha_doc_type: Doc type match weight
        gamma: Adjacency prior weight
        delta: Document scope prior weight
        kappa: Step coherence weight
        novelty_threshold: Base novelty detection threshold
        escape_threshold_k: Z-score threshold for soft escape (negative)
        top_k: Number of soft groups for missing metadata
        global_fallback_weight: Weight for global fallback
        min_weight_threshold: Minimum edge weight to keep
    """

    beta: float = 1.0
    alpha_device: float = 0.8
    alpha_doc_type: float = 0.5
    gamma: float = 0.3
    delta: float = 0.4
    kappa: float = 0.3
    novelty_threshold: float = 0.3
    escape_threshold_k: float = 2.0
    top_k: int = 3
    global_fallback_weight: float = 0.1
    min_weight_threshold: float = 0.05


@dataclass
class ChunkContext:
    """Context information for routing a chunk.

    Attributes:
        chunk_id: Chunk identifier
        embedding: Dense vector embedding
        device_name: Device name metadata (may be None)
        doc_type: Document type metadata (may be None)
        doc_id: Document ID for scope checking
        page: Page number
        adjacent_chunks: IDs of adjacent chunks (i-1, i+1)
        step_info: Step/procedure info extracted from content
        metadata: Additional metadata
    """

    chunk_id: str
    embedding: list[float]
    device_name: str | None = None
    doc_type: str | None = None
    doc_id: str = ""
    page: int = 0
    adjacent_chunks: list[str] = field(default_factory=list)
    step_info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_es_doc(
        cls,
        doc: "EsChunkDocument",
        adjacent_chunks: list[str] | None = None,
    ) -> "ChunkContext":
        """Create from ES document."""
        return cls(
            chunk_id=doc.chunk_id,
            embedding=doc.embedding,
            device_name=doc.device_name or None,
            doc_type=doc.doc_type or None,
            doc_id=doc.doc_id,
            page=doc.page,
            adjacent_chunks=adjacent_chunks or [],
            metadata={
                "chapter": doc.chapter,
                "content": doc.content,
            },
        )

    @property
    def has_metadata(self) -> bool:
        """Check if metadata is available."""
        return self.device_name is not None or self.doc_type is not None


class SoftRouter:
    """Soft membership router with novelty detection.

    Implements probabilistic routing:
    - Meta-aware routing when metadata exists
    - Soft routing to top-k groups when metadata missing
    - Escape hatch when semantic mismatch detected
    - Novelty detection for new group creation

    Args:
        config: Router configuration
        partitions: List of available partitions
    """

    def __init__(
        self,
        config: SoftRouterConfig | None = None,
        partitions: list[Partition] | None = None,
    ) -> None:
        self.config = config or SoftRouterConfig()
        self._partitions: dict[str, Partition] = {}
        self._step_pattern = re.compile(
            r"(?:step|단계|순서|절차)\s*[:#]?\s*(\d+)",
            re.IGNORECASE,
        )

        if partitions:
            for p in partitions:
                self._partitions[p.key] = p

    def set_partitions(self, partitions: list[Partition]) -> None:
        """Set available partitions for routing."""
        self._partitions = {p.key: p for p in partitions}

    def route_leaf(
        self,
        context: ChunkContext,
        adjacent_assignments: dict[str, str] | None = None,
    ) -> RoutingResult:
        """Route a leaf chunk to group(s).

        Routing logic:
        1. If metadata exists: primary group from meta, check escape
        2. If metadata missing: soft route to top-k groups
        3. Check novelty for potential new group

        Args:
            context: Chunk context with embedding and metadata
            adjacent_assignments: Map of adjacent chunk IDs to their group assignments

        Returns:
            RoutingResult with group edges
        """
        edges: list[GroupEdge] = []
        adjacent_assignments = adjacent_assignments or {}

        # Compute scores for all groups
        group_scores = self._compute_all_scores(context, adjacent_assignments)

        # Check novelty
        is_novel, novelty_score = self._detect_novelty(context, group_scores)

        if context.has_metadata:
            # Meta-aware routing
            primary_key = Partition.create_key(context.device_name, context.doc_type)

            if primary_key in self._partitions:
                primary_partition = self._partitions[primary_key]
                primary_score = group_scores.get(primary_key, 0.0)

                # Add primary edge
                edges.append(
                    GroupEdge(
                        leaf_id=context.chunk_id,
                        group_id=primary_key,
                        weight=self._normalize_weight(primary_score, group_scores),
                        edge_type=EdgeType.PRIMARY,
                        score=primary_score,
                    )
                )

                # Check for escape hatch
                escape_edges = self._check_escape_hatch(
                    context, primary_partition, group_scores
                )
                edges.extend(escape_edges)
            else:
                # Metadata exists but partition doesn't - treat as soft
                edges = self._soft_route(context, group_scores)
        else:
            # No metadata - soft route to top-k
            edges = self._soft_route(context, group_scores)

        # Filter edges below threshold
        edges = [e for e in edges if e.weight >= self.config.min_weight_threshold]

        # Ensure at least one edge
        if not edges and group_scores:
            best_group = max(group_scores, key=group_scores.get)
            edges.append(
                GroupEdge(
                    leaf_id=context.chunk_id,
                    group_id=best_group,
                    weight=1.0,
                    edge_type=EdgeType.SOFT,
                    score=group_scores[best_group],
                )
            )

        return RoutingResult(
            leaf_id=context.chunk_id,
            edges=edges,
            is_novel=is_novel,
            novelty_score=novelty_score,
            metadata={
                "device_name": context.device_name,
                "doc_type": context.doc_type,
            },
        )

    def _compute_all_scores(
        self,
        context: ChunkContext,
        adjacent_assignments: dict[str, str],
    ) -> dict[str, float]:
        """Compute routing scores for all groups.

        score(g) = β·sim(e, c_g) + α_d·match_device + α_t·match_type
                 + γ·adjacency + δ·doc_scope + κ·step_coherence

        Args:
            context: Chunk context
            adjacent_assignments: Adjacent chunk -> group mapping

        Returns:
            Dictionary of group_id -> score
        """
        scores: dict[str, float] = {}
        embedding = np.array(context.embedding)

        for key, partition in self._partitions.items():
            if partition.stats is None or partition.stats.centroid is None:
                continue

            score = 0.0

            # Semantic similarity
            centroid = np.array(partition.stats.centroid)
            similarity = self._cosine_similarity(embedding, centroid)
            score += self.config.beta * similarity

            # Metadata match
            if context.device_name and partition.device_name:
                if context.device_name == partition.device_name:
                    score += self.config.alpha_device
            if context.doc_type and partition.doc_type:
                if context.doc_type == partition.doc_type:
                    score += self.config.alpha_doc_type

            # Structure priors
            structure_prior = self._compute_structure_prior(
                context, partition, adjacent_assignments
            )
            score += (
                self.config.gamma * structure_prior.adjacency_score
                + self.config.delta * structure_prior.doc_scope_score
                + self.config.kappa * structure_prior.step_coherence_score
            )

            scores[key] = score

        return scores

    def _compute_structure_prior(
        self,
        context: ChunkContext,
        partition: Partition,
        adjacent_assignments: dict[str, str],
    ) -> StructurePrior:
        """Compute structure-based prior scores.

        Args:
            context: Chunk context
            partition: Target partition
            adjacent_assignments: Adjacent chunk assignments

        Returns:
            StructurePrior with component scores
        """
        adjacency_score = 0.0
        doc_scope_score = 0.0
        step_coherence_score = 0.0

        # Adjacency prior: adjacent chunks assigned to this group
        if adjacent_assignments:
            adjacent_in_group = sum(
                1
                for adj_id in context.adjacent_chunks
                if adjacent_assignments.get(adj_id) == partition.key
            )
            if context.adjacent_chunks:
                adjacency_score = adjacent_in_group / len(context.adjacent_chunks)

        # Doc scope prior: same document/chapter
        if context.doc_id:
            # Check if doc_id is associated with this partition
            # (simplified - in practice would check chunk metadata)
            if any(
                cid.startswith(context.doc_id)
                for cid in partition.chunk_ids[:100]  # Sample
            ):
                doc_scope_score = 0.5

            # Chapter match adds more
            chapter = context.metadata.get("chapter", "")
            if chapter and partition.key.lower() in chapter.lower():
                doc_scope_score += 0.5

        # Step coherence: procedural pattern matching
        content = context.metadata.get("content", "")
        step_info = self._extract_step_info(content)

        if step_info.get("has_step_pattern"):
            # Check if partition likely contains procedural content
            if partition.doc_type in ("sop", "procedure", "manual", "guide"):
                step_coherence_score = 0.7
            elif "절차" in content or "step" in content.lower():
                step_coherence_score = 0.5

        return StructurePrior(
            adjacency_score=adjacency_score,
            doc_scope_score=doc_scope_score,
            step_coherence_score=step_coherence_score,
        )

    def _extract_step_info(self, content: str) -> dict[str, Any]:
        """Extract step/procedure information from content.

        Args:
            content: Text content

        Returns:
            Step info dictionary
        """
        matches = self._step_pattern.findall(content)
        return {
            "has_step_pattern": bool(matches),
            "step_numbers": [int(m) for m in matches] if matches else [],
        }

    def _detect_novelty(
        self,
        context: ChunkContext,
        group_scores: dict[str, float],
    ) -> tuple[bool, float]:
        """Detect if chunk is novel (doesn't fit existing groups).

        Uses adaptive threshold per group (DP mixture inspired):
        - τ_g = 5th percentile of within-group similarities
        - Chunk is novel if similarity < τ_g for all groups

        Args:
            context: Chunk context
            group_scores: Precomputed group scores

        Returns:
            (is_novel, novelty_score)
        """
        if not self._partitions or not group_scores:
            return True, 0.0

        embedding = np.array(context.embedding)

        # Track best fit across all groups
        best_margin = float("-inf")  # How much above threshold
        best_similarity = 0.0

        for key, partition in self._partitions.items():
            if partition.stats is None or partition.stats.centroid is None:
                continue

            centroid = np.array(partition.stats.centroid)
            similarity = self._cosine_similarity(embedding, centroid)

            # Use 5th percentile threshold (τ_g) from GroupStats
            # This is computed during partition centroid computation
            threshold = partition.stats.threshold
            if threshold <= 0:
                # Fallback: use configurable base threshold
                threshold = self.config.novelty_threshold

            # Compute margin above threshold
            margin = similarity - threshold

            if margin > best_margin:
                best_margin = margin
                best_similarity = similarity

        # Novel if best similarity is below all group thresholds
        is_novel = best_margin < 0

        # Novelty score: how far below threshold (larger = more novel)
        novelty_score = -best_margin if is_novel else 0.0

        return is_novel, novelty_score

    def _check_escape_hatch(
        self,
        context: ChunkContext,
        primary_partition: Partition,
        group_scores: dict[str, float],
    ) -> list[GroupEdge]:
        """Check if chunk should escape to secondary groups.

        Escape condition: semantic similarity to primary group is
        significantly lower than expected (z-score < -k).

        Args:
            context: Chunk context
            primary_partition: Primary assigned partition
            group_scores: All group scores

        Returns:
            List of secondary escape edges
        """
        edges: list[GroupEdge] = []

        if primary_partition.stats is None:
            return edges

        # Compute similarity to primary group
        embedding = np.array(context.embedding)
        centroid = np.array(primary_partition.stats.centroid)
        similarity = self._cosine_similarity(embedding, centroid)

        # Check z-score
        z_score = primary_partition.stats.compute_z_score(similarity)

        if z_score < -self.config.escape_threshold_k:
            # Semantic mismatch - add secondary groups
            logger.debug(
                f"Escape triggered for {context.chunk_id}: z-score={z_score:.2f}"
            )

            # Get top-k alternative groups
            sorted_groups = sorted(
                group_scores.items(), key=lambda x: x[1], reverse=True
            )

            for group_id, score in sorted_groups[: self.config.top_k]:
                if group_id == primary_partition.key:
                    continue

                weight = self._normalize_weight(score, group_scores)
                if weight >= self.config.min_weight_threshold:
                    edges.append(
                        GroupEdge(
                            leaf_id=context.chunk_id,
                            group_id=group_id,
                            weight=weight,
                            edge_type=EdgeType.SECONDARY,
                            score=score,
                        )
                    )

        return edges

    def _soft_route(
        self,
        context: ChunkContext,
        group_scores: dict[str, float],
    ) -> list[GroupEdge]:
        """Route to top-k groups with soft membership.

        Used when metadata is missing.

        Args:
            context: Chunk context
            group_scores: All group scores

        Returns:
            List of soft edges
        """
        edges: list[GroupEdge] = []

        if not group_scores:
            return edges

        # Sort by score
        sorted_groups = sorted(
            group_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Take top-k
        for group_id, score in sorted_groups[: self.config.top_k]:
            weight = self._normalize_weight(score, group_scores)
            if weight >= self.config.min_weight_threshold:
                edges.append(
                    GroupEdge(
                        leaf_id=context.chunk_id,
                        group_id=group_id,
                        weight=weight,
                        edge_type=EdgeType.SOFT,
                        score=score,
                    )
                )

        return edges

    def _normalize_weight(
        self,
        score: float,
        all_scores: dict[str, float],
    ) -> float:
        """Normalize score to weight using softmax.

        Args:
            score: Score to normalize
            all_scores: All group scores for normalization

        Returns:
            Normalized weight (0-1)
        """
        if not all_scores:
            return 1.0

        scores_array = np.array(list(all_scores.values()))

        # Softmax normalization
        exp_scores = np.exp(scores_array - np.max(scores_array))
        softmax_scores = exp_scores / (np.sum(exp_scores) + 1e-8)

        # Find position of this score
        idx = list(all_scores.values()).index(score)
        return float(softmax_scores[idx])

    def _cosine_similarity(
        self,
        vec1: NDArray[np.floating[Any]],
        vec2: NDArray[np.floating[Any]],
    ) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def batch_route(
        self,
        contexts: list[ChunkContext],
    ) -> list[RoutingResult]:
        """Route multiple chunks in batch.

        Maintains adjacency information across chunks.

        Args:
            contexts: List of chunk contexts (ordered by position)

        Returns:
            List of routing results
        """
        results: list[RoutingResult] = []
        adjacent_assignments: dict[str, str] = {}

        for i, context in enumerate(contexts):
            # Build adjacent chunk info
            if i > 0:
                context.adjacent_chunks.append(contexts[i - 1].chunk_id)
            if i < len(contexts) - 1:
                context.adjacent_chunks.append(contexts[i + 1].chunk_id)

            # Route
            result = self.route_leaf(context, adjacent_assignments)
            results.append(result)

            # Update adjacency map for next iteration
            if result.primary_group:
                adjacent_assignments[context.chunk_id] = result.primary_group
            elif result.edges:
                adjacent_assignments[context.chunk_id] = result.edges[0].group_id

        return results


class AdaptiveSoftRouter(SoftRouter):
    """Soft router with learned/adaptive weights.

    Extends SoftRouter to support:
    - Weight learning from supervision signals
    - Adaptive threshold tuning
    - Online weight updates

    Args:
        config: Router configuration
        partitions: Available partitions
        learning_rate: Weight update learning rate
    """

    def __init__(
        self,
        config: SoftRouterConfig | None = None,
        partitions: list[Partition] | None = None,
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__(config, partitions)
        self.learning_rate = learning_rate
        self._weight_history: list[dict[str, float]] = []

    def update_weights(
        self,
        positive_pairs: list[tuple[str, str]],
        negative_pairs: list[tuple[str, str]],
        embeddings: dict[str, list[float]],
    ) -> None:
        """Update routing weights from supervision.

        Uses pairwise learning:
        - Positive pairs: chunks that should be in same group
        - Negative pairs: chunks that should be in different groups

        Args:
            positive_pairs: (chunk_id_1, chunk_id_2) same group
            negative_pairs: (chunk_id_1, chunk_id_2) different groups
            embeddings: Chunk ID to embedding mapping
        """
        # Simplified gradient update
        # In practice, would use proper optimization

        for id1, id2 in positive_pairs:
            if id1 in embeddings and id2 in embeddings:
                sim = self._cosine_similarity(
                    np.array(embeddings[id1]), np.array(embeddings[id2])
                )
                # Increase beta if similar items are far
                if sim < 0.5:
                    self.config.beta += self.learning_rate

        for id1, id2 in negative_pairs:
            if id1 in embeddings and id2 in embeddings:
                sim = self._cosine_similarity(
                    np.array(embeddings[id1]), np.array(embeddings[id2])
                )
                # Decrease beta if dissimilar items are close
                if sim > 0.7:
                    self.config.beta -= self.learning_rate

        # Clamp weights
        self.config.beta = max(0.1, min(5.0, self.config.beta))

        self._weight_history.append({
            "beta": self.config.beta,
            "alpha_device": self.config.alpha_device,
            "alpha_doc_type": self.config.alpha_doc_type,
        })

    def get_weight_history(self) -> list[dict[str, float]]:
        """Get history of weight updates."""
        return self._weight_history


__all__ = [
    "SoftRouter",
    "AdaptiveSoftRouter",
    "SoftRouterConfig",
    "ChunkContext",
]
