"""GMM clustering with UMAP dimensionality reduction for RAPTOR.

This module implements the RaptorClusterer class that:
1. Reduces embedding dimensions using UMAP (768 -> 10)
2. Applies GMM soft clustering
3. Uses BIC to automatically determine optimal cluster count
4. Provides soft cluster assignments for tree building

Reference: RAPTOR paper (arXiv:2401.18059)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sklearn.mixture import GaussianMixture
    from umap import UMAP

logger = logging.getLogger(__name__)


@dataclass
class ClusterAssignment:
    """Soft cluster assignment for a chunk.

    Attributes:
        chunk_id: Chunk identifier
        cluster_id: Assigned cluster ID
        probability: Assignment probability (soft membership)
        all_probabilities: Probabilities for all clusters
    """

    chunk_id: str
    cluster_id: str
    probability: float
    all_probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def is_confident(self) -> bool:
        """Check if assignment is confident (probability > 0.7)."""
        return self.probability > 0.7

    def get_top_k_clusters(self, k: int = 3) -> list[tuple[str, float]]:
        """Get top-k cluster assignments by probability."""
        sorted_probs = sorted(
            self.all_probabilities.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_probs[:k]


@dataclass
class ClusteringResult:
    """Result of clustering operation.

    Attributes:
        assignments: List of cluster assignments per chunk
        cluster_ids: List of all cluster IDs
        cluster_centers: Cluster centroids in reduced space
        bic_score: BIC score for model selection
        n_clusters: Number of clusters
        reduced_embeddings: UMAP-reduced embeddings
    """

    assignments: list[ClusterAssignment]
    cluster_ids: list[str]
    cluster_centers: NDArray[np.floating[Any]]
    bic_score: float
    n_clusters: int
    reduced_embeddings: NDArray[np.floating[Any]] | None = None


class RaptorClusterer:
    """GMM-based clusterer for RAPTOR tree building.

    Uses UMAP for dimensionality reduction and GMM for soft clustering.
    Automatically determines optimal cluster count using BIC.

    Args:
        reduction_dim: Target dimension for UMAP reduction (default: 10)
        min_cluster_size: Minimum samples per cluster (default: 5)
        max_clusters: Maximum number of clusters to try (default: 50)
        random_state: Random seed for reproducibility
        umap_n_neighbors: UMAP n_neighbors parameter
        umap_min_dist: UMAP min_dist parameter
    """

    def __init__(
        self,
        reduction_dim: int = 10,
        min_cluster_size: int = 5,
        max_clusters: int = 50,
        random_state: int = 42,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
    ) -> None:
        self.reduction_dim = reduction_dim
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist

        self._umap_model: UMAP | None = None
        self._gmm_model: GaussianMixture | None = None

    def cluster_chunks(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]] | NDArray[np.floating[Any]],
        partition_key: str = "",
    ) -> ClusteringResult:
        """Cluster chunks using GMM with UMAP reduction.

        Args:
            chunk_ids: List of chunk identifiers
            embeddings: Embedding vectors (N x D)
            partition_key: Partition key for cluster ID naming

        Returns:
            ClusteringResult with soft assignments
        """
        embeddings_array = np.array(embeddings)
        n_samples = embeddings_array.shape[0]

        logger.info(
            f"Clustering {n_samples} chunks (dim={embeddings_array.shape[1]})"
        )

        # Skip if too few samples
        if n_samples < self.min_cluster_size:
            logger.warning(
                f"Too few samples ({n_samples}) for clustering, "
                f"returning single cluster"
            )
            return self._single_cluster_result(chunk_ids, embeddings_array, partition_key)

        # UMAP dimensionality reduction
        reduced = self._reduce_dimensions(embeddings_array)

        # Find optimal number of clusters using BIC
        optimal_k = self._find_optimal_k(reduced)
        logger.info(f"Optimal cluster count: {optimal_k}")

        # Fit GMM with optimal k
        gmm = self._fit_gmm(reduced, optimal_k)
        self._gmm_model = gmm

        # Get soft assignments
        probabilities = gmm.predict_proba(reduced)
        labels = gmm.predict(reduced)

        # Generate cluster IDs
        cluster_ids = [
            f"{partition_key}_c{i}" if partition_key else f"cluster_{i}"
            for i in range(optimal_k)
        ]

        # Build assignments
        assignments = []
        for idx, chunk_id in enumerate(chunk_ids):
            cluster_idx = labels[idx]
            prob = probabilities[idx, cluster_idx]

            all_probs = {
                cluster_ids[i]: float(probabilities[idx, i])
                for i in range(optimal_k)
            }

            assignment = ClusterAssignment(
                chunk_id=chunk_id,
                cluster_id=cluster_ids[cluster_idx],
                probability=float(prob),
                all_probabilities=all_probs,
            )
            assignments.append(assignment)

        return ClusteringResult(
            assignments=assignments,
            cluster_ids=cluster_ids,
            cluster_centers=gmm.means_,
            bic_score=gmm.bic(reduced),
            n_clusters=optimal_k,
            reduced_embeddings=reduced,
        )

    def _reduce_dimensions(
        self,
        embeddings: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Reduce embedding dimensions using UMAP.

        Args:
            embeddings: High-dimensional embeddings (N x D)

        Returns:
            Reduced embeddings (N x reduction_dim)
        """
        try:
            from umap import UMAP
        except ImportError:
            logger.warning("UMAP not available, using PCA fallback")
            return self._pca_fallback(embeddings)

        n_samples = embeddings.shape[0]
        n_neighbors = min(self.umap_n_neighbors, n_samples - 1)

        if n_neighbors < 2:
            logger.warning("Too few samples for UMAP, returning original")
            return embeddings

        target_dim = min(self.reduction_dim, embeddings.shape[1], n_samples - 1)

        self._umap_model = UMAP(
            n_components=target_dim,
            n_neighbors=n_neighbors,
            min_dist=self.umap_min_dist,
            random_state=self.random_state,
            metric="cosine",
        )

        reduced = self._umap_model.fit_transform(embeddings)
        logger.info(f"Reduced dimensions: {embeddings.shape[1]} -> {reduced.shape[1]}")
        return reduced

    def _pca_fallback(
        self,
        embeddings: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """PCA fallback when UMAP is not available.

        Args:
            embeddings: High-dimensional embeddings

        Returns:
            PCA-reduced embeddings
        """
        from sklearn.decomposition import PCA

        target_dim = min(self.reduction_dim, embeddings.shape[1], embeddings.shape[0])
        pca = PCA(n_components=target_dim, random_state=self.random_state)
        return pca.fit_transform(embeddings)

    def _find_optimal_k(
        self,
        embeddings: NDArray[np.floating[Any]],
    ) -> int:
        """Find optimal cluster count using BIC.

        Tests k from 1 to min(max_clusters, n_samples / min_cluster_size)
        and selects k with lowest BIC.

        Args:
            embeddings: Reduced embeddings

        Returns:
            Optimal number of clusters
        """
        from sklearn.mixture import GaussianMixture

        n_samples = embeddings.shape[0]
        max_k = min(
            self.max_clusters,
            n_samples // self.min_cluster_size,
            n_samples,
        )
        max_k = max(1, max_k)

        best_k = 1
        best_bic = float("inf")

        # Test different values of k
        k_values = list(range(1, max_k + 1))

        for k in k_values:
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    random_state=self.random_state,
                    n_init=3,
                    max_iter=300,
                )
                gmm.fit(embeddings)
                bic = gmm.bic(embeddings)

                if bic < best_bic:
                    best_bic = bic
                    best_k = k
            except Exception as e:
                logger.warning(f"GMM failed for k={k}: {e}")
                continue

        return best_k

    def _fit_gmm(
        self,
        embeddings: NDArray[np.floating[Any]],
        n_clusters: int,
    ) -> "GaussianMixture":
        """Fit GMM model with specified number of clusters.

        Args:
            embeddings: Reduced embeddings
            n_clusters: Number of clusters

        Returns:
            Fitted GaussianMixture model
        """
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=self.random_state,
            n_init=5,
            max_iter=500,
        )
        gmm.fit(embeddings)
        return gmm

    def _single_cluster_result(
        self,
        chunk_ids: list[str],
        embeddings: NDArray[np.floating[Any]],
        partition_key: str,
    ) -> ClusteringResult:
        """Create result with all chunks in a single cluster.

        Args:
            chunk_ids: Chunk identifiers
            embeddings: Embedding vectors
            partition_key: Partition key for naming

        Returns:
            ClusteringResult with single cluster
        """
        cluster_id = f"{partition_key}_c0" if partition_key else f"cluster_{uuid.uuid4().hex[:8]}"
        center = np.mean(embeddings, axis=0, keepdims=True)

        assignments = [
            ClusterAssignment(
                chunk_id=cid,
                cluster_id=cluster_id,
                probability=1.0,
                all_probabilities={cluster_id: 1.0},
            )
            for cid in chunk_ids
        ]

        return ClusteringResult(
            assignments=assignments,
            cluster_ids=[cluster_id],
            cluster_centers=center,
            bic_score=0.0,
            n_clusters=1,
            reduced_embeddings=embeddings,
        )

    def get_clusters_by_id(
        self,
        result: ClusteringResult,
    ) -> dict[str, list[str]]:
        """Group chunk IDs by cluster.

        Args:
            result: Clustering result

        Returns:
            Dictionary mapping cluster ID to list of chunk IDs
        """
        clusters: dict[str, list[str]] = {cid: [] for cid in result.cluster_ids}

        for assignment in result.assignments:
            clusters[assignment.cluster_id].append(assignment.chunk_id)

        return clusters

    def get_soft_memberships(
        self,
        result: ClusteringResult,
        threshold: float = 0.1,
    ) -> dict[str, list[tuple[str, float]]]:
        """Get soft cluster memberships above threshold.

        Args:
            result: Clustering result
            threshold: Minimum probability threshold

        Returns:
            Dictionary mapping chunk ID to list of (cluster_id, probability)
        """
        memberships: dict[str, list[tuple[str, float]]] = {}

        for assignment in result.assignments:
            significant = [
                (cid, prob)
                for cid, prob in assignment.all_probabilities.items()
                if prob >= threshold
            ]
            significant.sort(key=lambda x: x[1], reverse=True)
            memberships[assignment.chunk_id] = significant

        return memberships


class ConstrainedClusterer(RaptorClusterer):
    """Clusterer with must-link and cannot-link constraints.

    Extends RaptorClusterer to support constrained clustering for
    procedural documents where certain chunks must/cannot be grouped.

    Args:
        must_links: List of (chunk_id_1, chunk_id_2) that must be in same cluster
        cannot_links: List of (chunk_id_1, chunk_id_2) that cannot be in same cluster
        **kwargs: Arguments passed to RaptorClusterer
    """

    def __init__(
        self,
        must_links: list[tuple[str, str]] | None = None,
        cannot_links: list[tuple[str, str]] | None = None,
        constraint_weight: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.must_links = must_links or []
        self.cannot_links = cannot_links or []
        self.constraint_weight = constraint_weight

    def cluster_chunks(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]] | NDArray[np.floating[Any]],
        partition_key: str = "",
    ) -> ClusteringResult:
        """Cluster with constraints.

        First performs standard GMM clustering, then adjusts assignments
        to satisfy constraints.

        Args:
            chunk_ids: Chunk identifiers
            embeddings: Embedding vectors
            partition_key: Partition key

        Returns:
            Constrained clustering result
        """
        # Get initial clustering
        result = super().cluster_chunks(chunk_ids, embeddings, partition_key)

        if not self.must_links and not self.cannot_links:
            return result

        # Apply constraint adjustments
        result = self._apply_constraints(result, chunk_ids)

        return result

    def _apply_constraints(
        self,
        result: ClusteringResult,
        chunk_ids: list[str],
    ) -> ClusteringResult:
        """Apply must-link and cannot-link constraints.

        Args:
            result: Initial clustering result
            chunk_ids: Chunk identifiers

        Returns:
            Adjusted clustering result
        """
        chunk_to_idx = {cid: idx for idx, cid in enumerate(chunk_ids)}
        assignments = list(result.assignments)

        # Apply must-links: merge clusters
        for id1, id2 in self.must_links:
            if id1 not in chunk_to_idx or id2 not in chunk_to_idx:
                continue

            idx1 = chunk_to_idx[id1]
            idx2 = chunk_to_idx[id2]

            cluster1 = assignments[idx1].cluster_id
            cluster2 = assignments[idx2].cluster_id

            if cluster1 != cluster2:
                # Move all chunks from cluster2 to cluster1
                for i, assign in enumerate(assignments):
                    if assign.cluster_id == cluster2:
                        assignments[i] = ClusterAssignment(
                            chunk_id=assign.chunk_id,
                            cluster_id=cluster1,
                            probability=assign.probability * self.constraint_weight,
                            all_probabilities=assign.all_probabilities,
                        )

        # Apply cannot-links: reassign conflicting chunks
        for id1, id2 in self.cannot_links:
            if id1 not in chunk_to_idx or id2 not in chunk_to_idx:
                continue

            idx1 = chunk_to_idx[id1]
            idx2 = chunk_to_idx[id2]

            if assignments[idx1].cluster_id == assignments[idx2].cluster_id:
                # Move the chunk with lower probability to second-best cluster
                if assignments[idx1].probability < assignments[idx2].probability:
                    target_idx = idx1
                else:
                    target_idx = idx2

                assign = assignments[target_idx]
                top_k = assign.get_top_k_clusters(k=2)

                if len(top_k) > 1:
                    new_cluster = top_k[1][0]
                    new_prob = top_k[1][1]
                    assignments[target_idx] = ClusterAssignment(
                        chunk_id=assign.chunk_id,
                        cluster_id=new_cluster,
                        probability=new_prob,
                        all_probabilities=assign.all_probabilities,
                    )

        return ClusteringResult(
            assignments=assignments,
            cluster_ids=result.cluster_ids,
            cluster_centers=result.cluster_centers,
            bic_score=result.bic_score,
            n_clusters=result.n_clusters,
            reduced_embeddings=result.reduced_embeddings,
        )


__all__ = [
    "RaptorClusterer",
    "ConstrainedClusterer",
    "ClusterAssignment",
    "ClusteringResult",
]
