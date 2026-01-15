"""Metadata-based partitioning for RAPTOR hierarchical RAG.

This module implements the MetadataPartitioner class that:
1. Partitions chunks by metadata (device_name, doc_type)
2. Computes partition centroids for soft routing
3. Builds hierarchical meta-tree structure
4. Handles None-pool for missing metadata

The partition key format is: "{device_name}_{doc_type}" (e.g., "SUPRA_XP_sop")
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.llm_infrastructure.raptor.schemas import (
    GroupStats,
    Partition,
)

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

    from backend.llm_infrastructure.embedding.base import BaseEmbedder

logger = logging.getLogger(__name__)


@dataclass
class MetaTreeNode:
    """Node in the hierarchical metadata tree.

    Structure: Level 0 (root) -> Level 1 (device) -> Level 2 (doc_type) -> Partitions

    Attributes:
        key: Node identifier
        level: Tree level (0=root, 1=device, 2=doc_type)
        children: Child node keys
        partition_keys: Associated partition keys at this level
        metadata: Additional metadata
    """

    key: str
    level: int = 0
    children: list[str] = field(default_factory=list)
    partition_keys: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaTree:
    """Hierarchical metadata tree for partition navigation.

    Enables hierarchical routing: device -> doc_type -> partition

    Attributes:
        root: Root node
        nodes: All nodes indexed by key
        device_to_partitions: Device name to partition keys mapping
        doc_type_to_partitions: Doc type to partition keys mapping
    """

    root: MetaTreeNode
    nodes: dict[str, MetaTreeNode] = field(default_factory=dict)
    device_to_partitions: dict[str, list[str]] = field(default_factory=dict)
    doc_type_to_partitions: dict[str, list[str]] = field(default_factory=dict)

    def get_partitions_by_device(self, device_name: str) -> list[str]:
        """Get all partition keys for a device."""
        normalized = device_name.strip().replace(" ", "_")
        return self.device_to_partitions.get(normalized, [])

    def get_partitions_by_doc_type(self, doc_type: str) -> list[str]:
        """Get all partition keys for a doc type."""
        normalized = doc_type.strip().lower()
        return self.doc_type_to_partitions.get(normalized, [])


class MetadataPartitioner:
    """Partitions chunks by metadata for hierarchical RAPTOR indexing.

    This partitioner:
    1. Groups chunks by (device_name, doc_type) into partitions
    2. Computes centroid embeddings for each partition
    3. Builds a hierarchical meta-tree for navigation
    4. Maintains the None-pool for missing metadata

    Args:
        es_client: Elasticsearch client
        index_name: Index name for chunk retrieval
        embedder: Embedder for computing centroids (optional, uses mean if None)
    """

    def __init__(
        self,
        es_client: "Elasticsearch",
        index_name: str,
        embedder: "BaseEmbedder | None" = None,
    ) -> None:
        self.es_client = es_client
        self.index_name = index_name
        self.embedder = embedder
        self._partitions: dict[str, Partition] = {}
        self._meta_tree: MetaTree | None = None

    @property
    def partitions(self) -> dict[str, Partition]:
        """Get all partitions."""
        return self._partitions

    @property
    def meta_tree(self) -> MetaTree | None:
        """Get the hierarchical meta-tree."""
        return self._meta_tree

    def partition_by_metadata(
        self,
        size: int = 10000,
        include_embeddings: bool = True,
    ) -> list[Partition]:
        """Partition all chunks by metadata.

        Args:
            size: Max chunks to process per scroll
            include_embeddings: Whether to load embeddings for centroid computation

        Returns:
            List of Partition objects
        """
        logger.info(f"Partitioning chunks from index: {self.index_name}")

        # Aggregate by device_name and doc_type
        agg_query = {
            "size": 0,
            "aggs": {
                "by_device": {
                    "terms": {"field": "device_name", "size": 1000, "missing": ""},
                    "aggs": {
                        "by_doc_type": {
                            "terms": {
                                "field": "doc_type",
                                "size": 100,
                                "missing": "",
                            },
                            "aggs": {
                                "chunk_ids": {
                                    "terms": {"field": "chunk_id", "size": 50000},
                                },
                            },
                        },
                    },
                },
            },
        }

        response = self.es_client.search(index=self.index_name, body=agg_query)

        # Build partitions from aggregation
        partitions: dict[str, Partition] = {}
        device_buckets = response["aggregations"]["by_device"]["buckets"]

        for device_bucket in device_buckets:
            device_name = device_bucket["key"] or None
            doc_type_buckets = device_bucket["by_doc_type"]["buckets"]

            for doc_type_bucket in doc_type_buckets:
                doc_type = doc_type_bucket["key"] or None
                chunk_buckets = doc_type_bucket["chunk_ids"]["buckets"]
                chunk_ids = [b["key"] for b in chunk_buckets]

                partition = Partition.from_metadata(device_name, doc_type)
                partition.chunk_ids = chunk_ids
                partitions[partition.key] = partition

        self._partitions = partitions
        logger.info(f"Created {len(partitions)} partitions")

        # Compute centroids if embeddings available
        if include_embeddings:
            self.compute_partition_centroids()

        # Build meta-tree
        self._meta_tree = self.build_hierarchical_meta_tree()

        return list(partitions.values())

    def compute_partition_centroids(self) -> dict[str, GroupStats]:
        """Compute centroid and statistics for each partition.

        Returns:
            Dictionary of partition key to GroupStats
        """
        logger.info("Computing partition centroids...")

        stats_map: dict[str, GroupStats] = {}

        for key, partition in self._partitions.items():
            if not partition.chunk_ids:
                continue

            # Fetch embeddings for this partition
            embeddings = self._fetch_embeddings(partition.chunk_ids)
            if not embeddings:
                continue

            # Compute centroid and statistics
            embeddings_array = np.array(embeddings)
            centroid = np.mean(embeddings_array, axis=0)

            # Compute similarity statistics
            similarities = []
            for emb in embeddings_array:
                sim = float(np.dot(emb, centroid) / (
                    np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-8
                ))
                similarities.append(sim)

            mean_sim = float(np.mean(similarities))
            std_sim = float(np.std(similarities)) if len(similarities) > 1 else 1.0
            threshold = float(np.percentile(similarities, 5)) if similarities else 0.3

            stats = GroupStats(
                group_id=key,
                centroid=centroid.tolist(),
                mean_similarity=mean_sim,
                std_similarity=std_sim,
                threshold=threshold,
                member_count=len(partition.chunk_ids),
            )

            partition.stats = stats
            stats_map[key] = stats

        logger.info(f"Computed centroids for {len(stats_map)} partitions")
        return stats_map

    def _fetch_embeddings(
        self,
        chunk_ids: list[str],
        batch_size: int = 500,
    ) -> list[list[float]]:
        """Fetch embeddings for given chunk IDs.

        Args:
            chunk_ids: List of chunk IDs
            batch_size: Batch size for mget

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i in range(0, len(chunk_ids), batch_size):
            batch_ids = chunk_ids[i : i + batch_size]
            docs = self.es_client.mget(
                index=self.index_name,
                body={"ids": batch_ids},
                _source=["embedding"],
            )

            for doc in docs.get("docs", []):
                if doc.get("found") and "embedding" in doc.get("_source", {}):
                    embeddings.append(doc["_source"]["embedding"])

        return embeddings

    def build_hierarchical_meta_tree(self) -> MetaTree:
        """Build hierarchical meta-tree from partitions.

        Structure:
            root
            ├── device_1
            │   ├── sop -> partition_key
            │   └── ts -> partition_key
            ├── device_2
            │   └── guide -> partition_key
            └── none (missing metadata)

        Returns:
            MetaTree object
        """
        root = MetaTreeNode(key="root", level=0)
        nodes: dict[str, MetaTreeNode] = {"root": root}
        device_to_partitions: dict[str, list[str]] = defaultdict(list)
        doc_type_to_partitions: dict[str, list[str]] = defaultdict(list)

        for key, partition in self._partitions.items():
            # Handle device level
            device_key = partition.device_name or "none"
            device_key_normalized = device_key.strip().replace(" ", "_")

            if device_key_normalized not in nodes:
                device_node = MetaTreeNode(
                    key=device_key_normalized,
                    level=1,
                    metadata={"device_name": partition.device_name},
                )
                nodes[device_key_normalized] = device_node
                root.children.append(device_key_normalized)

            device_node = nodes[device_key_normalized]
            device_to_partitions[device_key_normalized].append(key)

            # Handle doc_type level
            doc_type_key = partition.doc_type or "none"
            doc_type_key_normalized = doc_type_key.strip().lower()
            composite_key = f"{device_key_normalized}_{doc_type_key_normalized}"

            if composite_key not in nodes:
                doc_type_node = MetaTreeNode(
                    key=composite_key,
                    level=2,
                    metadata={
                        "device_name": partition.device_name,
                        "doc_type": partition.doc_type,
                    },
                )
                nodes[composite_key] = doc_type_node
                device_node.children.append(composite_key)

            doc_type_node = nodes[composite_key]
            doc_type_node.partition_keys.append(key)
            doc_type_to_partitions[doc_type_key_normalized].append(key)

        meta_tree = MetaTree(
            root=root,
            nodes=nodes,
            device_to_partitions=dict(device_to_partitions),
            doc_type_to_partitions=dict(doc_type_to_partitions),
        )

        logger.info(
            f"Built meta-tree with {len(nodes)} nodes, "
            f"{len(device_to_partitions)} devices, "
            f"{len(doc_type_to_partitions)} doc types"
        )

        return meta_tree

    def get_partition_stats(self) -> dict[str, int]:
        """Get statistics about partitions.

        Returns:
            Dictionary with partition stats
        """
        if not self._partitions:
            return {"total_partitions": 0, "total_chunks": 0}

        chunk_counts = [len(p.chunk_ids) for p in self._partitions.values()]

        return {
            "total_partitions": len(self._partitions),
            "total_chunks": sum(chunk_counts),
            "max_chunks": max(chunk_counts) if chunk_counts else 0,
            "min_chunks": min(chunk_counts) if chunk_counts else 0,
            "avg_chunks": sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0,
            "none_pool_chunks": len(
                self._partitions.get("none", Partition(key="none")).chunk_ids
            ),
        }

    def get_partition(self, key: str) -> Partition | None:
        """Get partition by key.

        Args:
            key: Partition key

        Returns:
            Partition or None if not found
        """
        return self._partitions.get(key)

    def get_partitions_for_metadata(
        self,
        device_name: str | None = None,
        doc_type: str | None = None,
    ) -> list[Partition]:
        """Get partitions matching the given metadata.

        Args:
            device_name: Device name filter
            doc_type: Document type filter

        Returns:
            List of matching partitions
        """
        results = []

        for partition in self._partitions.values():
            if device_name is not None and partition.device_name != device_name:
                continue
            if doc_type is not None and partition.doc_type != doc_type:
                continue
            results.append(partition)

        return results

    def add_chunk_to_partition(
        self,
        chunk_id: str,
        device_name: str | None = None,
        doc_type: str | None = None,
    ) -> Partition:
        """Add a chunk to the appropriate partition.

        Creates partition if it doesn't exist.

        Args:
            chunk_id: Chunk ID to add
            device_name: Device name metadata
            doc_type: Document type metadata

        Returns:
            Target partition
        """
        key = Partition.create_key(device_name, doc_type)

        if key not in self._partitions:
            partition = Partition.from_metadata(device_name, doc_type)
            self._partitions[key] = partition
        else:
            partition = self._partitions[key]

        partition.add_chunk(chunk_id)
        return partition

    def get_candidate_partitions(
        self,
        device_name: str | None = None,
        doc_type: str | None = None,
        include_none_pool: bool = True,
        max_candidates: int = 10,
    ) -> list[Partition]:
        """Get candidate partitions for routing.

        Prioritizes exact metadata matches, then partial matches.

        Args:
            device_name: Device name to match
            doc_type: Document type to match
            include_none_pool: Whether to include None pool
            max_candidates: Maximum candidates to return

        Returns:
            List of candidate partitions sorted by relevance
        """
        candidates: list[tuple[int, Partition]] = []

        for partition in self._partitions.values():
            score = 0

            # Exact device match
            if device_name and partition.device_name == device_name:
                score += 2
            # Exact doc_type match
            if doc_type and partition.doc_type == doc_type:
                score += 1

            # Skip None pool unless explicitly requested
            if partition.is_none_pool and not include_none_pool:
                continue

            candidates.append((score, partition))

        # Sort by score (descending), then by member count
        candidates.sort(key=lambda x: (-x[0], -len(x[1].chunk_ids)))

        return [p for _, p in candidates[:max_candidates]]


__all__ = [
    "MetadataPartitioner",
    "MetaTreeNode",
    "MetaTree",
]
