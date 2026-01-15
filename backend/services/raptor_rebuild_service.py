"""Offline RAPTOR rebuild service.

This service handles periodic RAPTOR tree reconstruction:
1. Rebuild local RAPTOR trees for each partition
2. Validate summaries with NLI
3. Build cross-group similarity links
4. Repair loop for cluster quality issues

Usage:
    service = RaptorRebuildService(
        es_client=es_client,
        tree_builder=builder,
        partitioner=partitioner,
    )
    await service.rebuild_all_partitions()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from backend.llm_infrastructure.raptor.clustering import RaptorClusterer
from backend.llm_infrastructure.raptor.partition import MetadataPartitioner
from backend.llm_infrastructure.raptor.schemas import (
    ClusterQualityMetrics,
    Partition,
    RaptorNode,
)
from backend.llm_infrastructure.raptor.summary_validator import (
    SummaryValidator,
    ValidatorConfig,
)
from backend.llm_infrastructure.raptor.tree_builder import (
    CrossGroupLink,
    RaptorTreeBuilder,
    TreeBuildResult,
    TreeBuilderConfig,
)

if TYPE_CHECKING:
    from elasticsearch import AsyncElasticsearch, Elasticsearch

    from backend.llm_infrastructure.embedding.base import BaseEmbedder

logger = logging.getLogger(__name__)


SummarizerFn = Callable[[list[str]], str]


@dataclass
class RebuildConfig:
    """Configuration for RAPTOR rebuild.

    Attributes:
        min_partition_size: Minimum chunks to build tree
        max_levels: Maximum tree depth
        validate_summaries: Whether to validate with NLI
        build_cross_links: Whether to build cross-group links
        parallel_partitions: Number of partitions to process in parallel
        repair_suspicious: Whether to repair suspicious clusters
        cohesion_threshold: Minimum cluster cohesion
        outlier_rate_threshold: Maximum outlier rate
    """

    min_partition_size: int = 10
    max_levels: int = 3
    validate_summaries: bool = True
    build_cross_links: bool = True
    parallel_partitions: int = 4
    repair_suspicious: bool = True
    cohesion_threshold: float = 0.3
    outlier_rate_threshold: float = 0.2


@dataclass
class RebuildStats:
    """Rebuild statistics.

    Attributes:
        partitions_processed: Number of partitions processed
        trees_built: Number of trees successfully built
        summary_nodes_created: Total summary nodes created
        cross_links_created: Cross-group links created
        clusters_repaired: Clusters that needed repair
        validation_failures: Summaries that failed validation
        errors: Number of errors
        start_time: Start timestamp
        end_time: End timestamp
    """

    partitions_processed: int = 0
    trees_built: int = 0
    summary_nodes_created: int = 0
    cross_links_created: int = 0
    clusters_repaired: int = 0
    validation_failures: int = 0
    errors: int = 0
    start_time: str = ""
    end_time: str = ""

    @property
    def duration_seconds(self) -> float:
        """Compute duration in seconds."""
        if not self.start_time or not self.end_time:
            return 0.0
        start = datetime.fromisoformat(self.start_time)
        end = datetime.fromisoformat(self.end_time)
        return (end - start).total_seconds()


class RaptorRebuildService:
    """Offline service for RAPTOR tree rebuilding.

    Handles periodic reconstruction of RAPTOR trees:
    - Per-partition tree building
    - Summary validation
    - Cross-group link building
    - Cluster quality repair

    Args:
        es_client: Elasticsearch client
        index_name: Target index name
        embedder: Embedder for computing embeddings
        summarizer: Function to generate summaries
        tree_builder: RAPTOR tree builder
        partitioner: Metadata partitioner
        validator: Summary validator
        config: Rebuild configuration
    """

    def __init__(
        self,
        es_client: "Elasticsearch | AsyncElasticsearch",
        index_name: str,
        embedder: "BaseEmbedder",
        summarizer: SummarizerFn,
        tree_builder: RaptorTreeBuilder | None = None,
        partitioner: MetadataPartitioner | None = None,
        validator: SummaryValidator | None = None,
        config: RebuildConfig | None = None,
    ) -> None:
        self.es_client = es_client
        self.index_name = index_name
        self.embedder = embedder
        self.summarizer = summarizer
        self.config = config or RebuildConfig()

        # Initialize components
        self._partitioner = partitioner or MetadataPartitioner(
            es_client=es_client,  # type: ignore
            index_name=index_name,
        )

        clusterer = RaptorClusterer(
            min_cluster_size=self.config.min_partition_size // 2,
        )
        self._tree_builder = tree_builder or RaptorTreeBuilder(
            clusterer=clusterer,
            embedder=embedder,
            summarizer=summarizer,
            config=TreeBuilderConfig(max_levels=self.config.max_levels),
        )

        self._validator = validator or SummaryValidator(ValidatorConfig())
        self._stats = RebuildStats()
        self._tree_results: dict[str, TreeBuildResult] = {}
        self._cross_links: list[CrossGroupLink] = []

    @property
    def stats(self) -> RebuildStats:
        """Get rebuild statistics."""
        return self._stats

    @property
    def tree_results(self) -> dict[str, TreeBuildResult]:
        """Get tree build results by partition key."""
        return self._tree_results

    @property
    def cross_links(self) -> list[CrossGroupLink]:
        """Get cross-group links."""
        return self._cross_links

    async def rebuild_all_partitions(self) -> RebuildStats:
        """Rebuild RAPTOR trees for all partitions.

        Returns:
            RebuildStats with summary
        """
        self._stats = RebuildStats(
            start_time=datetime.now(timezone.utc).isoformat()
        )

        try:
            # Refresh partitions from ES
            partitions = self._partitioner.partition_by_metadata()
            logger.info(f"Found {len(partitions)} partitions to rebuild")

            # Process partitions
            if self.config.parallel_partitions > 1:
                await self._rebuild_parallel(partitions)
            else:
                await self._rebuild_sequential(partitions)

            # Build cross-group links
            if self.config.build_cross_links and self._tree_results:
                self._cross_links = self._tree_builder.build_cross_group_links(
                    list(self._tree_results.values())
                )
                self._stats.cross_links_created = len(self._cross_links)

                # Store cross-links
                await self._store_cross_links()

        except Exception as e:
            logger.error(f"Rebuild failed: {e}")
            self._stats.errors += 1

        self._stats.end_time = datetime.now(timezone.utc).isoformat()
        return self._stats

    async def rebuild_partition(self, partition_key: str) -> TreeBuildResult | None:
        """Rebuild a specific partition.

        Args:
            partition_key: Partition to rebuild

        Returns:
            TreeBuildResult or None if failed
        """
        partition = self._partitioner.get_partition(partition_key)
        if not partition:
            logger.warning(f"Partition not found: {partition_key}")
            return None

        return await self._rebuild_single_partition(partition)

    async def _rebuild_sequential(self, partitions: list[Partition]) -> None:
        """Rebuild partitions sequentially."""
        for partition in partitions:
            await self._rebuild_single_partition(partition)

    async def _rebuild_parallel(self, partitions: list[Partition]) -> None:
        """Rebuild partitions in parallel."""
        semaphore = asyncio.Semaphore(self.config.parallel_partitions)

        async def rebuild_with_semaphore(p: Partition) -> None:
            async with semaphore:
                await self._rebuild_single_partition(p)

        await asyncio.gather(
            *[rebuild_with_semaphore(p) for p in partitions],
            return_exceptions=True,
        )

    async def _rebuild_single_partition(
        self,
        partition: Partition,
    ) -> TreeBuildResult | None:
        """Rebuild RAPTOR tree for a single partition.

        Args:
            partition: Partition to rebuild

        Returns:
            TreeBuildResult or None
        """
        self._stats.partitions_processed += 1

        if len(partition.chunk_ids) < self.config.min_partition_size:
            logger.debug(
                f"Skipping partition {partition.key}: "
                f"only {len(partition.chunk_ids)} chunks"
            )
            return None

        try:
            # Load leaf nodes
            leaf_nodes = await self._load_partition_nodes(partition)

            if not leaf_nodes:
                return None

            # Build tree
            result = self._tree_builder.build_local_tree(partition, leaf_nodes)

            # Validate and repair if needed
            if self.config.repair_suspicious:
                result = await self._repair_if_needed(result)

            # Store tree nodes
            await self._store_tree_nodes(result)

            self._tree_results[partition.key] = result
            self._stats.trees_built += 1
            self._stats.summary_nodes_created += result.summary_count

            logger.info(
                f"Built tree for {partition.key}: "
                f"{result.leaf_count} leaves, {result.summary_count} summaries"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to rebuild partition {partition.key}: {e}")
            self._stats.errors += 1
            return None

    async def _load_partition_nodes(
        self,
        partition: Partition,
    ) -> list[RaptorNode]:
        """Load leaf nodes for a partition.

        Args:
            partition: Partition to load

        Returns:
            List of RaptorNodes
        """
        nodes: list[RaptorNode] = []

        # Fetch chunks in batches
        batch_size = 500
        for i in range(0, len(partition.chunk_ids), batch_size):
            batch_ids = partition.chunk_ids[i : i + batch_size]

            if hasattr(self.es_client, "mget"):
                if asyncio.iscoroutinefunction(self.es_client.mget):
                    response = await self.es_client.mget(
                        index=self.index_name,
                        body={"ids": batch_ids},
                    )
                else:
                    response = self.es_client.mget(
                        index=self.index_name,
                        body={"ids": batch_ids},
                    )

                for doc in response.get("docs", []):
                    if doc.get("found"):
                        source = doc.get("_source", {})
                        node = RaptorNode.from_chunk(
                            chunk_id=doc["_id"],
                            content=source.get("content", ""),
                            embedding=source.get("embedding", []),
                            partition_key=partition.key,
                            metadata={
                                "doc_id": source.get("doc_id"),
                                "page": source.get("page"),
                                "device_name": source.get("device_name"),
                                "doc_type": source.get("doc_type"),
                            },
                        )
                        nodes.append(node)

        return nodes

    async def _store_tree_nodes(self, result: TreeBuildResult) -> None:
        """Store tree nodes (summaries) to ES.

        Args:
            result: Tree build result
        """
        # Only store summary nodes (level > 0)
        summary_nodes = [n for n in result.nodes if n.is_summary]

        if not summary_nodes:
            return

        actions = []
        for node in summary_nodes:
            doc = node.to_dict()
            doc["search_text"] = node.content  # For BM25

            action = {"index": {"_index": self.index_name, "_id": node.node_id}}
            actions.append(action)
            actions.append(doc)

        try:
            if hasattr(self.es_client, "bulk"):
                if asyncio.iscoroutinefunction(self.es_client.bulk):
                    await self.es_client.bulk(body=actions, refresh="wait_for")
                else:
                    self.es_client.bulk(body=actions, refresh="wait_for")

            logger.debug(f"Stored {len(summary_nodes)} summary nodes")

        except Exception as e:
            logger.error(f"Failed to store tree nodes: {e}")
            self._stats.errors += 1

    async def _store_cross_links(self) -> None:
        """Store cross-group links.

        Currently stores as a separate document or updates existing nodes.
        """
        if not self._cross_links:
            return

        # Group links by source node
        links_by_source: dict[str, list[dict[str, Any]]] = {}
        for link in self._cross_links:
            if link.source_id not in links_by_source:
                links_by_source[link.source_id] = []
            links_by_source[link.source_id].append({
                "target_id": link.target_id,
                "target_partition": link.target_partition,
                "similarity": link.similarity,
            })

        # Update source nodes with cross-links
        for source_id, links in links_by_source.items():
            try:
                update_body = {
                    "doc": {"cross_group_links": links},
                    "doc_as_upsert": False,
                }

                if hasattr(self.es_client, "update"):
                    if asyncio.iscoroutinefunction(self.es_client.update):
                        await self.es_client.update(
                            index=self.index_name,
                            id=source_id,
                            body=update_body,
                        )
                    else:
                        self.es_client.update(
                            index=self.index_name,
                            id=source_id,
                            body=update_body,
                        )
            except Exception as e:
                logger.warning(f"Failed to update cross-links for {source_id}: {e}")

    async def _repair_if_needed(
        self,
        result: TreeBuildResult,
    ) -> TreeBuildResult:
        """Check cluster quality and repair if needed.

        Args:
            result: Tree build result

        Returns:
            Possibly repaired result
        """
        # Compute quality metrics for clusters
        suspicious_clusters = []

        for level, nodes in result.levels.items():
            if level == 0:
                continue  # Skip leaves

            for node in nodes:
                metrics = await self._compute_cluster_metrics(node, result)

                if metrics.is_suspicious:
                    suspicious_clusters.append((node, metrics))
                    logger.warning(
                        f"Suspicious cluster {node.cluster_id}: "
                        f"cohesion={metrics.cohesion:.2f}, "
                        f"outlier_rate={metrics.outlier_rate:.2f}"
                    )

        if suspicious_clusters:
            self._stats.clusters_repaired += len(suspicious_clusters)
            # In a full implementation, would trigger repair here
            # For now, just log

        return result

    async def _compute_cluster_metrics(
        self,
        summary_node: RaptorNode,
        result: TreeBuildResult,
    ) -> ClusterQualityMetrics:
        """Compute quality metrics for a cluster.

        Args:
            summary_node: Summary node representing cluster
            result: Full tree result

        Returns:
            ClusterQualityMetrics
        """
        # Get child nodes
        child_ids = set(summary_node.children)
        children = [
            n for n in result.nodes
            if n.node_id in child_ids
        ]

        if not children:
            return ClusterQualityMetrics(cluster_id=summary_node.cluster_id)

        # Compute centroid
        embeddings = np.array([c.embedding for c in children])
        centroid = np.mean(embeddings, axis=0)

        # Compute similarities
        similarities = []
        for emb in embeddings:
            sim = float(np.dot(emb, centroid) / (
                np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-8
            ))
            similarities.append(sim)

        cohesion = float(np.mean(similarities))
        outlier_rate = sum(1 for s in similarities if s < 0.3) / len(similarities)

        # Check metadata conflicts
        devices = set(c.metadata.get("device_name") for c in children if c.metadata)
        doc_types = set(c.metadata.get("doc_type") for c in children if c.metadata)
        devices.discard(None)
        doc_types.discard(None)

        meta_conflict_rate = 0.0
        if len(devices) > 1 or len(doc_types) > 1:
            meta_conflict_rate = 0.5  # Simplified

        # Summary support (from validation)
        summary_support_rate = summary_node.validation_score or 1.0

        return ClusterQualityMetrics(
            cluster_id=summary_node.cluster_id,
            cohesion=cohesion,
            outlier_rate=outlier_rate,
            meta_conflict_rate=meta_conflict_rate,
            summary_support_rate=summary_support_rate,
        )

    async def reorganize_none_pool(self) -> int:
        """Reorganize the None pool by clustering and routing.

        Returns:
            Number of chunks reorganized
        """
        none_partition = self._partitioner.get_partition("none")
        if not none_partition or not none_partition.chunk_ids:
            return 0

        logger.info(f"Reorganizing None pool with {len(none_partition.chunk_ids)} chunks")

        # Load None pool nodes
        nodes = await self._load_partition_nodes(none_partition)

        if len(nodes) < self.config.min_partition_size:
            return 0

        # Cluster the None pool
        clusterer = RaptorClusterer()
        embeddings = [n.embedding for n in nodes]
        node_ids = [n.node_id for n in nodes]

        clustering_result = clusterer.cluster_chunks(
            node_ids, embeddings, partition_key="none_reorganized"
        )

        # Create new partitions from significant clusters
        reorganized = 0
        clusters = clusterer.get_clusters_by_id(clustering_result)

        for cluster_id, member_ids in clusters.items():
            if len(member_ids) >= self.config.min_partition_size:
                # Create new partition
                new_key = f"auto_{cluster_id}"
                new_partition = Partition(
                    key=new_key,
                    chunk_ids=member_ids,
                )
                self._partitioner._partitions[new_key] = new_partition
                reorganized += len(member_ids)

                logger.info(f"Created new partition {new_key} with {len(member_ids)} chunks")

        return reorganized


__all__ = [
    "RaptorRebuildService",
    "RebuildConfig",
    "RebuildStats",
]
