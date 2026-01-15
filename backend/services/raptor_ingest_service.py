"""Online RAPTOR ingestion service.

This service handles real-time chunk ingestion with soft routing:
1. Route incoming chunks to partitions
2. Store with group edges (soft membership)
3. Handle novelty detection for new groups
4. Queue chunks for offline RAPTOR tree building

Usage:
    service = RaptorIngestService(
        es_client=es_client,
        embedder=embedder,
        soft_router=router,
    )
    await service.ingest_chunk(chunk)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from backend.llm_infrastructure.raptor.partition import MetadataPartitioner
from backend.llm_infrastructure.raptor.schemas import (
    EdgeType,
    GroupEdge,
    Partition,
    RaptorNode,
)
from backend.llm_infrastructure.raptor.soft_router import (
    ChunkContext,
    SoftRouter,
    SoftRouterConfig,
)

if TYPE_CHECKING:
    from elasticsearch import AsyncElasticsearch, Elasticsearch

    from backend.llm_infrastructure.elasticsearch.document import EsChunkDocument
    from backend.llm_infrastructure.embedding.base import BaseEmbedder

logger = logging.getLogger(__name__)


@dataclass
class IngestConfig:
    """Configuration for RAPTOR ingestion.

    Attributes:
        batch_size: Batch size for bulk indexing
        novelty_queue_threshold: Queue chunks for review if novelty > threshold
        auto_create_partitions: Whether to auto-create new partitions
        store_edges: Whether to store group edges
        update_centroids: Whether to update partition centroids online
    """

    batch_size: int = 100
    novelty_queue_threshold: float = 0.5
    auto_create_partitions: bool = True
    store_edges: bool = True
    update_centroids: bool = False


@dataclass
class IngestStats:
    """Ingestion statistics.

    Attributes:
        total_processed: Total chunks processed
        total_routed: Chunks successfully routed
        novel_chunks: Chunks flagged as novel
        new_partitions: New partitions created
        errors: Number of errors
        start_time: Start timestamp
    """

    total_processed: int = 0
    total_routed: int = 0
    novel_chunks: int = 0
    new_partitions: int = 0
    errors: int = 0
    start_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def success_rate(self) -> float:
        """Compute success rate."""
        if self.total_processed == 0:
            return 0.0
        return self.total_routed / self.total_processed


class RaptorIngestService:
    """Online service for RAPTOR-based chunk ingestion.

    Handles real-time routing and indexing of chunks:
    - Soft routing to partition(s)
    - Edge-based multi-membership storage
    - Novelty detection and queuing

    Args:
        es_client: Elasticsearch client
        index_name: Target index name
        embedder: Embedder for computing embeddings
        soft_router: Soft router for partition assignment
        partitioner: Metadata partitioner
        config: Ingestion configuration
    """

    def __init__(
        self,
        es_client: "Elasticsearch | AsyncElasticsearch",
        index_name: str,
        embedder: "BaseEmbedder",
        soft_router: SoftRouter | None = None,
        partitioner: MetadataPartitioner | None = None,
        config: IngestConfig | None = None,
    ) -> None:
        self.es_client = es_client
        self.index_name = index_name
        self.embedder = embedder
        self.config = config or IngestConfig()

        # Initialize partitioner if not provided
        self._partitioner = partitioner or MetadataPartitioner(
            es_client=es_client,  # type: ignore
            index_name=index_name,
        )

        # Initialize router if not provided
        self._router = soft_router or SoftRouter(
            config=SoftRouterConfig(),
            partitions=list(self._partitioner.partitions.values()),
        )

        self._stats = IngestStats()
        self._novel_queue: list[ChunkContext] = []
        self._pending_batch: list[dict[str, Any]] = []

    @property
    def stats(self) -> IngestStats:
        """Get ingestion statistics."""
        return self._stats

    @property
    def novel_queue(self) -> list[ChunkContext]:
        """Get queue of novel chunks awaiting review."""
        return self._novel_queue

    async def ingest_chunk(
        self,
        chunk: "EsChunkDocument",
        adjacent_chunks: list[str] | None = None,
    ) -> bool:
        """Ingest a single chunk with soft routing.

        Args:
            chunk: Chunk document to ingest
            adjacent_chunks: IDs of adjacent chunks for structure prior

        Returns:
            True if successfully ingested
        """
        self._stats.total_processed += 1

        try:
            # Ensure embedding
            if not chunk.embedding:
                chunk.embedding = self.embedder.embed(chunk.content)

            # Create context
            context = ChunkContext.from_es_doc(chunk, adjacent_chunks)

            # Route
            routing_result = self._router.route_leaf(context)

            # Handle novelty
            if routing_result.is_novel:
                self._stats.novel_chunks += 1
                if routing_result.novelty_score > self.config.novelty_queue_threshold:
                    self._novel_queue.append(context)

                # Auto-create partition if enabled
                if self.config.auto_create_partitions:
                    partition = self._create_partition(chunk)
                    routing_result.edges.append(
                        GroupEdge(
                            leaf_id=chunk.chunk_id,
                            group_id=partition.key,
                            weight=1.0,
                            edge_type=EdgeType.PRIMARY,
                        )
                    )
                    self._stats.new_partitions += 1

            # Prepare document for indexing
            doc = self._prepare_document(chunk, routing_result.edges)

            # Add to batch
            self._pending_batch.append(doc)

            # Flush if batch is full
            if len(self._pending_batch) >= self.config.batch_size:
                await self._flush_batch()

            self._stats.total_routed += 1
            return True

        except Exception as e:
            logger.error(f"Failed to ingest chunk {chunk.chunk_id}: {e}")
            self._stats.errors += 1
            return False

    async def ingest_batch(
        self,
        chunks: list["EsChunkDocument"],
    ) -> int:
        """Ingest a batch of chunks.

        Args:
            chunks: List of chunk documents

        Returns:
            Number of successfully ingested chunks
        """
        # Sort by position for adjacency
        sorted_chunks = sorted(chunks, key=lambda c: (c.doc_id, c.page))

        success_count = 0
        adjacent_map: dict[str, list[str]] = {}

        # Build adjacency map
        for i, chunk in enumerate(sorted_chunks):
            adjacent = []
            if i > 0 and sorted_chunks[i - 1].doc_id == chunk.doc_id:
                adjacent.append(sorted_chunks[i - 1].chunk_id)
            if i < len(sorted_chunks) - 1 and sorted_chunks[i + 1].doc_id == chunk.doc_id:
                adjacent.append(sorted_chunks[i + 1].chunk_id)
            adjacent_map[chunk.chunk_id] = adjacent

        # Ingest with adjacency info
        for chunk in sorted_chunks:
            adjacent = adjacent_map.get(chunk.chunk_id, [])
            if await self.ingest_chunk(chunk, adjacent):
                success_count += 1

        # Flush remaining
        await self._flush_batch()

        return success_count

    def _prepare_document(
        self,
        chunk: "EsChunkDocument",
        edges: list[GroupEdge],
    ) -> dict[str, Any]:
        """Prepare document for ES indexing.

        Args:
            chunk: Original chunk document
            edges: Group edges from routing

        Returns:
            ES document dictionary
        """
        doc = chunk.to_es_doc()

        # Add RAPTOR fields
        doc["raptor_level"] = 0  # Leaf node
        doc["is_summary_node"] = False

        # Set primary partition key
        primary_edge = next(
            (e for e in edges if e.edge_type == EdgeType.PRIMARY),
            edges[0] if edges else None,
        )
        if primary_edge:
            doc["partition_key"] = primary_edge.group_id

        # Add group edges if configured
        if self.config.store_edges and edges:
            doc["group_edges"] = [e.to_dict() for e in edges]

        return doc

    def _create_partition(
        self,
        chunk: "EsChunkDocument",
    ) -> Partition:
        """Create a new partition for a novel chunk.

        Args:
            chunk: Chunk that triggered novelty

        Returns:
            New Partition
        """
        partition = Partition.from_metadata(
            device_name=chunk.device_name or None,
            doc_type=chunk.doc_type or None,
        )

        # Add to partitioner
        self._partitioner._partitions[partition.key] = partition

        # Update router
        self._router.set_partitions(list(self._partitioner.partitions.values()))

        logger.info(f"Created new partition: {partition.key}")
        return partition

    async def _flush_batch(self) -> None:
        """Flush pending batch to Elasticsearch."""
        if not self._pending_batch:
            return

        try:
            # Build bulk request
            actions = []
            for doc in self._pending_batch:
                action = {"index": {"_index": self.index_name, "_id": doc["chunk_id"]}}
                actions.append(action)
                actions.append(doc)

            # Execute bulk
            if hasattr(self.es_client, "bulk"):
                if asyncio.iscoroutinefunction(self.es_client.bulk):
                    await self.es_client.bulk(body=actions, refresh="wait_for")
                else:
                    self.es_client.bulk(body=actions, refresh="wait_for")

            logger.debug(f"Flushed {len(self._pending_batch)} documents")
            self._pending_batch.clear()

        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            self._stats.errors += len(self._pending_batch)
            self._pending_batch.clear()

    async def update_partitions(self) -> None:
        """Refresh partition data from ES."""
        self._partitioner.partition_by_metadata()
        self._router.set_partitions(list(self._partitioner.partitions.values()))
        logger.info("Updated partitions from ES")

    def process_novel_queue(
        self,
        handler: Any = None,
    ) -> int:
        """Process queued novel chunks.

        Args:
            handler: Optional handler for novel chunks

        Returns:
            Number of processed chunks
        """
        processed = 0

        for context in self._novel_queue:
            if handler:
                try:
                    handler(context)
                    processed += 1
                except Exception as e:
                    logger.warning(f"Novel chunk handler failed: {e}")
            else:
                # Default: log for manual review
                logger.info(
                    f"Novel chunk: {context.chunk_id}, "
                    f"device={context.device_name}, doc_type={context.doc_type}"
                )
                processed += 1

        self._novel_queue.clear()
        return processed

    def get_routing_diagnostics(
        self,
        chunk: "EsChunkDocument",
    ) -> dict[str, Any]:
        """Get routing diagnostics for a chunk.

        Useful for debugging routing decisions.

        Args:
            chunk: Chunk to diagnose

        Returns:
            Diagnostic information
        """
        if not chunk.embedding:
            chunk.embedding = self.embedder.embed(chunk.content)

        context = ChunkContext.from_es_doc(chunk)
        routing_result = self._router.route_leaf(context)

        return {
            "chunk_id": chunk.chunk_id,
            "has_metadata": context.has_metadata,
            "device_name": context.device_name,
            "doc_type": context.doc_type,
            "is_novel": routing_result.is_novel,
            "novelty_score": routing_result.novelty_score,
            "edges": [
                {
                    "group_id": e.group_id,
                    "weight": e.weight,
                    "edge_type": e.edge_type.value,
                }
                for e in routing_result.edges
            ],
            "primary_group": routing_result.primary_group,
            "secondary_groups": routing_result.secondary_groups,
        }


class SyncRaptorIngestService(RaptorIngestService):
    """Synchronous version of RaptorIngestService.

    For use in non-async contexts.
    """

    def ingest_chunk_sync(
        self,
        chunk: "EsChunkDocument",
        adjacent_chunks: list[str] | None = None,
    ) -> bool:
        """Synchronous chunk ingestion."""
        return asyncio.get_event_loop().run_until_complete(
            self.ingest_chunk(chunk, adjacent_chunks)
        )

    def ingest_batch_sync(
        self,
        chunks: list["EsChunkDocument"],
    ) -> int:
        """Synchronous batch ingestion."""
        return asyncio.get_event_loop().run_until_complete(
            self.ingest_batch(chunks)
        )


__all__ = [
    "RaptorIngestService",
    "SyncRaptorIngestService",
    "IngestConfig",
    "IngestStats",
]
