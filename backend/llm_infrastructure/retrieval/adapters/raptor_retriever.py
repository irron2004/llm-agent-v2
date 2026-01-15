"""RAPTOR hierarchical retriever adapter.

Provides a BaseRetriever implementation for Meta-guided Hierarchical RAG:
1. Routes queries to relevant partitions using MoE
2. Searches within local RAPTOR trees (collapsed tree strategy)
3. Expands summary nodes to leaf evidence
4. Combines results using RRF

Usage:
    from backend.llm_infrastructure.retrieval.adapters.raptor_retriever import (
        RaptorHierarchicalRetriever,
    )

    retriever = RaptorHierarchicalRetriever(
        es_engine=engine,
        embedder=embedder,
        query_router=router,
    )
    results = retriever.retrieve("search query", top_k=10)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from backend.llm_infrastructure.retrieval.base import BaseRetriever, RetrievalResult
from backend.llm_infrastructure.retrieval.registry import register_retriever

if TYPE_CHECKING:
    from backend.llm_infrastructure.embedding.base import BaseEmbedder
    from backend.llm_infrastructure.raptor.query_router import QueryRouter
    from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchEngine

logger = logging.getLogger(__name__)


@dataclass
class RaptorRetrieverConfig:
    """Configuration for RAPTOR hierarchical retriever.

    Attributes:
        tree_strategy: Search strategy ("collapsed", "tree_traversal", "layer_wise")
        min_weight_threshold: Minimum routing weight to search group
        include_summaries: Whether to include summary nodes in results
        expand_summaries: Whether to expand summaries to leaf nodes
        max_expansion_depth: Maximum depth for summary expansion
        use_rrf: Whether to use RRF for combining group results
        rrf_k: RRF constant
        rerank: Whether to apply cross-encoder reranking
        global_search_weight: Weight for global fallback search
    """

    tree_strategy: str = "collapsed"
    min_weight_threshold: float = 0.05
    include_summaries: bool = True
    expand_summaries: bool = True
    max_expansion_depth: int = 2
    use_rrf: bool = True
    rrf_k: int = 60
    rerank: bool = False
    global_search_weight: float = 0.1


@dataclass
class ExpandedResult:
    """Result with expansion information.

    Attributes:
        result: Base retrieval result
        is_summary: Whether this is a summary node
        children_ids: Child node IDs if summary
        expanded_from: Summary ID if this was expanded
        partition_key: Source partition
        raptor_level: RAPTOR tree level
    """

    result: RetrievalResult
    is_summary: bool = False
    children_ids: list[str] = field(default_factory=list)
    expanded_from: str | None = None
    partition_key: str = ""
    raptor_level: int = 0


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize a vector."""
    norm = np.linalg.norm(vec)
    norm = max(norm, eps)
    return vec / norm


@register_retriever("raptor_hierarchical", version="v1")
class RaptorHierarchicalRetriever(BaseRetriever):
    """RAPTOR hierarchical retriever using MoE routing.

    Implements the retrieval strategy:
    1. Route query to partitions using QueryRouter
    2. Search each partition's RAPTOR tree
    3. Optionally expand summary nodes to leaves
    4. Combine using RRF
    """

    def __init__(
        self,
        es_engine: "EsSearchEngine",
        embedder: "BaseEmbedder",
        query_router: "QueryRouter",
        *,
        config: RaptorRetrieverConfig | None = None,
        reranker: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RAPTOR retriever.

        Args:
            es_engine: EsSearchEngine instance
            embedder: Embedding model for query vectorization
            query_router: QueryRouter for partition routing
            config: Retriever configuration
            reranker: Optional cross-encoder reranker
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.es_engine = es_engine
        self.embedder = embedder
        self.query_router = query_router
        self.config = config or RaptorRetrieverConfig()
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        tenant_id: str | None = None,
        project_id: str | None = None,
        device_name: str | None = None,
        doc_type: str | None = None,
        include_global: bool = True,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Retrieve relevant documents using RAPTOR hierarchy.

        Args:
            query: Search query text
            top_k: Number of results to return
            tenant_id: Optional tenant filter
            project_id: Optional project filter
            device_name: Optional device filter (overrides routing)
            doc_type: Optional doc type filter (overrides routing)
            include_global: Whether to include global fallback
            **kwargs: Additional parameters

        Returns:
            List of RetrievalResult sorted by score
        """
        # Step 1: Route query to partitions
        routing = self.query_router.route(query)
        logger.debug(f"Routing distribution: {routing.group_weights}")

        # Step 2: Embed query
        query_vec = self._embed_query(query)

        # Step 3: Search each group
        all_results: list[list[ExpandedResult]] = []
        group_weights: list[float] = []

        for group_id, weight in routing.group_weights.items():
            if weight < self.config.min_weight_threshold:
                continue

            if group_id == "global":
                if include_global:
                    # Global search across all partitions
                    results = self._search_global(
                        query_vec, query, top_k, tenant_id, project_id
                    )
                    all_results.append(results)
                    group_weights.append(weight)
            else:
                # Search specific partition
                results = self._search_partition(
                    query_vec, query, group_id, top_k, tenant_id, project_id
                )
                all_results.append(results)
                group_weights.append(weight)

        # Step 4: Combine results
        if self.config.use_rrf:
            combined = self._combine_rrf(all_results, group_weights, top_k)
        else:
            combined = self._combine_weighted(all_results, group_weights, top_k)

        # Step 5: Optional reranking
        if self.config.rerank and self.reranker:
            combined = self._rerank(query, combined, top_k)

        return [r.result for r in combined[:top_k]]

    def _embed_query(self, query: str) -> list[float]:
        """Embed query text to vector."""
        if hasattr(self.embedder, "embed"):
            vec = self.embedder.embed(query)
        elif hasattr(self.embedder, "embed_batch"):
            vec = self.embedder.embed_batch([query])[0]
        else:
            raise TypeError("embedder must implement embed() or embed_batch()")

        arr = np.asarray(vec, dtype=np.float32)
        arr = _l2_normalize(arr)
        return arr.tolist()

    def _search_partition(
        self,
        query_vec: list[float],
        query_text: str,
        partition_key: str,
        top_k: int,
        tenant_id: str | None,
        project_id: str | None,
    ) -> list[ExpandedResult]:
        """Search within a specific partition.

        Uses collapsed tree strategy: search all levels and combine.
        """
        results: list[ExpandedResult] = []

        # Build filter for partition
        filters = self._build_partition_filter(
            partition_key, tenant_id, project_id
        )

        if self.config.tree_strategy == "collapsed":
            results = self._collapsed_tree_search(
                query_vec, query_text, filters, top_k, partition_key
            )
        elif self.config.tree_strategy == "tree_traversal":
            results = self._tree_traversal_search(
                query_vec, query_text, filters, top_k, partition_key
            )
        else:
            # Default to collapsed
            results = self._collapsed_tree_search(
                query_vec, query_text, filters, top_k, partition_key
            )

        return results

    def _collapsed_tree_search(
        self,
        query_vec: list[float],
        query_text: str,
        filters: dict[str, Any],
        top_k: int,
        partition_key: str,
    ) -> list[ExpandedResult]:
        """Collapsed tree search: query all levels simultaneously.

        This is the recommended RAPTOR retrieval strategy.
        """
        results: list[ExpandedResult] = []

        # Search with hybrid (dense + BM25)
        hits = self.es_engine.hybrid_search(
            query_vector=query_vec,
            query_text=query_text,
            top_k=top_k * 2,  # Over-fetch for filtering
            filters=filters,
            use_rrf=True,
        )

        for hit in hits:
            is_summary = hit.source.get("is_summary_node", False)
            raptor_level = hit.source.get("raptor_level", 0)
            children_ids = hit.source.get("raptor_children_ids", [])

            expanded = ExpandedResult(
                result=hit.to_retrieval_result(),
                is_summary=is_summary,
                children_ids=children_ids,
                partition_key=partition_key,
                raptor_level=raptor_level,
            )
            results.append(expanded)

        # Optionally expand summaries
        if self.config.expand_summaries:
            results = self._expand_summary_nodes(results, query_vec, top_k)

        return results

    def _tree_traversal_search(
        self,
        query_vec: list[float],
        query_text: str,
        filters: dict[str, Any],
        top_k: int,
        partition_key: str,
    ) -> list[ExpandedResult]:
        """Tree traversal search: start from root, traverse down.

        Alternative strategy for more structured retrieval.
        """
        results: list[ExpandedResult] = []

        # Start from top level (summaries)
        summary_filters = {**filters, "is_summary_node": True}
        summary_hits = self.es_engine.dense_search(
            query_vector=query_vec,
            top_k=top_k,
            filters=summary_filters,
        )

        # Get top summaries and expand
        for hit in summary_hits[:3]:  # Top 3 summaries
            children_ids = hit.source.get("raptor_children_ids", [])

            # Fetch children
            if children_ids:
                child_hits = self._fetch_nodes_by_ids(children_ids)
                for child_hit in child_hits:
                    results.append(
                        ExpandedResult(
                            result=child_hit.to_retrieval_result(),
                            is_summary=child_hit.source.get("is_summary_node", False),
                            children_ids=child_hit.source.get("raptor_children_ids", []),
                            expanded_from=hit.doc_id,
                            partition_key=partition_key,
                        )
                    )

        return results

    def _expand_summary_nodes(
        self,
        results: list[ExpandedResult],
        query_vec: list[float],
        top_k: int,
    ) -> list[ExpandedResult]:
        """Expand summary nodes to their leaf children.

        Adds leaf nodes as evidence for summary matches.
        """
        expanded_results: list[ExpandedResult] = []
        seen_ids: set[str] = set()

        for result in results:
            # Always include the original result
            if result.result.doc_id not in seen_ids:
                expanded_results.append(result)
                seen_ids.add(result.result.doc_id)

            # Expand if summary
            if result.is_summary and result.children_ids:
                child_hits = self._fetch_nodes_by_ids(result.children_ids)

                for child_hit in child_hits:
                    child_id = child_hit.doc_id
                    if child_id not in seen_ids:
                        expanded_results.append(
                            ExpandedResult(
                                result=child_hit.to_retrieval_result(),
                                is_summary=child_hit.source.get("is_summary_node", False),
                                children_ids=child_hit.source.get("raptor_children_ids", []),
                                expanded_from=result.result.doc_id,
                                partition_key=result.partition_key,
                            )
                        )
                        seen_ids.add(child_id)

        return expanded_results

    def _fetch_nodes_by_ids(self, node_ids: list[str]) -> list[Any]:
        """Fetch nodes by their IDs."""
        if not node_ids:
            return []

        try:
            # Use mget for batch fetch
            response = self.es_engine.es_client.mget(
                index=self.es_engine.index_name,
                body={"ids": node_ids},
            )

            hits = []
            for doc in response.get("docs", []):
                if doc.get("found"):
                    from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchHit
                    hit = EsSearchHit(
                        doc_id=doc["_id"],
                        score=1.0,  # No score for mget
                        source=doc.get("_source", {}),
                    )
                    hits.append(hit)
            return hits
        except Exception as e:
            logger.warning(f"Failed to fetch nodes: {e}")
            return []

    def _search_global(
        self,
        query_vec: list[float],
        query_text: str,
        top_k: int,
        tenant_id: str | None,
        project_id: str | None,
    ) -> list[ExpandedResult]:
        """Global search across all partitions (fallback)."""
        filters = self.es_engine.build_filter(
            tenant_id=tenant_id,
            project_id=project_id,
        )

        # Prefer leaf nodes for global search
        filters["is_summary_node"] = False

        hits = self.es_engine.hybrid_search(
            query_vector=query_vec,
            query_text=query_text,
            top_k=top_k,
            filters=filters,
            use_rrf=True,
        )

        return [
            ExpandedResult(
                result=hit.to_retrieval_result(),
                is_summary=False,
                partition_key="global",
            )
            for hit in hits
        ]

    def _build_partition_filter(
        self,
        partition_key: str,
        tenant_id: str | None,
        project_id: str | None,
    ) -> dict[str, Any]:
        """Build ES filter for a specific partition."""
        filters = self.es_engine.build_filter(
            tenant_id=tenant_id,
            project_id=project_id,
        )
        filters["partition_key"] = partition_key
        return filters

    def _combine_rrf(
        self,
        result_lists: list[list[ExpandedResult]],
        weights: list[float],
        top_k: int,
    ) -> list[ExpandedResult]:
        """Combine results using RRF (Reciprocal Rank Fusion)."""
        rrf_scores: dict[str, tuple[float, ExpandedResult]] = {}
        k = self.config.rrf_k

        for result_list, weight in zip(result_lists, weights):
            for rank, result in enumerate(result_list):
                doc_id = result.result.doc_id
                rrf_score = weight * (1.0 / (k + rank + 1))

                if doc_id in rrf_scores:
                    current_score, existing = rrf_scores[doc_id]
                    rrf_scores[doc_id] = (current_score + rrf_score, existing)
                else:
                    rrf_scores[doc_id] = (rrf_score, result)

        # Sort by RRF score
        sorted_results = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)

        # Update scores in results
        final_results = []
        for score, result in sorted_results[:top_k]:
            result.result.score = score
            final_results.append(result)

        return final_results

    def _combine_weighted(
        self,
        result_lists: list[list[ExpandedResult]],
        weights: list[float],
        top_k: int,
    ) -> list[ExpandedResult]:
        """Combine results using weighted scores."""
        combined_scores: dict[str, tuple[float, ExpandedResult]] = {}

        for result_list, weight in zip(result_lists, weights):
            for result in result_list:
                doc_id = result.result.doc_id
                weighted_score = weight * result.result.score

                if doc_id in combined_scores:
                    current_score, existing = combined_scores[doc_id]
                    combined_scores[doc_id] = (current_score + weighted_score, existing)
                else:
                    combined_scores[doc_id] = (weighted_score, result)

        sorted_results = sorted(
            combined_scores.values(), key=lambda x: x[0], reverse=True
        )

        final_results = []
        for score, result in sorted_results[:top_k]:
            result.result.score = score
            final_results.append(result)

        return final_results

    def _rerank(
        self,
        query: str,
        results: list[ExpandedResult],
        top_k: int,
    ) -> list[ExpandedResult]:
        """Apply cross-encoder reranking."""
        if not self.reranker or not results:
            return results

        try:
            pairs = [(query, r.result.content) for r in results]
            scores = self.reranker.predict(pairs)

            for result, score in zip(results, scores):
                result.result.score = float(score)

            results.sort(key=lambda x: x.result.score, reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return results


@register_retriever("raptor_flat", version="v1")
class RaptorFlatRetriever(RaptorHierarchicalRetriever):
    """Simplified RAPTOR retriever without tree hierarchy.

    Uses partition routing but searches flat (leaf nodes only).
    Useful as baseline for ablation studies.
    """

    def _search_partition(
        self,
        query_vec: list[float],
        query_text: str,
        partition_key: str,
        top_k: int,
        tenant_id: str | None,
        project_id: str | None,
    ) -> list[ExpandedResult]:
        """Search partition without tree hierarchy (leaf only)."""
        filters = self._build_partition_filter(
            partition_key, tenant_id, project_id
        )
        # Only search leaf nodes
        filters["raptor_level"] = 0

        hits = self.es_engine.hybrid_search(
            query_vector=query_vec,
            query_text=query_text,
            top_k=top_k,
            filters=filters,
            use_rrf=True,
        )

        return [
            ExpandedResult(
                result=hit.to_retrieval_result(),
                is_summary=False,
                partition_key=partition_key,
                raptor_level=0,
            )
            for hit in hits
        ]


__all__ = [
    "RaptorHierarchicalRetriever",
    "RaptorFlatRetriever",
    "RaptorRetrieverConfig",
    "ExpandedResult",
]
