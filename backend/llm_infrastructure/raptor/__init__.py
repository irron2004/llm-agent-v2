"""RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) module.

This module implements Meta-guided Hierarchical RAG with Robust Routing:
- Meta-prior Routing with Probabilistic Soft Membership
- Local RAPTOR with Robust Indexing
- Soft escape hatch for metadata misclassification handling

Key components:
- schemas: Data classes for RaptorNode, GroupEdge, Partition
- partition: Metadata-based partitioning
- clustering: GMM clustering with UMAP dimensionality reduction
- soft_router: Soft membership routing with novelty detection
- tree_builder: Local RAPTOR tree construction
- query_router: Mixture-of-Experts query routing
- summary_validator: NLI-based summary validation

Usage:
    from backend.llm_infrastructure.raptor import (
        RaptorNode,
        SoftRouter,
        RaptorTreeBuilder,
        QueryRouter,
    )

    # Build RAPTOR tree
    tree_result = tree_builder.build_local_tree(partition, leaf_nodes)

    # Route query
    routing = query_router.route(query)
"""

# Schemas
from backend.llm_infrastructure.raptor.schemas import (
    ClusterQualityMetrics,
    EdgeType,
    GroupEdge,
    GroupStats,
    NodeLevel,
    Partition,
    RaptorNode,
    RoutingResult,
    StructurePrior,
    ValidationResult,
)

# Partition
from backend.llm_infrastructure.raptor.partition import (
    MetadataPartitioner,
    MetaTree,
    MetaTreeNode,
)

# Clustering
from backend.llm_infrastructure.raptor.clustering import (
    ClusterAssignment,
    ClusteringResult,
    ConstrainedClusterer,
    RaptorClusterer,
)

# Soft Router
from backend.llm_infrastructure.raptor.soft_router import (
    AdaptiveSoftRouter,
    ChunkContext,
    SoftRouter,
    SoftRouterConfig,
)

# Tree Builder
from backend.llm_infrastructure.raptor.tree_builder import (
    CrossGroupLink,
    EnsembleTreeBuilder,
    IncrementalTreeBuilder,
    RaptorTreeBuilder,
    TreeBuildResult,
    TreeBuilderConfig,
)

# Query Router
from backend.llm_infrastructure.raptor.query_router import (
    AdaptiveQueryRouter,
    QueryRouter,
    QueryRouterConfig,
    RoutingDistribution,
)

# Summary Validator
from backend.llm_infrastructure.raptor.summary_validator import (
    BatchSummaryValidator,
    SentenceEvidence,
    SummaryValidator,
    ValidatorConfig,
)

__all__ = [
    # Schemas
    "EdgeType",
    "NodeLevel",
    "RaptorNode",
    "GroupEdge",
    "Partition",
    "GroupStats",
    "RoutingResult",
    "ValidationResult",
    "ClusterQualityMetrics",
    "StructurePrior",
    # Partition
    "MetadataPartitioner",
    "MetaTree",
    "MetaTreeNode",
    # Clustering
    "RaptorClusterer",
    "ConstrainedClusterer",
    "ClusterAssignment",
    "ClusteringResult",
    # Soft Router
    "SoftRouter",
    "AdaptiveSoftRouter",
    "SoftRouterConfig",
    "ChunkContext",
    # Tree Builder
    "RaptorTreeBuilder",
    "IncrementalTreeBuilder",
    "EnsembleTreeBuilder",
    "TreeBuilderConfig",
    "TreeBuildResult",
    "CrossGroupLink",
    # Query Router
    "QueryRouter",
    "AdaptiveQueryRouter",
    "QueryRouterConfig",
    "RoutingDistribution",
    # Summary Validator
    "SummaryValidator",
    "BatchSummaryValidator",
    "ValidatorConfig",
    "SentenceEvidence",
]
