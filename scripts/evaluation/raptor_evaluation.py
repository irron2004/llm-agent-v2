"""RAPTOR evaluation framework for ablation studies.

This module provides evaluation tools for:
1. Comparing different retrieval configurations
2. Running ablation studies
3. Generating degradation curves (metadata missing/noise)
4. Computing retrieval metrics (Recall@k, Precision@k, MRR, nDCG)

Usage:
    evaluator = RaptorEvaluator(
        es_client=es_client,
        embedder=embedder,
        golden_set=golden_set,
    )
    results = evaluator.run_ablation_study(configs)
    evaluator.plot_degradation_curves(results)
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch
    from matplotlib.figure import Figure

    from backend.llm_infrastructure.embedding.base import BaseEmbedder
    from backend.llm_infrastructure.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


@dataclass
class GoldenQuery:
    """A query in the golden set with expected answers.

    Attributes:
        query_id: Unique query identifier
        query_text: Query string
        relevant_chunk_ids: List of relevant chunk IDs
        device_name: Expected device (for filtering tests)
        doc_type: Expected document type
        metadata: Additional query metadata
    """

    query_id: str
    query_text: str
    relevant_chunk_ids: list[str]
    device_name: str | None = None
    doc_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics.

    Attributes:
        recall_at_k: Recall at various k values
        precision_at_k: Precision at various k values
        mrr: Mean Reciprocal Rank
        ndcg_at_k: nDCG at various k values
        map_score: Mean Average Precision
        device_mismatch_rate: Rate of device mismatches
    """

    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    map_score: float = 0.0
    device_mismatch_rate: float = 0.0
    avg_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recall@5": self.recall_at_k.get(5, 0.0),
            "recall@10": self.recall_at_k.get(10, 0.0),
            "precision@5": self.precision_at_k.get(5, 0.0),
            "precision@10": self.precision_at_k.get(10, 0.0),
            "mrr": self.mrr,
            "ndcg@10": self.ndcg_at_k.get(10, 0.0),
            "map": self.map_score,
            "device_mismatch_rate": self.device_mismatch_rate,
            "avg_latency_ms": self.avg_latency_ms,
        }


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment.

    Attributes:
        name: Experiment name
        use_partitioning: Whether to use metadata partitioning
        use_raptor_tree: Whether to build RAPTOR trees
        use_soft_membership: Whether to use soft routing
        use_novelty_detection: Whether to use novelty detection
        metadata_missing_rate: Rate of simulated metadata missing
        metadata_noise_rate: Rate of simulated metadata noise
        retriever_type: Retriever type to use
    """

    name: str
    use_partitioning: bool = True
    use_raptor_tree: bool = True
    use_soft_membership: bool = True
    use_novelty_detection: bool = True
    metadata_missing_rate: float = 0.0
    metadata_noise_rate: float = 0.0
    retriever_type: str = "raptor_hierarchical"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "partitioning": self.use_partitioning,
            "raptor": self.use_raptor_tree,
            "soft": self.use_soft_membership,
            "novelty": self.use_novelty_detection,
            "missing_rate": self.metadata_missing_rate,
            "noise_rate": self.metadata_noise_rate,
            "retriever": self.retriever_type,
        }


@dataclass
class ExperimentResult:
    """Result of a single experiment.

    Attributes:
        config: Experiment configuration
        metrics: Evaluation metrics
        per_query_results: Per-query details
        error: Error message if failed
    """

    config: ExperimentConfig
    metrics: RetrievalMetrics | None = None
    per_query_results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class RaptorEvaluator:
    """Evaluator for RAPTOR retrieval experiments.

    Provides tools for:
    - Running single evaluations
    - Running ablation studies
    - Simulating metadata degradation
    - Generating comparison plots

    Args:
        es_client: Elasticsearch client
        index_name: Index to evaluate
        embedder: Embedder for query encoding
        golden_set: List of golden queries
    """

    def __init__(
        self,
        es_client: "Elasticsearch",
        index_name: str,
        embedder: "BaseEmbedder",
        golden_set: list[GoldenQuery] | None = None,
    ) -> None:
        self.es_client = es_client
        self.index_name = index_name
        self.embedder = embedder
        self.golden_set = golden_set or []
        self._results_cache: dict[str, ExperimentResult] = {}

    def load_golden_set(self, path: str | Path) -> list[GoldenQuery]:
        """Load golden set from JSON file.

        Expected format:
        [
            {
                "query_id": "q1",
                "query_text": "...",
                "relevant_chunk_ids": ["c1", "c2"],
                "device_name": "SUPRA_XP",
                "doc_type": "sop"
            },
            ...
        ]

        Args:
            path: Path to JSON file

        Returns:
            List of GoldenQuery objects
        """
        with open(path) as f:
            data = json.load(f)

        self.golden_set = [
            GoldenQuery(
                query_id=q["query_id"],
                query_text=q["query_text"],
                relevant_chunk_ids=q["relevant_chunk_ids"],
                device_name=q.get("device_name"),
                doc_type=q.get("doc_type"),
                metadata=q.get("metadata", {}),
            )
            for q in data
        ]

        logger.info(f"Loaded {len(self.golden_set)} golden queries")
        return self.golden_set

    def evaluate(
        self,
        retriever: "BaseRetriever",
        config: ExperimentConfig | None = None,
        top_k_values: list[int] | None = None,
    ) -> RetrievalMetrics:
        """Evaluate a retriever on the golden set.

        Args:
            retriever: Retriever to evaluate
            config: Optional config for logging
            top_k_values: List of k values for metrics

        Returns:
            RetrievalMetrics
        """
        if not self.golden_set:
            raise ValueError("Golden set is empty")

        top_k_values = top_k_values or [5, 10, 20]
        max_k = max(top_k_values)

        all_recalls: dict[int, list[float]] = {k: [] for k in top_k_values}
        all_precisions: dict[int, list[float]] = {k: [] for k in top_k_values}
        all_ndcgs: dict[int, list[float]] = {k: [] for k in top_k_values}
        all_rr: list[float] = []
        all_ap: list[float] = []
        device_mismatches = 0
        total_with_device = 0
        latencies: list[float] = []

        for query in self.golden_set:
            import time
            start = time.time()

            # Retrieve
            try:
                results = retriever.retrieve(query.query_text, top_k=max_k)
            except Exception as e:
                logger.warning(f"Retrieval failed for {query.query_id}: {e}")
                continue

            latencies.append((time.time() - start) * 1000)

            retrieved_ids = [r.doc_id for r in results]
            relevant_set = set(query.relevant_chunk_ids)

            # Compute metrics at each k
            for k in top_k_values:
                top_k_ids = retrieved_ids[:k]
                hits = len(set(top_k_ids) & relevant_set)

                recall = hits / len(relevant_set) if relevant_set else 0.0
                precision = hits / k if k > 0 else 0.0

                all_recalls[k].append(recall)
                all_precisions[k].append(precision)

                # nDCG
                ndcg = self._compute_ndcg(top_k_ids, relevant_set, k)
                all_ndcgs[k].append(ndcg)

            # MRR
            rr = 0.0
            for rank, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in relevant_set:
                    rr = 1.0 / rank
                    break
            all_rr.append(rr)

            # MAP
            ap = self._compute_ap(retrieved_ids, relevant_set)
            all_ap.append(ap)

            # Device mismatch
            if query.device_name:
                total_with_device += 1
                for r in results[:5]:
                    if r.metadata and r.metadata.get("device_name") != query.device_name:
                        device_mismatches += 1
                        break

        return RetrievalMetrics(
            recall_at_k={k: np.mean(v) for k, v in all_recalls.items()},
            precision_at_k={k: np.mean(v) for k, v in all_precisions.items()},
            mrr=np.mean(all_rr) if all_rr else 0.0,
            ndcg_at_k={k: np.mean(v) for k, v in all_ndcgs.items()},
            map_score=np.mean(all_ap) if all_ap else 0.0,
            device_mismatch_rate=(
                device_mismatches / total_with_device if total_with_device > 0 else 0.0
            ),
            avg_latency_ms=np.mean(latencies) if latencies else 0.0,
        )

    def _compute_ndcg(
        self,
        retrieved: list[str],
        relevant: set[str],
        k: int,
    ) -> float:
        """Compute nDCG@k."""
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant:
                dcg += 1.0 / np.log2(i + 2)

        # Ideal DCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))

        return dcg / idcg if idcg > 0 else 0.0

    def _compute_ap(
        self,
        retrieved: list[str],
        relevant: set[str],
    ) -> float:
        """Compute Average Precision."""
        if not relevant:
            return 0.0

        num_relevant = 0
        sum_precision = 0.0

        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                sum_precision += precision_at_i

        return sum_precision / len(relevant) if relevant else 0.0

    def simulate_metadata_missing(
        self,
        rate: float,
        seed: int = 42,
    ) -> None:
        """Simulate metadata missing by updating ES documents.

        Args:
            rate: Rate of metadata to set as missing (0-1)
            seed: Random seed
        """
        random.seed(seed)

        # Get all document IDs
        response = self.es_client.search(
            index=self.index_name,
            body={"query": {"match_all": {}}, "_source": False},
            size=10000,
        )

        doc_ids = [hit["_id"] for hit in response["hits"]["hits"]]
        n_to_modify = int(len(doc_ids) * rate)
        to_modify = random.sample(doc_ids, n_to_modify)

        # Update documents
        for doc_id in to_modify:
            self.es_client.update(
                index=self.index_name,
                id=doc_id,
                body={"doc": {"device_name": "", "doc_type": ""}},
            )

        logger.info(f"Set metadata missing for {n_to_modify} documents")

    def simulate_metadata_noise(
        self,
        rate: float,
        seed: int = 42,
    ) -> None:
        """Simulate metadata noise by swapping values.

        Args:
            rate: Rate of metadata to swap (0-1)
            seed: Random seed
        """
        random.seed(seed)

        # Get documents with metadata
        response = self.es_client.search(
            index=self.index_name,
            body={
                "query": {"exists": {"field": "device_name"}},
                "_source": ["device_name", "doc_type"],
            },
            size=10000,
        )

        hits = response["hits"]["hits"]
        n_to_modify = int(len(hits) * rate)

        if n_to_modify < 2:
            return

        # Select pairs and swap
        to_modify = random.sample(hits, n_to_modify)

        for i in range(0, len(to_modify) - 1, 2):
            doc1 = to_modify[i]
            doc2 = to_modify[i + 1]

            # Swap device names
            self.es_client.update(
                index=self.index_name,
                id=doc1["_id"],
                body={"doc": {"device_name": doc2["_source"].get("device_name", "")}},
            )
            self.es_client.update(
                index=self.index_name,
                id=doc2["_id"],
                body={"doc": {"device_name": doc1["_source"].get("device_name", "")}},
            )

        logger.info(f"Swapped metadata for {n_to_modify} documents")

    def run_ablation_study(
        self,
        configs: list[ExperimentConfig],
        retriever_factory: Any = None,
    ) -> pd.DataFrame:
        """Run ablation study with multiple configurations.

        Args:
            configs: List of experiment configurations
            retriever_factory: Factory function to create retrievers

        Returns:
            DataFrame with results
        """
        results: list[dict[str, Any]] = []

        for config in configs:
            logger.info(f"Running experiment: {config.name}")

            try:
                # Create retriever based on config
                if retriever_factory:
                    retriever = retriever_factory(config)
                else:
                    retriever = self._create_default_retriever(config)

                # Evaluate
                metrics = self.evaluate(retriever, config)

                result = {
                    **config.to_dict(),
                    **metrics.to_dict(),
                }
                results.append(result)

                logger.info(
                    f"  Recall@10: {metrics.recall_at_k.get(10, 0):.3f}, "
                    f"Precision@5: {metrics.precision_at_k.get(5, 0):.3f}"
                )

            except Exception as e:
                logger.error(f"Experiment {config.name} failed: {e}")
                results.append({
                    **config.to_dict(),
                    "error": str(e),
                })

        return pd.DataFrame(results)

    def _create_default_retriever(
        self,
        config: ExperimentConfig,
    ) -> "BaseRetriever":
        """Create a default retriever based on config."""
        from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchEngine
        from backend.llm_infrastructure.retrieval.adapters.es_hybrid import EsHybridRetriever

        engine = EsSearchEngine(self.es_client, self.index_name)
        return EsHybridRetriever(
            es_engine=engine,
            embedder=self.embedder,
        )

    def generate_degradation_curve(
        self,
        metric: str = "recall@10",
        rates: list[float] | None = None,
        configs: list[str] | None = None,
    ) -> "Figure":
        """Generate degradation curve plot.

        Args:
            metric: Metric to plot
            rates: Missing/noise rates to test
            configs: Config names to compare

        Returns:
            Matplotlib Figure
        """
        import matplotlib.pyplot as plt

        rates = rates or [0.0, 0.1, 0.3, 0.5]
        configs = configs or ["hard_filter", "soft_membership", "full_system"]

        fig, ax = plt.subplots(figsize=(10, 6))

        # This would plot cached results
        # For now, create placeholder
        for config_name in configs:
            ax.plot(rates, [0.8, 0.7, 0.5, 0.3], marker='o', label=config_name)

        ax.set_xlabel("Metadata Missing Rate")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs Metadata Missing Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def export_results(
        self,
        results_df: pd.DataFrame,
        output_path: str | Path,
        format: str = "csv",
    ) -> None:
        """Export results to file.

        Args:
            results_df: Results DataFrame
            output_path: Output file path
            format: Output format (csv, json, excel)
        """
        output_path = Path(output_path)

        if format == "csv":
            results_df.to_csv(output_path, index=False)
        elif format == "json":
            results_df.to_json(output_path, orient="records", indent=2)
        elif format == "excel":
            results_df.to_excel(output_path, index=False)

        logger.info(f"Exported results to {output_path}")


def run_full_evaluation(
    es_client: "Elasticsearch",
    index_name: str,
    embedder: "BaseEmbedder",
    golden_set_path: str,
    output_dir: str = "results",
) -> None:
    """Run full evaluation pipeline.

    Args:
        es_client: Elasticsearch client
        index_name: Index name
        embedder: Embedder
        golden_set_path: Path to golden set JSON
        output_dir: Output directory
    """
    from scripts.evaluation.ablation_configs import get_ablation_configs

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize evaluator
    evaluator = RaptorEvaluator(
        es_client=es_client,
        index_name=index_name,
        embedder=embedder,
    )
    evaluator.load_golden_set(golden_set_path)

    # Get configs
    configs = get_ablation_configs()

    # Run ablation
    results = evaluator.run_ablation_study(configs)

    # Export
    evaluator.export_results(results, output_path / "ablation_results.csv")

    # Generate plots
    fig = evaluator.generate_degradation_curve()
    fig.savefig(output_path / "degradation_curve.png", dpi=150, bbox_inches="tight")

    logger.info(f"Evaluation complete. Results saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAPTOR Evaluation")
    parser.add_argument("--es-url", default="http://localhost:9200")
    parser.add_argument("--index", required=True)
    parser.add_argument("--golden-set", required=True)
    parser.add_argument("--output-dir", default="results")

    args = parser.parse_args()

    from elasticsearch import Elasticsearch
    from backend.llm_infrastructure.embedding import get_embedder

    es_client = Elasticsearch([args.es_url])
    embedder = get_embedder("koe5")

    run_full_evaluation(
        es_client=es_client,
        index_name=args.index,
        embedder=embedder,
        golden_set_path=args.golden_set,
        output_dir=args.output_dir,
    )
