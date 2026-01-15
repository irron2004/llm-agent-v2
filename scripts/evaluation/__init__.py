"""RAPTOR evaluation scripts.

This package provides evaluation tools for the Meta-guided Hierarchical RAG system:
- raptor_evaluation: Evaluation framework with metrics computation
- ablation_configs: Experiment configurations for ablation studies

Usage:
    from scripts.evaluation import (
        RaptorEvaluator,
        get_ablation_configs,
        get_baseline_configs,
    )

    evaluator = RaptorEvaluator(es_client, index_name, embedder)
    configs = get_ablation_configs()
    results = evaluator.run_ablation_study(configs)
"""

from scripts.evaluation.ablation_configs import (
    EXPERIMENT_DESCRIPTIONS,
    ExperimentConfig,
    HyperparamConfig,
    get_ablation_configs,
    get_all_configs,
    get_baseline_configs,
    get_degradation_configs,
    get_hyperparameter_configs,
    get_paper_figure_configs,
)
from scripts.evaluation.raptor_evaluation import (
    GoldenQuery,
    RaptorEvaluator,
    RetrievalMetrics,
    run_full_evaluation,
)

__all__ = [
    # Evaluator
    "RaptorEvaluator",
    "GoldenQuery",
    "RetrievalMetrics",
    "run_full_evaluation",
    # Configs
    "ExperimentConfig",
    "HyperparamConfig",
    "get_ablation_configs",
    "get_baseline_configs",
    "get_degradation_configs",
    "get_hyperparameter_configs",
    "get_all_configs",
    "get_paper_figure_configs",
    "EXPERIMENT_DESCRIPTIONS",
]
