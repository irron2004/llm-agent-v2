from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
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


def __getattr__(name: str) -> object:
    if name in {
        "ExperimentConfig",
        "HyperparamConfig",
        "get_ablation_configs",
        "get_baseline_configs",
        "get_degradation_configs",
        "get_hyperparameter_configs",
        "get_all_configs",
        "get_paper_figure_configs",
        "EXPERIMENT_DESCRIPTIONS",
    }:
        from scripts.evaluation import ablation_configs as module

        return cast(object, getattr(module, name))

    if name in {
        "RaptorEvaluator",
        "GoldenQuery",
        "RetrievalMetrics",
        "run_full_evaluation",
    }:
        from scripts.evaluation import raptor_evaluation as module

        return cast(object, getattr(module, name))

    raise AttributeError(name)
