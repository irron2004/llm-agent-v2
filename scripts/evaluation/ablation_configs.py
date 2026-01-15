"""Ablation study configurations for RAPTOR evaluation.

This module defines experiment configurations for:
1. Baseline comparisons
2. Component ablation
3. Metadata degradation tests
4. Hyperparameter sensitivity analysis

Usage:
    from scripts.evaluation.ablation_configs import (
        get_ablation_configs,
        get_degradation_configs,
        get_baseline_configs,
    )

    configs = get_ablation_configs()
    evaluator.run_ablation_study(configs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from scripts.evaluation.raptor_evaluation import ExperimentConfig


# =============================================================================
# Baseline Configurations
# =============================================================================

def get_baseline_configs() -> list[ExperimentConfig]:
    """Get baseline experiment configurations.

    Baselines:
    1. Flat Hybrid: No meta, no RAPTOR, just dense+BM25
    2. Hard Meta Filter: Filter by meta before flat retrieval
    3. Global RAPTOR: RAPTOR without meta partitioning
    4. Meta + Flat: Meta partitioning without RAPTOR trees

    Returns:
        List of baseline configs
    """
    return [
        ExperimentConfig(
            name="flat_hybrid",
            use_partitioning=False,
            use_raptor_tree=False,
            use_soft_membership=False,
            use_novelty_detection=False,
            retriever_type="es_hybrid",
        ),
        ExperimentConfig(
            name="hard_meta_filter",
            use_partitioning=True,
            use_raptor_tree=False,
            use_soft_membership=False,
            use_novelty_detection=False,
            retriever_type="es_hybrid",  # With metadata filter
        ),
        ExperimentConfig(
            name="global_raptor",
            use_partitioning=False,
            use_raptor_tree=True,
            use_soft_membership=False,
            use_novelty_detection=False,
            retriever_type="raptor_hierarchical",
        ),
        ExperimentConfig(
            name="meta_flat",
            use_partitioning=True,
            use_raptor_tree=False,
            use_soft_membership=False,
            use_novelty_detection=False,
            retriever_type="raptor_flat",
        ),
        ExperimentConfig(
            name="meta_raptor_hard",
            use_partitioning=True,
            use_raptor_tree=True,
            use_soft_membership=False,
            use_novelty_detection=False,
            retriever_type="raptor_hierarchical",
        ),
        ExperimentConfig(
            name="full_system",
            use_partitioning=True,
            use_raptor_tree=True,
            use_soft_membership=True,
            use_novelty_detection=True,
            retriever_type="raptor_hierarchical",
        ),
    ]


# =============================================================================
# Ablation Configurations
# =============================================================================

def get_ablation_configs() -> list[ExperimentConfig]:
    """Get ablation study configurations.

    Tests each component's contribution by removing it.

    Returns:
        List of ablation configs
    """
    return [
        # Baseline: all components on
        ExperimentConfig(
            name="full_system",
            use_partitioning=True,
            use_raptor_tree=True,
            use_soft_membership=True,
            use_novelty_detection=True,
        ),
        # Remove partitioning
        ExperimentConfig(
            name="no_partitioning",
            use_partitioning=False,
            use_raptor_tree=True,
            use_soft_membership=True,
            use_novelty_detection=True,
        ),
        # Remove RAPTOR trees
        ExperimentConfig(
            name="no_raptor",
            use_partitioning=True,
            use_raptor_tree=False,
            use_soft_membership=True,
            use_novelty_detection=True,
        ),
        # Remove soft membership
        ExperimentConfig(
            name="no_soft_membership",
            use_partitioning=True,
            use_raptor_tree=True,
            use_soft_membership=False,
            use_novelty_detection=True,
        ),
        # Remove novelty detection
        ExperimentConfig(
            name="no_novelty",
            use_partitioning=True,
            use_raptor_tree=True,
            use_soft_membership=True,
            use_novelty_detection=False,
        ),
        # Only partitioning
        ExperimentConfig(
            name="partition_only",
            use_partitioning=True,
            use_raptor_tree=False,
            use_soft_membership=False,
            use_novelty_detection=False,
        ),
        # Partitioning + RAPTOR
        ExperimentConfig(
            name="partition_raptor",
            use_partitioning=True,
            use_raptor_tree=True,
            use_soft_membership=False,
            use_novelty_detection=False,
        ),
        # Soft without novelty
        ExperimentConfig(
            name="soft_no_novelty",
            use_partitioning=True,
            use_raptor_tree=True,
            use_soft_membership=True,
            use_novelty_detection=False,
        ),
    ]


# =============================================================================
# Metadata Degradation Configurations
# =============================================================================

def get_degradation_configs() -> list[ExperimentConfig]:
    """Get metadata degradation test configurations.

    Tests robustness to metadata missing and noise.

    Returns:
        List of degradation configs
    """
    configs = []

    # Missing metadata tests
    for rate in [0.0, 0.1, 0.3, 0.5, 0.7]:
        configs.append(
            ExperimentConfig(
                name=f"missing_{int(rate*100)}pct",
                use_partitioning=True,
                use_raptor_tree=True,
                use_soft_membership=True,
                use_novelty_detection=True,
                metadata_missing_rate=rate,
            )
        )

    # Noise tests
    for rate in [0.0, 0.1, 0.3, 0.5]:
        configs.append(
            ExperimentConfig(
                name=f"noise_{int(rate*100)}pct",
                use_partitioning=True,
                use_raptor_tree=True,
                use_soft_membership=True,
                use_novelty_detection=True,
                metadata_noise_rate=rate,
            )
        )

    # Combined
    configs.append(
        ExperimentConfig(
            name="missing_30_noise_10",
            use_partitioning=True,
            use_raptor_tree=True,
            use_soft_membership=True,
            use_novelty_detection=True,
            metadata_missing_rate=0.3,
            metadata_noise_rate=0.1,
        )
    )

    return configs


# =============================================================================
# Hyperparameter Sensitivity Configurations
# =============================================================================

@dataclass
class HyperparamConfig:
    """Hyperparameter configuration for sensitivity analysis.

    Attributes:
        name: Configuration name
        beta: Semantic similarity weight
        alpha_device: Device name match weight
        alpha_doc_type: Doc type match weight
        novelty_threshold: Novelty detection threshold
        escape_threshold: Soft escape threshold
        top_k: Soft routing top-k
        max_levels: RAPTOR tree depth
        global_fallback_weight: Global fallback weight
    """

    name: str
    beta: float = 1.0
    alpha_device: float = 0.8
    alpha_doc_type: float = 0.5
    novelty_threshold: float = 0.3
    escape_threshold: float = 0.5
    top_k: int = 3
    max_levels: int = 3
    global_fallback_weight: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "beta": self.beta,
            "alpha_device": self.alpha_device,
            "alpha_doc_type": self.alpha_doc_type,
            "novelty_threshold": self.novelty_threshold,
            "escape_threshold": self.escape_threshold,
            "top_k": self.top_k,
            "max_levels": self.max_levels,
            "global_fallback_weight": self.global_fallback_weight,
        }


def get_hyperparameter_configs() -> list[HyperparamConfig]:
    """Get hyperparameter sensitivity test configurations.

    Tests impact of key hyperparameters.

    Returns:
        List of hyperparameter configs
    """
    configs = []

    # Beta (semantic weight) sensitivity
    for beta in [0.5, 1.0, 1.5, 2.0]:
        configs.append(HyperparamConfig(name=f"beta_{beta}", beta=beta))

    # Alpha device sensitivity
    for alpha in [0.2, 0.5, 0.8, 1.0]:
        configs.append(HyperparamConfig(name=f"alpha_device_{alpha}", alpha_device=alpha))

    # Novelty threshold sensitivity
    for threshold in [0.2, 0.3, 0.4, 0.5]:
        configs.append(
            HyperparamConfig(name=f"novelty_{threshold}", novelty_threshold=threshold)
        )

    # Escape threshold sensitivity
    for threshold in [0.3, 0.5, 0.7]:
        configs.append(
            HyperparamConfig(name=f"escape_{threshold}", escape_threshold=threshold)
        )

    # Top-k sensitivity (precision vs recall tradeoff)
    for k in [2, 3, 5]:
        configs.append(HyperparamConfig(name=f"top_k_{k}", top_k=k))

    # Max levels sensitivity
    for levels in [2, 3, 4]:
        configs.append(HyperparamConfig(name=f"levels_{levels}", max_levels=levels))

    # Global fallback sensitivity
    for weight in [0.05, 0.1, 0.2, 0.3]:
        configs.append(
            HyperparamConfig(name=f"fallback_{weight}", global_fallback_weight=weight)
        )

    return configs


# =============================================================================
# All Configurations Combined
# =============================================================================

def get_all_configs() -> dict[str, list[ExperimentConfig | HyperparamConfig]]:
    """Get all experiment configurations organized by type.

    Returns:
        Dictionary with config lists by category
    """
    return {
        "baselines": get_baseline_configs(),
        "ablation": get_ablation_configs(),
        "degradation": get_degradation_configs(),
        "hyperparameters": get_hyperparameter_configs(),
    }


# =============================================================================
# Configuration Descriptions (for paper/documentation)
# =============================================================================

EXPERIMENT_DESCRIPTIONS = {
    "flat_hybrid": "Baseline: Dense+BM25 hybrid without metadata or hierarchy",
    "hard_meta_filter": "Filter by metadata, then flat retrieval",
    "global_raptor": "RAPTOR tree without metadata partitioning",
    "meta_flat": "Metadata partitioning without RAPTOR trees",
    "meta_raptor_hard": "Metadata + RAPTOR without soft membership",
    "full_system": "Complete system: Meta + RAPTOR + Soft + Novelty",
    "missing_*pct": "Simulated metadata missing at various rates",
    "noise_*pct": "Simulated metadata noise (swapped values)",
}


# =============================================================================
# Paper Figure Configurations
# =============================================================================

def get_paper_figure_configs() -> dict[str, list[ExperimentConfig]]:
    """Get configurations for paper figures.

    Returns:
        Dictionary with configs for each figure
    """
    return {
        # Figure 1: Ablation bar chart
        "ablation_bar": [
            ExperimentConfig(name="full_system", use_partitioning=True, use_raptor_tree=True, use_soft_membership=True, use_novelty_detection=True),
            ExperimentConfig(name="no_soft", use_partitioning=True, use_raptor_tree=True, use_soft_membership=False, use_novelty_detection=True),
            ExperimentConfig(name="no_raptor", use_partitioning=True, use_raptor_tree=False, use_soft_membership=True, use_novelty_detection=True),
            ExperimentConfig(name="no_partition", use_partitioning=False, use_raptor_tree=True, use_soft_membership=True, use_novelty_detection=True),
            ExperimentConfig(name="baseline", use_partitioning=False, use_raptor_tree=False, use_soft_membership=False, use_novelty_detection=False),
        ],

        # Figure 2: Metadata missing degradation curve
        "missing_curve": [
            ExperimentConfig(name=f"full_{r}", use_partitioning=True, use_raptor_tree=True, use_soft_membership=True, use_novelty_detection=True, metadata_missing_rate=r)
            for r in [0.0, 0.1, 0.3, 0.5, 0.7]
        ] + [
            ExperimentConfig(name=f"hard_{r}", use_partitioning=True, use_raptor_tree=True, use_soft_membership=False, use_novelty_detection=False, metadata_missing_rate=r)
            for r in [0.0, 0.1, 0.3, 0.5, 0.7]
        ],

        # Figure 3: Metadata noise degradation curve
        "noise_curve": [
            ExperimentConfig(name=f"full_{r}", use_partitioning=True, use_raptor_tree=True, use_soft_membership=True, use_novelty_detection=True, metadata_noise_rate=r)
            for r in [0.0, 0.1, 0.3, 0.5]
        ] + [
            ExperimentConfig(name=f"hard_{r}", use_partitioning=True, use_raptor_tree=True, use_soft_membership=False, use_novelty_detection=False, metadata_noise_rate=r)
            for r in [0.0, 0.1, 0.3, 0.5]
        ],

        # Figure 4: Baseline comparison
        "baselines": get_baseline_configs(),
    }


__all__ = [
    "ExperimentConfig",
    "HyperparamConfig",
    "get_baseline_configs",
    "get_ablation_configs",
    "get_degradation_configs",
    "get_hyperparameter_configs",
    "get_all_configs",
    "get_paper_figure_configs",
    "EXPERIMENT_DESCRIPTIONS",
]
