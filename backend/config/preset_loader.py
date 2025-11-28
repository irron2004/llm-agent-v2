"""Preset configuration loader for retrieval pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PRESET_DIR = Path(__file__).parent / "presets"


def load_preset(name: str) -> dict[str, Any]:
    """Load a preset configuration by name.

    Args:
        name: Preset name (with or without .yaml extension)

    Returns:
        Dictionary containing preset configuration

    Raises:
        FileNotFoundError: If preset file doesn't exist
        yaml.YAMLError: If preset file is invalid
    """
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"

    preset_path = PRESET_DIR / name
    if not preset_path.exists():
        available = [p.stem for p in PRESET_DIR.glob("*.yaml")]
        raise FileNotFoundError(
            f"Preset '{name}' not found. Available presets: {available}"
        )

    with preset_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_presets() -> list[str]:
    """List all available preset names."""
    if not PRESET_DIR.exists():
        return []
    return [p.stem for p in PRESET_DIR.glob("*.yaml")]


def get_retrieval_config(preset_name: str) -> dict[str, Any]:
    """Extract retrieval-specific config from preset.

    Returns:
        Dict with keys: method, version, top_k, dense_weight, sparse_weight, rrf_k
    """
    preset = load_preset(preset_name)

    retrieval = preset.get("retrieval", {})
    hybrid = preset.get("hybrid", {})
    dense = preset.get("dense", {})
    sparse = preset.get("sparse", {})

    return {
        "method": retrieval.get("method"),
        "version": retrieval.get("version"),
        "top_k": retrieval.get("top_k"),
        "dense_weight": hybrid.get("dense_weight"),
        "sparse_weight": hybrid.get("sparse_weight"),
        "rrf_k": hybrid.get("rrf_k"),
        "similarity_threshold": dense.get("similarity_threshold"),
    }


__all__ = ["load_preset", "list_presets", "get_retrieval_config"]
