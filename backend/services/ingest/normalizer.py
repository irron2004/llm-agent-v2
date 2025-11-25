"""Factory for ingest-time normalizer selection."""

from __future__ import annotations

from typing import Callable, Optional

from ...llm_infrastructure.preprocessing.normalize_engine import build_normalizer


def get_normalizer(
    level: str = "L3",
    variant_map: Optional[dict] = None,
    keep_newlines: bool = True,
) -> Callable[[str], str]:
    """Return a text normalizer callable for ingest pipelines."""
    return build_normalizer(
        level=level,
        variant_map=variant_map or {},
        keep_newlines=keep_newlines,
    )
