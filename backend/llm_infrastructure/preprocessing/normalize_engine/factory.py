"""Factory for building text normalizers (L0~L5)."""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

from .base import (
    NormLevel,
    clean_variants,
    clean_variants_fast,
    normalize_text,
    sanitize_variant_map,
)
from .domain import (
        preprocess_l4_advanced_domain,
        preprocess_l5_enhanced_domain,
        preprocess_semiconductor_domain,
)
from .utils import dump_normalization_log, dump_normalization_log_simple

log = logging.getLogger("llm_infrastructure.preprocessing.normalize_engine")


def _attach_profile(fn: Callable[[str], str], *, level: str, keep_newlines: bool, variants: int) -> Callable[[str], str]:
    fn.__safe_profile__ = {  # type: ignore[attr-defined]
        "level": level,
        "sanitized_variants": variants,
        "keep_newlines": keep_newlines,
        "semiconductor_domain": level in {"L3", "L4", "L5"},
        "advanced_entity_extraction": level in {"L4", "L5"},
        "enhanced_variant_mapping": level == "L5",
    }
    return fn


def build_normalizer(
    level: str = "L1",
    variant_map: Optional[Dict[str, str]] = None,
    keep_newlines: bool = True,
) -> Callable[[str], str]:
    """Return a normalizer callable for the given level."""
    level = (level or "L1").upper()
    vm = variant_map or {}
    if level == "L0":
        fn = lambda s: normalize_text(s, keep_newlines=keep_newlines)
    elif level in {"L1", "L2"}:
        fn = lambda s: clean_variants(normalize_text(s, keep_newlines=keep_newlines), vm)
    elif level == "L3":
        fn = lambda s: preprocess_semiconductor_domain(s, vm)
    elif level == "L4":
        fn = lambda s: preprocess_l4_advanced_domain(s, vm)
    elif level == "L5":
        fn = lambda s: preprocess_l5_enhanced_domain(s, vm)
    else:
        raise ValueError(f"Unknown normalization level: {level}")
    return _attach_profile(fn, level=level, keep_newlines=keep_newlines, variants=len(vm))


def build_normalizer_by_level(
    level: str = "L1",
    variant_map: Optional[Dict[str, str]] = None,
    keep_newlines: bool = True,
) -> Callable[[str], str]:
    """Compatibility wrapper; delegates to build_normalizer."""
    return build_normalizer(level=level, variant_map=variant_map, keep_newlines=keep_newlines)


def build_normalizers_by_level(
    variant_map: Optional[Dict[str, str]] = None,
    keep_newlines: bool = True,
) -> Dict[str, Callable[[str], str]]:
    """Return a mapping of L0~L5 level names to normalizer callables."""
    levels = ["L0", "L1", "L2", "L3", "L4", "L5"]
    return {lvl: build_normalizer(lvl, variant_map, keep_newlines) for lvl in levels}


def build_normalizer_simple(
    variant_map: Optional[Dict[str, str]] = None,
    *,
    keep_newlines: bool = True,
) -> Callable[[str], str]:
    """Simplified normalizer: L1-equivalent with optional variant map."""
    vm = variant_map or {}

    def _normalize_one(line: str) -> str:
        t = normalize_text(line, keep_newlines=False)
        if vm:
            t = clean_variants(t, vm)
        return t

    def _normalize(text: str) -> str:
        if not text:
            return ""
        if keep_newlines:
            return "\n".join(_normalize_one(p) for p in text.splitlines())
        return _normalize_one(text)

    return _attach_profile(
        _normalize, level="L1", keep_newlines=keep_newlines, variants=len(vm)
    )


__all__ = [
    "NormLevel",
    "build_normalizer",
    "build_normalizer_by_level",
    "build_normalizers_by_level",
    "build_normalizer_simple",
    "clean_variants_fast",
    "sanitize_variant_map",
    "dump_normalization_log",
    "dump_normalization_log_simple",
]
