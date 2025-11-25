"""Normalization engine exports (L0~L5)."""

from .base import (
    NormLevel,
    STOP_VARIANTS,
    VARIANT_REWRITE,
    clean_variants,
    clean_variants_fast,
    normalize_text,
    sanitize_variant_map,
)
from .domain import (
    prejoin_domain_terms,
    preprocess_l4_advanced_domain,
    preprocess_l5_enhanced_domain,
    preprocess_semiconductor_domain,
)
from .factory import (
    build_normalizer,
    build_normalizer_by_level,
    build_normalizers_by_level,
    build_normalizer_simple,
    dump_normalization_log,
    dump_normalization_log_simple,
)
from .utils import build_vocabulary, create_buckets, tokenize_ko_en

__all__ = [
    "NormLevel",
    "STOP_VARIANTS",
    "VARIANT_REWRITE",
    "normalize_text",
    "clean_variants",
    "clean_variants_fast",
    "sanitize_variant_map",
    "prejoin_domain_terms",
    "preprocess_semiconductor_domain",
    "preprocess_l4_advanced_domain",
    "preprocess_l5_enhanced_domain",
    "build_normalizer",
    "build_normalizer_by_level",
    "build_normalizers_by_level",
    "build_normalizer_simple",
    "dump_normalization_log",
    "dump_normalization_log_simple",
    "build_vocabulary",
    "create_buckets",
    "tokenize_ko_en",
]
