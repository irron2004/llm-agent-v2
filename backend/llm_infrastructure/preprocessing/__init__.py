"""Document preprocessing pipeline.

Engine implementations live under `normalize_engine/`.
Adapters (registry-registered preprocessors) live under `adapters/`.
"""

# Ensure adapters register themselves on import
from .adapters import (
    StandardPreprocessor,
    DomainSpecificPreprocessor,
    NormalizationPreprocessor,
)

__all__ = [
    "StandardPreprocessor",
    "DomainSpecificPreprocessor",
    "NormalizationPreprocessor",
]
