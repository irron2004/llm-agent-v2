"""Preprocessing adapters.

Import all adapters here to ensure they are registered.
"""

from .standard import StandardPreprocessor
from .domain_specific import DomainSpecificPreprocessor
from .normalize import NormalizationPreprocessor

__all__ = [
    "StandardPreprocessor",
    "DomainSpecificPreprocessor",
    "NormalizationPreprocessor",
]
