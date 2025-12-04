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
from .registry import (
    PreprocessorRegistry,
    get_preprocessor,
    register_preprocessor,
)

__all__ = [
    "StandardPreprocessor",
    "DomainSpecificPreprocessor",
    "NormalizationPreprocessor",
    "PreprocessorRegistry",
    "get_preprocessor",
    "register_preprocessor",
]

# Parsers (PDF, etc.)
from .parsers import (
    DeepDocPdfAdapter,
    DeepSeekVLPdfAdapter,
    PlainPdfAdapter,
    VlmPdfAdapter,
)  # noqa: E402,F401

# Backward-compatible aliases
DeepDocPdfParser = DeepDocPdfAdapter
PlainPdfParser = PlainPdfAdapter
DeepSeekVLPdfParser = DeepSeekVLPdfAdapter
VlmPdfParser = VlmPdfAdapter

__all__ += [
    "DeepDocPdfAdapter",
    "DeepSeekVLPdfAdapter",
    "VlmPdfAdapter",
    "PlainPdfAdapter",
    "DeepDocPdfParser",
    "DeepSeekVLPdfParser",
    "VlmPdfParser",
    "PlainPdfParser",
]
