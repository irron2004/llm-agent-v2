"""Embedding adapters for registry."""

# 레지스트리 어댑터 자동 로드
from . import sentence  # noqa: F401
try:
    from . import tei  # noqa: F401
except ImportError:
    pass
try:
    from . import langchain  # noqa: F401
except ImportError:
    pass

__all__ = []
