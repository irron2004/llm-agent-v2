"""Base embedder for SentenceTransformer engine (internal use)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


class BaseEmbedder(ABC):
    """엔진 내부용 BaseEmbedder (레지스트리용 BaseEmbedder와 구분)."""

    @abstractmethod
    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """배치 임베딩."""
        ...

    def encode_query(self, text: str) -> np.ndarray:
        """단일 텍스트 임베딩."""
        return self.encode([text])[0]

    def get_dimension(self) -> int:
        """임베딩 차원 조회."""
        vec = self.encode_query(" ")
        return int(vec.shape[-1])


__all__ = ["BaseEmbedder"]
