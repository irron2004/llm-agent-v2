"""Embedding service: registry 기반 임베딩 호출."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from backend.config.settings import rag_settings
from backend.llm_infrastructure.embedding.registry import get_embedder


class EmbeddingService:
    """질문/문서 임베딩을 공통으로 처리하는 서비스."""

    def __init__(
        self,
        *,
        method: Optional[str] = None,
        version: Optional[str] = None,
        device: Optional[str] = None,
        use_cache: Optional[bool] = None,
        cache_dir: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.method = method or rag_settings.embedding_method
        self.version = version or rag_settings.embedding_version
        self.device = device or rag_settings.embedding_device
        self.use_cache = use_cache if use_cache is not None else rag_settings.embedding_use_cache
        self.cache_dir = cache_dir or rag_settings.embedding_cache_dir
        self.model_name = model_name
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder(
                self.method,
                version=self.version,
                device=self.device,
                use_cache=self.use_cache,
                cache_dir=self.cache_dir,
                model_name=self.model_name,
            )
        return self._embedder

    def embed_query(self, text: str) -> np.ndarray:
        """단일 텍스트 임베딩."""
        return self._get_embedder().embed(text)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """여러 텍스트 임베딩."""
        return self._get_embedder().embed_batch(list(texts))

    def dimension(self) -> int:
        """임베딩 차원."""
        return self._get_embedder().get_dimension()

    def get_raw_embedder(self):
        """레지스트리에서 생성된 실제 임베더 인스턴스를 반환한다."""
        return self._get_embedder()


__all__ = ["EmbeddingService"]
