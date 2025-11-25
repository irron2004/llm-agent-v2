"""SentenceTransformer embedding engine."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .base import BaseEmbedder
from .cache import EmbeddingCache
from .utils import l2_normalize, pick_device


class SentenceTransformerEmbedder(BaseEmbedder):
    """SentenceTransformer 엔진 래퍼."""

    def __init__(
        self,
        model_name: str = "nlpai-lab/KoE5",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        use_cache: bool = False,
        cache_dir: str = ".cache/embeddings",
        show_progress_bar: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers가 필요합니다. `pip install sentence-transformers` 로 설치하세요."
            ) from exc

        self.device = pick_device(device)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar

        self.cache: Optional[EmbeddingCache] = None
        if use_cache:
            self.cache = EmbeddingCache(cache_dir)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts_list = list(texts)
        if not texts_list:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)

        # 캐시 조회
        if self.cache:
            hit = self.cache.get(texts_list, self.model_name, self.normalize_embeddings)
            if hit is not None:
                return hit

        vecs = self.model.encode(
            texts_list,
            normalize_embeddings=False,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress_bar,
        )
        if self.normalize_embeddings:
            vecs = l2_normalize(vecs)

        if self.cache:
            self.cache.set(texts_list, self.model_name, self.normalize_embeddings, vecs)

        return vecs

    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


__all__ = ["SentenceTransformerEmbedder"]
