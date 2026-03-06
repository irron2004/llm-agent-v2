"""SentenceTransformer embedding engine."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from ...base import BaseEmbedder
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
        trust_remote_code: bool = False,
        truncate_dim: Optional[int] = None,
        max_seq_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers가 필요합니다. `pip install sentence-transformers` 로 설치하세요."
            ) from exc

        self.device = pick_device(device)
        self.model_name = model_name
        st_kwargs: dict[str, Any] = {"device": self.device}
        if trust_remote_code:
            st_kwargs["trust_remote_code"] = True
            st_kwargs["model_kwargs"] = {"trust_remote_code": True}
        if truncate_dim is not None:
            st_kwargs["truncate_dim"] = truncate_dim
        self.model = SentenceTransformer(model_name, **st_kwargs)
        if max_seq_length is not None:
            self.model.max_seq_length = int(max_seq_length)
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar
        self.truncate_dim = truncate_dim
        self.max_seq_length = max_seq_length

        self.cache: Optional[EmbeddingCache] = None
        if use_cache:
            self.cache = EmbeddingCache(cache_dir)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string."""
        return self.encode([text])[0]

    def _embedding_dim(self) -> int:
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError("Could not determine sentence embedding dimension")
        return int(dim)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of strings."""
        return self.encode(texts, batch_size=batch_size)

    def encode(self, texts: Iterable[str], batch_size: Optional[int] = None) -> np.ndarray:
        texts_list = list(texts)
        if not texts_list:
            dim = self._embedding_dim()
            return np.zeros((0, dim), dtype=np.float32)

        # 캐시 조회
        if self.cache:
            hit = self.cache.get(texts_list, self.model_name, self.normalize_embeddings)
            if hit is not None:
                return hit

        if batch_size is None:
            vecs = self.model.encode(
                texts_list,
                normalize_embeddings=False,
                convert_to_numpy=True,
                show_progress_bar=self.show_progress_bar,
            )
        else:
            vecs = self.model.encode(
                texts_list,
                batch_size=int(batch_size),
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
        return self._embedding_dim()


__all__ = ["SentenceTransformerEmbedder"]
