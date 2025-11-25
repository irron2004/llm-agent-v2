"""SentenceTransformer adapter registered in the embedder registry."""

from __future__ import annotations

from typing import Iterable, Any

import numpy as np

from ..base import BaseEmbedder
from ..registry import register_embedder
from ..engines.sentence import create_embedder


DEFAULT_MODELS = {
    # alias : hf_name
    "koe5": "nlpai-lab/KoE5",
    "multilingual_e5": "intfloat/multilingual-e5-large",
    "bge_base": "BAAI/bge-base-en-v1.5",
}


@register_embedder("koe5", version="v1")
@register_embedder("multilingual_e5", version="v1")
@register_embedder("bge_base", version="v1")
class SentenceEmbedderAdapter(BaseEmbedder):
    """레지스트리용 SentenceTransformer 어댑터."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        normalize_embeddings: bool = True,
        use_cache: bool = False,
        cache_dir: str = ".cache/embeddings",
        show_progress_bar: bool = False,
        alias: str | None = None,
        **kwargs: Any,
    ) -> None:
        # alias는 레지스트리에서 전달되는 이름(koe5, bge_base 등)
        requested = alias
        # alias가 없으면 기본값(koe5)로 사용, model_name이 우선
        resolved_model = model_name or DEFAULT_MODELS.get(
            requested, DEFAULT_MODELS["koe5"]
        )

        super().__init__(
            model_name=resolved_model,
            device=device,
            normalize_embeddings=normalize_embeddings,
            use_cache=use_cache,
            cache_dir=cache_dir,
            show_progress_bar=show_progress_bar,
            alias=requested,
            **kwargs,
        )
        self.engine = create_embedder(
            typ="sentence",
            model_name=resolved_model,
            device=device,
            normalize_embeddings=normalize_embeddings,
            use_cache=use_cache,
            cache_dir=cache_dir,
            show_progress_bar=show_progress_bar,
        )
        self.dimension = self.engine.get_dimension()

    def embed(self, text: str) -> np.ndarray:
        return self.engine.encode([text])[0]

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        # batch_size는 SentenceTransformer 내부에서 처리되므로 전달하지 않음
        return self.engine.encode(texts)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """레지스트리 BaseEmbedder 호환 편의 메서드."""
        return self.engine.encode(texts)


__all__ = ["SentenceEmbedderAdapter"]
