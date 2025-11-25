"""Factory function for creating embedders."""

from __future__ import annotations

from typing import Any, Optional

from .embedder import SentenceTransformerEmbedder


def create_embedder(
    typ: str = "sentence",
    model_name: str = "nlpai-lab/KoE5",
    device: Optional[str] = None,
    normalize_embeddings: bool = True,
    use_cache: bool = False,
    cache_dir: str = ".cache/embeddings",
    show_progress_bar: bool = False,
    **_: Any,
) -> SentenceTransformerEmbedder:
    """팩토리: typ에 따라 엔진을 생성."""
    if typ in {None, "", "sentence"}:
        return SentenceTransformerEmbedder(
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
            use_cache=use_cache,
            cache_dir=cache_dir,
            show_progress_bar=show_progress_bar,
        )
    raise ValueError(f"Unknown embedder type: {typ}")


__all__ = ["create_embedder"]
