"""Disk caching for embeddings."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .utils import make_cache_key


class EmbeddingCache:
    """디스크 캐시 래퍼(diskcache 기반)."""

    def __init__(self, cache_dir: str = ".cache/embeddings") -> None:
        try:
            import diskcache  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "diskcache가 필요합니다. `pip install diskcache` 로 설치하세요."
            ) from exc

        self.cache_path = Path(cache_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(str(self.cache_path))

    def get(self, texts: Iterable[str], model_name: str, normalize: bool) -> Optional[np.ndarray]:
        key = make_cache_key(texts, model_name, normalize)
        hit = self._cache.get(key)
        if hit is None:
            return None
        if isinstance(hit, bytes):
            return pickle.loads(hit)
        return hit

    def set(self, texts: Iterable[str], model_name: str, normalize: bool, value: np.ndarray) -> None:
        key = make_cache_key(texts, model_name, normalize)
        try:
            payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            payload = value
        self._cache.set(key, payload)


__all__ = ["EmbeddingCache"]
