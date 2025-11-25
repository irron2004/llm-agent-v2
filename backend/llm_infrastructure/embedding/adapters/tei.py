"""Text Embeddings Inference (TEI) client embedder (adapter)."""

from __future__ import annotations

from typing import Any, List

import numpy as np
import numpy.typing as npt

from ..base import BaseEmbedder
from ..registry import register_embedder


@register_embedder("tei", version="v1")
class TEIEmbedder(BaseEmbedder):
    """TEI (Text Embeddings Inference) 서버 클라이언트."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        try:
            import httpx  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "httpx가 필요합니다. `pip install httpx` 로 설치하세요."
            ) from exc

        self.endpoint_url = self.config.get("endpoint_url")
        if not self.endpoint_url:
            raise ValueError("endpoint_url is required for TEI embedder")

        self.timeout = self.config.get("timeout", 30)
        self.normalize = self.config.get("normalize", True)
        self.client = httpx.Client(timeout=self.timeout)

        # Dimension은 지연 초기화
        self._dimension: int | None = None

    def _infer_dimension(self) -> int:
        test_embedding = self.embed("test")
        return int(test_embedding.shape[-1])

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        response = self.client.post(
            f"{self.endpoint_url}/embed",
            json={"inputs": text, "normalize": self.normalize},
        )
        response.raise_for_status()
        embedding = np.array(response.json(), dtype=np.float32)
        if self._dimension is None:
            self._dimension = embedding.shape[-1]
        return embedding

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> npt.NDArray[np.float32]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.post(
                f"{self.endpoint_url}/embed",
                json={"inputs": batch, "normalize": self.normalize},
            )
            response.raise_for_status()
            batch_embeddings = np.array(response.json(), dtype=np.float32)
            all_embeddings.append(batch_embeddings)
        stacked = np.vstack(all_embeddings) if all_embeddings else np.zeros((0, 0), dtype=np.float32)
        if self._dimension is None and stacked.size > 0:
            self._dimension = stacked.shape[-1]
        return stacked

    def get_dimension(self) -> int:
        if self._dimension is None:
            self._dimension = self._infer_dimension()
        return self._dimension

    def __del__(self):
        if hasattr(self, "client"):
            try:
                self.client.close()
            except Exception:
                pass


__all__ = ["TEIEmbedder"]
