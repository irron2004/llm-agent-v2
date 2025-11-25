"""Text Embeddings Inference (TEI) client embedder."""

from typing import Any
import numpy as np
import numpy.typing as npt

from ..base import BaseEmbedder
from ..registry import register_embedder


@register_embedder("tei", version="v1")
class TEIEmbedder(BaseEmbedder):
    """TEI (Text Embeddings Inference) server client.

    Config options:
        endpoint_url: str - TEI server URL (e.g., "http://tei:80")
        timeout: int = 30 - Request timeout in seconds
        normalize: bool = True - L2 normalize embeddings
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx not installed. Install with: pip install httpx"
            )

        self.endpoint_url = self.config.get("endpoint_url")
        if not self.endpoint_url:
            raise ValueError("endpoint_url is required for TEI embedder")

        self.timeout = self.config.get("timeout", 30)
        self.normalize = self.config.get("normalize", True)
        self.client = httpx.Client(timeout=self.timeout)

        # Infer dimension from server
        self.dimension = self._get_dimension_from_server()

    def _get_dimension_from_server(self) -> int:
        """Get embedding dimension from TEI server."""
        # Embed a short test string to get dimension
        test_embedding = self.embed("test")
        return len(test_embedding)

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        """Embed a single text."""
        response = self.client.post(
            f"{self.endpoint_url}/embed",
            json={"inputs": text, "normalize": self.normalize},
        )
        response.raise_for_status()
        embedding = np.array(response.json(), dtype=np.float32)
        return embedding

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> npt.NDArray[np.float32]:
        """Embed multiple texts in batches."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = self.client.post(
                f"{self.endpoint_url}/embed",
                json={"inputs": batch, "normalize": self.normalize},
            )
            response.raise_for_status()

            batch_embeddings = np.array(response.json(), dtype=np.float32)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "client"):
            self.client.close()
