"""SentenceTransformer-based embedders."""

from typing import Any
import numpy as np
import numpy.typing as npt

from ..base import BaseEmbedder
from ..registry import register_embedder


@register_embedder("bge_base", version="v1")
class SentenceTransformerEmbedder(BaseEmbedder):
    """BGE-base embedding model using SentenceTransformers.

    Config options:
        model_name: str = "BAAI/bge-base-en-v1.5" - HuggingFace model ID
        device: str = "cuda" - Device to use
        normalize_embeddings: bool = True - L2 normalize embeddings
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Import here to make it optional
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        model_name = self.config.get("model_name", "BAAI/bge-base-en-v1.5")
        device = self.config.get("device", "cuda")

        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = self.config.get("normalize_embeddings", True)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        """Embed a single text."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embedding

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> npt.NDArray[np.float32]:
        """Embed multiple texts in batches."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings


@register_embedder("bge_large", version="v1")
class BGELargeEmbedder(SentenceTransformerEmbedder):
    """BGE-large embedding model."""

    def __init__(self, **kwargs: Any) -> None:
        # Override default model_name
        kwargs.setdefault("model_name", "BAAI/bge-large-en-v1.5")
        super().__init__(**kwargs)


@register_embedder("multilingual_e5", version="v1")
class MultilingualE5Embedder(SentenceTransformerEmbedder):
    """Multilingual E5 embedding model."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("model_name", "intfloat/multilingual-e5-large")
        super().__init__(**kwargs)
