"""Base classes for embedding."""

from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np
import numpy.typing as npt


class BaseEmbedder(ABC):
    """Base class for all embedding methods.

    Each embedding method should:
    1. Inherit from this class
    2. Implement the embed() and embed_batch() methods
    3. Register itself using @register_embedder decorator

    Example:
        ```python
        from .registry import register_embedder

        @register_embedder("my_embedder", version="v1")
        class MyEmbedder(BaseEmbedder):
            def embed(self, text):
                return np.random.rand(768)  # Example

            def embed_batch(self, texts):
                return [self.embed(t) for t in texts]
        ```
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize embedder with optional config."""
        self.config = kwargs
        self.dimension: int | None = None

    @abstractmethod
    def embed(self, text: str) -> npt.NDArray[np.float32]:
        """Embed a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector (1D numpy array)
        """
        raise NotImplementedError

    @abstractmethod
    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> npt.NDArray[np.float32]:
        """Embed multiple texts in batches.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            Embedding matrix (2D numpy array: [n_texts, dimension])
        """
        raise NotImplementedError

    def get_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            Embedding vector dimension
        """
        if self.dimension is None:
            # Infer dimension by embedding empty string
            sample_embedding = self.embed("")
            self.dimension = len(sample_embedding)
        return self.dimension

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
