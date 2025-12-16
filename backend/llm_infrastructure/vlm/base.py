"""Base class for VLM (Vision-Language Model) clients.

All VLM clients must implement the BaseVlmClient interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image


class BaseVlmClient(ABC):
    """Abstract base class for VLM clients.

    VLM clients are used to interact with vision-language models for tasks like
    OCR, image description, and document parsing.

    Subclasses must implement the `generate` method.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the VLM client.

        Args:
            **kwargs: Configuration options for the client.
        """
        self.config = kwargs

    @abstractmethod
    def generate(
        self,
        image: "Image.Image | bytes | str",
        prompt: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Generate text from an image and prompt.

        Args:
            image: Input image as PIL Image, bytes, or file path.
            prompt: Text prompt describing the task.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 for deterministic).
            **kwargs: Additional model-specific parameters.

        Returns:
            Generated text response from the model.

        Raises:
            VlmClientError: If generation fails.
        """
        raise NotImplementedError

    def get_model_name(self) -> str:
        """Return the model name/identifier.

        Returns:
            Model name string.
        """
        return self.config.get("model", "unknown")


class VlmClientError(Exception):
    """Base exception for VLM client errors."""

    pass


class VlmConnectionError(VlmClientError):
    """Raised when connection to VLM server fails."""

    pass


class VlmGenerationError(VlmClientError):
    """Raised when text generation fails."""

    pass
