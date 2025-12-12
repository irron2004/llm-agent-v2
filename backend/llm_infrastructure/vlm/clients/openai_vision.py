"""OpenAI Vision API compatible VLM client.

Supports vLLM-served Qwen-VL and other OpenAI-compatible vision models.

Usage:
    from backend.llm_infrastructure.vlm.clients import OpenAIVisionClient

    client = OpenAIVisionClient(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen2-VL-7B-Instruct",
    )
    result = client.generate(image, "Extract text from this image")
"""

from __future__ import annotations

import base64
import io
import logging
from typing import TYPE_CHECKING, Any

import httpx

from backend.llm_infrastructure.vlm.base import (
    BaseVlmClient,
    VlmConnectionError,
    VlmGenerationError,
)
from backend.llm_infrastructure.vlm.registry import register_vlm_client

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


@register_vlm_client("openai_vision", version="v1")
class OpenAIVisionClient(BaseVlmClient):
    """OpenAI Vision API compatible client.

    Connects to vLLM or other OpenAI-compatible servers serving vision models
    like Qwen2-VL, LLaVA, etc.

    The client uses the /v1/chat/completions endpoint with image content
    encoded as base64 data URLs.

    Attributes:
        base_url: API base URL (e.g., "http://localhost:8000/v1").
        model: Model identifier.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2-VL-7B-Instruct",
        timeout: int = 600,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI Vision client.

        Args:
            base_url: API base URL.
            model: Model name/identifier.
            timeout: Request timeout in seconds.
            **kwargs: Additional configuration options.
        """
        super().__init__(base_url=base_url, model=model, timeout=timeout, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

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
            temperature: Sampling temperature.
            **kwargs: Additional parameters (passed to API).

        Returns:
            Generated text response.

        Raises:
            VlmConnectionError: If connection to server fails.
            VlmGenerationError: If generation fails.
        """
        image_data_url = self._encode_image(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            logger.error("Failed to connect to VLM server: %s", e)
            raise VlmConnectionError(f"Failed to connect to {self.base_url}: {e}") from e
        except httpx.HTTPStatusError as e:
            logger.error("VLM server returned error: %s", e.response.text)
            raise VlmGenerationError(
                f"VLM server error ({e.response.status_code}): {e.response.text}"
            ) from e
        except httpx.TimeoutException as e:
            logger.error("VLM request timed out after %ds", self.timeout)
            raise VlmConnectionError(f"Request timed out after {self.timeout}s") from e

        try:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return content.strip()
        except (KeyError, IndexError) as e:
            logger.error("Unexpected response format: %s", response.text[:500])
            raise VlmGenerationError(f"Unexpected response format: {e}") from e

    def _encode_image(self, image: "Image.Image | bytes | str") -> str:
        """Encode image to base64 data URL.

        Args:
            image: PIL Image, bytes, or file path.

        Returns:
            Base64 data URL string.
        """
        from PIL import Image as PILImage

        if isinstance(image, str):
            with open(image, "rb") as f:
                image_bytes = f.read()
        elif isinstance(image, bytes):
            image_bytes = image
        elif isinstance(image, PILImage.Image):
            buffer = io.BytesIO()
            image_format = image.format or "PNG"
            image.save(buffer, format=image_format)
            image_bytes = buffer.getvalue()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        base64_data = base64.b64encode(image_bytes).decode("utf-8")

        # Detect MIME type from magic bytes
        mime_type = self._detect_mime_type(image_bytes)

        return f"data:{mime_type};base64,{base64_data}"

    def _detect_mime_type(self, data: bytes) -> str:
        """Detect image MIME type from magic bytes.

        Args:
            data: Image bytes.

        Returns:
            MIME type string.
        """
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        elif data[:2] == b"\xff\xd8":
            return "image/jpeg"
        elif data[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
        else:
            return "image/png"  # Default fallback

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "OpenAIVisionClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def create_vlm_client_from_settings() -> OpenAIVisionClient:
    """Create VLM client from application settings.

    Returns:
        Configured OpenAIVisionClient instance.
    """
    from backend.config.settings import vlm_client_settings

    return OpenAIVisionClient(
        base_url=vlm_client_settings.base_url,
        model=vlm_client_settings.model,
        timeout=vlm_client_settings.timeout,
    )
