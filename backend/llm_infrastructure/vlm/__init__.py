"""VLM (Vision-Language Model) client module.

Provides a unified interface for interacting with VLM models via OpenAI-compatible APIs.

Usage:
    from backend.llm_infrastructure.vlm import get_vlm_client, BaseVlmClient

    # Get client from registry
    client = get_vlm_client("openai_vision", base_url="http://localhost:8000/v1")
    result = client.generate(image, "Extract text from this image")

    # Or create directly
    from backend.llm_infrastructure.vlm.clients import OpenAIVisionClient
    client = OpenAIVisionClient(base_url="http://localhost:8000/v1")
"""

from backend.llm_infrastructure.vlm.base import BaseVlmClient
from backend.llm_infrastructure.vlm.registry import (
    VlmClientRegistry,
    get_vlm_client,
    register_vlm_client,
)

__all__ = [
    "BaseVlmClient",
    "VlmClientRegistry",
    "get_vlm_client",
    "register_vlm_client",
]
