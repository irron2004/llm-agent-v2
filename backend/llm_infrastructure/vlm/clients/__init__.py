"""VLM client implementations.

Available clients:
- OpenAIVisionClient: For OpenAI-compatible Vision APIs (vLLM, etc.)
"""

from backend.llm_infrastructure.vlm.clients.openai_vision import OpenAIVisionClient

__all__ = ["OpenAIVisionClient"]
