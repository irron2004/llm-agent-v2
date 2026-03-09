"""LLM adapters (registered implementations)."""

from .vllm import VLLMAdapter
from .ollama import OllamaAdapter

__all__ = ["VLLMAdapter", "OllamaAdapter"]
