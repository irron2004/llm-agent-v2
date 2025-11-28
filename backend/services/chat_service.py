"""Chat service orchestrating LLM calls (vLLM by default)."""

from __future__ import annotations

from typing import Iterable, Optional

from backend.config.settings import rag_settings, vllm_settings
from backend.llm_infrastructure.llm import get_llm, LLMResponse


class ChatService:
    """Simple chat orchestrator using the LLM registry."""

    def __init__(
        self,
        llm_method: str | None = None,
        llm_version: str | None = None,
    ) -> None:
        self.method = llm_method or "vllm"
        self.version = llm_version or "v1"
        self._llm = get_llm(
            self.method,
            version=self.version,
            base_url=vllm_settings.base_url,
            model=vllm_settings.model_name,
            temperature=vllm_settings.temperature,
            max_tokens=vllm_settings.max_tokens,
            timeout=vllm_settings.timeout,
        )

    def chat(
        self,
        user_message: str,
        *,
        history: Iterable[dict[str, str]] | None = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(list(history))
        messages.append({"role": "user", "content": user_message})
        return self._llm.generate(messages, **kwargs)


__all__ = ["ChatService"]
