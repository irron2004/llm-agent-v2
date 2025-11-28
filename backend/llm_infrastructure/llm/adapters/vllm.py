"""vLLM adapter registered in the LLM registry."""

from __future__ import annotations

from typing import Any, Iterable

from ..base import BaseLLM, LLMResponse
from ..engines.vllm import VLLMClient
from ..registry import register_llm


@register_llm("vllm", version="v1")
class VLLMAdapter(BaseLLM):
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.engine = VLLMClient(
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            client=client,
        )

    def generate(
        self,
        messages: Iterable[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        return self.engine.generate(messages, **kwargs)


__all__ = ["VLLMAdapter"]
