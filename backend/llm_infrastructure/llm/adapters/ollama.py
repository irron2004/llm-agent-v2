from __future__ import annotations

from typing import Any, Iterable, TypeVar, overload

from pydantic import BaseModel

from ..base import BaseLLM, LLMResponse
from ..engines.ollama import OllamaClient
from ..registry import register_llm

T = TypeVar("T", bound=BaseModel)


@register_llm("ollama", version="v1")
class OllamaAdapter(BaseLLM):
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
        self.engine = OllamaClient(
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            client=client,
        )

    @overload
    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        response_model: type[T],
        **kwargs: Any,
    ) -> T: ...

    @overload
    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        response_model: None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        response_model: type[T] | None = None,
        **kwargs: Any,
    ) -> LLMResponse | T:
        return self.engine.generate(messages, response_model=response_model, **kwargs)


__all__ = ["OllamaAdapter"]
