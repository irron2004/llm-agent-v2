"""vLLM engine (OpenAI-compatible API)."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import httpx

from backend.config.settings import vllm_settings
from ..base import LLMResponse


class VLLMClient:
    """Thin client for vLLM OpenAI-compatible /v1/chat/completions."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url or vllm_settings.base_url
        self.model = model or vllm_settings.model_name
        self.temperature = temperature if temperature is not None else vllm_settings.temperature
        self.max_tokens = max_tokens if max_tokens is not None else vllm_settings.max_tokens
        self.timeout = timeout if timeout is not None else vllm_settings.timeout
        self._client = client or httpx.Client(timeout=self.timeout)

    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": stream,
        }
        # Allow extra OpenAI-compatible params
        payload.update(kwargs)

        resp = self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return LLMResponse(text=text, raw=data)


__all__ = ["VLLMClient"]
