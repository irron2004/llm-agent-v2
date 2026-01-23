"""vLLM engine (OpenAI-compatible API)."""

from __future__ import annotations

from typing import Any, Iterable, Optional, TypeVar

import httpx
from pydantic import BaseModel

from backend.config.settings import vllm_settings
from ..base import LLMResponse

T = TypeVar("T", bound=BaseModel)


class VLLMClient:
    """Thin client for vLLM OpenAI-compatible /v1/chat/completions."""

    _ALLOWED_REASONING_EFFORTS = {"low", "medium", "high"}

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url or vllm_settings.base_url
        self.model = model or vllm_settings.model_name
        self.temperature = temperature if temperature is not None else vllm_settings.temperature
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None else vllm_settings.reasoning_effort
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
        response_model: Optional[type[T]] = None,
        **kwargs: Any,
    ) -> LLMResponse | T:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": stream,
        }

        # If response_model is provided, force JSON mode
        if response_model is not None:
            payload["response_format"] = {"type": "json_object"}

        effort = kwargs.pop("reasoning_effort", None)
        if effort is None:
            effort = self.reasoning_effort
        if effort in self._ALLOWED_REASONING_EFFORTS:
            payload["reasoning_effort"] = effort

        # Allow extra OpenAI-compatible params
        payload.update(kwargs)

        try:
            resp = self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # Surface provider error details to the caller for easier debugging.
            detail = exc.response.text
            raise RuntimeError(
                f"vLLM request failed: status={exc.response.status_code}, body={detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"vLLM request failed: base_url={self.base_url}, error={exc}"
            ) from exc

        data = resp.json()
        # Some OpenAI-compatible providers return tool_calls with content=None.
        # Normalize to a non-None string so downstream `.strip()` calls do not crash.
        choice = data["choices"][0]
        message = choice.get("message", {})
        text = message.get("content")

        # Extract reasoning content from reasoning models
        reasoning = message.get("reasoning_content") or message.get("reasoning")

        if text is None:
            # Reasoning models may return reasoning_content instead of content
            text = reasoning or choice.get("text", "") or ""
            # If we used reasoning as text, clear it to avoid duplication
            if text == reasoning:
                reasoning = None

        # Parse to Pydantic model if requested
        if response_model is not None:
            return response_model.model_validate_json(text)

        return LLMResponse(text=text, raw=data, reasoning=reasoning)


__all__ = ["VLLMClient"]
