from __future__ import annotations

from typing import Any, Iterable, Optional, TypeVar

import httpx
from pydantic import BaseModel

from backend.config.settings import ollama_settings

from ..base import LLMResponse

T = TypeVar("T", bound=BaseModel)


class OllamaClient:
    _ALLOWED_REASONING_EFFORTS = {"low", "medium", "high"}

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = (base_url or ollama_settings.base_url).rstrip("/")
        self.model = model or ollama_settings.model_name
        self.temperature = temperature if temperature is not None else ollama_settings.temperature
        self.max_tokens = max_tokens if max_tokens is not None else ollama_settings.max_tokens
        self.timeout = timeout if timeout is not None else ollama_settings.timeout
        self.repeat_penalty = repeat_penalty if repeat_penalty is not None else ollama_settings.repeat_penalty
        self._client = client or httpx.Client(timeout=self.timeout)

    def _is_openai_compatible(self) -> bool:
        return self.base_url.endswith("/v1")

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
        if self._is_openai_compatible():
            return self._generate_openai_compatible(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                response_model=response_model,
                **kwargs,
            )

        return self._generate_native(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            response_model=response_model,
            **kwargs,
        )

    def _generate_openai_compatible(
        self,
        messages: Iterable[dict[str, str]],
        *,
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
        response_model: Optional[type[T]],
        **kwargs: Any,
    ) -> LLMResponse | T:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": stream,
        }

        if self.repeat_penalty and self.repeat_penalty != 1.0:
            payload["frequency_penalty"] = min(self.repeat_penalty - 1.0, 2.0)

        if response_model is not None:
            payload["response_format"] = {"type": "json_object"}

        effort = kwargs.pop("reasoning_effort", None)
        if effort in self._ALLOWED_REASONING_EFFORTS:
            payload["reasoning_effort"] = effort

        payload.update(kwargs)

        data = self._post_json(f"{self.base_url}/chat/completions", payload)
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        text = message.get("content")
        reasoning = message.get("reasoning_content") or message.get("reasoning")

        if text is None:
            text = reasoning or choice.get("text", "") or ""
            if text == reasoning:
                reasoning = None

        if response_model is not None:
            return response_model.model_validate_json(text)

        return LLMResponse(text=text, raw=data, reasoning=reasoning)

    def _generate_native(
        self,
        messages: Iterable[dict[str, str]],
        *,
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
        response_model: Optional[type[T]],
        **kwargs: Any,
    ) -> LLMResponse | T:
        native_kwargs = dict(kwargs)
        native_kwargs.pop("reasoning_effort", None)

        response_format = native_kwargs.pop("response_format", None)
        options = native_kwargs.pop("options", None)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": stream,
        }

        ollama_options: dict[str, Any] = {}
        if isinstance(options, dict):
            ollama_options.update(options)

        resolved_temperature = temperature if temperature is not None else self.temperature
        resolved_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        if resolved_temperature is not None:
            ollama_options["temperature"] = resolved_temperature
        if resolved_max_tokens is not None:
            ollama_options["num_predict"] = resolved_max_tokens
        if self.repeat_penalty and self.repeat_penalty != 1.0:
            ollama_options["repeat_penalty"] = self.repeat_penalty
        if ollama_options:
            payload["options"] = ollama_options

        if response_model is not None:
            payload["format"] = "json"
        elif isinstance(response_format, dict) and response_format.get("type") == "json_object":
            payload["format"] = "json"

        payload.update(native_kwargs)

        data = self._post_json(f"{self.base_url}/api/chat", payload)
        message = data.get("message")
        text = ""
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                text = content
        if not text:
            response = data.get("response")
            if isinstance(response, str):
                text = response

        if response_model is not None:
            return response_model.model_validate_json(text)

        return LLMResponse(text=text, raw=data)

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self._client.post(url, json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            raise RuntimeError(
                f"Ollama request failed: status={exc.response.status_code}, body={detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Ollama request failed: base_url={self.base_url}, error={exc}"
            ) from exc

        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError("Ollama request failed: response is not JSON object")
        return data


__all__ = ["OllamaClient"]
