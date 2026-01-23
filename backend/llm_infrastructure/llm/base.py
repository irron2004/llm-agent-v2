"""Base classes and response model for LLM engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, TypeVar, overload

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResponse:
    """Normalized LLM response."""

    text: str
    raw: dict[str, Any] | None = None
    reasoning: str | None = None  # Reasoning content from reasoning models


class BaseLLM(ABC):
    """Common interface for all LLM implementations."""

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs

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

    @abstractmethod
    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        response_model: type[T] | None = None,
        **kwargs: Any,
    ) -> LLMResponse | T:
        """Generate a response from messages.

        Args:
            messages: Chat messages.
            response_model: If provided, parse JSON response into this Pydantic model.
            **kwargs: Additional parameters for the LLM.

        Returns:
            LLMResponse if response_model is None, otherwise the parsed model instance.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"


__all__ = ["BaseLLM", "LLMResponse"]
