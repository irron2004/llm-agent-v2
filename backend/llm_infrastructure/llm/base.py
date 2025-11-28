"""Base classes and response model for LLM engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class LLMResponse:
    """Normalized LLM response."""

    text: str
    raw: dict[str, Any] | None = None


class BaseLLM(ABC):
    """Common interface for all LLM implementations."""

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs

    @abstractmethod
    def generate(
        self,
        messages: Iterable[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from messages."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"


__all__ = ["BaseLLM", "LLMResponse"]
