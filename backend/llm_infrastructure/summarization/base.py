"""Base classes for summarization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SummaryResult:
    """Normalized summarization result."""

    original_text: str
    summary: str
    summary_length: int
    compression_ratio: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSummarizer(ABC):
    """Abstract base class for all summarization implementations."""

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs

    @abstractmethod
    def summarize(
        self,
        text: str,
        *,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> SummaryResult:
        """Summarize input text.

        Args:
            text: Input text to summarize.
            max_length: Optional maximum summary length hint.
            **kwargs: Additional summarizer-specific options.

        Returns:
            SummaryResult containing summary and metadata.
        """
        raise NotImplementedError

    def summarize_batch(
        self,
        texts: list[str],
        *,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> list[SummaryResult]:
        """Summarize multiple texts (default: sequential).

        Subclasses may override for parallel/batch processing.
        """
        return [self.summarize(t, max_length=max_length, **kwargs) for t in texts]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"


__all__ = ["BaseSummarizer", "SummaryResult"]
