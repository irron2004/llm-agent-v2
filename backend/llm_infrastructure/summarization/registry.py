"""Registry for summarization implementations."""

from __future__ import annotations

from typing import Any, Type

from .base import BaseSummarizer


class SummarizerRegistry:
    """Global registry for summarizer implementations."""

    _registry: dict[str, dict[str, Type[BaseSummarizer]]] = {}

    @classmethod
    def register(
        cls, name: str, summarizer_cls: Type[BaseSummarizer], version: str = "v1"
    ) -> None:
        cls._registry.setdefault(name, {})
        if version in cls._registry[name]:
            raise ValueError(
                f"Summarizer '{name}' version '{version}' already registered"
            )
        cls._registry[name][version] = summarizer_cls

    @classmethod
    def get(cls, name: str, version: str = "v1", **kwargs: Any) -> BaseSummarizer:
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys()) or "(none)"
            raise ValueError(f"Unknown summarizer '{name}'. Available: {available}")
        if version not in cls._registry[name]:
            versions = ", ".join(cls._registry[name].keys())
            raise ValueError(
                f"Unknown version '{version}' for '{name}'. Available: {versions}"
            )
        summarizer_cls = cls._registry[name][version]
        kwargs.setdefault("alias", name)
        return summarizer_cls(**kwargs)

    @classmethod
    def list_methods(cls) -> dict[str, list[str]]:
        return {name: list(versions.keys()) for name, versions in cls._registry.items()}


def register_summarizer(name: str, version: str = "v1"):
    """Decorator to register a summarizer adapter."""

    def decorator(cls: Type[BaseSummarizer]) -> Type[BaseSummarizer]:
        SummarizerRegistry.register(name, cls, version=version)
        return cls

    return decorator


def get_summarizer(name: str, version: str = "v1", **kwargs: Any) -> BaseSummarizer:
    """Get a summarizer instance by name and version."""
    return SummarizerRegistry.get(name, version=version, **kwargs)


__all__ = ["SummarizerRegistry", "register_summarizer", "get_summarizer"]
