"""Registry for LLM implementations."""

from __future__ import annotations

from typing import Any, Type

from .base import BaseLLM


class LLMRegistry:
    """Global registry for LLM engines."""

    _registry: dict[str, dict[str, Type[BaseLLM]]] = {}

    @classmethod
    def register(cls, name: str, llm_cls: Type[BaseLLM], version: str = "v1") -> None:
        cls._registry.setdefault(name, {})
        if version in cls._registry[name]:
            raise ValueError(f"LLM '{name}' version '{version}' already registered")
        cls._registry[name][version] = llm_cls

    @classmethod
    def get(cls, name: str, version: str = "v1", **kwargs: Any) -> BaseLLM:
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown LLM '{name}'. Available: {available}")
        if version not in cls._registry[name]:
            versions = ", ".join(cls._registry[name].keys())
            raise ValueError(f"Unknown version '{version}' for '{name}'. Available: {versions}")
        llm_cls = cls._registry[name][version]
        kwargs.setdefault("alias", name)
        return llm_cls(**kwargs)

    @classmethod
    def list_methods(cls) -> dict[str, list[str]]:
        return {name: list(versions.keys()) for name, versions in cls._registry.items()}


def register_llm(name: str, version: str = "v1"):
    """Decorator to register an LLM adapter."""
    def decorator(cls: Type[BaseLLM]) -> Type[BaseLLM]:
        LLMRegistry.register(name, cls, version=version)
        return cls
    return decorator


def get_llm(name: str, version: str = "v1", **kwargs: Any) -> BaseLLM:
    return LLMRegistry.get(name, version=version, **kwargs)


__all__ = ["LLMRegistry", "register_llm", "get_llm"]
