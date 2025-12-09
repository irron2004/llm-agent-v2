"""Registry for reranking methods."""

from typing import Any, Type

from .base import BaseReranker


class RerankerRegistry:
    """Global registry for reranking methods.

    Example:
        ```python
        # Register
        @register_reranker("cross_encoder", version="v1")
        class CrossEncoderReranker(BaseReranker):
            ...

        # Use
        reranker = get_reranker("cross_encoder", version="v1")
        results = reranker.rerank(query, retrieval_results, top_k=5)
        ```
    """

    _registry: dict[str, dict[str, Type[BaseReranker]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        reranker_cls: Type[BaseReranker],
        version: str = "v1",
    ) -> None:
        """Register a reranker class.

        Args:
            name: Method name (e.g., "cross_encoder", "llm")
            reranker_cls: Reranker class to register
            version: Version string (default: "v1")
        """
        if name not in cls._registry:
            cls._registry[name] = {}

        if version in cls._registry[name]:
            raise ValueError(
                f"Reranker '{name}' version '{version}' already registered"
            )

        cls._registry[name][version] = reranker_cls

    @classmethod
    def get(
        cls,
        name: str,
        version: str = "v1",
        **kwargs: Any,
    ) -> BaseReranker:
        """Get a reranker instance.

        Args:
            name: Method name
            version: Version string (default: "v1")
            **kwargs: Additional config passed to reranker __init__

        Returns:
            Reranker instance

        Raises:
            ValueError: If method not found
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys()) if cls._registry else "(none)"
            raise ValueError(
                f"Unknown reranking method: '{name}'. "
                f"Available: {available}"
            )

        if version not in cls._registry[name]:
            available_versions = ", ".join(cls._registry[name].keys())
            raise ValueError(
                f"Unknown version '{version}' for method '{name}'. "
                f"Available versions: {available_versions}"
            )

        reranker_cls = cls._registry[name][version]
        return reranker_cls(**kwargs)

    @classmethod
    def list_methods(cls) -> dict[str, list[str]]:
        """List all registered methods and their versions.

        Returns:
            Dict mapping method names to list of versions
        """
        return {
            name: list(versions.keys())
            for name, versions in cls._registry.items()
        }


def register_reranker(name: str, version: str = "v1"):
    """Decorator to register a reranker class.

    Args:
        name: Method name
        version: Version string (default: "v1")

    Example:
        ```python
        @register_reranker("cross_encoder", version="v1")
        class CrossEncoderReranker(BaseReranker):
            def rerank(self, query, results, top_k=5):
                ...
        ```
    """
    def decorator(cls: Type[BaseReranker]) -> Type[BaseReranker]:
        RerankerRegistry.register(name, cls, version=version)
        return cls
    return decorator


def get_reranker(name: str, version: str = "v1", **kwargs: Any) -> BaseReranker:
    """Get a reranker instance (convenience function).

    Args:
        name: Method name
        version: Version string
        **kwargs: Config passed to reranker

    Returns:
        Reranker instance
    """
    return RerankerRegistry.get(name, version=version, **kwargs)


__all__ = [
    "RerankerRegistry",
    "register_reranker",
    "get_reranker",
]
