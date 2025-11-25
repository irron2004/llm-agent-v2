"""Registry for retrieval methods."""

from typing import Any, Type

from .base import BaseRetriever


class RetrieverRegistry:
    """Global registry for retrieval methods.

    Example:
        ```python
        # Register
        @register_retriever("hybrid_rrf", version="v1")
        class HybridRRFRetriever(BaseRetriever):
            ...

        # Use
        retriever = get_retriever("hybrid_rrf", version="v1")
        results = retriever.retrieve("query text", top_k=10)
        ```
    """

    _registry: dict[str, dict[str, Type[BaseRetriever]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        retriever_cls: Type[BaseRetriever],
        version: str = "v1",
    ) -> None:
        """Register a retriever class.

        Args:
            name: Method name (e.g., "dense", "hybrid_rrf")
            retriever_cls: Retriever class to register
            version: Version string (default: "v1")
        """
        if name not in cls._registry:
            cls._registry[name] = {}

        if version in cls._registry[name]:
            raise ValueError(
                f"Retriever '{name}' version '{version}' already registered"
            )

        cls._registry[name][version] = retriever_cls

    @classmethod
    def get(
        cls,
        name: str,
        version: str = "v1",
        **kwargs: Any,
    ) -> BaseRetriever:
        """Get a retriever instance.

        Args:
            name: Method name
            version: Version string (default: "v1")
            **kwargs: Additional config passed to retriever __init__

        Returns:
            Retriever instance

        Raises:
            ValueError: If method not found
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown retrieval method: '{name}'. "
                f"Available: {available}"
            )

        if version not in cls._registry[name]:
            available_versions = ", ".join(cls._registry[name].keys())
            raise ValueError(
                f"Unknown version '{version}' for method '{name}'. "
                f"Available versions: {available_versions}"
            )

        retriever_cls = cls._registry[name][version]
        return retriever_cls(**kwargs)

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


def register_retriever(name: str, version: str = "v1"):
    """Decorator to register a retriever class.

    Args:
        name: Method name
        version: Version string (default: "v1")

    Example:
        ```python
        @register_retriever("my_retriever", version="v1")
        class MyRetriever(BaseRetriever):
            def retrieve(self, query, top_k=10):
                ...
        ```
    """
    def decorator(cls: Type[BaseRetriever]) -> Type[BaseRetriever]:
        RetrieverRegistry.register(name, cls, version=version)
        return cls
    return decorator


def get_retriever(name: str, version: str = "v1", **kwargs: Any) -> BaseRetriever:
    """Get a retriever instance (convenience function).

    Args:
        name: Method name
        version: Version string
        **kwargs: Config passed to retriever

    Returns:
        Retriever instance
    """
    return RetrieverRegistry.get(name, version=version, **kwargs)
