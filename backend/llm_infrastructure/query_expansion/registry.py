"""Registry for query expansion methods."""

from typing import Any, Type

from .base import BaseQueryExpander


class QueryExpanderRegistry:
    """Global registry for query expansion methods.

    Example:
        ```python
        # Register
        @register_query_expander("llm", version="v1")
        class LLMQueryExpander(BaseQueryExpander):
            ...

        # Use
        expander = get_query_expander("llm", version="v1")
        expanded = expander.expand("original query", n=3)
        ```
    """

    _registry: dict[str, dict[str, Type[BaseQueryExpander]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        expander_cls: Type[BaseQueryExpander],
        version: str = "v1",
    ) -> None:
        """Register a query expander class.

        Args:
            name: Method name (e.g., "llm", "synonym")
            expander_cls: QueryExpander class to register
            version: Version string (default: "v1")
        """
        if name not in cls._registry:
            cls._registry[name] = {}

        if version in cls._registry[name]:
            raise ValueError(
                f"QueryExpander '{name}' version '{version}' already registered"
            )

        cls._registry[name][version] = expander_cls

    @classmethod
    def get(
        cls,
        name: str,
        version: str = "v1",
        **kwargs: Any,
    ) -> BaseQueryExpander:
        """Get a query expander instance.

        Args:
            name: Method name
            version: Version string (default: "v1")
            **kwargs: Additional config passed to expander __init__

        Returns:
            QueryExpander instance

        Raises:
            ValueError: If method not found
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys()) if cls._registry else "(none)"
            raise ValueError(
                f"Unknown query expansion method: '{name}'. "
                f"Available: {available}"
            )

        if version not in cls._registry[name]:
            available_versions = ", ".join(cls._registry[name].keys())
            raise ValueError(
                f"Unknown version '{version}' for method '{name}'. "
                f"Available versions: {available_versions}"
            )

        expander_cls = cls._registry[name][version]
        return expander_cls(**kwargs)

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


def register_query_expander(name: str, version: str = "v1"):
    """Decorator to register a query expander class.

    Args:
        name: Method name
        version: Version string (default: "v1")

    Example:
        ```python
        @register_query_expander("llm", version="v1")
        class LLMQueryExpander(BaseQueryExpander):
            def expand(self, query, n=3, include_original=True):
                ...
        ```
    """
    def decorator(cls: Type[BaseQueryExpander]) -> Type[BaseQueryExpander]:
        QueryExpanderRegistry.register(name, cls, version=version)
        return cls
    return decorator


def get_query_expander(name: str, version: str = "v1", **kwargs: Any) -> BaseQueryExpander:
    """Get a query expander instance (convenience function).

    Args:
        name: Method name
        version: Version string
        **kwargs: Config passed to expander

    Returns:
        QueryExpander instance
    """
    return QueryExpanderRegistry.get(name, version=version, **kwargs)


__all__ = [
    "QueryExpanderRegistry",
    "register_query_expander",
    "get_query_expander",
]
