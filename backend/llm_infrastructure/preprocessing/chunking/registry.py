"""Chunker registry module for text splitting methods."""

from typing import Any, Type

from .base import BaseChunker, ChunkParams


class ChunkerRegistry:
    """Registry for chunker implementations.

    Allows runtime selection of chunking method based on configuration.

    Example:
        ```python
        @register_chunker("fixed_size", version="v1")
        class FixedSizeChunker(BaseChunker):
            ...

        chunker = get_chunker("fixed_size", chunk_size=512)
        chunks = chunker.chunk(text)
        ```
    """

    _registry: dict[str, dict[str, Type[BaseChunker]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        chunker_cls: Type[BaseChunker],
        version: str = "v1",
    ) -> None:
        """Register a chunker class."""
        if name not in cls._registry:
            cls._registry[name] = {}

        if version in cls._registry[name]:
            raise ValueError(
                f"Chunker '{name}' version '{version}' already registered"
            )

        cls._registry[name][version] = chunker_cls

    @classmethod
    def get(
        cls,
        name: str,
        version: str = "v1",
        params: ChunkParams | None = None,
        **kwargs: Any,
    ) -> BaseChunker:
        """Get a chunker instance by name and version.

        Args:
            name: Chunker name (e.g., "fixed_size", "recursive").
            version: Chunker version (default: "v1").
            params: Optional ChunkParams instance.
            **kwargs: Additional parameters passed to chunker.

        Returns:
            Initialized BaseChunker instance.

        Raises:
            ValueError: If chunker name or version is not registered.
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys()) or "(none)"
            raise ValueError(
                f"Unknown chunker: '{name}'. Available: {available}"
            )

        if version not in cls._registry[name]:
            available_versions = ", ".join(cls._registry[name].keys())
            raise ValueError(
                f"Unknown version '{version}' for chunker '{name}'. "
                f"Available versions: {available_versions}"
            )

        chunker_cls = cls._registry[name][version]
        return chunker_cls(params=params, **kwargs)

    @classmethod
    def list_chunkers(cls) -> dict[str, list[str]]:
        """List all registered chunkers and their versions.

        Returns:
            Dict mapping chunker names to list of available versions.
        """
        return {
            name: list(versions.keys())
            for name, versions in cls._registry.items()
        }

    @classmethod
    def is_registered(cls, name: str, version: str = "v1") -> bool:
        """Check if a chunker is registered.

        Args:
            name: Chunker name.
            version: Chunker version.

        Returns:
            True if registered, False otherwise.
        """
        return name in cls._registry and version in cls._registry[name]


def register_chunker(name: str, version: str = "v1"):
    """Decorator to register a chunker class.

    Args:
        name: Name to register the chunker under.
        version: Version string (default: "v1").

    Example:
        ```python
        @register_chunker("my_chunker", version="v1")
        class MyChunker(BaseChunker):
            def chunk(self, text, doc_id="", metadata=None):
                ...
        ```
    """
    def decorator(cls: Type[BaseChunker]) -> Type[BaseChunker]:
        ChunkerRegistry.register(name, cls, version=version)
        return cls
    return decorator


def get_chunker(
    name: str,
    version: str = "v1",
    params: ChunkParams | None = None,
    **kwargs: Any,
) -> BaseChunker:
    """Convenience function to get a chunker instance.

    Args:
        name: Chunker name.
        version: Chunker version (default: "v1").
        params: Optional ChunkParams instance.
        **kwargs: Additional parameters.

    Returns:
        Initialized BaseChunker instance.
    """
    return ChunkerRegistry.get(name, version=version, params=params, **kwargs)
