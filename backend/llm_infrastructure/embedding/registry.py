"""Registry for embedding methods."""

from typing import Any, Type

from .base import BaseEmbedder


class EmbedderRegistry:
    """Global registry for embedding methods.

    Example:
        ```python
        # Register
        @register_embedder("bge_base", version="v1")
        class BGEBaseEmbedder(BaseEmbedder):
            ...

        # Use
        embedder = get_embedder("bge_base", version="v1")
        embedding = embedder.embed("hello world")
        ```
    """

    _registry: dict[str, dict[str, Type[BaseEmbedder]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        embedder_cls: Type[BaseEmbedder],
        version: str = "v1",
    ) -> None:
        """Register an embedder class.

        Args:
            name: Method name (e.g., "bge_base", "openai_ada")
            embedder_cls: Embedder class to register
            version: Version string (default: "v1")
        """
        if name not in cls._registry:
            cls._registry[name] = {}

        if version in cls._registry[name]:
            raise ValueError(
                f"Embedder '{name}' version '{version}' already registered"
            )

        cls._registry[name][version] = embedder_cls

    @classmethod
    def get(
        cls,
        name: str,
        version: str = "v1",
        **kwargs: Any,
    ) -> BaseEmbedder:
        """Get an embedder instance.

        Args:
            name: Method name
            version: Version string (default: "v1")
            **kwargs: Additional config passed to embedder __init__

        Returns:
            Embedder instance

        Raises:
            ValueError: If method not found
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown embedding method: '{name}'. "
                f"Available: {available}"
            )

        if version not in cls._registry[name]:
            available_versions = ", ".join(cls._registry[name].keys())
            raise ValueError(
                f"Unknown version '{version}' for method '{name}'. "
                f"Available versions: {available_versions}"
            )

        embedder_cls = cls._registry[name][version]
        # 레지스트리에서 요청한 별칭(alias)을 그대로 전달하여 기본 모델 선택 등에 활용
        kwargs.setdefault("alias", name)
        return embedder_cls(**kwargs)

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


def register_embedder(name: str, version: str = "v1"):
    """Decorator to register an embedder class.

    Args:
        name: Method name
        version: Version string (default: "v1")

    Example:
        ```python
        @register_embedder("my_embedder", version="v1")
        class MyEmbedder(BaseEmbedder):
            def embed(self, text):
                ...
        ```
    """
    def decorator(cls: Type[BaseEmbedder]) -> Type[BaseEmbedder]:
        EmbedderRegistry.register(name, cls, version=version)
        return cls
    return decorator


def get_embedder(name: str, version: str = "v1", **kwargs: Any) -> BaseEmbedder:
    """Get an embedder instance (convenience function).

    Args:
        name: Method name
        version: Version string
        **kwargs: Config passed to embedder

    Returns:
        Embedder instance
    """
    return EmbedderRegistry.get(name, version=version, **kwargs)
