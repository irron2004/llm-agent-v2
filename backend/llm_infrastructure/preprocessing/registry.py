"""Registry for preprocessing methods."""

from typing import Any, Type

from .base import BasePreprocessor


class PreprocessorRegistry:
    """Global registry for preprocessing methods.

    This allows dynamic selection of preprocessing methods at runtime
    based on configuration (e.g., from .env or preset files).

    Example:
        ```python
        # Register
        @register_preprocessor("method_a", version="v1")
        class PreprocessorA(BasePreprocessor):
            ...

        # Use
        preprocessor = get_preprocessor("method_a", version="v1")
        results = preprocessor.preprocess(docs)
        ```
    """

    _registry: dict[str, dict[str, Type[BasePreprocessor]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        preprocessor_cls: Type[BasePreprocessor],
        version: str = "v1",
    ) -> None:
        """Register a preprocessor class.

        Args:
            name: Method name (e.g., "standard", "domain_specific")
            preprocessor_cls: Preprocessor class to register
            version: Version string (default: "v1")
        """
        if name not in cls._registry:
            cls._registry[name] = {}

        if version in cls._registry[name]:
            raise ValueError(
                f"Preprocessor '{name}' version '{version}' already registered"
            )

        cls._registry[name][version] = preprocessor_cls

    @classmethod
    def get(
        cls,
        name: str,
        version: str = "v1",
        **kwargs: Any,
    ) -> BasePreprocessor:
        """Get a preprocessor instance.

        Args:
            name: Method name
            version: Version string (default: "v1")
            **kwargs: Additional config passed to preprocessor __init__

        Returns:
            Preprocessor instance

        Raises:
            ValueError: If method not found
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown preprocessing method: '{name}'. "
                f"Available: {available}"
            )

        if version not in cls._registry[name]:
            available_versions = ", ".join(cls._registry[name].keys())
            raise ValueError(
                f"Unknown version '{version}' for method '{name}'. "
                f"Available versions: {available_versions}"
            )

        preprocessor_cls = cls._registry[name][version]
        return preprocessor_cls(**kwargs)

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


def register_preprocessor(name: str, version: str = "v1"):
    """Decorator to register a preprocessor class.

    Args:
        name: Method name
        version: Version string (default: "v1")

    Example:
        ```python
        @register_preprocessor("my_method", version="v1")
        class MyPreprocessor(BasePreprocessor):
            def preprocess(self, docs):
                ...
        ```
    """
    def decorator(cls: Type[BasePreprocessor]) -> Type[BasePreprocessor]:
        PreprocessorRegistry.register(name, cls, version=version)
        return cls
    return decorator


def get_preprocessor(name: str, version: str = "v1", **kwargs: Any) -> BasePreprocessor:
    """Get a preprocessor instance (convenience function).

    Args:
        name: Method name
        version: Version string
        **kwargs: Config passed to preprocessor

    Returns:
        Preprocessor instance
    """
    return PreprocessorRegistry.get(name, version=version, **kwargs)
