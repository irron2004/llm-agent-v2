"""VLM client registry for managing VLM client implementations.

Follows the same registry pattern used by embedders and retrievers.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Type

from backend.llm_infrastructure.vlm.base import BaseVlmClient

logger = logging.getLogger(__name__)


class VlmClientRegistry:
    """Registry for VLM client implementations.

    Example:
        @register_vlm_client("openai_vision", version="v1")
        class OpenAIVisionClient(BaseVlmClient):
            ...

        client = get_vlm_client("openai_vision", base_url="http://localhost:8000/v1")
    """

    _registry: dict[str, dict[str, Type[BaseVlmClient]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        client_cls: Type[BaseVlmClient],
        version: str = "v1",
    ) -> None:
        """Register a VLM client class.

        Args:
            name: Client identifier (e.g., "openai_vision").
            client_cls: VLM client class to register.
            version: Version string (default: "v1").
        """
        if name not in cls._registry:
            cls._registry[name] = {}
        cls._registry[name][version] = client_cls
        logger.debug("Registered VLM client: %s (version=%s)", name, version)

    @classmethod
    def get(
        cls,
        name: str,
        version: str = "v1",
        **kwargs: Any,
    ) -> BaseVlmClient:
        """Get a VLM client instance.

        Args:
            name: Client identifier.
            version: Version string.
            **kwargs: Arguments passed to client constructor.

        Returns:
            Instantiated VLM client.

        Raises:
            KeyError: If client not found.
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(f"VLM client '{name}' not found. Available: {available}")

        versions = cls._registry[name]
        if version not in versions:
            available = list(versions.keys())
            raise KeyError(
                f"Version '{version}' not found for VLM client '{name}'. "
                f"Available: {available}"
            )

        client_cls = versions[version]
        return client_cls(**kwargs)

    @classmethod
    def list_clients(cls) -> dict[str, list[str]]:
        """List all registered clients and their versions.

        Returns:
            Dict mapping client names to available versions.
        """
        return {name: list(versions.keys()) for name, versions in cls._registry.items()}


def register_vlm_client(
    name: str,
    version: str = "v1",
) -> Callable[[Type[BaseVlmClient]], Type[BaseVlmClient]]:
    """Decorator to register a VLM client class.

    Args:
        name: Client identifier.
        version: Version string.

    Returns:
        Decorator function.

    Example:
        @register_vlm_client("openai_vision", version="v1")
        class OpenAIVisionClient(BaseVlmClient):
            ...
    """

    def decorator(cls: Type[BaseVlmClient]) -> Type[BaseVlmClient]:
        VlmClientRegistry.register(name, cls, version)
        return cls

    return decorator


def get_vlm_client(
    name: str,
    version: str = "v1",
    **kwargs: Any,
) -> BaseVlmClient:
    """Get a VLM client instance from the registry.

    Args:
        name: Client identifier.
        version: Version string.
        **kwargs: Arguments passed to client constructor.

    Returns:
        Instantiated VLM client.
    """
    return VlmClientRegistry.get(name, version, **kwargs)
