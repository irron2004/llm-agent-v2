"""Vector database registry for managing implementations.

Follows the same registry pattern used by embedders and retrievers.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Type

from backend.llm_infrastructure.vectordb.base import VectorDBClient

logger = logging.getLogger(__name__)


class VectorDBRegistry:
    """Registry for vector database implementations.

    Example:
        @register_vectordb("elasticsearch")
        class EsVectorDB(VectorDBClient):
            ...

        db = get_vectordb("elasticsearch", es_client=client, index_name="idx")
    """

    _registry: dict[str, Type[VectorDBClient]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        db_cls: Type[VectorDBClient],
    ) -> None:
        """Register a vector DB implementation.

        Args:
            name: Database identifier (e.g., "elasticsearch", "pinecone").
            db_cls: VectorDBClient subclass to register.
        """
        if name in cls._registry:
            logger.warning("Overwriting existing vectordb registration: %s", name)
        cls._registry[name] = db_cls
        logger.debug("Registered vectordb: %s", name)

    @classmethod
    def get(
        cls,
        name: str,
        **kwargs: Any,
    ) -> VectorDBClient:
        """Get a vector DB instance.

        Args:
            name: Database identifier.
            **kwargs: Arguments passed to DB constructor.

        Returns:
            Instantiated VectorDBClient.

        Raises:
            KeyError: If database not found.
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(f"VectorDB '{name}' not found. Available: {available}")

        db_cls = cls._registry[name]
        return db_cls(**kwargs)

    @classmethod
    def list_databases(cls) -> list[str]:
        """List all registered databases.

        Returns:
            List of registered database names.
        """
        return list(cls._registry.keys())


def register_vectordb(
    name: str,
) -> Callable[[Type[VectorDBClient]], Type[VectorDBClient]]:
    """Decorator to register a vector DB implementation.

    Args:
        name: Database identifier.

    Returns:
        Decorator function.

    Example:
        @register_vectordb("elasticsearch")
        class EsVectorDB(VectorDBClient):
            ...
    """

    def decorator(cls: Type[VectorDBClient]) -> Type[VectorDBClient]:
        VectorDBRegistry.register(name, cls)
        return cls

    return decorator


def get_vectordb(
    name: str,
    **kwargs: Any,
) -> VectorDBClient:
    """Get a vector DB instance from the registry.

    Args:
        name: Database identifier.
        **kwargs: Arguments passed to DB constructor.

    Returns:
        Instantiated VectorDBClient.
    """
    return VectorDBRegistry.get(name, **kwargs)


__all__ = [
    "VectorDBRegistry",
    "register_vectordb",
    "get_vectordb",
]
