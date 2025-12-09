"""Elasticsearch index manager for RAG chunks.

Handles index creation, deletion, and alias management with
environment-based naming and rolling update strategy.

Usage:
    >>> from backend.llm_infrastructure.elasticsearch import EsIndexManager
    >>> manager = EsIndexManager(es_host="http://localhost:9200", env="dev")
    >>> manager.create_index(version=1, dims=1024)
    >>> manager.switch_alias(version=1)
"""

from __future__ import annotations

import logging
from typing import Any

from elasticsearch import Elasticsearch, NotFoundError

from .mappings import get_rag_chunks_mapping, get_index_settings, get_index_meta

logger = logging.getLogger(__name__)


class EsIndexManager:
    """Elasticsearch index manager for RAG chunks.

    Naming convention:
        - Index: rag_chunks_{env}_v{version}
        - Alias: rag_chunks_{env}_current

    Attributes:
        es: Elasticsearch client
        env: Environment name (dev, staging, prod)
        index_prefix: Base prefix for index names (default: rag_chunks)
    """

    def __init__(
        self,
        es_host: str = "http://localhost:9200",
        env: str = "dev",
        index_prefix: str = "rag_chunks",
        es_user: str | None = None,
        es_password: str | None = None,
        verify_certs: bool = True,
        es_client: Elasticsearch | None = None,
    ) -> None:
        """Initialize the index manager.

        Args:
            es_host: Elasticsearch host URL
            env: Environment name (dev, staging, prod)
            index_prefix: Base prefix for index names
            es_user: Elasticsearch username (optional)
            es_password: Elasticsearch password (optional)
            verify_certs: Verify SSL certificates
            es_client: Pre-configured Elasticsearch client (optional)
        """
        self.env = env
        self.index_prefix = index_prefix

        if es_client is not None:
            self.es = es_client
        else:
            client_kwargs: dict[str, Any] = {
                "hosts": [es_host],
                "verify_certs": verify_certs,
            }
            if es_user and es_password:
                client_kwargs["basic_auth"] = (es_user, es_password)

            self.es = Elasticsearch(**client_kwargs)

    # =========================================================================
    # Naming Helpers
    # =========================================================================

    def get_index_name(self, version: int) -> str:
        """Get index name for a specific version.

        Args:
            version: Index version number

        Returns:
            Index name (e.g., rag_chunks_dev_v1)
        """
        return f"{self.index_prefix}_{self.env}_v{version}"

    def get_alias_name(self) -> str:
        """Get alias name for current environment.

        Returns:
            Alias name (e.g., rag_chunks_dev_current)
        """
        return f"{self.index_prefix}_{self.env}_current"

    # =========================================================================
    # Index Operations
    # =========================================================================

    def index_exists(self, version: int) -> bool:
        """Check if an index exists.

        Args:
            version: Index version number

        Returns:
            True if index exists
        """
        index_name = self.get_index_name(version)
        return self.es.indices.exists(index=index_name)

    def create_index(
        self,
        version: int,
        dims: int = 1024,
        number_of_shards: int = 1,
        number_of_replicas: int = 0,
        embedding_model: str = "nlpai-lab/KoE5",
        chunking_method: str = "fixed_size",
        chunking_size: int = 512,
        chunking_overlap: int = 50,
        preprocess_method: str = "normalize",
        skip_if_exists: bool = False,
    ) -> dict[str, Any]:
        """Create a new index with the RAG chunks mapping.

        Args:
            version: Index version number
            dims: Embedding vector dimensions
            number_of_shards: Number of primary shards
            number_of_replicas: Number of replica shards
            embedding_model: Model name for _meta
            chunking_method: Chunking method name for _meta
            chunking_size: Chunk size for _meta
            chunking_overlap: Chunk overlap for _meta
            preprocess_method: Preprocessing method for _meta
            skip_if_exists: Skip creation if index already exists

        Returns:
            Elasticsearch response

        Raises:
            Exception: If index creation fails
        """
        index_name = self.get_index_name(version)

        if skip_if_exists and self.index_exists(version):
            logger.info(f"Index {index_name} already exists, skipping creation")
            return {"acknowledged": True, "skipped": True}

        body = {
            "settings": get_index_settings(
                number_of_shards=number_of_shards,
                number_of_replicas=number_of_replicas,
            ),
            "mappings": {
                **get_rag_chunks_mapping(dims=dims),
                "_meta": get_index_meta(
                    embedding_model=embedding_model,
                    embedding_dim=dims,
                    chunking_method=chunking_method,
                    chunking_size=chunking_size,
                    chunking_overlap=chunking_overlap,
                    preprocess_method=preprocess_method,
                ),
            },
        }

        logger.info(f"Creating index {index_name} with dims={dims}")
        response = self.es.indices.create(index=index_name, body=body)
        logger.info(f"Index {index_name} created successfully")
        return response

    def delete_index(self, version: int, ignore_not_found: bool = True) -> dict[str, Any]:
        """Delete an index.

        Args:
            version: Index version number
            ignore_not_found: Don't raise error if index doesn't exist

        Returns:
            Elasticsearch response
        """
        index_name = self.get_index_name(version)
        logger.warning(f"Deleting index {index_name}")

        try:
            response = self.es.indices.delete(index=index_name)
            logger.info(f"Index {index_name} deleted successfully")
            return response
        except NotFoundError:
            if ignore_not_found:
                logger.info(f"Index {index_name} not found, nothing to delete")
                return {"acknowledged": True, "not_found": True}
            raise

    # =========================================================================
    # Alias Operations
    # =========================================================================

    def get_alias_target(self) -> str | None:
        """Get the current index pointed to by the alias.

        Returns:
            Index name or None if alias doesn't exist
        """
        alias_name = self.get_alias_name()
        try:
            response = self.es.indices.get_alias(name=alias_name)
            # Response format: {index_name: {aliases: {alias_name: {}}}}
            indices = list(response.keys())
            return indices[0] if indices else None
        except NotFoundError:
            return None

    def switch_alias(self, version: int) -> dict[str, Any]:
        """Switch alias to point to a specific index version.

        This is an atomic operation that removes the alias from the old
        index and adds it to the new index in a single request.

        Args:
            version: Target index version number

        Returns:
            Elasticsearch response

        Raises:
            ValueError: If target index doesn't exist
        """
        alias_name = self.get_alias_name()
        new_index = self.get_index_name(version)

        if not self.index_exists(version):
            raise ValueError(f"Target index {new_index} does not exist")

        actions = []

        # Remove alias from current index (if any)
        current_index = self.get_alias_target()
        if current_index:
            actions.append({
                "remove": {
                    "index": current_index,
                    "alias": alias_name,
                }
            })
            logger.info(f"Removing alias {alias_name} from {current_index}")

        # Add alias to new index
        actions.append({
            "add": {
                "index": new_index,
                "alias": alias_name,
            }
        })
        logger.info(f"Adding alias {alias_name} to {new_index}")

        response = self.es.indices.update_aliases(body={"actions": actions})
        logger.info(f"Alias {alias_name} now points to {new_index}")
        return response

    def remove_alias(self, version: int | None = None) -> dict[str, Any]:
        """Remove alias from an index.

        Args:
            version: Index version (if None, removes from current target)

        Returns:
            Elasticsearch response
        """
        alias_name = self.get_alias_name()

        if version is not None:
            index_name = self.get_index_name(version)
        else:
            index_name = self.get_alias_target()
            if not index_name:
                logger.info(f"Alias {alias_name} doesn't exist, nothing to remove")
                return {"acknowledged": True, "not_found": True}

        logger.info(f"Removing alias {alias_name} from {index_name}")
        response = self.es.indices.delete_alias(index=index_name, name=alias_name)
        return response

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def list_indices(self) -> list[str]:
        """List all indices matching the prefix and environment.

        Returns:
            List of index names
        """
        pattern = f"{self.index_prefix}_{self.env}_v*"
        response = self.es.indices.get(index=pattern, ignore_unavailable=True)
        return list(response.keys())

    def get_index_info(self, version: int) -> dict[str, Any] | None:
        """Get index information including settings and mappings.

        Args:
            version: Index version number

        Returns:
            Index info dict or None if not found
        """
        index_name = self.get_index_name(version)
        try:
            response = self.es.indices.get(index=index_name)
            return response.get(index_name)
        except NotFoundError:
            return None

    def get_latest_version(self) -> int | None:
        """Find the latest index version number.

        Returns:
            Latest version number or None if no indices exist
        """
        indices = self.list_indices()
        if not indices:
            return None

        versions = []
        for index in indices:
            # Parse version from index name: rag_chunks_dev_v1 -> 1
            try:
                version_str = index.split("_v")[-1]
                versions.append(int(version_str))
            except (ValueError, IndexError):
                continue

        return max(versions) if versions else None

    def health_check(self) -> dict[str, Any]:
        """Check Elasticsearch cluster health.

        Returns:
            Cluster health information
        """
        return self.es.cluster.health()


__all__ = ["EsIndexManager"]
