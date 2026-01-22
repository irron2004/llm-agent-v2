"""Device and DocType cache for auto-parsing.

This module provides a local cache of device names and doc types
fetched from Elasticsearch. The cache is initialized on first use
and can be refreshed manually.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DeviceCache:
    """Singleton cache for device names and doc types."""

    _instance: Optional["DeviceCache"] = None

    def __new__(cls) -> "DeviceCache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._devices: List[Dict[str, Any]] = []
            cls._instance._doc_types: List[Dict[str, Any]] = []
            cls._instance._initialized = False
        return cls._instance

    @property
    def devices(self) -> List[Dict[str, Any]]:
        """Get cached device list."""
        return self._devices

    @property
    def doc_types(self) -> List[Dict[str, Any]]:
        """Get cached doc type list."""
        return self._doc_types

    @property
    def device_names(self) -> List[str]:
        """Get list of device names only."""
        return [d.get("name", "") for d in self._devices if d.get("name")]

    @property
    def doc_type_names(self) -> List[str]:
        """Get list of doc type names only."""
        return [d.get("name", "") for d in self._doc_types if d.get("name")]

    @property
    def is_initialized(self) -> bool:
        """Check if cache has been initialized."""
        return self._initialized

    def initialize(self, search_service: Any) -> bool:
        """Initialize cache from Elasticsearch.

        Args:
            search_service: SearchService instance with es_engine.

        Returns:
            True if initialization successful, False otherwise.
        """
        if not hasattr(search_service, 'es_engine') or search_service.es_engine is None:
            logger.warning("ES engine not available for device cache initialization")
            return False

        try:
            es = search_service.es_engine.es
            index = search_service.es_engine.index_name

            agg_query = {
                "size": 0,
                "aggs": {
                    "devices": {
                        "terms": {
                            "field": "device_name",
                            "size": 500,  # Get more devices for better matching
                            "order": {"_count": "desc"},
                        },
                        "aggs": {
                            "unique_docs": {
                                "cardinality": {
                                    "field": "doc_id"
                                }
                            }
                        }
                    },
                    "doc_types": {
                        "terms": {
                            "field": "doc_type",
                            "size": 50,
                            "order": {"_count": "desc"},
                        },
                        "aggs": {
                            "unique_docs": {
                                "cardinality": {
                                    "field": "doc_id"
                                }
                            }
                        }
                    },
                }
            }

            result = es.search(index=index, body=agg_query)

            device_buckets = result.get("aggregations", {}).get("devices", {}).get("buckets", [])
            doc_type_buckets = result.get("aggregations", {}).get("doc_types", {}).get("buckets", [])

            self._devices = [
                {
                    "name": bucket["key"],
                    "doc_count": bucket.get("unique_docs", {}).get("value", bucket["doc_count"])
                }
                for bucket in device_buckets
                if bucket.get("key")
            ]

            self._doc_types = [
                {
                    "name": bucket["key"],
                    "doc_count": bucket.get("unique_docs", {}).get("value", bucket["doc_count"])
                }
                for bucket in doc_type_buckets
                if bucket.get("key")
            ]

            self._initialized = True
            logger.info(
                f"Device cache initialized: {len(self._devices)} devices, "
                f"{len(self._doc_types)} doc types"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize device cache: {e}")
            return False

    def refresh(self, search_service: Any) -> bool:
        """Refresh cache from Elasticsearch.

        Args:
            search_service: SearchService instance with es_engine.

        Returns:
            True if refresh successful, False otherwise.
        """
        self._initialized = False
        return self.initialize(search_service)

    def get_all(self) -> Dict[str, Any]:
        """Get all cached data.

        Returns:
            Dict with 'devices' and 'doc_types' lists.
        """
        return {
            "devices": self._devices,
            "doc_types": self._doc_types,
        }


# Global singleton instance
device_cache = DeviceCache()


def get_device_cache() -> DeviceCache:
    """Get the global device cache instance."""
    return device_cache


def ensure_device_cache_initialized(search_service: Any) -> DeviceCache:
    """Ensure device cache is initialized, initialize if needed.

    Args:
        search_service: SearchService instance with es_engine.

    Returns:
        Initialized DeviceCache instance.
    """
    cache = get_device_cache()
    if not cache.is_initialized:
        cache.initialize(search_service)
    return cache
