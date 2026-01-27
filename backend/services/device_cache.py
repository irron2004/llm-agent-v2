"""Device and DocType cache for auto-parsing.

This module provides a local cache of device names and doc types
fetched from Elasticsearch. The cache is initialized on first use
and can be refreshed manually.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CATALOG_PATH = _REPO_ROOT / "data" / "device_catalog.json"


def _unique_doc_count_agg(field: str = "doc_id") -> Dict[str, Any]:
    """Cardinality aggregation for unique document count.

    Falls back to base doc id when doc_id includes chunk suffixes.
    """
    return {
        "cardinality": {
            "script": {
                "lang": "painless",
                "source": (
                    "def v = doc.containsKey(params.f) && !doc[params.f].empty ? doc[params.f].value : null;"
                    "if (v == null) return null;"
                    "int idx = v.indexOf('#');"
                    "if (idx == -1) idx = v.indexOf(':');"
                    "return idx > 0 ? v.substring(0, idx) : v;"
                ),
                "params": {"f": field},
            }
        }
    }


def _compute_visible_devices(devices: List[Dict[str, Any]], limit: int = 10) -> List[str]:
    """Compute visible device list based on document counts (top N)."""
    ranked = []
    for item in devices or []:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        try:
            count = int(item.get("doc_count", 0))
        except Exception:
            count = 0
        ranked.append((name, count))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[: max(0, int(limit))]]


def _load_catalog_from_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        devices = data.get("devices") or []
        doc_types = data.get("doc_types") or []
        vis = data.get("vis") or data.get("visible_devices") or []
        if not isinstance(devices, list) or not isinstance(doc_types, list) or not isinstance(vis, list):
            return None
        return {"devices": devices, "doc_types": doc_types, "vis": vis}
    except Exception:
        logger.exception("Failed to load device catalog file: %s", path)
        return None


def _save_catalog_to_file(path: Path, devices: List[Dict[str, Any]], doc_types: List[Dict[str, Any]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        visible_devices = _compute_visible_devices(devices)
        payload = {
            "devices": devices,
            "doc_types": doc_types,
            "vis": visible_devices,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Failed to save device catalog file: %s", path)


class DeviceCache:
    """Singleton cache for device names and doc types."""

    _instance: Optional["DeviceCache"] = None

    def __new__(cls) -> "DeviceCache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._devices: List[Dict[str, Any]] = []
            cls._instance._doc_types: List[Dict[str, Any]] = []
            cls._instance._visible_devices: List[str] = []
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
    def visible_devices(self) -> List[str]:
        """Get list of visible devices (top by doc count)."""
        return self._visible_devices

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
        # 1) Try local catalog first
        local_catalog = _load_catalog_from_file(_CATALOG_PATH)
        if local_catalog is not None:
            self._devices = local_catalog.get("devices", [])
            self._doc_types = local_catalog.get("doc_types", [])
            self._visible_devices = local_catalog.get("vis", []) or _compute_visible_devices(self._devices)
            self._initialized = True
            logger.info(
                "Device catalog loaded from file: %s (devices=%d, doc_types=%d)",
                _CATALOG_PATH,
                len(self._devices),
                len(self._doc_types),
            )
            return True

        # 2) Fallback to ES on first creation only
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
                            "unique_docs": _unique_doc_count_agg("doc_id")
                        }
                    },
                    "doc_types": {
                        "terms": {
                            "field": "doc_type",
                            "size": 50,
                            "order": {"_count": "desc"},
                        },
                        "aggs": {
                            "unique_docs": _unique_doc_count_agg("doc_id")
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

            self._visible_devices = _compute_visible_devices(self._devices)
            self._initialized = True
            _save_catalog_to_file(_CATALOG_PATH, self._devices, self._doc_types)
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
            "vis": self._visible_devices,
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
