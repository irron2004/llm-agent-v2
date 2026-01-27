"""Devices API router.

Provides device list for device selection UI.
Uses ES terms aggregation to get unique device names from indexed documents.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api.dependencies import get_search_service
from backend.services.device_cache import ensure_device_cache_initialized
from backend.services.es_search_service import EsSearchService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["devices"])


class DeviceInfo(BaseModel):
    """Device information with document count."""
    name: str = Field(..., description="Device name")
    doc_count: int = Field(..., description="Number of documents for this device")


class DeviceListResponse(BaseModel):
    """Response containing list of available devices."""
    devices: list[DeviceInfo] = Field(default_factory=list, description="List of devices")
    total: int = Field(..., description="Total number of unique devices")


class DocTypeInfo(BaseModel):
    """Document type information with document count."""
    name: str = Field(..., description="Document type name")
    doc_count: int = Field(..., description="Number of documents for this doc type")


class DeviceCatalogResponse(BaseModel):
    """Response containing device and doc type catalog."""
    devices: list[DeviceInfo] = Field(default_factory=list, description="List of devices")
    doc_types: list[DocTypeInfo] = Field(default_factory=list, description="List of doc types")
    vis: list[str] = Field(default_factory=list, description="Visible device names")


@router.get("/devices", response_model=DeviceListResponse)
async def list_devices(
    limit: int = 100,
    search_service: EsSearchService = Depends(get_search_service),
) -> DeviceListResponse:
    """Get list of available devices from indexed documents.

    Returns unique device names with their document counts,
    sorted by document count (descending).

    Args:
        limit: Maximum number of devices to return (default: 100)
        search_service: ES search service (injected)

    Returns:
        DeviceListResponse with list of devices and total count.
    """
    # Verify we have ES engine access
    if not hasattr(search_service, 'es_engine') or search_service.es_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Elasticsearch engine not available"
        )

    es = search_service.es_engine.es
    index = search_service.es_engine.index_name

    # Terms aggregation on device_name field with cardinality sub-aggregation for unique doc_id count
    agg_query = {
        "size": 0,
        "aggs": {
            "devices": {
                "terms": {
                    "field": "device_name",
                    "size": limit,
                    "order": {"_count": "desc"},
                },
                "aggs": {
                    "unique_docs": {
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
                                "params": {"f": "doc_id"},
                            }
                        }
                    }
                }
            }
        }
    }

    try:
        result = es.search(index=index, body=agg_query)
        buckets = result.get("aggregations", {}).get("devices", {}).get("buckets", [])

        devices = [
            DeviceInfo(
                name=bucket["key"],
                doc_count=bucket.get("unique_docs", {}).get("value", bucket["doc_count"])
            )
            for bucket in buckets
            if bucket["key"]  # Filter out empty device names
        ]

        return DeviceListResponse(
            devices=devices,
            total=len(devices),
        )
    except Exception as e:
        logger.error(f"Failed to fetch device list: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch device list: {str(e)}"
        )


@router.get("/device-catalog", response_model=DeviceCatalogResponse)
async def get_device_catalog(
    search_service: EsSearchService = Depends(get_search_service),
) -> DeviceCatalogResponse:
    """Get device/doc type catalog using local cache (ES only on first load)."""
    cache = ensure_device_cache_initialized(search_service)

    devices: list[DeviceInfo] = []
    for d in cache.devices or []:
        name = str(d.get("name", "")).strip()
        if not name:
            continue
        try:
            doc_count = int(d.get("doc_count", 0))
        except Exception:
            doc_count = 0
        devices.append(DeviceInfo(name=name, doc_count=doc_count))

    doc_types: list[DocTypeInfo] = []
    for d in cache.doc_types or []:
        name = str(d.get("name", "")).strip()
        if not name:
            continue
        try:
            doc_count = int(d.get("doc_count", 0))
        except Exception:
            doc_count = 0
        doc_types.append(DocTypeInfo(name=name, doc_count=doc_count))

    visible_devices = [str(name).strip() for name in cache.visible_devices or [] if str(name).strip()]

    return DeviceCatalogResponse(devices=devices, doc_types=doc_types, vis=visible_devices)
