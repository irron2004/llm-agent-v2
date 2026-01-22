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
                            "field": "doc_id"
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
