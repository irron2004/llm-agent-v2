"""Assets API for serving document page images from MinIO."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from backend.services.storage import get_page_image, sanitize_doc_id

router = APIRouter(prefix="/assets", tags=["Assets"])
logger = logging.getLogger(__name__)


@router.get("/docs/{doc_id}/pages/{page}")
async def get_document_page_image(doc_id: str, page: int):
    """Get a document page image from MinIO.

    Args:
        doc_id: Document identifier
        page: Page number (1-indexed)

    Returns:
        PNG image response

    Raises:
        HTTPException 404: If image not found
        HTTPException 500: If storage error
    """
    if page < 1:
        raise HTTPException(status_code=400, detail="Page number must be >= 1")

    # Sanitize doc_id for security
    safe_doc_id = sanitize_doc_id(doc_id)

    try:
        image_data = get_page_image(safe_doc_id, page)

        if image_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Page image not found: doc_id={doc_id}, page={page}",
            )

        return Response(
            content=image_data,
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=86400",  # Cache for 1 day
                "Content-Disposition": f'inline; filename="{safe_doc_id}_page_{page}.png"',
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching page image: doc_id=%s, page=%d, error=%s", doc_id, page, e)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve page image",
        ) from e


__all__ = ["router"]
