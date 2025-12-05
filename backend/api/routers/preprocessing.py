"""Preprocessing API router."""

from typing import Callable

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from backend.api.dependencies import get_preprocessor_factory
from backend.llm_infrastructure.preprocessing.base import BasePreprocessor

router = APIRouter(prefix="/preprocessing", tags=["preprocessing"])


class PreprocessRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "pm 2 alarm (1234) spec out helium leak 4.0x10^-9 mt 5",
                "level": "L3"
            }
        }
    )

    text: str
    level: str | None = None


class PreprocessResponse(BaseModel):
    processed_text: str


@router.post("/apply", response_model=PreprocessResponse)
async def apply_preprocessing(
    body: PreprocessRequest,
    preprocessor_factory: Callable[[str | None], BasePreprocessor] = Depends(
        get_preprocessor_factory
    ),
):
    """Apply preprocessor to a single text. Level can be overridden per request."""
    preprocessor = preprocessor_factory(body.level)
    processed = list(preprocessor.preprocess([body.text]))
    if not processed:
        raise HTTPException(status_code=400, detail="Empty result after preprocessing")
    return PreprocessResponse(processed_text=processed[0])
