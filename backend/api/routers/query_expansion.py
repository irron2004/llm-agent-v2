"""Query Expansion Service API."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend.api.dependencies import get_query_expander
from backend.llm_infrastructure.query_expansion.base import BaseQueryExpander
from backend.llm_infrastructure.query_expansion.prompts import list_prompt_templates

router = APIRouter(prefix="/expand-query", tags=["Query Expansion Service"])


class ExpandQueryRequest(BaseModel):
    """Request body for query expansion."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "반도체 장비 PM 점검 방법",
                "n": 3,
                "include_original": True,
            }
        }
    )

    query: str = Field(..., description="Original query to expand")
    n: int = Field(default=3, ge=1, le=10, description="Number of expanded queries to generate")
    include_original: bool = Field(
        default=True, description="Include original query in results"
    )


class ExpandQueryResponse(BaseModel):
    """Response from query expansion."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "original_query": "반도체 장비 PM 점검 방법",
                "expanded_queries": [
                    "반도체 장비 예방 정비 절차",
                    "반도체 제조 장비 유지보수 가이드",
                    "반도체 생산 설비 정기 점검 항목",
                ],
                "all_queries": [
                    "반도체 장비 PM 점검 방법",
                    "반도체 장비 예방 정비 절차",
                    "반도체 제조 장비 유지보수 가이드",
                    "반도체 생산 설비 정기 점검 항목",
                ],
                "total": 4,
                "include_original": True,
            }
        }
    )

    original_query: str = Field(..., description="Original query")
    expanded_queries: List[str] = Field(..., description="Generated expanded queries")
    all_queries: List[str] = Field(
        ..., description="All queries (original + expanded if include_original)"
    )
    total: int = Field(..., description="Total number of queries")
    include_original: bool = Field(..., description="Whether original was included")


@router.post("", response_model=ExpandQueryResponse)
async def expand_query(
    request: ExpandQueryRequest,
    expander: BaseQueryExpander = Depends(get_query_expander),
):
    """Expand a query into multiple related search queries.

    Uses LLM to generate semantically related queries that can improve
    retrieval recall by searching with multiple query variations.
    """
    try:
        expanded = expander.expand(
            query=request.query,
            n=request.n,
            include_original=request.include_original,
        )

        return ExpandQueryResponse(
            original_query=expanded.original_query,
            expanded_queries=expanded.expanded_queries,
            all_queries=expanded.get_all_queries(),
            total=len(expanded),
            include_original=expanded.include_original,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/prompts", response_model=dict)
async def list_prompts():
    """List available prompt templates for query expansion."""
    from backend.llm_infrastructure.query_expansion import QueryExpanderRegistry

    return {
        "methods": QueryExpanderRegistry.list_methods(),
        "prompt_templates": list_prompt_templates(),
        "descriptions": {
            "general_mq_v1": "General multi-query expansion (English)",
            "general_mq_v1_ko": "General multi-query expansion (Korean)",
            "technical_mq_v1": "Technical/domain-specific expansion",
            "semiconductor_mq_v1": "Semiconductor domain expansion (Korean)",
        },
    }


__all__ = [
    "router",
    "ExpandQueryRequest",
    "ExpandQueryResponse",
]
