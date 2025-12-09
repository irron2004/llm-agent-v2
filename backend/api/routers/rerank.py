"""Rerank Service API."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from backend.api.dependencies import get_reranker
from backend.llm_infrastructure.reranking.base import BaseReranker
from backend.llm_infrastructure.retrieval.base import RetrievalResult

router = APIRouter(prefix="/rerank", tags=["Rerank Service"])


class DocumentInput(BaseModel):
    """Input document for reranking."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "doc_id": "doc-001",
                "content": "Machine learning is a subset of artificial intelligence.",
                "score": 0.85,
            }
        }
    )

    doc_id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content text")
    score: float = Field(default=0.0, description="Original retrieval score")
    metadata: dict = Field(default_factory=dict, description="Optional metadata")


class RerankRequest(BaseModel):
    """Request body for reranking."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What is machine learning?",
                "documents": [
                    {
                        "doc_id": "doc-001",
                        "content": "Machine learning is a subset of artificial intelligence.",
                        "score": 0.85,
                    },
                    {
                        "doc_id": "doc-002",
                        "content": "Python is a programming language.",
                        "score": 0.75,
                    },
                ],
                "top_k": 5,
            }
        }
    )

    query: str = Field(..., description="Query to rerank documents against")
    documents: List[DocumentInput] = Field(..., description="Documents to rerank")
    top_k: Optional[int] = Field(
        default=None, ge=1, description="Number of results to return (None = all)"
    )


class RerankResultItem(BaseModel):
    """Single reranked document result."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rank": 1,
                "doc_id": "doc-001",
                "content": "Machine learning is a subset of artificial intelligence.",
                "score": 0.95,
                "original_score": 0.85,
            }
        }
    )

    rank: int = Field(..., description="Rank after reranking (1-based)")
    doc_id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    score: float = Field(..., description="Reranking score")
    original_score: float = Field(..., description="Original retrieval score")
    metadata: dict = Field(default_factory=dict, description="Document metadata")


class RerankResponse(BaseModel):
    """Response from reranking."""

    query: str = Field(..., description="Original query")
    results: List[RerankResultItem] = Field(..., description="Reranked documents")
    total: int = Field(..., description="Total number of reranked documents")
    model: str = Field(..., description="Reranking model used")


@router.post("", response_model=RerankResponse)
async def rerank_documents(
    request: RerankRequest,
    reranker: BaseReranker = Depends(get_reranker),
):
    """Rerank documents based on query relevance.

    Takes a query and a list of documents, returns documents reranked
    by relevance to the query using a cross-encoder model.
    """
    try:
        # Build a map of original scores from input for fallback
        original_scores = {doc.doc_id: doc.score for doc in request.documents}

        # Convert input documents to RetrievalResult
        retrieval_results = [
            RetrievalResult(
                doc_id=doc.doc_id,
                content=doc.content,
                score=doc.score,
                metadata=doc.metadata,
                raw_text=doc.content,  # Use content as raw_text for reranking
            )
            for doc in request.documents
        ]

        # Perform reranking
        reranked = reranker.rerank(
            query=request.query,
            results=retrieval_results,
            top_k=request.top_k,
        )

        # Convert to response format
        result_items = [
            RerankResultItem(
                rank=idx + 1,
                doc_id=result.doc_id,
                content=result.content,
                score=result.score,
                # Fallback to input score if reranker didn't set original_score
                original_score=(
                    result.metadata.get("original_score")
                    if result.metadata and "original_score" in result.metadata
                    else original_scores.get(result.doc_id, 0.0)
                ),
                metadata={
                    k: v
                    for k, v in (result.metadata or {}).items()
                    if k not in ("original_score", "rerank_model")
                },
            )
            for idx, result in enumerate(reranked)
        ]

        return RerankResponse(
            query=request.query,
            results=result_items,
            total=len(result_items),
            model=reranker.model_name if hasattr(reranker, "model_name") else "unknown",
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/models", response_model=dict)
async def list_rerank_models():
    """List available reranking models and methods."""
    from backend.llm_infrastructure.reranking import RerankerRegistry

    return {
        "methods": RerankerRegistry.list_methods(),
        "recommended_models": [
            {
                "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "description": "Fast, good quality (English)",
                "size": "small",
            },
            {
                "name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
                "description": "Balanced speed/quality (English)",
                "size": "medium",
            },
            {
                "name": "BAAI/bge-reranker-base",
                "description": "Multilingual support",
                "size": "base",
            },
            {
                "name": "BAAI/bge-reranker-v2-m3",
                "description": "High quality multilingual",
                "size": "large",
            },
        ],
    }


__all__ = [
    "router",
    "RerankRequest",
    "RerankResponse",
    "RerankResultItem",
    "DocumentInput",
]