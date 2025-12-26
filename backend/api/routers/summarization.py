"""Document Summarization API."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.services.document_summarization_service import (
    DocumentSummarizationService,
)
from backend.services.es_summarization_service import (
    EsSummarizationService,
)

router = APIRouter(prefix="/summarization", tags=["Summarization"])


# ─── Request/Response Models ───


class TextSummarizationRequest(BaseModel):
    """Request for single text summarization."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "이 문서는 장비의 PM 절차에 대해 설명합니다...",
                "max_length": 200,
            }
        }
    )

    text: str = Field(..., min_length=10, description="Text to summarize")
    max_length: int = Field(default=200, ge=50, le=2000, description="Max summary length hint")


class TextSummarizationResponse(BaseModel):
    """Response for single text summarization."""

    summary: str = Field(..., description="Generated summary")
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords")
    actions: list[str] = Field(default_factory=list, description="Action steps (if any)")
    warnings: list[str] = Field(default_factory=list, description="Warnings/cautions (if any)")
    compression_ratio: float = Field(..., description="Summary/original length ratio")


class DocumentSummarizationRequest(BaseModel):
    """Request for full document summarization."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pages": [
                    "목차\n1. 소개 .......... 2\n2. 절차 .......... 5\n",
                    "1. 소개\n이 문서는 장비의 PM 절차에 대해...",
                ],
                "doc_id": "SOP_001",
                "toc_page_indices": [0],
            }
        }
    )

    pages: list[str] = Field(..., min_length=1, description="List of page texts (0-indexed)")
    doc_id: str = Field(default="DOC_001", description="Document identifier")
    toc_page_indices: list[int] = Field(
        default=[0], description="Indices of TOC pages (0-based)"
    )
    chunk_size: int = Field(default=900, ge=100, le=4000, description="Chunk size for splitting")
    chunk_overlap: int = Field(default=120, ge=0, le=500, description="Overlap between chunks")

    @model_validator(mode="after")
    def validate_overlap_less_than_size(self) -> "DocumentSummarizationRequest":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        return self


class TOCEntryResponse(BaseModel):
    """TOC entry in response."""

    title: str
    start_page: int
    level: int


class ChunkSummaryResponse(BaseModel):
    """Chunk summary in response."""

    chunk_id: str
    summary: str
    keywords: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list, description="Action steps")
    warnings: list[str] = Field(default_factory=list, description="Warnings/cautions")


class ChapterSummaryResponse(BaseModel):
    """Chapter summary in response."""

    chapter_title: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class DocumentMetadataResponse(BaseModel):
    """Document metadata in response."""

    device_name: str | None = None
    doc_type: str | None = None
    doc_version: str | None = None
    doc_date: str | None = None


class DocumentSummaryResponse(BaseModel):
    """Document-level summary in response."""

    summary: str
    key_points: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    doc_metadata: DocumentMetadataResponse | None = Field(
        default=None, description="Extracted document metadata"
    )


class DocumentSummarizationResponse(BaseModel):
    """Response for full document summarization."""

    doc_id: str
    toc_entries: list[TOCEntryResponse] = Field(default_factory=list)
    page_offset: int = Field(default=0, description="Inferred page offset")
    num_pages: int
    num_chunks: int
    chunk_summaries: list[ChunkSummaryResponse] = Field(default_factory=list)
    chapter_summaries: list[ChapterSummaryResponse] = Field(default_factory=list)
    document_summary: DocumentSummaryResponse
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── ES-based Summarization Models ───


class EsDocumentInfoResponse(BaseModel):
    """Document info from ES."""

    doc_id: str = Field(..., description="Document identifier")
    chunk_count: int = Field(..., description="Number of chunks in ES")


class EsDocumentsListResponse(BaseModel):
    """List of documents in ES."""

    documents: list[EsDocumentInfoResponse] = Field(default_factory=list)
    total: int = Field(..., description="Total number of documents")


class EsSummarizationRequest(BaseModel):
    """Request for ES-based document summarization."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "force_regenerate": False,
                "update_es": True,
            }
        }
    )

    force_regenerate: bool = Field(
        default=False, description="Regenerate summaries even if they exist"
    )
    update_es: bool = Field(
        default=True, description="Update ES documents with summaries"
    )


class EsSummarizationResponse(BaseModel):
    """Response for ES-based document summarization."""

    doc_id: str
    chunks_updated: int = Field(..., description="Number of chunks updated in ES")
    document_summary: DocumentSummaryResponse
    chapter_summaries: list[ChapterSummaryResponse] = Field(default_factory=list)


# ─── Dependencies ───


def get_summarization_service() -> DocumentSummarizationService:
    """Get document summarization service instance."""
    return DocumentSummarizationService()


def get_es_summarization_service() -> EsSummarizationService:
    """Get ES summarization service instance."""
    return EsSummarizationService()


# ─── Endpoints ───


@router.post("/text", response_model=TextSummarizationResponse)
async def summarize_text(
    req: TextSummarizationRequest,
    service: DocumentSummarizationService = Depends(get_summarization_service),
):
    """Summarize a single text chunk.

    Returns summary with keywords, action steps, and warnings extracted.
    """
    try:
        result = service.summarize_text(req.text, max_length=req.max_length)

        return TextSummarizationResponse(
            summary=result.summary,
            keywords=result.keywords,
            actions=result.actions,
            warnings=result.warnings,
            compression_ratio=len(result.summary) / max(len(req.text), 1),
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/document", response_model=DocumentSummarizationResponse)
async def summarize_document(
    req: DocumentSummarizationRequest,
    service: DocumentSummarizationService = Depends(get_summarization_service),
):
    """Summarize a full document with hierarchical structure.

    Workflow:
    1. Parse TOC from specified page(s)
    2. Assign chapters to pages
    3. Split pages into chunks
    4. Generate chunk -> chapter -> document summaries
    """
    try:
        # Create service with request-specific settings
        svc = DocumentSummarizationService(
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
        )

        result = svc.process_document(
            pages=req.pages,
            doc_id=req.doc_id,
            toc_page_indices=req.toc_page_indices,
        )

        # Convert to response models
        toc_entries = [
            TOCEntryResponse(
                title=e.title,
                start_page=e.start_page,
                level=e.level,
            )
            for e in result.toc_entries
        ]

        chunk_summaries = [
            ChunkSummaryResponse(
                chunk_id=chunk_id,
                summary=cs.summary,
                keywords=cs.keywords,
                actions=cs.actions,
                warnings=cs.warnings,
            )
            for chunk_id, cs in result.chunk_summaries.items()
        ]

        chapter_summaries = [
            ChapterSummaryResponse(
                chapter_title=cs.chapter_title,
                summary=cs.summary,
                key_points=cs.key_points,
                keywords=cs.keywords,
            )
            for cs in result.chapter_summaries.values()
        ]

        # Build document metadata response if available
        doc_metadata = None
        if result.document_summary.metadata:
            meta = result.document_summary.metadata
            doc_metadata = DocumentMetadataResponse(
                device_name=meta.device_name,
                doc_type=meta.doc_type,
                doc_version=meta.doc_version,
                doc_date=meta.doc_date,
            )

        document_summary = DocumentSummaryResponse(
            summary=result.document_summary.summary,
            key_points=result.document_summary.key_points,
            keywords=result.document_summary.keywords,
            doc_metadata=doc_metadata,
        )

        return DocumentSummarizationResponse(
            doc_id=result.doc_id,
            toc_entries=toc_entries,
            page_offset=result.page_offset,
            num_pages=len(result.page_docs),
            num_chunks=len(result.chunk_docs),
            chunk_summaries=chunk_summaries,
            chapter_summaries=chapter_summaries,
            document_summary=document_summary,
            metadata={
                "chunk_size": req.chunk_size,
                "chunk_overlap": req.chunk_overlap,
            },
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─── ES-based Endpoints ───


@router.get("/es/documents", response_model=EsDocumentsListResponse)
async def list_es_documents(
    size: int = 100,
    service: EsSummarizationService = Depends(get_es_summarization_service),
):
    """List all documents in the ES index.

    Returns document IDs and their chunk counts.
    The 'total' field represents the total number of unique documents in the index,
    regardless of the 'size' parameter.
    """
    try:
        # Get actual total count
        total = service.count_documents()

        # Get document list (up to size)
        docs = service.list_documents(size=size)

        return EsDocumentsListResponse(
            documents=[
                EsDocumentInfoResponse(doc_id=d.doc_id, chunk_count=d.chunk_count)
                for d in docs
            ],
            total=total,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/es/summarize/{doc_id}", response_model=EsSummarizationResponse)
async def summarize_es_document(
    doc_id: str,
    req: EsSummarizationRequest = EsSummarizationRequest(),
    service: EsSummarizationService = Depends(get_es_summarization_service),
):
    """Summarize a document stored in Elasticsearch.

    Workflow:
    1. Fetch all chunks for the doc_id from ES
    2. Reconstruct pages from chunks
    3. Generate hierarchical summaries (chunk -> chapter -> document)
    4. Update ES documents with summaries (if update_es=True)
    """
    try:
        result = service.summarize_document(
            doc_id=doc_id,
            update_es=req.update_es,
            force_regenerate=req.force_regenerate,
        )

        # Build document metadata response if available
        doc_metadata = None
        if result.document_summary.metadata:
            meta = result.document_summary.metadata
            doc_metadata = DocumentMetadataResponse(
                device_name=meta.device_name,
                doc_type=meta.doc_type,
                doc_version=meta.doc_version,
                doc_date=meta.doc_date,
            )

        document_summary = DocumentSummaryResponse(
            summary=result.document_summary.summary,
            key_points=result.document_summary.key_points,
            keywords=result.document_summary.keywords,
            doc_metadata=doc_metadata,
        )

        chapter_summaries = [
            ChapterSummaryResponse(
                chapter_title=cs.chapter_title,
                summary=cs.summary,
                key_points=cs.key_points,
                keywords=cs.keywords,
            )
            for cs in result.chapter_summaries.values()
        ]

        return EsSummarizationResponse(
            doc_id=result.doc_id,
            chunks_updated=result.chunks_updated,
            document_summary=document_summary,
            chapter_summaries=chapter_summaries,
        )

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


__all__ = [
    "router",
    "TextSummarizationRequest",
    "TextSummarizationResponse",
    "DocumentSummarizationRequest",
    "DocumentSummarizationResponse",
    "EsDocumentsListResponse",
    "EsSummarizationRequest",
    "EsSummarizationResponse",
]
