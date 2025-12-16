"""Elasticsearch document schema for RAG chunks.

This module provides the data class for ES documents and conversion utilities
from Section objects produced by DocumentIngestService.

Usage:
    from backend.llm_infrastructure.elasticsearch.document import EsChunkDocument

    doc = EsChunkDocument.from_section(
        section=section,
        doc_id="doc_001",
        chunk_index=0,
        embedding=[0.1, 0.2, ...],
        doc_type="sop",
    )
    es_doc = doc.to_es_doc()
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.services.ingest.document_ingest_service import Section


@dataclass
class EsChunkDocument:
    """Elasticsearch RAG chunk document schema.

    Maps to the fields defined in mappings.get_rag_chunks_mapping().
    """

    # Primary Keys / Location
    doc_id: str
    chunk_id: str
    page: int

    # Text Fields
    content: str
    search_text: str

    # Vector Embedding
    embedding: list[float] = field(default_factory=list)

    # Metadata / Filter Fields
    lang: str = "ko"
    doc_type: str = "generic"
    tenant_id: str = ""
    project_id: str = ""
    pipeline_version: str = "v1"
    content_hash: str = ""

    # Optional Fields
    page_image_path: str | None = None
    bbox: dict[str, Any] | None = None
    quality_score: float | None = None
    summary: str | None = None
    caption: str | None = None
    tags: list[str] = field(default_factory=list)

    # Timestamps
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        """Set default values after initialization."""
        if not self.content_hash:
            self.content_hash = compute_content_hash(self.content)
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_es_doc(self) -> dict[str, Any]:
        """Convert to Elasticsearch document dictionary.

        Returns:
            Dict suitable for ES bulk indexing.
        """
        doc: dict[str, Any] = {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "page": self.page,
            "content": self.content,
            "search_text": self.search_text,
            "embedding": self.embedding,
            "lang": self.lang,
            "doc_type": self.doc_type,
            "tenant_id": self.tenant_id,
            "project_id": self.project_id,
            "pipeline_version": self.pipeline_version,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

        # Add optional fields only if set
        if self.page_image_path:
            doc["page_image_path"] = self.page_image_path
        if self.bbox:
            doc["bbox"] = self.bbox
        if self.quality_score is not None:
            doc["quality_score"] = self.quality_score
        if self.summary:
            doc["summary"] = self.summary
        if self.caption:
            doc["caption"] = self.caption
        if self.tags:
            doc["tags"] = self.tags

        return doc

    @classmethod
    def from_section(
        cls,
        section: "Section",
        doc_id: str,
        chunk_index: int,
        embedding: list[float],
        *,
        lang: str = "ko",
        doc_type: str = "generic",
        tenant_id: str = "",
        project_id: str = "",
        pipeline_version: str = "v1",
        vlm_model: str = "",
        tags: list[str] | None = None,
    ) -> "EsChunkDocument":
        """Create EsChunkDocument from a Section.

        Args:
            section: Section object from DocumentIngestService.
            doc_id: Document identifier.
            chunk_index: Zero-based chunk index within the document.
            embedding: Embedding vector.
            lang: Language code.
            doc_type: Document type (sop, ts, guide, etc.).
            tenant_id: Tenant identifier.
            project_id: Project identifier.
            pipeline_version: Ingestion pipeline version.
            vlm_model: VLM model used for parsing (for metadata).
            tags: Optional tags for the chunk.

        Returns:
            EsChunkDocument instance.
        """
        content = section.text
        title = section.title or ""
        search_text = build_search_text(content, title, tags)
        page = section.page_start or 0
        chunk_id = f"{doc_id}#{chunk_index:04d}"

        doc_tags = list(tags) if tags else []
        if vlm_model:
            doc_tags.append(f"vlm:{vlm_model}")

        # Extract page_image_path from section metadata if available
        section_meta = section.metadata or {}
        page_image_path = section_meta.get("page_image_path")

        return cls(
            doc_id=doc_id,
            chunk_id=chunk_id,
            page=page,
            content=content,
            search_text=search_text,
            embedding=embedding,
            lang=lang,
            doc_type=doc_type,
            tenant_id=tenant_id,
            project_id=project_id,
            pipeline_version=pipeline_version,
            tags=doc_tags,
            page_image_path=page_image_path,
        )


def build_search_text(
    content: str,
    title: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Build combined search text for BM25 retrieval.

    Combines content, title, and tags into a single searchable text field.

    Args:
        content: Main content text.
        title: Section title (optional).
        tags: Tags (optional).

    Returns:
        Combined search text.
    """
    parts = []
    if title:
        parts.append(title)
    parts.append(content)
    if tags:
        parts.append(" ".join(tags))
    return " ".join(parts)


def compute_content_hash(content: str) -> str:
    """Compute content hash for deduplication.

    Args:
        content: Text content.

    Returns:
        16-character hex hash.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "EsChunkDocument",
    "build_search_text",
    "compute_content_hash",
]
