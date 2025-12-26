"""ES-integrated document summarization service.

Retrieves documents from ES, generates summaries, and updates ES with results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from backend.config.settings import search_settings
from backend.llm_infrastructure.summarization.schemas import (
    ChapterSummary,
    ChunkSummary,
    DocumentMetadata,
    DocumentSummary,
)
from backend.services.document_summarization_service import (
    DocumentSummarizationService,
)

logger = logging.getLogger(__name__)


@dataclass
class EsDocumentInfo:
    """Document info from ES aggregation."""

    doc_id: str
    chunk_count: int


@dataclass
class EsChunk:
    """A chunk retrieved from ES."""

    es_id: str  # ES document _id
    chunk_id: str
    doc_id: str
    page: int
    content: str
    chapter: str = ""
    chunk_summary: str = ""
    chunk_keywords: list[str] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class EsSummarizationResult:
    """Result of ES document summarization."""

    doc_id: str
    chunks_updated: int
    document_summary: DocumentSummary
    chapter_summaries: dict[str, ChapterSummary]


class EsSummarizationService:
    """Service for summarizing documents stored in Elasticsearch.

    Workflow:
    1. Fetch all chunks for a doc_id
    2. Group chunks by page/chapter
    3. Generate summaries using DocumentSummarizationService
    4. Update ES documents with summaries
    """

    def __init__(
        self,
        es_client: Elasticsearch | None = None,
        index: str | None = None,
    ) -> None:
        # ES client
        if es_client is None:
            client_kwargs: dict[str, Any] = {
                "hosts": [search_settings.es_host],
            }
            if search_settings.es_user and search_settings.es_password:
                client_kwargs["basic_auth"] = (
                    search_settings.es_user,
                    search_settings.es_password,
                )
            es_client = Elasticsearch(**client_kwargs)

        self.es = es_client

        # Index name
        if index is None:
            index = f"{search_settings.es_index_prefix}_{search_settings.es_env}_current"
        self.index = index

        # Summarization service
        self._summarizer = DocumentSummarizationService()

    def count_documents(self) -> int:
        """Count total unique documents in the index.

        Returns:
            Total number of unique doc_ids.
        """
        result = self.es.search(
            index=self.index,
            body={
                "size": 0,
                "aggs": {
                    "unique_docs": {
                        "cardinality": {
                            "field": "doc_id.keyword"
                        }
                    }
                },
            },
        )
        return int(result["aggregations"]["unique_docs"]["value"])

    def list_documents(self, size: int = 100) -> list[EsDocumentInfo]:
        """List all documents in the index.

        Args:
            size: Maximum number of documents to return.

        Returns:
            List of EsDocumentInfo with doc_id and chunk count.
        """
        result = self.es.search(
            index=self.index,
            body={
                "size": 0,
                "aggs": {
                    "docs": {"terms": {"field": "doc_id.keyword", "size": size}}
                },
            },
        )

        docs = []
        for bucket in result["aggregations"]["docs"]["buckets"]:
            docs.append(
                EsDocumentInfo(
                    doc_id=bucket["key"],
                    chunk_count=bucket["doc_count"],
                )
            )
        return docs

    def get_document_chunks(
        self,
        doc_id: str,
        max_chunks: int = 500,
    ) -> list[EsChunk]:
        """Fetch all chunks for a document.

        Args:
            doc_id: Document identifier.
            max_chunks: Maximum chunks to fetch.

        Returns:
            List of EsChunk sorted by page.
        """
        result = self.es.search(
            index=self.index,
            body={
                "query": {"term": {"doc_id.keyword": doc_id}},
                "size": max_chunks,
                "sort": [{"page": "asc"}],
                "_source": {"excludes": ["embedding"]},
            },
        )

        chunks = []
        for hit in result["hits"]["hits"]:
            source = hit["_source"]
            chunks.append(
                EsChunk(
                    es_id=hit["_id"],
                    chunk_id=source.get("chunk_id", ""),
                    doc_id=source.get("doc_id", doc_id),
                    page=source.get("page", 0),
                    content=source.get("content", ""),
                    chapter=source.get("chapter", ""),
                    chunk_summary=source.get("chunk_summary", ""),
                    chunk_keywords=source.get("chunk_keywords"),
                    metadata={
                        "device_name": source.get("device_name", ""),
                        "doc_description": source.get("doc_description", ""),
                        "doc_type": source.get("doc_type", ""),
                    },
                )
            )
        return chunks

    def summarize_document(
        self,
        doc_id: str,
        *,
        update_es: bool = True,
        force_regenerate: bool = False,
    ) -> EsSummarizationResult:
        """Summarize a document from ES and optionally update ES.

        Args:
            doc_id: Document identifier.
            update_es: Whether to update ES with summaries.
            force_regenerate: Regenerate even if summaries exist.

        Returns:
            EsSummarizationResult with summaries.
        """
        # 1. Fetch chunks
        chunks = self.get_document_chunks(doc_id)
        if not chunks:
            raise ValueError(f"No chunks found for doc_id: {doc_id}")

        logger.info(f"Fetched {len(chunks)} chunks for doc_id={doc_id}")

        # 2. Check if already summarized (unless force)
        if not force_regenerate:
            has_summaries = any(c.chunk_summary for c in chunks)
            if has_summaries:
                logger.info(f"Document {doc_id} already has summaries, skipping")
                return self._build_result_from_existing(doc_id, chunks)

        # 3. Group chunks by page to reconstruct pages
        pages = self._chunks_to_pages(chunks)

        # 4. Run summarization pipeline
        result = self._summarizer.process_document(
            pages=pages,
            doc_id=doc_id,
        )

        # 5. Map summaries back to chunks
        chunk_summaries = self._map_summaries_to_chunks(
            chunks, result.chunk_summaries
        )

        # 6. Update ES if requested
        if update_es:
            self._update_es_chunks(
                chunks,
                chunk_summaries,
                result.chapter_summaries,
                result.document_summary,
            )

        return EsSummarizationResult(
            doc_id=doc_id,
            chunks_updated=len(chunks) if update_es else 0,
            document_summary=result.document_summary,
            chapter_summaries=result.chapter_summaries,
        )

    def _chunks_to_pages(self, chunks: list[EsChunk]) -> list[str]:
        """한페이지의 여러 chunk가 있을 때, 한 페이지로 합치는 코드
        """
        from collections import defaultdict

        page_contents: dict[int, list[str]] = defaultdict(list)
        for chunk in chunks:
            page_contents[chunk.page].append(chunk.content)

        # Sort by page number and join
        max_page = max(page_contents.keys()) if page_contents else 0
        pages = []
        for page_num in range(max_page + 1):
            if page_num in page_contents:
                pages.append("\n\n".join(page_contents[page_num]))
            else:
                pages.append("")

        return pages

    def _map_summaries_to_chunks(
        self,
        chunks: list[EsChunk],
        chunk_summaries: dict[str, ChunkSummary],
    ) -> dict[str, ChunkSummary]:
        """Map summarization chunk_ids to ES chunks.

        The summarization service uses p{page}_c{index} format.
        We need to map these back to ES chunk_ids.
        """
        # Group chunks by page
        from collections import defaultdict

        page_chunks: dict[int, list[EsChunk]] = defaultdict(list)
        for chunk in chunks:
            page_chunks[chunk.page].append(chunk)

        # Sort chunks within each page (maintain order)
        for page_num in page_chunks:
            page_chunks[page_num].sort(key=lambda c: c.chunk_id)

        # Map summary chunk_ids to ES chunk_ids
        result: dict[str, ChunkSummary] = {}
        for summary_id, summary in chunk_summaries.items():
            # Parse p{page}_c{index}
            try:
                parts = summary_id.split("_")
                page_num = int(parts[0][1:])  # Remove 'p' prefix
                chunk_idx = int(parts[1][1:])  # Remove 'c' prefix

                if page_num in page_chunks and chunk_idx < len(page_chunks[page_num]):
                    es_chunk = page_chunks[page_num][chunk_idx]
                    result[es_chunk.es_id] = summary
            except (ValueError, IndexError):
                continue

        return result

    def _update_es_chunks(
        self,
        chunks: list[EsChunk],
        chunk_summaries: dict[str, ChunkSummary],
        chapter_summaries: dict[str, ChapterSummary],
        document_summary: DocumentSummary,
    ) -> None:
        """Update ES documents with summaries."""
        # Extract document-level metadata
        doc_metadata = {}
        if document_summary.metadata:
            if document_summary.metadata.device_name:
                doc_metadata["device_name"] = document_summary.metadata.device_name
            if document_summary.metadata.doc_type:
                doc_metadata["doc_type"] = document_summary.metadata.doc_type

        # Build bulk update actions
        actions = []
        for chunk in chunks:
            update_doc: dict[str, Any] = {}

            # Chunk-level summary
            if chunk.es_id in chunk_summaries:
                cs = chunk_summaries[chunk.es_id]
                update_doc["chunk_summary"] = cs.summary
                if cs.keywords:
                    update_doc["chunk_keywords"] = cs.keywords

            # Chapter assignment (from chapter summaries)
            for chapter_title in chapter_summaries:
                # Simple matching: check if chunk is in this chapter
                # (In practice, this should use the chapter assignment from summarization)
                if chapter_title != "FRONT_MATTER" and chapter_title != "UNKNOWN":
                    if not update_doc.get("chapter"):
                        update_doc["chapter"] = chapter_title

            # Document-level metadata
            update_doc.update(doc_metadata)

            if update_doc:
                actions.append({
                    "_op_type": "update",
                    "_index": self.index,
                    "_id": chunk.es_id,
                    "doc": update_doc,
                })

        if actions:
            success, errors = bulk(self.es, actions, raise_on_error=False)
            logger.info(f"Updated {success} ES documents, {len(errors)} errors")

    def _build_result_from_existing(
        self,
        doc_id: str,
        chunks: list[EsChunk],
    ) -> EsSummarizationResult:
        """Build result from existing summaries in ES."""
        # Aggregate existing data
        chapters: dict[str, list[str]] = {}
        for chunk in chunks:
            chapter = chunk.chapter or "UNKNOWN"
            if chapter not in chapters:
                chapters[chapter] = []
            if chunk.chunk_summary:
                chapters[chapter].append(chunk.chunk_summary)

        # Build chapter summaries from existing data
        chapter_summaries = {
            title: ChapterSummary(
                chapter_title=title,
                summary="\n".join(summaries[:5]),  # First 5 chunk summaries
            )
            for title, summaries in chapters.items()
            if summaries
        }

        return EsSummarizationResult(
            doc_id=doc_id,
            chunks_updated=0,
            document_summary=DocumentSummary(summary="(using existing summaries)"),
            chapter_summaries=chapter_summaries,
        )


__all__ = [
    "EsSummarizationService",
    "EsSummarizationResult",
    "EsDocumentInfo",
    "EsChunk",
]
