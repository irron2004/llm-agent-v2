"""Document summarization service.

Orchestrates hierarchical summarization: chunk -> chapter -> document.
Supports VLM-extracted pages with TOC-based chapter grouping.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from backend.config.settings import summarization_settings, vllm_settings

logger = logging.getLogger(__name__)
from backend.llm_infrastructure.llm import get_llm
from backend.llm_infrastructure.llm.base import BaseLLM
from backend.llm_infrastructure.summarization.adapters.llm import LLMSummarizer
from backend.llm_infrastructure.summarization.schemas import (
    ChapterSummary,
    ChunkSummary,
    DocumentSummary,
    TOCEntry,
)
from backend.llm_infrastructure.summarization.toc_parser import (
    match_toc_to_pages,
    parse_toc,
)


@dataclass
class PageDocument:
    """A page with metadata."""

    page_number: int
    content: str
    chapter_title: str = "UNKNOWN"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkDocument:
    """A chunk with metadata."""

    chunk_id: str
    content: str
    page_number: int
    chunk_index: int
    chapter_title: str
    summary: ChunkSummary | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentSummarizationResult:
    """Complete result of document summarization."""

    doc_id: str
    toc_entries: list[TOCEntry]
    page_offset: int
    page_docs: list[PageDocument]
    chunk_docs: list[ChunkDocument]
    chunk_summaries: dict[str, ChunkSummary]
    chapter_summaries: dict[str, ChapterSummary]
    document_summary: DocumentSummary


class DocumentSummarizationService:
    """Hierarchical document summarization service.

    Workflow:
    1. Parse TOC from first page(s)
    2. Infer page offset (TOC page numbers vs actual indices)
    3. Assign chapters to pages
    4. Split pages into chunks
    5. Generate chunk summaries
    6. Generate chapter summaries (from chunk summaries)
    7. Generate document summary (from chapter summaries)
    """

    def __init__(
        self,
        llm_method: str | None = None,
        llm_version: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        prompt_version: str | None = None,
    ) -> None:
        # Use summarization_settings as defaults
        self.llm_method = llm_method or summarization_settings.llm_method
        self.llm_version = llm_version or summarization_settings.llm_version
        self.chunk_size = chunk_size if chunk_size is not None else summarization_settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else summarization_settings.chunk_overlap
        self.prompt_version = prompt_version or summarization_settings.prompt_version

        # Initialize LLM
        self._llm: BaseLLM = get_llm(
            self.llm_method,
            version=self.llm_version,
            base_url=vllm_settings.base_url,
            model=vllm_settings.model_name,
            temperature=0,
            max_tokens=vllm_settings.max_tokens,
            timeout=vllm_settings.timeout,
        )

        # Initialize summarizer
        self._summarizer:LLMSummarizer = LLMSummarizer(
            llm_method=self.llm_method,
            llm_version=self.llm_version,
            prompt_version=self.prompt_version,
        )

    def process_document(
        self,
        pages: list[str],
        doc_id: str = "DOC_001",
        toc_page_indices: list[int] | None = None,
    ) -> DocumentSummarizationResult:
        """Process a document through the full summarization pipeline.

        Args:
            pages: List of page texts (0-indexed).
            doc_id: Document identifier.
            toc_page_indices: Indices of TOC pages (default: [0]).

        Returns:
            DocumentSummarizationResult with all summaries.
        """
        if toc_page_indices is None:
            toc_page_indices = [0, 1, 2]

        total_start = time.time()

        # 1. 지정한 페이지에서 목차 추출
        t0 = time.time()
        toc_text = "\n".join(pages[i] for i in toc_page_indices if i < len(pages))
        toc_entries = parse_toc(toc_text, llm=self._llm)
        logger.info(f"[1/7] TOC 파싱 완료: {len(toc_entries)}개 항목, {time.time() - t0:.2f}s")

        # 2. 모든 페이지를 순회하면서 각 TOC 제목이 실제로 나타나는 페이지 찾기
        t0 = time.time()
        toc_entries = match_toc_to_pages(toc_entries=toc_entries, pages=pages, threshold=70, toc_page_indices=toc_page_indices)
        logger.info(f"[2/7] TOC-페이지 매칭 완료: {time.time() - t0:.2f}s")

        # 3. toc_entries의 start_page로 챕터 할당 (offset=0, 이미 실제 페이지 번호)
        t0 = time.time()
        page_docs = self._assign_chapters_to_pages(pages, toc_entries, offset=0)
        logger.info(f"[3/7] 챕터 할당 완료: {len(page_docs)}개 페이지, {time.time() - t0:.2f}s")

        # 4. Split into chunks
        t0 = time.time()
        chunk_docs = self._split_pages_to_chunks(page_docs)
        logger.info(f"[4/7] 청크 분할 완료: {len(chunk_docs)}개 청크, {time.time() - t0:.2f}s")

        # 5. Generate chunk summaries (병렬 처리)
        t0 = time.time()
        chunk_summaries = self._summarize_chunks_parallel(chunk_docs)
        logger.info(f"[5/7] 청크 요약 완료: {len(chunk_summaries)}개, {time.time() - t0:.2f}s")

        # 6. Generate chapter summaries
        t0 = time.time()
        chapter_summaries = self._summarize_chapters(chunk_docs, chunk_summaries)
        logger.info(f"[6/7] 챕터 요약 완료: {len(chapter_summaries)}개, {time.time() - t0:.2f}s")

        # 7. Generate document summary
        t0 = time.time()
        document_summary = self._summarize_document(chapter_summaries)
        logger.info(f"[7/7] 문서 요약 완료: {time.time() - t0:.2f}s")

        logger.info(f"전체 처리 완료: {time.time() - total_start:.2f}s")

        return DocumentSummarizationResult(
            doc_id=doc_id,
            toc_entries=toc_entries,
            page_offset=0,  # No longer used - pages are matched directly
            page_docs=page_docs,
            chunk_docs=chunk_docs,
            chunk_summaries=chunk_summaries,
            chapter_summaries=chapter_summaries,
            document_summary=document_summary,
        )

    def _assign_chapters_to_pages(
        self,
        pages: list[str],
        toc_entries: list[TOCEntry],
        offset: int,
    ) -> list[PageDocument]:
        """Assign chapter titles to pages based on TOC."""
        docs = [
            PageDocument(page_number=i + 1, content=txt or "")
            for i, txt in enumerate(pages)
        ]

        # Filter to chapter-level entries (level=1)
        chapter_entries = [e for e in toc_entries if e.level == 1]
        if not chapter_entries:
            for d in docs:
                d.chapter_title = "UNKNOWN"
            return docs

        # Convert TOC pages to 0-based indices
        starts: list[tuple[int, str]] = []
        for e in chapter_entries:
            start_idx = (e.start_page + offset) - 1
            starts.append((start_idx, e.title))
        starts.sort(key=lambda x: x[0])

        # Default: FRONT_MATTER for pages before first chapter
        for d in docs:
            d.chapter_title = "FRONT_MATTER"

        # Assign chapters by range
        for idx, (start_idx, title) in enumerate(starts):
            if start_idx < 0:
                start_idx = 0
            end_idx = (starts[idx + 1][0] - 1) if idx + 1 < len(starts) else (len(docs) - 1)
            if start_idx >= len(docs):
                continue
            end_idx = min(end_idx, len(docs) - 1)
            for p in range(start_idx, end_idx + 1):
                docs[p].chapter_title = title

        return docs

    def _split_pages_to_chunks(
        self,
        page_docs: list[PageDocument],
    ) -> list[ChunkDocument]:
        """Split pages into overlapping chunks."""
        chunk_docs: list[ChunkDocument] = []

        for page_doc in page_docs:
            content = page_doc.content
            if not content:
                continue

            # Simple character-based chunking with overlap
            chunks = self._chunk_text(content)

            for j, chunk_text in enumerate(chunks):
                chunk_docs.append(
                    ChunkDocument(
                        chunk_id=f"p{page_doc.page_number}_c{j}",
                        content=chunk_text,
                        page_number=page_doc.page_number,
                        chunk_index=j,
                        chapter_title=page_doc.chapter_title,
                    )
                )

        return chunk_docs

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Ensure overlap is less than chunk_size to prevent infinite loop
        effective_overlap = min(self.chunk_overlap, self.chunk_size - 1)

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence/paragraph boundary
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + self.chunk_size // 2:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    for sep in [".\n", ". ", "。\n", "。 "]:
                        sent_break = text.rfind(sep, start, end)
                        if sent_break > start + self.chunk_size // 2:
                            end = sent_break + len(sep)
                            break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start with overlap (guaranteed to advance at least 1 char)
            start = end - effective_overlap
            if start >= len(text):
                break

        return chunks

    def _summarize_chunks(
        self,
        chunk_docs: list[ChunkDocument],
    ) -> dict[str, ChunkSummary]:
        """Generate summaries for all chunks (sequential)."""
        summaries: dict[str, ChunkSummary] = {}

        for chunk in chunk_docs:
            if not chunk.content.strip():
                continue

            summary = self._summarizer.summarize_chunk(chunk.content)
            chunk.summary = summary
            summaries[chunk.chunk_id] = summary

        return summaries

    def _summarize_chunks_parallel(
        self,
        chunk_docs: list[ChunkDocument],
        max_workers: int = 24,
    ) -> dict[str, ChunkSummary]:
        """Generate summaries for all chunks (parallel)."""
        summaries: dict[str, ChunkSummary] = {}

        # Filter chunks with content
        valid_chunks = [c for c in chunk_docs if c.content.strip()]

        if not valid_chunks:
            return summaries

        def summarize_one(chunk: ChunkDocument) -> tuple[str, ChunkSummary]:
            summary = self._summarizer.summarize_chunk(chunk.content)
            return chunk.chunk_id, summary

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(summarize_one, chunk): chunk for chunk in valid_chunks}

            for i, future in enumerate(as_completed(futures)):
                chunk = futures[future]
                try:
                    chunk_id, summary = future.result()
                    chunk.summary = summary
                    summaries[chunk_id] = summary
                    if (i + 1) % 10 == 0:
                        logger.info(f"  청크 요약 진행: {i + 1}/{len(valid_chunks)}")
                except Exception as e:
                    logger.error(f"  청크 {chunk.chunk_id} 요약 실패: {e}")

        return summaries

    def _summarize_chapters(
        self,
        chunk_docs: list[ChunkDocument],
        chunk_summaries: dict[str, ChunkSummary],
    ) -> dict[str, ChapterSummary]:
        """Generate chapter summaries from chunk summaries."""
        # Group chunks by chapter
        by_chapter: dict[str, list[ChunkDocument]] = defaultdict(list)
        for chunk in chunk_docs:
            by_chapter[chunk.chapter_title].append(chunk)

        chapter_summaries: dict[str, ChapterSummary] = {}

        for chapter_title, chunks in by_chapter.items():
            # Sort by page and chunk index
            chunks_sorted = sorted(
                chunks, key=lambda x: (x.page_number, x.chunk_index)
            )

            # Collect chunk summary texts
            chunk_summary_texts = []
            for chunk in chunks_sorted:
                if chunk.chunk_id in chunk_summaries:
                    chunk_summary_texts.append(chunk_summaries[chunk.chunk_id].summary)

            if not chunk_summary_texts:
                continue

            # Reduce chunk summaries in batches to avoid prompt overflow.
            reduced_summaries = self._reduce_chunk_summaries(
                chapter_title=chapter_title,
                chunk_summaries=chunk_summary_texts,
            )

            # Generate chapter summary
            chapter_summary = self._summarizer.summarize_chapter(
                chapter_title=chapter_title,
                chunk_summaries=reduced_summaries,
            )
            chapter_summaries[chapter_title] = chapter_summary

        return chapter_summaries

    def _reduce_chunk_summaries(
        self,
        chapter_title: str,
        chunk_summaries: list[str],
        batch_size: int = 40,
    ) -> list[str]:
        """Batch-reduce chunk summaries so the chapter prompt stays small.

        챕터 요약 프롬프트가 너무 길어지는 경우를 막기 위해 배치 축약을 수행한다.
        이유: vLLM은 prompt_tokens + max_tokens > max_model_len이면 요청을 400으로 거절하며,
             이때 "max_tokens ... got negative" 오류가 발생한다.
        예시: chunk 요약이 120개이고 batch_size=40이면
             40개씩 3개 배치 요약 -> 3개 요약으로 최종 챕터 요약 수행.
        """
        if len(chunk_summaries) <= batch_size:
            return chunk_summaries

        current = chunk_summaries
        round_idx = 1
        while len(current) > batch_size:
            logger.info(
                "  Chapter batch reduction round %d: %s (%d items)",
                round_idx,
                chapter_title,
                len(current),
            )
            batch_summaries: list[str] = []
            for i in range(0, len(current), batch_size):
                batch = current[i : i + batch_size]
                batch_summary = self._summarizer.summarize_chapter(
                    chapter_title=f"{chapter_title} (batch {i // batch_size + 1})",
                    chunk_summaries=batch,
                )
                batch_summaries.append(batch_summary.summary)
            current = batch_summaries
            round_idx += 1

        logger.info(
            "  Chapter batch reduction done: %s (%d items)",
            chapter_title,
            len(current),
        )
        return current

    def _summarize_document(
        self,
        chapter_summaries: dict[str, ChapterSummary],
    ) -> DocumentSummary:
        """Generate document summary from chapter summaries."""
        # Pass all summaries to summarize_document
        # The method will use FRONT_MATTER for metadata extraction
        # but exclude it from the main summary content
        all_summaries = list(chapter_summaries.values())

        if not all_summaries:
            return DocumentSummary(summary="No content to summarize.")

        return self._summarizer.summarize_document(all_summaries)

    def summarize_text(self, text: str, **kwargs: Any) -> ChunkSummary:
        """Summarize a single text (convenience method)."""
        return self._summarizer.summarize_chunk(text, **kwargs)


__all__ = [
    "DocumentSummarizationService",
    "DocumentSummarizationResult",
    "PageDocument",
    "ChunkDocument",
]
