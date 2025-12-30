"""Elasticsearch ingestion pipeline for VLM-parsed sections.

This service takes sectioned text (e.g., from DocumentIngestService.for_vlm),
optionally preprocesses/chunks/embeds it, and bulk-indexes into Elasticsearch
using the RAG chunk mapping.

Usage:
    # Basic usage with settings
    svc = EsIngestService.from_settings()
    result = svc.ingest_pdf(pdf_file, doc_id="doc_001", doc_type="sop")

    # With custom VLM client
    from backend.llm_infrastructure.vlm.clients import OpenAIVisionClient
    vlm_client = OpenAIVisionClient(base_url="http://localhost:8000/v1")
    svc = EsIngestService.from_settings(vlm_client=vlm_client)
    result = svc.ingest_pdf(pdf_file, doc_id="doc_001")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterable, List, Optional, Sequence

import numpy as np
import yaml
from elasticsearch import Elasticsearch, helpers

from backend.config.settings import rag_settings, search_settings, vllm_settings
from backend.llm_infrastructure.elasticsearch.document import EsChunkDocument
from backend.llm_infrastructure.llm import get_llm
from backend.llm_infrastructure.preprocessing.registry import get_preprocessor
from backend.services.embedding_service import EmbeddingService
from backend.services.ingest.document_ingest_service import Section, DocumentIngestService
from backend.services.ingest.txt_parser import parse_maintenance_txt

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result of document ingestion."""

    doc_id: str
    index: str
    total_sections: int
    total_chunks: int
    indexed_chunks: int
    failed_chunks: int = 0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _l2_normalize(vecs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize vectors. Handles both 1D and 2D arrays."""
    if vecs.ndim == 1:
        # Single vector: reshape to 2D, normalize, then squeeze back
        vecs = vecs.reshape(1, -1)
        norm = np.linalg.norm(vecs, axis=1, keepdims=True)
        norm = np.maximum(norm, eps)
        return (vecs / norm).squeeze(0)
    else:
        norm = np.linalg.norm(vecs, axis=1, keepdims=True)
        norm = np.maximum(norm, eps)
        return vecs / norm


class EsIngestService:
    """Ingest sectioned documents into Elasticsearch as RAG chunks.

    Uses Section = Chunk strategy: VLM-parsed sections become chunks directly.

    Supports two modes:
    1. ingest_sections(): Ingest pre-parsed sections
    2. ingest_pdf(): Full pipeline from PDF file using VLM parser
    """

    def __init__(
        self,
        *,
        es_client: Elasticsearch,
        index: str,
        embedder: Any,
        preprocessor: Any = None,
        normalize_vectors: bool = True,
        vlm_client: Any = None,
        document_ingest_service: DocumentIngestService | None = None,
        pipeline_version: str = "v1",
    ) -> None:
        self.es = es_client
        self.index = index
        self.embedder = embedder
        self.preprocessor = preprocessor
        self.normalize_vectors = normalize_vectors
        self.vlm_client = vlm_client
        self._document_ingest_service = document_ingest_service
        self.pipeline_version = pipeline_version

    @classmethod
    def from_settings(
        cls,
        *,
        index: Optional[str] = None,
        es_client: Elasticsearch | None = None,
        vlm_client: Any = None,
        pipeline_version: str = "v1",
    ) -> "EsIngestService":
        """Build EsIngestService using global settings.

        Uses Section = Chunk strategy (no additional chunking).

        Args:
            index: ES index/alias name. Defaults to current alias.
            es_client: Pre-configured ES client. Created from settings if None.
            vlm_client: VLM client for PDF parsing. Created from settings if None.
            pipeline_version: Pipeline version for metadata tracking.

        Returns:
            Configured EsIngestService instance.
        """
        # Elasticsearch client
        if es_client is None:
            client_kwargs: dict[str, Any] = {
                "hosts": [search_settings.es_host],
                "verify_certs": True,
            }
            if search_settings.es_user and search_settings.es_password:
                client_kwargs["basic_auth"] = (
                    search_settings.es_user,
                    search_settings.es_password,
                )
            es_client = Elasticsearch(**client_kwargs)

        # Embedder
        embed_svc = EmbeddingService(
            method=rag_settings.embedding_method,
            version=rag_settings.embedding_version,
            device=rag_settings.embedding_device,
            use_cache=rag_settings.embedding_use_cache,
            cache_dir=rag_settings.embedding_cache_dir,
        )

        # Preprocessor (for search_text preparation)
        preprocessor = get_preprocessor(
            rag_settings.preprocess_method,
            version=rag_settings.preprocess_version,
            level=rag_settings.preprocess_level,
        )

        # VLM client (optional - created from settings if not provided)
        if vlm_client is None:
            try:
                from backend.llm_infrastructure.vlm.clients.openai_vision import (
                    create_vlm_client_from_settings,
                )
                vlm_client = create_vlm_client_from_settings()
            except Exception as e:
                logger.warning("Could not create VLM client from settings: %s", e)
                vlm_client = None

        # Index alias (defaults to current alias)
        if index is None:
            index = f"{search_settings.es_index_prefix}_{search_settings.es_env}_current"

        return cls(
            es_client=es_client,
            index=index,
            embedder=embed_svc.get_raw_embedder(),
            preprocessor=preprocessor,
            normalize_vectors=rag_settings.vector_normalize,
            vlm_client=vlm_client,
            pipeline_version=pipeline_version,
        )

    def _preprocess(self, texts: Sequence[str]) -> List[str]:
        """Preprocess texts for embedding/search."""
        if self.preprocessor is None:
            return list(texts)
        return list(self.preprocessor.preprocess(texts))

    def _embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of texts.

        Returns:
            Embedding vectors as 2D numpy array (n_texts, embedding_dim).
        """
        raw_embeddings = self.embedder.embed_batch(list(texts))
        embeddings = np.asarray(raw_embeddings, dtype=np.float32)

        # Ensure 2D array
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if self.normalize_vectors:
            embeddings = _l2_normalize(embeddings)
        return embeddings

    def ingest_sections(
        self,
        base_doc_id: str,
        sections: Iterable[Section],
        *,
        doc_type: str = "generic",
        lang: str = "ko",
        tenant_id: str = "",
        project_id: str = "",
        vlm_model: str = "",
        tags: list[str] | None = None,
        refresh: bool = False,
    ) -> dict[str, int | str]:
        """Ingest parsed sections into Elasticsearch using EsChunkDocument schema.

        Uses Section = Chunk strategy: each section becomes one ES document.

        Args:
            base_doc_id: Original document identifier (user-provided)
            sections: Parsed sections from DocumentIngestService
            doc_type: Logical document type/category
            lang: Language code
            tenant_id: Tenant identifier for multi-tenancy
            project_id: Project identifier
            vlm_model: VLM model name for metadata
            tags: Optional tags to attach to chunks
            refresh: Whether to refresh index after bulk insert

        Returns:
            Dict with index name and number of indexed chunks.
        """
        # Convert to list for multiple iterations
        section_list = list(sections)
        if not section_list:
            logger.info("No sections to ingest for doc_id=%s", base_doc_id)
            return {"index": self.index, "indexed": 0}

        # Preprocess texts for embedding (Section = Chunk, no additional splitting)
        raw_texts = [s.text for s in section_list]
        processed_texts = self._preprocess(raw_texts)

        # Generate embeddings
        embeddings = self._embed_batch(processed_texts)

        # Validate embedding dimensions
        if embeddings.size > 0:
            embed_dim = embeddings.shape[1]
            expected = search_settings.es_embedding_dims
            if expected and embed_dim != expected:
                raise ValueError(
                    f"Embedding dimension mismatch: got {embed_dim}, expected {expected}"
                )

        # Convert sections to EsChunkDocuments
        es_docs: list[EsChunkDocument] = []
        for idx, (section, emb) in enumerate(zip(section_list, embeddings)):
            es_doc = EsChunkDocument.from_section(
                section=section,
                doc_id=base_doc_id,
                chunk_index=idx,
                embedding=emb.tolist(),
                lang=lang,
                doc_type=doc_type,
                tenant_id=tenant_id,
                project_id=project_id,
                pipeline_version=self.pipeline_version,
                vlm_model=vlm_model,
                tags=tags,
            )
            es_docs.append(es_doc)

        # Build bulk actions using EsChunkDocument.to_es_doc()
        actions = [
            {
                "_index": self.index,
                "_id": doc.chunk_id,
                "_source": doc.to_es_doc(),
            }
            for doc in es_docs
        ]

        helpers.bulk(self.es, actions, refresh=refresh)
        logger.info("Indexed %d chunks into %s", len(actions), self.index)
        return {"index": self.index, "indexed": len(actions)}

    def ingest_pdf(
        self,
        file: BinaryIO,
        doc_id: str,
        *,
        doc_type: str = "generic",
        tenant_id: str = "",
        project_id: str = "",
        lang: str = "ko",
        tags: list[str] | None = None,
        refresh: bool = False,
    ) -> IngestResult:
        """Full pipeline: PDF → VLM parse → sections → ES index.

        Args:
            file: PDF file binary.
            doc_id: Document identifier.
            doc_type: Document type (sop, ts, guide, etc.).
            tenant_id: Tenant identifier for multi-tenancy.
            project_id: Project identifier.
            lang: Language code.
            tags: Optional tags to attach to chunks.
            refresh: Whether to refresh ES index after bulk insert.

        Returns:
            IngestResult with indexed chunk counts.

        Raises:
            RuntimeError: If VLM client is not configured.
        """
        if self.vlm_client is None:
            raise RuntimeError(
                "VLM client not configured. "
                "Pass vlm_client to constructor or set VLM_CLIENT_* env vars."
            )

        # Create or reuse DocumentIngestService
        if self._document_ingest_service is None:
            self._document_ingest_service = DocumentIngestService.for_vlm(
                vlm_client=self.vlm_client,
            )

        # Parse PDF with VLM
        logger.info("Parsing PDF with VLM for doc_id=%s", doc_id)
        result = self._document_ingest_service.ingest_pdf(file, doc_type=doc_type)

        sections_data = result.get("sections", [])
        if not sections_data:
            logger.warning("No sections extracted from PDF for doc_id=%s", doc_id)
            return IngestResult(
                doc_id=doc_id,
                index=self.index,
                total_sections=0,
                total_chunks=0,
                indexed_chunks=0,
                metadata=result.get("metadata", {}),
            )

        # Convert section dicts to Section objects
        sections = [
            Section(
                title=s.get("title", ""),
                text=s.get("text", ""),
                page_start=s.get("page_start"),
                page_end=s.get("page_end"),
                metadata=s.get("metadata", {}),
            )
            for s in sections_data
        ]

        # Get VLM model name for metadata
        vlm_model = ""
        if hasattr(self.vlm_client, "get_model_name"):
            vlm_model = self.vlm_client.get_model_name()

        # Ingest sections to ES using EsChunkDocument schema
        ingest_result = self.ingest_sections(
            base_doc_id=doc_id,
            sections=sections,
            doc_type=doc_type,
            lang=lang,
            tenant_id=tenant_id,
            project_id=project_id,
            vlm_model=vlm_model,
            tags=tags,
            refresh=refresh,
        )

        return IngestResult(
            doc_id=doc_id,
            index=self.index,
            total_sections=len(sections),
            total_chunks=ingest_result.get("indexed", 0),
            indexed_chunks=ingest_result.get("indexed", 0),
            metadata={
                **result.get("metadata", {}),
                "tenant_id": tenant_id,
                "project_id": project_id,
                "lang": lang,
                "vlm_model": vlm_model,
            },
        )

    def ingest_txt(
        self,
        file: BinaryIO | str,
        doc_id: str,
        *,
        doc_type: str = "maintenance",
        tenant_id: str = "",
        project_id: str = "",
        lang: str = "ko",
        tags: list[str] | None = None,
        refresh: bool = False,
        use_llm_summary: bool = True,
    ) -> IngestResult:
        """Full pipeline: txt → parse → LLM summary → ES index.

        Ingests maintenance report text files with structured sections.
        Creates separate ES documents for each section (status/action/cause/result).

        Args:
            file: Text file (BinaryIO) or file path (str).
            doc_id: Document identifier (e.g., Order No.).
            doc_type: Document type (default: "maintenance").
            tenant_id: Tenant identifier for multi-tenancy.
            project_id: Project identifier.
            lang: Language code.
            tags: Optional tags to attach to chunks.
            refresh: Whether to refresh ES index after insert.
            use_llm_summary: Whether to generate LLM summaries (default: True).

        Returns:
            IngestResult with indexed chunk count.

        Note:
            Each section becomes a separate ES document with the same doc_id.
            Use doc_id to retrieve all sections of the same maintenance event.
        """
        # Read file content
        if isinstance(file, str):
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = file.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")

        # Parse maintenance report
        logger.info("Parsing maintenance report for doc_id=%s", doc_id)
        report = parse_maintenance_txt(content)

        if not report.sections:
            logger.warning("No sections extracted from txt for doc_id=%s", doc_id)
            return IngestResult(
                doc_id=doc_id,
                index=self.index,
                total_sections=0,
                total_chunks=0,
                indexed_chunks=0,
                metadata=report.meta,
            )

        # Generate LLM summary if enabled
        llm_analysis = {}
        if use_llm_summary:
            try:
                llm_analysis = self._generate_maintenance_summary(report)
            except Exception as e:
                logger.error("Failed to generate LLM summary: %s", e)
                # Continue without summary

        # Extract device info from meta
        device_name = report.meta.get("Model Name", "") or report.meta.get("Equip. NO", "")
        doc_description = report.meta.get("Title", "")
        order_no = report.meta.get("Order No.", "")

        # Build common metadata for all sections
        common_meta = {
            "device_name": device_name,
            "doc_description": doc_description,
            "order_no": order_no,
        }

        # Add overall summary/keywords from LLM if available
        if "chunk_summary" in llm_analysis:
            common_meta["overall_summary"] = llm_analysis["chunk_summary"]
        if "chunk_keywords" in llm_analysis:
            common_meta["overall_keywords"] = llm_analysis["chunk_keywords"]

        # Create Section objects for each section
        sections = []
        section_names = ["status", "action", "cause", "result"]

        for section_name in section_names:
            section_text = report.sections.get(section_name, "")
            if not section_text.strip():
                continue

            # Build section-specific metadata
            section_meta = common_meta.copy()
            section_meta["section_type"] = section_name
            section_meta["chapter"] = section_name  # For ES chapter field

            # Add LLM-generated section summary/keywords
            llm_sections = llm_analysis.get("llm_analysis", {}).get("sections", {})
            if section_name in llm_sections:
                section_info = llm_sections[section_name]
                if "summary" in section_info:
                    section_meta["chunk_summary"] = section_info["summary"]
                if "keywords" in section_info:
                    section_meta["chunk_keywords"] = section_info["keywords"]

            # Create Section object
            section = Section(
                title=section_name,
                text=section_text,
                page_start=0,
                page_end=0,
                metadata=section_meta,
            )
            sections.append(section)

        if not sections:
            logger.warning("No valid sections to ingest for doc_id=%s", doc_id)
            return IngestResult(
                doc_id=doc_id,
                index=self.index,
                total_sections=0,
                total_chunks=0,
                indexed_chunks=0,
                metadata=report.meta,
            )

        # Ingest all sections (each becomes a separate ES document)
        ingest_result = self.ingest_sections(
            base_doc_id=doc_id,
            sections=sections,
            doc_type=doc_type,
            lang=lang,
            tenant_id=tenant_id,
            project_id=project_id,
            vlm_model="",
            tags=tags,
            refresh=refresh,
        )

        return IngestResult(
            doc_id=doc_id,
            index=self.index,
            total_sections=len(sections),
            total_chunks=ingest_result.get("indexed", 0),
            indexed_chunks=ingest_result.get("indexed", 0),
            metadata={
                **report.meta,
                "tenant_id": tenant_id,
                "project_id": project_id,
                "lang": lang,
                "llm_analysis": llm_analysis,
            },
        )

    def _generate_maintenance_summary(
        self,
        report,
    ) -> dict[str, Any]:
        """Generate LLM-based summary and keywords for maintenance report.

        Args:
            report: Parsed MaintenanceReport.

        Returns:
            Dict with chunk_summary, chunk_keywords, and sections metadata.
        """
        # Load prompt
        prompt_path = (
            Path(__file__).parent.parent
            / "llm_infrastructure"
            / "summarization"
            / "prompts"
            / "maintenance_summary_v1.yaml"
        )

        with open(prompt_path, encoding="utf-8") as f:
            prompt_data = yaml.safe_load(f)

        system_msg = prompt_data.get("system", "")
        user_template = prompt_data["messages"][0]["content"]

        # Format user message
        user_msg = user_template.format(report_text=report.full_text)

        # Initialize LLM
        llm = get_llm(
            "vllm",
            version="v1",
            base_url=vllm_settings.base_url,
            model=vllm_settings.model_name,
            temperature=0,
            max_tokens=vllm_settings.max_tokens,
            timeout=vllm_settings.timeout,
        )

        # Generate summary
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        response = llm.generate(messages, response_format={"type": "json_object"})

        # Parse response
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            return {}

        # Extract summary and keywords for ES schema
        result = {
            "chunk_summary": data.get("summary", ""),
            "chunk_keywords": data.get("keywords", []),
        }

        # Add device info if extracted
        if "device_info" in data:
            device_info = data["device_info"]
            if device_info.get("device_name"):
                result["device_name"] = device_info["device_name"]
            if device_info.get("part_changed"):
                result["part_changed"] = device_info["part_changed"]
            if device_info.get("issue_type"):
                result["issue_type"] = device_info["issue_type"]

        # Store full LLM response in metadata
        result["llm_analysis"] = data

        return result


__all__ = ["EsIngestService", "IngestResult"]
