"""LLM-based summarization adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from backend.config.settings import vllm_settings
from backend.llm_infrastructure.llm import get_llm
from backend.llm_infrastructure.llm.base import BaseLLM

from ..base import BaseSummarizer, SummaryResult
from ..registry import register_summarizer
from ..schemas import ChapterSummary, ChunkSummary, DocumentMetadata, DocumentSummary


# Prompt directory
PROMPT_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(name: str, version: str = "v1") -> dict[str, str]:
    """Load prompt from YAML file."""
    path = PROMPT_DIR / f"{name}_{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    system = data.get("system", "")
    user = ""
    for msg in data.get("messages", []):
        if msg.get("role") == "user":
            user = msg.get("content", "")
            break
    return {"system": system, "user": user}


@register_summarizer("llm", version="v1")
class LLMSummarizer(BaseSummarizer):
    """LLM-based summarizer using vLLM or compatible API."""

    def __init__(
        self,
        llm_method: str = "vllm",
        llm_version: str = "v1",
        prompt_name: str = "chunk_summary",
        prompt_version: str = "v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.prompt_name = prompt_name
        self.prompt_version = prompt_version
        self._prompt = _load_prompt(prompt_name, prompt_version)

        # Initialize LLM
        self._llm: BaseLLM = get_llm(
            llm_method,
            version=llm_version,
            base_url=kwargs.get("base_url", vllm_settings.base_url),
            model=kwargs.get("model", vllm_settings.model_name),
            temperature=kwargs.get("temperature", 0),  # Deterministic for summarization
            max_tokens=kwargs.get("max_tokens", vllm_settings.max_tokens),
            timeout=kwargs.get("timeout", vllm_settings.timeout),
        )

    def summarize(
        self,
        text: str,
        *,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> SummaryResult:
        """Summarize text using LLM."""
        user_content = self._prompt["user"].format(
            chunk_text=text,
            max_length=max_length or 200,
        )

        messages = [
            {"role": "system", "content": self._prompt["system"]},
            {"role": "user", "content": user_content},
        ]

        # Generate with JSON mode
        response = self._llm.generate(
            messages,
            response_format={"type": "json_object"},
            **kwargs,
        )

        # Parse response
        try:
            data = json.loads(response.text)
            summary = data.get("summary", response.text)
        except json.JSONDecodeError:
            summary = response.text.strip()
            data = {}

        return SummaryResult(
            original_text=text,
            summary=summary,
            summary_length=len(summary),
            compression_ratio=len(summary) / max(len(text), 1),
            metadata={
                "keywords": data.get("keywords", []),
                "actions": data.get("actions", []),
                "warnings": data.get("warnings", []),
                "raw_response": data,
            },
        )

    def summarize_chunk(
        self,
        text: str,
        *,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> ChunkSummary:
        """Summarize a chunk and return structured ChunkSummary."""
        prompt = _load_prompt("chunk_summary", self.prompt_version)
        user_content = prompt["user"].format(
            chunk_text=text,
            max_length=max_length or 200,
        )

        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": user_content},
        ]

        response = self._llm.generate(
            messages,
            response_format={"type": "json_object"},
            **kwargs,
        )

        try:
            data = json.loads(response.text)
            return ChunkSummary(
                summary=data.get("summary", ""),
                keywords=data.get("keywords", []),
                actions=data.get("actions", []),
                warnings=data.get("warnings", []),
            )
        except (json.JSONDecodeError, ValueError):
            return ChunkSummary(summary=response.text.strip())

    def summarize_chapter(
        self,
        chapter_title: str,
        chunk_summaries: list[str],
        **kwargs: Any,
    ) -> ChapterSummary:
        """Summarize a chapter from chunk summaries."""
        prompt = _load_prompt("chapter_summary", self.prompt_version)

        # Format chunk summaries
        summaries_text = "\n".join(f"- {s}" for s in chunk_summaries)
        user_content = prompt["user"].format(
            chapter_title=chapter_title,
            chunk_summaries=summaries_text,
        )

        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": user_content},
        ]

        response = self._llm.generate(
            messages,
            response_format={"type": "json_object"},
            **kwargs,
        )

        try:
            data = json.loads(response.text)
            return ChapterSummary(
                chapter_title=chapter_title,
                summary=data.get("summary", ""),
                key_points=data.get("key_points", []),
                keywords=data.get("keywords", []),
            )
        except (json.JSONDecodeError, ValueError):
            return ChapterSummary(
                chapter_title=chapter_title,
                summary=response.text.strip(),
            )

    def summarize_document(
        self,
        chapter_summaries: list[ChapterSummary],
        *,
        front_matter_text: str | None = None,
        **kwargs: Any,
    ) -> DocumentSummary:
        """Summarize entire document from chapter summaries."""
        prompt = _load_prompt("document_summary", self.prompt_version)

        # Format chapter summaries (exclude FRONT_MATTER/UNKNOWN from main content)
        summaries_text = "\n\n".join(
            f"[{cs.chapter_title}]\n{cs.summary}\nKey points: {', '.join(cs.key_points[:8])}"
            for cs in chapter_summaries
            if cs.chapter_title not in ("FRONT_MATTER", "UNKNOWN")
        )

        # Include FRONT_MATTER for metadata extraction
        front_matter_section = ""
        front_matter_summary = next(
            (cs for cs in chapter_summaries if cs.chapter_title == "FRONT_MATTER"),
            None,
        )
        if front_matter_summary:
            front_matter_section = (
                f"\n\n[FRONT_MATTER (title page/preface - use for metadata extraction)]\n"
                f"{front_matter_summary.summary}"
            )
        elif front_matter_text:
            front_matter_section = (
                f"\n\n[FRONT_MATTER (title page/preface - use for metadata extraction)]\n"
                f"{front_matter_text[:2000]}"
            )

        user_content = prompt["user"].format(
            chapter_summaries=summaries_text + front_matter_section
        )

        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": user_content},
        ]

        response = self._llm.generate(
            messages,
            response_format={"type": "json_object"},
            **kwargs,
        )

        try:
            data = json.loads(response.text)

            # Parse metadata if present
            metadata = None
            if "metadata" in data and isinstance(data["metadata"], dict):
                meta_data = data["metadata"]
                metadata = DocumentMetadata(
                    device_name=meta_data.get("device_name"),
                    doc_type=meta_data.get("doc_type"),
                    doc_version=meta_data.get("doc_version"),
                    doc_date=meta_data.get("doc_date"),
                )

            return DocumentSummary(
                summary=data.get("summary", ""),
                key_points=data.get("key_points", []),
                keywords=data.get("keywords", []),
                metadata=metadata,
            )
        except (json.JSONDecodeError, ValueError):
            return DocumentSummary(summary=response.text.strip())


__all__ = ["LLMSummarizer"]
