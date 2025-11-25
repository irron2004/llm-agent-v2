"""Preprocessor wrapper around the normalization engine."""

from typing import Iterable, Any

from ..normalize_engine import build_normalizer
from ..base import BasePreprocessor
from ..registry import register_preprocessor


@register_preprocessor("normalize", version="v1")
class NormalizationPreprocessor(BasePreprocessor):
    """Apply text normalization with configurable level and variant map."""

    def __init__(
        self,
        level: str = "L3",
        variant_map: dict[str, str] | None = None,
        keep_newlines: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            level=level,
            variant_map=variant_map or {},
            keep_newlines=keep_newlines,
            **kwargs,
        )
        self.level = level or "L3"
        self.variant_map = variant_map or {}
        self.keep_newlines = keep_newlines
        self._normalizer = build_normalizer(
            level=self.level,
            variant_map=self.variant_map,
            keep_newlines=self.keep_newlines,
        )

    def preprocess(self, docs: Iterable[Any]) -> Iterable[Any]:
        """Normalize each document, preserving metadata if present."""
        for doc in docs:
            if isinstance(doc, dict):
                text = doc.get("content", "")
                metadata = {k: v for k, v in doc.items() if k != "content"}
            else:
                text = str(doc)
                metadata = {}

            if not text:
                continue

            normalized = self._normalizer(text)

            if metadata:
                yield {"content": normalized, **metadata}
            else:
                yield normalized
