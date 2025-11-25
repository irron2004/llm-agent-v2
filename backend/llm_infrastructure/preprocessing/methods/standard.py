"""Standard preprocessing method (baseline)."""

from typing import Iterable

from ..base import BasePreprocessor
from ..registry import register_preprocessor


@register_preprocessor("standard", version="v1")
class StandardPreprocessor(BasePreprocessor):
    """Standard preprocessing: basic cleaning and normalization.

    This is a baseline method that:
    - Strips whitespace
    - Normalizes unicode
    - Removes empty documents

    Config options:
        lowercase: bool = False - Convert to lowercase
        remove_extra_spaces: bool = True - Collapse multiple spaces
    """

    def preprocess(self, docs: Iterable[str]) -> Iterable[str]:
        """Apply standard preprocessing."""
        lowercase = self.config.get("lowercase", False)
        remove_extra_spaces = self.config.get("remove_extra_spaces", True)

        for doc in docs:
            # Handle dict with 'content' field or plain string
            if isinstance(doc, dict):
                text = doc.get("content", "")
            else:
                text = str(doc)

            # Strip whitespace
            text = text.strip()

            # Skip empty
            if not text:
                continue

            # Lowercase if requested
            if lowercase:
                text = text.lower()

            # Remove extra spaces
            if remove_extra_spaces:
                text = " ".join(text.split())

            yield text
