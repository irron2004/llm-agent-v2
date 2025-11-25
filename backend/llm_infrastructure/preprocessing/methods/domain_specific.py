"""Domain-specific preprocessing for PE Agent.

This is an example of how to implement domain-specific preprocessing
that handles technical terminology, abbreviations, etc.
"""

from typing import Iterable
import re

from ..base import BasePreprocessor
from ..registry import register_preprocessor


@register_preprocessor("pe_domain", version="v1")
class DomainSpecificPreprocessor(BasePreprocessor):
    """PE domain-specific preprocessing.

    Handles:
    - Technical abbreviations normalization
    - Unit standardization
    - Special character handling for equations

    Config options:
        normalize_units: bool = True - Standardize unit formats
        expand_abbreviations: bool = True - Expand common abbreviations
    """

    # Example abbreviation mappings (customize for your domain)
    ABBREVIATIONS = {
        "temp": "temperature",
        "pres": "pressure",
        "conc": "concentration",
        # Add more as needed
    }

    def preprocess(self, docs: Iterable[str]) -> Iterable[str]:
        """Apply domain-specific preprocessing."""
        normalize_units = self.config.get("normalize_units", True)
        expand_abbr = self.config.get("expand_abbreviations", True)

        for doc in docs:
            # Handle dict with 'content' field or plain string
            if isinstance(doc, dict):
                text = doc.get("content", "")
                metadata = {k: v for k, v in doc.items() if k != "content"}
            else:
                text = str(doc)
                metadata = {}

            text = text.strip()
            if not text:
                continue

            # Normalize units (example: standardize temperature units)
            if normalize_units:
                text = self._normalize_units(text)

            # Expand abbreviations
            if expand_abbr:
                text = self._expand_abbreviations(text)

            # Return with metadata if present
            if metadata:
                yield {"content": text, **metadata}
            else:
                yield text

    def _normalize_units(self, text: str) -> str:
        """Normalize unit representations."""
        # Example: 100°C → 100 degC, 50 C → 50 degC
        text = re.sub(r'(\d+)\s*°\s*C', r'\1 degC', text)
        text = re.sub(r'(\d+)\s*°\s*F', r'\1 degF', text)

        # Pressure units
        text = re.sub(r'(\d+)\s*psi', r'\1 PSI', text)

        return text

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        # Simple word boundary replacement
        for abbr, full in self.ABBREVIATIONS.items():
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text = re.sub(pattern, full, text, flags=re.IGNORECASE)

        return text


@register_preprocessor("pe_domain", version="v2")
class DomainSpecificPreprocessorV2(DomainSpecificPreprocessor):
    """Enhanced version with additional normalization.

    This shows how to version preprocessing methods.
    V2 adds normalization for chemical formulas.
    """

    def preprocess(self, docs: Iterable[str]) -> Iterable[str]:
        """Apply v2 preprocessing with chemical formula handling."""
        # Call parent preprocessing
        for doc in super().preprocess(docs):
            if isinstance(doc, dict):
                text = doc["content"]
                metadata = {k: v for k, v in doc.items() if k != "content"}
            else:
                text = doc
                metadata = {}

            # Additional v2 feature: normalize chemical formulas
            text = self._normalize_chemical_formulas(text)

            if metadata:
                yield {"content": text, **metadata}
            else:
                yield text

    def _normalize_chemical_formulas(self, text: str) -> str:
        """Normalize chemical formula representations."""
        # Example: H2O → H₂O (subscript)
        # In practice, you might want to keep ASCII for search
        # This is just an example
        return text
