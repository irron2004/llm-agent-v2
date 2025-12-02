"""Registry-registered parser adapters that wrap engines."""

from .pdf_plain import PlainPdfAdapter
from .pdf_deepdoc import DeepDocPdfAdapter
from .pdf_vlm import VlmPdfAdapter, DeepSeekVLPdfAdapter

__all__ = ["PlainPdfAdapter", "DeepDocPdfAdapter", "VlmPdfAdapter", "DeepSeekVLPdfAdapter"]
