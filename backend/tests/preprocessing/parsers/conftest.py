"""Pytest fixtures for parser tests."""

import io
from pathlib import Path

import pytest


@pytest.fixture
def sample_pdf_bytes():
    """Sample PDF file content (minimal valid PDF)."""
    return b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000056 00000 n\n0000000115 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"


@pytest.fixture
def sample_pdf_file(sample_pdf_bytes):
    """Sample PDF file as BytesIO."""
    return io.BytesIO(sample_pdf_bytes)


@pytest.fixture
def mock_deepdoc_output():
    """Sample DeepDoc parser output structure."""
    return {
        "chunks": [
            {
                "text": "This is the first paragraph from page 1.",
                "page": 1,
                "bbox": {"x0": 72, "y0": 100, "x1": 540, "y1": 150},
                "label": "paragraph",
                "confidence": 0.98,
            },
            {"text": "Second paragraph on page 1.", "page": 1, "bbox": [72, 160, 540, 200], "label": "paragraph"},
            {
                "text": "Introduction",
                "page": 1,
                "bbox": {"left": 72, "top": 50, "right": 300, "bottom": 80},
                "label": "title",
                "confidence": 0.99,
            },
        ],
        "tables": [
            {
                "page": 1,
                "bbox": [100, 300, 500, 500],
                "html": "<table><tr><th>Header1</th><th>Header2</th></tr></table>",
                "text": "Header1 | Header2",
                "image": "/tmp/table_p1.png",
            }
        ],
        "figures": [{"page": 1, "bbox": [100, 550, 400, 700], "caption": "Figure 1: Sample chart", "image_path": "/tmp/fig_p1.png"}],
    }


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Temporary directory for model files."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir
