"""Integration tests for PDF parsers with real PDF files."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_infrastructure.preprocessing.parsers import get_parser
from llm_infrastructure.preprocessing.parsers.base import ParsedDocument, PdfParseOptions


def create_minimal_pdf_bytes():
    """Create a minimal valid PDF without external dependencies."""
    # Minimal PDF with embedded text
    return b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000056 00000 n
0000000115 00000 n
trailer<</Size 4/Root 1 0 R>>
startxref
190
%%EOF"""


@pytest.fixture
def sample_pdf_bytes():
    """Fixture that provides a minimal PDF."""
    return create_minimal_pdf_bytes()


class MockVLMClientForIntegration:
    """Mock VLM client that returns realistic text for integration tests."""

    def __init__(self):
        self.call_count = 0

    def generate(self, image, prompt, **kwargs):
        """Generate realistic markdown-like text for each page."""
        self.call_count += 1

        # Simulate VLM output with markdown formatting
        return """# Sample Document for Parser Testing

This is the first paragraph of our test document. It contains multiple sentences
to test text extraction capabilities. The parser should be able to extract this
text accurately and preserve the structure.

This is the second paragraph. It discusses various aspects of document parsing,
including layout recognition, text extraction, and table structure detection.
These are important features for any robust PDF parser.

## Comparison Table

| Feature | DeepDoc | VLM |
|---------|---------|-----|
| OCR Support | Yes | Yes |
| Layout Recognition | Yes | Yes |
| Table Extraction | Yes | Limited |
| Bounding Boxes | Yes | No |

In conclusion, both parsers have their strengths and use cases.
DeepDoc excels at structure extraction, while VLM parsers provide
more contextual understanding of the content.
"""


class TestPlainPdfParserIntegration:
    """Integration tests for PlainPdfEngine with real PDFs."""

    @pytest.mark.skipif(True, reason="Requires pdfplumber")
    def test_parse_pdf_with_pdfplumber(self, sample_pdf_bytes):
        """Test parsing a real PDF with pdfplumber (PlainPdfEngine)."""
        parser = get_parser("pdf_plain")

        pdf_file = io.BytesIO(sample_pdf_bytes)
        result = parser.parse(pdf_file, options=PdfParseOptions())

        # Verify result structure
        assert isinstance(result, ParsedDocument)
        assert result.content_type == "application/pdf"
        assert result.metadata["parser"] == "pdf_plain"

        # Should have at least one page
        assert len(result.pages) >= 1

    @pytest.mark.skipif(True, reason="Requires pdfplumber")
    def test_parse_multiple_pages_sequential(self, sample_pdf_bytes):
        """Test parsing the same PDF multiple times."""
        parser = get_parser("pdf_plain")

        # Parse twice
        result1 = parser.parse(io.BytesIO(sample_pdf_bytes))
        result2 = parser.parse(io.BytesIO(sample_pdf_bytes))

        # Results should be consistent
        assert len(result1.pages) == len(result2.pages)


class TestDeepDocPdfParserIntegration:
    """Integration tests for DeepDocPdfEngine with real PDFs."""

    def test_parse_with_deepdoc_mocked_backend(self, sample_pdf_bytes):
        """Test DeepDoc parser with mocked backend."""
        # Mock the DeepDoc backend
        mock_backend_cls = MagicMock()
        mock_backend_instance = MagicMock()
        mock_backend_cls.return_value = mock_backend_instance

        # Mock parse method to return realistic output
        mock_backend_instance.return_value = {
            "chunks": [
                {"text": "Sample text from page 1", "page": 1, "bbox": [10, 20, 100, 200]}
            ],
            "tables": [],
            "figures": []
        }

        from llm_infrastructure.preprocessing.parsers.engines.pdf_deepdoc_engine import DeepDocPdfEngine

        with patch.object(DeepDocPdfEngine, "_load_backend_class", return_value=mock_backend_cls):
            parser = get_parser("pdf_deepdoc")
            pdf_file = io.BytesIO(sample_pdf_bytes)
            result = parser.parse(pdf_file)

            # Verify result structure
            assert isinstance(result, ParsedDocument)
            assert result.content_type == "application/pdf"
            assert len(result.blocks) >= 1

    def test_deepdoc_fallback_to_plain_without_backend(self, sample_pdf_bytes):
        """Test DeepDoc falls back to plain when backend not available."""
        from llm_infrastructure.preprocessing.parsers.engines.pdf_deepdoc_engine import DeepDocPdfEngine

        with patch.object(DeepDocPdfEngine, "_load_backend_class", return_value=None):
            # Mock PlainPdfEngine to avoid pdfplumber dependency
            mock_plain_result = ParsedDocument(
                pages=[],
                blocks=[],
                tables=[],
                figures=[],
                metadata={"parser": "pdf_plain"},
                content_type="application/pdf"
            )

            with patch("llm_infrastructure.preprocessing.parsers.engines.pdf_deepdoc_engine.PlainPdfEngine") as mock_plain:
                mock_plain_instance = MagicMock()
                mock_plain_instance.run.return_value = mock_plain_result
                mock_plain.return_value = mock_plain_instance

                parser = get_parser("pdf_deepdoc")
                pdf_file = io.BytesIO(sample_pdf_bytes)
                result = parser.parse(pdf_file, options=PdfParseOptions(fallback_to_plain=True))

                # Should have used fallback
                assert result.metadata.get("used_fallback") is True
                assert "fallback_reason" in result.metadata


class TestVlmPdfParserIntegration:
    """Integration tests for VlmPdfEngine with real PDFs."""

    def test_parse_with_vlm_mock_client(self, sample_pdf_bytes):
        """Test VLM parser with mock client on real PDF."""
        mock_client = MockVLMClientForIntegration()

        # Mock renderer to return a simple image representation
        def mock_pdf_to_images(pdf_bytes):
            return ["mock_image_page1"]

        parser = get_parser("pdf_vlm", vlm_client=mock_client, renderer=mock_pdf_to_images)

        pdf_file = io.BytesIO(sample_pdf_bytes)
        result = parser.parse(
            pdf_file,
            options=PdfParseOptions(
                vlm_model="deepseek-vl2",
                vlm_prompt="Extract all text from this page",
                vlm_temperature=0.1
            )
        )

        # Verify result structure
        assert isinstance(result, ParsedDocument)
        assert result.content_type == "application/pdf"
        assert result.metadata["parser"] == "pdf_vlm"
        assert result.metadata["vlm_model"] == "deepseek-vl2"

        # Should have parsed at least one page
        assert len(result.pages) >= 1
        assert len(result.blocks) >= 1

        # Should have extracted text via VLM
        merged_text = result.merged_text()
        assert len(merged_text) > 0
        assert "Sample Document" in merged_text

        # VLM should have been called
        assert mock_client.call_count >= 1

        # VLM doesn't provide structured tables/figures
        assert result.tables == []
        assert result.figures == []

    def test_vlm_with_different_models(self, sample_pdf_bytes):
        """Test VLM parser with different model configurations."""
        def mock_pdf_to_images(pdf_bytes):
            return ["mock_image"]

        models_to_test = [
            ("deepseek-vl2", 0.1, 4096),
            ("qwen-vl-plus", 0.2, 2048),
            ("gpt-4-vision-preview", 0.0, 8192),
        ]

        for model_name, temp, max_tokens in models_to_test:
            mock_client = MockVLMClientForIntegration()
            parser = get_parser("pdf_vlm", vlm_client=mock_client, renderer=mock_pdf_to_images)

            pdf_file = io.BytesIO(sample_pdf_bytes)
            result = parser.parse(
                pdf_file,
                options=PdfParseOptions(
                    vlm_model=model_name,
                    vlm_temperature=temp,
                    vlm_max_new_tokens=max_tokens
                )
            )

            # Verify model-specific metadata
            assert result.metadata["vlm_model"] == model_name
            assert len(result.pages) >= 1
            assert mock_client.call_count >= 1


class TestParserComparison:
    """Integration tests comparing different parsers on the same PDF."""

    def test_compare_deepdoc_and_vlm_parsers(self, sample_pdf_bytes):
        """Compare DeepDoc and VLM parsers on the same document."""
        # Mock DeepDoc backend
        mock_deepdoc_backend = MagicMock()
        mock_deepdoc_instance = MagicMock()
        mock_deepdoc_backend.return_value = mock_deepdoc_instance
        mock_deepdoc_instance.return_value = {
            "chunks": [{"text": "DeepDoc extracted text", "page": 1}],
            "tables": [],
            "figures": []
        }

        from llm_infrastructure.preprocessing.parsers.engines.pdf_deepdoc_engine import DeepDocPdfEngine

        with patch.object(DeepDocPdfEngine, "_load_backend_class", return_value=mock_deepdoc_backend):
            deepdoc_parser = get_parser("pdf_deepdoc")
            deepdoc_result = deepdoc_parser.parse(io.BytesIO(sample_pdf_bytes))

        # Parse with VLM
        mock_vlm_client = MockVLMClientForIntegration()
        vlm_parser = get_parser(
            "pdf_vlm",
            vlm_client=mock_vlm_client,
            renderer=lambda pdf: ["mock_image"]
        )
        vlm_result = vlm_parser.parse(io.BytesIO(sample_pdf_bytes))

        # Both should produce valid results
        assert isinstance(deepdoc_result, ParsedDocument)
        assert isinstance(vlm_result, ParsedDocument)

        # VLM should produce more structured markdown output
        vlm_text = vlm_result.merged_text()
        assert "#" in vlm_text or "|" in vlm_text  # Markdown formatting

    def test_all_parsers_handle_same_pdf(self, sample_pdf_bytes):
        """Test that all parsers can handle the same PDF without errors."""
        results = []

        # Test VLM parser
        mock_vlm_client = MockVLMClientForIntegration()
        vlm_parser = get_parser(
            "pdf_vlm",
            vlm_client=mock_vlm_client,
            renderer=lambda pdf: ["mock_image"]
        )
        vlm_result = vlm_parser.parse(io.BytesIO(sample_pdf_bytes))
        results.append(("pdf_vlm", vlm_result))

        # Test DeepDoc parser with mock
        mock_backend = MagicMock()
        mock_instance = MagicMock()
        mock_backend.return_value = mock_instance
        mock_instance.return_value = {
            "chunks": [{"text": "Test", "page": 1}],
            "tables": [],
            "figures": []
        }

        from llm_infrastructure.preprocessing.parsers.engines.pdf_deepdoc_engine import DeepDocPdfEngine

        with patch.object(DeepDocPdfEngine, "_load_backend_class", return_value=mock_backend):
            deepdoc_parser = get_parser("pdf_deepdoc")
            deepdoc_result = deepdoc_parser.parse(io.BytesIO(sample_pdf_bytes))
            results.append(("pdf_deepdoc", deepdoc_result))

        # All parsers should have produced results
        assert len(results) >= 2

        # All results should be valid ParsedDocuments
        for parser_id, result in results:
            assert isinstance(result, ParsedDocument), f"{parser_id} didn't return ParsedDocument"
            assert len(result.pages) >= 1 or len(result.blocks) >= 1, f"{parser_id} didn't extract content"


class TestEdgeCases:
    """Integration tests for edge cases with real PDFs."""

    def test_vlm_empty_pdf_handling(self, sample_pdf_bytes):
        """Test VLM handling of minimal PDFs."""
        mock_client = MockVLMClientForIntegration()
        parser = get_parser(
            "pdf_vlm",
            vlm_client=mock_client,
            renderer=lambda pdf: ["mock_image"]
        )

        result = parser.parse(io.BytesIO(sample_pdf_bytes))

        # Should handle gracefully
        assert isinstance(result, ParsedDocument)
        assert len(result.pages) >= 1

    def test_vlm_with_custom_prompt(self, sample_pdf_bytes):
        """Test VLM parser with custom extraction prompt."""
        mock_client = MockVLMClientForIntegration()
        parser = get_parser(
            "pdf_vlm",
            vlm_client=mock_client,
            renderer=lambda pdf: ["mock_image"]
        )

        custom_prompt = "Extract only the headings and table structure from this page."

        result = parser.parse(
            io.BytesIO(sample_pdf_bytes),
            options=PdfParseOptions(vlm_prompt=custom_prompt)
        )

        # Should have used custom prompt (reflected in metadata)
        assert result.metadata["vlm_prompt"] == custom_prompt
        assert mock_client.call_count >= 1


class TestPerformanceAndReliability:
    """Integration tests for performance and reliability."""

    def test_vlm_parser_consistency(self, sample_pdf_bytes):
        """Test that VLM parser produces consistent results across multiple runs."""
        mock_client = MockVLMClientForIntegration()
        parser = get_parser(
            "pdf_vlm",
            vlm_client=mock_client,
            renderer=lambda pdf: ["mock_image"]
        )

        results = []
        for _ in range(3):
            # Reset mock client
            mock_client.call_count = 0
            pdf_file = io.BytesIO(sample_pdf_bytes)
            result = parser.parse(pdf_file)
            results.append(result.merged_text())

        # All results should be identical
        assert results[0] == results[1] == results[2]

    def test_vlm_factory_lazy_initialization(self, sample_pdf_bytes):
        """Test that VLM factory is only called when needed."""
        factory_calls = []

        def counting_factory(options):
            factory_calls.append(options.vlm_model)
            return MockVLMClientForIntegration()

        parser = get_parser(
            "pdf_vlm",
            vlm_factory=counting_factory,
            renderer=lambda pdf: ["mock_image"]
        )

        # Factory should not be called yet
        assert len(factory_calls) == 0

        # Parse document
        result = parser.parse(
            io.BytesIO(sample_pdf_bytes),
            options=PdfParseOptions(vlm_model="deepseek-vl2")
        )

        # Factory should be called exactly once
        assert len(factory_calls) == 1
        assert factory_calls[0] == "deepseek-vl2"
