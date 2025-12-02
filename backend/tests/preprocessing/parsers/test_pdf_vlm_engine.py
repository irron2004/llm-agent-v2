"""Tests for VlmPdfEngine."""

import io
from unittest.mock import MagicMock, call, patch

import pytest

from llm_infrastructure.preprocessing.parsers.base import ParsedDocument, PdfParseOptions
from llm_infrastructure.preprocessing.parsers.engines.pdf_vlm_engine import VlmPdfEngine


class MockVLMClient:
    """Mock VLM client for testing."""

    def __init__(self):
        self.calls = []

    def generate(self, image, prompt, **kwargs):
        self.calls.append({"image": image, "prompt": prompt, "kwargs": kwargs})
        # Return different text for different pages (based on call count)
        page_num = len(self.calls)
        return f"# Page {page_num}\n\nThis is content from page {page_num}."


class TestVlmPdfEngine:
    """Test VlmPdfEngine functionality."""

    def test_initialization_default(self):
        """Test engine initialization with defaults."""
        engine = VlmPdfEngine()

        assert engine.vlm_client is None
        assert engine.vlm_factory is None
        assert engine.renderer is None
        assert engine.default_prompt.startswith("Read this page")
        assert engine.content_type == "application/pdf"

    def test_initialization_with_vlm_client(self):
        """Test engine initialization with VLM client."""
        mock_client = MagicMock()
        engine = VlmPdfEngine(vlm_client=mock_client)

        assert engine.vlm_client is mock_client

    def test_initialization_with_vlm_factory(self):
        """Test engine initialization with VLM factory."""
        mock_factory = MagicMock(return_value=MagicMock())
        engine = VlmPdfEngine(vlm_factory=mock_factory)

        assert engine.vlm_factory is mock_factory

    def test_initialization_with_custom_renderer(self):
        """Test engine initialization with custom renderer."""
        mock_renderer = MagicMock(return_value=["image1", "image2"])
        engine = VlmPdfEngine(renderer=mock_renderer)

        assert engine.renderer is mock_renderer

    def test_initialization_with_custom_prompt(self):
        """Test engine initialization with custom default prompt."""
        custom_prompt = "Extract all text from this document page."
        engine = VlmPdfEngine(default_prompt=custom_prompt)

        assert engine.default_prompt == custom_prompt

    def test_render_pdf_with_custom_renderer(self):
        """Test PDF rendering with custom renderer."""
        mock_images = ["image1", "image2", "image3"]
        mock_renderer = MagicMock(return_value=mock_images)
        engine = VlmPdfEngine(renderer=mock_renderer)

        pdf_bytes = b"%PDF-1.4 dummy content"
        images = engine._render_pdf(pdf_bytes)

        mock_renderer.assert_called_once_with(pdf_bytes)
        assert images == mock_images

    def test_render_pdf_without_renderer_raises_import_error(self):
        """Test that rendering without renderer and without pdf2image raises ImportError."""
        engine = VlmPdfEngine()

        # Mock the import to fail
        import sys
        with patch.dict(sys.modules, {"pdf2image": None}):
            with pytest.raises(ImportError, match="pdf2image is required"):
                engine._render_pdf(b"dummy")

    def test_run_raises_when_no_vlm_client(self):
        """Test that run() raises ImportError when no VLM client is provided."""
        engine = VlmPdfEngine()

        with pytest.raises(ImportError, match="vlm_client is required"):
            engine.run(io.BytesIO(b"dummy"))

    def test_run_with_vlm_client(self):
        """Test successful PDF parsing with VLM client."""
        mock_client = MockVLMClient()
        mock_images = ["image1", "image2"]
        mock_renderer = MagicMock(return_value=mock_images)

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        pdf_bytes = b"%PDF-1.4 dummy content"
        result = engine.run(io.BytesIO(pdf_bytes))

        # Verify rendering was called
        mock_renderer.assert_called_once_with(pdf_bytes)

        # Verify VLM client was called for each page
        assert len(mock_client.calls) == 2
        assert mock_client.calls[0]["image"] == "image1"
        assert mock_client.calls[1]["image"] == "image2"

        # Verify result structure
        assert isinstance(result, ParsedDocument)
        assert len(result.pages) == 2
        assert len(result.blocks) == 2
        assert result.pages[0].number == 1
        assert result.pages[1].number == 2
        assert "Page 1" in result.pages[0].text
        assert "Page 2" in result.pages[1].text
        assert result.blocks[0].page == 1
        assert result.blocks[1].page == 2
        assert result.blocks[0].label == "page"
        assert result.tables == []
        assert result.figures == []
        assert result.metadata["parser"] == "pdf_vlm"

    def test_run_with_vlm_factory(self):
        """Test that vlm_factory is called when vlm_client is None."""
        mock_client = MockVLMClient()
        mock_factory = MagicMock(return_value=mock_client)
        mock_renderer = MagicMock(return_value=["image1"])

        engine = VlmPdfEngine(vlm_factory=mock_factory, renderer=mock_renderer)

        opts = PdfParseOptions(vlm_model="deepseek-vl2")
        result = engine.run(io.BytesIO(b"dummy"), options=opts)

        # Verify factory was called with options
        mock_factory.assert_called_once_with(opts)

        # Verify client was used
        assert len(mock_client.calls) == 1
        assert isinstance(result, ParsedDocument)

    def test_run_uses_default_prompt(self):
        """Test that default prompt is used when no custom prompt in options."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1"])
        default_prompt = "Custom default prompt"

        engine = VlmPdfEngine(
            vlm_client=mock_client,
            renderer=mock_renderer,
            default_prompt=default_prompt
        )

        engine.run(io.BytesIO(b"dummy"))

        assert mock_client.calls[0]["prompt"] == default_prompt

    def test_run_uses_custom_prompt_from_options(self):
        """Test that custom prompt from options overrides default."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1"])

        engine = VlmPdfEngine(
            vlm_client=mock_client,
            renderer=mock_renderer,
            default_prompt="Default prompt"
        )

        custom_prompt = "Extract formulas and tables"
        opts = PdfParseOptions(vlm_prompt=custom_prompt)
        engine.run(io.BytesIO(b"dummy"), options=opts)

        assert mock_client.calls[0]["prompt"] == custom_prompt

    def test_run_passes_max_new_tokens(self):
        """Test that max_new_tokens is passed to VLM client."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1"])

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        opts = PdfParseOptions(vlm_max_new_tokens=2048)
        engine.run(io.BytesIO(b"dummy"), options=opts)

        assert mock_client.calls[0]["kwargs"]["max_new_tokens"] == 2048

    def test_run_passes_temperature(self):
        """Test that temperature is passed to VLM client."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1"])

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        opts = PdfParseOptions(vlm_temperature=0.1)
        engine.run(io.BytesIO(b"dummy"), options=opts)

        assert mock_client.calls[0]["kwargs"]["temperature"] == 0.1

    def test_run_passes_all_vlm_options(self):
        """Test that all VLM options are passed correctly."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1"])

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        opts = PdfParseOptions(
            vlm_model="deepseek-vl2",
            vlm_prompt="Custom prompt",
            vlm_max_new_tokens=4096,
            vlm_temperature=0.2
        )
        result = engine.run(io.BytesIO(b"dummy"), options=opts)

        # Check VLM client call
        assert mock_client.calls[0]["prompt"] == "Custom prompt"
        assert mock_client.calls[0]["kwargs"]["max_new_tokens"] == 4096
        assert mock_client.calls[0]["kwargs"]["temperature"] == 0.2

        # Check metadata
        assert result.metadata["vlm_model"] == "deepseek-vl2"
        assert result.metadata["vlm_prompt"] == "Custom prompt"

    def test_run_handles_multiple_pages(self):
        """Test processing multiple pages."""
        mock_client = MockVLMClient()
        mock_images = [f"image{i}" for i in range(5)]
        mock_renderer = MagicMock(return_value=mock_images)

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        result = engine.run(io.BytesIO(b"dummy"))

        assert len(result.pages) == 5
        assert len(result.blocks) == 5
        assert len(mock_client.calls) == 5

        # Verify page numbers are sequential
        for i, page in enumerate(result.pages, start=1):
            assert page.number == i
            assert f"Page {i}" in page.text

    def test_run_seeks_file_to_beginning(self):
        """Test that file is seeked to beginning before reading."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1"])

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        file = io.BytesIO(b"dummy content")
        file.read()  # Move cursor to end
        assert file.tell() == len(b"dummy content")

        engine.run(file)

        # File should have been read (cursor at end again after reading)
        assert file.tell() == len(b"dummy content")

    def test_run_handles_non_seekable_file(self):
        """Test that non-seekable files are handled gracefully."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1"])

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        # Create a file-like object without seek
        class NonSeekableFile:
            def read(self):
                return b"dummy content"

        file = NonSeekableFile()
        result = engine.run(file)  # Should not raise

        assert isinstance(result, ParsedDocument)

    def test_metadata_includes_parser_name(self):
        """Test that metadata includes parser name."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1"])

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        result = engine.run(io.BytesIO(b"dummy"))

        assert result.metadata["parser"] == "pdf_vlm"

    def test_metadata_includes_vlm_model(self):
        """Test that metadata includes VLM model name."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1"])

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        opts = PdfParseOptions(vlm_model="deepseek-vl2-large")
        result = engine.run(io.BytesIO(b"dummy"), options=opts)

        assert result.metadata["vlm_model"] == "deepseek-vl2-large"

    def test_metadata_includes_prompt(self):
        """Test that metadata includes the prompt used."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1"])
        default_prompt = "Default extraction prompt"

        engine = VlmPdfEngine(
            vlm_client=mock_client,
            renderer=mock_renderer,
            default_prompt=default_prompt
        )

        result = engine.run(io.BytesIO(b"dummy"))

        assert result.metadata["vlm_prompt"] == default_prompt

    def test_blocks_have_correct_structure(self):
        """Test that blocks have correct structure."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1", "image2"])

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        result = engine.run(io.BytesIO(b"dummy"))

        for i, block in enumerate(result.blocks, start=1):
            assert block.page == i
            assert block.label == "page"
            assert block.text == result.pages[i-1].text
            assert block.bbox is None  # VLM doesn't provide coordinates
            assert block.confidence is None

    def test_pages_have_correct_structure(self):
        """Test that pages have correct structure."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1", "image2"])

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        result = engine.run(io.BytesIO(b"dummy"))

        for i, page in enumerate(result.pages, start=1):
            assert page.number == i
            assert isinstance(page.text, str)
            assert len(page.text) > 0
            assert page.width is None  # VLM doesn't provide dimensions
            assert page.height is None

    def test_no_tables_or_figures_in_output(self):
        """Test that tables and figures are empty (VLM returns continuous text)."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1"])

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        result = engine.run(io.BytesIO(b"dummy"))

        assert result.tables == []
        assert result.figures == []

    def test_content_type(self):
        """Test that content_type is set correctly."""
        engine = VlmPdfEngine()

        assert engine.content_type == "application/pdf"

    def test_run_with_empty_pdf(self):
        """Test handling of PDF with no pages (renderer returns empty list)."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=[])

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        result = engine.run(io.BytesIO(b"dummy"))

        assert len(result.pages) == 0
        assert len(result.blocks) == 0
        assert len(mock_client.calls) == 0

    def test_vlm_factory_called_only_once(self):
        """Test that vlm_factory is called only once even for multiple pages."""
        mock_client = MockVLMClient()
        mock_factory = MagicMock(return_value=mock_client)
        mock_renderer = MagicMock(return_value=["image1", "image2", "image3"])

        engine = VlmPdfEngine(vlm_factory=mock_factory, renderer=mock_renderer)

        engine.run(io.BytesIO(b"dummy"))

        # Factory should be called exactly once
        assert mock_factory.call_count == 1

    def test_vlm_client_persists_across_calls(self):
        """Test that vlm_client created by factory persists across run() calls."""
        mock_client = MockVLMClient()
        mock_factory = MagicMock(return_value=mock_client)
        mock_renderer = MagicMock(return_value=["image1"])

        engine = VlmPdfEngine(vlm_factory=mock_factory, renderer=mock_renderer)

        # First call
        engine.run(io.BytesIO(b"dummy1"))
        assert mock_factory.call_count == 1
        assert len(mock_client.calls) == 1

        # Second call should reuse client
        engine.run(io.BytesIO(b"dummy2"))
        assert mock_factory.call_count == 1  # Still 1
        assert len(mock_client.calls) == 2  # But client was called again

    def test_options_default_to_none(self):
        """Test that None options are handled gracefully."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1"])

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        # Pass no options
        result = engine.run(io.BytesIO(b"dummy"), options=None)

        # Should use defaults
        assert isinstance(result, ParsedDocument)
        assert mock_client.calls[0]["kwargs"] == {}  # No extra kwargs passed

    def test_vlm_options_none_values_not_passed(self):
        """Test that None VLM option values are not passed to client."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1"])

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        opts = PdfParseOptions(
            vlm_max_new_tokens=None,
            vlm_temperature=None
        )
        engine.run(io.BytesIO(b"dummy"), options=opts)

        # Should not include None values in kwargs
        assert "max_new_tokens" not in mock_client.calls[0]["kwargs"]
        assert "temperature" not in mock_client.calls[0]["kwargs"]


class TestVlmPdfEngineWithDifferentModels:
    """Test VlmPdfEngine with various VLM model configurations."""

    def test_run_with_deepseek_vl2_model(self):
        """Test parsing with DeepSeek-VL2 model."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1"])

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        opts = PdfParseOptions(vlm_model="deepseek-vl2")
        result = engine.run(io.BytesIO(b"dummy"), options=opts)

        assert result.metadata["vlm_model"] == "deepseek-vl2"
        assert len(mock_client.calls) == 1
        assert isinstance(result, ParsedDocument)

    def test_run_with_qwen_vl_model(self):
        """Test parsing with Qwen-VL model."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1"])

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        opts = PdfParseOptions(vlm_model="qwen-vl-plus")
        result = engine.run(io.BytesIO(b"dummy"), options=opts)

        assert result.metadata["vlm_model"] == "qwen-vl-plus"
        assert len(result.pages) == 1

    def test_run_with_gpt4_vision_model(self):
        """Test parsing with GPT-4-Vision model."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1"])

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        opts = PdfParseOptions(vlm_model="gpt-4-vision-preview")
        result = engine.run(io.BytesIO(b"dummy"), options=opts)

        assert result.metadata["vlm_model"] == "gpt-4-vision-preview"
        assert len(result.pages) == 1

    def test_run_with_custom_vlm_model(self):
        """Test parsing with custom/generic VLM model name."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1", "image2"])

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        opts = PdfParseOptions(vlm_model="custom-vision-model-v1")
        result = engine.run(io.BytesIO(b"dummy"), options=opts)

        assert result.metadata["vlm_model"] == "custom-vision-model-v1"
        assert len(result.pages) == 2

    def test_multiple_models_sequential_parsing(self):
        """Test parsing same document with different models sequentially."""
        mock_renderer = MagicMock(return_value=["image1"])
        pdf_content = io.BytesIO(b"dummy content")

        models = ["deepseek-vl2", "qwen-vl-plus", "gpt-4-vision-preview"]
        results = []

        for model_name in models:
            mock_client = MockVLMClient()
            engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

            pdf_content.seek(0)  # Reset file position
            opts = PdfParseOptions(vlm_model=model_name)
            result = engine.run(pdf_content, options=opts)
            results.append(result)

            # Verify model name in metadata
            assert result.metadata["vlm_model"] == model_name
            assert len(mock_client.calls) == 1

        # All results should be ParsedDocuments
        assert all(isinstance(r, ParsedDocument) for r in results)
        assert len(results) == 3

    def test_model_specific_temperature_settings(self):
        """Test that different models can use different temperature settings."""
        mock_renderer = MagicMock(return_value=["image1"])

        # Model configurations with different temperatures
        model_configs = [
            ("deepseek-vl2", 0.1),
            ("qwen-vl-plus", 0.3),
            ("gpt-4-vision-preview", 0.0),
        ]

        for model_name, temperature in model_configs:
            mock_client = MockVLMClient()
            engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

            opts = PdfParseOptions(
                vlm_model=model_name,
                vlm_temperature=temperature
            )
            result = engine.run(io.BytesIO(b"dummy"), options=opts)

            # Verify temperature was passed correctly
            assert mock_client.calls[0]["kwargs"]["temperature"] == temperature
            assert result.metadata["vlm_model"] == model_name

    def test_model_specific_max_tokens_settings(self):
        """Test that different models can use different max_tokens settings."""
        mock_renderer = MagicMock(return_value=["image1"])

        # Model configurations with different max tokens
        model_configs = [
            ("deepseek-vl2", 4096),
            ("qwen-vl-plus", 2048),
            ("gpt-4-vision-preview", 8192),
        ]

        for model_name, max_tokens in model_configs:
            mock_client = MockVLMClient()
            engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

            opts = PdfParseOptions(
                vlm_model=model_name,
                vlm_max_new_tokens=max_tokens
            )
            result = engine.run(io.BytesIO(b"dummy"), options=opts)

            # Verify max_tokens was passed correctly
            assert mock_client.calls[0]["kwargs"]["max_new_tokens"] == max_tokens
            assert result.metadata["vlm_model"] == model_name

    def test_model_with_version_suffix(self):
        """Test parsing with model names that include version suffixes."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1"])

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        # Test various version formats
        model_names = [
            "deepseek-vl2-7b",
            "qwen-vl-plus-v1.5",
            "gpt-4-vision-preview-2024",
            "custom-model:latest",
        ]

        for model_name in model_names:
            mock_client.calls.clear()  # Reset calls
            opts = PdfParseOptions(vlm_model=model_name)
            result = engine.run(io.BytesIO(b"dummy"), options=opts)

            assert result.metadata["vlm_model"] == model_name
            assert len(mock_client.calls) == 1

    def test_vlm_factory_receives_model_name(self):
        """Test that VLM factory receives the model name from options."""
        mock_client = MockVLMClient()

        def mock_factory(options):
            # Verify factory receives options with model name
            assert options.vlm_model == "deepseek-vl2-large"
            return mock_client

        mock_renderer = MagicMock(return_value=["image1"])
        engine = VlmPdfEngine(vlm_factory=mock_factory, renderer=mock_renderer)

        opts = PdfParseOptions(vlm_model="deepseek-vl2-large")
        result = engine.run(io.BytesIO(b"dummy"), options=opts)

        assert result.metadata["vlm_model"] == "deepseek-vl2-large"

    def test_no_model_name_specified(self):
        """Test parsing when no model name is specified in options."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["image1"])

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        opts = PdfParseOptions()  # No vlm_model specified
        result = engine.run(io.BytesIO(b"dummy"), options=opts)

        # Should work fine, metadata should have None for vlm_model
        assert result.metadata["vlm_model"] is None
        assert len(result.pages) == 1

    def test_model_name_persistence_across_pages(self):
        """Test that model name is consistent across multi-page documents."""
        mock_client = MockVLMClient()
        mock_renderer = MagicMock(return_value=["img1", "img2", "img3"])

        engine = VlmPdfEngine(vlm_client=mock_client, renderer=mock_renderer)

        opts = PdfParseOptions(vlm_model="deepseek-vl2")
        result = engine.run(io.BytesIO(b"dummy"), options=opts)

        # Metadata should be set once for the entire document
        assert result.metadata["vlm_model"] == "deepseek-vl2"

        # All pages should be processed
        assert len(result.pages) == 3
        assert len(mock_client.calls) == 3

    def test_model_name_in_error_context(self):
        """Test that model name is available in error context."""
        def failing_factory(options):
            raise ImportError(f"Failed to load model: {options.vlm_model}")

        engine = VlmPdfEngine(vlm_factory=failing_factory, renderer=MagicMock(return_value=[]))

        opts = PdfParseOptions(vlm_model="non-existent-model")

        with pytest.raises(ImportError, match="non-existent-model"):
            engine.run(io.BytesIO(b"dummy"), options=opts)
