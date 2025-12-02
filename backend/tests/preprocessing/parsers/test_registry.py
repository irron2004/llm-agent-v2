"""Tests for parser registry."""

import pytest

from llm_infrastructure.preprocessing.parsers.base import BaseParser, ParsedDocument, PdfParseOptions
from llm_infrastructure.preprocessing.parsers.registry import (
    PARSER_REGISTRY,
    get_parser,
    list_parsers,
    register_parser,
)


class MockParser(BaseParser):
    content_type = "test/mock"

    def __init__(self, test_arg=None):
        self.test_arg = test_arg

    def parse(self, file, options=None):
        return ParsedDocument(metadata={"parser": "mock", "test_arg": self.test_arg})


class TestRegistry:
    def setup_method(self):
        """Clear registry before each test."""
        PARSER_REGISTRY.clear()

    def test_register_parser(self):
        register_parser("mock", MockParser)
        assert "mock" in PARSER_REGISTRY
        assert PARSER_REGISTRY["mock"] == MockParser

    def test_register_empty_id_raises_error(self):
        with pytest.raises(ValueError, match="must be a non-empty string"):
            register_parser("", MockParser)

    def test_get_parser_success(self):
        register_parser("mock", MockParser)
        parser = get_parser("mock")
        assert isinstance(parser, MockParser)
        assert parser.test_arg is None

    def test_get_parser_with_kwargs(self):
        register_parser("mock", MockParser)
        parser = get_parser("mock", test_arg="custom_value")
        assert isinstance(parser, MockParser)
        assert parser.test_arg == "custom_value"

    def test_get_parser_not_found(self):
        register_parser("mock1", MockParser)
        with pytest.raises(KeyError, match="Parser 'unknown' is not registered"):
            get_parser("unknown")

    def test_get_parser_error_message_shows_available(self):
        register_parser("parser_a", MockParser)
        register_parser("parser_b", MockParser)
        with pytest.raises(KeyError, match="Available: parser_a, parser_b"):
            get_parser("not_registered")

    def test_list_parsers_empty(self):
        parsers = list(list_parsers())
        assert parsers == []

    def test_list_parsers_multiple(self):
        register_parser("parser_a", MockParser)
        register_parser("parser_b", MockParser)
        parsers = sorted(list_parsers())
        assert parsers == ["parser_a", "parser_b"]

    def test_overwrite_existing_parser(self):
        """Registry allows overwriting existing parsers."""
        register_parser("mock", MockParser)
        register_parser("mock", MockParser)  # Overwrite
        assert "mock" in PARSER_REGISTRY

    def test_registry_is_shared(self):
        """Test that registry is a module-level singleton."""
        from llm_infrastructure.preprocessing.parsers import registry

        registry.register_parser("test_shared", MockParser)
        assert "test_shared" in PARSER_REGISTRY