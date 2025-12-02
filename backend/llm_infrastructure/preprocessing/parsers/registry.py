"""Parser registry for infrastructure-level parsers."""

from __future__ import annotations

from typing import Dict, Iterable, Type

from .base import BaseParser

PARSER_REGISTRY: Dict[str, Type[BaseParser]] = {}


def register_parser(parser_id: str, parser_cls: Type[BaseParser]) -> None:
    if not parser_id:
        raise ValueError("parser_id must be a non-empty string")
    PARSER_REGISTRY[parser_id] = parser_cls


def get_parser(parser_id: str, **kwargs) -> BaseParser:
    parser_cls = PARSER_REGISTRY.get(parser_id)
    if not parser_cls:
        available = ", ".join(sorted(PARSER_REGISTRY.keys()))
        raise KeyError(f"Parser '{parser_id}' is not registered. Available: {available}")
    return parser_cls(**kwargs)


def list_parsers() -> Iterable[str]:
    return PARSER_REGISTRY.keys()


__all__ = ["PARSER_REGISTRY", "get_parser", "list_parsers", "register_parser"]
