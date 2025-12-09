"""Query expansion module for generating multiple search queries."""

from .base import BaseQueryExpander, ExpandedQueries
from .registry import (
    QueryExpanderRegistry,
    register_query_expander,
    get_query_expander,
)

# Import adapters to trigger registration
from . import adapters  # noqa: F401

__all__ = [
    "BaseQueryExpander",
    "ExpandedQueries",
    "QueryExpanderRegistry",
    "register_query_expander",
    "get_query_expander",
]