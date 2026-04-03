"""Component dictionary for relation-based chunk linking.

Builds a canonical dictionary of component/topic terms from the chunk_v3 manifest.
Used to extract `components` field from chunk content during ingestion and batch update.

Usage:
    from backend.domain.component_dictionary import get_component_dict, match_components

    comp_dict = get_component_dict()
    matched = match_components("Baratron Gauge 교체 후 Zero Adj 수행", comp_dict)
    # → ["BARATRON GAUGE"]
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Minimum token length to avoid false positives (e.g., "8" matching everywhere)
_MIN_MATCH_LEN = 3

# Topics too generic to be useful as component identifiers.
# These match too broadly and create noise in relation linking.
# Will be refined after analyzing extraction results.
_BLOCKLIST = frozenset({
    "ALL",
    "ETC",
    "PM",
    "8계통 CHECK",
})

# Default manifest path
_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data" / "chunk_v3_manifest.json"
_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "component_dictionary.json"


@dataclass(frozen=True)
class ComponentEntry:
    """A canonical component/topic entry."""

    canonical: str  # Uppercase canonical name (e.g., "BARATRON GAUGE")
    pattern: re.Pattern[str]  # Compiled regex for matching
    module: str = ""  # Parent module if available (e.g., "PM", "EFEM")


def _build_pattern(name: str) -> re.Pattern[str]:
    """Build a case-insensitive regex for a component name.

    Splits on whitespace/underscore, escapes each word, then joins
    with a flexible whitespace/underscore pattern.
    """
    parts = re.split(r"[\s_]+", name)
    parts = [re.escape(p) for p in parts if p]
    if not parts:
        return re.compile(re.escape(name), re.IGNORECASE)
    pattern_str = r"[\s_]+".join(parts)
    return re.compile(pattern_str, re.IGNORECASE)


def build_component_dict_from_manifest(
    manifest_path: str | Path | None = None,
) -> list[ComponentEntry]:
    """Build component dictionary from chunk_v3_manifest.json.

    Extracts unique topic values from SOP/TS entries and creates
    ComponentEntry objects with regex patterns for matching.

    Args:
        manifest_path: Path to manifest JSON. Uses default if None.

    Returns:
        List of ComponentEntry objects sorted by name length (longest first).
    """
    path = Path(manifest_path) if manifest_path else _MANIFEST_PATH
    if not path.exists():
        logger.warning("Manifest not found: %s", path)
        return []

    with open(path, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)

    # Collect unique topics with module info
    topic_modules: dict[str, str] = {}  # canonical -> module
    for entry in data:
        topic = (entry.get("topic") or "").strip()
        if not topic or len(topic) < _MIN_MATCH_LEN:
            continue
        canonical = topic.upper()
        if canonical in _BLOCKLIST:
            continue
        if canonical not in topic_modules:
            module = (entry.get("module") or "").strip().upper()
            topic_modules[canonical] = module

    entries = [
        ComponentEntry(
            canonical=canonical,
            pattern=_build_pattern(canonical),
            module=module,
        )
        for canonical, module in topic_modules.items()
    ]

    # Sort longest first to match "EFEM CONTROLLER" before "CONTROLLER"
    entries.sort(key=lambda e: len(e.canonical), reverse=True)

    logger.info("Built component dictionary: %d entries", len(entries))
    return entries


def save_component_dict(
    entries: list[ComponentEntry],
    output_path: str | Path | None = None,
) -> Path:
    """Save component dictionary to JSON cache file.

    Args:
        entries: List of ComponentEntry objects.
        output_path: Output path. Uses default if None.

    Returns:
        Path to saved file.
    """
    path = Path(output_path) if output_path else _CACHE_PATH
    data = [
        {"canonical": e.canonical, "module": e.module}
        for e in entries
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Saved component dictionary to %s (%d entries)", path, len(data))
    return path


def load_component_dict(
    cache_path: str | Path | None = None,
) -> list[ComponentEntry]:
    """Load component dictionary from JSON cache file.

    Args:
        cache_path: Path to cache JSON. Uses default if None.

    Returns:
        List of ComponentEntry objects.
    """
    path = Path(cache_path) if cache_path else _CACHE_PATH
    if not path.exists():
        logger.info("Cache not found, building from manifest: %s", path)
        entries = build_component_dict_from_manifest()
        if entries:
            save_component_dict(entries, path)
        return entries

    with open(path, encoding="utf-8") as f:
        data: list[dict[str, str]] = json.load(f)

    entries = [
        ComponentEntry(
            canonical=item["canonical"],
            pattern=_build_pattern(item["canonical"]),
            module=item.get("module", ""),
        )
        for item in data
    ]
    entries.sort(key=lambda e: len(e.canonical), reverse=True)
    logger.info("Loaded component dictionary from cache: %d entries", len(entries))
    return entries


@lru_cache(maxsize=1)
def get_component_dict() -> tuple[ComponentEntry, ...]:
    """Get cached component dictionary (singleton).

    Returns:
        Tuple of ComponentEntry objects (immutable for caching).
    """
    return tuple(load_component_dict())


def match_components(
    text: str,
    comp_dict: tuple[ComponentEntry, ...] | list[ComponentEntry] | None = None,
    *,
    max_components: int = 10,
) -> list[str]:
    """Match component names in text using the dictionary.

    Args:
        text: Text to search for component names.
        comp_dict: Component dictionary. Uses cached singleton if None.
        max_components: Maximum number of components to return.

    Returns:
        List of canonical component names found in text.
    """
    if not text:
        return []

    if comp_dict is None:
        comp_dict = get_component_dict()

    matched: list[str] = []
    matched_spans: list[tuple[int, int]] = []  # Track matched positions

    for entry in comp_dict:
        if len(matched) >= max_components:
            break

        m = entry.pattern.search(text)
        if m is None:
            continue

        # Check for overlapping matches (longer match takes priority)
        span = m.span()
        overlapping = any(
            span[0] < existing[1] and span[1] > existing[0]
            for existing in matched_spans
        )
        if overlapping:
            continue

        matched.append(entry.canonical)
        matched_spans.append(span)

    return matched


def match_components_fuzzy(
    text: str,
    comp_dict: tuple[ComponentEntry, ...] | list[ComponentEntry] | None = None,
    *,
    threshold: int = 85,
    max_components: int = 10,
) -> list[str]:
    """Match component names using fuzzy matching (for non-standardized text).

    Uses rapidfuzz for partial ratio matching. Falls back to exact matching
    if rapidfuzz is not available.

    Args:
        text: Text to search (typically doc_description or title).
        comp_dict: Component dictionary. Uses cached singleton if None.
        threshold: Minimum fuzzy match score (0-100).
        max_components: Maximum number of components to return.

    Returns:
        List of canonical component names matched.
    """
    if not text:
        return []

    if comp_dict is None:
        comp_dict = get_component_dict()

    try:
        from rapidfuzz import fuzz
    except ImportError:
        logger.debug("rapidfuzz not available, falling back to exact matching")
        return match_components(text, comp_dict, max_components=max_components)

    text_upper = text.upper()
    scored: list[tuple[str, int]] = []

    for entry in comp_dict:
        if len(entry.canonical) < _MIN_MATCH_LEN:
            continue
        score = fuzz.partial_ratio(entry.canonical, text_upper)
        if score >= threshold:
            scored.append((entry.canonical, score))

    # Sort by score descending, take top N
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [name for name, _ in scored[:max_components]]


__all__ = [
    "ComponentEntry",
    "build_component_dict_from_manifest",
    "save_component_dict",
    "load_component_dict",
    "get_component_dict",
    "match_components",
    "match_components_fuzzy",
]
