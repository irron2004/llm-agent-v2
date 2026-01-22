"""Doc type grouping and normalization helpers."""

from __future__ import annotations

import re
from typing import Iterable


DOC_TYPE_GROUPS: dict[str, list[str]] = {
    "myservice": [
        "myservice",
    ],
    "SOP": [
        "SOP",
        "SOP/Manual",
        "Global SOP",
        "Global_SOP",
        "SOP/Manual/Guide",
        "SOP Appendix",
        "SOP/Set‑up Manual",
        "generic",
    ],
    "ts": [
        "문제 해결 가이드",
        "Trouble Shooting Guide",
        "Guide",
        "ts",
        "Troubleshooting Guide",
    ],
    "setup": [
        "Installation Manual",
        "매뉴얼",
        "설치 매뉴얼",
        "setup",
        "Manual",
    ],
    "gcb": [
        "gcb",
        "maintenance",
    ],
}


_UNICODE_HYPHENS = re.compile(r"[\u2010-\u2015\u2212]")
_WS = re.compile(r"\s+")
_WS_SLASH = re.compile(r"\s*/\s*")


def normalize_doc_type(value: str) -> str:
    if value is None:
        return ""
    cleaned = str(value).strip().lower()
    cleaned = cleaned.replace("_", " ")
    cleaned = _UNICODE_HYPHENS.sub("-", cleaned)
    cleaned = _WS_SLASH.sub("/", cleaned)
    cleaned = _WS.sub(" ", cleaned)
    return cleaned


_GROUP_ORDER = ["myservice", "SOP", "ts", "setup", "gcb"]
_GROUP_NAME_BY_NORMALIZED = {
    normalize_doc_type(name): name for name in _GROUP_ORDER
}
_GROUP_VARIANTS_NORMALIZED = {
    name: {normalize_doc_type(v) for v in variants}
    for name, variants in DOC_TYPE_GROUPS.items()
}


def expand_doc_type_selection(selected: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for value in selected or []:
        if value is None:
            continue
        normalized = normalize_doc_type(value)
        if not normalized:
            continue
        group_name = _GROUP_NAME_BY_NORMALIZED.get(normalized)
        if group_name:
            expanded.extend(DOC_TYPE_GROUPS.get(group_name, []))
            continue
        expanded.append(str(value).strip())
    return _dedupe(expanded)


def group_doc_type_buckets(buckets: Iterable[dict], use_unique_docs: bool = False) -> list[dict]:
    counts = {name: 0 for name in _GROUP_ORDER}
    for bucket in buckets or []:
        key = bucket.get("key")
        if not key:
            continue
        normalized = normalize_doc_type(str(key))
        for group_name, variants in _GROUP_VARIANTS_NORMALIZED.items():
            if normalized in variants:
                # Use unique doc count if available and requested
                if use_unique_docs:
                    count = bucket.get("unique_docs", {}).get("value", bucket.get("doc_count", 0))
                else:
                    count = bucket.get("doc_count", 0)
                counts[group_name] += int(count)
                break
    return [{"name": name, "doc_count": counts[name]} for name in _GROUP_ORDER]


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out
