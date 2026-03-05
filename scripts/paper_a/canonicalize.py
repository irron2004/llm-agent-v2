from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.domain.doc_type_mapping import DOC_TYPE_GROUPS, normalize_doc_type


_COMPACT_RE = re.compile(r"[\s\-_./]+")
_VLM_NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9가-힣]")
_VLM_MULTI_UNDERSCORE_RE = re.compile(r"_+")

_BATCH_INVALID_RE = re.compile(r"[^a-z0-9_-]+")
_BATCH_SQUEEZE_UNDERSCORE_RE = re.compile(r"_+")


def compact_key(text: str | None) -> str:
    """Match `backend/llm_infrastructure/llm/langgraph_agent.py:_compact_text` semantics."""
    if text is None:
        return ""
    return _COMPACT_RE.sub("", str(text).lower())


def doc_id_variant_vlm(name: str | None) -> str:
    """Mirror `scripts/vlm_es_ingest.py:generate_doc_id` doc_id normalization (name only)."""
    if name is None:
        return "doc"

    doc_id = _VLM_NON_ALNUM_RE.sub("_", str(name))
    doc_id = _VLM_MULTI_UNDERSCORE_RE.sub("_", doc_id)
    doc_id = doc_id.strip("_").lower()
    return doc_id or "doc"


def doc_id_variant_batch_sop(name: str | None) -> str:
    """Mirror `scripts/batch_ingest_sop.sh` doc_id pipeline (name only)."""
    if name is None:
        return ""

    doc_id = str(name).lower()
    doc_id = doc_id.replace(" ", "_")
    doc_id = _BATCH_INVALID_RE.sub("_", doc_id)
    doc_id = _BATCH_SQUEEZE_UNDERSCORE_RE.sub("_", doc_id)
    return doc_id


_DOC_TYPE_ES_KEY_BY_GROUP: dict[str, str] = {
    "SOP": "sop",
    "ts": "ts",
    "setup": "setup",
    "myservice": "myservice",
    "gcb": "gcb",
}

_DOC_TYPE_ES_BY_NORMALIZED: dict[str, str] = {}
for group_name, variants in DOC_TYPE_GROUPS.items():
    es_key = _DOC_TYPE_ES_KEY_BY_GROUP.get(group_name)
    if not es_key:
        continue
    for raw in [group_name, *variants]:
        normalized = normalize_doc_type(raw)
        if normalized and normalized not in _DOC_TYPE_ES_BY_NORMALIZED:
            _DOC_TYPE_ES_BY_NORMALIZED[normalized] = es_key

for raw, es_key in {
    "sop_pdf": "sop",
    "sop_pptx": "sop",
    "setup_manual": "setup",
    "ts_pdf": "ts",
}.items():
    normalized = normalize_doc_type(raw)
    if normalized and normalized not in _DOC_TYPE_ES_BY_NORMALIZED:
        _DOC_TYPE_ES_BY_NORMALIZED[normalized] = es_key


def normalize_doc_type_es(value: str | None) -> str:
    """Normalize doc_type values to ES keys: {sop, ts, setup, myservice, gcb}."""
    normalized = normalize_doc_type(value or "")
    if not normalized:
        return ""

    es_key = _DOC_TYPE_ES_BY_NORMALIZED.get(normalized)
    if es_key:
        return es_key

    # Corpus values like sop_pdf/setup_manual normalize into "sop pdf"/"setup manual".
    head = normalized.split(" ", 1)[0]
    if head in {"sop", "ts", "setup", "myservice", "gcb"}:
        return "sop" if head == "sop" else head

    return ""


def canonicalize_device_name(name: str | None, candidates: Iterable[str] | None) -> str | None:
    """Return the candidate string whose compact form matches `name` (or None)."""
    target = compact_key(name)
    if not target:
        return None

    for candidate in candidates or []:
        if compact_key(candidate) == target:
            return str(candidate)

    return None
