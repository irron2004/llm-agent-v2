"""5종 문서별 chunker 구현.

- chunk_vlm_parsed: SOP, TS, Setup Manual (VLM JSON -> ChunkV3Document)
- chunk_myservice: MyService TXT -> 4섹션 통합 chunk
- chunk_gcb: GCB raw JSON -> Title prefix + fixed-size chunking
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .common import (
    ChunkV3Document,
    canonicalize_doc_type,
    compute_content_hash,
    generate_chunk_id,
    load_vlm_result,
)


# =============================================================================
# 파일명 -> 메타데이터 파싱
# =============================================================================


def _parse_sop_filename(filename: str) -> dict[str, str]:
    """SOP 파일명에서 device_name, chapter 추출.

    패턴: Global SOP_{DEVICE}_{CHAPTER}.pdf/.pptx
    예: Global SOP_GENEVA xp_REP_PM_CHUCK MOTOR.pdf
        -> device_name: GENEVA XP, chapter: REP_PM_CHUCK MOTOR
    """
    stem = Path(filename).stem
    match = re.match(r"Global\s+SOP[_\s]+(.+)", stem, re.IGNORECASE)
    if not match:
        return {"device_name": "", "chapter": ""}

    rest = match.group(1)
    parts = rest.split("_", 1)
    device_name = parts[0].strip().upper()
    chapter = parts[1].strip() if len(parts) > 1 else ""

    # _EN, _KR suffix 제거
    chapter = re.sub(r"[_\s]+(EN|KR)\s*$", "", chapter, flags=re.IGNORECASE)

    return {"device_name": device_name, "chapter": chapter}


def _parse_ts_filename(filename: str) -> dict[str, str]:
    """TS 파일명에서 device_name, chapter 추출.

    패턴: {DEVICE}_{MODULE}_Trouble_Shooting_Guide_{TOPIC}.pdf
    예: PRECIA_PM_Trouble_Shooting_Guide_Chuck Abnormal.pdf
        -> device_name: PRECIA, chapter: PM_Chuck Abnormal
    """
    stem = Path(filename).stem
    match = re.match(
        r"(.+?)_Trouble[_\s]+Shooting[_\s]+Guide[_\s]+(.*)",
        stem,
        re.IGNORECASE,
    )
    if not match:
        return {"device_name": "", "chapter": stem}

    prefix = match.group(1).strip()
    topic = match.group(2).strip()

    prefix_parts = prefix.split("_", 1)
    device_name = prefix_parts[0].strip().upper()
    module = prefix_parts[1].strip() if len(prefix_parts) > 1 else ""

    chapter = f"{module}_{topic}" if module else topic
    return {"device_name": device_name, "chapter": chapter}


def _parse_setup_manual_filename(filename: str) -> dict[str, str]:
    """Setup Manual 파일명에서 device_name 추출.

    패턴: Set_Up_Manual_{DEVICE}.pdf
    예: Set_Up_Manual_SUPRA_Nm.pdf -> device_name: SUPRA NM
    """
    stem = Path(filename).stem
    match = re.match(r"Set[_\s]+Up[_\s]+Manual[_\s]+(.*)", stem, re.IGNORECASE)
    if not match:
        return {"device_name": "", "chapter": ""}

    device_raw = match.group(1).strip()
    device_name = device_raw.replace("_", " ").upper()
    return {"device_name": device_name, "chapter": ""}


FILENAME_PARSERS = {
    "sop": _parse_sop_filename,
    "ts": _parse_ts_filename,
    "setup": _parse_setup_manual_filename,
}

DEFAULT_MANIFEST_PATH = Path("data/chunk_v3_manifest.json")


@lru_cache(maxsize=8)
def _load_manifest_index(manifest_path: str) -> dict[tuple[str, str], dict[str, Any]]:
    """manifest를 (doc_type, file_name) 키 인덱스로 로드."""
    path = Path(manifest_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}

    index: dict[tuple[str, str], dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        doc_type = canonicalize_doc_type(str(item.get("doc_type", "")).strip().lower())
        file_name = str(item.get("file_name", "")).strip().lower()
        if not doc_type or not file_name:
            continue
        index[(doc_type, file_name)] = item
    return index


def _find_manifest_meta(
    doc_type: str,
    source_file: str,
    manifest_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """doc_type + source_file로 manifest 메타 조회."""
    if not source_file:
        return None
    manifest = str(Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH)
    index = _load_manifest_index(manifest)
    if not index:
        return None
    key = (
        canonicalize_doc_type(doc_type.strip().lower()),
        Path(source_file).name.strip().lower(),
    )
    return index.get(key)


def _word_window_split(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= max_tokens:
        return [" ".join(words).strip()]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        segment = " ".join(words[start:end]).strip()
        if segment:
            chunks.append(segment)
        if end >= len(words):
            break
        start = max(end - overlap_tokens, start + 1)
    return chunks


def _split_blocks(text: str) -> list[str]:
    return [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]


def _chunk_page_by_heading(
    text: str,
    heading_re: re.Pattern[str],
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, str]]:
    blocks = _split_blocks(text)
    if not blocks:
        return []

    sections: list[tuple[str, list[str]]] = []
    title = ""
    section_blocks: list[str] = []
    for block in blocks:
        first = block.splitlines()[0].strip() if block.splitlines() else ""
        if first and heading_re.search(first):
            if section_blocks:
                sections.append((title, section_blocks))
            title = first
            section_blocks = []
            continue
        section_blocks.append(block)

    if section_blocks:
        sections.append((title, section_blocks))
    if not sections:
        sections = [("", blocks)]

    out: list[tuple[str, str]] = []
    for section_title, blocks_in_section in sections:
        merged = "\n\n".join(blocks_in_section).strip()
        if not merged:
            continue
        segments = _word_window_split(merged, max_tokens, overlap_tokens) or [merged]
        for segment in segments:
            if segment.strip():
                out.append((section_title, segment.strip()))
    return out


# =============================================================================
# VLM parsed 문서 chunker (SOP, TS, Setup Manual)
# =============================================================================


def chunk_vlm_parsed(
    doc_type: str,
    vlm_json_path: str | Path,
    lang: str = "ko",
    manifest_path: str | Path | None = None,
) -> list[ChunkV3Document]:
    """VLM JSON -> ChunkV3Document[] (SOP, TS, Setup Manual 공용).

    페이지 = chunk 단위. 메타데이터는 파일명에서 파싱.

    Args:
        doc_type: 문서 종류 (sop, ts, setup_manual)
        vlm_json_path: VLM 파싱 결과 JSON 경로
        lang: 언어 코드
        manifest_path: 파일명 정규화 manifest 경로 (없으면 filename parser fallback)

    Returns:
        ChunkV3Document 리스트
    """
    data = load_vlm_result(vlm_json_path)
    doc_id = data["doc_id"]
    source_file = data.get("source_file", "")

    raw_doc_type = str(doc_type)
    canonical_doc_type = canonicalize_doc_type(doc_type)

    parser_fn = FILENAME_PARSERS.get(
        canonical_doc_type, lambda _f: {"device_name": "", "chapter": ""}
    )
    parsed_meta = parser_fn(source_file)
    manifest_meta = _find_manifest_meta(
        canonical_doc_type, source_file, manifest_path=manifest_path
    )

    device_name = str(parsed_meta.get("device_name", "") or "")
    chapter = str(parsed_meta.get("chapter", "") or "")
    work_type = ""
    module = ""
    topic = ""
    meta_source = "filename_parser"

    if manifest_meta:
        meta_source = "manifest"
        device_name = str(manifest_meta.get("device_name", "") or device_name)
        work_type = str(manifest_meta.get("work_type", "") or "")
        module = str(manifest_meta.get("module", "") or "")
        topic = str(manifest_meta.get("topic", "") or "")
        if topic:
            if canonical_doc_type == "ts" and module:
                chapter = f"{module}_{topic}"
            else:
                chapter = topic

    chunks: list[ChunkV3Document] = []
    next_chunk_index = 0
    setup_sequence_no = 1

    sop_heading = re.compile(
        r"(^STEP\s*\d+|^\d+(?:\.\d+)*\s+|^(목적|범위|준비물|절차|주의|주의사항|경고|참고|점검|교체|조정)\b)",
        re.IGNORECASE,
    )
    ts_heading = re.compile(
        r"^(증상|원인|조치|결과|symptom|cause|action|solution|countermeasure|result)\b",
        re.IGNORECASE,
    )
    setup_heading = re.compile(
        r"(^(chapter|unit|section)\s+\d+|^(장|챕터|유닛|절)\s*\d+|^STEP\s*\d+|^\d+\))",
        re.IGNORECASE,
    )

    for page_idx, page in enumerate(data.get("pages", [])):
        text = page.get("text", "").strip()
        if not text:
            continue

        if canonical_doc_type == "sop":
            page_segments = _chunk_page_by_heading(text, sop_heading, 700, 80)
        elif canonical_doc_type == "ts":
            page_segments = _chunk_page_by_heading(text, ts_heading, 650, 60)
        elif canonical_doc_type == "setup":
            page_segments = _chunk_page_by_heading(text, setup_heading, 750, 80)
        else:
            page_segments = [("", text)]

        alarm_match = re.search(
            r"(?:alarm\s*[(:]?\s*(\d{3,8})\b|error\s*code\s*[:#]?\s*(\w+))",
            text,
            re.IGNORECASE,
        )
        alarm_code = ""
        if alarm_match:
            alarm_code = (alarm_match.group(1) or alarm_match.group(2) or "").strip()

        page_no = page.get("page", page_idx + 1)
        for section_title, segment in page_segments:
            content = segment.strip()
            if not content:
                continue

            extra_meta = {
                "source_file": source_file,
                "source_type": data.get("source_type", ""),
                "vlm_model": data.get("vlm_model", ""),
                "meta_source": meta_source,
                "work_type": work_type,
                "module": module,
                "topic": topic,
                "source_doc_type": raw_doc_type,
            }
            if section_title:
                extra_meta["section_title"] = section_title
            if canonical_doc_type == "ts" and alarm_code:
                extra_meta["alarm_code"] = alarm_code
            if canonical_doc_type == "setup":
                extra_meta["sequence_no"] = setup_sequence_no
                setup_sequence_no += 1

            chunk_id = generate_chunk_id(canonical_doc_type, doc_id, next_chunk_index)
            next_chunk_index += 1
            chunks.append(
                ChunkV3Document(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page=page_no,
                    lang=lang,
                    content=content,
                    search_text=content,
                    doc_type=canonical_doc_type,
                    device_name=device_name,
                    equip_id="",
                    chapter=chapter,
                    content_hash=compute_content_hash(content),
                    extra_meta=extra_meta,
                )
            )

    return chunks


# =============================================================================
# MyService chunker
# =============================================================================


def _parse_myservice_txt(txt_path: str | Path) -> dict[str, Any] | None:
    """MyService TXT 파일 파싱.

    [meta] JSON 블록과 [status/action/cause/result] 섹션을 추출한다.

    Returns:
        None if empty completeness, otherwise dict with meta and sections.
    """
    content = Path(txt_path).read_text(encoding="utf-8")
    lines = content.split("\n")

    meta: dict[str, Any] = {}
    sections: dict[str, str] = {"status": "", "action": "", "cause": "", "result": ""}

    current_section: str | None = None
    meta_lines: list[str] = []
    in_meta = False

    for line in lines:
        stripped = line.strip()

        if stripped == "[meta]":
            in_meta = True
            current_section = None
            continue
        elif stripped in ("[status]", "[action]", "[cause]", "[result]"):
            in_meta = False
            current_section = stripped[1:-1]
            continue

        if in_meta:
            meta_lines.append(line)
        elif current_section:
            sections[current_section] += line + "\n"

    # Parse meta JSON
    meta_text = "\n".join(meta_lines).strip()
    if meta_text:
        if not meta_text.startswith("{"):
            meta_text = "{" + meta_text
        if not meta_text.endswith("}"):
            meta_text = meta_text + "}"
        try:
            meta = json.loads(meta_text)
        except json.JSONDecodeError:
            meta = {}

    # Skip empty documents
    completeness = meta.get("completeness", "")
    if completeness == "empty":
        return None

    for key in sections:
        sections[key] = sections[key].strip()

    return {"meta": meta, "sections": sections}


def chunk_myservice(txt_path: str | Path) -> list[ChunkV3Document]:
    """MyService TXT -> 4섹션 통합 chunk.

    1. [meta] JSON 파싱 -> 메타데이터
    2. completeness == "empty" -> skip
    3. [status/action/cause/result] 통합
    4. search_text = title + cause + status + action + result

    Args:
        txt_path: MyService TXT 파일 경로

    Returns:
        ChunkV3Document 리스트 (0 or 1개)
    """
    parsed = _parse_myservice_txt(txt_path)
    if parsed is None:
        return []

    meta = parsed["meta"]
    sections = parsed["sections"]

    sections_present = {
        key: bool((sections.get(key) or "").strip())
        for key in ("status", "action", "cause", "result")
    }
    if not any(sections_present.values()):
        return []

    title = str(meta.get("Title", "") or "").strip()
    cause_text = str(sections.get("cause", "") or "").strip()
    status_text = str(sections.get("status", "") or "").strip()
    doc_id = Path(txt_path).stem

    max_tokens_by_section = {
        "status": 450,
        "action": 700,
        "cause": 250,
        "result": 250,
    }
    overlap_by_section = {
        "status": 50,
        "action": 80,
        "cause": 30,
        "result": 30,
    }

    out: list[ChunkV3Document] = []
    chunk_index = 0
    for section in ("status", "action", "cause", "result"):
        section_text = str(sections.get(section, "") or "").strip()
        if not section_text:
            continue
        parts = _word_window_split(
            section_text,
            max_tokens=max_tokens_by_section[section],
            overlap_tokens=overlap_by_section[section],
        ) or [section_text]

        for part in parts:
            content = f"[{section}] {part.strip()}".strip()
            if not content:
                continue
            search_text = f"{title} {cause_text} {status_text} {part.strip()}".strip()
            chunk_id = generate_chunk_id("myservice", doc_id, chunk_index)
            chunk_index += 1
            out.append(
                ChunkV3Document(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page=0,
                    lang="ko",
                    content=content,
                    search_text=search_text,
                    doc_type="myservice",
                    device_name=str(meta.get("Model Name", "") or ""),
                    equip_id=str(meta.get("Equip_ID", "") or ""),
                    chapter=section,
                    content_hash=compute_content_hash(content),
                    extra_meta={
                        "order_no": meta.get("Order No.", ""),
                        "activity_type": meta.get("Activity Type", ""),
                        "country": meta.get("Country", ""),
                        "reception_date": meta.get("Reception Date", ""),
                        "completeness": meta.get("completeness", ""),
                        "section": section,
                        "sections_present": sections_present,
                    },
                )
            )

    return out


# =============================================================================
# GCB chunker
# =============================================================================


def _fixed_size_split(text: str, size: int = 512, overlap: int = 50) -> list[str]:
    """고정 크기 텍스트 분할.

    Args:
        text: 분할할 텍스트
        size: chunk 크기 (문자 수)
        overlap: 겹침 크기

    Returns:
        분할된 텍스트 리스트
    """
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    pos = 0
    while pos < len(text):
        end = min(pos + size, len(text))
        chunks.append(text[pos:end])
        if end >= len(text):
            break
        pos = end - overlap
        if pos <= (end - size):
            pos = end

    return chunks


def _split_gcb_sections(content: str) -> list[tuple[str, str]]:
    header_re = re.compile(
        r"^(description|cause|result|background|request|설명|원인|결과|요청|배경)\s*[:：]",
        re.IGNORECASE,
    )
    date_re = re.compile(r"\b20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b")

    sections: list[tuple[str, str]] = []
    current_name = "detail"
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        merged = "\n".join(current_lines).strip()
        if merged:
            sections.append((current_name, merged))
        current_lines = []

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            current_lines.append(line)
            continue
        header = header_re.match(stripped)
        if header:
            flush()
            current_name = header.group(1).lower()
            continue
        if date_re.search(stripped) and len(" ".join(current_lines).split()) > 220:
            flush()
            current_name = "timeline"
        current_lines.append(line)

    flush()
    if not sections and content.strip():
        return [("detail", content.strip())]
    return sections


def chunk_gcb(
    json_path: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_content_len: int = 10,
) -> list[ChunkV3Document]:
    """GCB raw JSON -> Title prefix + fixed-size chunks.

    1. Content < min_content_len -> skip
    2. prefix = f"[GCB {num}] {title}\\nModel: {model}\\n\\n"
    3. len <= chunk_size -> single chunk
    4. len > chunk_size -> fixed-size split

    Args:
        json_path: GCB raw JSON 파일 경로
        chunk_size: chunk 크기 (문자 수)
        chunk_overlap: 겹침 크기
        min_content_len: 최소 Content 길이 (미만은 스킵)

    Returns:
        ChunkV3Document 리스트
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_chunks: list[ChunkV3Document] = []

    for entry in data:
        content_raw = str(entry.get("Content", "") or "").strip()
        if len(content_raw) < min_content_len:
            continue

        gcb_number = str(entry.get("GCB_number", "") or "")
        status = str(entry.get("Status", "") or "")
        title = str(entry.get("Title", "") or "")
        model_name = str(entry.get("Model Name", "") or "")
        equip_id = str(entry.get("Equip_ID", "") or "")
        request_type = str(entry.get("Request_Item2", "") or "")

        summary = (
            f"[GCB {gcb_number}] {title}\n"
            f"Request: {request_type}\n"
            f"Status: {status}\n"
            f"Model: {model_name}"
        ).strip()

        detail_sections = _split_gcb_sections(content_raw)
        detail_chunks: list[tuple[str, str]] = []
        for section_name, section_text in detail_sections:
            parts = _word_window_split(section_text, 700, 60) or [section_text]
            for part in parts:
                if part.strip():
                    detail_chunks.append((section_name, part.strip()))

        chunk_of = 1 + len(detail_chunks)
        doc_id = f"gcb_{gcb_number}"
        next_idx = 0

        summary_id = generate_chunk_id("gcb", doc_id, next_idx)
        next_idx += 1
        all_chunks.append(
            ChunkV3Document(
                chunk_id=summary_id,
                doc_id=doc_id,
                page=0,
                lang="ko",
                content=summary,
                search_text=summary,
                doc_type="gcb",
                device_name=model_name,
                equip_id=equip_id,
                chapter="summary",
                content_hash=compute_content_hash(summary),
                extra_meta={
                    "gcb_number": gcb_number,
                    "request_type": request_type,
                    "status": status,
                    "chunk_of": chunk_of,
                    "chunk_tier": "summary",
                },
            )
        )

        for section_name, detail in detail_chunks:
            chunk_id = generate_chunk_id("gcb", doc_id, next_idx)
            next_idx += 1
            all_chunks.append(
                ChunkV3Document(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page=0,
                    lang="ko",
                    content=detail,
                    search_text=detail,
                    doc_type="gcb",
                    device_name=model_name,
                    equip_id=equip_id,
                    chapter=section_name,
                    content_hash=compute_content_hash(detail),
                    extra_meta={
                        "gcb_number": gcb_number,
                        "request_type": request_type,
                        "status": status,
                        "chunk_of": chunk_of,
                        "chunk_tier": "detail",
                        "section": section_name,
                    },
                )
            )

    return all_chunks


__all__ = [
    "chunk_gcb",
    "chunk_myservice",
    "chunk_vlm_parsed",
]
