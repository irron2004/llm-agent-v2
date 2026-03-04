"""5종 문서별 chunker 구현.

- chunk_vlm_parsed: SOP, TS, Setup Manual (VLM JSON -> ChunkV3Document)
- chunk_myservice: MyService TXT -> 4섹션 통합 chunk
- chunk_gcb: GCB raw JSON -> Title prefix + fixed-size chunking
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .common import (
    ChunkV3Document,
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
    "sop_pdf": _parse_sop_filename,
    "sop_pptx": _parse_sop_filename,
    "sop": _parse_sop_filename,
    "ts": _parse_ts_filename,
    "setup_manual": _parse_setup_manual_filename,
}


# =============================================================================
# VLM parsed 문서 chunker (SOP, TS, Setup Manual)
# =============================================================================

def chunk_vlm_parsed(
    doc_type: str,
    vlm_json_path: str | Path,
    lang: str = "ko",
) -> list[ChunkV3Document]:
    """VLM JSON -> ChunkV3Document[] (SOP, TS, Setup Manual 공용).

    페이지 = chunk 단위. 메타데이터는 파일명에서 파싱.

    Args:
        doc_type: 문서 종류 (sop_pdf, sop_pptx, ts, setup_manual)
        vlm_json_path: VLM 파싱 결과 JSON 경로
        lang: 언어 코드

    Returns:
        ChunkV3Document 리스트
    """
    data = load_vlm_result(vlm_json_path)
    doc_id = data["doc_id"]
    source_file = data.get("source_file", "")

    parser_fn = FILENAME_PARSERS.get(doc_type, lambda f: {"device_name": "", "chapter": ""})
    meta = parser_fn(source_file)

    chunks: list[ChunkV3Document] = []
    for idx, page in enumerate(data.get("pages", [])):
        text = page.get("text", "").strip()
        if not text:
            continue

        chunk_id = generate_chunk_id(doc_type, doc_id, idx)
        chunks.append(
            ChunkV3Document(
                chunk_id=chunk_id,
                doc_id=doc_id,
                page=page.get("page", idx + 1),
                lang=lang,
                content=text,
                search_text=text,
                doc_type=doc_type.split("_")[0] if "_" in doc_type else doc_type,
                device_name=meta.get("device_name", ""),
                equip_id="",
                chapter=meta.get("chapter", ""),
                content_hash=compute_content_hash(text),
                extra_meta={
                    "source_file": source_file,
                    "source_type": data.get("source_type", ""),
                    "vlm_model": data.get("vlm_model", ""),
                },
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

    # 4섹션 통합 content
    section_parts: list[str] = []
    for key in ("status", "action", "cause", "result"):
        text = sections.get(key, "").strip()
        if text:
            section_parts.append(f"[{key}] {text}")

    content = "\n".join(section_parts)
    if not content.strip():
        return []

    # search_text: title + cause + status를 앞에 배치 (BM25 가중)
    title = meta.get("Title", "")
    cause = sections.get("cause", "")
    status = sections.get("status", "")
    action = sections.get("action", "")
    result = sections.get("result", "")
    search_text = f"{title} {cause} {status} {action} {result}".strip()

    doc_id = Path(txt_path).stem
    chunk_id = generate_chunk_id("myservice", doc_id, 0)

    return [
        ChunkV3Document(
            chunk_id=chunk_id,
            doc_id=doc_id,
            page=0,
            lang="ko",
            content=content,
            search_text=search_text,
            doc_type="myservice",
            device_name=meta.get("Model Name", ""),
            equip_id=meta.get("Equip_ID", ""),
            chapter="",
            content_hash=compute_content_hash(content),
            extra_meta={
                "order_no": meta.get("Order No.", ""),
                "activity_type": meta.get("Activity Type", ""),
                "country": meta.get("Country", ""),
                "completeness": meta.get("completeness", ""),
            },
        )
    ]


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
        content_raw = entry.get("Content", "").strip()
        if len(content_raw) < min_content_len:
            continue

        gcb_number = str(entry.get("GCB_number", ""))
        title = entry.get("Title", "")
        model_name = entry.get("Model Name", "")
        equip_id = entry.get("Equip_ID", "")

        prefix = f"[GCB {gcb_number}] {title}\nModel: {model_name}\n\n"
        full_text = prefix + content_raw

        text_chunks = _fixed_size_split(full_text, size=chunk_size, overlap=chunk_overlap)

        doc_id = f"gcb_{gcb_number}"

        for idx, chunk_text in enumerate(text_chunks):
            chunk_id = generate_chunk_id("gcb", doc_id, idx)
            all_chunks.append(
                ChunkV3Document(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page=0,
                    lang="ko",
                    content=chunk_text,
                    search_text=chunk_text,
                    doc_type="gcb",
                    device_name=model_name,
                    equip_id=equip_id,
                    chapter="",
                    content_hash=compute_content_hash(chunk_text),
                    extra_meta={
                        "gcb_number": gcb_number,
                        "request_type": entry.get("Request_Item2", ""),
                        "status": entry.get("Status", ""),
                        "chunk_of": len(text_chunks),
                    },
                )
            )

    return all_chunks


__all__ = [
    "chunk_gcb",
    "chunk_myservice",
    "chunk_vlm_parsed",
]
