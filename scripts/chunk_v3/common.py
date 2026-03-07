"""chunk_v3 공통 데이터 모델과 유틸리티."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ChunkV3Document:
    """chunk_v3 통합 chunk 포맷.

    5종 문서(SOP, TS, Setup Manual, MyService, GCB)가
    모두 이 포맷으로 변환되어 Phase 3/4에서 통합 처리된다.
    """

    chunk_id: str  # {doc_type}_{doc_id}#{chunk_index:04d}
    doc_id: str
    page: int
    lang: str
    content: str
    search_text: str
    doc_type: str  # sop, ts, setup, myservice, gcb
    device_name: str
    equip_id: str
    chapter: str
    content_hash: str  # SHA256[:16]
    chunk_version: str = "v3"
    pipeline_version: str = "1.0.0"
    extra_meta: dict[str, Any] = field(default_factory=dict)
    section_chapter: str = ""
    section_number: int = -1
    chapter_source: str = "none"
    chapter_ok: bool = False


DOC_TYPE_ALIASES: dict[str, str] = {
    "sop": "sop",
    "ts": "ts",
    "trouble_shooting": "ts",
    "trouble_shooting_guide": "ts",
    "troubleshooting": "ts",
    "t/s": "ts",
    "setup": "setup",
    "setup_manual": "setup",
    "set_up_manual": "setup",
    "installation manual": "setup",
    "myservice": "myservice",
    "gcb": "gcb",
}


def canonicalize_doc_type(doc_type: str) -> str:
    raw = str(doc_type or "").strip().lower()
    return DOC_TYPE_ALIASES.get(raw, raw)


def generate_chunk_id(doc_type: str, doc_id: str, index: int) -> str:
    """chunk_id 생성.

    Args:
        doc_type: 문서 종류 (sop, ts, setup, myservice, gcb)
        doc_id: 문서 ID
        index: chunk 인덱스 (0-based)

    Returns:
        chunk_id (e.g., "sop_doc1#0000")
    """
    return f"{canonicalize_doc_type(doc_type)}_{doc_id}#{index:04d}"


def compute_content_hash(text: str) -> str:
    """텍스트의 SHA256 해시 앞 16자리 반환.

    Args:
        text: 해시할 텍스트

    Returns:
        SHA256[:16] hex string
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def save_vlm_result(
    doc_type: str,
    doc_id: str,
    pages: list[dict[str, Any]],
    source_file: str,
    vlm_model: str,
    output_dir: str = "data/vlm_parsed",
    source_type: str = "pdf",
) -> Path:
    """VLM 파싱 결과를 JSON으로 저장.

    Args:
        doc_type: 문서 종류
        doc_id: 문서 ID
        pages: [{"page": 1, "text": "...", "confidence": null}, ...]
        source_file: 원본 파일명
        vlm_model: 사용된 VLM 모델명
        output_dir: 출력 기본 경로
        source_type: 소스 파일 형식 ("pdf" or "pptx")

    Returns:
        저장된 JSON 파일 경로
    """
    out_dir = Path(output_dir) / doc_type
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "doc_id": doc_id,
        "source_file": source_file,
        "source_type": source_type,
        "total_pages": len(pages),
        "vlm_model": vlm_model,
        "parsed_at": datetime.now(timezone.utc).isoformat(),
        "pages": pages,
    }

    out_path = out_dir / f"{doc_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return out_path


def load_vlm_result(path: str | Path) -> dict[str, Any]:
    """저장된 VLM 파싱 결과 JSON 로드.

    Args:
        path: JSON 파일 경로

    Returns:
        파싱 결과 딕셔너리
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_chunks_jsonl(path: str | Path, chunks: list[ChunkV3Document]) -> None:
    """ChunkV3Document 리스트를 JSONL로 저장.

    Args:
        path: 출력 JSONL 파일 경로
        chunks: 저장할 chunk 리스트
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            line = json.dumps(asdict(chunk), ensure_ascii=False)
            f.write(line + "\n")


def load_chunks_jsonl(path: str | Path) -> list[ChunkV3Document]:
    """JSONL에서 ChunkV3Document 리스트 로드.

    Args:
        path: JSONL 파일 경로

    Returns:
        ChunkV3Document 리스트
    """
    chunks: list[ChunkV3Document] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            chunks.append(ChunkV3Document(**data))
    return chunks


__all__ = [
    "ChunkV3Document",
    "DOC_TYPE_ALIASES",
    "canonicalize_doc_type",
    "compute_content_hash",
    "generate_chunk_id",
    "load_chunks_jsonl",
    "load_vlm_result",
    "save_chunks_jsonl",
    "save_vlm_result",
]
