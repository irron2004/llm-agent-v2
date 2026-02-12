"""
GCB Equip_ID ES 동기화 검증 테스트

scraped_gcb.json의 전체 데이터를 기준으로:
1. gcb_number + title로 GCB 텍스트 파일 매칭
2. ES에 해당 doc_id가 있는지 확인
3. 있다면 equip_id가 기대값과 일치하는지 검증
"""

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass

import pytest
import requests

ES_URL = os.getenv("ES_URL", "http://localhost:8002")
ES_INDEX = os.getenv("ES_INDEX", "rag_chunks_dev_v2")

SCRAPED_GCB_PATH = (
    "/home/llm-share/datasets/pe_agent_data"
    "/pe_preprocess_data/gcb_raw/20260126/scraped_gcb.json"
)
GCB_TXT_DIR = (
    "/home/llm-share/datasets/pe_agent_data"
    "/pe_preprocess_data/gcb/"
)

GARBAGE_EQUIP_IDS = {"", "-", ".", "/", "1", "NA", "N/A", "na", "n/a"}


# ── helpers ──────────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _extract_title_from_file(filepath: str) -> str | None:
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("**Title**:"):
                return line.strip().replace("**Title**:", "").strip()
    return None


def _is_valid_equip_id(eid: str | None) -> bool:
    return bool(eid and eid.strip() not in GARBAGE_EQUIP_IDS)


# ── data loading (module-level cache) ────────────────────────────────


@dataclass
class MatchResult:
    doc_id: str
    gcb_number: str
    file_title: str | None
    json_title: str | None
    expected_equip_id: str | None
    category: str  # single_valid | single_null | dup_matched | not_in_json


def _build_expected_mapping() -> list[MatchResult]:
    """GCB 텍스트 파일 × scraped_gcb.json 매칭 → 기대 equip_id 목록."""
    with open(SCRAPED_GCB_PATH, encoding="utf-8") as f:
        raw_data = json.load(f)

    json_groups: dict[str, list[dict]] = defaultdict(list)
    for entry in raw_data:
        json_groups[entry.get("GCB_number")].append(entry)

    results: list[MatchResult] = []

    for fname in sorted(os.listdir(GCB_TXT_DIR)):
        m = re.match(r"GCB_(\d+)\.txt", fname)
        if not m:
            continue
        gcb_num = m.group(1)
        doc_id = f"GCB_{gcb_num}"
        filepath = os.path.join(GCB_TXT_DIR, fname)
        file_title = _extract_title_from_file(filepath)
        entries = json_groups.get(gcb_num, [])

        if not entries:
            results.append(MatchResult(
                doc_id=doc_id,
                gcb_number=gcb_num,
                file_title=file_title,
                json_title=None,
                expected_equip_id=None,
                category="not_in_json",
            ))
            continue

        if len(entries) == 1:
            entry = entries[0]
            eid = entry.get("Equip_ID")
            results.append(MatchResult(
                doc_id=doc_id,
                gcb_number=gcb_num,
                file_title=file_title,
                json_title=entry.get("Title"),
                expected_equip_id=eid.strip() if _is_valid_equip_id(eid) else None,
                category="single_valid" if _is_valid_equip_id(eid) else "single_null",
            ))
            continue

        # 중복 → Title 비교
        file_title_norm = _normalize(file_title or "")
        matched_entry = None
        for entry in entries:
            jt_norm = _normalize(entry.get("Title", ""))
            if (
                file_title_norm == jt_norm
                or file_title_norm in jt_norm
                or jt_norm in file_title_norm
            ):
                matched_entry = entry
                break

        if matched_entry:
            eid = matched_entry.get("Equip_ID")
            results.append(MatchResult(
                doc_id=doc_id,
                gcb_number=gcb_num,
                file_title=file_title,
                json_title=matched_entry.get("Title"),
                expected_equip_id=eid.strip() if _is_valid_equip_id(eid) else None,
                category="dup_matched",
            ))
        else:
            results.append(MatchResult(
                doc_id=doc_id,
                gcb_number=gcb_num,
                file_title=file_title,
                json_title=None,
                expected_equip_id=None,
                category="not_in_json",
            ))

    return results


def _fetch_es_equip_ids() -> dict[str, str | None]:
    """ES에서 GCB 전체 doc_id → equip_id 매핑 조회 (scroll)."""
    result: dict[str, str | None] = {}
    resp = requests.post(
        f"{ES_URL}/{ES_INDEX}/_search?scroll=2m",
        json={
            "size": 1000,
            "query": {"term": {"doc_type": "gcb"}},
            "_source": ["doc_id", "equip_id"],
        },
        timeout=30,
    ).json()

    scroll_id = resp["_scroll_id"]
    hits = resp["hits"]["hits"]

    while hits:
        for h in hits:
            doc_id = h["_source"]["doc_id"]
            equip_id = h["_source"].get("equip_id")
            if doc_id not in result:
                result[doc_id] = equip_id
        resp = requests.post(
            f"{ES_URL}/_search/scroll",
            json={"scroll": "2m", "scroll_id": scroll_id},
            timeout=30,
        ).json()
        hits = resp["hits"]["hits"]

    requests.delete(
        f"{ES_URL}/_search/scroll",
        json={"scroll_id": scroll_id},
        timeout=10,
    )
    return result


# ── fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def expected_mapping() -> list[MatchResult]:
    return _build_expected_mapping()


@pytest.fixture(scope="module")
def es_equip_ids() -> dict[str, str | None]:
    return _fetch_es_equip_ids()


# ── tests ────────────────────────────────────────────────────────────


class TestGcbEquipIdSync:
    """ES에 저장된 equip_id가 scraped_gcb.json 원본과 일치하는지 전수 검증."""

    def test_es_gcb_documents_exist(self, es_equip_ids: dict):
        """ES에 GCB 문서가 존재하는지 확인."""
        assert len(es_equip_ids) > 0, "ES에 GCB 문서가 없습니다"

    def test_equip_id_consistency(
        self,
        expected_mapping: list[MatchResult],
        es_equip_ids: dict[str, str | None],
    ):
        """전체 GCB 파일에 대해 ES equip_id가 기대값과 일치하는지 검증."""
        mismatches: list[dict] = []
        checked = 0
        skipped_not_in_es = 0

        for mr in expected_mapping:
            if mr.doc_id not in es_equip_ids:
                skipped_not_in_es += 1
                continue

            checked += 1
            actual = es_equip_ids[mr.doc_id]

            if actual != mr.expected_equip_id:
                mismatches.append({
                    "doc_id": mr.doc_id,
                    "category": mr.category,
                    "expected": mr.expected_equip_id,
                    "actual": actual,
                    "file_title": mr.file_title,
                    "json_title": mr.json_title,
                })

        summary = (
            f"checked={checked}, "
            f"skipped(not_in_es)={skipped_not_in_es}, "
            f"mismatches={len(mismatches)}"
        )
        if mismatches:
            detail = json.dumps(mismatches[:20], indent=2, ensure_ascii=False)
            pytest.fail(f"equip_id 불일치 {len(mismatches)}건\n{summary}\n{detail}")

    def test_valid_equip_id_not_empty_in_es(
        self,
        expected_mapping: list[MatchResult],
        es_equip_ids: dict[str, str | None],
    ):
        """유효 equip_id가 있어야 하는 문서가 ES에서 null이 아닌지 확인."""
        missing: list[str] = []

        for mr in expected_mapping:
            if mr.expected_equip_id is None:
                continue
            if mr.doc_id not in es_equip_ids:
                continue
            if es_equip_ids[mr.doc_id] is None:
                missing.append(mr.doc_id)

        assert not missing, (
            f"유효 equip_id가 ES에 누락된 문서 {len(missing)}건: "
            f"{missing[:20]}"
        )

    def test_null_equip_id_stays_null_in_es(
        self,
        expected_mapping: list[MatchResult],
        es_equip_ids: dict[str, str | None],
    ):
        """equip_id가 없어야 하는 문서가 ES에서도 null인지 확인."""
        unexpected: list[dict] = []

        for mr in expected_mapping:
            if mr.expected_equip_id is not None:
                continue
            if mr.doc_id not in es_equip_ids:
                continue
            actual = es_equip_ids[mr.doc_id]
            if actual is not None:
                unexpected.append({"doc_id": mr.doc_id, "actual": actual})

        assert not unexpected, (
            f"equip_id=null이어야 하는데 값이 있는 문서 {len(unexpected)}건: "
            f"{unexpected[:20]}"
        )

    def test_category_distribution(
        self,
        expected_mapping: list[MatchResult],
        es_equip_ids: dict[str, str | None],
    ):
        """카테고리별 건수가 합리적인지 확인 (regression guard)."""
        in_es = [mr for mr in expected_mapping if mr.doc_id in es_equip_ids]
        cats = defaultdict(int)
        for mr in in_es:
            cats[mr.category] += 1

        total = len(in_es)
        with_equip = sum(1 for mr in in_es if mr.expected_equip_id is not None)

        # 기본 sanity checks
        assert total > 3000, f"ES에 매칭된 GCB가 너무 적음: {total}"
        assert with_equip > 2000, f"유효 equip_id가 너무 적음: {with_equip}"
        assert cats.get("single_valid", 0) > 0
        assert cats.get("dup_matched", 0) > 0

        print(f"\n=== 카테고리별 분포 (ES 존재 기준) ===")
        print(f"  전체: {total}")
        for cat, cnt in sorted(cats.items()):
            print(f"  {cat}: {cnt}")
        print(f"  유효 equip_id 보유: {with_equip}")
