"""
myservice Equip_ID ES 동기화 검증 테스트

myservice_psk.csv의 전체 데이터를 기준으로:
1. Order No. = ES doc_id 로 매칭
2. ES에 해당 doc_id가 있는지 확인
3. 있다면 equip_id가 CSV의 Equip_ID와 일치하는지 검증
"""

import csv
import json
import os
from collections import defaultdict

import pytest
import requests

ES_URL = os.getenv("ES_URL", "http://localhost:8002")
ES_INDEX = os.getenv("ES_INDEX", "rag_chunks_dev_v2")

MYSERVICE_CSV_PATH = (
    "/home/llm-share/datasets/pe_agent_data"
    "/pe_preprocess_data/myservice_psk.csv"
)

GARBAGE_EQUIP_IDS = {"", "-", ".", "/", "1", "NA", "N/A", "na", "n/a"}


# ── helpers ──────────────────────────────────────────────────────────


def _is_valid_equip_id(eid: str | None) -> bool:
    return bool(eid and eid.strip() not in GARBAGE_EQUIP_IDS)


def _load_csv_mapping() -> dict[str, str | None]:
    """CSV에서 Order No. → Equip_ID 매핑 로드. 쓰레기 값은 None 처리."""
    mapping: dict[str, str | None] = {}
    with open(MYSERVICE_CSV_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            order_no = row.get("Order No.", "").strip()
            eid = row.get("Equip_ID", "").strip()
            if not order_no:
                continue
            mapping[order_no] = eid if _is_valid_equip_id(eid) else None
    return mapping


def _fetch_es_equip_ids() -> dict[str, str | None]:
    """ES에서 myservice 전체 doc_id → equip_id 매핑 조회 (scroll)."""
    result: dict[str, str | None] = {}
    resp = requests.post(
        f"{ES_URL}/{ES_INDEX}/_search?scroll=2m",
        json={
            "size": 1000,
            "query": {"term": {"doc_type": "myservice"}},
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
def csv_mapping() -> dict[str, str | None]:
    return _load_csv_mapping()


@pytest.fixture(scope="module")
def es_equip_ids() -> dict[str, str | None]:
    return _fetch_es_equip_ids()


# ── tests ────────────────────────────────────────────────────────────


class TestMyserviceEquipIdSync:
    """ES에 저장된 myservice equip_id가 CSV 원본과 일치하는지 전수 검증."""

    def test_es_myservice_documents_exist(self, es_equip_ids: dict):
        """ES에 myservice 문서가 존재하는지 확인."""
        assert len(es_equip_ids) > 0, "ES에 myservice 문서가 없습니다"

    def test_csv_has_data(self, csv_mapping: dict):
        """CSV에서 데이터를 로드했는지 확인."""
        assert len(csv_mapping) > 90000, (
            f"CSV 데이터가 너무 적음: {len(csv_mapping)}"
        )

    def test_equip_id_consistency(
        self,
        csv_mapping: dict[str, str | None],
        es_equip_ids: dict[str, str | None],
    ):
        """전체 myservice doc_id에 대해 ES equip_id가 CSV와 일치하는지 검증."""
        mismatches: list[dict] = []
        checked = 0
        skipped_not_in_csv = 0

        for doc_id, actual_equip_id in es_equip_ids.items():
            if doc_id not in csv_mapping:
                skipped_not_in_csv += 1
                continue

            checked += 1
            expected = csv_mapping[doc_id]

            if actual_equip_id != expected:
                mismatches.append({
                    "doc_id": doc_id,
                    "expected": expected,
                    "actual": actual_equip_id,
                })

        summary = (
            f"checked={checked}, "
            f"skipped(not_in_csv)={skipped_not_in_csv}, "
            f"mismatches={len(mismatches)}"
        )
        assert not mismatches, (
            f"equip_id 불일치 {len(mismatches)}건\n{summary}\n"
            f"{json.dumps(mismatches[:20], indent=2, ensure_ascii=False)}"
        )

    def test_valid_equip_id_not_empty_in_es(
        self,
        csv_mapping: dict[str, str | None],
        es_equip_ids: dict[str, str | None],
    ):
        """CSV에 유효 Equip_ID가 있는 문서가 ES에서 누락되지 않았는지 확인."""
        missing: list[str] = []

        for doc_id, expected in csv_mapping.items():
            if expected is None:
                continue
            if doc_id not in es_equip_ids:
                continue
            if es_equip_ids[doc_id] is None:
                missing.append(doc_id)

        assert not missing, (
            f"유효 equip_id가 ES에 누락된 문서 {len(missing)}건: "
            f"{missing[:20]}"
        )

    def test_null_equip_id_stays_null_in_es(
        self,
        csv_mapping: dict[str, str | None],
        es_equip_ids: dict[str, str | None],
    ):
        """CSV에서 Equip_ID가 없는 문서가 ES에서도 null인지 확인."""
        unexpected: list[dict] = []

        for doc_id, expected in csv_mapping.items():
            if expected is not None:
                continue
            if doc_id not in es_equip_ids:
                continue
            actual = es_equip_ids[doc_id]
            if actual is not None:
                unexpected.append({"doc_id": doc_id, "actual": actual})

        assert not unexpected, (
            f"equip_id=null이어야 하는데 값이 있는 문서 {len(unexpected)}건: "
            f"{unexpected[:20]}"
        )

    def test_coverage(
        self,
        csv_mapping: dict[str, str | None],
        es_equip_ids: dict[str, str | None],
    ):
        """매칭 커버리지가 99% 이상인지 확인 (regression guard)."""
        es_doc_ids = set(es_equip_ids.keys())
        csv_doc_ids = set(csv_mapping.keys())
        matched = es_doc_ids & csv_doc_ids
        coverage = len(matched) / len(es_doc_ids) * 100

        with_equip = sum(
            1 for did in matched
            if csv_mapping[did] is not None
        )

        print(f"\n=== myservice equip_id 커버리지 ===")
        print(f"  ES 고유 doc_id:   {len(es_doc_ids)}")
        print(f"  CSV Order No.:    {len(csv_doc_ids)}")
        print(f"  매칭:             {len(matched)} ({coverage:.1f}%)")
        print(f"  유효 equip_id:    {with_equip}")
        print(f"  미매칭:           {len(es_doc_ids - csv_doc_ids)}")

        assert coverage > 99.0, f"커버리지가 너무 낮음: {coverage:.1f}%"
