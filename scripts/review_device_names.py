#!/usr/bin/env python3
"""
Device name 검토 및 수정 스크립트

device_catalog.json의 device_name들을 Elasticsearch에서 조회하고,
잘못된 이름을 수정하거나 삭제할 수 있습니다.

Usage:
    python scripts/review_device_names.py
    python scripts/review_device_names.py --start-from "SUPRA N"  # 특정 장비부터 시작
    python scripts/review_device_names.py --min-docs 5  # doc_count가 5 이하인 것만
"""

import json
import sys
from pathlib import Path
from elasticsearch import Elasticsearch

# Configuration
ES_HOST = "http://localhost:8002"
ES_INDEX = "rag_chunks_dev_v2"
CATALOG_PATH = Path(__file__).parent.parent / "data" / "device_catalog.json"


def load_catalog():
    """device_catalog.json 로드"""
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_catalog(data):
    """device_catalog.json 저장"""
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_sample_docs(es: Elasticsearch, device_name: str, size: int = 3):
    """특정 device_name의 샘플 문서 조회"""
    result = es.search(
        index=ES_INDEX,
        body={
            "size": size,
            "query": {"term": {"device_name": device_name}},
            "_source": ["doc_id", "device_name", "doc_type", "title", "content"],
        },
    )
    return result["hits"]["hits"]


def get_doc_count(es: Elasticsearch, device_name: str) -> int:
    """특정 device_name의 문서 수 조회"""
    result = es.count(
        index=ES_INDEX,
        body={"query": {"term": {"device_name": device_name}}},
    )
    return result["count"]


def update_device_name(es: Elasticsearch, old_name: str, new_name: str) -> int:
    """ES에서 device_name 일괄 변경"""
    result = es.update_by_query(
        index=ES_INDEX,
        body={
            "query": {"term": {"device_name": old_name}},
            "script": {
                "source": "ctx._source.device_name = params.new_name",
                "lang": "painless",
                "params": {"new_name": new_name},
            },
        },
        refresh=True,
    )
    return result["updated"]


def delete_device_docs(es: Elasticsearch, device_name: str) -> int:
    """ES에서 특정 device_name의 문서 삭제"""
    result = es.delete_by_query(
        index=ES_INDEX,
        body={"query": {"term": {"device_name": device_name}}},
        refresh=True,
    )
    return result["deleted"]


def print_sample_docs(docs):
    """샘플 문서 출력"""
    if not docs:
        print("  (문서 없음)")
        return

    for i, hit in enumerate(docs, 1):
        src = hit["_source"]
        print(f"\n  [{i}] doc_id: {src.get('doc_id', 'N/A')}")
        print(f"      doc_type: {src.get('doc_type', 'N/A')}")
        title = src.get("title", "")
        if title:
            print(f"      title: {title[:100]}...")
        content = src.get("content", "")
        if content:
            print(f"      content: {content[:200]}...")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Device name 검토 및 수정")
    parser.add_argument("--start-from", help="특정 장비 이름부터 시작")
    parser.add_argument("--min-docs", type=int, help="doc_count가 이 값 이하인 것만 검토")
    parser.add_argument("--max-docs", type=int, help="doc_count가 이 값 이하인 것만 검토")
    args = parser.parse_args()

    # ES 연결
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        print("ERROR: Elasticsearch에 연결할 수 없습니다.")
        sys.exit(1)

    # Catalog 로드
    catalog = load_catalog()
    devices = catalog.get("devices", [])

    print(f"총 {len(devices)}개 장비 검토 시작\n")
    print("=" * 60)

    # 시작 위치 찾기
    start_idx = 0
    if args.start_from:
        for i, d in enumerate(devices):
            if d.get("name") == args.start_from:
                start_idx = i
                break

    reviewed = 0
    modified = 0
    deleted_count = 0

    for i, device in enumerate(devices[start_idx:], start_idx + 1):
        name = device.get("name", "")
        catalog_doc_count = device.get("doc_count", 0)

        # 필터링
        if args.min_docs and catalog_doc_count > args.min_docs:
            continue
        if args.max_docs and catalog_doc_count > args.max_docs:
            continue

        # ES에서 실제 문서 수 확인
        actual_count = get_doc_count(es, name)

        print(f"\n[{i}/{len(devices)}] 장비명: {name}")
        print(f"  catalog doc_count: {catalog_doc_count}, ES 실제: {actual_count}")

        if actual_count == 0:
            print("  → ES에 문서 없음 (catalog에서만 존재)")
        else:
            # 샘플 문서 조회
            samples = get_sample_docs(es, name)
            print_sample_docs(samples)

        print("\n  명령어:")
        print("    [Enter] 다음으로")
        print("    [r] 이름 변경 (rename)")
        print("    [d] 문서 삭제 (delete)")
        print("    [s] 건너뛰기 (skip to next)")
        print("    [q] 종료")

        while True:
            cmd = input("\n  > ").strip().lower()

            if cmd == "" or cmd == "s":
                break

            elif cmd == "r":
                new_name = input("  새 이름 입력: ").strip()
                if not new_name:
                    print("  취소됨")
                    continue

                confirm = input(f"  '{name}' → '{new_name}'로 변경? (y/n): ").strip().lower()
                if confirm == "y":
                    updated = update_device_name(es, name, new_name)
                    print(f"  ✓ {updated}개 문서 업데이트됨")
                    modified += 1
                    break
                else:
                    print("  취소됨")

            elif cmd == "d":
                confirm = input(f"  '{name}' 문서 {actual_count}개 삭제? (y/n): ").strip().lower()
                if confirm == "y":
                    deleted = delete_device_docs(es, name)
                    print(f"  ✓ {deleted}개 문서 삭제됨")
                    deleted_count += 1
                    break
                else:
                    print("  취소됨")

            elif cmd == "q":
                print(f"\n종료. 검토: {reviewed}, 수정: {modified}, 삭제: {deleted_count}")
                sys.exit(0)

            else:
                print("  알 수 없는 명령어")

        reviewed += 1
        print("-" * 60)

    print(f"\n완료! 검토: {reviewed}, 수정: {modified}, 삭제: {deleted_count}")
    print("\ndevice_catalog.json을 갱신하려면:")
    print("  python -c \"from backend.services.device_cache import DeviceCache; DeviceCache().refresh(...)\"")
    print("  또는 서버 재시작 후 /api/devices/refresh 호출")


if __name__ == "__main__":
    main()
