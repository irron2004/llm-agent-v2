#!/usr/bin/env python3
"""
Doc type 검토 및 수정 스크립트

Usage:
    python scripts/review_doc_types.py
    python scripts/review_doc_types.py --max-docs 100  # doc_count가 100 이하인 것만
"""

import json
import sys
from pathlib import Path
from elasticsearch import Elasticsearch

ES_HOST = "http://localhost:8002"
ES_INDEX = "rag_chunks_dev_v2"
CATALOG_PATH = Path(__file__).parent.parent / "data" / "device_catalog.json"


def load_catalog():
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_sample_docs(es, doc_type, size=3):
    result = es.search(
        index=ES_INDEX,
        body={
            "size": size,
            "query": {"term": {"doc_type": doc_type}},
            "_source": ["doc_id", "device_name", "doc_type", "title", "content"],
        },
    )
    return result["hits"]["hits"]


def get_doc_count(es, doc_type):
    result = es.count(
        index=ES_INDEX,
        body={"query": {"term": {"doc_type": doc_type}}},
    )
    return result["count"]


def update_doc_type(es, old_type, new_type):
    result = es.update_by_query(
        index=ES_INDEX,
        body={
            "query": {"term": {"doc_type": old_type}},
            "script": {
                "source": "ctx._source.doc_type = params.new_type",
                "lang": "painless",
                "params": {"new_type": new_type},
            },
        },
        conflicts="proceed",
        scroll_size=500,
        refresh=True,
    )
    return result["updated"]


def print_sample_docs(docs):
    if not docs:
        print("  (문서 없음)")
        return
    for i, hit in enumerate(docs, 1):
        src = hit["_source"]
        print(f"\n  [{i}] doc_id: {src.get('doc_id', 'N/A')}")
        print(f"      device_name: {src.get('device_name', 'N/A')}")
        title = src.get("title", "")
        if title:
            print(f"      title: {title[:80]}")
        content = src.get("content", "")
        if content:
            print(f"      content: {content[:150]}...")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from", help="특정 doc_type부터 시작")
    parser.add_argument("--max-docs", type=int, help="doc_count가 이 값 이하인 것만")
    args = parser.parse_args()

    es = Elasticsearch(ES_HOST)
    if not es.ping():
        print("ERROR: ES 연결 실패")
        sys.exit(1)

    catalog = load_catalog()
    doc_types = catalog.get("doc_types", [])
    print(f"총 {len(doc_types)}개 doc_type\n" + "=" * 60)

    start_idx = 0
    if args.start_from:
        for i, d in enumerate(doc_types):
            if d.get("name") == args.start_from:
                start_idx = i
                break

    modified = 0

    for i, dt in enumerate(doc_types[start_idx:], start_idx + 1):
        name = dt.get("name", "")
        catalog_count = dt.get("doc_count", 0)

        if args.max_docs and catalog_count > args.max_docs:
            continue

        actual_count = get_doc_count(es, name)
        print(f"\n[{i}/{len(doc_types)}] doc_type: {name}")
        print(f"  catalog: {catalog_count}, ES 실제: {actual_count}")

        if actual_count > 0:
            print_sample_docs(get_sample_docs(es, name))

        print("\n  [Enter] 다음 | [r] 이름변경 | [q] 종료")

        while True:
            cmd = input("  > ").strip().lower()
            if cmd == "" or cmd == "s":
                break
            elif cmd == "r":
                new_type = input("  새 doc_type: ").strip()
                if new_type:
                    confirm = input(f"  '{name}' → '{new_type}'? (y/n): ").strip().lower()
                    if confirm == "y":
                        updated = update_doc_type(es, name, new_type)
                        print(f"  ✓ {updated}개 업데이트")
                        modified += 1
                        break
            elif cmd == "q":
                print(f"\n종료. 수정: {modified}")
                sys.exit(0)

        print("-" * 60)

    print(f"\n완료! 수정: {modified}")


if __name__ == "__main__":
    main()
