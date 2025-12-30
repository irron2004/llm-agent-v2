"""Query myservice documents from Elasticsearch.

Usage in Jupyter:
    %run query_myservice.py
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
ROOT = Path.cwd().parent if Path.cwd().name == "jupyter" else Path.cwd()
sys.path.insert(0, str(ROOT))

from elasticsearch import Elasticsearch
from backend.config.settings import search_settings

# ES 클라이언트 초기화
es = Elasticsearch(hosts=[search_settings.es_host])
index = f"{search_settings.es_index_prefix}_{search_settings.es_env}_current"

print(f"Connected to ES: {search_settings.es_host}")
print(f"Index: {index}")
print("=" * 80)

# 1. Myservice 문서 통계
print("\n1️⃣  Myservice 문서 통계")
print("-" * 80)

count_result = es.count(
    index=index,
    body={"query": {"term": {"doc_type.keyword": "myservice"}}}
)
print(f"Total myservice documents: {count_result['count']}")

# 섹션별 분포
section_result = es.search(
    index=index,
    body={
        "query": {"term": {"doc_type.keyword": "myservice"}},
        "size": 0,
        "aggs": {
            "by_chapter": {
                "terms": {"field": "chapter.keyword", "size": 10}
            }
        }
    }
)

print("\nSections breakdown:")
for bucket in section_result["aggregations"]["by_chapter"]["buckets"]:
    print(f"  - {bucket['key']:>10}: {bucket['doc_count']:>6} docs")

# Unique doc_ids
unique_result = es.search(
    index=index,
    body={
        "query": {"term": {"doc_type.keyword": "myservice"}},
        "size": 0,
        "aggs": {
            "unique_docs": {
                "cardinality": {"field": "doc_id.keyword"}
            }
        }
    }
)
print(f"\nUnique maintenance events: {unique_result['aggregations']['unique_docs']['value']}")

# 2. 최근 myservice 문서 샘플 조회
print("\n2️⃣  최근 myservice 문서 (5개)")
print("-" * 80)

recent_result = es.search(
    index=index,
    body={
        "query": {"term": {"doc_type.keyword": "myservice"}},
        "size": 5,
        "sort": [{"created_at": "desc"}],
        "_source": {
            "includes": ["doc_id", "chapter", "content", "device_name", "chunk_summary", "chunk_keywords"]
        }
    }
)

for i, hit in enumerate(recent_result["hits"]["hits"], 1):
    source = hit["_source"]
    print(f"\n[{i}] Doc: {source.get('doc_id')} | Chapter: {source.get('chapter')}")
    print(f"Device: {source.get('device_name', 'N/A')}")
    content = source.get('content', '')
    print(f"Content: {content[:150]}{'...' if len(content) > 150 else ''}")
    if source.get('chunk_summary'):
        print(f"Summary: {source['chunk_summary'][:100]}...")
    if source.get('chunk_keywords'):
        print(f"Keywords: {', '.join(source['chunk_keywords'][:5])}")
    print("-" * 80)

# 3. 특정 doc_id의 모든 섹션 조회 함수
def get_myservice_doc(doc_id):
    """특정 doc_id의 모든 섹션을 조회합니다."""
    result = es.search(
        index=index,
        body={
            "query": {
                "bool": {
                    "must": [
                        {"term": {"doc_type.keyword": "myservice"}},
                        {"term": {"doc_id.keyword": doc_id}}
                    ]
                }
            },
            "size": 10,
            "sort": [{"chapter.keyword": "asc"}],
            "_source": {"excludes": ["embedding"]}
        }
    )

    print(f"\n3️⃣  Maintenance Event: {doc_id}")
    print("=" * 80)

    if result["hits"]["total"]["value"] == 0:
        print(f"No documents found for doc_id: {doc_id}")
        return

    for hit in result["hits"]["hits"]:
        source = hit["_source"]
        print(f"\n[{source.get('chapter', 'N/A').upper()}]")
        print(f"Chunk ID: {source.get('chunk_id')}")
        print(f"Content:\n{source.get('content', '')}")

        if source.get('chunk_summary'):
            print(f"\nSummary: {source['chunk_summary']}")
        if source.get('chunk_keywords'):
            print(f"Keywords: {', '.join(source['chunk_keywords'])}")
        print("-" * 80)

    return result


# 4. 키워드로 myservice 문서 검색
def search_myservice(keyword, size=10):
    """키워드로 myservice 문서를 검색합니다."""
    result = es.search(
        index=index,
        body={
            "query": {
                "bool": {
                    "must": [
                        {"term": {"doc_type.keyword": "myservice"}},
                        {"match": {"content": keyword}}
                    ]
                }
            },
            "size": size,
            "_source": {
                "includes": ["doc_id", "chapter", "content", "device_name", "chunk_summary"]
            },
            "highlight": {
                "fields": {
                    "content": {"fragment_size": 150}
                }
            }
        }
    )

    print(f"\n4️⃣  Search Results for: '{keyword}'")
    print("=" * 80)
    print(f"Found {result['hits']['total']['value']} documents")

    for i, hit in enumerate(result["hits"]["hits"], 1):
        source = hit["_source"]
        print(f"\n[{i}] Score: {hit['_score']:.2f}")
        print(f"Doc: {source.get('doc_id')} | Chapter: {source.get('chapter')}")
        print(f"Device: {source.get('device_name', 'N/A')}")

        # Highlight 표시
        if 'highlight' in hit and 'content' in hit['highlight']:
            print(f"Match: ...{hit['highlight']['content'][0]}...")
        else:
            content = source.get('content', '')
            print(f"Content: {content[:150]}...")

        if source.get('chunk_summary'):
            print(f"Summary: {source['chunk_summary'][:100]}...")
        print("-" * 80)

    return result


print("\n" + "=" * 80)
print("✓ Functions available:")
print("  - get_myservice_doc(doc_id)   : 특정 doc_id의 모든 섹션 조회")
print("  - search_myservice(keyword)   : 키워드로 검색")
print("\nExample:")
print("  get_myservice_doc('40045191')")
print("  search_myservice('FCIP')")
print("=" * 80)
