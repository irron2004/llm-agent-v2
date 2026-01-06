"""Check myservice documents in detail."""

import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from elasticsearch import Elasticsearch
from backend.config.settings import search_settings

# Connect to ES
es = Elasticsearch(hosts=[search_settings.es_host])
index = f"{search_settings.es_index_prefix}_{search_settings.es_env}_current"

print("=" * 80)
print("MYSERVICE DOCUMENTS DETAILED ANALYSIS")
print("=" * 80)

# Total docs in index
total_count = es.count(index=index)["count"]
print(f"\n📊 Total documents in index: {total_count}")

# Myservice docs count
myservice_count = es.count(
    index=index,
    body={"query": {"term": {"doc_type.keyword": "myservice"}}}
)["count"]
print(f"📊 Myservice documents: {myservice_count}")

# Myservice by chapter
print("\n" + "-" * 80)
print("Myservice breakdown by chapter:")
print("-" * 80)

result = es.search(
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

for bucket in result["aggregations"]["by_chapter"]["buckets"]:
    chapter = bucket["key"]
    count = bucket["doc_count"]
    print(f"  {chapter:>10}: {count:>6} docs")

# Unique doc_ids (maintenance events)
print("\n" + "-" * 80)
print("Unique maintenance events (doc_id):")
print("-" * 80)

doc_id_result = es.search(
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

unique_docs = doc_id_result["aggregations"]["unique_docs"]["value"]
print(f"Unique doc_ids: {unique_docs}")

if unique_docs > 0:
    avg_sections_per_doc = myservice_count / unique_docs
    print(f"Average sections per doc: {avg_sections_per_doc:.2f}")

# Sample doc_ids
print("\n" + "-" * 80)
print("Sample doc_ids:")
print("-" * 80)

sample_result = es.search(
    index=index,
    body={
        "query": {"term": {"doc_type.keyword": "myservice"}},
        "size": 0,
        "aggs": {
            "sample_docs": {
                "terms": {"field": "doc_id.keyword", "size": 10}
            }
        }
    }
)

for bucket in sample_result["aggregations"]["sample_docs"]["buckets"]:
    doc_id = bucket["key"]
    count = bucket["doc_count"]
    print(f"  {doc_id}: {count} sections")

print("=" * 80)
