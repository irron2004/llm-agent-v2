"""Check all doc_type in ES index."""

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
print("ALL DOC_TYPES IN ES INDEX")
print("=" * 80)

# Get all doc_types
result = es.search(
    index=index,
    body={
        "size": 0,
        "aggs": {
            "doc_types": {
                "terms": {
                    "field": "doc_type.keyword",
                    "size": 50
                }
            }
        }
    }
)

total_docs = es.count(index=index)["count"]
print(f"\nTotal documents in index: {total_docs}")
print("\nDoc types breakdown:")
print("-" * 80)

for bucket in result["aggregations"]["doc_types"]["buckets"]:
    doc_type = bucket["key"]
    count = bucket["doc_count"]
    percentage = (count / total_docs * 100) if total_docs > 0 else 0
    print(f"{doc_type:>20}: {count:>8} docs ({percentage:>5.1f}%)")

print("=" * 80)
print("\n⚠️  delete_myservice_docs.py will ONLY delete doc_type='myservice'")
print("=" * 80)
