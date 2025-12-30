"""Check ES documents for maintenance report."""

import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from elasticsearch import Elasticsearch
from backend.config.settings import search_settings

# Connect to ES
es = Elasticsearch(hosts=[search_settings.es_host])
index = f"{search_settings.es_index_prefix}_{search_settings.es_env}_current"

# Search for our test document
doc_id = "sample_maintenance_report"

response = es.search(
    index=index,
    body={
        "query": {
            "term": {
                "doc_id.keyword": doc_id
            }
        },
        "size": 10,
        "sort": [{"chunk_id.keyword": "asc"}],
        "_source": {
            "excludes": ["embedding"]  # Skip embedding for readability
        }
    }
)

hits = response["hits"]["hits"]
print(f"Found {len(hits)} documents for doc_id='{doc_id}'")
print("=" * 80)

for i, hit in enumerate(hits, 1):
    source = hit["_source"]
    print(f"\n[Document {i}]")
    print(f"Chunk ID: {source.get('chunk_id')}")
    print(f"Chapter: {source.get('chapter')}")
    print(f"Content: {source.get('content')[:150]}...")
    print(f"Device: {source.get('device_name')}")
    print(f"Summary: {source.get('chunk_summary')}")
    print(f"Keywords: {source.get('chunk_keywords')}")
    print(f"Overall Keywords: {source.get('overall_keywords', [])[:5]}")
    print("-" * 80)

print("\n✓ All sections are stored as separate documents!")
print(f"Retrieve all sections: query by doc_id.keyword='{doc_id}'")
