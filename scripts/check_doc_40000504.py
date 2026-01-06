"""Check specific document in ES."""

import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from elasticsearch import Elasticsearch
from backend.config.settings import search_settings

# Connect to ES
es = Elasticsearch(hosts=[search_settings.es_host])
index = f"{search_settings.es_index_prefix}_{search_settings.es_env}_current"

# Search for the specific document
doc_id = "40000504"

response = es.search(
    index=index,
    body={
        "query": {
            "term": {
                "doc_id.keyword": doc_id
            }
        },
        "size": 10,
        "_source": {
            "excludes": ["embedding"]
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
    print(f"Content (full): '{source.get('content')}'")
    print(f"Device: {source.get('device_name')}")
    print(f"Doc Description: {source.get('doc_description')}")
    print(f"Order No: {source.get('order_no')}")
    print("-" * 80)
