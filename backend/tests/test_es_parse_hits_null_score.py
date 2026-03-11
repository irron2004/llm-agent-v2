# Test for ES hit parsing with null/missing _score values.
# Verifies fix for page/chunk fetch stability.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

try:
    from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchEngine
except ModuleNotFoundError:
    from llm_infrastructure.retrieval.engines.es_search import EsSearchEngine


def test_parse_hits_score_none():
    """Test that _score=None does not crash and defaults to 0.0."""
    engine = EsSearchEngine(es_client=None, index_name="test-index")
    resp = {
        "hits": {
            "hits": [
                {
                    "_id": "doc_1",
                    "_score": None,  # Explicit None - would crash with float(None)
                    "_source": {
                        "doc_id": "doc_1",
                        "chunk_id": "chunk_1",
                        "content": "test content",
                        "page": 1,
                    },
                }
            ]
        }
    }
    hits = engine._parse_hits(resp)
    assert len(hits) == 1
    assert hits[0].score == 0.0
    assert hits[0].chunk_id == "chunk_1"


def test_parse_hits_score_missing():
    """Test that missing _score defaults to 0.0."""
    engine = EsSearchEngine(es_client=None, index_name="test-index")
    resp = {
        "hits": {
            "hits": [
                {
                    "_id": "doc_2",
                    # No _score key at all
                    "_source": {
                        "doc_id": "doc_2",
                        "chunk_id": "chunk_2",
                        "content": "test content",
                        "page": 2,
                    },
                }
            ]
        }
    }
    hits = engine._parse_hits(resp)
    assert len(hits) == 1
    assert hits[0].score == 0.0


def test_parse_hits_score_string():
    """Test that non-numeric _score does not crash and defaults to 0.0."""
    engine = EsSearchEngine(es_client=None, index_name="test-index")
    resp = {
        "hits": {
            "hits": [
                {
                    "_id": "doc_3",
                    "_score": "not_a_number",  # String - would crash with float()
                    "_source": {
                        "doc_id": "doc_3",
                        "chunk_id": "chunk_3",
                        "content": "test content",
                        "page": 3,
                    },
                }
            ]
        }
    }
    hits = engine._parse_hits(resp)
    assert len(hits) == 1
    assert hits[0].score == 0.0


def test_parse_hits_metadata_always_has_chunk_id():
    """Test that metadata always contains chunk_id, even when source lacks it."""
    engine = EsSearchEngine(es_client=None, index_name="test-index")
    resp = {
        "hits": {
            "hits": [
                {
                    "_id": "doc_4",
                    "_score": 1.5,
                    "_source": {
                        "doc_id": "doc_4",
                        # No chunk_id in _source - should fallback to _id
                        "content": "test content",
                        "page": 4,
                        "extra_field": "value",
                    },
                }
            ]
        }
    }
    hits = engine._parse_hits(resp)
    assert len(hits) == 1
    # chunk_id should fallback to _id
    assert hits[0].chunk_id == "doc_4"
    # metadata should have chunk_id
    assert "chunk_id" in hits[0].metadata
    assert hits[0].metadata["chunk_id"] == "doc_4"
    # Other source fields should still be in metadata
    assert "extra_field" in hits[0].metadata
    assert hits[0].metadata["extra_field"] == "value"


def test_parse_hits_normal_score():
    """Test that normal numeric scores still work correctly."""
    engine = EsSearchEngine(es_client=None, index_name="test-index")
    resp = {
        "hits": {
            "hits": [
                {
                    "_id": "doc_5",
                    "_score": 3.14159,
                    "_source": {
                        "doc_id": "doc_5",
                        "chunk_id": "chunk_5",
                        "content": "test content",
                        "page": 5,
                    },
                }
            ]
        }
    }
    hits = engine._parse_hits(resp)
    assert len(hits) == 1
    assert hits[0].score == 3.14159


def test_parse_hits_zero_score():
    """Test that explicit 0.0 score is preserved."""
    engine = EsSearchEngine(es_client=None, index_name="test-index")
    resp = {
        "hits": {
            "hits": [
                {
                    "_id": "doc_6",
                    "_score": 0.0,
                    "_source": {
                        "doc_id": "doc_6",
                        "chunk_id": "chunk_6",
                        "content": "test content",
                        "page": 6,
                    },
                }
            ]
        }
    }
    hits = engine._parse_hits(resp)
    assert len(hits) == 1
    assert hits[0].score == 0.0


if __name__ == "__main__":
    test_parse_hits_score_none()
    test_parse_hits_score_missing()
    test_parse_hits_score_string()
    test_parse_hits_metadata_always_has_chunk_id()
    test_parse_hits_normal_score()
    test_parse_hits_zero_score()
    print("All tests passed!")
