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


def _collect_term_fields(clause: dict) -> set[str]:
    fields: set[str] = set()
    for term_clause in clause.get("bool", {}).get("should", []):
        term = term_clause.get("term", {})
        fields.update(term.keys())
    return fields


def _collect_term_values(clause: dict) -> set[str]:
    values: set[str] = set()
    for term_clause in clause.get("bool", {}).get("should", []):
        term = term_clause.get("term", {})
        for value in term.values():
            values.add(str(value))
    return values


def test_build_filter_includes_doc_id_and_keyword_clauses() -> None:
    engine = EsSearchEngine(es_client=None, index_name="test-index")

    filters = engine.build_filter(doc_ids=["a", "b"])

    assert filters is not None
    fields = _collect_term_fields(filters)
    assert "doc_id" in fields
    assert "doc_id.keyword" in fields
    values = _collect_term_values(filters)
    assert values == {"a", "b"}


def test_build_filter_combines_doc_ids_with_doc_type() -> None:
    engine = EsSearchEngine(es_client=None, index_name="test-index")

    filters = engine.build_filter(doc_type="sop", doc_ids=["a", "b"])

    assert filters is not None
    must_clauses = filters["bool"]["must"]
    assert len(must_clauses) == 2

    doc_id_clause = next(
        clause
        for clause in must_clauses
        if "doc_id" in _collect_term_fields(clause)
        or "doc_id.keyword" in _collect_term_fields(clause)
    )
    doc_type_clause = next(
        clause
        for clause in must_clauses
        if "doc_type" in _collect_term_fields(clause)
        or "doc_type.keyword" in _collect_term_fields(clause)
    )

    assert _collect_term_fields(doc_id_clause) == {"doc_id", "doc_id.keyword"}
    assert _collect_term_values(doc_id_clause) == {"a", "b"}
    assert "doc_type" in _collect_term_fields(doc_type_clause)
    assert "doc_type.keyword" in _collect_term_fields(doc_type_clause)
