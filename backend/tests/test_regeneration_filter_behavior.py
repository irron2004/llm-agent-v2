from backend.api.routers.agent import AgentRequest, _build_state_overrides
from backend.llm_infrastructure.llm.langgraph_agent import retrieve_node
from backend.llm_infrastructure.retrieval.base import RetrievalResult


class MixedDocTypeRetriever:
    def retrieve(self, query: str, top_k: int = 10, **kwargs):
        return [
            RetrievalResult(
                doc_id="doc_sop",
                content="sop content",
                score=1.0,
                metadata={"doc_type": "sop", "device_name": "SUPRA V"},
                raw_text="sop raw",
            ),
            RetrievalResult(
                doc_id="doc_generic",
                content="generic content",
                score=0.9,
                metadata={"doc_type": "generic", "device_name": "SUPRA V"},
                raw_text="generic raw",
            ),
            RetrievalResult(
                doc_id="doc_myservice",
                content="myservice content",
                score=0.8,
                metadata={"doc_type": "myservice", "device_name": "SUPRA V"},
                raw_text="myservice raw",
            ),
        ]


def test_build_state_overrides_marks_doc_type_strict() -> None:
    req = AgentRequest(message="test", filter_doc_types=["sop"])
    overrides = _build_state_overrides(req)

    selected_doc_types = overrides["selected_doc_types"]
    normalized = {str(doc_type).strip().lower() for doc_type in selected_doc_types}
    assert "sop" in normalized
    assert "sop/manual" in normalized
    assert overrides["selected_doc_types_strict"] is True
    assert overrides["detected_language"] == "en"


def test_build_state_overrides_sets_detected_language_when_overrides_exist() -> None:
    req = AgentRequest(message="장비 점검 방법", filter_devices=["SUPRA V"])
    overrides = _build_state_overrides(req)

    assert overrides["selected_devices"] == ["SUPRA V"]
    assert overrides["detected_language"] == "ko"


def test_retrieve_node_strict_doc_type_excludes_group_variants() -> None:
    retriever = MixedDocTypeRetriever()
    state = {
        "query": "ashing rate",
        "route": "ts",
        "search_queries": ["ashing rate"],
        "selected_doc_types": ["sop"],
        "selected_doc_types_strict": True,
    }

    result = retrieve_node(
        state,
        retriever=retriever,
        reranker=None,
        retrieval_top_k=10,
        final_top_k=10,
    )

    doc_types = {(doc.metadata or {}).get("doc_type") for doc in result["docs"]}
    assert doc_types == {"sop"}


def test_retrieve_node_non_strict_doc_type_keeps_group_variants() -> None:
    retriever = MixedDocTypeRetriever()
    state = {
        "query": "ashing rate",
        "route": "ts",
        "search_queries": ["ashing rate"],
        "selected_doc_types": ["sop"],
    }

    result = retrieve_node(
        state,
        retriever=retriever,
        reranker=None,
        retrieval_top_k=10,
        final_top_k=10,
    )

    doc_types = {(doc.metadata or {}).get("doc_type") for doc in result["docs"]}
    assert "sop" in doc_types
    assert "generic" in doc_types
