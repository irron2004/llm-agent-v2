from backend.llm_infrastructure.llm.langgraph_agent import retrieve_node
from backend.llm_infrastructure.retrieval.base import RetrievalResult


class MixedEquipIdRetriever:
    def retrieve(self, query: str, top_k: int = 10, **kwargs):
        return [
            RetrievalResult(
                doc_id="doc_epag50",
                content="equip EPAG50 content",
                score=1.0,
                metadata={"doc_type": "gcb", "device_name": "SUPRA V", "equip_id": "EPAG50"},
                raw_text="equip EPAG50 raw",
            ),
            RetrievalResult(
                doc_id="doc_epag51",
                content="equip EPAG51 content",
                score=0.9,
                metadata={"doc_type": "gcb", "device_name": "SUPRA V", "equip_id": "EPAG51"},
                raw_text="equip EPAG51 raw",
            ),
        ]


def test_retrieve_node_strict_equip_id_filter_excludes_other_equip_ids() -> None:
    retriever = MixedEquipIdRetriever()
    state = {
        "query": "gcb",
        "route": "general",
        "search_queries": ["gcb"],
        "selected_equip_ids": ["epag50"],
    }

    result = retrieve_node(
        state,
        retriever=retriever,
        reranker=None,
        retrieval_top_k=10,
        final_top_k=10,
    )

    equip_ids = {(doc.metadata or {}).get("equip_id") for doc in result["docs"]}
    assert equip_ids == {"EPAG50"}


def test_retrieve_node_without_equip_id_filter_keeps_all_equip_ids() -> None:
    retriever = MixedEquipIdRetriever()
    state = {
        "query": "gcb",
        "route": "general",
        "search_queries": ["gcb"],
    }

    result = retrieve_node(
        state,
        retriever=retriever,
        reranker=None,
        retrieval_top_k=10,
        final_top_k=10,
    )

    equip_ids = {(doc.metadata or {}).get("equip_id") for doc in result["docs"]}
    assert "EPAG50" in equip_ids
    assert "EPAG51" in equip_ids
