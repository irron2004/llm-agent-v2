from backend.llm_infrastructure.llm.langgraph_agent import retrieve_node
from backend.llm_infrastructure.retrieval.base import RetrievalResult


class MixedDeviceRetriever:
    def retrieve(self, query: str, top_k: int = 10, **kwargs):
        return [
            RetrievalResult(
                doc_id="doc_supra_v",
                content="supra v content",
                score=1.0,
                metadata={"doc_type": "sop", "device_name": "SUPRA V"},
                raw_text="supra v raw",
            ),
            RetrievalResult(
                doc_id="doc_omni",
                content="omni content",
                score=0.9,
                metadata={"doc_type": "sop", "device_name": "OMNI"},
                raw_text="omni raw",
            ),
        ]


def test_retrieve_node_strict_device_filter_excludes_other_devices() -> None:
    retriever = MixedDeviceRetriever()
    state = {
        "query": "power cal",
        "route": "setup",
        "search_queries": ["power cal"],
        "selected_devices": ["SUPRA V"],
    }

    result = retrieve_node(
        state,
        retriever=retriever,
        reranker=None,
        retrieval_top_k=10,
        final_top_k=10,
    )

    device_names = {(doc.metadata or {}).get("device_name") for doc in result["docs"]}
    assert device_names == {"SUPRA V"}


def test_retrieve_node_without_device_filter_keeps_all_devices() -> None:
    retriever = MixedDeviceRetriever()
    state = {
        "query": "power cal",
        "route": "setup",
        "search_queries": ["power cal"],
    }

    result = retrieve_node(
        state,
        retriever=retriever,
        reranker=None,
        retrieval_top_k=10,
        final_top_k=10,
    )

    device_names = {(doc.metadata or {}).get("device_name") for doc in result["docs"]}
    assert "SUPRA V" in device_names
    assert "OMNI" in device_names
