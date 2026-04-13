from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_infrastructure.llm.base import BaseLLM, LLMResponse
from backend.llm_infrastructure.llm import langgraph_agent as langgraph_agent_module
from backend.llm_infrastructure.llm.langgraph_agent import (
    PromptSpec,
    RetrievalResult,
    _build_setup_work_procedure_ref_map,
    _enrich_setup_doc_groups_with_work_procedure,
    answer_node,
)
from backend.llm_infrastructure.llm.prompt_loader import PromptTemplate


class NoopLLM(BaseLLM):
    def generate(
        self,
        messages: Iterable[dict[str, str]],
        *,
        response_model=None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(text="yes")


class FakeSectionHit:
    def __init__(self, result: RetrievalResult) -> None:
        self._result = result

    def to_retrieval_result(self) -> RetrievalResult:
        return self._result


class FakeESEngine:
    def __init__(self, by_doc_id: dict[str, list[RetrievalResult]]) -> None:
        self._by_doc_id = by_doc_id

    def fetch_section_chunks_by_keyword(
        self,
        *,
        doc_id: str,
        keyword: str,
        max_pages: int,
        content_index: str | None = None,
    ) -> list[FakeSectionHit]:
        results = self._by_doc_id.get(doc_id, [])[:max_pages]
        return [FakeSectionHit(result) for result in results]


def _prompt(name: str) -> PromptTemplate:
    return PromptTemplate(
        name=name, version="v1", system="", user="{sys.query}\n{ref_text}", raw={}
    )


def _make_spec() -> PromptSpec:
    base = _prompt("base")
    return PromptSpec(
        router=_prompt("router"),
        setup_mq=_prompt("setup_mq"),
        ts_mq=_prompt("ts_mq"),
        general_mq=_prompt("general_mq"),
        st_gate=_prompt("st_gate"),
        st_mq=_prompt("st_mq"),
        setup_ans=base,
        ts_ans=base,
        general_ans=base,
        judge_setup_sys="",
        judge_ts_sys="",
        judge_general_sys="",
        issue_ans=base,
        issue_detail_ans=base,
    )


def _result(doc_id: str, content: str, *, section: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        doc_id=doc_id,
        content=content,
        score=score,
        metadata={"section_chapter": section, "doc_type": "sop", "page": 1},
        raw_text=content,
    )


def test_enrich_setup_doc_groups_adds_work_procedure_per_doc_id() -> None:
    doc_groups = [
        (
            "group-1",
            [
                {
                    "doc_id": "doc1#001",
                    "content": "doc1 chapter A",
                    "metadata": {"section": "3. 사고 사례"},
                },
                {
                    "doc_id": "doc2#001",
                    "content": "doc2 chapter B",
                    "metadata": {"section": "4. Flow Chart"},
                },
                {
                    "doc_id": "doc3#001",
                    "content": "doc3 chapter A",
                    "metadata": {"section": "3. 사고 사례"},
                },
            ],
        )
    ]
    wp_map = {
        "doc1": [
            {
                "doc_id": "doc1#wp",
                "content": "doc1 work procedure",
                "metadata": {"section": "6. Work Procedure"},
            }
        ],
        "doc2": [
            {
                "doc_id": "doc2#wp",
                "content": "doc2 work procedure",
                "metadata": {"section": "6. Work Procedure"},
            }
        ],
        "doc3": [
            {
                "doc_id": "doc3#wp",
                "content": "doc3 work procedure",
                "metadata": {"section": "6. Work Procedure"},
            }
        ],
    }

    enriched = _enrich_setup_doc_groups_with_work_procedure(
        doc_groups,
        work_procedure_ref_map=wp_map,
        max_refs_per_doc=1,
    )

    refs = enriched[0][1]
    assert [ref["doc_id"] for ref in refs] == [
        "doc1#001",
        "doc1#wp",
        "doc2#001",
        "doc2#wp",
        "doc3#001",
        "doc3#wp",
    ]


def test_build_setup_work_procedure_ref_map_uses_existing_or_fetched_wp() -> None:
    docs = [
        _result("doc1#001", "doc1 chapter A", section="3. 사고 사례"),
        _result("doc2#001", "doc2 work procedure", section="6. Work Procedure"),
    ]
    engine = FakeESEngine(
        {
            "doc1#001": [_result("doc1#wp", "doc1 fetched wp", section="6. Work Procedure")],
        }
    )

    wp_map = _build_setup_work_procedure_ref_map(docs, es_engine=engine, max_refs_per_doc=1)

    assert "doc1" in wp_map
    assert "doc2" in wp_map
    assert wp_map["doc1"][0]["content"] == "doc1 fetched wp"
    assert wp_map["doc2"][0]["content"] == "doc2 work procedure"


def test_answer_node_enriches_setup_groups_before_relevance_check(monkeypatch) -> None:
    captured_doc_texts: list[str] = []

    def fake_check_doc_relevance(query: str, doc_ref_text: str, *, llm: BaseLLM):
        captured_doc_texts.append(doc_ref_text)
        return "doc1 work procedure" in doc_ref_text

    def fake_invoke_with_reasoning(*args: Any, **kwargs: Any):
        return "# 제목\n\n## 작업 절차\n1. 절차\n\n## 참고문헌\n[1] ref", ""

    monkeypatch.setattr(langgraph_agent_module, "_check_doc_relevance", fake_check_doc_relevance)
    monkeypatch.setattr(
        langgraph_agent_module, "_invoke_llm_with_reasoning", fake_invoke_with_reasoning
    )
    monkeypatch.setattr(
        langgraph_agent_module,
        "_validate_answer_format",
        lambda *args, **kwargs: {
            "ok": True,
            "title_ok": True,
            "missing_sections": [],
            "numbering_ok": True,
            "has_emoji_numbering": False,
            "has_markdown_table": False,
            "citations_ok": True,
            "references_ok": True,
            "language_ok": True,
        },
    )

    state = {
        "route": "setup",
        "query": "ZEDIUS XP source 교체 방법",
        "original_query": "ZEDIUS XP source 교체 방법",
        "target_language": "en",
        "answer_ref_json": [
            {
                "doc_id": "doc1#001",
                "content": "doc1 chapter A",
                "metadata": {"section": "3. 사고 사례", "rerank_hit_count": 1},
            },
            {
                "doc_id": "doc2#001",
                "content": "doc2 chapter B",
                "metadata": {"section": "4. Flow Chart", "rerank_hit_count": 1},
            },
        ],
        "setup_work_procedure_ref_map": {
            "doc1": [
                {
                    "doc_id": "doc1#wp",
                    "content": "doc1 work procedure",
                    "metadata": {"section": "6. Work Procedure", "rerank_hit_count": 1},
                }
            ],
            "doc2": [
                {
                    "doc_id": "doc2#wp",
                    "content": "doc2 work procedure",
                    "metadata": {"section": "6. Work Procedure", "rerank_hit_count": 1},
                }
            ],
        },
    }

    result = answer_node(state, llm=NoopLLM(), spec=_make_spec())

    assert captured_doc_texts, "expected setup relevance check to run"
    assert "doc1 work procedure" in captured_doc_texts[0]
    assert result["answer_ref_json"][0]["doc_id"] == "doc1#001"
    assert any(ref["doc_id"] == "doc1#wp" for ref in result["answer_ref_json"])
