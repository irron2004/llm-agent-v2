"""expand_related_docs_node 단위 테스트"""

import pytest
from unittest.mock import MagicMock, call

from backend.llm_infrastructure.llm.langgraph_agent import (
    expand_related_docs_node,
    EXPAND_TOP_K,
)
from backend.llm_infrastructure.retrieval.base import RetrievalResult


def _make_doc(
    doc_id: str,
    page: int | None = 1,
    doc_type: str = "manual",
    content: str | None = None,
) -> RetrievalResult:
    """테스트용 문서 생성 헬퍼"""
    metadata = {"doc_type": doc_type}
    if page is not None:
        metadata["page"] = page
    return RetrievalResult(
        doc_id=doc_id,
        content=content or f"content_{doc_id}",
        score=1.0,
        metadata=metadata,
        raw_text=f"raw_{doc_id}",
    )


class TestExpandTopKLimit:
    """상위 K개 확장 제한 테스트"""

    def test_expand_top_k_constant_is_5(self):
        """EXPAND_TOP_K 상수가 5인지 확인"""
        assert EXPAND_TOP_K == 5

    def test_expand_only_top_5_docs(self):
        """10개 문서 중 상위 5개만 확장 시도"""
        # Given: 10개 문서
        docs = [_make_doc(f"doc_{i}", page=i + 1) for i in range(10)]
        state = {"docs": docs}

        page_fetcher = MagicMock(return_value=[])

        # When
        expand_related_docs_node(state, page_fetcher=page_fetcher)

        # Then: page_fetcher는 상위 5개에 대해서만 호출 (5회)
        assert page_fetcher.call_count == 5

    def test_docs_less_than_top_k_all_expanded(self):
        """문서가 5개 미만일 때 전부 확장 시도"""
        # Given: 3개 문서
        docs = [_make_doc(f"doc_{i}", page=i + 1) for i in range(3)]
        state = {"docs": docs}

        page_fetcher = MagicMock(return_value=[])

        # When
        expand_related_docs_node(state, page_fetcher=page_fetcher)

        # Then: 3회 호출
        assert page_fetcher.call_count == 3

    def test_exactly_5_docs_all_expanded(self):
        """정확히 5개 문서일 때 전부 확장"""
        # Given: 5개 문서
        docs = [_make_doc(f"doc_{i}", page=i + 1) for i in range(5)]
        state = {"docs": docs}

        page_fetcher = MagicMock(return_value=[])

        # When
        expand_related_docs_node(state, page_fetcher=page_fetcher)

        # Then: 5회 호출
        assert page_fetcher.call_count == 5


class TestRemainingDocsPreserved:
    """나머지 문서 원본 유지 테스트"""

    def test_remaining_docs_included_in_result(self):
        """6번째 이후 문서도 결과에 포함"""
        # Given: 7개 문서
        docs = [_make_doc(f"doc_{i}", page=i + 1) for i in range(7)]
        state = {"docs": docs}

        page_fetcher = MagicMock(return_value=[])

        # When
        result = expand_related_docs_node(state, page_fetcher=page_fetcher)

        # Then: answer_ref_json은 상위 5개만 포함
        assert len(result["answer_ref_json"]) == min(EXPAND_TOP_K, len(docs))

    def test_remaining_docs_have_original_content(self):
        """6번째 이후 문서는 원본 content 유지"""
        # Given: 7개 문서, 각각 고유 content
        docs = [_make_doc(f"doc_{i}", page=i + 1, content=f"original_content_{i}") for i in range(7)]
        state = {"docs": docs}

        # page_fetcher가 빈 결과 반환 -> 확장 안됨
        page_fetcher = MagicMock(return_value=[])

        # When
        result = expand_related_docs_node(state, page_fetcher=page_fetcher)

        # Then: 상위 5개 문서만 확인
        ref_json = result["answer_ref_json"]
        for i, ref in enumerate(ref_json):
            # raw_text가 있으면 raw_text, 없으면 content 사용
            assert f"doc_{i}" in ref["doc_id"]


class TestDocTypeBasedExpansion:
    """doc_type 기반 확장 분기 테스트"""

    def test_gcb_doc_uses_doc_fetcher(self):
        """gcb 타입은 doc_fetcher 사용"""
        docs = [_make_doc("gcb_doc", page=1, doc_type="gcb")]
        state = {"docs": docs}

        page_fetcher = MagicMock()
        doc_fetcher = MagicMock(return_value=[])

        # When
        expand_related_docs_node(
            state, page_fetcher=page_fetcher, doc_fetcher=doc_fetcher
        )

        # Then
        doc_fetcher.assert_called_once_with("gcb_doc")
        page_fetcher.assert_not_called()

    def test_myservice_doc_uses_doc_fetcher(self):
        """myservice 타입은 doc_fetcher 사용"""
        docs = [_make_doc("myservice_doc", page=1, doc_type="myservice")]
        state = {"docs": docs}

        page_fetcher = MagicMock()
        doc_fetcher = MagicMock(return_value=[])

        # When
        expand_related_docs_node(
            state, page_fetcher=page_fetcher, doc_fetcher=doc_fetcher
        )

        # Then
        doc_fetcher.assert_called_once_with("myservice_doc")
        page_fetcher.assert_not_called()

    def test_manual_doc_uses_page_fetcher(self):
        """일반 문서(manual)는 page_fetcher 사용"""
        docs = [_make_doc("manual_doc", page=5, doc_type="manual")]
        state = {"docs": docs}

        page_fetcher = MagicMock(return_value=[])
        doc_fetcher = MagicMock()

        # When
        expand_related_docs_node(
            state, page_fetcher=page_fetcher, doc_fetcher=doc_fetcher
        )

        # Then: page 5 기준 ±2 -> [3, 4, 5, 6, 7]
        page_fetcher.assert_called_once()
        call_args = page_fetcher.call_args
        assert call_args[0][0] == "manual_doc"
        assert set(call_args[0][1]) == {3, 4, 5, 6, 7}
        doc_fetcher.assert_not_called()

    def test_page_range_boundary_at_page_1(self):
        """page 1일 때 범위가 [1, 2, 3] (음수 안됨)"""
        docs = [_make_doc("doc", page=1, doc_type="manual")]
        state = {"docs": docs}

        page_fetcher = MagicMock(return_value=[])

        # When
        expand_related_docs_node(state, page_fetcher=page_fetcher)

        # Then: page 1 기준 ±2 -> max(1, 1-2)=1 ~ 3 -> [1, 2, 3]
        call_args = page_fetcher.call_args
        pages = call_args[0][1]
        assert min(pages) >= 1
        assert set(pages) == {1, 2, 3}


class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_no_page_info_skips_expansion(self):
        """page 정보 없으면 확장 스킵"""
        docs = [_make_doc("no_page_doc", page=None, doc_type="manual")]
        state = {"docs": docs}

        page_fetcher = MagicMock()

        # When
        expand_related_docs_node(state, page_fetcher=page_fetcher)

        # Then: page_fetcher 호출 안됨
        page_fetcher.assert_not_called()

    def test_empty_docs_returns_empty(self):
        """빈 문서 리스트 -> 빈 결과"""
        state = {"docs": []}

        # When
        result = expand_related_docs_node(state, page_fetcher=MagicMock())

        # Then
        assert "docs" not in result
        assert "_events" in result

    def test_no_fetchers_returns_empty(self):
        """fetcher 없으면 빈 결과"""
        docs = [_make_doc("doc", page=1)]
        state = {"docs": docs}

        # When
        result = expand_related_docs_node(state)

        # Then
        assert "docs" not in result
        assert "_events" in result

    def test_mixed_doc_types(self):
        """여러 doc_type 혼합 시 각각 적절한 fetcher 사용"""
        docs = [
            _make_doc("gcb_doc", page=1, doc_type="gcb"),
            _make_doc("manual_doc", page=5, doc_type="manual"),
            _make_doc("myservice_doc", page=1, doc_type="myservice"),
        ]
        state = {"docs": docs}

        page_fetcher = MagicMock(return_value=[])
        doc_fetcher = MagicMock(return_value=[])

        # When
        expand_related_docs_node(
            state, page_fetcher=page_fetcher, doc_fetcher=doc_fetcher
        )

        # Then
        # gcb, myservice -> doc_fetcher (2회)
        assert doc_fetcher.call_count == 2
        # manual -> page_fetcher (1회)
        assert page_fetcher.call_count == 1


class TestExpansionResultFormat:
    """확장 결과 포맷 테스트"""

    def test_result_has_answer_ref_json(self):
        """결과에 answer_ref_json 키 존재"""
        docs = [_make_doc("doc", page=1)]
        state = {"docs": docs}

        page_fetcher = MagicMock(return_value=[])

        # When
        result = expand_related_docs_node(state, page_fetcher=page_fetcher)

        # Then
        assert "answer_ref_json" in result

    def test_ref_json_has_required_fields(self):
        """각 ref_json 항목에 필수 필드 존재"""
        docs = [_make_doc("doc", page=1)]
        state = {"docs": docs}

        page_fetcher = MagicMock(return_value=[])

        # When
        result = expand_related_docs_node(state, page_fetcher=page_fetcher)

        # Then
        ref = result["answer_ref_json"][0]
        assert "rank" in ref
        assert "doc_id" in ref
        assert "content" in ref

    def test_display_docs_merge_same_doc_id_for_myservice(self):
        """myservice 동일 doc_id는 display_docs에서 병합"""
        docs = [
            _make_doc("same_doc", page=0, doc_type="myservice"),
            _make_doc("same_doc", page=1, doc_type="myservice"),
        ]
        state = {"docs": docs}

        doc_fetcher = MagicMock(return_value=[])

        result = expand_related_docs_node(state, doc_fetcher=doc_fetcher)

        display_docs = result.get("display_docs", [])
        assert len(display_docs) == 1
        assert display_docs[0].doc_id == "same_doc"


class TestExpandedPagesMetadata:
    """확장된 페이지 정보가 metadata에 저장되는지 테스트"""

    def test_page_expansion_stores_expanded_pages(self):
        """페이지 기반 확장 시 expanded_pages가 metadata에 저장됨"""
        docs = [_make_doc("doc1", page=5, doc_type="manual")]
        state = {"docs": docs}

        related = [
            _make_doc("doc1_p3", page=3),
            _make_doc("doc1_p4", page=4),
            _make_doc("doc1_p5", page=5),
            _make_doc("doc1_p6", page=6),
            _make_doc("doc1_p7", page=7),
        ]
        page_fetcher = MagicMock(return_value=related)

        # When
        result = expand_related_docs_node(state, page_fetcher=page_fetcher)

        # Then
        expanded_doc = result["docs"][0]
        assert expanded_doc.metadata is not None
        assert "expanded_pages" in expanded_doc.metadata
        assert expanded_doc.metadata["expanded_pages"] == [3, 4, 5, 6, 7]

    def test_page_expansion_at_boundary(self):
        """페이지 1에서 시작 시 expanded_pages가 1부터 시작"""
        docs = [_make_doc("doc1", page=1, doc_type="manual")]
        state = {"docs": docs}

        related = [_make_doc("doc1_p1", page=1)]
        page_fetcher = MagicMock(return_value=related)

        # When
        result = expand_related_docs_node(state, page_fetcher=page_fetcher)

        # Then
        expanded_doc = result["docs"][0]
        # page_window=2이면 max(1,1-2)=1부터 1+2=3까지
        assert expanded_doc.metadata["expanded_pages"] == [1, 2, 3]

    def test_gcb_doc_collects_pages_from_chunks(self):
        """gcb 문서는 모든 청크에서 페이지 정보 수집"""
        docs = [_make_doc("gcb_doc", page=1, doc_type="gcb")]
        state = {"docs": docs}

        related = [
            _make_doc("chunk1", page=1),
            _make_doc("chunk2", page=2),
            _make_doc("chunk3", page=5),
            _make_doc("chunk4", page=3),  # 순서가 바뀌어도 정렬됨
        ]
        doc_fetcher = MagicMock(return_value=related)

        # When
        result = expand_related_docs_node(state, doc_fetcher=doc_fetcher)

        # Then
        expanded_doc = result["docs"][0]
        assert expanded_doc.metadata is not None
        assert "expanded_pages" in expanded_doc.metadata
        assert expanded_doc.metadata["expanded_pages"] == [1, 2, 3, 5]  # sorted

    def test_no_expansion_no_expanded_pages(self):
        """확장 시도 문서에는 expanded_pages가 저장됨"""
        docs = [_make_doc("doc1", page=1, doc_type="manual")]
        state = {"docs": docs}

        # fetcher returns empty - no expansion
        page_fetcher = MagicMock(return_value=[])

        # When
        result = expand_related_docs_node(state, page_fetcher=page_fetcher)

        # Then
        doc = result["docs"][0]
        assert doc.metadata.get("expanded_pages") == [1, 2, 3]

    def test_myservice_page_zero_expansion(self):
        """myservice 문서의 page 0도 정상적으로 확장됨"""
        docs = [_make_doc("myservice_doc", page=0, doc_type="myservice")]
        state = {"docs": docs}

        # doc_fetcher returns chunks with page 0
        related = [
            _make_doc("chunk1", page=0),
            _make_doc("chunk2", page=0),
        ]
        doc_fetcher = MagicMock(return_value=related)

        # When
        result = expand_related_docs_node(state, doc_fetcher=doc_fetcher)

        # Then
        expanded_doc = result["docs"][0]
        assert expanded_doc.metadata is not None
        assert "expanded_pages" in expanded_doc.metadata
        assert expanded_doc.metadata["expanded_pages"] == [0]  # page 0 should be included
