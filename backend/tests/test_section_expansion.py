"""Section extraction + expander 통합 테스트.

Tests:
- SOP section extraction (TOC parsing, header matching, carry-forward)
- TS section extraction (alpha TOC, X-N. subsection)
- PEMS no-op extraction
- SectionExpander (expand, ordering, dedup, settings)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path for scripts.chunk_v3 imports
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest
from unittest.mock import MagicMock

from scripts.chunk_v3.section_extractor import SectionInfo, extract_sections
from backend.llm_infrastructure.retrieval.postprocessors.section_expander import (
    ExpandedGroup,
    SectionExpandResult,
    SectionExpander,
)
from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchHit


# =========================================================================
# Helpers
# =========================================================================

def _make_page(page_no: int, text: str) -> dict:
    return {"page": page_no, "text": text}


def _make_hit(
    doc_id: str = "doc1",
    chunk_id: str = "chunk1",
    content: str = "test",
    score: float = 1.0,
    page: int = 1,
    section_chapter: str = "",
    chapter_source: str = "none",
    chapter_ok: bool = False,
) -> EsSearchHit:
    return EsSearchHit(
        doc_id=doc_id,
        chunk_id=chunk_id,
        content=content,
        score=score,
        page=page,
        metadata={
            "section_chapter": section_chapter,
            "chapter_source": chapter_source,
            "chapter_ok": chapter_ok,
        },
    )


# =========================================================================
# SOP Section Extraction Tests
# =========================================================================

class TestSopSectionExtraction:
    """SOP 문서의 섹션 추출 테스트."""

    def test_basic_toc_and_headers(self):
        """TOC 파싱 후 ## N. Title 매칭."""
        pages = [
            _make_page(1, "```markdown\n# SOP Title\n## Contents\n1. Safety\n2. Preparation\n3. Procedure\n```"),
            _make_page(2, "```markdown\n## 1. Safety\nSafety instructions here.\n```"),
            _make_page(3, "```markdown\nMore safety details.\n```"),
            _make_page(4, "```markdown\n## 2. Preparation\nPrepare tools.\n```"),
            _make_page(5, "```markdown\n## 3. Procedure\nStep by step.\n```"),
        ]
        result = extract_sections(pages, "sop")
        assert len(result) == 5

        # Page 1 (TOC) should be skipped
        assert result[0].chapter_ok is False
        assert result[0].section_chapter == ""

        # Page 2: "1. Safety" matched via TOC
        assert result[1].section_chapter == "1. Safety"
        assert result[1].section_number == 1
        assert result[1].chapter_source == "toc_match"
        assert result[1].chapter_ok is True

        # Page 3: carry-forward from "1. Safety"
        assert result[2].section_chapter == "1. Safety"
        assert result[2].chapter_source == "carry"
        assert result[2].chapter_ok is True

        # Page 4: "2. Preparation"
        assert result[3].section_chapter == "2. Preparation"
        assert result[3].section_number == 2

        # Page 5: "3. Procedure"
        assert result[4].section_chapter == "3. Procedure"

    def test_pre_toc_pages_skipped(self):
        """TOC 이전 페이지는 섹션 할당 안 됨."""
        pages = [
            _make_page(1, "```markdown\n# Title\n## Scope\nScope text.\n```"),
            _make_page(2, "```markdown\n## Contents\n1. Safety\n2. Work\n```"),
            _make_page(3, "```markdown\n## 1. Safety\nSafety text.\n```"),
        ]
        result = extract_sections(pages, "sop")
        # Pages 0, 1 (before and including TOC) should be skipped
        assert result[0].chapter_ok is False
        assert result[1].chapter_ok is False
        # Page 2: matched
        assert result[2].section_chapter == "1. Safety"
        assert result[2].chapter_ok is True

    def test_non_toc_header_filtered(self):
        """TOC에 없는 번호의 헤더는 무시."""
        pages = [
            _make_page(1, "```markdown\n## Contents\n1. Safety\n3. Procedure\n```"),
            _make_page(2, "```markdown\n## 1. Safety\nSafety.\n```"),
            _make_page(3, "```markdown\n## 2. Something Not In TOC\nShould be ignored.\n```"),
            _make_page(4, "```markdown\n## 3. Procedure\nProcedure.\n```"),
        ]
        result = extract_sections(pages, "sop")
        # Page 2 (idx 1): 1. Safety
        assert result[1].section_chapter == "1. Safety"
        # Page 3 (idx 2): Header #2 is NOT in TOC -> carry from Safety
        assert result[2].section_chapter == "1. Safety"
        assert result[2].chapter_source == "carry"
        # Page 4 (idx 3): 3. Procedure
        assert result[3].section_chapter == "3. Procedure"

    def test_no_toc_fallback(self):
        """TOC 없으면 헤더만으로 추출."""
        pages = [
            _make_page(1, "```markdown\n# SOP Title\nScope.\n```"),
            _make_page(2, "```markdown\n## 1. Safety\nSafety.\n```"),
            _make_page(3, "```markdown\n## 2. Work\nWork.\n```"),
        ]
        result = extract_sections(pages, "sop")
        # No TOC -> headers accepted as 'title' source
        assert result[1].section_chapter == "1. Safety"
        assert result[1].chapter_source == "title"
        assert result[2].section_chapter == "2. Work"


# =========================================================================
# TS Section Extraction Tests
# =========================================================================

class TestTsSectionExtraction:
    """TS 문서의 섹션 추출 테스트."""

    def test_alpha_toc_and_subsections(self):
        """Alpha TOC 파싱 후 X-N. 서브섹션 매칭."""
        pages = [
            _make_page(1, "```markdown\n# TS Guide\n- A. FFU Pressure error\n- B. FFU fan error\n- C. Communication error\n```"),
            _make_page(2, "```markdown\n## A-1. Side door check\nCheck door.\n```"),
            _make_page(3, "```markdown\n## B-1. Fan status\nCheck fan.\n```"),
        ]
        result = extract_sections(pages, "ts")
        assert len(result) == 3

        # Page 1 (TOC) skipped
        assert result[0].chapter_ok is False

        # Page 2: A-1 maps to "A. FFU Pressure error"
        assert result[1].section_chapter == "A. FFU Pressure error"
        assert result[1].chapter_source == "toc_match"
        assert result[1].chapter_ok is True

        # Page 3: B-1 maps to "B. FFU fan error"
        assert result[2].section_chapter == "B. FFU fan error"

    def test_ts_carry_forward(self):
        """TS 서브섹션 없는 페이지는 carry-forward."""
        pages = [
            _make_page(1, "```markdown\n- A. Topic A\n- B. Topic B\n```"),
            _make_page(2, "```markdown\n## A-1. Check\nDetails.\n```"),
            _make_page(3, "```markdown\nMore details about A.\n```"),
            _make_page(4, "```markdown\n## B-1. Check B\nDetails B.\n```"),
        ]
        result = extract_sections(pages, "ts")
        assert result[2].section_chapter == "A. Topic A"
        assert result[2].chapter_source == "carry"
        assert result[3].section_chapter == "B. Topic B"


# =========================================================================
# PEMS (No-op) Section Extraction Tests
# =========================================================================

class TestPemsSectionExtraction:
    """PEMS 문서는 섹션 추출 없음."""

    def test_noop_all_empty(self):
        pages = [_make_page(1, "text"), _make_page(2, "text")]
        result = extract_sections(pages, "pems")
        assert len(result) == 2
        assert all(r.section_chapter == "" for r in result)
        assert all(r.chapter_ok is False for r in result)

    def test_unknown_doc_type(self):
        pages = [_make_page(1, "text")]
        result = extract_sections(pages, "unknown_type")
        assert len(result) == 1
        assert result[0].chapter_ok is False


# =========================================================================
# SectionExpander Tests
# =========================================================================

class TestSectionExpander:
    """SectionExpander 후처리기 테스트."""

    def _make_mock_engine(self, section_chunks_map: dict[tuple[str, str], list[EsSearchHit]]):
        """Mock ES engine that returns predetermined section chunks."""
        engine = MagicMock()

        def fetch_side_effect(doc_id, section_chapter, max_pages=8, content_index=None):
            key = (doc_id, section_chapter)
            return section_chunks_map.get(key, [])

        engine.fetch_section_chunks.side_effect = fetch_side_effect
        return engine

    def test_basic_expansion(self):
        """기본 섹션 확장."""
        hits = [
            _make_hit("doc1", "c1", score=5.0, page=3,
                       section_chapter="1. Safety", chapter_source="toc_match", chapter_ok=True),
            _make_hit("doc2", "c2", score=3.0, page=1),
        ]
        section_chunks = [
            _make_hit("doc1", "c1_p2", page=2),
            _make_hit("doc1", "c1", page=3),
            _make_hit("doc1", "c1_p4", page=4),
        ]
        engine = self._make_mock_engine({("doc1", "1. Safety"): section_chunks})

        expander = SectionExpander(enabled=True, top_groups=2, max_pages=8)
        result = expander.expand(hits, engine)

        assert len(result.expanded_groups) == 1
        assert result.expanded_groups[0].doc_id == "doc1"
        assert result.expanded_groups[0].section_chapter == "1. Safety"

    def test_disabled(self):
        """비활성화시 원본 반환."""
        hits = [_make_hit("doc1", "c1", section_chapter="1. Safety",
                          chapter_source="toc_match", chapter_ok=True)]
        engine = MagicMock()

        expander = SectionExpander(enabled=False)
        result = expander.expand(hits, engine)

        assert result.expanded_groups == []
        assert result.original_hits == hits
        engine.fetch_section_chunks.assert_not_called()

    def test_empty_hits(self):
        """빈 결과."""
        engine = MagicMock()
        expander = SectionExpander(enabled=True)
        result = expander.expand([], engine)
        assert result.expanded_groups == []

    def test_top_groups_limit(self):
        """top_groups 제한."""
        hits = [
            _make_hit("doc1", "c1", score=5.0, section_chapter="1. Safety",
                       chapter_source="toc_match", chapter_ok=True),
            _make_hit("doc2", "c2", score=4.0, section_chapter="A. Topic",
                       chapter_source="toc_match", chapter_ok=True),
            _make_hit("doc3", "c3", score=3.0, section_chapter="2. Work",
                       chapter_source="toc_match", chapter_ok=True),
        ]
        engine = self._make_mock_engine({
            ("doc1", "1. Safety"): [_make_hit("doc1", "c1_exp", page=1)],
            ("doc2", "A. Topic"): [_make_hit("doc2", "c2_exp", page=1)],
            ("doc3", "2. Work"): [_make_hit("doc3", "c3_exp", page=1)],
        })

        expander = SectionExpander(enabled=True, top_groups=2)
        result = expander.expand(hits, engine)
        assert len(result.expanded_groups) == 2

    def test_source_filter(self):
        """allowed_sources 필터링."""
        hits = [
            _make_hit("doc1", "c1", section_chapter="1. Safety",
                       chapter_source="carry", chapter_ok=True),
        ]
        engine = MagicMock()

        expander = SectionExpander(
            enabled=True,
            allowed_sources={"toc_match", "title"},  # "carry" not allowed
        )
        result = expander.expand(hits, engine)
        assert len(result.expanded_groups) == 0
        engine.fetch_section_chunks.assert_not_called()

    def test_dedup_same_group(self):
        """같은 (doc_id, section_chapter) 그룹은 한 번만 확장."""
        hits = [
            _make_hit("doc1", "c1", score=5.0, page=2,
                       section_chapter="1. Safety", chapter_source="toc_match", chapter_ok=True),
            _make_hit("doc1", "c2", score=4.0, page=3,
                       section_chapter="1. Safety", chapter_source="toc_match", chapter_ok=True),
        ]
        engine = self._make_mock_engine({
            ("doc1", "1. Safety"): [
                _make_hit("doc1", "c1", page=2),
                _make_hit("doc1", "c2", page=3),
            ],
        })

        expander = SectionExpander(enabled=True, top_groups=5)
        result = expander.expand(hits, engine)
        assert len(result.expanded_groups) == 1

    def test_from_settings(self):
        """from_settings 클래스메서드."""
        settings = MagicMock()
        settings.section_expand_enabled = True
        settings.section_expand_top_groups = 3
        settings.section_expand_max_pages = 10
        settings.section_expand_allowed_sources = "toc_match,title,rule"

        expander = SectionExpander.from_settings(settings)
        assert expander.enabled is True
        assert expander.top_groups == 3
        assert expander.max_pages == 10
        assert expander.allowed_sources == {"toc_match", "title", "rule"}

    def test_ranking_order_preserved(self):
        """all_results_ordered에서 원래 순위 유지."""
        hits = [
            _make_hit("doc1", "c1", score=5.0, page=3,
                       section_chapter="1. Safety", chapter_source="toc_match", chapter_ok=True),
            _make_hit("doc2", "c5", score=4.0, page=1),
            _make_hit("doc3", "c6", score=3.0, page=2),
        ]
        section_chunks = [
            _make_hit("doc1", "c1_p2", page=2),
            _make_hit("doc1", "c1", page=3),
            _make_hit("doc1", "c1_p4", page=4),
        ]
        engine = self._make_mock_engine({("doc1", "1. Safety"): section_chunks})

        expander = SectionExpander(enabled=True, top_groups=2)
        result = expander.expand(hits, engine)
        ordered = result.all_results_ordered()

        # First should be expanded group chunks (c1_p2, c1, c1_p4)
        # Then unexpanded hits (c5, c6)
        chunk_ids = [h.chunk_id for h in ordered]
        assert "c1_p2" in chunk_ids
        assert "c1" in chunk_ids
        assert "c1_p4" in chunk_ids
        assert "c5" in chunk_ids
        assert "c6" in chunk_ids

        # c1 group comes before c5 and c6
        c1_idx = chunk_ids.index("c1")
        c5_idx = chunk_ids.index("c5")
        c6_idx = chunk_ids.index("c6")
        assert c1_idx < c5_idx
        assert c5_idx < c6_idx

        # No duplicates
        assert len(chunk_ids) == len(set(chunk_ids))
