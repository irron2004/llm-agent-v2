"""SOP 평가 결과 뷰어.

Usage:
    uv run streamlit run backend/scripts/eval_viewer.py
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

RESULTS_PATH = Path("data/eval_sop_results_variant.json")


def load_results() -> dict:
    if not RESULTS_PATH.exists():
        return {"summary": {}, "results": []}
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def main():
    st.set_page_config(page_title="SOP 평가 결과", layout="wide")
    st.title("SOP Retrieval 평가 결과")

    # Reload button
    if st.button("새로고침"):
        st.rerun()

    data = load_results()
    summary = data.get("summary", {})
    results = data.get("results", [])

    if not results:
        st.warning("평가 결과가 없습니다. 평가 스크립트 실행 후 새로고침하세요.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 질문", summary.get("total", len(results)))
    col2.metric("문서 검색 정확도", f"{summary.get('retrieval_accuracy', 0)}%",
                delta=f"{summary.get('retrieval_hits', 0)}/{summary.get('total', 0)}")
    col3.metric("페이지 적중률", f"{summary.get('page_accuracy', 0)}%",
                delta=f"{summary.get('page_hits', 0)}/{summary.get('total', 0)}")
    col4.metric("답변 생성 성공", f"{summary.get('answer_rate', 0)}%",
                delta=f"{summary.get('answer_ok', 0)}/{summary.get('total', 0)}")

    st.divider()

    # Split into success / fail
    failed = [r for r in results if not r.get("retrieval_hit") or r.get("answer_empty") or r.get("error")]
    success = [r for r in results if r.get("retrieval_hit") and not r.get("answer_empty") and not r.get("error")]

    tab_fail, tab_success, tab_all = st.tabs([
        f"실패 ({len(failed)})",
        f"성공 ({len(success)})",
        f"전체 ({len(results)})",
    ])

    def render_question_list(items: list[dict], key_prefix: str):
        if not items:
            st.info("항목이 없습니다.")
            return

        # Question selector
        options = [
            f"{'X' if not r.get('retrieval_hit') else 'O'} | "
            f"{r.get('query', '')[:80]}..."
            for r in items
        ]
        selected_idx = st.selectbox(
            "질문 선택",
            range(len(options)),
            format_func=lambda i: options[i],
            key=f"{key_prefix}_select",
        )

        if selected_idx is not None:
            r = items[selected_idx]

            # Status badges
            cols = st.columns(4)
            cols[0].markdown(
                f"**문서검색**: {'✅' if r.get('retrieval_hit') else '❌'}"
            )
            cols[1].markdown(
                f"**페이지**: {'✅' if r.get('page_hit') else '❌'}"
            )
            cols[2].markdown(
                f"**답변**: {'❌ 실패' if r.get('answer_empty') else '✅ 성공'}"
            )
            cols[3].markdown(f"**소요시간**: {r.get('elapsed', '?')}s")

            # Ground truth
            st.markdown("---")
            gt_col1, gt_col2 = st.columns(2)
            gt_col1.markdown(f"**정답 문서**: `{r.get('ground_truth_doc', '')}`")
            gt_col2.markdown(f"**정답 페이지**: `{r.get('ground_truth_pages', '')}`")

            # Matched docs
            matched = r.get("matched_doc_ids", [])
            if matched:
                unique_matched = list(dict.fromkeys(matched))
                st.markdown(f"**매칭된 문서**: {', '.join(f'`{d}`' for d in unique_matched[:5])}")
            matched_pages = r.get("matched_pages", [])
            if matched_pages:
                st.markdown(f"**매칭된 페이지**: {matched_pages}")

            # Error
            if r.get("error"):
                st.error(f"에러: {r['error']}")

            # Answer
            st.markdown("---")
            st.markdown("### 답변")
            answer = r.get("answer", "(답변 없음)")
            if answer:
                st.markdown(answer)
            else:
                st.warning("답변이 저장되지 않았습니다.")

            # Judge
            judge = r.get("judge", {})
            if judge:
                st.markdown("---")
                st.markdown("### Judge 판정")
                faithful = judge.get("faithful", False)
                st.markdown(f"**Faithful**: {'✅ Yes' if faithful else '❌ No'}")
                issues = judge.get("issues", [])
                if issues:
                    st.markdown(f"**Issues**: {', '.join(str(i) for i in issues)}")
                hint = judge.get("hint", "")
                if hint:
                    st.markdown(f"**Hint**: {hint}")

            # Full query
            with st.expander("전체 질문 텍스트"):
                st.text(r.get("query", ""))

            # Raw JSON
            with st.expander("Raw JSON"):
                display_r = {k: v for k, v in r.items() if k != "answer"}
                st.json(display_r)

    with tab_fail:
        render_question_list(failed, "fail")

    with tab_success:
        render_question_list(success, "success")

    with tab_all:
        render_question_list(results, "all")


if __name__ == "__main__":
    main()
