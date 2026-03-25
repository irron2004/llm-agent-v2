"""SOP 평가 결과 뷰어.

Usage:
    uv run streamlit run backend/scripts/eval_viewer.py
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

RESULTS_PATH = Path("data/eval_sop_results_variant.json")
RETEST_PATH = Path("data/eval_sop_results_failed_retest.json")
GOLD_MASTER_PATH = Path("data/paper_a/eval/query_gold_master_v0_6_generated_full.jsonl")
GOLD_MASTER_EVAL_PATH = Path("data/eval_gold_master_base_v1.json")
VPLUS_EVAL_DIR = Path("data/eval_vplus_retrieval")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {"summary": {}, "results": []}
    return json.loads(path.read_text(encoding="utf-8"))


def render_question_list(items: list[dict], key_prefix: str):
    """단일 결과 리스트를 렌더링."""
    if not items:
        st.info("항목이 없습니다.")
        return

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
        _render_single_result(r)


def _render_single_result(r: dict):
    """단일 결과 상세 렌더링."""
    cols = st.columns(4)
    cols[0].markdown(f"**문서검색**: {'✅' if r.get('retrieval_hit') else '❌'}")
    cols[1].markdown(f"**페이지**: {'✅' if r.get('page_hit') else '❌'}")
    cols[2].markdown(f"**답변**: {'❌ 실패' if r.get('answer_empty') else '✅ 성공'}")
    cols[3].markdown(f"**소요시간**: {r.get('elapsed', '?')}s")

    st.markdown("---")
    gt_col1, gt_col2 = st.columns(2)
    gt_col1.markdown(f"**정답 문서**: `{r.get('ground_truth_doc', '')}`")
    gt_col2.markdown(f"**정답 페이지**: `{r.get('ground_truth_pages', '')}`")

    matched = r.get("matched_doc_ids", [])
    if matched:
        unique_matched = list(dict.fromkeys(matched))
        st.markdown(f"**매칭된 문서**: {', '.join(f'`{d}`' for d in unique_matched[:5])}")
    matched_pages = r.get("matched_pages", [])
    if matched_pages:
        st.markdown(f"**매칭된 페이지**: {matched_pages}")

    if r.get("error"):
        st.error(f"에러: {r['error']}")

    st.markdown("---")
    st.markdown("### 답변")
    answer = r.get("answer", "(답변 없음)")
    if answer:
        st.markdown(answer)
    else:
        st.warning("답변이 저장되지 않았습니다.")

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

    with st.expander("전체 질문 텍스트"):
        st.text(r.get("query", ""))

    with st.expander("Raw JSON"):
        display_r = {k: v for k, v in r.items() if k != "answer"}
        st.json(display_r)


def page_baseline():
    """기존 평가 결과 페이지."""
    st.title("SOP Retrieval 평가 결과")

    if st.button("새로고침", key="refresh_baseline"):
        st.rerun()

    data = load_json(RESULTS_PATH)
    summary = data.get("summary", {})
    results = data.get("results", [])

    if not results:
        st.warning("평가 결과가 없습니다. 평가 스크립트 실행 후 새로고침하세요.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 질문", summary.get("total", len(results)))
    col2.metric("문서 검색 정확도", f"{summary.get('retrieval_accuracy', 0)}%",
                delta=f"{summary.get('retrieval_hits', 0)}/{summary.get('total', 0)}")
    col3.metric("페이지 적중률", f"{summary.get('page_accuracy', 0)}%",
                delta=f"{summary.get('page_hits', 0)}/{summary.get('total', 0)}")
    col4.metric("답변 생성 성공", f"{summary.get('answer_rate', 0)}%",
                delta=f"{summary.get('answer_ok', 0)}/{summary.get('total', 0)}")

    st.divider()

    failed = [r for r in results if not r.get("retrieval_hit") or r.get("answer_empty") or r.get("error")]
    success = [r for r in results if r.get("retrieval_hit") and not r.get("answer_empty") and not r.get("error")]

    tab_fail, tab_success, tab_all = st.tabs([
        f"실패 ({len(failed)})",
        f"성공 ({len(success)})",
        f"전체 ({len(results)})",
    ])

    with tab_fail:
        render_question_list(failed, "fail")
    with tab_success:
        render_question_list(success, "success")
    with tab_all:
        render_question_list(results, "all")


def page_improvement():
    """1차 개선 비교 페이지."""
    st.title("1차 개선 — 기존 vs 개선 비교")

    if st.button("새로고침", key="refresh_improve"):
        st.rerun()

    baseline_data = load_json(RESULTS_PATH)
    retest_data = load_json(RETEST_PATH)

    baseline_results = baseline_data.get("results", [])
    retest_results = retest_data.get("results", [])

    if not retest_results:
        st.warning("재테스트 결과가 없습니다. 재테스트 실행 후 새로고침하세요.")
        return

    # Build lookup: query -> baseline result
    baseline_map: dict[str, dict] = {}
    for r in baseline_results:
        baseline_map[r.get("query", "")] = r

    # 개선 요약 통계
    improved = 0
    unchanged = 0
    degraded = 0
    comparisons: list[dict] = []

    for retest in retest_results:
        query = retest.get("query", "")
        baseline = baseline_map.get(query, {})

        base_ok = baseline.get("retrieval_hit", False) and not baseline.get("answer_empty", True) and not baseline.get("error")
        new_ok = retest.get("retrieval_hit", False) and not retest.get("answer_empty", True) and not retest.get("error")

        if not base_ok and new_ok:
            status = "개선"
            improved += 1
        elif base_ok and not new_ok:
            status = "악화"
            degraded += 1
        elif not base_ok and not new_ok:
            status = "미해결"
            unchanged += 1
        else:
            status = "유지(성공)"
            improved += 1  # was already ok, still ok

        comparisons.append({
            "query": query,
            "status": status,
            "baseline": baseline,
            "retest": retest,
        })

    # Summary metrics
    total = len(comparisons)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("대상 질문", total)
    col2.metric("개선", improved, delta=f"+{improved}", delta_color="normal")
    col3.metric("미해결", unchanged)
    col4.metric("악화", degraded, delta=f"-{degraded}" if degraded else "0", delta_color="inverse")

    # Retest summary
    retest_summary = retest_data.get("summary", {})
    if retest_summary:
        st.divider()
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("재테스트 문서검색", f"{retest_summary.get('retrieval_accuracy', 0)}%")
        rc2.metric("재테스트 페이지적중", f"{retest_summary.get('page_accuracy', 0)}%")
        rc3.metric("재테스트 답변성공", f"{retest_summary.get('answer_rate', 0)}%")

    st.divider()

    # Filter tabs
    improved_items = [c for c in comparisons if c["status"] == "개선"]
    unresolved_items = [c for c in comparisons if c["status"] == "미해결"]
    degraded_items = [c for c in comparisons if c["status"] == "악화"]

    tab_improved, tab_unresolved, tab_degraded, tab_all = st.tabs([
        f"개선 ({len(improved_items)})",
        f"미해결 ({len(unresolved_items)})",
        f"악화 ({len(degraded_items)})",
        f"전체 ({total})",
    ])

    def render_comparison_list(items: list[dict], key_prefix: str):
        if not items:
            st.info("항목이 없습니다.")
            return

        options = [
            f"{c['status']} | {c['query'][:75]}..."
            for c in items
        ]
        selected_idx = st.selectbox(
            "질문 선택",
            range(len(options)),
            format_func=lambda i: options[i],
            key=f"{key_prefix}_select",
        )

        if selected_idx is not None:
            comp = items[selected_idx]
            baseline = comp["baseline"]
            retest = comp["retest"]

            # Status badge
            status_colors = {"개선": "🟢", "미해결": "🟡", "악화": "🔴", "유지(성공)": "🟢"}
            st.markdown(f"### {status_colors.get(comp['status'], '⚪')} {comp['status']}")

            # Ground truth
            gt_doc = retest.get("ground_truth_doc", "") or baseline.get("ground_truth_doc", "")
            gt_pages = retest.get("ground_truth_pages", "") or baseline.get("ground_truth_pages", "")
            st.markdown(f"**정답 문서**: `{gt_doc}` | **정답 페이지**: `{gt_pages}`")

            st.divider()

            # Side-by-side comparison
            left, right = st.columns(2)

            with left:
                st.markdown("### 기존 답변")
                if baseline:
                    bcols = st.columns(3)
                    bcols[0].markdown(f"**문서검색**: {'✅' if baseline.get('retrieval_hit') else '❌'}")
                    bcols[1].markdown(f"**페이지**: {'✅' if baseline.get('page_hit') else '❌'}")
                    bcols[2].markdown(f"**답변**: {'❌' if baseline.get('answer_empty') else '✅'}")

                    b_matched = baseline.get("matched_doc_ids", [])
                    if b_matched:
                        st.markdown(f"**매칭문서**: `{list(dict.fromkeys(b_matched))[:3]}`")

                    st.markdown("---")
                    b_answer = baseline.get("answer", "(답변 없음)")
                    if b_answer:
                        st.markdown(b_answer)
                    else:
                        st.warning("답변 없음")

                    b_judge = baseline.get("judge", {})
                    if b_judge:
                        st.caption(f"Judge: faithful={b_judge.get('faithful')} | {b_judge.get('hint', '')[:100]}")
                else:
                    st.warning("기존 결과 없음")

            with right:
                st.markdown("### 개선 답변")
                rcols = st.columns(3)
                rcols[0].markdown(f"**문서검색**: {'✅' if retest.get('retrieval_hit') else '❌'}")
                rcols[1].markdown(f"**페이지**: {'✅' if retest.get('page_hit') else '❌'}")
                rcols[2].markdown(f"**답변**: {'❌' if retest.get('answer_empty') else '✅'}")

                r_matched = retest.get("matched_doc_ids", [])
                if r_matched:
                    st.markdown(f"**매칭문서**: `{list(dict.fromkeys(r_matched))[:3]}`")

                st.markdown("---")
                r_answer = retest.get("answer", "(답변 없음)")
                if r_answer:
                    st.markdown(r_answer)
                else:
                    st.warning("답변 없음")

                r_judge = retest.get("judge", {})
                if r_judge:
                    st.caption(f"Judge: faithful={r_judge.get('faithful')} | {r_judge.get('hint', '')[:100]}")

            # Timing comparison
            st.divider()
            b_time = baseline.get("elapsed", 0) if baseline else 0
            r_time = retest.get("elapsed", 0)
            st.markdown(f"**소요시간**: 기존 {b_time}s → 개선 {r_time}s")

            # Raw JSON
            with st.expander("Raw JSON (기존)"):
                if baseline:
                    st.json({k: v for k, v in baseline.items() if k != "answer"})
            with st.expander("Raw JSON (개선)"):
                st.json({k: v for k, v in retest.items() if k != "answer"})

    with tab_improved:
        render_comparison_list(improved_items, "imp")
    with tab_unresolved:
        render_comparison_list(unresolved_items, "unres")
    with tab_degraded:
        render_comparison_list(degraded_items, "deg")
    with tab_all:
        render_comparison_list(comparisons, "all_comp")


def page_gold_master():
    """Gold Master 데이터 탐색 페이지."""
    st.title("Gold Master 데이터 탐색")

    if st.button("새로고침", key="refresh_gold"):
        st.rerun()

    if not GOLD_MASTER_PATH.exists():
        st.warning(f"파일 없음: {GOLD_MASTER_PATH}")
        return

    from collections import Counter

    data: list[dict] = []
    with open(GOLD_MASTER_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    if not data:
        st.warning("데이터가 비어 있습니다.")
        return

    # ── Summary metrics ──
    st.subheader("요약")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 질문", len(data))
    splits = Counter(r["split"] for r in data)
    c2.metric("test", splits.get("test", 0))
    c3.metric("dev", splits.get("dev", 0))
    intents = Counter(r["intent_primary"] for r in data)
    c4.metric("intent 종류", len(intents))

    # ── 분포 차트 ──
    st.divider()
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Intent 분포")
        intent_df = [{"intent": k, "count": v} for k, v in intents.most_common()]
        st.bar_chart(data=intent_df, x="intent", y="count")

    with chart_col2:
        st.subheader("Scope 분포")
        scopes = Counter(r["scope_observability"] for r in data)
        scope_df = [{"scope": k, "count": v} for k, v in scopes.most_common()]
        st.bar_chart(data=scope_df, x="scope", y="count")

    st.divider()
    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.subheader("설비(Device) 분포")
        devices = Counter(r["canonical_device_name"] for r in data)
        dev_df = [{"device": k, "count": v} for k, v in devices.most_common(15)]
        st.bar_chart(data=dev_df, x="device", y="count")

    with chart_col4:
        st.subheader("Doc Type 분포")
        doc_types: Counter = Counter()
        for r in data:
            for dt in r.get("preferred_doc_types", []):
                doc_types[dt] += 1
        dt_df = [{"doc_type": k, "count": v} for k, v in doc_types.most_common()]
        st.bar_chart(data=dt_df, x="doc_type", y="count")

    # ── Candidate relevance 분포 ──
    st.divider()
    st.subheader("Retrieved Candidates — Relevance 분포")
    rels: Counter = Counter()
    for r in data:
        for c in r.get("retrieved_candidates", []):
            rels[c.get("relevance", -1)] += 1
    rel_labels = {0: "무관(0)", 1: "부분관련(1)", 2: "정답(2)"}
    rel_df = [{"relevance": rel_labels.get(k, str(k)), "count": v} for k, v in sorted(rels.items())]
    st.bar_chart(data=rel_df, x="relevance", y="count")

    # ── 필터 & 개별 탐색 ──
    st.divider()
    st.subheader("개별 질문 탐색")

    # Filters
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    with filter_col1:
        f_split = st.selectbox("Split", ["전체"] + sorted(splits.keys()), key="gm_split")
    with filter_col2:
        f_intent = st.selectbox("Intent", ["전체"] + sorted(intents.keys()), key="gm_intent")
    with filter_col3:
        f_scope = st.selectbox("Scope", ["전체"] + sorted(scopes.keys()), key="gm_scope")
    with filter_col4:
        f_device = st.selectbox("Device", ["전체"] + sorted(devices.keys()), key="gm_device")

    filtered = data
    if f_split != "전체":
        filtered = [r for r in filtered if r["split"] == f_split]
    if f_intent != "전체":
        filtered = [r for r in filtered if r["intent_primary"] == f_intent]
    if f_scope != "전체":
        filtered = [r for r in filtered if r["scope_observability"] == f_scope]
    if f_device != "전체":
        filtered = [r for r in filtered if r["canonical_device_name"] == f_device]

    st.caption(f"필터 결과: {len(filtered)}건")

    if not filtered:
        st.info("필터 조건에 맞는 항목이 없습니다.")
        return

    # Question selector
    options = [
        f"[{r['q_id']}] {r['question'][:70]}..."
        for r in filtered
    ]
    selected_idx = st.selectbox(
        "질문 선택",
        range(len(options)),
        format_func=lambda i: options[i],
        key="gm_select",
    )

    if selected_idx is not None:
        r = filtered[selected_idx]

        # Basic info
        info_cols = st.columns(4)
        info_cols[0].markdown(f"**q_id**: `{r['q_id']}`")
        info_cols[1].markdown(f"**split**: `{r['split']}`")
        info_cols[2].markdown(f"**intent**: `{r['intent_primary']}`")
        info_cols[3].markdown(f"**scope**: `{r['scope_observability']}`")

        st.markdown("---")

        # Question
        st.markdown("### 질문")
        st.markdown(f"> {r['question']}")
        if r.get("question_masked") and r["question_masked"] != r["question"]:
            st.caption(f"마스킹: {r['question_masked']}")

        # Scope info
        st.markdown("---")
        scope_cols = st.columns(3)
        scope_cols[0].markdown(f"**Device**: `{r.get('canonical_device_name', '')}`")
        scope_cols[1].markdown(f"**Equip ID**: `{r.get('canonical_equip_id', '') or '-'}`")
        scope_cols[2].markdown(f"**Scope Level**: `{r.get('target_scope_level', '')}`")

        allowed_devs = r.get("allowed_devices", [])
        if allowed_devs:
            st.markdown(f"**Allowed Devices**: {', '.join(f'`{d}`' for d in allowed_devs)}")

        # Gold docs
        st.markdown("---")
        st.markdown("### 정답 문서")
        gold_ids = r.get("gold_doc_ids", [])
        gold_strict = r.get("gold_doc_ids_strict", [])

        if gold_ids:
            for gid in gold_ids:
                is_strict = gid in gold_strict
                label = " **(strict)**" if is_strict else ""
                st.markdown(f"- `{gid}`{label}")
        else:
            st.info("gold_doc_ids 없음")

        # Retrieved candidates
        candidates = r.get("retrieved_candidates", [])
        if candidates:
            st.markdown("---")
            st.markdown(f"### Retrieved Candidates ({len(candidates)}건)")

            for i, cand in enumerate(candidates):
                rel = cand.get("relevance", -1)
                rel_emoji = {0: "⬜", 1: "🟡", 2: "🟢"}.get(rel, "❓")
                doc_id = cand.get("doc_id", "")
                score = cand.get("score", 0)
                reason = cand.get("judge_reason", "")

                with st.expander(f"{rel_emoji} [{rel}] `{doc_id}` (score={score:.2f}) — {reason}"):
                    cand_cols = st.columns(3)
                    cand_cols[0].markdown(f"**Device**: `{cand.get('device_name', '')}`")
                    cand_cols[1].markdown(f"**Doc Type**: `{cand.get('doc_type', '')}`")
                    cand_cols[2].markdown(f"**Topic Overlap**: {cand.get('topic_overlap_count', 0)}")

                    snippet = cand.get("snippet", "")
                    if snippet:
                        st.code(snippet[:500], language="markdown")

        # Notes
        notes = r.get("notes", "")
        if notes:
            st.markdown("---")
            st.caption(f"Notes: {notes}")

        # Raw JSON
        with st.expander("Raw JSON"):
            st.json(r)


def page_gold_master_eval():
    """Gold Master base_v1 평가 결과 페이지."""
    st.title("Gold Master base_v1 — 평가 결과")

    if st.button("새로고침", key="refresh_gm_eval"):
        st.rerun()

    data = load_json(GOLD_MASTER_EVAL_PATH)
    summary = data.get("summary", {})
    type_stats = data.get("type_stats", {})
    results = data.get("results", [])

    if not results:
        st.warning("평가 결과가 없습니다. `uv run python -u scripts/eval_gold_master.py` 실행 후 새로고침하세요.")
        return

    from collections import Counter

    # ── Summary metrics (strict 기준) ──
    st.subheader("전체 요약 (strict 기준)")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("총 질문", summary.get("total", len(results)))
    col2.metric("문서 검색(strict)", f"{summary.get('retrieval_accuracy', 0)}%",
                delta=f"{summary.get('retrieval_hits', 0)}/{summary.get('total', 0)}")
    col3.metric("문서 검색(lenient)", f"{summary.get('retrieval_lenient_accuracy', 0)}%",
                delta=f"{summary.get('retrieval_hits_lenient', 0)}/{summary.get('total', 0)}")
    col4.metric("페이지 적중률", f"{summary.get('page_accuracy', 0)}%",
                delta=f"{summary.get('page_hits', 0)}/{summary.get('total', 0)}")
    col5.metric("답변 생성 성공", f"{summary.get('answer_rate', 0)}%",
                delta=f"{summary.get('answer_ok', 0)}/{summary.get('total', 0)}")

    # ── Per-type breakdown ──
    if type_stats:
        st.divider()
        st.subheader("문서 타입별 성능")
        type_rows = []
        for dtype, stats in sorted(type_stats.items(), key=lambda x: x[1]["total"], reverse=True):
            t = stats["total"]
            type_rows.append({
                "타입": dtype,
                "총 건수": t,
                "검색 성공": stats["retrieval_hit"],
                "검색 정확도": f"{stats['retrieval_hit']/t*100:.1f}%" if t else "0%",
                "답변 성공": stats["answer_ok"],
                "답변 성공률": f"{stats['answer_ok']/t*100:.1f}%" if t else "0%",
                "에러": stats["error"],
            })
        st.table(type_rows)

    # ── 분포 차트 ──
    st.divider()
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Intent별 검색 성공률")
        intent_stats: dict[str, dict[str, int]] = {}
        for r in results:
            intent = r.get("intent_primary", "unknown")
            if intent not in intent_stats:
                intent_stats[intent] = {"total": 0, "hit": 0, "answer_ok": 0}
            intent_stats[intent]["total"] += 1
            if r.get("retrieval_hit"):
                intent_stats[intent]["hit"] += 1
            if not r.get("answer_empty") and not r.get("error"):
                intent_stats[intent]["answer_ok"] += 1

        intent_df = [
            {
                "intent": k,
                "검색 정확도(%)": round(v["hit"] / v["total"] * 100, 1) if v["total"] else 0,
                "답변 성공률(%)": round(v["answer_ok"] / v["total"] * 100, 1) if v["total"] else 0,
            }
            for k, v in sorted(intent_stats.items(), key=lambda x: x[1]["total"], reverse=True)
        ]
        st.bar_chart(data=intent_df, x="intent", y=["검색 정확도(%)", "답변 성공률(%)"])

    with chart_col2:
        st.subheader("Device별 검색 성공률 (상위 15)")
        device_stats: dict[str, dict[str, int]] = {}
        for r in results:
            dev = r.get("canonical_device_name", "unknown")
            if dev not in device_stats:
                device_stats[dev] = {"total": 0, "hit": 0, "answer_ok": 0}
            device_stats[dev]["total"] += 1
            if r.get("retrieval_hit"):
                device_stats[dev]["hit"] += 1
            if not r.get("answer_empty") and not r.get("error"):
                device_stats[dev]["answer_ok"] += 1

        dev_df = [
            {
                "device": k,
                "검색 정확도(%)": round(v["hit"] / v["total"] * 100, 1) if v["total"] else 0,
                "답변 성공률(%)": round(v["answer_ok"] / v["total"] * 100, 1) if v["total"] else 0,
            }
            for k, v in sorted(device_stats.items(), key=lambda x: x[1]["total"], reverse=True)[:15]
        ]
        st.bar_chart(data=dev_df, x="device", y=["검색 정확도(%)", "답변 성공률(%)"])

    # ── 필터 & 개별 탐색 ──
    st.divider()
    st.subheader("개별 질문 탐색")

    # Filters
    all_types = sorted({dt for r in results for dt in r.get("preferred_doc_types", [])})
    all_intents = sorted({r.get("intent_primary", "") for r in results})
    all_devices = sorted({r.get("canonical_device_name", "") for r in results})

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        f_status = st.selectbox("상태", ["전체", "검색성공", "검색실패", "답변성공", "답변실패", "에러"], key="gme_status")
    with fc2:
        f_type = st.selectbox("Doc Type", ["전체"] + all_types, key="gme_type")
    with fc3:
        f_intent = st.selectbox("Intent", ["전체"] + all_intents, key="gme_intent")
    with fc4:
        f_device = st.selectbox("Device", ["전체"] + all_devices, key="gme_device")

    filtered = results
    if f_status == "검색성공":
        filtered = [r for r in filtered if r.get("retrieval_hit")]
    elif f_status == "검색실패":
        filtered = [r for r in filtered if not r.get("retrieval_hit")]
    elif f_status == "답변성공":
        filtered = [r for r in filtered if not r.get("answer_empty") and not r.get("error")]
    elif f_status == "답변실패":
        filtered = [r for r in filtered if r.get("answer_empty") and not r.get("error")]
    elif f_status == "에러":
        filtered = [r for r in filtered if r.get("error")]

    if f_type != "전체":
        filtered = [r for r in filtered if f_type in r.get("preferred_doc_types", [])]
    if f_intent != "전체":
        filtered = [r for r in filtered if r.get("intent_primary") == f_intent]
    if f_device != "전체":
        filtered = [r for r in filtered if r.get("canonical_device_name") == f_device]

    st.caption(f"필터 결과: {len(filtered)}건")

    if not filtered:
        st.info("필터 조건에 맞는 항목이 없습니다.")
        return

    options = []
    for r in filtered:
        r_mark = "O" if r.get("retrieval_hit") else "X"
        a_mark = "O" if (not r.get("answer_empty") and not r.get("error")) else "X"
        options.append(f"[{r_mark}/{a_mark}] [{r.get('q_id','')}] {r.get('query','')[:65]}...")

    selected_idx = st.selectbox(
        "질문 선택",
        range(len(options)),
        format_func=lambda i: options[i],
        key="gme_select",
    )

    if selected_idx is not None:
        r = filtered[selected_idx]

        # Status badges
        cols = st.columns(5)
        cols[0].markdown(f"**문서검색(strict)**: {'✅' if r.get('retrieval_hit') else '❌'}")
        cols[1].markdown(f"**문서검색(lenient)**: {'✅' if r.get('retrieval_hit_lenient') else '❌'}")
        cols[2].markdown(f"**페이지**: {'✅' if r.get('page_hit') else '❌'}")
        cols[3].markdown(f"**답변**: {'❌ 실패' if r.get('answer_empty') else '✅ 성공'}")
        cols[4].markdown(f"**소요시간**: {r.get('elapsed', '?')}s")

        st.markdown("---")

        # Metadata
        meta_cols = st.columns(4)
        meta_cols[0].markdown(f"**q_id**: `{r.get('q_id', '')}`")
        meta_cols[1].markdown(f"**intent**: `{r.get('intent_primary', '')}`")
        meta_cols[2].markdown(f"**doc_type**: `{', '.join(r.get('preferred_doc_types', []))}`")
        meta_cols[3].markdown(f"**device**: `{r.get('canonical_device_name', '')}`")

        st.markdown("---")

        # Gold docs
        st.markdown("### 정답 문서 (strict)")
        gold_strict = r.get("gold_doc_ids_strict", [])
        gold_all = r.get("gold_doc_ids", [])
        matched_gold = r.get("matched_gold_ids", [])

        if gold_strict:
            for gid in gold_strict:
                hit = "✅" if gid in matched_gold else "❌"
                st.markdown(f"- {hit} `{gid}`")
        else:
            st.info("gold_doc_ids_strict 없음")

        if gold_all and gold_all != gold_strict:
            with st.expander(f"전체 gold_doc_ids ({len(gold_all)}건)"):
                for gid in gold_all:
                    st.markdown(f"- `{gid}`")

        # Retrieved docs
        retrieved = r.get("all_retrieved_doc_ids", [])
        if retrieved:
            st.markdown("---")
            st.markdown(f"### 검색된 문서 ({len(retrieved)}건)")
            matched_docs = set(r.get("matched_doc_ids", []))
            for doc_id in retrieved:
                hit = "✅" if doc_id in matched_docs else "⬜"
                st.markdown(f"- {hit} `{doc_id}`")

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

        judge = r.get("judge", {})
        if judge:
            st.markdown("---")
            st.markdown("### Judge 판정")
            faithful = judge.get("faithful", False)
            st.markdown(f"**Faithful**: {'✅ Yes' if faithful else '❌ No'}")
            hint = judge.get("hint", "")
            if hint:
                st.markdown(f"**Hint**: {hint}")

        with st.expander("Raw JSON"):
            display_r = {k: v for k, v in r.items() if k != "answer"}
            st.json(display_r)


def _load_vplus_results() -> list[dict]:
    """모든 variant의 결과를 통합 로드."""
    results: list[dict] = []
    for p in sorted(VPLUS_EVAL_DIR.glob("results_*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    return results


def _classify_work_type(gold_doc: str) -> str:
    gold = gold_doc.lower()
    if "_rep_" in gold:
        return "REP (교체)"
    elif "_adj_" in gold:
        return "ADJ (조정)"
    elif "_cln_" in gold:
        return "CLN (세정)"
    elif "_sw_" in gold:
        return "SW"
    elif "_modify_" in gold:
        return "MODIFY"
    return "기타"


def _classify_module(gold_doc: str) -> str:
    gold = gold_doc.lower()
    for mod in ["efem", "pm", "tm", "sub_unit", "rack", "load_port"]:
        if f"_{mod}_" in gold:
            return mod.upper().replace("_", " ")
    return "기타"


def page_vplus_eval():
    """Vplus SOP 검색 평가 결과 페이지 (전체 variant 통합)."""
    st.title("Vplus SOP 검색 평가")

    if st.button("새로고침", key="refresh_vplus"):
        st.rerun()

    results = _load_vplus_results()

    if not results:
        st.warning("평가 결과가 없습니다. `uv run python -u scripts/eval_vplus_retrieval.py --variant all` 실행 후 새로고침하세요.")
        return

    # CSV에서 전체 질문 수 로드
    import csv as _csv
    csv_path = Path("data/eval_sop_question_list_vplus.csv")
    total_questions = 0
    if csv_path.exists():
        with open(csv_path, encoding="utf-8-sig") as f:
            total_questions = sum(1 for _ in _csv.DictReader(f))

    # ── Variant별 요약 ──
    from collections import Counter, defaultdict

    variant_stats: dict[str, dict[str, int]] = {}
    for r in results:
        v = r.get("variant", "unknown")
        if v not in variant_stats:
            variant_stats[v] = {"total": 0, "doc_hit": 0, "page_hit": 0, "error": 0}
        variant_stats[v]["total"] += 1
        if r.get("hit_doc"):
            variant_stats[v]["doc_hit"] += 1
        if r.get("hit_page"):
            variant_stats[v]["page_hit"] += 1
        if r.get("error"):
            variant_stats[v]["error"] += 1

    completed = len(results)
    doc_hits = sum(1 for r in results if r.get("hit_doc"))
    page_hits = sum(1 for r in results if r.get("hit_page"))
    errors = sum(1 for r in results if r.get("error"))

    st.subheader("전체 요약")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("총 질문", total_questions or completed)
    col2.metric("완료", f"{completed}/{total_questions or completed}")
    col3.metric("문서 검색", f"{doc_hits/completed*100:.1f}%" if completed else "0%",
                delta=f"{doc_hits}/{completed}")
    col4.metric("페이지 적중", f"{page_hits/completed*100:.1f}%" if completed else "0%",
                delta=f"{page_hits}/{completed}")
    col5.metric("에러(스킵)", errors)

    # Variant별 비교 테이블
    if len(variant_stats) > 1:
        st.divider()
        st.subheader("Variant별 성능 비교")
        var_rows = []
        for v in ["original", "sim1", "sim2", "sim3"]:
            if v not in variant_stats:
                continue
            s = variant_stats[v]
            t = s["total"]
            var_rows.append({
                "Variant": v,
                "건수": t,
                "문서 검색": f"{s['doc_hit']}/{t} ({s['doc_hit']/t*100:.1f}%)" if t else "-",
                "페이지 적중": f"{s['page_hit']}/{t} ({s['page_hit']/t*100:.1f}%)" if t else "-",
                "에러": s["error"],
            })
        st.table(var_rows)

    # ── 작업 타입별 성능 ──
    st.divider()
    st.subheader("작업 타입별 성능")
    type_stats: dict[str, dict[str, int]] = {}
    for r in results:
        wtype = _classify_work_type(r.get("gold_doc", ""))
        if wtype not in type_stats:
            type_stats[wtype] = {"total": 0, "doc_hit": 0, "page_hit": 0}
        type_stats[wtype]["total"] += 1
        if r.get("hit_doc"):
            type_stats[wtype]["doc_hit"] += 1
        if r.get("hit_page"):
            type_stats[wtype]["page_hit"] += 1

    type_rows = []
    for wtype, stats in sorted(type_stats.items(), key=lambda x: x[1]["total"], reverse=True):
        t = stats["total"]
        type_rows.append({
            "타입": wtype,
            "건수": t,
            "문서 검색": f"{stats['doc_hit']}/{t} ({stats['doc_hit']/t*100:.1f}%)",
            "페이지 적중": f"{stats['page_hit']}/{t} ({stats['page_hit']/t*100:.1f}%)",
        })
    st.table(type_rows)

    # ── 모듈별 성능 차트 ──
    st.divider()
    st.subheader("모듈별 성능")
    mod_stats: dict[str, dict[str, int]] = {}
    for r in results:
        mod_label = _classify_module(r.get("gold_doc", ""))
        if mod_label not in mod_stats:
            mod_stats[mod_label] = {"total": 0, "doc_hit": 0, "page_hit": 0}
        mod_stats[mod_label]["total"] += 1
        if r.get("hit_doc"):
            mod_stats[mod_label]["doc_hit"] += 1
        if r.get("hit_page"):
            mod_stats[mod_label]["page_hit"] += 1

    mod_df = [
        {
            "모듈": k,
            "문서 검색(%)": round(v["doc_hit"] / v["total"] * 100, 1) if v["total"] else 0,
            "페이지 적중(%)": round(v["page_hit"] / v["total"] * 100, 1) if v["total"] else 0,
        }
        for k, v in sorted(mod_stats.items(), key=lambda x: x[1]["total"], reverse=True)
    ]
    st.bar_chart(data=mod_df, x="모듈", y=["문서 검색(%)", "페이지 적중(%)"])

    # ── 개별 질문 탐색 ──
    st.divider()
    st.subheader("개별 질문 탐색")

    available_variants = sorted({r.get("variant", "") for r in results})

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        f_status = st.selectbox("상태", ["전체", "검색 성공", "검색 실패", "에러"], key="vplus_status")
    with fc2:
        f_type = st.selectbox("작업 타입", ["전체"] + sorted(type_stats.keys()), key="vplus_type_filter")
    with fc3:
        f_variant = st.selectbox("Variant", ["전체"] + available_variants, key="vplus_variant_filter")

    filtered = results
    if f_status == "검색 성공":
        filtered = [r for r in filtered if r.get("hit_doc")]
    elif f_status == "검색 실패":
        filtered = [r for r in filtered if not r.get("hit_doc") and not r.get("error")]
    elif f_status == "에러":
        filtered = [r for r in filtered if r.get("error")]

    if f_type != "전체":
        type_key = f_type.split(" ")[0].lower()
        filtered = [r for r in filtered if f"_{type_key}_" in r.get("gold_doc", "").lower()]

    if f_variant != "전체":
        filtered = [r for r in filtered if r.get("variant") == f_variant]

    st.caption(f"필터 결과: {len(filtered)}건")

    if not filtered:
        st.info("필터 조건에 맞는 항목이 없습니다.")
        return

    options = []
    for r in filtered:
        d_mark = "O" if r.get("hit_doc") else "X"
        p_mark = "O" if r.get("hit_page") else "X"
        rank_str = f"@{r['hit_rank']}" if r.get("hit_rank") else ""
        v_label = f"[{r.get('variant', '')}]" if r.get("variant") else ""
        options.append(f"[{d_mark}/{p_mark}] {rank_str} {v_label} {r.get('question', '')[:60]}...")

    selected_idx = st.selectbox(
        "질문 선택",
        range(len(options)),
        format_func=lambda i: options[i],
        key="vplus_select",
    )

    if selected_idx is not None:
        r = filtered[selected_idx]

        # Status
        cols = st.columns(5)
        cols[0].markdown(f"**문서 검색**: {'✅' if r.get('hit_doc') else '❌'}")
        cols[1].markdown(f"**페이지 적중**: {'✅' if r.get('hit_page') else '❌'}")
        cols[2].markdown(f"**검색 순위**: {r.get('hit_rank', '-')}")
        cols[3].markdown(f"**Variant**: `{r.get('variant', '')}`")
        cols[4].markdown(f"**소요시간**: {r.get('elapsed', '?')}s")

        st.markdown("---")

        # Gold doc info
        gc1, gc2 = st.columns(2)
        gc1.markdown(f"**정답 문서**: `{r.get('gold_doc', '')}`")
        gc2.markdown(f"**정답 페이지**: `{r.get('gold_pages', '')}`")

        if r.get("matched_pages"):
            st.markdown(f"**매칭된 페이지**: {r['matched_pages']}")

        if r.get("error"):
            st.error(f"에러: {r['error']}")

        # Retrieved docs
        retrieved = r.get("retrieved_doc_ids", [])
        if retrieved:
            st.markdown("---")
            st.markdown(f"### 검색된 문서 (상위 {len(retrieved)}건)")
            import re as _re
            gold_norm = _re.sub(r"\.(pdf|pptx)$", "", r.get("gold_doc", "").lower())
            gold_norm = _re.sub(r"[^a-z0-9]+", "_", gold_norm).strip("_")
            for i, doc_id in enumerate(retrieved, 1):
                doc_norm = _re.sub(r"[^a-z0-9]+", "_", doc_id.lower()).strip("_")
                hit = "✅" if gold_norm and gold_norm in doc_norm else "⬜"
                st.markdown(f"{i}. {hit} `{doc_id}`")

        # Answer
        answer = r.get("answer", "")
        if answer:
            st.markdown("---")
            st.markdown("### 답변 (일부)")
            st.markdown(answer)

        # Raw JSON
        with st.expander("Raw JSON"):
            display_r = {k: v for k, v in r.items() if k != "answer"}
            st.json(display_r)


def main():
    st.set_page_config(page_title="SOP 평가 결과", layout="wide")

    page = st.sidebar.radio(
        "페이지",
        ["기존 평가", "1차 개선", "Gold Master", "Gold Master base_v1", "Vplus 검색 평가"],
        index=0,
    )

    if page == "기존 평가":
        page_baseline()
    elif page == "1차 개선":
        page_improvement()
    elif page == "Gold Master":
        page_gold_master()
    elif page == "Gold Master base_v1":
        page_gold_master_eval()
    elif page == "Vplus 검색 평가":
        page_vplus_eval()


if __name__ == "__main__":
    main()
