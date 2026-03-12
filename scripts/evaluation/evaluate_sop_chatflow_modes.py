#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from backend.api.dependencies import (
    get_default_llm,
    get_prompt_spec_cached,
    get_search_service,
)
from backend.api.main import _configure_search_service
from backend.domain.doc_type_mapping import expand_doc_type_selection
from backend.llm_infrastructure.llm.langgraph_agent import (
    PromptSpec,
    _detect_language_rule_based,
)
from backend.services.agents.langgraph_rag_agent import LangGraphRAGAgent
from backend.services.search_service import SearchService


@dataclass(frozen=True)
class QueryCase:
    qid: str
    base_idx: int
    variant: str
    question: str
    device: str
    gold_doc: str
    gold_pages: str
    note: str
    source_language: str


@dataclass
class RetrievalEval:
    qid: str
    variant: str
    mode: str
    question: str
    route: str | None
    search_queries: list[str]
    gold_doc: str
    gold_pages: str
    hit_doc_at_10: bool
    hit_page_at_10: bool
    elapsed_ms: float
    top_docs: list[dict[str, Any]]


@dataclass
class AnswerEval:
    qid: str
    variant: str
    mode: str
    question: str
    route: str | None
    search_queries: list[str]
    gold_doc: str
    gold_pages: str
    hit_doc_at_10: bool
    hit_page_at_10: bool
    elapsed_ms: float
    top_docs: list[dict[str, Any]]
    answer: str
    judge: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SOP chat-flow modes")
    parser.add_argument(
        "--csv-path",
        default="data/eval_sop_question_list_박진우_변형.csv",
        help="Base CSV path",
    )
    parser.add_argument(
        "--variants",
        choices=("original", "all"),
        default="original",
        help="Evaluate only 질문내용 or all variants",
    )
    parser.add_argument(
        "--out-dir",
        default="data/eval_results/sop_chatflow_modes_20260311",
        help="Output directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick runs",
    )
    parser.add_argument(
        "--answer-samples",
        type=int,
        default=8,
        help="Number of representative queries for full answer generation",
    )
    return parser.parse_args()


def _normalize(text: str) -> str:
    return text.lower().strip().replace(" ", "_").replace("-", "_").replace("/", "_")


def _parse_pages(raw: str) -> set[int]:
    pages: set[int] = set()
    for token in (raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            try:
                start_s, end_s = token.split("-", 1)
                start, end = int(start_s), int(end_s)
            except ValueError:
                continue
            for page in range(start, end + 1):
                pages.add(page)
            continue
        try:
            pages.add(int(token))
        except ValueError:
            continue
    return pages


def _doc_matches(gold_doc: str, doc_id: str) -> bool:
    gold = _normalize(gold_doc)
    candidate = _normalize(doc_id)
    return gold in candidate or candidate in gold


def _compute_hits(
    gold_doc: str, gold_pages: str, docs: list[dict[str, Any]]
) -> tuple[bool, bool]:
    gold_page_set = _parse_pages(gold_pages)
    hit_doc = False
    hit_page = False
    for doc in docs[:10]:
        doc_id = str(doc.get("doc_id") or "")
        if not _doc_matches(gold_doc, doc_id):
            continue
        hit_doc = True
        page = doc.get("page")
        if page is None:
            continue
        try:
            page_int = int(page)
        except (TypeError, ValueError):
            continue
        if page_int in gold_page_set:
            hit_page = True
    return hit_doc, hit_page


def _summarize_docs(result_docs: list[Any]) -> list[dict[str, Any]]:
    summarized: list[dict[str, Any]] = []
    for rank, doc in enumerate(result_docs[:10], start=1):
        metadata = getattr(doc, "metadata", None) or {}
        summarized.append(
            {
                "rank": rank,
                "doc_id": getattr(doc, "doc_id", None),
                "page": metadata.get("page"),
                "score": getattr(doc, "score", None),
                "doc_type": metadata.get("doc_type"),
                "chunk_id": metadata.get("chunk_id"),
                "device_name": metadata.get("device_name"),
                "title": metadata.get("title") or metadata.get("source"),
            }
        )
    return summarized


def _load_cases(csv_path: Path, variants: str, limit: int | None) -> list[QueryCase]:
    cases: list[QueryCase] = []
    with csv_path.open(encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for idx, row in enumerate(reader, start=1):
            variant_items = [("original", row.get("질문내용", "").strip())]
            if variants == "all":
                variant_items.extend(
                    [
                        ("sim1", row.get("유사질문1", "").strip()),
                        ("sim2", row.get("유사질문2", "").strip()),
                        ("sim3", row.get("유사질문3", "").strip()),
                    ]
                )
            for variant, question in variant_items:
                if not question:
                    continue
                qid = f"q{idx:03d}-{variant}"
                cases.append(
                    QueryCase(
                        qid=qid,
                        base_idx=idx,
                        variant=variant,
                        question=question,
                        device=(row.get("장비") or "").strip(),
                        gold_doc=(row.get("정답문서") or "").strip(),
                        gold_pages=(row.get("정답 페이지") or "").strip(),
                        note=(row.get("특이점") or "").strip(),
                        source_language=(row.get("원본언어") or "").strip(),
                    )
                )
                if limit is not None and len(cases) >= limit:
                    return cases
    return cases


def _build_filter_state(case: QueryCase) -> dict[str, Any]:
    selected_doc_types = expand_doc_type_selection(["sop"])
    detected_language = _detect_language_rule_based(case.question)
    return {
        "selected_devices": [case.device] if case.device else [],
        "selected_doc_types": selected_doc_types,
        "selected_doc_types_strict": True,
        "detected_language": detected_language,
        "parsed_query": {
            "selected_devices": [case.device] if case.device else [],
            "selected_doc_types": selected_doc_types,
            "doc_types_strict": True,
            "detected_language": detected_language,
        },
        "mq_mode": "fallback",
    }


def _build_task_mode_state(case: QueryCase) -> dict[str, Any]:
    detected_language = _detect_language_rule_based(case.question)
    selected_doc_types = expand_doc_type_selection(["sop"])
    return {
        "guided_confirm": True,
        "auto_parse_confirmed": True,
        "target_language": detected_language,
        "task_mode": "sop",
        "selected_devices": [case.device] if case.device else [],
        "selected_doc_types": selected_doc_types,
        "selected_doc_types_strict": True,
        "selected_equip_ids": [],
        "mq_mode": "fallback",
    }


def _run_retrieval_eval(
    cases: list[QueryCase],
    *,
    filter_agent: LangGraphRAGAgent,
    task_agent: LangGraphRAGAgent,
) -> dict[str, list[RetrievalEval]]:
    results: dict[str, list[RetrievalEval]] = {"sop_filter": [], "task_mode_sop": []}
    total = len(cases)

    for idx, case in enumerate(cases, start=1):
        for mode, agent, state in (
            ("sop_filter", filter_agent, _build_filter_state(case)),
            ("task_mode_sop", task_agent, _build_task_mode_state(case)),
        ):
            started = time.perf_counter()
            result = agent.run(
                case.question,
                thread_id=f"{mode}-{case.qid}",
                state_overrides=state,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            docs = _summarize_docs(result.get("docs") or [])
            hit_doc, hit_page = _compute_hits(case.gold_doc, case.gold_pages, docs)
            eval_row = RetrievalEval(
                qid=case.qid,
                variant=case.variant,
                mode=mode,
                question=case.question,
                route=result.get("route"),
                search_queries=[str(q) for q in (result.get("search_queries") or [])],
                gold_doc=case.gold_doc,
                gold_pages=case.gold_pages,
                hit_doc_at_10=hit_doc,
                hit_page_at_10=hit_page,
                elapsed_ms=elapsed_ms,
                top_docs=docs,
            )
            results[mode].append(eval_row)
            print(
                f"[retrieval {idx}/{total}] {case.qid} {mode:<13s} "
                f"doc={'Y' if hit_doc else 'N'} page={'Y' if hit_page else 'N'} "
                f"{elapsed_ms / 1000:.1f}s top1={docs[0]['doc_id'] if docs else '-'}",
                flush=True,
            )

    return results


def _pick_answer_samples(
    retrieval_results: dict[str, list[RetrievalEval]],
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []

    filter_map = {row.qid: row for row in retrieval_results["sop_filter"]}
    task_map = {row.qid: row for row in retrieval_results["task_mode_sop"]}

    task_better: list[str] = []
    filter_better: list[str] = []
    both_miss: list[str] = []
    both_hit: list[str] = []

    for qid, filter_row in filter_map.items():
        task_row = task_map[qid]
        if task_row.hit_doc_at_10 and not filter_row.hit_doc_at_10:
            task_better.append(qid)
        elif filter_row.hit_doc_at_10 and not task_row.hit_doc_at_10:
            filter_better.append(qid)
        elif not filter_row.hit_doc_at_10 and not task_row.hit_doc_at_10:
            both_miss.append(qid)
        elif filter_row.hit_doc_at_10 and task_row.hit_doc_at_10:
            both_hit.append(qid)

    picked: list[str] = []
    for bucket in (task_better, filter_better, both_miss, both_hit):
        for qid in bucket:
            if qid in picked:
                continue
            picked.append(qid)
            if len(picked) >= limit:
                return picked
    return picked[:limit]


def _run_answer_samples(
    cases_by_qid: dict[str, QueryCase],
    sample_qids: list[str],
    *,
    filter_agent: LangGraphRAGAgent,
    task_agent: LangGraphRAGAgent,
) -> list[AnswerEval]:
    results: list[AnswerEval] = []
    total = len(sample_qids)

    for idx, qid in enumerate(sample_qids, start=1):
        case = cases_by_qid[qid]
        for mode, agent, state in (
            ("sop_filter", filter_agent, _build_filter_state(case)),
            ("task_mode_sop", task_agent, _build_task_mode_state(case)),
        ):
            started = time.perf_counter()
            result = agent.run(
                case.question,
                thread_id=f"answer-{mode}-{case.qid}",
                state_overrides=state,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            docs = _summarize_docs(
                result.get("retrieved_docs") or result.get("docs") or []
            )
            hit_doc, hit_page = _compute_hits(case.gold_doc, case.gold_pages, docs)
            answer_row = AnswerEval(
                qid=case.qid,
                variant=case.variant,
                mode=mode,
                question=case.question,
                route=result.get("route"),
                search_queries=[str(q) for q in (result.get("search_queries") or [])],
                gold_doc=case.gold_doc,
                gold_pages=case.gold_pages,
                hit_doc_at_10=hit_doc,
                hit_page_at_10=hit_page,
                elapsed_ms=elapsed_ms,
                top_docs=docs,
                answer=str(result.get("answer") or ""),
                judge=result.get("judge") or {},
            )
            results.append(answer_row)
            print(
                f"[answer {idx}/{total}] {case.qid} {mode:<13s} "
                f"doc={'Y' if hit_doc else 'N'} faithful={answer_row.judge.get('faithful')} "
                f"{elapsed_ms / 1000:.1f}s",
                flush=True,
            )
    return results


def _variant_breakdown(rows: list[RetrievalEval]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[RetrievalEval]] = {}
    for row in rows:
        grouped.setdefault(row.variant, []).append(row)

    breakdown: dict[str, dict[str, float]] = {}
    for variant, items in grouped.items():
        total = len(items)
        breakdown[variant] = {
            "count": total,
            "doc_hit_at_10": sum(1 for item in items if item.hit_doc_at_10)
            / max(total, 1),
            "page_hit_at_10": sum(1 for item in items if item.hit_page_at_10)
            / max(total, 1),
            "avg_elapsed_ms": sum(item.elapsed_ms for item in items) / max(total, 1),
        }
    return breakdown


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def _write_summary(
    out_dir: Path,
    cases: list[QueryCase],
    retrieval_results: dict[str, list[RetrievalEval]],
    answer_rows: list[AnswerEval],
) -> None:
    lines: list[str] = []
    lines.append("# SOP Chat-Flow Eval")
    lines.append("")
    lines.append(f"- cases: {len(cases)}")
    lines.append(f"- answer_samples: {len(answer_rows) // 2}")
    lines.append("")

    for mode in ("sop_filter", "task_mode_sop"):
        rows = retrieval_results[mode]
        total = len(rows)
        doc_hits = sum(1 for row in rows if row.hit_doc_at_10)
        page_hits = sum(1 for row in rows if row.hit_page_at_10)
        avg_elapsed = sum(row.elapsed_ms for row in rows) / max(total, 1)
        lines.append(f"## {mode}")
        lines.append(
            f"- doc_hit@10: {doc_hits}/{total} ({doc_hits / max(total, 1) * 100:.2f}%)"
        )
        lines.append(
            f"- page_hit@10: {page_hits}/{total} ({page_hits / max(total, 1) * 100:.2f}%)"
        )
        lines.append(f"- avg_elapsed_ms: {avg_elapsed:.0f}")
        lines.append("- variant_breakdown:")
        for variant, stats in _variant_breakdown(rows).items():
            lines.append(
                f"  - {variant}: doc@10={stats['doc_hit_at_10'] * 100:.2f}% "
                f"page@10={stats['page_hit_at_10'] * 100:.2f}% avg={stats['avg_elapsed_ms']:.0f}ms"
            )
        lines.append("")

    filter_map = {row.qid: row for row in retrieval_results["sop_filter"]}
    task_map = {row.qid: row for row in retrieval_results["task_mode_sop"]}
    better_task = [
        qid
        for qid, row in filter_map.items()
        if task_map[qid].hit_doc_at_10 and not row.hit_doc_at_10
    ]
    better_filter = [
        qid
        for qid, row in filter_map.items()
        if row.hit_doc_at_10 and not task_map[qid].hit_doc_at_10
    ]
    both_miss = [
        qid
        for qid, row in filter_map.items()
        if (not row.hit_doc_at_10) and (not task_map[qid].hit_doc_at_10)
    ]

    lines.append("## Comparison")
    lines.append(f"- task_mode_sop_only_better: {len(better_task)}")
    lines.append(f"- sop_filter_only_better: {len(better_filter)}")
    lines.append(f"- both_miss: {len(both_miss)}")
    lines.append("")

    if answer_rows:
        lines.append("## Answer Samples")
        for row in answer_rows:
            lines.append(
                f"- [{row.mode}] {row.qid} "
                f"doc@10={'Y' if row.hit_doc_at_10 else 'N'} "
                f"faithful={row.judge.get('faithful')} "
                f"top1={(row.top_docs[0]['doc_id'] if row.top_docs else '-')}"
            )
            lines.append(f"  - question: {row.question}")
            lines.append(f"  - search_queries: {row.search_queries}")
            lines.append(f"  - answer: {row.answer[:300].replace(chr(10), ' ')}")
        lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv_path)
    cases = _load_cases(csv_path, variants=args.variants, limit=args.limit)
    cases_by_qid = {case.qid: case for case in cases}

    _configure_search_service()
    llm = get_default_llm()
    search_service = cast(SearchService, cast(object, get_search_service()))
    prompt_spec = cast(PromptSpec, get_prompt_spec_cached())
    chatflow_mode = "verified"
    chatflow_final_top_k = 20
    chatflow_retrieval_top_k = 50

    retrieval_filter_agent = LangGraphRAGAgent(
        llm=llm,
        search_service=search_service,
        prompt_spec=prompt_spec,
        top_k=chatflow_final_top_k,
        retrieval_top_k=chatflow_retrieval_top_k,
        mode=chatflow_mode,
        ask_user_after_retrieve=True,
        auto_parse_enabled=False,
        use_canonical_retrieval=False,
    )
    retrieval_task_agent = LangGraphRAGAgent(
        llm=llm,
        search_service=search_service,
        prompt_spec=prompt_spec,
        top_k=chatflow_final_top_k,
        retrieval_top_k=chatflow_retrieval_top_k,
        mode=chatflow_mode,
        ask_user_after_retrieve=True,
        auto_parse_enabled=True,
        use_canonical_retrieval=False,
    )
    answer_filter_agent = LangGraphRAGAgent(
        llm=llm,
        search_service=search_service,
        prompt_spec=prompt_spec,
        top_k=chatflow_final_top_k,
        retrieval_top_k=chatflow_retrieval_top_k,
        mode=chatflow_mode,
        ask_user_after_retrieve=False,
        auto_parse_enabled=False,
        use_canonical_retrieval=False,
        checkpointer=None,
    )
    answer_task_agent = LangGraphRAGAgent(
        llm=llm,
        search_service=search_service,
        prompt_spec=prompt_spec,
        top_k=chatflow_final_top_k,
        retrieval_top_k=chatflow_retrieval_top_k,
        mode=chatflow_mode,
        ask_user_after_retrieve=False,
        auto_parse_enabled=True,
        use_canonical_retrieval=False,
        checkpointer=None,
    )

    retrieval_results = _run_retrieval_eval(
        cases,
        filter_agent=retrieval_filter_agent,
        task_agent=retrieval_task_agent,
    )
    _write_jsonl(
        out_dir / "retrieval_sop_filter.jsonl", retrieval_results["sop_filter"]
    )
    _write_jsonl(
        out_dir / "retrieval_task_mode_sop.jsonl", retrieval_results["task_mode_sop"]
    )

    sample_qids = _pick_answer_samples(retrieval_results, args.answer_samples)
    answer_rows = _run_answer_samples(
        cases_by_qid,
        sample_qids,
        filter_agent=answer_filter_agent,
        task_agent=answer_task_agent,
    )
    _write_jsonl(out_dir / "answer_samples.jsonl", answer_rows)
    _write_jsonl(out_dir / "query_cases.jsonl", cases)
    _write_summary(out_dir, cases, retrieval_results, answer_rows)

    stats = {
        mode: {
            "count": len(rows),
            "doc_hit_at_10": sum(1 for row in rows if row.hit_doc_at_10),
            "page_hit_at_10": sum(1 for row in rows if row.hit_page_at_10),
            "top1_counter": Counter(
                row.top_docs[0]["doc_id"] if row.top_docs else "<none>" for row in rows
            ).most_common(20),
        }
        for mode, rows in retrieval_results.items()
    }
    (out_dir / "summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote evaluation outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
