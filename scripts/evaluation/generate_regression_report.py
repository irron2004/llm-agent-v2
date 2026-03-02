#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True)
class CliArgs:
    run_root: str
    out: str


def _read_json(path: Path) -> dict[str, object]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc
    try:
        parsed = cast(object, json.loads(raw))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    raw_map = cast(dict[object, object], parsed)
    out: dict[str, object] = {}
    for key, value in raw_map.items():
        out[str(key)] = value
    return out


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        with path.open(encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    parsed = cast(object, json.loads(text))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSONL at {path}:{line_no}: {exc}"
                    ) from exc
                if not isinstance(parsed, dict):
                    raise RuntimeError(f"Expected object at {path}:{line_no}")
                raw_map = cast(dict[object, object], parsed)
                row: dict[str, object] = {}
                for key, value in raw_map.items():
                    row[str(key)] = value
                rows.append(row)
    except OSError as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc
    return rows


def _as_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _fmt_float(value: float | None, *, digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def _fmt_delta(value: float | None, *, digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return "-"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{digits}f}"


def _coerce_str_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    raw = cast(dict[object, object], value)
    out: dict[str, object] = {}
    for key, item in raw.items():
        out[str(key)] = item
    return out


def _get_pair_metrics(pair_obj: dict[str, object]) -> dict[str, dict[str, object]]:
    metrics_raw = pair_obj.get("metrics")
    if not isinstance(metrics_raw, dict):
        return {}
    metrics: dict[str, dict[str, object]] = {}
    for key, item in cast(dict[object, object], metrics_raw).items():
        key_str = str(key)
        mapped = _coerce_str_mapping(item)
        if mapped is None:
            continue
        metrics[key_str] = mapped
    return metrics


def _make_retrieval_table(
    deltas: dict[str, object],
    *,
    rows: list[tuple[str, str]],
    metrics: list[str],
    digits: int = 4,
) -> str:
    pairs_raw = deltas.get("pairs")
    pairs = _coerce_str_mapping(pairs_raw) or {}

    header = ["configuration"] + [m for m in metrics]
    lines: list[str] = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for label, key in rows:
        pair_raw = pairs.get(key)
        pair = _coerce_str_mapping(pair_raw) or {}
        present = bool(pair.get("present"))
        if not present:
            lines.append("| " + " | ".join([label] + ["-"] * len(metrics)) + " |")
            continue
        metric_map = _get_pair_metrics(pair)

        cells: list[str] = [label]
        for m in metrics:
            triple = metric_map.get(m) or {}
            b = _as_float(triple.get("before"))
            a = _as_float(triple.get("after"))
            d = _as_float(triple.get("delta"))
            cells.append(
                f"{_fmt_float(b, digits=digits)} -> {_fmt_float(a, digits=digits)} ({_fmt_delta(d, digits=digits)})"
            )
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _select_top_diffs(
    agent_delta: dict[str, object], *, n: int = 6
) -> list[dict[str, object]]:
    per_qid_raw = agent_delta.get("per_qid")
    if not isinstance(per_qid_raw, list):
        return []
    scored: list[tuple[float, dict[str, object]]] = []
    for item in cast(list[object], per_qid_raw):
        row = _coerce_str_mapping(item)
        if row is None:
            continue
        if row.get("status") != "ok":
            continue
        score = _as_float(row.get("jaccard_at_10"))
        if score is None:
            continue
        scored.append((score, row))
    scored.sort(key=lambda t: t[0])
    return [row for _, row in scored[:n]]


def _parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Generate before/after regression report"
    )
    _ = parser.add_argument("--run-root", required=True)
    _ = parser.add_argument("--out", required=True)
    parsed = parser.parse_args()
    run_root = str(getattr(parsed, "run_root", "")).strip()
    out = str(getattr(parsed, "out", "")).strip()
    if not run_root:
        raise RuntimeError("--run-root must be non-empty")
    if not out:
        raise RuntimeError("--out must be non-empty")
    return CliArgs(run_root=run_root, out=out)


def main() -> int:
    args = _parse_args()
    run_root = Path(args.run_root).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(run_root / "manifest.json")
    before_sha = str(manifest.get("before_sha") or "")
    after_sha = str(manifest.get("after_sha") or "")
    run_id = str(manifest.get("run_id") or run_root.name)

    compare_summary = _read_json(run_root / "deltas" / "compare_summary.json")
    retrieval_delta = compare_summary.get("retrieval")
    agent_delta = compare_summary.get("agent")
    retrieval_obj = _coerce_str_mapping(retrieval_delta) or {}
    agent_obj = _coerce_str_mapping(agent_delta) or {}

    retrieval_metrics = [
        "hit@10",
        "MRR",
        "RepeatJaccard@10",
        "ParaphraseJaccard@10",
        "p95_latency_ms",
    ]
    subset_table = _make_retrieval_table(
        retrieval_obj,
        rows=[
            ("subset_48 det_true", "subset_48_det_true"),
            ("subset_48 det_false", "subset_48_det_false"),
        ],
        metrics=retrieval_metrics,
        digits=4,
    )
    full_table = _make_retrieval_table(
        retrieval_obj,
        rows=[
            ("full_240 det_true", "full_240_det_true"),
            ("full_240 det_false (skipped)", "full_240_det_false"),
        ],
        metrics=retrieval_metrics,
        digits=4,
    )

    doc_j = _coerce_str_mapping(agent_obj.get("doc_id_jaccard_at_10")) or {}
    top_diffs = _select_top_diffs(agent_obj, n=6)

    qid_to_query: dict[str, str] = {}
    before_run_path = run_root / "before" / "agent" / "subset_48" / "run.jsonl"
    if before_run_path.exists():
        for row in _read_jsonl(before_run_path):
            qid = str(row.get("qid") or "").strip()
            query = str(row.get("query") or "").strip()
            if qid and query:
                qid_to_query[qid] = query

    lines: list[str] = []
    lines.append(f"# Before/After Regression Compare ({run_id})")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- before_sha: `{before_sha}`")
    lines.append(f"- after_sha: `{after_sha}`")
    lines.append(f"- run_root: `{run_root}`")

    query_files = _coerce_str_mapping(manifest.get("query_files")) or {}
    subset_q = _coerce_str_mapping(query_files.get("subset")) or {}
    full_q = _coerce_str_mapping(query_files.get("full")) or {}
    if subset_q:
        lines.append(
            f"- subset_48: `{subset_q.get('path')}` (lines={subset_q.get('line_count')}, sha256={subset_q.get('sha256')})"
        )
    if full_q:
        lines.append(
            f"- full_240: `{full_q.get('path')}` (lines={full_q.get('line_count')}, sha256={full_q.get('sha256')})"
        )

    es_cfg = _coerce_str_mapping(manifest.get("es_config")) or {}
    if es_cfg:
        lines.append(
            f"- ES: host={es_cfg.get('host')}, env={es_cfg.get('env')}, index_prefix={es_cfg.get('index_prefix')}"
        )

    lines.append("")
    lines.append("## Retrieval Metrics")
    lines.append("")
    lines.append("### subset_48")
    lines.append("")
    lines.append(subset_table)
    lines.append("")
    lines.append("### full_240")
    lines.append("")
    lines.append(full_table)

    lines.append("")
    lines.append("## Agent Signals (subset_48)")
    lines.append("")
    lines.append(f"- common_qids: {agent_obj.get('qids_common', '-')}")
    lines.append(f"- error_rows: {agent_obj.get('error_rows', '-')}")
    lines.append(f"- skipped_rows: {agent_obj.get('skipped_rows', '-')}")
    lines.append(
        f"- doc_id_jaccard@10: mean={_fmt_float(_as_float(doc_j.get('mean')), digits=4)}, p95={_fmt_float(_as_float(doc_j.get('p95')), digits=4)}, count={doc_j.get('count', '-')}"
    )

    for key in ("route", "mq_used", "attempts", "faithful"):
        dist = _coerce_str_mapping(agent_obj.get(key)) or {}
        b = dist.get("before")
        a = dist.get("after")
        lines.append(f"- {key}: before={b} | after={a}")

    lines.append("")
    lines.append("## Representative Diffs (lowest doc-id Jaccard@10)")
    lines.append("")

    if not top_diffs:
        lines.append("- (no diffs available yet)")
    else:
        for item in top_diffs:
            qid = str(item.get("qid") or "")
            query = qid_to_query.get(qid, "")
            score = _as_float(item.get("jaccard_at_10"))
            before_obj = _coerce_str_mapping(item.get("before")) or {}
            after_obj = _coerce_str_mapping(item.get("after")) or {}
            lines.append(f"### {qid} (jaccard@10={_fmt_float(score, digits=4)})")
            lines.append("")
            if query:
                lines.append(f"- query: {query}")
            lines.append(
                f"- before: route={before_obj.get('route')}, faithful={before_obj.get('faithful')}"
            )
            lines.append(
                f"- after: route={after_obj.get('route')}, faithful={after_obj.get('faithful')}"
            )
            lines.append(f"- before_doc_ids: {before_obj.get('doc_ids')}")
            lines.append(f"- after_doc_ids: {after_obj.get('doc_ids')}")
            lines.append("")

    _ = out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
