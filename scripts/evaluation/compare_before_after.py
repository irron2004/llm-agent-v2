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
class RetrievalPair:
    label: str
    before_metrics: Path
    after_metrics: Path


@dataclass(frozen=True)
class CliArgs:
    run_root: str
    out_dir: str


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
    obj = cast(dict[object, object], parsed)
    out: dict[str, object] = {}
    for key, value in obj.items():
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


def _coerce_str_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    raw = cast(dict[object, object], value)
    out: dict[str, object] = {}
    for key, item in raw.items():
        out[str(key)] = item
    return out


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


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _jaccard(left: list[str], right: list[str]) -> float:
    a = set([x for x in left if x])
    b = set([x for x in right if x])
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = int(math.ceil(0.95 * len(ordered))) - 1
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def _count_by(values: list[object]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        if value is None:
            key = "<none>"
        elif isinstance(value, bool):
            key = "true" if value else "false"
        else:
            key = str(value)
        out[key] = out.get(key, 0) + 1
    return out


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def _gate_drop(
    before: float | None, after: float | None, *, max_drop: float
) -> dict[str, object]:
    d = _delta(before, after)
    if d is None:
        return {"pass": None, "max_drop": max_drop, "delta": None}
    return {"pass": (d >= (-1.0 * max_drop)), "max_drop": max_drop, "delta": d}


def _resolve_retrieval_pairs(run_root: Path) -> list[RetrievalPair]:
    def _pair(label: str, rel: str) -> RetrievalPair:
        before = run_root / "before" / "retrieval" / rel / "metrics.json"
        after = run_root / "after" / "retrieval" / rel / "metrics.json"
        return RetrievalPair(label=label, before_metrics=before, after_metrics=after)

    return [
        _pair("subset_48_det_true", "subset_48/det_true"),
        _pair("subset_48_det_false", "subset_48/det_false"),
        _pair("full_240_det_true", "full_240/det_true"),
        _pair("full_240_det_false", "full_240/det_false"),
    ]


def _compare_retrieval(run_root: Path) -> dict[str, object]:
    keyset = [
        "hit@5",
        "hit@10",
        "MRR",
        "RepeatJaccard@10",
        "ParaphraseJaccard@10",
        "p95_latency_ms",
        "query_count",
        "deterministic_repeats",
    ]

    gates = {
        "hit@10": 0.02,
        "MRR": 0.02,
        "RepeatJaccard@10": 0.02,
        "ParaphraseJaccard@10": 0.02,
    }

    results: dict[str, object] = {}
    failures: list[dict[str, object]] = []

    for pair in _resolve_retrieval_pairs(run_root):
        if not pair.before_metrics.exists() or not pair.after_metrics.exists():
            results[pair.label] = {
                "present": False,
                "before_path": str(pair.before_metrics),
                "after_path": str(pair.after_metrics),
            }
            continue

        before = _read_json(pair.before_metrics)
        after = _read_json(pair.after_metrics)

        row: dict[str, object] = {
            "present": True,
            "before_path": str(pair.before_metrics),
            "after_path": str(pair.after_metrics),
            "metrics": {},
            "gates": {},
        }

        metrics_out: dict[str, object] = {}
        for key in keyset:
            b = _as_float(before.get(key))
            a = _as_float(after.get(key))
            if b is None:
                b_int = _as_int(before.get(key))
                b = float(b_int) if b_int is not None else None
            if a is None:
                a_int = _as_int(after.get(key))
                a = float(a_int) if a_int is not None else None
            metrics_out[key] = {"before": b, "after": a, "delta": _delta(b, a)}
        row["metrics"] = metrics_out

        gates_out: dict[str, object] = {}
        for key, max_drop in gates.items():
            b = _as_float(before.get(key))
            a = _as_float(after.get(key))
            gate = _gate_drop(b, a, max_drop=max_drop)
            gates_out[key] = gate
            if gate.get("pass") is False:
                failures.append({"pair": pair.label, "metric": key, **gate})
        row["gates"] = gates_out
        results[pair.label] = row

    overall_pass = all(item.get("pass") is not False for item in failures)
    return {"overall_pass": overall_pass, "pairs": results, "failures": failures}


def _compare_agent(run_root: Path) -> dict[str, object]:
    before_path = run_root / "before" / "agent" / "subset_48" / "run.jsonl"
    after_path = run_root / "after" / "agent" / "subset_48" / "run.jsonl"

    if not before_path.exists() or not after_path.exists():
        return {
            "present": False,
            "before_path": str(before_path),
            "after_path": str(after_path),
        }

    before_rows = _read_jsonl(before_path)
    after_rows = _read_jsonl(after_path)
    before_by_qid = {
        str(r.get("qid") or ""): r for r in before_rows if str(r.get("qid") or "")
    }
    after_by_qid = {
        str(r.get("qid") or ""): r for r in after_rows if str(r.get("qid") or "")
    }
    common_qids = sorted(set(before_by_qid.keys()) & set(after_by_qid.keys()))

    route_before: list[object] = []
    route_after: list[object] = []
    mq_used_before: list[object] = []
    mq_used_after: list[object] = []
    attempts_before: list[object] = []
    attempts_after: list[object] = []
    faithful_before: list[object] = []
    faithful_after: list[object] = []

    jaccard_at_10: list[float] = []
    per_qid: list[dict[str, object]] = []

    missing: list[str] = []
    skipped: int = 0
    error_rows: int = 0

    for qid in common_qids:
        b = before_by_qid[qid]
        a = after_by_qid[qid]

        b_err = b.get("error")
        a_err = a.get("error")
        if b_err is not None or a_err is not None:
            error_rows += 1
            per_qid.append(
                {
                    "qid": qid,
                    "status": "error",
                    "before_error": b_err,
                    "after_error": a_err,
                }
            )
            continue

        b_run = cast(
            dict[str, object] | None,
            b.get("run") if isinstance(b.get("run"), dict) else None,
        )
        a_run = cast(
            dict[str, object] | None,
            a.get("run") if isinstance(a.get("run"), dict) else None,
        )
        if b_run is None or a_run is None:
            skipped += 1
            per_qid.append({"qid": qid, "status": "missing_run"})
            continue

        b_meta = cast(
            dict[str, object] | None,
            b_run.get("metadata") if isinstance(b_run.get("metadata"), dict) else None,
        )
        a_meta = cast(
            dict[str, object] | None,
            a_run.get("metadata") if isinstance(a_run.get("metadata"), dict) else None,
        )
        if b_meta is None or a_meta is None:
            skipped += 1
            per_qid.append({"qid": qid, "status": "missing_metadata"})
            continue

        route_before.append(b_meta.get("route"))
        route_after.append(a_meta.get("route"))
        mq_used_before.append(b_meta.get("mq_used"))
        mq_used_after.append(a_meta.get("mq_used"))
        attempts_before.append(b_meta.get("attempts"))
        attempts_after.append(a_meta.get("attempts"))

        b_judge = cast(
            dict[str, object] | None,
            b_run.get("judge") if isinstance(b_run.get("judge"), dict) else None,
        )
        a_judge = cast(
            dict[str, object] | None,
            a_run.get("judge") if isinstance(a_run.get("judge"), dict) else None,
        )
        faithful_before.append(b_judge.get("faithful") if b_judge else None)
        faithful_after.append(a_judge.get("faithful") if a_judge else None)

        b_ids_raw = b_run.get("retrieved_doc_ids")
        a_ids_raw = a_run.get("retrieved_doc_ids")
        b_ids = [
            str(x).strip()
            for x in cast(list[object], b_ids_raw or [])
            if str(x).strip()
        ]
        a_ids = [
            str(x).strip()
            for x in cast(list[object], a_ids_raw or [])
            if str(x).strip()
        ]
        score = _jaccard(b_ids[:10], a_ids[:10])
        jaccard_at_10.append(score)

        per_qid.append(
            {
                "qid": qid,
                "status": "ok",
                "jaccard_at_10": score,
                "before": {
                    "route": b_meta.get("route"),
                    "mq_used": b_meta.get("mq_used"),
                    "attempts": b_meta.get("attempts"),
                    "faithful": (b_judge or {}).get("faithful"),
                    "doc_ids": b_ids[:10],
                },
                "after": {
                    "route": a_meta.get("route"),
                    "mq_used": a_meta.get("mq_used"),
                    "attempts": a_meta.get("attempts"),
                    "faithful": (a_judge or {}).get("faithful"),
                    "doc_ids": a_ids[:10],
                },
            }
        )

    if not common_qids:
        missing = ["no common qids"]

    return {
        "present": True,
        "before_path": str(before_path),
        "after_path": str(after_path),
        "qids_common": len(common_qids),
        "rows_before": len(before_rows),
        "rows_after": len(after_rows),
        "error_rows": error_rows,
        "skipped_rows": skipped,
        "route": {"before": _count_by(route_before), "after": _count_by(route_after)},
        "mq_used": {
            "before": _count_by(mq_used_before),
            "after": _count_by(mq_used_after),
        },
        "attempts": {
            "before": _count_by(attempts_before),
            "after": _count_by(attempts_after),
        },
        "faithful": {
            "before": _count_by(faithful_before),
            "after": _count_by(faithful_after),
        },
        "doc_id_jaccard_at_10": {
            "mean": _mean(jaccard_at_10),
            "p95": _p95(jaccard_at_10),
            "count": len(jaccard_at_10),
        },
        "missing": missing,
        "per_qid": per_qid,
    }


def _parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(description="Compare before/after evidence")
    _ = parser.add_argument("--run-root", required=True)
    _ = parser.add_argument("--out-dir", required=True)
    parsed = parser.parse_args()
    run_root = str(getattr(parsed, "run_root", "")).strip()
    out_dir = str(getattr(parsed, "out_dir", "")).strip()
    if not run_root:
        raise RuntimeError("--run-root must be non-empty")
    if not out_dir:
        raise RuntimeError("--out-dir must be non-empty")
    return CliArgs(run_root=run_root, out_dir=out_dir)


def main() -> int:
    args = _parse_args()
    run_root = Path(args.run_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    retrieval = _compare_retrieval(run_root)
    agent = _compare_agent(run_root)

    pairs_raw = retrieval.get("pairs")
    pairs = _coerce_str_mapping(pairs_raw) or {}

    retrieval_subset = {
        "overall_pass": retrieval.get("overall_pass"),
        "subset_48_det_true": pairs.get("subset_48_det_true", {}),
        "subset_48_det_false": pairs.get("subset_48_det_false", {}),
        "failures": retrieval.get("failures"),
    }
    retrieval_full = {
        "overall_pass": retrieval.get("overall_pass"),
        "full_240_det_true": pairs.get("full_240_det_true", {}),
        "full_240_det_false": pairs.get("full_240_det_false", {}),
        "failures": retrieval.get("failures"),
    }

    _ = (out_dir / "retrieval_subset_48.json").write_text(
        json.dumps(retrieval_subset, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _ = (out_dir / "retrieval_full_240.json").write_text(
        json.dumps(retrieval_full, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _ = (out_dir / "agent_subset_48.json").write_text(
        json.dumps(agent, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _ = (out_dir / "compare_summary.json").write_text(
        json.dumps(
            {"retrieval": retrieval, "agent": agent}, indent=2, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
