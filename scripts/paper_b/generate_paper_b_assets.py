#!/usr/bin/env python3
# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnusedCallResult=false
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt


DEFAULT_EVIDENCE_ROOT = ".sisyphus/evidence/paper-b/task-10"
DEFAULT_ASSETS_DIR = "docs/paper/paper_b_assets"
DEFAULT_QUERIES = "data/synth_benchmarks/stability_bench_v1/queries.jsonl"

DRIVER_BUCKETS = ["abbr", "mixed_lang", "error_code", "near_dup"]


@dataclass(frozen=True)
class ConfigSpec:
    key: str
    label: str
    metrics_path: Path
    results_path: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Paper B table/figures")
    _ = parser.add_argument("--evidence-root", default=DEFAULT_EVIDENCE_ROOT)
    _ = parser.add_argument("--assets-dir", default=DEFAULT_ASSETS_DIR)
    _ = parser.add_argument("--queries", default=DEFAULT_QUERIES)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected object JSON in {path}")
    raw = dict(data)
    return {str(k): cast(object, v) for k, v in raw.items()}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            payload = line.split("|", 1)[-1].strip()
            if not payload:
                continue
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                rows.append({str(k): v for k, v in dict(parsed).items()})
    return rows


def _to_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise RuntimeError(f"Cannot convert value to float: {value!r}")


def _jaccard_at_k(left: list[str], right: list[str], k: int) -> float:
    left_set = set(left[:k])
    right_set = set(right[:k])
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def _parse_doc_ids(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _load_query_tags(path: Path) -> dict[str, list[str]]:
    qid_to_tags: dict[str, list[str]] = {}
    for row in _load_jsonl(path):
        qid = str(row.get("qid", "")).strip()
        if not qid:
            continue
        tags_raw = row.get("tags")
        if isinstance(tags_raw, list):
            qid_to_tags[qid] = [
                str(item).strip() for item in tags_raw if str(item).strip()
            ]
        else:
            qid_to_tags[qid] = []
    return qid_to_tags


def _driver_stability_by_bucket(
    results_rows: list[dict[str, object]],
    qid_to_tags: dict[str, list[str]],
) -> dict[str, float]:
    qid_to_repeats: dict[str, list[list[str]]] = defaultdict(list)
    for row in results_rows:
        qid = str(row.get("qid", ""))
        qid_to_repeats[qid].append(_parse_doc_ids(row.get("top_k_doc_ids")))

    bucket_scores: dict[str, list[float]] = defaultdict(list)
    for qid, repeats in qid_to_repeats.items():
        if len(repeats) <= 1:
            continue
        anchor = repeats[0]
        per_qid = [_jaccard_at_k(anchor, candidate, 10) for candidate in repeats[1:]]
        if not per_qid:
            continue
        score = sum(per_qid) / len(per_qid)
        for tag in qid_to_tags.get(qid, []):
            if tag in DRIVER_BUCKETS:
                bucket_scores[tag].append(score)

    return {
        bucket: (sum(scores) / len(scores) if scores else 0.0)
        for bucket, scores in (
            (bucket, bucket_scores.get(bucket, [])) for bucket in DRIVER_BUCKETS
        )
    }


def main() -> int:
    args = _parse_args()
    evidence_root = Path(args.evidence_root)
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ConfigSpec(
            key="baseline_v1_nondet",
            label="Baseline (v1, nondet)",
            metrics_path=evidence_root / "baseline_v1_nondet" / "metrics.json",
            results_path=evidence_root / "baseline_v1_nondet" / "results.jsonl",
        ),
        ConfigSpec(
            key="deterministic_protocol",
            label="Deterministic protocol",
            metrics_path=evidence_root / "deterministic_protocol" / "metrics.json",
            results_path=evidence_root / "deterministic_protocol" / "results.jsonl",
        ),
        ConfigSpec(
            key="stability_aware_mq_v2",
            label="Stability-aware MQ (v2, nondet)",
            metrics_path=evidence_root / "stability_aware_mq_v2" / "metrics.json",
            results_path=evidence_root / "stability_aware_mq_v2" / "results.jsonl",
        ),
    ]

    qid_to_tags = _load_query_tags(Path(args.queries))
    table_rows: list[dict[str, float | int | str]] = []
    driver_breakdown: dict[str, dict[str, float]] = {}

    for config in configs:
        if not config.metrics_path.exists() or not config.results_path.exists():
            raise RuntimeError(
                f"Missing evidence for config {config.key}: {config.metrics_path} / {config.results_path}"
            )

        metrics = _load_json(config.metrics_path)
        all_rows = _load_jsonl(config.results_path)
        primary_mode = str(metrics.get("primary_mode_name", ""))
        primary_rows = [
            row for row in all_rows if str(row.get("mode", "")) == primary_mode
        ]
        driver_breakdown[config.key] = _driver_stability_by_bucket(
            primary_rows, qid_to_tags
        )

        table_rows.append(
            {
                "configuration": config.label,
                "hit@5": _to_float(metrics.get("hit_at_5", 0.0)),
                "hit@10": _to_float(metrics.get("hit_at_10", 0.0)),
                "MRR": _to_float(metrics.get("mrr", 0.0)),
                "Stability@10 (repeat)": _to_float(
                    metrics.get("repeat_stability_jaccard_at_10", 0.0)
                ),
                "Stability@10 (paraphrase)": _to_float(
                    metrics.get("paraphrase_stability_jaccard_at_10", 0.0)
                ),
                "p95_latency_ms": _to_float(metrics.get("p95_latency_ms", 0.0)),
                "query_count": int(_to_float(metrics.get("query_count", 0))),
                "repeats": int(_to_float(metrics.get("deterministic_repeats", 0))),
            }
        )

    table_csv_path = assets_dir / "table_1_metrics.csv"
    headers = list(table_rows[0].keys())
    with table_csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=headers)
        writer.writeheader()
        for row in table_rows:
            writer.writerow(row)

    table_md_path = assets_dir / "table_1_metrics.md"
    with table_md_path.open("w", encoding="utf-8") as fp:
        _ = fp.write("| " + " | ".join(headers) + " |\n")
        _ = fp.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in table_rows:
            cells = []
            for key in headers:
                value = row[key]
                if isinstance(value, float):
                    cells.append(f"{value:.4f}")
                else:
                    cells.append(str(value))
            _ = fp.write("| " + " | ".join(cells) + " |\n")

    fig1_png = assets_dir / "figure_1_stability_vs_recall.png"
    fig1_svg = assets_dir / "figure_1_stability_vs_recall.svg"
    plt.figure(figsize=(7.5, 5.0))
    x = [float(row["hit@10"]) for row in table_rows]
    y = [float(row["Stability@10 (repeat)"]) for row in table_rows]
    labels = [str(row["configuration"]) for row in table_rows]
    plt.plot(x, y, marker="o", linewidth=1.2)
    for idx, label in enumerate(labels):
        plt.annotate(label, (x[idx], y[idx]), textcoords="offset points", xytext=(6, 4))
    plt.xlabel("Recall proxy (hit@10)")
    plt.ylabel("Stability@10 (repeat Jaccard)")
    plt.title("Figure 1. Stability vs Recall trade-off")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig1_png, dpi=180)
    plt.savefig(fig1_svg)
    plt.close()

    fig2_png = assets_dir / "figure_2_driver_breakdown.png"
    fig2_svg = assets_dir / "figure_2_driver_breakdown.svg"
    plt.figure(figsize=(8.5, 5.2))
    config_order = [cfg.key for cfg in configs]
    labels_short = {
        "baseline_v1_nondet": "Baseline",
        "deterministic_protocol": "Deterministic",
        "stability_aware_mq_v2": "MQ v2",
    }
    width = 0.22
    x_pos = list(range(len(DRIVER_BUCKETS)))
    for idx, config_key in enumerate(config_order):
        values = [driver_breakdown[config_key][bucket] for bucket in DRIVER_BUCKETS]
        shifted = [pos + (idx - 1) * width for pos in x_pos]
        plt.bar(shifted, values, width=width, label=labels_short[config_key])
    plt.xticks(x_pos, DRIVER_BUCKETS)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Stability@10 (repeat Jaccard)")
    plt.title("Figure 2. Stability driver breakdown by query bucket")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig2_png, dpi=180)
    plt.savefig(fig2_svg)
    plt.close()

    print(f"Wrote {table_csv_path}")
    print(f"Wrote {table_md_path}")
    print(f"Wrote {fig1_png}")
    print(f"Wrote {fig1_svg}")
    print(f"Wrote {fig2_png}")
    print(f"Wrote {fig2_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
