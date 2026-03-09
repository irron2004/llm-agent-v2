#!/usr/bin/env python3
"""
report_split_quality.py - Split quality + power analysis report for Paper A.

Usage:
    python scripts/paper_a/report_split_quality.py \
        --eval-set data/paper_a/eval/query_gold_master_v0_5.jsonl \
        --split-report data/paper_a/eval/query_gold_master_v0_5_split_report.json \
        --out-json .sisyphus/evidence/paper-a/reports/T6_split_quality_2026-03-09.json \
        --out-md .sisyphus/evidence/paper-a/reports/T6_split_quality_2026-03-09.md
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import date


OBSERVABILITY_ORDER = ["explicit_device", "explicit_equip", "implicit", "ambiguous"]

# Power analysis constants
ALPHA = 0.05
POWER = 0.8
P_WORST = 0.5
Z_ALPHA2 = 1.96   # z_{0.975} for two-sided alpha=0.05
Z_BETA = 0.842    # z_{0.8} for power=0.8


def mde(n: int) -> float:
    """Minimum detectable effect for hit@5 (absolute), two-sided."""
    if n <= 0:
        return float("inf")
    return (Z_ALPHA2 + Z_BETA) * math.sqrt(2 * P_WORST * (1 - P_WORST) / n)


def load_eval(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: line {lineno} parse error: {e}", file=sys.stderr)
    return rows


def counts_by_field(rows: list[dict], split: str, field: str) -> dict:
    counter: dict[str, int] = defaultdict(int)
    for r in rows:
        if r.get("split") == split:
            counter[r.get(field, "")] += 1
    return dict(counter)


def gold_coverage(rows: list[dict]) -> list[dict]:
    """For each (split, scope_observability), count non-empty gold_doc_ids vs total."""
    groups: dict[tuple, dict] = defaultdict(lambda: {"total": 0, "with_gold": 0})
    for r in rows:
        key = (r.get("split", ""), r.get("scope_observability", ""))
        groups[key]["total"] += 1
        if r.get("gold_doc_ids"):
            groups[key]["with_gold"] += 1

    result = []
    for (split, obs), counts in sorted(groups.items()):
        n = counts["total"]
        w = counts["with_gold"]
        result.append({
            "split": split,
            "scope_observability": obs,
            "total": n,
            "with_gold": w,
            "pct": round(w / n * 100, 1) if n > 0 else 0.0,
        })
    return result


def power_analysis(rows: list[dict]) -> list[dict]:
    """MDE for test total and each test slice by scope_observability."""
    test_rows = [r for r in rows if r.get("split") == "test"]
    n_total = len(test_rows)

    slices: dict[str, int] = defaultdict(int)
    for r in test_rows:
        slices[r.get("scope_observability", "")] += 1

    results = [{"slice": "test_total", "n": n_total, "mde": round(mde(n_total), 4)}]
    for obs in OBSERVABILITY_ORDER:
        n = slices.get(obs, 0)
        results.append({"slice": f"test_{obs}", "n": n, "mde": round(mde(n), 4) if n > 0 else None})
    return results


def build_limitations_paragraph(rows: list[dict], pa: list[dict]) -> str:
    test_rows = [r for r in rows if r.get("split") == "test"]
    n_total = len(test_rows)

    pa_map = {p["slice"]: p for p in pa}
    mde_total = pa_map.get("test_total", {}).get("mde", "N/A")
    mde_explicit_device = pa_map.get("test_explicit_device", {}).get("mde", "N/A")
    n_explicit_device = pa_map.get("test_explicit_device", {}).get("n", "N/A")

    return (
        f"Our evaluation uses a test set of {n_total} queries drawn from explicit-device, "
        f"explicit-equip, and implicit scope-observability strata (ambiguous queries are "
        f"held in the dev set only). "
        f"The minimum detectable effect (MDE) for hit@5 at alpha=0.05 and power=0.8 is "
        f"{mde_total:.3f} ({mde_total*100:.1f} pp) for the full test set. "
        f"For smaller slices like explicit_device (n={n_explicit_device}), the MDE increases "
        f"to {mde_explicit_device:.3f} ({mde_explicit_device*100:.1f} pp). "
        f"These constraints limit our ability to detect small improvements but are sufficient "
        f"to detect the large contamination differences observed between scoped and unscoped systems."
    )


def render_md(
    counts_obs: dict,
    counts_intent: dict,
    gold_cov: list[dict],
    pa: list[dict],
    limitations: str,
) -> str:
    lines = []
    lines.append("# Paper A – Split Quality & Power Analysis Report")
    lines.append("")
    lines.append(f"Generated: {date.today().isoformat()}")
    lines.append("")

    # --- Counts by observability ---
    lines.append("## 1. Query Counts by Scope Observability")
    lines.append("")
    all_obs = sorted(
        set(list(counts_obs.get("dev", {}).keys()) + list(counts_obs.get("test", {}).keys()))
    )
    lines.append("| scope_observability | dev | test |")
    lines.append("|---|---|---|")
    for obs in OBSERVABILITY_ORDER:
        d = counts_obs.get("dev", {}).get(obs, 0)
        t = counts_obs.get("test", {}).get(obs, 0)
        lines.append(f"| {obs} | {d} | {t} |")
    lines.append("")

    # --- Counts by intent ---
    lines.append("## 2. Query Counts by Intent (Primary)")
    lines.append("")
    all_intents = sorted(
        set(list(counts_intent.get("dev", {}).keys()) + list(counts_intent.get("test", {}).keys()))
    )
    lines.append("| intent_primary | dev | test |")
    lines.append("|---|---|---|")
    for intent in all_intents:
        d = counts_intent.get("dev", {}).get(intent, 0)
        t = counts_intent.get("test", {}).get(intent, 0)
        lines.append(f"| {intent} | {d} | {t} |")
    lines.append("")

    # --- Gold coverage ---
    lines.append("## 3. Gold Coverage by Split x Scope Observability")
    lines.append("")
    lines.append("| split | scope_observability | total | with_gold | pct |")
    lines.append("|---|---|---|---|---|")
    for row in gold_cov:
        lines.append(
            f"| {row['split']} | {row['scope_observability']} "
            f"| {row['total']} | {row['with_gold']} | {row['pct']}% |"
        )
    lines.append("")

    # --- Power analysis ---
    lines.append("## 4. Power / MDE Analysis (hit@5, alpha=0.05, power=0.8)")
    lines.append("")
    lines.append(f"Parameters: alpha={ALPHA}, power={POWER}, p={P_WORST} (worst-case), "
                 f"z_alpha/2={Z_ALPHA2}, z_beta={Z_BETA}")
    lines.append("")
    lines.append("Formula: MDE(n) = (z_alpha/2 + z_beta) * sqrt(2 * p * (1-p) / n)")
    lines.append("")
    lines.append("| slice | n | MDE (absolute) | MDE (pp) |")
    lines.append("|---|---|---|---|")
    for row in pa:
        mde_val = row["mde"]
        if mde_val is None:
            lines.append(f"| {row['slice']} | {row['n']} | N/A | N/A |")
        else:
            lines.append(f"| {row['slice']} | {row['n']} | {mde_val:.4f} | {mde_val*100:.1f} pp |")
    lines.append("")

    # --- Limitations ---
    lines.append("## 5. Limitations")
    lines.append("")
    lines.append(limitations)
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Paper A split quality + power analysis report")
    parser.add_argument("--eval-set", required=True, help="Path to JSONL eval set")
    parser.add_argument("--split-report", required=True, help="Path to split report JSON")
    parser.add_argument("--out-json", required=True, help="Output JSON path")
    parser.add_argument("--out-md", required=True, help="Output MD path")
    args = parser.parse_args()

    # Load data
    print(f"Loading eval set: {args.eval_set}")
    rows = load_eval(args.eval_set)
    print(f"  Loaded {len(rows)} rows")

    print(f"Loading split report: {args.split_report}")
    with open(args.split_report, encoding="utf-8") as f:
        split_report = json.load(f)
    print(f"  Split report keys: {list(split_report.keys())[:5]} ...")

    # Compute metrics
    counts_obs = {
        "dev": counts_by_field(rows, "dev", "scope_observability"),
        "test": counts_by_field(rows, "test", "scope_observability"),
    }
    counts_intent = {
        "dev": counts_by_field(rows, "dev", "intent_primary"),
        "test": counts_by_field(rows, "test", "intent_primary"),
    }
    gold_cov = gold_coverage(rows)
    pa = power_analysis(rows)
    limitations = build_limitations_paragraph(rows, pa)

    # Build output JSON
    out_data = {
        "counts_by_observability": counts_obs,
        "counts_by_intent": counts_intent,
        "gold_coverage": gold_cov,
        "power_analysis": pa,
        "generated_at": date.today().isoformat(),
    }

    # Write outputs
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"Wrote JSON: {args.out_json}")

    md_content = render_md(counts_obs, counts_intent, gold_cov, pa, limitations)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Wrote MD:   {args.out_md}")

    # Print summary to stdout
    print("\n=== Summary ===")
    print(f"Total rows: {len(rows)}")
    print(f"Dev: {sum(counts_obs['dev'].values())}  Test: {sum(counts_obs['test'].values())}")
    print("\nScope observability (dev | test):")
    for obs in OBSERVABILITY_ORDER:
        d = counts_obs["dev"].get(obs, 0)
        t = counts_obs["test"].get(obs, 0)
        print(f"  {obs:20s}: {d:4d} | {t:4d}")
    print("\nPower analysis:")
    for row in pa:
        mde_val = row["mde"]
        if mde_val is not None:
            print(f"  {row['slice']:30s}: n={row['n']:4d}  MDE={mde_val:.4f} ({mde_val*100:.1f} pp)")
        else:
            print(f"  {row['slice']:30s}: n={row['n']:4d}  MDE=N/A")
    print(f"\nLimitations paragraph:\n{limitations}")


if __name__ == "__main__":
    main()
