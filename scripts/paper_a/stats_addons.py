"""Stats add-ons for Paper A: H5, H10, H11 analyses.

Uses only Python stdlib (csv, json, random, math, argparse, pathlib).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import date
from pathlib import Path


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def bootstrap_mean_diff(values_a: list[float], values_b: list[float], seed: int, samples: int) -> list[float]:
    """Paired bootstrap: resample with replacement, return sorted deltas (b - a)."""
    rng = random.Random(seed)
    n = len(values_a)
    deltas: list[float] = []
    for _ in range(samples):
        idx = [rng.randrange(n) for _ in range(n)]
        d = sum(values_b[i] - values_a[i] for i in idx) / n
        deltas.append(d)
    deltas.sort()
    return deltas


def bootstrap_mean_single(values: list[float], seed: int, samples: int) -> list[float]:
    """Non-paired bootstrap of the mean for a single sample."""
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(samples):
        idx = [rng.randrange(n) for _ in range(n)]
        m = sum(values[i] for i in idx) / n
        means.append(m)
    means.sort()
    return means


def ci_bounds(sorted_vals: list[float], alpha: float) -> tuple[float, float]:
    """Return (lower, upper) percentile CI from sorted bootstrap distribution."""
    n = len(sorted_vals)
    lo_idx = int(math.floor(alpha / 2 * n))
    hi_idx = int(math.ceil((1 - alpha / 2) * n)) - 1
    hi_idx = min(hi_idx, n - 1)
    return sorted_vals[lo_idx], sorted_vals[hi_idx]


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(val: str) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# H5: Cross-slice CI — B3 implicit vs explicit_device, adj_cont@5
# ---------------------------------------------------------------------------

def analyse_h5(baselines_rows: list[dict[str, str]], seed: int, samples: int) -> dict:
    """Bootstrap 95% CI for difference in means: implicit minus explicit_device for B3."""
    implicit_vals: list[float] = []
    explicit_vals: list[float] = []

    for row in baselines_rows:
        if row["system_id"] != "B3":
            continue
        if row["status"] != "ok":
            continue
        v = safe_float(row.get("adj_cont@5", ""))
        if v is None:
            continue
        obs = row["scope_observability"]
        if obs == "implicit":
            implicit_vals.append(v)
        elif obs == "explicit_device":
            explicit_vals.append(v)

    n_imp = len(implicit_vals)
    n_exp = len(explicit_vals)

    if n_imp == 0 or n_exp == 0:
        return {
            "n_implicit": n_imp,
            "n_explicit": n_exp,
            "error": "insufficient data for bootstrap",
        }

    mean_imp = sum(implicit_vals) / n_imp
    mean_exp = sum(explicit_vals) / n_exp
    delta = mean_imp - mean_exp

    # Non-paired: bootstrap difference of means by resampling each group independently
    rng = random.Random(seed)
    boot_deltas: list[float] = []
    for _ in range(samples):
        idx_a = [rng.randrange(n_imp) for _ in range(n_imp)]
        idx_b = [rng.randrange(n_exp) for _ in range(n_exp)]
        d = (sum(implicit_vals[i] for i in idx_a) / n_imp) - (sum(explicit_vals[i] for i in idx_b) / n_exp)
        boot_deltas.append(d)
    boot_deltas.sort()

    ci_lo, ci_hi = ci_bounds(boot_deltas, alpha=0.05)

    return {
        "n_implicit": n_imp,
        "n_explicit": n_exp,
        "mean_implicit": round(mean_imp, 6),
        "mean_explicit": round(mean_exp, 6),
        "delta": round(delta, 6),
        "ci_lower": round(ci_lo, 6),
        "ci_upper": round(ci_hi, 6),
        "metric": "adj_cont@5",
        "system": "B3",
    }


# ---------------------------------------------------------------------------
# H10: Paired bootstrap CI — explicit_device, B1 vs B0, adj_cont@5
# ---------------------------------------------------------------------------

def analyse_h10(baselines_rows: list[dict[str, str]], seed: int, samples: int) -> dict:
    """Paired bootstrap 95% CI: B1 - B0 for explicit_device slice."""
    b0: dict[str, float] = {}
    b1: dict[str, float] = {}

    for row in baselines_rows:
        if row["scope_observability"] != "explicit_device":
            continue
        if row["status"] != "ok":
            continue
        v = safe_float(row.get("adj_cont@5", ""))
        if v is None:
            continue
        qid = row["q_id"]
        sid = row["system_id"]
        if sid == "B0":
            b0[qid] = v
        elif sid == "B1":
            b1[qid] = v

    common_qids = sorted(set(b0) & set(b1))
    n = len(common_qids)

    if n == 0:
        return {"n_paired": 0, "error": "no paired queries found"}

    vals_a = [b0[q] for q in common_qids]
    vals_b = [b1[q] for q in common_qids]

    delta_mean = sum(b - a for a, b in zip(vals_a, vals_b)) / n

    boot_deltas = bootstrap_mean_diff(vals_a, vals_b, seed=seed, samples=samples)
    ci_lo, ci_hi = ci_bounds(boot_deltas, alpha=0.05)

    return {
        "n_paired": n,
        "delta_mean": round(delta_mean, 6),
        "ci_lower": round(ci_lo, 6),
        "ci_upper": round(ci_hi, 6),
        "metric": "adj_cont@5",
        "comparison": "B1 - B0",
        "slice": "explicit_device",
    }


# ---------------------------------------------------------------------------
# H11: Equivalence test — explicit_device, B3 vs B2, adj_cont@5
# ---------------------------------------------------------------------------

def analyse_h11(
    baselines_rows: list[dict[str, str]],
    seed: int,
    samples: int,
    margin: float,
) -> dict:
    """Paired bootstrap 90% CI for B3 - B2; decide equivalence if CI ⊂ [-margin, +margin]."""
    b2: dict[str, float] = {}
    b3: dict[str, float] = {}

    for row in baselines_rows:
        if row["scope_observability"] != "explicit_device":
            continue
        if row["status"] != "ok":
            continue
        v = safe_float(row.get("adj_cont@5", ""))
        if v is None:
            continue
        qid = row["q_id"]
        sid = row["system_id"]
        if sid == "B2":
            b2[qid] = v
        elif sid == "B3":
            b3[qid] = v

    common_qids = sorted(set(b2) & set(b3))
    n = len(common_qids)

    if n == 0:
        return {"n_paired": 0, "error": "no paired queries found"}

    vals_a = [b2[q] for q in common_qids]
    vals_b = [b3[q] for q in common_qids]

    delta_mean = sum(b - a for a, b in zip(vals_a, vals_b)) / n

    boot_deltas = bootstrap_mean_diff(vals_a, vals_b, seed=seed, samples=samples)
    # 90% CI (alpha=0.10)
    ci_lo, ci_hi = ci_bounds(boot_deltas, alpha=0.10)

    is_equivalent = bool(ci_lo >= -margin and ci_hi <= margin)

    return {
        "n_paired": n,
        "delta_mean": round(delta_mean, 6),
        "ci_lower_90": round(ci_lo, 6),
        "ci_upper_90": round(ci_hi, 6),
        "margin": margin,
        "is_equivalent": is_equivalent,
        "metric": "adj_cont@5",
        "comparison": "B3 - B2",
        "slice": "explicit_device",
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    return f"{v:.4f}"


def build_markdown(results: dict) -> str:
    lines: list[str] = []
    lines.append("# Paper A — Stats Add-ons Report")
    lines.append(f"\nGenerated: {results['generated_at']}\n")

    # H5
    lines.append("## H5: Cross-Slice CI — B3 Implicit vs Explicit-Device (adj_cont@5)")
    h5 = results["H5_cross_slice"]
    if "error" in h5:
        lines.append(f"\n**Error**: {h5['error']}\n")
    else:
        lines.append(f"""
| Slice | N | Mean adj_cont@5 |
|---|---|---|
| implicit | {h5['n_implicit']} | {_fmt(h5['mean_implicit'])} |
| explicit_device | {h5['n_explicit']} | {_fmt(h5['mean_explicit'])} |

- **Delta** (implicit − explicit_device): {_fmt(h5['delta'])}
- **95% Bootstrap CI**: [{_fmt(h5['ci_lower'])}, {_fmt(h5['ci_upper'])}]
""")
        delta = h5["delta"]
        ci_lo = h5["ci_lower"]
        ci_hi = h5["ci_upper"]
        if ci_lo > 0:
            interp = (
                f"The 95% CI [{_fmt(ci_lo)}, {_fmt(ci_hi)}] lies entirely above zero, "
                f"suggesting that B3 performs meaningfully better on implicit queries than on "
                f"explicit-device queries (delta = {_fmt(delta)})."
            )
        elif ci_hi < 0:
            interp = (
                f"The 95% CI [{_fmt(ci_lo)}, {_fmt(ci_hi)}] lies entirely below zero, "
                f"suggesting B3 performs worse on implicit queries than on explicit-device queries "
                f"(delta = {_fmt(delta)})."
            )
        else:
            interp = (
                f"The 95% CI [{_fmt(ci_lo)}, {_fmt(ci_hi)}] straddles zero (delta = {_fmt(delta)}), "
                f"so there is no strong evidence of a performance difference between slices for B3."
            )
        lines.append(f"**Interpretation**: {interp}\n")

    # H10
    lines.append("## H10: Paired Bootstrap CI — B1 vs B0, Explicit-Device Slice (adj_cont@5)")
    h10 = results["H10_paired_bootstrap"]
    if "error" in h10:
        lines.append(f"\n**Error**: {h10['error']}\n")
    else:
        lines.append(f"""
- **N paired queries**: {h10['n_paired']}
- **Delta mean** (B1 − B0): {_fmt(h10['delta_mean'])}
- **95% Bootstrap CI**: [{_fmt(h10['ci_lower'])}, {_fmt(h10['ci_upper'])}]
""")
        delta = h10["delta_mean"]
        ci_lo = h10["ci_lower"]
        ci_hi = h10["ci_upper"]
        if ci_lo > 0:
            interp = (
                f"The paired 95% CI [{_fmt(ci_lo)}, {_fmt(ci_hi)}] is entirely positive, "
                f"providing evidence that B1 outperforms B0 on the explicit-device slice "
                f"(mean gain = {_fmt(delta)})."
            )
        elif ci_hi < 0:
            interp = (
                f"The paired 95% CI [{_fmt(ci_lo)}, {_fmt(ci_hi)}] is entirely negative, "
                f"indicating B1 underperforms B0 on the explicit-device slice "
                f"(mean loss = {_fmt(delta)})."
            )
        else:
            interp = (
                f"The paired 95% CI [{_fmt(ci_lo)}, {_fmt(ci_hi)}] crosses zero "
                f"(delta = {_fmt(delta)}), so there is insufficient evidence of a systematic "
                f"advantage for B1 over B0 on the explicit-device slice."
            )
        lines.append(f"**Interpretation**: {interp}\n")

    # H11
    lines.append("## H11: Equivalence Test — B3 vs B2, Explicit-Device Slice (adj_cont@5)")
    h11 = results["H11_equivalence"]
    if "error" in h11:
        lines.append(f"\n**Error**: {h11['error']}\n")
    else:
        equiv_str = "YES" if h11["is_equivalent"] else "NO"
        lines.append(f"""
- **N paired queries**: {h11['n_paired']}
- **Delta mean** (B3 − B2): {_fmt(h11['delta_mean'])}
- **90% Bootstrap CI**: [{_fmt(h11['ci_lower_90'])}, {_fmt(h11['ci_upper_90'])}]
- **Equivalence margin**: ±{h11['margin']}
- **Equivalent**: {equiv_str}
""")
        delta = h11["delta_mean"]
        ci_lo = h11["ci_lower_90"]
        ci_hi = h11["ci_upper_90"]
        margin = h11["margin"]
        if h11["is_equivalent"]:
            interp = (
                f"The 90% CI [{_fmt(ci_lo)}, {_fmt(ci_hi)}] falls entirely within the equivalence "
                f"band [−{margin}, +{margin}]. B3 and B2 are statistically equivalent on the "
                f"explicit-device slice at this margin (delta = {_fmt(delta)})."
            )
        else:
            interp = (
                f"The 90% CI [{_fmt(ci_lo)}, {_fmt(ci_hi)}] extends outside the equivalence "
                f"band [−{margin}, +{margin}] (delta = {_fmt(delta)}). Equivalence between B3 "
                f"and B2 cannot be claimed at the {margin} margin."
            )
        lines.append(f"**Interpretation**: {interp}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Paper A stats add-ons: H5, H10, H11")
    parser.add_argument("--baselines-per-query", required=True, help="Path to baselines per_query.csv")
    parser.add_argument("--core-per-query", required=True, help="Path to core per_query.csv")
    parser.add_argument("--out-json", required=True, help="Output JSON path")
    parser.add_argument("--out-md", required=True, help="Output MD path")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260309)
    parser.add_argument("--equiv-margin", type=float, default=0.02)
    args = parser.parse_args()

    baselines_rows = load_csv(args.baselines_per_query)
    # core rows available for future extensions; currently H5/H10/H11 use baselines only
    # core_rows = load_csv(args.core_per_query)

    seed = args.seed
    samples = args.bootstrap_samples
    margin = args.equiv_margin

    h5 = analyse_h5(baselines_rows, seed=seed, samples=samples)
    h10 = analyse_h10(baselines_rows, seed=seed, samples=samples)
    h11 = analyse_h11(baselines_rows, seed=seed, samples=samples, margin=margin)

    results = {
        "H5_cross_slice": h5,
        "H10_paired_bootstrap": h10,
        "H11_equivalence": h11,
        "generated_at": str(date.today()),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON written: {out_json}")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(build_markdown(results), encoding="utf-8")
    print(f"MD written: {out_md}")

    # Print summary to stdout
    print("\n=== H5 Cross-Slice CI ===")
    for k, v in h5.items():
        print(f"  {k}: {v}")
    print("\n=== H10 Paired Bootstrap CI ===")
    for k, v in h10.items():
        print(f"  {k}: {v}")
    print("\n=== H11 Equivalence Test ===")
    for k, v in h11.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
