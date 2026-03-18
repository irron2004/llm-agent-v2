"""P6/P7/P7+ soft scoring re-experiment on masked queries (Paper A).

Strategy:
  - Load masked_hybrid_results.json which has B3_masked top_doc_ids per query.
  - Use rank-based base score: base(rank) = 1 / (rank + 1)  (rank 0-indexed)
  - Apply soft penalty:
      P6: score = base - 0.05 * v_scope(doc, target_device)
      P7: score = base - lambda_q * v_scope(doc, target_device)
          where lambda_q = 0.05 * (n_out_of_scope / len(docs))
  - Re-sort by adjusted score, compute contamination@10 and gold_hit@10.
  - Compare: B3_masked, B4_masked, B4.5_masked, P6_masked, P7_masked, P7plus_masked.

Outputs:
  data/paper_a/masked_p6p7_results.json
  Console summary table
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_a._io import JsonValue, read_jsonl, write_json

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MASKED_HYBRID_PATH = ROOT / "data/paper_a/masked_hybrid_results.json"
DOC_SCOPE_PATH = ROOT / ".sisyphus/evidence/paper-a/policy/doc_scope.jsonl"
SHARED_IDS_PATH = ROOT / ".sisyphus/evidence/paper-a/policy/shared_doc_ids.txt"
OUT_PATH = ROOT / "data/paper_a/masked_p6p7_results.json"

TOP_K = 10
LAMBDA_FIXED = 0.05

P7PLUS_T_HIGH = 0.75
P7PLUS_T_MID = 0.55
P7PLUS_SHARED_CAP_HIGH = 0
P7PLUS_SHARED_CAP_MID = 1
P7PLUS_SHARED_CAP_LOW = 2
P7PLUS_EARLY_WINDOW = 5
P7PLUS_CANDIDATE_MAX = 30
P7PLUS_B4_WEIGHT = 1.15
P7PLUS_B45_WEIGHT = 0.90
P7PLUS_LAMBDA_BASE = 0.04
P7PLUS_LAMBDA_SPAN = 0.06
P7PLUS_MU_BASE = 0.01
P7PLUS_MU_SPAN = 0.04
P7PLUS_ETA_BASE = 0.02
P7PLUS_ETA_SPAN = 0.08


# ---------------------------------------------------------------------------
# Policy data loading
# ---------------------------------------------------------------------------
def load_doc_scope(path: Path) -> dict[str, str]:
    """Return {es_doc_id -> es_device_name}."""
    result: dict[str, str] = {}
    for row in read_jsonl(path):
        assert isinstance(row, dict)
        doc_id = str(row.get("es_doc_id") or "").strip()
        device = str(row.get("es_device_name") or "").strip()
        if doc_id:
            result[doc_id] = device
    return result


def load_shared_ids(path: Path) -> set[str]:
    result: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            result.add(line)
    return result


# ---------------------------------------------------------------------------
# v_scope: penalty indicator
# ---------------------------------------------------------------------------
def v_scope(
    doc_id: str,
    target_device: str,
    doc_device_map: dict[str, str],
    shared_ids: set[str],
) -> int:
    """Return 1 if doc is out-of-scope (different device AND not shared), else 0."""
    if not doc_id:
        return 0
    if doc_id in shared_ids:
        return 0
    doc_dev = doc_device_map.get(doc_id, "").strip().upper()
    tgt = target_device.strip().upper()
    return 0 if doc_dev == tgt else 1


# ---------------------------------------------------------------------------
# Soft scoring: apply penalty and re-rank
# ---------------------------------------------------------------------------
def apply_soft_scoring(
    doc_ids: list[str],
    target_device: str,
    doc_device_map: dict[str, str],
    shared_ids: set[str],
    lambda_val: float,
) -> list[str]:
    """Return re-ranked doc_id list after soft penalty.

    base_score(rank) = 1.0 / (rank + 1)  (rank 0-indexed)
    adjusted = base - lambda_val * v_scope(doc_id, target_device)
    """
    scored: list[tuple[str, float]] = []
    for rank, doc_id in enumerate(doc_ids):
        base = 1.0 / (rank + 1)
        penalty = lambda_val * v_scope(
            doc_id, target_device, doc_device_map, shared_ids
        )
        scored.append((doc_id, base - penalty))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in scored]


def lambda_adaptive(
    doc_ids: list[str],
    target_device: str,
    doc_device_map: dict[str, str],
    shared_ids: set[str],
) -> float:
    """lambda_q = LAMBDA_FIXED * (n_out_of_scope / len(docs))."""
    if not doc_ids:
        return LAMBDA_FIXED
    n_out = sum(
        1 for d in doc_ids if v_scope(d, target_device, doc_device_map, shared_ids) == 1
    )
    return LAMBDA_FIXED * (n_out / len(doc_ids))


def _dedupe_keep_order(doc_ids: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for doc_id in doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def _rank_score_map(doc_ids: list[str], weight: float) -> dict[str, float]:
    score_map: dict[str, float] = {}
    for rank, doc_id in enumerate(doc_ids):
        score = weight / (rank + 1)
        prev = score_map.get(doc_id)
        if prev is None or score > prev:
            score_map[doc_id] = score
    return score_map


def scope_confidence_proxy(scope_observability: str) -> float:
    scope = scope_observability.strip().lower()
    if scope == "explicit_device":
        return 0.85
    if scope == "explicit_equip":
        return 0.60
    if scope == "implicit":
        return 0.45
    return 0.50


def build_p7plus_ranking(
    *,
    b3_doc_ids: list[str],
    b4_doc_ids: list[str],
    b45_doc_ids: list[str],
    target_device: str,
    scope_observability: str,
    doc_device_map: dict[str, str],
    shared_ids: set[str],
) -> tuple[list[str], dict[str, Any]]:
    conf = scope_confidence_proxy(scope_observability)
    if conf >= P7PLUS_T_HIGH:
        seed = b4_doc_ids + b3_doc_ids + b45_doc_ids
        shared_cap = P7PLUS_SHARED_CAP_HIGH
    elif conf >= P7PLUS_T_MID:
        seed = b4_doc_ids + b45_doc_ids + b3_doc_ids
        shared_cap = P7PLUS_SHARED_CAP_MID
    else:
        seed = b3_doc_ids + b45_doc_ids + b4_doc_ids
        shared_cap = P7PLUS_SHARED_CAP_LOW

    candidates = _dedupe_keep_order(seed)[:P7PLUS_CANDIDATE_MAX]

    b3_score_map = _rank_score_map(b3_doc_ids, 1.0)
    b4_score_map = _rank_score_map(b4_doc_ids, P7PLUS_B4_WEIGHT)
    b45_score_map = _rank_score_map(b45_doc_ids, P7PLUS_B45_WEIGHT)

    lambda_q = P7PLUS_LAMBDA_BASE + P7PLUS_LAMBDA_SPAN * (1.0 - conf)
    mu_q = P7PLUS_MU_BASE + P7PLUS_MU_SPAN * conf
    eta_q = P7PLUS_ETA_BASE + P7PLUS_ETA_SPAN * conf

    scored: list[tuple[str, float, int]] = []
    tgt = target_device.strip().upper()
    for doc_id in candidates:
        base = max(
            b3_score_map.get(doc_id, 0.0),
            b4_score_map.get(doc_id, 0.0),
            b45_score_map.get(doc_id, 0.0),
        )
        oos = v_scope(doc_id, target_device, doc_device_map, shared_ids)
        is_shared = 1 if doc_id in shared_ids else 0
        doc_dev = doc_device_map.get(doc_id, "").strip().upper()
        is_target = 1 if doc_dev and doc_dev == tgt else 0
        score = base - lambda_q * oos - mu_q * is_shared + eta_q * is_target
        scored.append((doc_id, score, is_shared))

    scored.sort(key=lambda x: x[1], reverse=True)

    selected: list[str] = []
    deferred: list[tuple[str, float, int]] = []
    shared_in_early = 0
    for doc_id, score, is_shared in scored:
        if len(selected) >= TOP_K:
            break

        in_early_window = len(selected) < P7PLUS_EARLY_WINDOW
        if in_early_window and is_shared == 1 and shared_in_early >= shared_cap:
            deferred.append((doc_id, score, is_shared))
            continue

        selected.append(doc_id)
        if len(selected) <= P7PLUS_EARLY_WINDOW and is_shared == 1:
            shared_in_early += 1

    if len(selected) < TOP_K:
        for doc_id, _, _ in deferred:
            if doc_id in selected:
                continue
            selected.append(doc_id)
            if len(selected) >= TOP_K:
                break

    if len(selected) < TOP_K:
        for doc_id, _, _ in scored:
            if doc_id in selected:
                continue
            selected.append(doc_id)
            if len(selected) >= TOP_K:
                break

    diag: dict[str, Any] = {
        "conf_proxy": round(conf, 4),
        "lambda_p7plus": round(lambda_q, 6),
        "mu_p7plus": round(mu_q, 6),
        "eta_p7plus": round(eta_q, 6),
        "shared_cap": shared_cap,
        "candidate_count": len(candidates),
    }
    return selected[:TOP_K], diag


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_contamination(
    doc_ids: list[str],
    target_device: str,
    doc_device_map: dict[str, str],
    shared_ids: set[str],
    k: int,
) -> float:
    """cont@k = |{d in top-k : out_of_scope}| / k"""
    top_k = doc_ids[:k]
    if not top_k:
        return 0.0
    contaminated = sum(
        1 for d in top_k if v_scope(d, target_device, doc_device_map, shared_ids) == 1
    )
    return contaminated / k


def compute_gold_hit(doc_ids: list[str], gold_ids: list[str], k: int) -> bool:
    gold_set = set(gold_ids)
    top_k_set = set(doc_ids[:k])
    return bool(gold_set & top_k_set)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run() -> None:
    print("Loading policy data...")
    doc_device_map = load_doc_scope(DOC_SCOPE_PATH)
    shared_ids = load_shared_ids(SHARED_IDS_PATH)
    print(f"  doc_scope: {len(doc_device_map)} entries, shared: {len(shared_ids)}")

    print(f"Loading masked hybrid results from {MASKED_HYBRID_PATH} ...")
    with MASKED_HYBRID_PATH.open(encoding="utf-8") as f:
        masked_hybrid: list[dict[str, Any]] = json.load(f)
    print(f"  Loaded {len(masked_hybrid)} query entries.")

    per_query: list[dict[str, Any]] = []

    for qi, q in enumerate(masked_hybrid):
        q_id = str(q.get("q_id") or qi)
        target_device = str(q.get("target_device") or "")
        scope_obs = str(q.get("scope_observability") or "")
        gold_ids_loose = [str(g) for g in (q.get("gold_ids_loose") or [])]
        gold_ids_strict = [str(g) for g in (q.get("gold_ids_strict") or [])]

        if not target_device or not gold_ids_loose:
            continue

        conditions = q.get("conditions") or {}

        # Grab existing conditions B3/B4/B4.5 masked results
        b3_entry = conditions.get("B3_masked") or {}
        b4_entry = conditions.get("B4_masked") or {}
        b45_entry = conditions.get("B4.5_masked") or {}

        b3_doc_ids: list[str] = [str(d) for d in (b3_entry.get("top_doc_ids") or [])]
        b4_doc_ids: list[str] = [str(d) for d in (b4_entry.get("top_doc_ids") or [])]
        b45_doc_ids: list[str] = [str(d) for d in (b45_entry.get("top_doc_ids") or [])]

        # P6: fixed lambda on B3 base scores
        p6_doc_ids = apply_soft_scoring(
            b3_doc_ids, target_device, doc_device_map, shared_ids, LAMBDA_FIXED
        )

        # P7: adaptive lambda on B3 base scores
        lam_q = lambda_adaptive(b3_doc_ids, target_device, doc_device_map, shared_ids)
        p7_doc_ids = apply_soft_scoring(
            b3_doc_ids, target_device, doc_device_map, shared_ids, lam_q
        )

        p7plus_doc_ids, p7plus_diag = build_p7plus_ranking(
            b3_doc_ids=b3_doc_ids,
            b4_doc_ids=b4_doc_ids,
            b45_doc_ids=b45_doc_ids,
            target_device=target_device,
            scope_observability=scope_obs,
            doc_device_map=doc_device_map,
            shared_ids=shared_ids,
        )

        def make_cond_result(doc_ids: list[str]) -> dict[str, Any]:
            return {
                "cont@10": compute_contamination(
                    doc_ids, target_device, doc_device_map, shared_ids, TOP_K
                ),
                "gold_hit_loose": compute_gold_hit(doc_ids, gold_ids_loose, TOP_K),
                "gold_hit_strict": compute_gold_hit(doc_ids, gold_ids_strict, TOP_K),
                "top_doc_ids": doc_ids[:TOP_K],
            }

        q_result: dict[str, Any] = {
            "q_id": q_id,
            "target_device": target_device,
            "scope_observability": scope_obs,
            "gold_ids_loose": gold_ids_loose,
            "gold_ids_strict": gold_ids_strict,
            "lambda_p7": round(lam_q, 6),
            "lambda_p7plus": p7plus_diag["lambda_p7plus"],
            "mu_p7plus": p7plus_diag["mu_p7plus"],
            "eta_p7plus": p7plus_diag["eta_p7plus"],
            "conf_p7plus": p7plus_diag["conf_proxy"],
            "shared_cap_p7plus": p7plus_diag["shared_cap"],
            "conditions": {
                "B3_masked": {
                    "cont@10": b3_entry.get("cont@10", 0.0),
                    "gold_hit_loose": b3_entry.get("gold_hit_loose", False),
                    "gold_hit_strict": b3_entry.get("gold_hit_strict", False),
                    "top_doc_ids": b3_doc_ids[:TOP_K],
                },
                "B4_masked": {
                    "cont@10": b4_entry.get("cont@10", 0.0),
                    "gold_hit_loose": b4_entry.get("gold_hit_loose", False),
                    "gold_hit_strict": b4_entry.get("gold_hit_strict", False),
                    "top_doc_ids": [
                        str(d) for d in (b4_entry.get("top_doc_ids") or [])
                    ],
                },
                "B4.5_masked": {
                    "cont@10": b45_entry.get("cont@10", 0.0),
                    "gold_hit_loose": b45_entry.get("gold_hit_loose", False),
                    "gold_hit_strict": b45_entry.get("gold_hit_strict", False),
                    "top_doc_ids": [
                        str(d) for d in (b45_entry.get("top_doc_ids") or [])
                    ],
                },
                "P6_masked": make_cond_result(p6_doc_ids),
                "P7_masked": make_cond_result(p7_doc_ids),
                "P7plus_masked": make_cond_result(p7plus_doc_ids),
            },
        }
        per_query.append(q_result)

    print(f"\nSaving {len(per_query)} query results to {OUT_PATH} ...")
    serializable_rows: list[JsonValue] = [cast(JsonValue, row) for row in per_query]
    write_json(OUT_PATH, serializable_rows)
    print("Saved.")

    _print_summary(per_query)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
CONDITIONS_ORDER = [
    "B3_masked",
    "B4_masked",
    "B4.5_masked",
    "P6_masked",
    "P7_masked",
    "P7plus_masked",
]


def _print_summary(per_query: list[dict[str, Any]]) -> None:
    n_total = len(per_query)

    agg: dict[str, dict[str, list[float]]] = {
        c: {"cont": [], "gold_strict": [], "gold_loose": []} for c in CONDITIONS_ORDER
    }

    for q in per_query:
        for cond in CONDITIONS_ORDER:
            cdata = q.get("conditions", {}).get(cond)
            if not cdata or "error" in cdata:
                continue
            agg[cond]["cont"].append(float(cdata.get("cont@10") or 0.0))
            agg[cond]["gold_strict"].append(
                1.0 if cdata.get("gold_hit_strict") else 0.0
            )
            agg[cond]["gold_loose"].append(1.0 if cdata.get("gold_hit_loose") else 0.0)

    def _row(cond: str, data: dict[str, list[float]], n_ref: int) -> str:
        n = len(data["cont"])
        if n == 0:
            return f"  {cond:<20}  {'N/A':>8}  {'N/A':>14}  {'N/A':>12}"
        cont_avg = sum(data["cont"]) / n
        gs = int(sum(data["gold_strict"]))
        gl = int(sum(data["gold_loose"]))
        return f"  {cond:<20}  {cont_avg:>8.3f}  {gs:>5}/{n_ref:<5}    {gl:>5}/{n_ref}"

    print(f"\n{'=' * 72}")
    print(f"P6/P7 Soft Scoring on Masked Queries  (n={n_total})")
    print(f"  P6: lambda=0.05 (fixed)   P7: lambda=adaptive (contamination-based)")
    print(f"{'=' * 72}")
    print(f"{'condition':<22}  {'cont@10':>8}  {'gold_strict':>14}  {'gold_loose':>12}")
    print("-" * 68)
    for cond in CONDITIONS_ORDER:
        print(_row(cond, agg[cond], n_total))

    # Delta vs B3
    print("\n--- Delta vs B3_masked ---")
    b3_cont = agg["B3_masked"]["cont"]
    b3_cont_avg = sum(b3_cont) / len(b3_cont) if b3_cont else 0.0
    b3_gs = (
        sum(agg["B3_masked"]["gold_strict"]) / len(agg["B3_masked"]["gold_strict"])
        if agg["B3_masked"]["gold_strict"]
        else 0.0
    )
    b3_gl = (
        sum(agg["B3_masked"]["gold_loose"]) / len(agg["B3_masked"]["gold_loose"])
        if agg["B3_masked"]["gold_loose"]
        else 0.0
    )
    print(
        f"  {'condition':<20}  {'Δcont@10':>10}  {'Δgold_strict':>14}  {'Δgold_loose':>12}"
    )
    print("-" * 68)
    for cond in CONDITIONS_ORDER[1:]:
        data = agg[cond]
        n = len(data["cont"])
        if n == 0:
            continue
        c_avg = sum(data["cont"]) / n
        gs_avg = sum(data["gold_strict"]) / n
        gl_avg = sum(data["gold_loose"]) / n
        dc = c_avg - b3_cont_avg
        dgs = gs_avg - b3_gs
        dgl = gl_avg - b3_gl
        sign_c = "+" if dc >= 0 else ""
        sign_gs = "+" if dgs >= 0 else ""
        sign_gl = "+" if dgl >= 0 else ""
        print(
            f"  {cond:<20}  {sign_c}{dc:>9.3f}  "
            f"{sign_gs}{dgs:>13.3f}  {sign_gl}{dgl:>11.3f}"
        )

    # By scope_observability
    scope_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in per_query:
        scope_groups[q.get("scope_observability") or "unknown"].append(q)

    for scope, rows in sorted(scope_groups.items()):
        ns = len(rows)
        print(f"\n--- scope={scope} (n={ns}) ---")
        print(
            f"{'condition':<22}  {'cont@10':>8}  {'gold_strict':>14}  {'gold_loose':>12}"
        )
        print("-" * 68)
        for cond in CONDITIONS_ORDER:
            conts, gss, gls = [], [], []
            for q in rows:
                cdata = q.get("conditions", {}).get(cond)
                if not cdata or "error" in cdata:
                    continue
                conts.append(float(cdata.get("cont@10") or 0.0))
                gss.append(1.0 if cdata.get("gold_hit_strict") else 0.0)
                gls.append(1.0 if cdata.get("gold_hit_loose") else 0.0)
            if not conts:
                continue
            print(
                f"  {cond:<20}  {sum(conts) / len(conts):>8.3f}  "
                f"{int(sum(gss)):>5}/{ns:<5}    "
                f"{int(sum(gls)):>5}/{ns}"
            )

    # P7 lambda distribution
    lam_vals = [
        q.get("lambda_p7", 0.0) for q in per_query if q.get("lambda_p7") is not None
    ]
    if lam_vals:
        lam_nonzero = [v for v in lam_vals if v > 0]
        print(f"\n--- P7 adaptive lambda stats ---")
        print(
            f"  mean={sum(lam_vals) / len(lam_vals):.5f}  "
            f"max={max(lam_vals):.5f}  "
            f"n_nonzero={len(lam_nonzero)}/{len(lam_vals)}"
        )

    p7plus_lam_vals = [
        float(q.get("lambda_p7plus", 0.0))
        for q in per_query
        if q.get("lambda_p7plus") is not None
    ]
    p7plus_mu_vals = [
        float(q.get("mu_p7plus", 0.0))
        for q in per_query
        if q.get("mu_p7plus") is not None
    ]
    p7plus_eta_vals = [
        float(q.get("eta_p7plus", 0.0))
        for q in per_query
        if q.get("eta_p7plus") is not None
    ]
    p7plus_conf_vals = [
        float(q.get("conf_p7plus", 0.0))
        for q in per_query
        if q.get("conf_p7plus") is not None
    ]
    if p7plus_lam_vals and p7plus_mu_vals and p7plus_eta_vals and p7plus_conf_vals:
        print(f"\n--- P7+ parameter stats ---")
        print(
            f"  lambda_mean={sum(p7plus_lam_vals) / len(p7plus_lam_vals):.5f}  "
            f"mu_mean={sum(p7plus_mu_vals) / len(p7plus_mu_vals):.5f}  "
            f"eta_mean={sum(p7plus_eta_vals) / len(p7plus_eta_vals):.5f}"
        )
        print(
            f"  conf_mean={sum(p7plus_conf_vals) / len(p7plus_conf_vals):.5f}  "
            f"conf_min={min(p7plus_conf_vals):.5f}  "
            f"conf_max={max(p7plus_conf_vals):.5f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()
