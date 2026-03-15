from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

MASKED_HYBRID_PATH = ROOT / "data/paper_a/masked_hybrid_results.json"
DOC_SCOPE_PATH = ROOT / ".sisyphus/evidence/paper-a/policy/doc_scope.jsonl"
SHARED_IDS_PATH = ROOT / ".sisyphus/evidence/paper-a/policy/shared_doc_ids.txt"
OUT_MD_PATH = (
    ROOT
    / "docs/papers/20_paper_a_scope/evidence/2026-03-14_b45_failure_decomposition.md"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _read_lines(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _classify_case(
    top_doc_ids: list[str],
    target_device: str,
    doc_device_map: dict[str, str],
    shared_ids: set[str],
) -> tuple[str, dict[str, int]]:
    shared_docs = 0
    target_docs = 0
    oos_docs = 0
    unknown_docs = 0

    for doc_id in top_doc_ids:
        if doc_id in shared_ids:
            shared_docs += 1
            continue
        doc_device = doc_device_map.get(doc_id, "").strip()
        if not doc_device:
            unknown_docs += 1
        elif doc_device == target_device:
            target_docs += 1
        else:
            oos_docs += 1

    stats = {
        "shared_docs": shared_docs,
        "target_docs": target_docs,
        "oos_docs": oos_docs,
        "unknown_docs": unknown_docs,
        "n_docs": len(top_doc_ids),
    }

    if shared_docs >= 5:
        return "shared_overload", stats
    if oos_docs > 0:
        return "wrong_shared_or_oos_intrusion", stats
    if len(top_doc_ids) < 10:
        return "sparse_result_set", stats
    return "ranking_dilution", stats


def main() -> None:
    masked_results = _read_json(MASKED_HYBRID_PATH)
    if not isinstance(masked_results, list):
        raise ValueError("masked_hybrid_results.json must contain a list")

    doc_scope_rows = _read_jsonl(DOC_SCOPE_PATH)
    shared_ids = _read_lines(SHARED_IDS_PATH)
    doc_device_map = {
        str(row.get("es_doc_id") or "").strip(): str(
            row.get("es_device_name") or ""
        ).strip()
        for row in doc_scope_rows
        if str(row.get("es_doc_id") or "").strip()
    }

    total_queries = 0
    b4_loose_hits = 0
    b45_loose_hits = 0
    paradox_cases: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    category_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in masked_results:
        if not isinstance(row, dict):
            continue
        total_queries += 1
        conditions = row.get("conditions")
        if not isinstance(conditions, dict):
            continue

        b4 = conditions.get("B4_masked")
        b45 = conditions.get("B4.5_masked")
        if not isinstance(b4, dict) or not isinstance(b45, dict):
            continue

        b4_hit = bool(b4.get("gold_hit_loose"))
        b45_hit = bool(b45.get("gold_hit_loose"))
        if b4_hit:
            b4_loose_hits += 1
        if b45_hit:
            b45_loose_hits += 1

        if not (b4_hit and not b45_hit):
            continue

        target_device = str(row.get("target_device") or "").strip()
        b45_top_doc_ids = [str(v) for v in (b45.get("top_doc_ids") or []) if str(v)]
        category, stats = _classify_case(
            b45_top_doc_ids, target_device, doc_device_map, shared_ids
        )
        category_counts[category] += 1

        case = {
            "q_id": str(row.get("q_id") or ""),
            "scope_observability": str(row.get("scope_observability") or ""),
            "target_device": target_device,
            "category": category,
            "b4_top_doc_ids": [str(v) for v in (b4.get("top_doc_ids") or []) if str(v)][
                :5
            ],
            "b45_top_doc_ids": b45_top_doc_ids[:5],
            **stats,
        }
        paradox_cases.append(case)
        if len(category_examples[category]) < 3:
            category_examples[category].append(case)

    paradox_total = len(paradox_cases)
    estimated_fix_category = (
        category_counts.most_common(1)[0][0] if paradox_total else None
    )
    estimated_gain = (
        category_counts[estimated_fix_category] / total_queries
        if estimated_fix_category
        else 0.0
    )

    lines = [
        "# B4.5 Failure Decomposition (2026-03-14)",
        "",
        "Date: 2026-03-14",
        "Status: generated from `scripts/paper_a/analyze_b45_failure_decomposition.py`",
        "",
        "## Inputs",
        "",
        f"- Results: `{MASKED_HYBRID_PATH.relative_to(ROOT)}`",
        f"- Doc scope: `{DOC_SCOPE_PATH.relative_to(ROOT)}`",
        f"- Shared doc ids: `{SHARED_IDS_PATH.relative_to(ROOT)}`",
        "",
        "## Command",
        "",
        "```bash",
        "cd /home/hskim/work/llm-agent-v2",
        "uv run python scripts/paper_a/analyze_b45_failure_decomposition.py",
        "```",
        "",
        "## Summary",
        "",
        f"- Total masked queries analyzed: {total_queries}",
        f"- B4 loose hit@10: {b4_loose_hits}/{total_queries} ({b4_loose_hits / total_queries:.1%})",
        f"- B4.5 loose hit@10: {b45_loose_hits}/{total_queries} ({b45_loose_hits / total_queries:.1%})",
        f"- Paradox cases (B4 hit, B4.5 miss): {paradox_total}/{total_queries} ({paradox_total / total_queries:.1%})",
        "",
        "## Category Breakdown",
        "",
    ]

    for category, count in category_counts.most_common():
        lines.append(
            f"- {category}: {count}/{paradox_total} ({count / paradox_total:.1%})"
        )

    lines.extend(["", "## Representative Cases", ""])
    for category, examples in category_examples.items():
        lines.append(f"### {category}")
        lines.append("")
        for case in examples:
            lines.append(
                "- "
                + f"`{case['q_id']}` ({case['scope_observability']}, {case['target_device']}): "
                + f"shared={case['shared_docs']}, target={case['target_docs']}, oos={case['oos_docs']}, "
                + f"top5 B4.5={case['b45_top_doc_ids']}"
            )
        lines.append("")

    lines.extend(
        [
            "## Proposed Policy Fix",
            "",
            "- Recommendation: when a target device is already known, rank target-device docs before shared docs and cap shared-doc exposure in the early top-k window.",
        ]
    )
    if estimated_fix_category is not None:
        lines.append(
            "- "
            + f"If the dominant category (`{estimated_fix_category}`) were fully recovered to B4-level behavior, the expected loose hit@10 gain would be about +{estimated_gain * 100:.1f}%p on the full masked set."
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- This decomposition uses the masked hybrid result set because it provides per-query top_doc_ids for B4 and B4.5.",
            "- The same recall inversion pattern (B4.5 < B4) also appears in the broader 2026-03-14 execution narrative, so this analysis is intended as a diagnosis aid, not a final claim by itself.",
        ]
    )

    OUT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved failure decomposition to: {OUT_MD_PATH}")


if __name__ == "__main__":
    main()
