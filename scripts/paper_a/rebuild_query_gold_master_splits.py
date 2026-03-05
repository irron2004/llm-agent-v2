from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_a.canonicalize import compact_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild Paper A query_gold_master split assignments (v0.5)"
    )
    _ = parser.add_argument(
        "--in",
        required=True,
        dest="in_path",
        help="Input master eval JSONL (v0.4)",
    )
    _ = parser.add_argument(
        "--out",
        required=True,
        dest="out_path",
        help="Output master eval JSONL (v0.5)",
    )
    _ = parser.add_argument("--seed", type=int, default=20260305, help="Base seed")
    _ = parser.add_argument(
        "--max-seed-tries",
        type=int,
        default=100,
        help="Try seed..seed+N-1 until balance constraints pass",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                loaded = cast(object, json.loads(raw))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at line {line_no}: {exc}") from exc

            if not isinstance(loaded, dict):
                raise RuntimeError(f"Row must be object at line {line_no}")

            row = cast(dict[str, object], loaded)
            if not isinstance(row.get("q_id"), str):
                raise RuntimeError(f"Missing q_id at line {line_no}")
            if not isinstance(row.get("question"), str):
                raise RuntimeError(f"Missing question at line {line_no}")
            if not isinstance(row.get("scope_observability"), str):
                raise RuntimeError(f"Missing scope_observability at line {line_no}")
            rows.append(row)

    if not rows:
        raise RuntimeError("No rows found in input JSONL")
    return rows


def _freeze_input(input_path: Path, frozen_path: Path) -> None:
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, frozen_path)


def _stable_bucket(key: str, seed: int) -> int:
    payload = f"{key}|{seed}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16) % 100


def _select_split(bucket: int) -> str:
    return "test" if bucket < 20 else "dev"


def _build_leak_key(row: dict[str, object]) -> str:
    question_masked = str(row.get("question_masked") or "")
    question = str(row.get("question") or "")
    basis = question_masked if question_masked.strip() else question
    return compact_key(basis)


def _assign_splits(
    rows: list[dict[str, object]], seed: int
) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_leak_key: dict[str, list[int]] = defaultdict(list)
    group_has_empty_gold: dict[str, bool] = {}
    for idx, row in enumerate(rows):
        leak_key = _build_leak_key(row)
        if not leak_key:
            leak_key = f"qid:{str(row.get('q_id') or '')}"
        by_leak_key[leak_key].append(idx)
        gold_doc_ids = row.get("gold_doc_ids", [])
        is_empty_gold = not isinstance(gold_doc_ids, list) or len(gold_doc_ids) == 0
        prev = group_has_empty_gold.get(leak_key, False)
        group_has_empty_gold[leak_key] = prev or is_empty_gold

    out_rows = [dict(row) for row in rows]

    for leak_key, indices in by_leak_key.items():
        if group_has_empty_gold.get(leak_key, False):
            split = "dev"
        else:
            split = _select_split(_stable_bucket(leak_key, seed))
        for idx in indices:
            out_rows[idx]["split"] = split

    split_scope_counter: dict[str, Counter[str]] = defaultdict(Counter)
    split_counter: Counter[str] = Counter()
    scope_counter: Counter[str] = Counter()
    scope_nonempty_gold_counter: Counter[str] = Counter()
    scope_nonempty_gold_test_counter: Counter[str] = Counter()
    split_leak_keys: dict[str, set[str]] = {"dev": set(), "test": set()}

    for row in out_rows:
        split = str(row.get("split") or "")
        scope = str(row.get("scope_observability") or "")
        leak_key = _build_leak_key(row)
        split_counter[split] += 1
        scope_counter[scope] += 1
        split_scope_counter[scope][split] += 1
        gold_doc_ids = row.get("gold_doc_ids", [])
        has_nonempty_gold = isinstance(gold_doc_ids, list) and len(gold_doc_ids) > 0
        if has_nonempty_gold:
            scope_nonempty_gold_counter[scope] += 1
            if split == "test":
                scope_nonempty_gold_test_counter[scope] += 1
        if split in split_leak_keys and leak_key:
            split_leak_keys[split].add(leak_key)

    overlap_leak_keys = sorted(split_leak_keys["dev"] & split_leak_keys["test"])

    report = {
        "seed": seed,
        "total_rows": len(out_rows),
        "groups_forced_dev_due_to_empty_gold": sum(
            1 for value in group_has_empty_gold.values() if value
        ),
        "split_counts": dict(sorted(split_counter.items())),
        "scope_counts": dict(sorted(scope_counter.items())),
        "scope_by_split": {
            scope: dict(sorted(counter.items()))
            for scope, counter in sorted(split_scope_counter.items())
        },
        "scope_nonempty_gold_counts": dict(sorted(scope_nonempty_gold_counter.items())),
        "scope_nonempty_gold_test_counts": dict(
            sorted(scope_nonempty_gold_test_counter.items())
        ),
        "unique_leak_keys_by_split": {
            "dev": len(split_leak_keys["dev"]),
            "test": len(split_leak_keys["test"]),
        },
        "overlap_leak_key_count": len(overlap_leak_keys),
        "overlap_leak_keys_preview": overlap_leak_keys[:20],
    }
    return out_rows, report


def _constraints_ok(report: dict[str, object]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    overlap = int(report.get("overlap_leak_key_count") or 0)
    if overlap > 0:
        errors.append(f"leak_key overlap across dev/test: {overlap}")

    scope_counts = cast(dict[str, object], report.get("scope_counts") or {})
    scope_by_split = cast(dict[str, object], report.get("scope_by_split") or {})
    scope_nonempty_gold_counts = cast(
        dict[str, object], report.get("scope_nonempty_gold_counts") or {}
    )
    scope_nonempty_gold_test_counts = cast(
        dict[str, object], report.get("scope_nonempty_gold_test_counts") or {}
    )

    for scope, total_obj in scope_counts.items():
        _ = int(total_obj)
        nonempty_total = int(scope_nonempty_gold_counts.get(scope, 0) or 0)
        if nonempty_total == 0:
            continue
        expected_min_test = max(1, math.floor(nonempty_total * 0.1))
        scope_stats_obj = scope_by_split.get(scope, {})
        if not isinstance(scope_stats_obj, dict):
            errors.append(f"invalid scope_by_split for {scope}")
            continue
        _ = cast(dict[str, object], scope_stats_obj)
        test_count = int(scope_nonempty_gold_test_counts.get(scope, 0) or 0)
        if test_count < expected_min_test:
            errors.append(
                (
                    f"scope={scope}: test_count={test_count} "
                    f"< min_required={expected_min_test} (nonempty-gold only)"
                )
            )

    return len(errors) == 0, errors


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            _ = f.write(json.dumps(row, ensure_ascii=False))
            _ = f.write("\n")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        _ = f.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> int:
    args = parse_args()
    input_path = Path(cast(str, args.in_path))
    output_path = Path(cast(str, args.out_path))
    base_seed = int(args.seed)
    max_seed_tries = int(args.max_seed_tries)

    if not input_path.exists() or not input_path.is_file():
        print(f"Input JSONL not found: {input_path}", file=sys.stderr)
        return 1
    if max_seed_tries <= 0:
        print("--max-seed-tries must be >= 1", file=sys.stderr)
        return 1

    rows = _load_jsonl(input_path)

    frozen_path = output_path.with_name("query_gold_master_v0_4_frozen.jsonl")
    split_report_path = output_path.with_name(
        "query_gold_master_v0_5_split_report.json"
    )
    failed_report_path = output_path.with_name(
        "query_gold_master_v0_5_split_report_failed.json"
    )

    _freeze_input(input_path, frozen_path)

    best_rows: list[dict[str, object]] | None = None
    best_report: dict[str, object] | None = None
    last_errors: list[str] = []

    for offset in range(max_seed_tries):
        seed_try = base_seed + offset
        candidate_rows, candidate_report = _assign_splits(rows, seed_try)
        ok, errors = _constraints_ok(candidate_report)
        if ok:
            best_rows = candidate_rows
            best_report = candidate_report
            break
        last_errors = errors

    if best_rows is None or best_report is None:
        failed_payload = {
            "input": str(input_path),
            "out": str(output_path),
            "base_seed": base_seed,
            "max_seed_tries": max_seed_tries,
            "errors": last_errors,
        }
        _write_json(failed_report_path, failed_payload)
        print(
            (
                "Failed to satisfy split constraints after seed retries. "
                f"See {failed_report_path}"
            ),
            file=sys.stderr,
        )
        return 1

    _write_jsonl(output_path, best_rows)

    best_report["input_path"] = str(input_path)
    best_report["output_path"] = str(output_path)
    best_report["frozen_input_path"] = str(frozen_path)
    best_report["input_sha256"] = _sha256(input_path)
    best_report["frozen_input_sha256"] = _sha256(frozen_path)
    best_report["output_sha256"] = _sha256(output_path)

    _write_json(split_report_path, best_report)

    print(f"Frozen input: {frozen_path}")
    print(f"Regenerated output: {output_path}")
    print(f"Split report: {split_report_path}")
    print(f"Selected seed: {best_report['seed']}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
