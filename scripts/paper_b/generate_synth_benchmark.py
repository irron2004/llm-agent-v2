#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast, override


GENERATOR_VERSION = "synth_benchmark_stability_v1"
DEFAULT_OUT_DIR = "data/synth_benchmarks/stability_bench_v1"

DOC_COUNT = 120
NEAR_DUP_PAIRS = 30
GROUP_COUNT = 60
QUERIES_PER_GROUP = 4
QUERY_COUNT = GROUP_COUNT * QUERIES_PER_GROUP

GROUP_MIN_ABBR = 20
GROUP_MIN_MIXED_LANG = 20
GROUP_MIN_ERROR_CODE = 10
GROUP_MIN_NEAR_DUP = 15

L2_MAX_LCS = 40
L3_MAX_JACCARD_5GRAM = 0.35


@dataclass(frozen=True)
class LeakageViolation(Exception):
    rule: str
    qid: str
    details: str

    @override
    def __str__(self) -> str:
        return f"{self.rule} violation on {self.qid}: {self.details}"


@dataclass(frozen=True)
class CliArgs:
    seed: int
    out: str
    selftest: bool


def _to_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{field_name} must be int")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise RuntimeError(f"{field_name} must be int") from exc
    raise RuntimeError(f"{field_name} must be int")


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _parse_args() -> CliArgs:
    class _ArgsNamespace(argparse.Namespace):
        seed: int = 0
        out: str = DEFAULT_OUT_DIR
        selftest: bool = False

    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic stability benchmark v1"
    )
    _ = parser.add_argument(
        "--seed", type=int, required=True, help="Deterministic RNG seed"
    )
    _ = parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    _ = parser.add_argument(
        "--selftest",
        action="store_true",
        help="Inject a leakage violation and ensure validator fails",
    )
    parsed = parser.parse_args(namespace=_ArgsNamespace())
    return CliArgs(
        seed=_to_int(cast(object, parsed.seed), "seed"),
        out=str(cast(object, parsed.out)),
        selftest=_to_bool(cast(object, parsed.selftest)),
    )


def _doc_id(idx: int) -> str:
    return f"DOC_{idx:04d}"


def _group_id(idx: int) -> str:
    return f"G_{idx:04d}"


def _query_id(group_idx: int, variant_idx: int) -> str:
    absolute_idx = group_idx * QUERIES_PER_GROUP + variant_idx
    return f"Q_{absolute_idx:04d}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            _ = fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _normalize_tokens(text: str) -> list[str]:
    cleaned = " ".join(text.lower().split())
    return cleaned.split(" ") if cleaned else []


def _as_string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise RuntimeError(f"{field_name} must be a list")
    return [str(item) for item in cast(list[object], value)]


def _token_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = _normalize_tokens(text)
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)}


def _jaccard_5gram(left: str, right: str) -> float:
    left_grams = _token_ngrams(left, 5)
    right_grams = _token_ngrams(right, 5)
    union = left_grams | right_grams
    if not union:
        return 0.0
    return len(left_grams & right_grams) / len(union)


def _longest_common_substring_len(left: str, right: str) -> int:
    if not left or not right:
        return 0
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    prev = [0] * (len(shorter) + 1)
    best = 0
    for ch in longer:
        curr = [0] * (len(shorter) + 1)
        for idx, s_ch in enumerate(shorter, start=1):
            if ch == s_ch:
                curr[idx] = prev[idx - 1] + 1
                if curr[idx] > best:
                    best = curr[idx]
        prev = curr
    return best


def _enforce_distribution(groups: list[dict[str, object]]) -> None:
    counts = {
        "abbr": 0,
        "mixed_lang": 0,
        "error_code": 0,
        "near_dup": 0,
    }
    for group in groups:
        tags = set(_as_string_list(group.get("group_tags"), "group_tags"))
        for key in counts:
            if key in tags:
                counts[key] += 1

    if counts["abbr"] < GROUP_MIN_ABBR:
        raise RuntimeError(
            f"Group constraint failed: abbr={counts['abbr']} < {GROUP_MIN_ABBR}"
        )
    if counts["mixed_lang"] < GROUP_MIN_MIXED_LANG:
        raise RuntimeError(
            f"Group constraint failed: mixed_lang={counts['mixed_lang']} < {GROUP_MIN_MIXED_LANG}"
        )
    if counts["error_code"] < GROUP_MIN_ERROR_CODE:
        raise RuntimeError(
            f"Group constraint failed: error_code={counts['error_code']} < {GROUP_MIN_ERROR_CODE}"
        )
    if counts["near_dup"] < GROUP_MIN_NEAR_DUP:
        raise RuntimeError(
            f"Group constraint failed: near_dup={counts['near_dup']} < {GROUP_MIN_NEAR_DUP}"
        )


def _enforce_leakage(
    queries: list[dict[str, object]],
    corpus_by_id: dict[str, dict[str, object]],
) -> None:
    for query_obj in queries:
        qid = str(query_obj["qid"])
        query_text = str(query_obj["query"])
        expected_doc_ids = _as_string_list(
            query_obj.get("expected_doc_ids"), "expected_doc_ids"
        )

        if "DOC_" in query_text or "SYNTH_" in query_text:
            raise LeakageViolation("L1", qid, "query contains forbidden token pattern")

        gold_parts: list[str] = []
        for doc_id in expected_doc_ids:
            doc = corpus_by_id.get(str(doc_id))
            if doc is None:
                raise RuntimeError(f"Unknown expected doc_id in query {qid}: {doc_id}")
            gold_parts.append(str(doc["content"]))
        gold_content = "\n".join(gold_parts)

        lcs_len = _longest_common_substring_len(query_text, gold_content)
        if lcs_len > L2_MAX_LCS:
            raise LeakageViolation(
                "L2",
                qid,
                f"longest common substring length {lcs_len} exceeds {L2_MAX_LCS}",
            )

        jaccard = _jaccard_5gram(query_text, gold_content)
        if jaccard > L3_MAX_JACCARD_5GRAM:
            raise LeakageViolation(
                "L3",
                qid,
                f"5-gram Jaccard {jaccard:.6f} exceeds {L3_MAX_JACCARD_5GRAM}",
            )


def _build_corpus(
    _rng: random.Random,
) -> tuple[list[dict[str, object]], list[tuple[str, str]]]:
    doc_types = ["manual", "troubleshooting", "setup"]
    near_dup_pairs: list[tuple[str, str]] = []
    docs: list[dict[str, object]] = []

    critical_classes = [
        ("abbr", "OHT", "OVERHEAD_TRANSFER"),
        ("error_code", "E-7712", "E-7713"),
        ("module", "MDL_A3", "MDL_B3"),
        ("unit", "8ms", "12ms"),
    ]

    for pair_idx in range(NEAR_DUP_PAIRS):
        crit_tag, token_a, token_b = critical_classes[pair_idx % len(critical_classes)]
        pair_group = f"PAIR_{pair_idx:03d}"
        device = f"SUPRA_{pair_idx % 12:02d}_SYNTH"
        doc_type = doc_types[pair_idx % len(doc_types)]

        fixed = (
            f"{device} maintenance note {pair_group}. "
            "Use canonical reset cadence and verify pressure profile before re-enable. "
            "Korean token 포함: 챔버 seal 확인, then run dry cycle. "
            "Error marker EC-2A7 present in telemetry. "
            "Critical marker"
        )

        content_a = f"{fixed} {token_a}."
        content_b = f"{fixed} {token_b}."

        first_id = _doc_id(pair_idx * 2)
        second_id = _doc_id(pair_idx * 2 + 1)
        near_dup_pairs.append((first_id, second_id))

        tags = ["near_dup", crit_tag, "mixed_lang", "high_risk"]
        docs.append(
            {
                "doc_id": first_id,
                "doc_type": doc_type,
                "device_name": device,
                "content": content_a,
                "tags": tags,
                "chapter": pair_group,
            }
        )
        docs.append(
            {
                "doc_id": second_id,
                "doc_type": doc_type,
                "device_name": device,
                "content": content_b,
                "tags": tags,
                "chapter": pair_group,
            }
        )

    abbr_tokens = ["PM", "RF", "ESC", "MFC", "OHT", "N2", "APC", "TMP"]
    error_tokens = ["E-1101", "E-2044", "E-5500", "E-9091", "ALM-77"]

    for local_idx in range(60):
        doc_idx = 60 + local_idx
        doc_id = _doc_id(doc_idx)
        doc_type = doc_types[(doc_idx + 1) % len(doc_types)]
        device = f"NOVA_{local_idx % 15:02d}_SYNTH"
        abbr = abbr_tokens[local_idx % len(abbr_tokens)]
        err = error_tokens[local_idx % len(error_tokens)]
        mixed = "ko/en mix: 온도 drift check and valve reopen"
        content = (
            f"{device} service chapter CH_{local_idx:03d}. "
            f"Keep {abbr} value stable and track fault {err}. "
            f"{mixed}. "
            "Reference timing window is short and should not be expanded."
        )
        tags = ["mixed_lang"]
        if local_idx % 2 == 0:
            tags.append("abbr")
        if local_idx % 3 == 0:
            tags.append("error_code")
        docs.append(
            {
                "doc_id": doc_id,
                "doc_type": doc_type,
                "device_name": device,
                "content": content,
                "tags": tags,
                "equip_id": f"EQP_{local_idx:03d}",
            }
        )

    if len(docs) != DOC_COUNT:
        raise RuntimeError(f"Corpus size mismatch: {len(docs)} != {DOC_COUNT}")

    return docs, near_dup_pairs


def _group_tags(group_idx: int) -> list[str]:
    tags: list[str] = []
    if group_idx < 20:
        tags.append("abbr")
    if 20 <= group_idx < 40:
        tags.append("mixed_lang")
    if 40 <= group_idx < 50:
        tags.append("error_code")
    if 45 <= group_idx < 60:
        tags.append("near_dup")
    if not tags:
        tags.append("ambiguous")
    return tags


def _build_queries(
    rng: random.Random,
    corpus: list[dict[str, object]],
    near_dup_pairs: list[tuple[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    corpus_by_id: dict[str, dict[str, object]] = {
        str(doc["doc_id"]): doc for doc in corpus
    }
    single_doc_candidates = [
        str(doc["doc_id"])
        for doc in corpus
        if "near_dup" not in _as_string_list(doc.get("tags"), "tags")
    ]
    rng.shuffle(single_doc_candidates)

    groups_meta: list[dict[str, object]] = []
    queries: list[dict[str, object]] = []
    levels = ["low", "mid", "high", "mid"]

    for group_idx in range(GROUP_COUNT):
        gid = _group_id(group_idx)
        tags = _group_tags(group_idx)

        expected: list[str]
        if "near_dup" in tags:
            pair_idx = group_idx - 45
            expected = [near_dup_pairs[pair_idx][0], near_dup_pairs[pair_idx][1]]
            key_doc = corpus_by_id[expected[0]]
        else:
            candidate_idx = group_idx % len(single_doc_candidates)
            expected = [single_doc_candidates[candidate_idx]]
            key_doc = corpus_by_id[expected[0]]

        device_name = str(key_doc["device_name"])
        content_tokens = _normalize_tokens(str(key_doc["content"]))

        abbr_hint = next(
            (tok for tok in content_tokens if tok.isupper() and len(tok) <= 6), "PM"
        )
        err_hint = next(
            (
                tok
                for tok in content_tokens
                if tok.startswith("e-") or tok.startswith("alm-")
            ),
            "E-1101",
        )

        if "abbr" in tags:
            canonical = f"{device_name} {abbr_hint} setting 유지 절차 알려줘"
            variants = [
                f"{device_name}에서 {abbr_hint} 값 유지하려면 어떤 점검을 먼저 해야 하나요?",
                f"Need steps to keep {abbr_hint} stable on {device_name}, 핵심만 정리해줘.",
                f"{abbr_hint} drift 줄이기 위해 {device_name} maintenance 순서가 궁금해.",
                f"{device_name} 장비에서 {abbr_hint} 안정화 루틴을 간단히 설명해줘.",
            ]
        elif "mixed_lang" in tags:
            canonical = f"{device_name} 챔버 pressure check workflow 요약"
            variants = [
                f"{device_name} 챔버 pressure check workflow를 짧게 알려줘.",
                f"For {device_name}, 점검 전에 pressure trend 어떻게 확인해?",
                f"{device_name} 장비에서 chamber pressure drift 대응 flow 알려줘.",
                f"Need ko/en mixed checklist for {device_name} pressure stability.",
            ]
        elif "error_code" in tags:
            canonical = f"{device_name} {err_hint} fault 처리 가이드"
            variants = [
                f"{device_name}에서 {err_hint} fault 뜨면 우선 조치가 뭐야?",
                f"{err_hint} 발생 시 {device_name} recovery sequence 알려줘.",
                f"Need quick triage for {err_hint} on {device_name}, 한국어로 요약.",
                f"{device_name} 장비 {err_hint} 에러 대응 절차 핵심만 정리해줘.",
            ]
        elif "near_dup" in tags:
            canonical = (
                f"{device_name} near-duplicate 정비 문서의 critical marker 차이 확인"
            )
            variants = [
                f"{device_name} 문서 두 개가 거의 같은데 critical marker 차이만 확인하고 싶어.",
                f"Show how to compare near-duplicate maintenance notes for {device_name} quickly.",
                f"{device_name} 유사 문서 pair에서 한 토큰 차이로 바뀌는 의미를 설명해줘.",
                f"Need checklist to disambiguate two almost identical docs for {device_name}.",
            ]
        else:
            canonical = f"{device_name} routine check procedure"
            variants = [
                f"{device_name} routine check 절차 알려줘.",
                f"Need routine maintenance sequence for {device_name}.",
                f"{device_name} 장비 기본 점검 순서를 정리해줘.",
                f"Give a concise check workflow for {device_name}.",
            ]

        groups_meta.append(
            {
                "group_id": gid,
                "group_tags": list(tags),
                "expected_doc_ids": list(expected),
            }
        )

        for variant_idx, query_text in enumerate(variants):
            queries.append(
                {
                    "qid": _query_id(group_idx, variant_idx),
                    "group_id": gid,
                    "canonical_query": canonical,
                    "query": query_text,
                    "paraphrase_level": levels[variant_idx],
                    "expected_doc_ids": list(expected),
                    "tags": list(tags),
                }
            )

    if len(queries) != QUERY_COUNT:
        raise RuntimeError(f"Query size mismatch: {len(queries)} != {QUERY_COUNT}")

    return groups_meta, queries


def _build_manifest(
    seed: int, corpus_path: Path, queries_path: Path
) -> dict[str, object]:
    return {
        "seed": seed,
        "generator_version": GENERATOR_VERSION,
        "files": {
            "corpus.jsonl": _sha256_file(corpus_path),
            "queries.jsonl": _sha256_file(queries_path),
        },
        "counts": {
            "docs": DOC_COUNT,
            "groups": GROUP_COUNT,
            "queries": QUERY_COUNT,
        },
        "leakage_rules": {
            "L1_forbidden_patterns": ["DOC_", "SYNTH_"],
            "L2_max_longest_common_substring": L2_MAX_LCS,
            "L3_max_5gram_jaccard": L3_MAX_JACCARD_5GRAM,
        },
    }


def _generate(
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(seed)
    corpus, near_dup_pairs = _build_corpus(rng)
    groups, queries = _build_queries(rng, corpus, near_dup_pairs)

    _enforce_distribution(groups)
    _enforce_leakage(queries, {str(doc["doc_id"]): doc for doc in corpus})

    return corpus, queries, groups


def _run_selftest(seed: int) -> int:
    corpus, queries, _ = _generate(seed)
    broken = [dict(item) for item in queries]
    broken[0]["query"] = str(broken[0]["query"]) + " DOC_9999"

    try:
        _enforce_leakage(broken, {str(doc["doc_id"]): doc for doc in corpus})
    except LeakageViolation as violation:
        print(f"SELFTEST expected failure: {violation}")
        return 1

    print("SELFTEST failed: leakage validator did not detect injected violation")
    return 2


def main() -> int:
    args = _parse_args()

    if args.selftest:
        return _run_selftest(args.seed)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        corpus, queries, _ = _generate(args.seed)
    except LeakageViolation as violation:
        print(f"FAIL: {violation}")
        return 1
    except Exception as exc:
        print(f"FAIL: generation error: {exc}")
        return 1

    corpus_path = output_dir / "corpus.jsonl"
    queries_path = output_dir / "queries.jsonl"
    manifest_path = output_dir / "manifest.json"

    _write_jsonl(corpus_path, corpus)
    _write_jsonl(queries_path, queries)

    manifest = _build_manifest(args.seed, corpus_path, queries_path)
    _ = manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote corpus: {corpus_path}")
    print(f"Wrote queries: {queries_path}")
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
