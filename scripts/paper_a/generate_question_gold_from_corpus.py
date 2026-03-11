from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config.settings import rag_settings, search_settings
from backend.llm_infrastructure.elasticsearch.manager import EsIndexManager
from backend.llm_infrastructure.retrieval.engines.es_search import EsSearchEngine
from backend.services.embedding_service import EmbeddingService
from elasticsearch import Elasticsearch

_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]+")
_NON_KEY_RE = re.compile(r"[^a-z0-9가-힣]+")

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "what",
    "when",
    "where",
    "which",
    "how",
    "관련",
    "절차",
    "방법",
    "설비",
    "장비",
    "확인",
    "점검",
    "작업",
    "가이드",
    "문서",
    "내용",
    "정리",
    "대해",
    "있는",
    "발생",
    "원인",
    "조치",
    "이력",
    "manual",
    "guide",
    "check",
    "all",
    "global",
    "rep",
    "adj",
    "pm",
    "tm",
    "efem",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Paper A questions from corpus docs and "
            "expand gold labels from retrieved answerable docs"
        )
    )
    _ = parser.add_argument(
        "--doc-meta",
        type=str,
        default=".sisyphus/evidence/paper-a/corpus/doc_meta.jsonl",
        help="Path to corpus doc_meta.jsonl",
    )
    _ = parser.add_argument(
        "--corpus-filter",
        type=str,
        default=".sisyphus/evidence/paper-a/corpus/corpus_doc_ids.txt",
        help="Path to corpus whitelist doc IDs",
    )
    _ = parser.add_argument(
        "--out-jsonl",
        type=str,
        default="data/paper_a/eval/query_gold_master_v0_6_generated.jsonl",
        help="Output eval JSONL path",
    )
    _ = parser.add_argument(
        "--out-report",
        type=str,
        default=".sisyphus/evidence/paper-a/reports/T17_generated_gold_expansion_2026-03-09.json",
        help="Output report JSON path",
    )
    _ = parser.add_argument(
        "--doc-types",
        type=str,
        default="sop,ts,setup,myservice,gcb",
        help="Comma-separated doc types to include",
    )
    _ = parser.add_argument(
        "--max-questions",
        type=int,
        default=120,
        help="Maximum number of generated questions",
    )
    _ = parser.add_argument(
        "--question-variants",
        type=int,
        default=1,
        help="Number of question variants per source document",
    )
    _ = parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k retrieval candidates per generated question",
    )
    _ = parser.add_argument(
        "--seed",
        type=int,
        default=20260309,
        help="Random seed",
    )
    _ = parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="ES alias/index override",
    )
    return parser.parse_args()


def _compact(text: str) -> str:
    lowered = str(text or "").strip().lower()
    return _NON_KEY_RE.sub("", lowered)


def _tokenize(text: str) -> set[str]:
    tokens = {tok.lower() for tok in _TOKEN_RE.findall(str(text or ""))}
    filtered: set[str] = set()
    for token in tokens:
        if token in _STOPWORDS:
            continue
        if len(token) < 2:
            continue
        if token.isdigit():
            continue
        filtered.add(token)
    return filtered


def _load_doc_meta(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            loaded = cast(object, json.loads(raw))
            if not isinstance(loaded, dict):
                raise RuntimeError(f"Invalid doc_meta line {line_no}: not object")
            row = cast(dict[str, object], loaded)
            item = {
                "es_doc_id": str(row.get("es_doc_id") or "").strip(),
                "es_device_name": str(row.get("es_device_name") or "").strip(),
                "es_doc_type": str(row.get("es_doc_type") or "").strip(),
                "es_equip_id": str(row.get("es_equip_id") or "").strip().upper(),
                "topic": str(row.get("topic") or "").strip(),
                "source_file": str(row.get("source_file") or "").strip(),
            }
            if not item["es_doc_id"]:
                continue
            rows.append(item)
    if not rows:
        raise RuntimeError(f"No rows loaded from doc_meta: {path}")
    return rows


def _load_corpus_doc_ids(path: Path) -> set[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return {ln for ln in lines if ln}


def _resolve_index(index_override: str | None) -> tuple[str, str]:
    manager = EsIndexManager(
        es_host=search_settings.es_host,
        env=search_settings.es_env,
        index_prefix=search_settings.es_index_prefix,
        es_user=search_settings.es_user or None,
        es_password=search_settings.es_password or None,
        verify_certs=True,
    )
    alias = index_override or manager.get_alias_name()
    target = manager.get_alias_target()
    if not target:
        return alias, alias
    return alias, target


def _build_es_client() -> Elasticsearch:
    kwargs: dict[str, Any] = {"hosts": [search_settings.es_host], "verify_certs": True}
    if search_settings.es_user and search_settings.es_password:
        kwargs["basic_auth"] = (search_settings.es_user, search_settings.es_password)
    return Elasticsearch(**kwargs)


def _question_variants_for_row(
    row: dict[str, str], variant_limit: int
) -> list[tuple[str, str, str, str]]:
    device = row["es_device_name"]
    equip = row["es_equip_id"]
    topic = row["topic"] or row["source_file"] or row["es_doc_id"]
    doc_type = row["es_doc_type"]

    variants: list[tuple[str, str, str, str]] = []

    if equip:
        templates = [
            f"{equip} 장비에서 {topic} 관련 이상 원인과 점검/조치 이력은 무엇인가?",
            f"{equip} 설비에서 {topic} 문제 발생 시 우선 확인할 점검 항목은 무엇인가?",
            f"{equip}에서 {topic} 이슈 재발 방지를 위한 조치 절차는 무엇인가?",
        ]
        for question in templates:
            masked = question.replace(equip, "[EQUIP]")
            variants.append((question, masked, "explicit_equip", "equip"))
    else:
        if doc_type == "ts":
            templates = [
                f"{device} 설비에서 {topic} 트러블슈팅 시 핵심 점검 포인트는 무엇인가?",
                f"{device} 장비에서 {topic} 장애의 원인 후보와 진단 순서는 무엇인가?",
                f"{device} 설비에서 {topic} 문제 발생 시 즉시 조치 절차는 무엇인가?",
            ]
        elif doc_type == "setup":
            templates = [
                f"{device} 설비에서 {topic} 설정/초기 점검 방법은 무엇인가?",
                f"{device} 장비의 {topic} 세팅 시 필수 체크리스트는 무엇인가?",
                f"{device} 설비에서 {topic} 작업 전 준비사항과 검증 항목은 무엇인가?",
            ]
        else:
            templates = [
                f"{device} 설비에서 {topic} 관련 절차와 주의사항은 무엇인가?",
                f"{device} 장비의 {topic} 작업 순서와 핵심 포인트를 알려줘.",
                f"{device} 설비에서 {topic} 수행 시 안전/품질 관점의 체크 항목은 무엇인가?",
            ]

        for question in templates:
            masked = question.replace(device, "[DEVICE]") if device else question
            variants.append((question, masked, "explicit_device", "device"))

    deduped: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()
    for item in variants:
        key = item[0].strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped[: max(1, variant_limit)]


def _split_for_qid(q_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{q_id}|{seed}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "test" if bucket < 20 else "dev"


def _stratified_select(
    rows: list[dict[str, str]], max_questions: int, seed: int
) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["es_doc_type"]].append(row)
    for dtype in groups:
        groups[dtype].sort(key=lambda item: item["es_doc_id"])

    all_rows = list(rows)
    if len(all_rows) <= max_questions:
        return all_rows

    rng = random.Random(seed)
    doc_types = sorted(groups.keys())
    n_types = max(1, len(doc_types))
    base_quota = max_questions // n_types
    remainder = max_questions % n_types

    selected: list[dict[str, str]] = []
    for i, dtype in enumerate(doc_types):
        quota = base_quota + (1 if i < remainder else 0)
        group = groups[dtype]
        if quota >= len(group):
            selected.extend(group)
        else:
            selected.extend(rng.sample(group, quota))
    selected.sort(key=lambda item: item["es_doc_id"])
    return selected[:max_questions]


def _retrieve_candidates(
    *,
    question: str,
    es_engine: EsSearchEngine,
    embed_svc: EmbeddingService,
    base_filter: dict[str, Any],
    top_k: int,
) -> list[dict[str, str | float]]:
    vector = np.asarray(embed_svc.embed_query(question), dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm

    hits = es_engine.hybrid_search(
        query_vector=list(vector),
        query_text=question,
        top_k=top_k,
        dense_weight=0.7,
        sparse_weight=0.3,
        filters=base_filter,
        use_rrf=True,
        rrf_k=60,
    )

    out: list[dict[str, str | float]] = []
    seen: set[str] = set()
    for hit in hits:
        did = str(hit.doc_id or "").strip()
        if not did or did in seen:
            continue
        seen.add(did)
        meta = hit.metadata or {}
        out.append(
            {
                "doc_id": did,
                "score": round(float(hit.score or 0.0), 4),
                "device_name": str(meta.get("device_name") or "").strip(),
                "doc_type": str(meta.get("doc_type") or "").strip(),
                "equip_id": str(meta.get("equip_id") or "").strip().upper(),
                "snippet": str(hit.content or "")[:240],
            }
        )
        if len(out) >= top_k:
            break
    return out


def _judge_candidate(
    *,
    source_doc_id: str,
    source_device: str,
    source_equip: str,
    source_topic: str,
    target_scope: str,
    candidate: dict[str, str | float],
    doc_meta_map: dict[str, dict[str, str]],
) -> tuple[int, str, int]:
    cand_doc_id = str(candidate.get("doc_id") or "")
    cand_device = str(candidate.get("device_name") or "")
    cand_equip = str(candidate.get("equip_id") or "")
    cand_snippet = str(candidate.get("snippet") or "")
    cand_meta = doc_meta_map.get(cand_doc_id, {})
    cand_topic = str(cand_meta.get("topic") or "")

    if cand_doc_id == source_doc_id:
        return 2, "source_doc", 999

    source_tokens = _tokenize(source_topic)
    if not source_tokens:
        source_tokens = _tokenize(source_doc_id)
    cand_tokens = _tokenize(f"{cand_doc_id} {cand_topic} {cand_snippet}")
    overlap = source_tokens & cand_tokens
    overlap_count = len(overlap)

    source_topic_compact = _compact(source_topic)
    cand_blob = _compact(f"{cand_doc_id} {cand_topic} {cand_snippet}")
    topic_compact_hit = bool(source_topic_compact) and source_topic_compact in cand_blob

    topical = overlap_count >= 1 or topic_compact_hit
    same_device = bool(source_device) and _compact(source_device) == _compact(
        cand_device
    )
    same_equip = bool(source_equip) and source_equip.upper() == cand_equip.upper()

    if not topical:
        return 0, "topic_mismatch", overlap_count

    if target_scope == "equip":
        if same_equip:
            return 2, "same_equip_topic_match", overlap_count
        if same_device:
            return 1, "same_device_topic_match", overlap_count
        return 1, "cross_device_topic_match", overlap_count

    if same_device:
        return 2, "same_device_topic_match", overlap_count
    return 1, "cross_device_topic_match", overlap_count


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            _ = f.write(json.dumps(row, ensure_ascii=False))
            _ = f.write("\n")


def run() -> int:
    args = parse_args()
    doc_meta_path = Path(args.doc_meta)
    corpus_filter_path = Path(args.corpus_filter)
    out_jsonl_path = Path(args.out_jsonl)
    out_report_path = Path(args.out_report)
    top_k = int(args.top_k)
    max_questions = int(args.max_questions)
    question_variants = max(1, int(args.question_variants))
    seed = int(args.seed)

    if not doc_meta_path.exists() or not doc_meta_path.is_file():
        print(f"doc-meta not found: {doc_meta_path}", file=sys.stderr)
        return 1
    if not corpus_filter_path.exists() or not corpus_filter_path.is_file():
        print(f"corpus-filter not found: {corpus_filter_path}", file=sys.stderr)
        return 1

    target_doc_types = {
        part.strip().lower() for part in str(args.doc_types).split(",") if part.strip()
    }
    if not target_doc_types:
        print("No doc types selected", file=sys.stderr)
        return 1

    doc_meta_rows = _load_doc_meta(doc_meta_path)
    corpus_doc_ids = _load_corpus_doc_ids(corpus_filter_path)

    filtered_rows = [
        row
        for row in doc_meta_rows
        if row["es_doc_id"] in corpus_doc_ids
        and row["es_doc_type"].lower() in target_doc_types
    ]
    if not filtered_rows:
        print("No rows remained after corpus/doc_type filtering", file=sys.stderr)
        return 1

    source_doc_budget = (max_questions + question_variants - 1) // question_variants
    selected_rows = _stratified_select(
        filtered_rows, max_questions=source_doc_budget, seed=seed
    )
    doc_meta_map = {row["es_doc_id"]: row for row in filtered_rows}

    alias_name, resolved_index = _resolve_index(args.index)
    es_engine = EsSearchEngine(
        es_client=_build_es_client(),
        index_name=resolved_index,
        text_fields=["search_text^1.0", "chunk_summary^0.7", "chunk_keywords^0.8"],
    )
    base_filter = es_engine.build_filter(doc_ids=sorted(corpus_doc_ids))
    if base_filter is None:
        print("Failed to build corpus whitelist filter", file=sys.stderr)
        return 1

    embed_svc = EmbeddingService(
        method=rag_settings.embedding_method,
        version=rag_settings.embedding_version,
        device=rag_settings.embedding_device,
        use_cache=rag_settings.embedding_use_cache,
        cache_dir=rag_settings.embedding_cache_dir,
    )

    output_rows: list[dict[str, object]] = []
    expanded_rows = 0
    added_gold_docs = 0
    strict_only_rows = 0

    q_index = 0
    for row in selected_rows:
        source_doc_id = row["es_doc_id"]
        source_device = row["es_device_name"]
        source_equip = row["es_equip_id"]
        source_topic = row["topic"]
        source_doc_type = row["es_doc_type"]

        variants = _question_variants_for_row(row, question_variants)
        for variant_idx, (
            question,
            question_masked,
            scope_obs,
            target_scope_level,
        ) in enumerate(variants, start=1):
            if len(output_rows) >= max_questions:
                break

            q_index += 1
            q_id = f"A-gen{q_index:04d}"

            retrieved = _retrieve_candidates(
                question=question,
                es_engine=es_engine,
                embed_svc=embed_svc,
                base_filter=cast(dict[str, Any], base_filter),
                top_k=top_k,
            )

            judged: list[dict[str, object]] = []
            loose_gold: set[str] = {source_doc_id}
            strict_gold: set[str] = {source_doc_id}
            for cand in retrieved:
                relevance, reason, overlap_count = _judge_candidate(
                    source_doc_id=source_doc_id,
                    source_device=source_device,
                    source_equip=source_equip,
                    source_topic=source_topic,
                    target_scope=target_scope_level,
                    candidate=cand,
                    doc_meta_map=doc_meta_map,
                )
                cand_doc_id = str(cand.get("doc_id") or "")
                if relevance >= 1 and cand_doc_id:
                    loose_gold.add(cand_doc_id)
                if relevance >= 2 and cand_doc_id:
                    strict_gold.add(cand_doc_id)
                judged.append(
                    {
                        **cand,
                        "relevance": relevance,
                        "judge_reason": reason,
                        "topic_overlap_count": overlap_count,
                    }
                )

            if len(loose_gold) > 1:
                expanded_rows += 1
                added_gold_docs += len(loose_gold) - 1
            if len(strict_gold) == 1 and len(loose_gold) > 1:
                strict_only_rows += 1

            split = _split_for_qid(q_id, seed)
            intent = (
                "troubleshooting"
                if source_doc_type in {"myservice", "gcb", "ts"}
                else "procedure"
            )
            output_row: dict[str, object] = {
                "q_id": q_id,
                "split": split,
                "source": "auto_generated_from_doc_meta_v0_6",
                "question": question,
                "question_masked": question_masked,
                "scope_observability": scope_obs,
                "intent_primary": intent,
                "intent_secondary": "",
                "target_scope_level": target_scope_level,
                "canonical_device_name": source_device,
                "canonical_equip_id": source_equip,
                "allowed_devices": [source_device] if source_device else [],
                "allowed_equips": [source_equip] if source_equip else [],
                "preferred_doc_types": [source_doc_type],
                "acceptable_doc_types": [],
                "gold_doc_ids": sorted(loose_gold),
                "gold_doc_ids_strict": sorted(strict_gold),
                "gold_pages": [],
                "gold_passages": [],
                "answerable_without_context": False,
                "shared_allowed": True,
                "family_allowed": False if target_scope_level == "equip" else True,
                "notes": (
                    f"generated_from={source_doc_id}; topic={source_topic}; "
                    + f"retrieval_top_k={top_k}; variant={variant_idx}; "
                    + f"expanded_gold={len(loose_gold) - 1}"
                ),
                "source_doc_id": source_doc_id,
                "source_topic": source_topic,
                "retrieved_candidates": judged,
            }
            output_rows.append(output_row)

        if len(output_rows) >= max_questions:
            break

    _write_jsonl(out_jsonl_path, output_rows)

    split_counts: dict[str, int] = defaultdict(int)
    scope_counts: dict[str, int] = defaultdict(int)
    avg_gold = 0.0
    for row in output_rows:
        split_counts[str(row["split"])] += 1
        scope_counts[str(row["scope_observability"])] += 1
        avg_gold += float(len(cast(list[object], row["gold_doc_ids"])))
    avg_gold = avg_gold / len(output_rows) if output_rows else 0.0

    report: dict[str, object] = {
        "generated_rows": len(output_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "scope_counts": dict(sorted(scope_counts.items())),
        "doc_types": sorted(target_doc_types),
        "question_variants": question_variants,
        "source_docs_selected": len(selected_rows),
        "max_questions": max_questions,
        "top_k": top_k,
        "expanded_rows": expanded_rows,
        "strict_only_rows": strict_only_rows,
        "added_gold_docs": added_gold_docs,
        "avg_gold_doc_count": round(avg_gold, 4),
        "doc_meta_path": str(doc_meta_path),
        "corpus_filter_path": str(corpus_filter_path),
        "alias": alias_name,
        "resolved_index": resolved_index,
        "output_jsonl": str(out_jsonl_path),
        "seed": seed,
    }
    _write_json(out_report_path, report)

    print(f"Generated rows: {len(output_rows)}")
    print(f"Expanded rows: {expanded_rows}")
    print(f"Added gold docs: {added_gold_docs}")
    print(f"Output JSONL: {out_jsonl_path}")
    print(f"Output report: {out_report_path}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
