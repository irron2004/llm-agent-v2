"""Utility functions for normalization (tokenization, dumping, helpers)."""

from __future__ import annotations

import json
import os
import pickle
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..io_utils import save_json
from .base import STOP_VARIANTS


def tokenize_ko_en(text: str) -> List[str]:
    """한국어/영어/숫자 토큰 추출."""
    if not text:
        return []
    ko_pattern = re.compile(r"[가-힣]+")
    en_pattern = re.compile(r"[a-z0-9]+")
    ko_tokens = ko_pattern.findall(text.lower())
    en_tokens = en_pattern.findall(text.lower())
    all_tokens = list(set(ko_tokens + en_tokens))
    all_tokens.sort()
    return all_tokens


def build_vocabulary(
    texts: List[str], min_freq: int = 1, max_vocab_size: int = 10000
) -> Dict[str, int]:
    """텍스트 컬렉션에서 어휘를 구축합니다."""
    vocab: Dict[str, int] = {}
    for text in texts:
        tokens = tokenize_ko_en(text)
        for token in tokens:
            if token not in STOP_VARIANTS:
                vocab[token] = vocab.get(token, 0) + 1
    filtered = {k: v for k, v in vocab.items() if v >= min_freq}
    sorted_vocab = dict(sorted(filtered.items(), key=lambda x: x[1], reverse=True))
    if max_vocab_size:
        sorted_vocab = dict(list(sorted_vocab.items())[:max_vocab_size])
    return sorted_vocab


def create_buckets(texts: List[str], bucket_size: int = 1000) -> List[List[str]]:
    """텍스트를 버킷으로 나눕니다."""
    return [texts[i : i + bucket_size] for i in range(0, len(texts), bucket_size)]


def _is_picklable(obj: Any) -> bool:
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


def _apply_normalizers_to_text(text: str, doc_id: str, normalizers):
    """단일 문서에 normalizers를 적용한 레코드(dict)를 생성합니다."""
    rec = {"doc_id": str(doc_id), "text": text}
    if isinstance(normalizers, dict):
        for level, fn in normalizers.items():
            rec[f"norm_{level}"] = fn(text)
    else:
        rec["norm_text"] = normalizers(text)
    return rec


def dump_normalization_log(
    docs: Union[List[Tuple[str, str]], List[str]],
    normalizers: Union[Dict[str, Callable[[str], str]], Callable[[str], str]],
    *,
    path: str = "normalized_result.json",
    limit: Optional[int] = None,
    parallel: bool = False,
    workers: Optional[int] = None,
    backend: str = "process",  # "process" | "thread"
    chunksize: int = 64,
) -> None:
    """
    원문과 정규화 결과를 한 파일에 기록합니다.
      - normalizers가 dict면 'norm_L0','norm_L1','norm_L2' 키로 저장
      - 함수 하나면 'norm_text' 키로 저장
      - docs는 ["text", ...] 또는 [("text","doc_id"), ...] 허용
    """
    from tqdm import tqdm

    N = len(docs)
    end = N if limit is None else min(N, limit)

    items: List[Tuple[str, str]] = []
    for i in range(end):
        if isinstance(docs[i], (tuple, list)):
            text, doc_id = docs[i][0], str(docs[i][1])
        else:
            text, doc_id = docs[i], f"doc_{i:06d}"
        items.append((text, doc_id))

    rows: List[Dict[str, str]] = [None] * len(items)

    exec_backend = backend
    if parallel and backend == "process" and not _is_picklable(normalizers):
        exec_backend = "thread"

    if not parallel or len(items) < 1000:
        for idx, (text, doc_id) in enumerate(tqdm(items, desc="Normalizing")):
            rows[idx] = _apply_normalizers_to_text(text, doc_id, normalizers)
        save_json(rows, path)
        return

    max_workers = workers or (os.cpu_count())
    Executor = ProcessPoolExecutor if exec_backend == "process" else ThreadPoolExecutor
    with Executor(max_workers=max_workers) as ex:
        def _work(pair: Tuple[str, str]):
            t, d = pair
            return _apply_normalizers_to_text(t, d, normalizers)

        it = ex.map(_work, items, chunksize=chunksize)
        for idx, rec in enumerate(
            tqdm(it, total=len(items), desc=f"Normalizing[{exec_backend}]")
        ):
            rows[idx] = rec
    save_json(rows, path)


def dump_normalization_log_simple(
    docs: List[Tuple[str, str]],
    normalizer: Callable[[str], str],
    path: str = "normalized_result.json",
    limit: Optional[int] = None,
) -> None:
    """원문/정규화 텍스트를 나란히 덤프하여 비교."""
    out = []
    N = len(docs) if limit is None else min(len(docs), limit)
    for i in range(N):
        text, _ = docs[i]
        out.append({"text": text, "norm_text": normalizer(text)})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


__all__ = [
    "tokenize_ko_en",
    "build_vocabulary",
    "create_buckets",
    "dump_normalization_log",
    "dump_normalization_log_simple",
]
