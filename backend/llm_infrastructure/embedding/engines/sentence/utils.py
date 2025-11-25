"""Utilities for SentenceTransformer engine."""

from __future__ import annotations

import hashlib
import os
from typing import Iterable, List, Sequence

import numpy as np

try:
    import torch
except Exception:  # torch optional
    torch = None  # type: ignore

_gpu_rr = 0


def pick_device(strategy: str | None = None) -> str:
    """GPU 자동 선택 전략 (컨테이너 안전)."""
    if strategy is None or strategy == "auto":
        if torch is not None and torch.cuda.is_available():
            try:
                free_mem = [
                    torch.cuda.mem_get_info(i)[0]
                    for i in range(torch.cuda.device_count())
                ]
                best_gpu = int(max(range(len(free_mem)), key=free_mem.__getitem__))
                return f"cuda:{best_gpu}"
            except Exception:
                # 메모리 조회 실패 시 첫 GPU 또는 CPU로 폴백
                return "cuda:0"
        return "cpu"

    if strategy == "round-robin":
        if torch is None or not torch.cuda.is_available():
            return "cpu"
        global _gpu_rr
        _gpu_rr = (_gpu_rr + 1) % torch.cuda.device_count()
        return f"cuda:{_gpu_rr}"

    # 명시 지정 (CUDA_VISIBLE_DEVICES에 따라 torch가 알아서 제한)
    return strategy


def l2_normalize(vectors: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """L2 정규화."""
    norm = np.linalg.norm(vectors, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return vectors / norm


def chunk_texts(texts: Sequence[str], max_tokens: int = 512, overlap: int = 50) -> List[str]:
    """간단한 토큰 길이 기준 청킹 (단어 단위)."""
    chunks: List[str] = []
    for text in texts:
        words = text.split()
        start = 0
        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start = max(0, end - overlap)
    return chunks


def make_cache_key(texts: Iterable[str], model_name: str, normalize: bool) -> str:
    """텍스트 목록 + 모델명 + 정규화 여부로 해시 키 생성."""
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    digest.update(b"\x01" if normalize else b"\x00")
    for t in texts:
        digest.update(str(t).encode("utf-8"))
    return digest.hexdigest()


__all__ = ["pick_device", "l2_normalize", "chunk_texts", "make_cache_key"]
