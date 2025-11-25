"""Base normalization primitives (L0/L1/L2) and variant utilities."""

from __future__ import annotations

import functools
import hashlib
import json
import re
from enum import IntEnum
from typing import Callable, Dict, Tuple

# Normalization levels enum
class NormLevel(IntEnum):
    """Normalization levels for text preprocessing."""

    L0 = 0  # Basic text normalization
    L1 = 1  # L0 + synonym/variant mapping
    L2 = 2  # L1 + domain-specific hooks
    L3 = 3  # Semiconductor domain specialized (recommended)
    L4 = 4  # Advanced entity extraction
    L5 = 5  # Enhanced variant mapping


# 1) 변형어로 무시할 쓰레기 토큰
STOP_VARIANTS = {
    "-", "—", "–", "―", "", "N/A", "n/a", "na", "없음", "해당없음", "해당 없음", "미정", "x", "X"
}

# 2) 흔한 기호표기를 표준화
VARIANT_REWRITE = {
    "µm": "um",
    "μm": "um",
    "Å": "angstrom",
    "Å": "angstrom",
    "°C": "celsius",
    "°F": "fahrenheit",
    "±": "plus_minus",
    "≤": "less_equal",
    "≥": "greater_equal",
    "≠": "not_equal",
    "≈": "approximately",
    "→": "arrow",
    "←": "arrow",
    "↔": "arrow",
}

# 모호/일반어는 변형어에서 제외
_AMBIGUOUS = {
    "test",
    "check",
    "clear",
    "run",
    "set",
    "replace",
    "issue",
    "request",
    "result",
    "analysis",
    "data",
    "테스트",
    "점검",
    "확인",
    "교체",
    "요청",
    "완료",
    "현상",
    "분석",
    "데이터",
    "진행",
}


def normalize_text(text: str, *, keep_newlines: bool = True) -> str:
    """
    텍스트 표준화 (기호 통일 + 공백 정리 + 영문 소문자화).
    keep_newlines=True 이면 줄바꿈(\n)을 보존합니다.
    """
    if not text:
        return ""

    # 1) 기호 표준화
    for old, new in VARIANT_REWRITE.items():
        text = text.replace(old, new)

    # 2) 공백 정리
    if keep_newlines:
        text = re.sub(r"[ \t\f\v]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text).strip()
    else:
        text = re.sub(r"\s+", " ", text).strip()

    # 3) 영문 소문자화
    text = re.sub(r"[A-Z]", lambda m: m.group(0).lower(), text)
    return text


# ---------- 변형어 치환기 ----------
def _digest_variant_map(variant_map: Dict[str, str]) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
    items = tuple(
        sorted(
            (k.strip(), v.strip())
            for k, v in variant_map.items()
            if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip()
        )
    )
    variants = tuple(k for k, _ in items)
    canonicals = tuple(v for _, v in items)
    h = hashlib.sha256(json.dumps(items, ensure_ascii=False).encode("utf-8")).hexdigest()
    return h, variants, canonicals


@functools.lru_cache(maxsize=8)
def _compile_variant_replacer(
    variant_map_digest: str, variants: Tuple[str, ...], canonicals: Tuple[str, ...]
) -> Tuple[re.Pattern, Callable[[re.Match], str]]:
    # 길이 긴 variant부터 매칭(겹침 방지)
    pairs = sorted(zip(variants, canonicals), key=lambda x: len(x[0]), reverse=True)
    alts = [re.escape(v) for v, _ in pairs]
    boundary = r"(?<![가-힣A-Za-z0-9])(?:%s)(?![가-힣A-Za-z0-9])" % "|".join(alts)
    regex = re.compile(boundary, flags=re.IGNORECASE)
    lower_map = {v.lower(): c for v, c in pairs}

    def _repl(m: re.Match) -> str:
        return lower_map.get(m.group(0).lower(), m.group(0))

    return regex, _repl


def clean_variants(text: str, variant_map: Dict[str, str]) -> str:
    """변형어를 표준형으로 변환합니다."""
    if not text or not variant_map:
        return text or ""

    digest, variants, canonicals = _digest_variant_map(variant_map)
    regex, repl = _compile_variant_replacer(digest, variants, canonicals)
    return regex.sub(repl, text)


def clean_variants_fast(text: str, variant_map: Dict[str, str]) -> str:
    """
    대규모 사전을 1회 정규식으로 빠르게/안전하게 치환.
    성능과 안정성을 모두 고려한 최적화된 치환기입니다.
    """
    if not text or not variant_map:
        return text or ""

    variants = sorted(variant_map.keys(), key=len, reverse=True)
    boundary = r"[가-힣A-Za-z0-9]"
    pat = re.compile(
        rf"(?<!{boundary})(?:{'|'.join(map(re.escape, variants))})(?!{boundary})",
        flags=re.IGNORECASE,
    )
    lower_map = {k.lower(): v for k, v in variant_map.items()}
    return pat.sub(lambda m: lower_map.get(m.group(0).lower(), m.group(0)), text)


def sanitize_variant_map(variant_map: Dict[str, str]) -> Dict[str, str]:
    """
    화이트리스트 중심의 안전한 사전만 남깁니다.
    모호한 일반어, 구두점, 빈 문자열은 제거합니다.
    """
    safe = {}
    for k, v in (variant_map or {}).items():
        if not k or not v:
            continue
        vk = str(k).strip().lower()
        vv = str(v).strip().lower()

        if vk in _AMBIGUOUS or vk in STOP_VARIANTS or vk in {"-", "_"}:
            continue

        # 약어(대문자/숫자 2~9자) 또는 길이>=4 용어만 허용
        if not re.fullmatch(r"[A-Z0-9\-]{2,9}", k) and len(vk) < 4:
            continue

        safe[vk] = vv
    return safe


__all__ = [
    "NormLevel",
    "STOP_VARIANTS",
    "VARIANT_REWRITE",
    "normalize_text",
    "clean_variants",
    "clean_variants_fast",
    "sanitize_variant_map",
]
