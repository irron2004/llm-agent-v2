"""Lightweight text quality heuristics for noisy OCR/VLM chunks."""

from __future__ import annotations

import re


def strip_noisy_lines(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    kept = []
    for line in lines:
        if not line.strip():
            continue
        if is_noisy_line(line):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def is_noisy_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    line_len = len(stripped)
    if line_len <= 6:
        return False
    alnum = sum(1 for ch in stripped if ch.isalnum())
    alnum_ratio = alnum / max(line_len, 1)
    pipe_count = stripped.count("|")
    symbol_run = re.search(r"([|=_-])\1{12,}", stripped)
    pipe_run = re.search(r"(?:\|\s*){12,}", stripped)

    if pipe_run and alnum_ratio < 0.15:
        return True
    if symbol_run and alnum_ratio < 0.2:
        return True
    if pipe_count >= 10 and line_len >= 40 and alnum_ratio < 0.15:
        return True
    if stripped.startswith("|") and stripped.endswith("|") and alnum_ratio < 0.1:
        return True
    return False


def is_noisy_chunk(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    if len(stripped) < 20:
        return False
    alnum = sum(1 for ch in stripped if ch.isalnum())
    alnum_ratio = alnum / max(len(stripped), 1)
    symbol_ratio = sum(
        1 for ch in stripped if not ch.isalnum() and not ch.isspace()
    ) / max(len(stripped), 1)

    if symbol_ratio > 0.6 and alnum_ratio < 0.1:
        return True
    if re.search(r"(?:\|\s*){40,}", stripped) and alnum_ratio < 0.15:
        return True
    if re.search(r"([|=_-])\1{30,}", stripped) and alnum_ratio < 0.2:
        return True
    return False
