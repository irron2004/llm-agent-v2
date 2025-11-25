"""Minimal I/O utilities used by normalization utils."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(obj: Any, path: str | Path) -> None:
    """Save Python object as JSON with UTF-8 and pretty indent."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


__all__ = ["save_json"]
