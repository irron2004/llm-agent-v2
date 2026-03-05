from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast


PathLike = str | Path
JsonValue = str | int | float | bool | None | dict[str, "JsonValue"] | list["JsonValue"]


def _to_path(path: PathLike) -> Path:
    return path if isinstance(path, Path) else Path(path)


def write_json(
    path: PathLike, obj: JsonValue, *, indent: int = 2, sort_keys: bool = True
) -> None:
    """Write JSON with UTF-8 encoding and a trailing newline."""
    p = _to_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
        _ = f.write("\n")


def write_jsonl(path: PathLike, rows: Iterable[JsonValue]) -> None:
    """Write JSON Lines (JSONL) with UTF-8 encoding; one record per line."""
    p = _to_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            _ = f.write(json.dumps(row, ensure_ascii=False))
            _ = f.write("\n")


def read_jsonl(path: PathLike) -> Iterator[JsonValue]:
    """Read JSONL with UTF-8 encoding. Blank lines are ignored."""
    p = _to_path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield cast(JsonValue, json.loads(line))
