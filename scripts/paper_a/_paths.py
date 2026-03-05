from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the project root directory as a Path."""
    return Path(__file__).resolve().parents[2]
