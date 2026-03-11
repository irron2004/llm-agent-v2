from __future__ import annotations

from pathlib import Path
import os
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


@pytest.fixture
def workbench_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "ai-workbench-home"
    monkeypatch.setenv("AI_WORKBENCH_HOME", str(home))
    return home


@pytest.fixture
def fake_tmux_bin_dir() -> Path:
    return Path(__file__).resolve().parent / "fake_bin"


@pytest.fixture
def fake_tmux_path(fake_tmux_bin_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    tmux_path = fake_tmux_bin_dir / "tmux"
    os.chmod(tmux_path, 0o755)
    current_path = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", f"{fake_tmux_bin_dir}:{current_path}")
    return tmux_path
