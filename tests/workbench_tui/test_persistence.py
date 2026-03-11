from __future__ import annotations

import json
from typing import cast

from ai_workbench.models import PaneProfile, Workspace
from ai_workbench.storage.workspace_repo import WorkspaceRepository


def test_list_empty_on_fresh_install(workbench_home) -> None:
    repo = WorkspaceRepository()
    summary = repo.to_json_summary()
    assert summary["workspaces"] == []


def test_backup_recovery_when_workspaces_json_corrupt(workbench_home) -> None:
    repo = WorkspaceRepository()
    workspace = Workspace(
        id="ws-1",
        name="Recoverable",
        session_name="aiwb-abcdef12",
        template="triple-agent",
        pane_profiles=[
            PaneProfile(role="claude", command="claude"),
            PaneProfile(role="codex", command="codex"),
        ],
    )
    repo.upsert_workspace(workspace)

    workspace.pinned = True
    repo.upsert_workspace(workspace)

    repo.paths.workspaces.write_text("{not-valid-json", encoding="utf-8")
    recovered = repo.to_json_summary()
    workspaces = cast(list[dict[str, object]], recovered["workspaces"])

    assert len(workspaces) == 1
    assert workspaces[0]["name"] == "Recoverable"
    assert recovered["warnings"]

    parsed = json.loads(repo.paths.workspaces.read_text(encoding="utf-8"))
    assert parsed["workspaces"][0]["name"] == "Recoverable"
