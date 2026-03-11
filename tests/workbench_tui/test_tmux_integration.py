from __future__ import annotations

import os
import shutil

import pytest

from ai_workbench.core.tmux_adapter import TmuxAdapter


@pytest.mark.skipif(
    os.getenv("AIWB_RUN_TMUX_IT") != "1", reason="Set AIWB_RUN_TMUX_IT=1 to enable"
)
def test_tmux_integration_smoke(workbench_home) -> None:
    if shutil.which("tmux") is None:
        pytest.skip("tmux is not installed")

    socket_name = "aiwb-test"
    adapter = TmuxAdapter(socket_name=socket_name, tmux_config="/dev/null")
    session_name = "aiwb-it-demo"
    workspace_id = "ws-it"

    if adapter.has_session(session_name):
        adapter.kill_session(session_name)

    pane_id = adapter.new_session(session_name, "echo integration")
    adapter.set_session_workspace_id(session_name, workspace_id)
    adapter.set_pane_role(pane_id, "claude")

    assert adapter.has_session(session_name)
    assert adapter.find_session_by_workspace_id(workspace_id) == session_name
    panes = adapter.list_panes(session_name)
    assert panes

    assert adapter.kill_managed_session(session_name, workspace_id)
