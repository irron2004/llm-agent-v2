from __future__ import annotations

from ai_workbench.core.tmux_adapter import TmuxAdapter


def test_tmux_adapter_lists_panes_from_fake_tmux(fake_tmux_path) -> None:
    adapter = TmuxAdapter()
    panes = adapter.list_panes("aiwb-demo")
    assert len(panes) == 3
    assert panes[0].role == "claude"
    assert panes[1].role == "codex"


def test_tmux_adapter_resolves_workspace_id_after_rename(fake_tmux_path) -> None:
    adapter = TmuxAdapter()
    discovered = adapter.find_session_by_workspace_id("ws-renamed")
    assert discovered == "renamed-session"


def test_tmux_adapter_lists_session_metadata(fake_tmux_path) -> None:
    adapter = TmuxAdapter()
    sessions = adapter.list_sessions()
    target = next(item for item in sessions if item.name == "renamed-session")
    assert target.session_id == "$2"
    assert target.socket_path == "/tmp/tmux-1002/default"


def test_tmux_adapter_dry_run_prints_commands(capsys) -> None:
    adapter = TmuxAdapter(dry_run=True, print_tmux=True)
    adapter.new_session("aiwb-demo", "claude")
    captured = capsys.readouterr()
    assert "tmux" in captured.out
    assert "new-session" in captured.out


def test_configure_workspace_bar_uses_alt_bindings(capsys) -> None:
    adapter = TmuxAdapter(dry_run=True, print_tmux=True)
    adapter.configure_workspace_bar("aiwb-demo")

    captured = capsys.readouterr()
    output = captured.out

    for index in range(1, 10):
        assert f"bind-key -T root M-{index} run-shell" in output
        assert f"cycle --index {index}" in output
        assert f"bind-key -T root C-{index} run-shell" not in output

    assert "bind-key -T root M-0 run-shell" in output
    assert "@ai_workbench_home" in output
    assert "detach-client" in output
