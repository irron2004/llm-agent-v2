from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Callable


@dataclass(slots=True)
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str


@dataclass(slots=True)
class TmuxSession:
    name: str
    workspace_id: str | None
    attached_clients: int
    session_id: str | None = None
    socket_path: str | None = None


@dataclass(slots=True)
class TmuxPane:
    pane_id: str
    index: int
    active: bool
    dead: bool
    current_command: str
    title: str
    role: str | None
    pid: int | None
    dead_status: int | None
    start_command: str


class TmuxCommandError(RuntimeError):
    def __init__(
        self, message: str, *, kind: str, result: CommandResult | None = None
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.result = result


class TmuxAdapter:
    def __init__(
        self,
        *,
        dry_run: bool = False,
        print_tmux: bool = False,
        socket_name: str | None = None,
        tmux_config: str | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.dry_run = dry_run
        self.print_tmux = print_tmux
        self.socket_name = socket_name
        self.tmux_config = tmux_config
        self._runner = runner or subprocess.run

    @property
    def prefix(self) -> list[str]:
        argv = ["tmux"]
        if self.socket_name:
            argv.extend(["-L", self.socket_name])
        if self.tmux_config:
            argv.extend(["-f", self.tmux_config])
        return argv

    def get_current_session(self) -> str | None:
        result = self._run(["display-message", "-p", "#{session_name}"], check=False)
        if result.returncode != 0:
            return None
        name = result.stdout.strip()
        return name or None

    def has_session(self, session_name: str) -> bool:
        self._validate_session_name(session_name)
        result = self._run(["has-session", "-t", session_name], check=False)
        return result.returncode == 0

    def list_sessions(self) -> list[TmuxSession]:
        fmt = (
            "#{session_name}\t#{@ai_workspace_id}\t#{session_attached}\t"
            "#{session_id}\t#{socket_path}"
        )
        result = self._run(["list-sessions", "-F", fmt], check=False)
        if result.returncode != 0:
            if self._classify_error(result) in {"server_missing", "not_installed"}:
                return []
            self._raise("Failed to list tmux sessions", result)

        sessions: list[TmuxSession] = []
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            name, workspace_id, attached, session_id, socket_path = (
                line.split("\t") + ["", "", "0", "", ""]
            )[:5]
            sessions.append(
                TmuxSession(
                    name=name,
                    workspace_id=workspace_id or None,
                    attached_clients=int(attached or "0"),
                    session_id=session_id or None,
                    socket_path=socket_path or None,
                )
            )
        return sessions

    def find_session_by_workspace_id(self, workspace_id: str) -> str | None:
        for session in self.list_sessions():
            if session.workspace_id == workspace_id:
                return session.name
        return None

    def new_session(
        self, session_name: str, command: str, *, cwd: str | None = None
    ) -> str:
        self._validate_session_name(session_name)
        args = ["new-session", "-d", "-s", session_name, "-P", "-F", "#{pane_id}"]
        if cwd:
            args.extend(["-c", cwd])
        args.append(self._bash_command(command))
        result = self._run(args, check=True)
        pane_id = result.stdout.strip() or "%0"
        return pane_id

    def split_window(
        self,
        session_name: str,
        command: str,
        *,
        cwd: str | None = None,
        horizontal: bool = True,
    ) -> str:
        self._validate_session_name(session_name)
        split_flag = "-h" if horizontal else "-v"
        args = [
            "split-window",
            split_flag,
            "-t",
            session_name,
            "-P",
            "-F",
            "#{pane_id}",
        ]
        if cwd:
            args.extend(["-c", cwd])
        args.append(self._bash_command(command))
        result = self._run(args, check=True)
        pane_id = result.stdout.strip() or "%0"
        return pane_id

    def select_layout(self, session_name: str, layout: str = "even-horizontal") -> None:
        self._validate_session_name(session_name)
        self._run(["select-layout", "-t", session_name, layout], check=True)

    def set_session_workspace_id(self, session_name: str, workspace_id: str) -> None:
        self._validate_session_name(session_name)
        self._run(
            ["set-option", "-t", session_name, "@ai_workspace_id", workspace_id],
            check=True,
        )

    def set_pane_role(self, pane_id: str, role: str) -> None:
        self._run(["set-option", "-p", "-t", pane_id, "@ai_role", role], check=True)
        self._run(["select-pane", "-t", pane_id, "-T", f"AIWB {role}"], check=True)

    def list_panes(self, session_name: str) -> list[TmuxPane]:
        self._validate_session_name(session_name)
        fmt = (
            "#{pane_id}\t#{pane_index}\t#{pane_active}\t#{pane_dead}\t"
            "#{pane_current_command}\t#{pane_title}\t#{@ai_role}\t"
            "#{pane_pid}\t#{pane_dead_status}\t#{pane_start_command}"
        )
        result = self._run(["list-panes", "-t", session_name, "-F", fmt], check=False)
        if result.returncode != 0:
            self._raise("Failed to list tmux panes", result)

        panes: list[TmuxPane] = []
        for raw_line in result.stdout.splitlines():
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = (line.split("\t") + ["", "", "", "", "", "", "", "", "", ""])[:10]
            panes.append(
                TmuxPane(
                    pane_id=parts[0],
                    index=int(parts[1] or "0"),
                    active=parts[2] == "1",
                    dead=parts[3] == "1",
                    current_command=parts[4],
                    title=parts[5],
                    role=parts[6] or None,
                    pid=int(parts[7]) if parts[7].isdigit() else None,
                    dead_status=int(parts[8]) if parts[8].isdigit() else None,
                    start_command=parts[9],
                )
            )
        return panes

    def find_pane_by_role(self, session_name: str, role: str) -> TmuxPane | None:
        panes = self.list_panes(session_name)
        for pane in panes:
            if pane.role == role:
                return pane
        return None

    def capture_pane(self, pane_id: str, *, lines: int = 200) -> str:
        start = f"-{max(lines, 1)}"
        result = self._run(
            ["capture-pane", "-p", "-S", start, "-J", "-t", pane_id], check=False
        )
        if result.returncode != 0:
            return ""
        return result.stdout

    def respawn_pane(
        self, pane_id: str, command: str, *, cwd: str | None = None
    ) -> None:
        args = ["respawn-pane", "-k", "-t", pane_id]
        if cwd:
            args.extend(["-c", cwd])
        args.append(self._bash_command(command))
        self._run(args, check=True)

    def apply_portrait_layout(
        self,
        session_name: str,
        *,
        active_pane_id: str | None,
        main_height_percent: int = 70,
    ) -> None:
        self._validate_session_name(session_name)
        self._run(["select-layout", "-t", session_name, "main-horizontal"], check=False)
        if active_pane_id:
            self._run(
                ["swap-pane", "-s", active_pane_id, "-t", f"{session_name}.0"],
                check=False,
            )
        height = max(30, min(main_height_percent, 90))
        self._run(
            [
                "set-option",
                "-w",
                "-t",
                session_name,
                "main-pane-height",
                f"{height}%",
            ],
            check=False,
        )

    def attach_or_switch(
        self,
        session_name: str,
        *,
        inside_tmux: bool,
        exec_attach: bool = True,
    ) -> str | None:
        self._validate_session_name(session_name)
        home_session: str | None = None
        if inside_tmux:
            home_session = self.get_current_session()
            if home_session:
                self._run(
                    [
                        "set-option",
                        "-t",
                        session_name,
                        "@ai_workbench_home",
                        home_session,
                    ],
                    check=False,
                )
                self._run(
                    [
                        "bind-key",
                        "-T",
                        "prefix",
                        "B",
                        "switch-client",
                        "-t",
                        home_session,
                    ],
                    check=False,
                )
            self._run(["switch-client", "-t", session_name], check=True)
            return home_session

        attach_args = self.prefix + ["attach-session", "-t", session_name]
        if self.print_tmux or self.dry_run:
            print(" ".join(shlex.quote(part) for part in attach_args))
        if self.dry_run:
            return
        if exec_attach:
            os.execvp("tmux", attach_args)
        completed = self._runner(
            attach_args, check=False, capture_output=True, text=True
        )
        if completed.returncode != 0:
            result = CommandResult(
                attach_args, completed.returncode, completed.stdout, completed.stderr
            )
            self._raise("Failed to attach tmux session", result)

    def configure_workspace_bar(self, session_name: str) -> None:
        self._validate_session_name(session_name)
        python = sys.executable
        statusbar_cmd = (
            f"#({python} -m ai_workbench statusbar --session '#{{session_name}}')"
        )
        cycle_next = f"{python} -m ai_workbench cycle --direction next"
        cycle_prev = f"{python} -m ai_workbench cycle --direction prev"
        quick_create = f"{python} -m ai_workbench quick-create"
        add_pane = f"{python} -m ai_workbench add-pane --session '#{{session_name}}'"
        go_home = (
            "home='#{@ai_workbench_home}'; "
            '[ -n "$home" ] && tmux switch-client -t "=$home" || tmux detach-client'
        )

        # Status bar: top, showing workspace tabs
        for opt, val in [
            ("status", "on"),
            ("status-position", "top"),
            ("status-left-length", "120"),
            ("status-left", statusbar_cmd),
            ("status-right", ""),
            ("status-style", "bg=colour235,fg=colour248"),
            ("status-interval", "3"),
        ]:
            self._run(["set-option", "-t", session_name, opt, val], check=False)

        # Keybindings (root table = no prefix needed)
        bindings: list[tuple[str, str]] = [
            ("C-Tab", cycle_next),
            ("C-S-Tab", cycle_prev),
            ("C-t", quick_create),  # Ctrl+T: new workspace
            ("M-t", add_pane),  # Alt+T: add terminal pane
            ("M-0", go_home),
        ]
        for i in range(1, 10):
            bindings.append((f"M-{i}", f"{python} -m ai_workbench cycle --index {i}"))

        for key, cmd in bindings:
            self._run(
                ["bind-key", "-T", "root", key, "run-shell", cmd],
                check=False,
            )

    def kill_session(self, session_name: str) -> None:
        self._validate_session_name(session_name)
        self._run(["kill-session", "-t", session_name], check=True)

    def kill_managed_session(self, session_name: str, workspace_id: str) -> bool:
        for session in self.list_sessions():
            if session.name == session_name and session.workspace_id == workspace_id:
                self.kill_session(session_name)
                return True
        return False

    def _bash_command(self, command: str) -> str:
        return f"bash -lc {shlex.quote(command)}"

    def _validate_session_name(self, session_name: str) -> None:
        if session_name.startswith("="):
            raise ValueError("Session names must not begin with '='")

    def _run(self, args: list[str], *, check: bool) -> CommandResult:
        argv = self.prefix + args
        if self.print_tmux or self.dry_run:
            print(" ".join(shlex.quote(part) for part in argv))
        if self.dry_run:
            return CommandResult(argv=argv, returncode=0, stdout="", stderr="")

        try:
            completed = self._runner(argv, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            result = CommandResult(
                argv=argv, returncode=127, stdout="", stderr=str(exc)
            )
            if check:
                self._raise("tmux command failed", result)
            return result

        result = CommandResult(
            argv=argv,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        if check and result.returncode != 0:
            self._raise("tmux command failed", result)
        return result

    def _raise(self, message: str, result: CommandResult) -> None:
        kind = self._classify_error(result)
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or "no output"
        raise TmuxCommandError(f"{message}: {detail}", kind=kind, result=result)

    def _classify_error(self, result: CommandResult) -> str:
        text = (result.stderr or result.stdout).lower()
        if "no server running" in text:
            return "server_missing"
        if "no such file or directory" in text or "not found" in text:
            return "not_installed"
        if "can't find pane" in text:
            return "pane_missing"
        if "can't find session" in text or "can't find window" in text:
            return "session_missing"
        if "permission denied" in text:
            return "permission"
        return "unknown"
