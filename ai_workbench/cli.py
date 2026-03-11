from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from typing import Sequence, cast

from ai_workbench.config import get_workbench_paths
from ai_workbench.core.tmux_adapter import TmuxAdapter
from ai_workbench.core.workspace_service import WorkspaceService
from ai_workbench.storage.workspace_repo import WorkspaceRepository
from ai_workbench.tui.app import WorkbenchApp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-workbench",
        description="AI Workspace TUI for tmux-managed agent panes",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print tmux commands without executing"
    )
    parser.add_argument(
        "--print-tmux", action="store_true", help="Print every tmux command"
    )
    parser.add_argument("--tmux-socket", help="Use custom tmux server socket with -L")
    parser.add_argument("--tmux-config", help="Use custom tmux config file with -f")

    subparsers = parser.add_subparsers(dest="command")

    doctor = subparsers.add_parser("doctor", help="Run environment checks")
    doctor.add_argument("--json", action="store_true", help="Print JSON output")

    workspaces = subparsers.add_parser("workspaces", help="Workspace management")
    workspace_subcommands = workspaces.add_subparsers(
        dest="workspace_command", required=True
    )

    list_cmd = workspace_subcommands.add_parser(
        "list", help="List persisted workspaces"
    )
    list_cmd.add_argument("--json", action="store_true", help="Print JSON output")

    create_cmd = workspace_subcommands.add_parser("create", help="Create a workspace")
    create_cmd.add_argument("--name", required=True, help="Workspace display name")
    create_cmd.add_argument(
        "--template",
        default="triple-agent",
        choices=["triple-agent", "research", "debug", "writing", "terminal"],
    )
    create_cmd.add_argument(
        "--project-path", default="", help="Project path for pane working directory"
    )
    create_cmd.add_argument(
        "--command",
        action="append",
        dest="role_commands",
        default=[],
        help="Override role command as role=command, can repeat",
    )
    create_cmd.add_argument("--json", action="store_true", help="Print JSON output")

    statusbar = subparsers.add_parser(
        "statusbar", help="Print workspace tab bar for tmux status-left"
    )
    statusbar.add_argument(
        "--session", default="", help="Current tmux session name"
    )

    cycle = subparsers.add_parser(
        "cycle", help="Switch tmux client to next/prev workspace session"
    )
    cycle.add_argument(
        "--direction", choices=["next", "prev"], default="next"
    )
    cycle.add_argument(
        "--index", type=int, default=0, help="Switch to workspace by 1-based index"
    )

    subparsers.add_parser(
        "quick-create", help="Create a new terminal workspace and switch to it"
    )

    add_pane_cmd = subparsers.add_parser(
        "add-pane", help="Add a terminal pane to the current workspace session"
    )
    add_pane_cmd.add_argument(
        "--session", default="", help="Current tmux session name"
    )

    return parser


def build_workspace_service(
    args: argparse.Namespace,
) -> tuple[WorkspaceService, WorkspaceRepository]:
    tmux = TmuxAdapter(
        dry_run=args.dry_run,
        print_tmux=args.print_tmux,
        socket_name=args.tmux_socket,
        tmux_config=args.tmux_config,
    )
    repo = WorkspaceRepository()
    service = WorkspaceService(repo=repo, tmux=tmux)
    return service, repo


def parse_command_overrides(entries: Sequence[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --command value: {entry}")
        role, command = entry.split("=", 1)
        role = role.strip()
        command = command.strip()
        if not role or not command:
            raise ValueError(f"Invalid --command value: {entry}")
        overrides[role] = command
    return overrides


def cmd_doctor(args: argparse.Namespace) -> int:
    paths = get_workbench_paths()
    payload = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "tmux_installed": shutil.which("tmux") is not None,
        "inside_tmux": bool(os.getenv("TMUX")),
        "workbench_home": str(paths.home),
        "dry_run": bool(args.dry_run),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


def cmd_workspaces_list(args: argparse.Namespace) -> int:
    _, repo = build_workspace_service(args)
    summary = repo.to_json_summary()
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        workspaces = cast(list[dict[str, object]], summary["workspaces"])
        for workspace in workspaces:
            print(
                f"{workspace['id']}\t{workspace['name']}\t"
                f"{workspace['status']}\t{workspace['session_name']}"
            )
    return 0


def cmd_workspaces_create(args: argparse.Namespace) -> int:
    service, _ = build_workspace_service(args)
    role_commands = cast(Sequence[str], getattr(args, "role_commands", []))
    overrides = parse_command_overrides(list(role_commands))
    workspace = service.create_workspace(
        name=args.name,
        template=args.template,
        project_path=args.project_path or None,
        command_overrides=overrides,
    )
    if args.json:
        print(
            json.dumps({"workspace": workspace.to_dict()}, ensure_ascii=False, indent=2)
        )
    else:
        print(
            f"created {workspace.name} ({workspace.id}) status={workspace.status.value}"
        )
    return 0


def cmd_statusbar(args: argparse.Namespace) -> int:
    repo = WorkspaceRepository()
    workspaces = repo.list_workspaces()
    current_session = args.session.strip()

    if not workspaces:
        print("AIWB: no workspaces")
        return 0

    labels: list[str] = []
    for workspace in workspaces:
        if workspace.status.value == "archived":
            continue
        name = workspace.name
        if workspace.session_name == current_session:
            labels.append(f"#[fg=colour46,bold][{name}]#[default]")
        else:
            labels.append(name)

    print(" | ".join(labels) if labels else "AIWB")
    return 0


def cmd_cycle(args: argparse.Namespace) -> int:
    tmux = TmuxAdapter()
    repo = WorkspaceRepository()
    workspaces = [
        ws for ws in repo.list_workspaces() if ws.status.value != "archived"
    ]
    if not workspaces:
        return 0

    current = tmux.get_current_session()
    session_names = [ws.session_name for ws in workspaces]

    # --index N: switch to Nth workspace (1-based)
    if args.index > 0:
        target_idx = min(args.index - 1, len(session_names) - 1)
    else:
        try:
            idx = session_names.index(current) if current else -1
        except ValueError:
            idx = -1
        if args.direction == "next":
            target_idx = (idx + 1) % len(session_names)
        else:
            target_idx = (idx - 1) % len(session_names)

    target = session_names[target_idx]
    if tmux.has_session(target):
        tmux._run(["switch-client", "-t", target], check=False)
    return 0


def cmd_quick_create(args: argparse.Namespace) -> int:
    tmux = TmuxAdapter()
    repo = WorkspaceRepository()
    service = WorkspaceService(repo=repo, tmux=tmux)

    existing = [ws.name for ws in repo.list_workspaces()]
    base = "workspace"
    name = base
    n = 1
    while name in existing:
        n += 1
        name = f"{base}-{n}"

    cwd = os.getcwd()
    workspace = service.create_workspace(
        name=name,
        template="terminal",
        project_path=cwd,
    )

    if workspace.status.value != "error" and tmux.has_session(workspace.session_name):
        tmux._run(["switch-client", "-t", workspace.session_name], check=False)
    return 0


def cmd_add_pane(args: argparse.Namespace) -> int:
    tmux = TmuxAdapter()
    repo = WorkspaceRepository()
    service = WorkspaceService(repo=repo, tmux=tmux)

    session = args.session.strip() or tmux.get_current_session() or ""
    if not session:
        return 1

    workspace = None
    for ws in repo.list_workspaces():
        if ws.session_name == session:
            workspace = ws
            break
    if workspace is None:
        return 1

    shell = os.getenv("SHELL") or "bash"
    existing_roles = {p.role for p in workspace.pane_profiles}
    role = "terminal"
    n = 2
    while role in existing_roles:
        role = f"terminal-{n}"
        n += 1

    service.add_pane_profile(
        workspace.id, role=role, command=shell, cwd=workspace.project_path
    )
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    app = WorkbenchApp(
        dry_run=args.dry_run,
        print_tmux=args.print_tmux,
        socket_name=args.tmux_socket,
        tmux_config=args.tmux_config,
    )
    app.run()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "doctor":
            return cmd_doctor(args)
        if args.command == "statusbar":
            return cmd_statusbar(args)
        if args.command == "cycle":
            return cmd_cycle(args)
        if args.command == "quick-create":
            return cmd_quick_create(args)
        if args.command == "add-pane":
            return cmd_add_pane(args)
        if args.command == "workspaces":
            if args.workspace_command == "list":
                return cmd_workspaces_list(args)
            if args.workspace_command == "create":
                return cmd_workspaces_create(args)
            parser.error(f"Unknown workspace command: {args.workspace_command}")

        return cmd_dashboard(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
