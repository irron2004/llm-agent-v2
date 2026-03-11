from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
import uuid


SCHEMA_VERSION = 2


class WorkspaceStatus(str, Enum):
    DRAFT = "draft"
    STARTING = "starting"
    RUNNING = "running"
    DETACHED = "detached"
    DEGRADED = "degraded"
    MISSING = "missing"
    RESTORING = "restoring"
    ARCHIVED = "archived"
    ERROR = "error"


class WorkspaceRecoveryState(str, Enum):
    LIVE = "live"
    RECOVERABLE = "recoverable"
    REBUILDABLE = "rebuildable"
    BROKEN = "broken"


class PaneLifecycle(str, Enum):
    CONFIGURED = "configured"
    LAUNCHING = "launching"
    LIVE = "live"
    EXITED = "exited"
    CRASHED = "crashed"
    RESTARTING = "restarting"
    FAILED = "failed"


class PaneActivity(str, Enum):
    IDLE = "idle"
    STREAMING = "streaming"
    BUSY = "busy"
    WAITING = "waiting"
    DONE = "done"
    WARN = "warn"
    ERROR = "error"


WORKSPACE_TRANSITIONS: dict[WorkspaceStatus, set[WorkspaceStatus]] = {
    WorkspaceStatus.DRAFT: {WorkspaceStatus.STARTING},
    WorkspaceStatus.STARTING: {WorkspaceStatus.RUNNING, WorkspaceStatus.ERROR},
    WorkspaceStatus.RUNNING: {
        WorkspaceStatus.DETACHED,
        WorkspaceStatus.DEGRADED,
        WorkspaceStatus.MISSING,
        WorkspaceStatus.ARCHIVED,
    },
    WorkspaceStatus.DETACHED: {
        WorkspaceStatus.RUNNING,
        WorkspaceStatus.MISSING,
        WorkspaceStatus.ARCHIVED,
    },
    WorkspaceStatus.DEGRADED: {
        WorkspaceStatus.RESTORING,
        WorkspaceStatus.MISSING,
    },
    WorkspaceStatus.MISSING: {
        WorkspaceStatus.RESTORING,
    },
    WorkspaceStatus.RESTORING: {
        WorkspaceStatus.RUNNING,
        WorkspaceStatus.DEGRADED,
        WorkspaceStatus.ERROR,
    },
    WorkspaceStatus.ERROR: {
        WorkspaceStatus.ARCHIVED,
    },
    WorkspaceStatus.ARCHIVED: set(),
}


PANE_TRANSITIONS: dict[PaneLifecycle, set[PaneLifecycle]] = {
    PaneLifecycle.CONFIGURED: {PaneLifecycle.LAUNCHING},
    PaneLifecycle.LAUNCHING: {PaneLifecycle.LIVE, PaneLifecycle.FAILED},
    PaneLifecycle.LIVE: {PaneLifecycle.EXITED, PaneLifecycle.CRASHED},
    PaneLifecycle.EXITED: {PaneLifecycle.RESTARTING},
    PaneLifecycle.CRASHED: {PaneLifecycle.RESTARTING},
    PaneLifecycle.RESTARTING: {PaneLifecycle.LIVE, PaneLifecycle.FAILED},
    PaneLifecycle.FAILED: set(),
}


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def validate_workspace_name(name: str) -> str:
    value = name.strip()
    if not value:
        raise ValueError("Workspace name must not be empty")
    if "\n" in value:
        raise ValueError("Workspace name must be a single line")
    if len(value) > 120:
        raise ValueError("Workspace name must be 120 characters or less")
    return value


def make_session_name(workspace_id: str) -> str:
    short = workspace_id.replace("-", "")[:8]
    session_name = f"aiwb-{short}"
    if session_name.startswith("="):
        raise ValueError("Session name must not start with '='")
    return session_name


@dataclass(slots=True)
class ToolResumeRef:
    kind: str
    session_id: str
    updated_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "session_id": self.session_id,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResumeRef":
        kind = str(data.get("kind", "")).strip()
        session_id = str(data.get("session_id", "")).strip()
        updated_at = str(data.get("updated_at", now_iso()))
        if not kind:
            raise ValueError("ToolResumeRef.kind is required")
        if not session_id:
            raise ValueError("ToolResumeRef.session_id is required")
        return cls(kind=kind, session_id=session_id, updated_at=updated_at)


@dataclass(slots=True)
class TmuxRuntimeHandle:
    socket_path: str | None = None
    session_name: str | None = None
    session_id_last_seen: str | None = None
    last_seen_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "socket_path": self.socket_path,
            "session_name": self.session_name,
            "session_id_last_seen": self.session_id_last_seen,
            "last_seen_at": self.last_seen_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TmuxRuntimeHandle":
        return cls(
            socket_path=data.get("socket_path"),
            session_name=data.get("session_name"),
            session_id_last_seen=data.get("session_id_last_seen"),
            last_seen_at=data.get("last_seen_at"),
        )


@dataclass(slots=True)
class PaneProfile:
    role: str
    command: str
    cwd: str | None = None
    pane_id_last_seen: str | None = None
    resume: ToolResumeRef | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "command": self.command,
            "cwd": self.cwd,
            "pane_id_last_seen": self.pane_id_last_seen,
            "resume": self.resume.to_dict() if self.resume is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaneProfile":
        role = str(data.get("role", "")).strip()
        command = str(data.get("command", "")).strip()
        cwd = data.get("cwd")
        pane_id_last_seen = data.get("pane_id_last_seen")
        resume_data = data.get("resume")
        if not role:
            raise ValueError("PaneProfile.role is required")
        if not command:
            raise ValueError("PaneProfile.command is required")
        resume = (
            ToolResumeRef.from_dict(resume_data)
            if isinstance(resume_data, dict)
            else None
        )
        return cls(
            role=role,
            command=command,
            cwd=cwd,
            pane_id_last_seen=pane_id_last_seen,
            resume=resume,
        )


@dataclass(slots=True)
class PaneRuntime:
    role: str
    pane_id: str | None = None
    lifecycle: PaneLifecycle = PaneLifecycle.CONFIGURED
    activity: PaneActivity = PaneActivity.IDLE
    last_output_at: str | None = None
    tail_preview: str = ""
    pane_dead: bool = False
    exit_status: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "pane_id": self.pane_id,
            "lifecycle": self.lifecycle.value,
            "activity": self.activity.value,
            "last_output_at": self.last_output_at,
            "tail_preview": self.tail_preview,
            "pane_dead": self.pane_dead,
            "exit_status": self.exit_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaneRuntime":
        lifecycle = PaneLifecycle(data.get("lifecycle", PaneLifecycle.CONFIGURED.value))
        activity = PaneActivity(data.get("activity", PaneActivity.IDLE.value))
        return cls(
            role=str(data.get("role", "")).strip(),
            pane_id=data.get("pane_id"),
            lifecycle=lifecycle,
            activity=activity,
            last_output_at=data.get("last_output_at"),
            tail_preview=str(data.get("tail_preview", "")),
            pane_dead=bool(data.get("pane_dead", False)),
            exit_status=data.get("exit_status"),
        )


@dataclass(slots=True)
class Workspace:
    id: str
    name: str
    session_name: str
    template: str
    pane_profiles: list[PaneProfile]
    status: WorkspaceStatus = WorkspaceStatus.DRAFT
    recovery_state: WorkspaceRecoveryState = WorkspaceRecoveryState.REBUILDABLE
    project_path: str | None = None
    tmux_runtime: TmuxRuntimeHandle | None = None
    tags: list[str] = field(default_factory=list)
    pinned: bool = False
    active_role: str | None = None
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    last_error: str | None = None

    def transition(self, next_status: WorkspaceStatus) -> None:
        if self.status == next_status:
            return
        allowed = WORKSPACE_TRANSITIONS.get(self.status, set())
        if next_status not in allowed:
            raise ValueError(
                f"Invalid workspace transition: {self.status.value} -> {next_status.value}"
            )
        self.status = next_status
        self.updated_at = now_iso()

    def touch(self) -> None:
        self.updated_at = now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "session_name": self.session_name,
            "template": self.template,
            "pane_profiles": [pane.to_dict() for pane in self.pane_profiles],
            "status": self.status.value,
            "recovery_state": self.recovery_state.value,
            "project_path": self.project_path,
            "tmux_runtime": self.tmux_runtime.to_dict()
            if self.tmux_runtime is not None
            else None,
            "tags": self.tags,
            "pinned": self.pinned,
            "active_role": self.active_role,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Workspace":
        pane_profiles = [
            PaneProfile.from_dict(item) for item in data.get("pane_profiles", [])
        ]
        tmux_runtime_data = data.get("tmux_runtime")
        tmux_runtime = (
            TmuxRuntimeHandle.from_dict(tmux_runtime_data)
            if isinstance(tmux_runtime_data, dict)
            else None
        )
        return cls(
            id=str(data["id"]),
            name=validate_workspace_name(str(data["name"])),
            session_name=str(data["session_name"]),
            template=str(data.get("template", "custom")),
            pane_profiles=pane_profiles,
            status=WorkspaceStatus(data.get("status", WorkspaceStatus.DRAFT.value)),
            recovery_state=WorkspaceRecoveryState(
                data.get("recovery_state", WorkspaceRecoveryState.REBUILDABLE.value)
            ),
            project_path=data.get("project_path"),
            tmux_runtime=tmux_runtime,
            tags=[str(tag) for tag in data.get("tags", [])],
            pinned=bool(data.get("pinned", False)),
            active_role=data.get("active_role"),
            created_at=str(data.get("created_at", now_iso())),
            updated_at=str(data.get("updated_at", now_iso())),
            last_error=data.get("last_error"),
        )


@dataclass(slots=True)
class WorkspaceStore:
    schema_version: int = SCHEMA_VERSION
    active_workspace_id: str | None = None
    workspaces: list[Workspace] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "active_workspace_id": self.active_workspace_id,
            "workspaces": [workspace.to_dict() for workspace in self.workspaces],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkspaceStore":
        schema_version = int(data.get("schema_version", SCHEMA_VERSION))
        workspaces = [Workspace.from_dict(item) for item in data.get("workspaces", [])]
        active_workspace_id = data.get("active_workspace_id")
        return cls(
            schema_version=schema_version,
            active_workspace_id=active_workspace_id,
            workspaces=workspaces,
        )


def new_workspace_id() -> str:
    return str(uuid.uuid4())
