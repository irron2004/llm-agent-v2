# AI Workspace TUI MVP (tmux + Textual)

## TL;DR
> **Summary**: Implement a portrait-first Textual TUI dashboard that persists “workspaces” and orchestrates tmux sessions/panes per workspace with slim previews, notes, and restore flows.
> **Deliverables**: `ai_workbench/` package + `ai-workbench` CLI, Textual dashboard (tabs/search/preview stack/note drawer/restore modal), tmux adapter + restore logic, JSON persistence under `~/.ai-workbench/`, automated tests.
> **Effort**: Large
> **Parallel**: YES - 3 waves
> **Critical Path**: Persistence schema → tmux adapter → runtime snapshot poller → Textual dashboard → restore/attach UX → tests

## Context
### Original Request
- Build a workspace-centric TUI for Claude/Codex/OpenCode “agent panes” backed by tmux.
- Portrait / narrow-width / vertical monitor optimized: active pane large, inactive panes slim but visible.
- MVP: dashboard does NOT embed full terminals; `Enter` attaches to tmux session.
- Includes: focus resize mode, notes drawer, lifecycle/health states, restore flows, keybindings.

### Interview Summary
- No additional interview required; user provided a PRD in chat. This plan embeds all MVP-critical specs so execution has zero judgment calls.

### Research Findings
- Repo CLI pattern: `argparse` subcommands, error handling, examples in `backend/llm_infrastructure/elasticsearch/cli.py:200`.
- Repo test pattern: `pytest` + fixtures, example in `tests/api/conftest.py:126`.
- Textual testing: `App.run_test()` / `Pilot.press()` / `Pilot.click()` / `size=(w,h)` in official docs.
- tmux exact target match: tmux man page states `-t =name` forces exact match (prevents prefix ambiguity).
- tmux formats: `list-... -F` formats are the canonical way to enumerate panes/sessions (tmux wiki Formats).

### Oracle Review (gaps addressed)
- Add stable identity mapping: persist `workspace_id` UUID; tag tmux session `@ai_workspace_id` and panes `@ai_role`.
- Treat tmux `pane_id` as ephemeral; resolve panes by `@ai_role` on every snapshot.
- Centralize tmux calls in one adapter; classify errors; support multiple tmux servers via socket selector.
- Use two-tier polling: cheap list-panes frequently; capture-pane only for visible previews / activity gated.
- Persistence hardening: atomic writes, file lock, `schema_version`, backups.

### Metis Review (gaps addressed)
- Decide packaging/test location: ship as root-level tool (`ai_workbench/`) driven by root `pyproject.toml` + `uv.lock`.
- Define nested-tmux behavior: inside tmux, `Enter` uses `switch-client` (accepted) and shows a warning toast.
- Define restore semantics: MVP restore = deterministic recreate (session + panes) from stored profiles; do not attempt to preserve arbitrary user-modified layouts.
- Guardrails: never create sessions named with `=`; validate workspace name; always use exact `-t =session` targeting.
- Add `--dry-run` / `--print-tmux` mode for deterministic tests.

## Work Objectives
### Core Objective
- A usable portrait-first dashboard that makes it easy to (1) resume workspaces after restart, (2) attach quickly, (3) see background activity via slim previews, (4) restore missing/degraded sessions.

### Deliverables
- Root Python package `ai_workbench/` implementing:
  - Textual app: dashboard + create + restore modal
  - Core services: workspace service, layout engine, preview service, health/restore service
  - tmux adapter (subprocess runner + command composition)
  - Storage: JSON metadata + markdown notes
- CLI entrypoint:
  - `ai-workbench` launches the dashboard
  - `python -m ai_workbench doctor --json` and `python -m ai_workbench workspaces list --json` support automated verification (console-script `ai-workbench` is optional convenience)
- Tests:
  - unit tests (layout, persistence, tmux command composition)
  - Textual pilot tests for keybindings + responsive layout
  - optional tmux integration tests behind env flag

### Definition of Done (agent-verifiable)
- `uv run python -m ai_workbench` opens the dashboard without exceptions.
- Creating a workspace from UI creates a tmux session and panes for the selected template.
- Dashboard shows: tabs, search, preview stack, note drawer; focus-resize keeps inactive panes slim.
- `Enter` attaches: outside tmux -> `tmux attach`; inside tmux -> `tmux switch-client`.
- Missing/degraded detection works and restore action recreates the session/panes.
- Notes persist and reload after restart.
- `uv run pytest -q` passes (including new TUI tests).

### Must Have
- Portrait-first default layout; compact mode when width < 100.
- Stable workspace identity and safe tmux targeting.
- No destructive operations against non-managed tmux sessions.

### Must NOT Have
- No embedding interactive terminals inside the dashboard.
- No modifications to user tmux config (`~/.tmux.conf`) or global tmux options.
- No reliance on persisted `pane_id` across restarts.
- No auto-switching active pane on output; only user focus changes active pane.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: tests-after (pytest + pytest-asyncio for Textual).
- Evidence policy: save Textual screenshots / stdout logs under `.sisyphus/evidence/task-*-*.{txt,svg}`.
- tmux tests must use an isolated server socket: `tmux -L aiwb-test -f /dev/null ...`.

## Execution Strategy
### Parallel Execution Waves
Wave 1 (foundation): packaging + models + persistence + tmux adapter + layout engine
Wave 2 (UI): Textual screens/widgets + polling + keybindings + create/restore flows
Wave 3 (verification): unit tests + pilot tests + optional tmux integration tests + docs

### Dependency Matrix (high level)
- T1 blocks all others (project scaffolding + deps)
- T2 (models/persistence) blocks T5+ (services/UI)
- T3 (tmux adapter) blocks T6+ (health/preview/restore/attach)
- T4 (layout engine) blocks T8+ (preview stack rendering)

### Agent Dispatch Summary
- Wave 1: quick + unspecified-high
- Wave 2: visual-engineering + unspecified-high
- Wave 3: deep + unspecified-high

## TODOs
> Implementation + test = ONE task.

- [ ] 1. Add `ai_workbench` package scaffold + dependencies + CLI entrypoint

  **What to do**:
  - Create root package `ai_workbench/` with `__init__.py`, `__main__.py`, and `cli.py`.
  - Update root `pyproject.toml`:
    - Add dependencies: `textual>=0.70.0`, `pytest-asyncio>=0.23.0` (root currently has `pytest`).
    - Add `[project.scripts]` entry: `ai-workbench = "ai_workbench.cli:main"`.
    - Update lockfile with UV (executor runs these commands):
      - `uv add textual pytest-asyncio`
  - Add minimal CLI surface:
    - Default action: run dashboard.
    - `ai-workbench doctor --json` returns basic environment checks (python version, tmux presence, workbench home path).
    - `ai-workbench workspaces list --json` returns persisted workspaces (for non-UI verification).
  - Ensure no backend packaging changes are required (root-level tool only).

  **Must NOT do**:
  - Do not modify `backend/pyproject.toml` packaging lists.
  - Do not introduce non-deterministic side effects in `doctor`/`list` commands.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — repo-wide wiring + packaging
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: [2-16] | Blocked By: []

  **References**:
  - Repo deps file: `pyproject.toml:1` — root dependency list
  - CLI pattern: `backend/llm_infrastructure/elasticsearch/cli.py:200` — argparse subcommands + return codes
  - Test pattern: `tests/api/conftest.py:126` — pytest fixtures conventions
  - Textual docs (testing): https://textual.textualize.io/guide/testing/

  **Acceptance Criteria**:
  - [ ] `uv run python -m ai_workbench --help` exits 0
  - [ ] `uv run python -m ai_workbench doctor --json` exits 0 and prints valid JSON
  - [ ] `uv run python -m ai_workbench --help` exits 0

  **QA Scenarios**:
  ```
  Scenario: CLI smoke
    Tool: Bash
    Steps: uv run python -m ai_workbench doctor --json
    Expected: exit code 0; JSON contains tmux_installed true/false
    Evidence: .sisyphus/evidence/task-1-cli-doctor.txt

  Scenario: Help output stable
    Tool: Bash
    Steps: uv run python -m ai_workbench --help
    Expected: exit code 0
    Evidence: .sisyphus/evidence/task-1-cli-help.txt
  ```

  **Commit**: YES | Message: `feat(workbench): add ai-workbench CLI scaffold` | Files: `pyproject.toml`, `ai_workbench/*`

- [ ] 2. Define data models + JSON schema + atomic persistence + safe paths

  **What to do**:
  - Implement models (Pydantic or dataclasses + manual validation; choose one and use consistently):
    - Workspace, PaneProfile, PaneRuntime (per PRD), plus `schema_version`.
  - Define storage layout under `~/.ai-workbench/` (override by env `AI_WORKBENCH_HOME`):
    - `workspaces.json` (workspace list + pane profiles)
    - `runtime.json` (optional cache; may be re-derived)
    - `notes/<workspace_id>.md`
    - `backups/workspaces.json.bak`
  - Implement atomic write:
    - write temp file, fsync, rename
    - keep last-known-good backup
  - Implement a file lock to prevent concurrent writers (simple lockfile).

  **Must NOT do**:
  - No sqlite in MVP.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — persistence correctness
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [5-13,14-16] | Blocked By: [1]

  **References**:
  - MVP spec (embedded): `.sisyphus/plans/2026-03-11-ai-workspace-tui-mvp.md`
  - Repo style: `backend/config/settings.py` (env-driven paths) (read during implementation)

  **Acceptance Criteria**:
  - [ ] `python -m ai_workbench workspaces list --json` returns an empty list on fresh install
  - [ ] Creating a workspace persists it and it reloads on next run
  - [ ] Corrupt `workspaces.json` falls back to `.bak` and surfaces warning

  **QA Scenarios**:
  ```
  Scenario: Atomic persistence
    Tool: Bash
    Steps: rm -rf ~/.ai-workbench; uv run python -m ai_workbench workspaces list --json
    Expected: exit 0; JSON {"workspaces": []}
    Evidence: .sisyphus/evidence/task-2-empty-list.txt

  Scenario: Backup recovery
    Tool: Bash
    Steps: (1) create workspace; (2) corrupt workspaces.json; (3) run list
    Expected: warning emitted; workspace list still present
    Evidence: .sisyphus/evidence/task-2-backup-recovery.txt
  ```

  **Commit**: YES | Message: `feat(workbench): add workspace models and JSON persistence` | Files: `ai_workbench/models/*`, `ai_workbench/storage/*`

- [ ] 3. Implement tmux adapter (subprocess runner + exact targeting + tagging + dry-run)

  **What to do**:
  - Create `ai_workbench/core/tmux_adapter.py` that:
    - builds argv lists (never `shell=True`)
    - supports `server selector`:
      - default: no socket args
      - optional: `-L <socket_name>` from workspace/server config
      - optional: `-f /dev/null` for tests
    - classifies errors: server missing vs session missing vs permission vs unknown
    - supports `--dry-run/--print-tmux`: prints commands without executing
  - Enumerate sessions/panes using `-F` formats and tab separators.
  - Exact targeting rule:
    - When addressing an existing session by name, always target as `-t =<session_name>` (never prefix match).
    - Never create session names beginning with `=`.
  - Tagging:
    - On session creation: `set-option -t =session @ai_workspace_id <uuid>`
    - On pane creation: `set-option -p -t %pane @ai_role <role>` and `select-pane -T "AIWB <role>"`.
  - Discovery by tag (supports session rename):
    - Implement `find_session_by_workspace_id(workspace_id)` by scanning:
      - `list-sessions -F '#{session_name}\t#{@ai_workspace_id}'`
    - If the stored session name is missing but a session with matching `@ai_workspace_id` exists, update the workspace record to the discovered session name.
  - Core commands needed:
    - `has-session`, `new-session -d`, `split-window -v`, `select-layout even-vertical`, `capture-pane -p -S -N -J`, `list-panes`, `list-sessions`, `attach-session` / `switch-client`, `kill-session` (only for managed sessions)

  **Must NOT do**:
  - Never kill or modify sessions not tagged with `@ai_workspace_id`.
  - Never depend on persisted `pane_id`.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — tmux edge cases
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [6-13,15] | Blocked By: [1,2]

  **References**:
  - tmux exact matching (`=name`): https://man7.org/linux/man-pages/man1/tmux.1.html
  - tmux formats wiki: https://github.com/tmux/tmux/wiki/Formats

  **Acceptance Criteria**:
  - [ ] With `--dry-run`, adapter prints tmux commands and does not mutate state
  - [ ] Adapter can list panes with stable parseable fields
  - [ ] Adapter uses `-t =session` on all session-targeted commands (except `new-session -s`)
  - [ ] Adapter can resolve a workspace session by `@ai_workspace_id` tag even after manual rename

  **QA Scenarios**:
  ```
  Scenario: Isolated tmux server smoke (non-interactive)
    Tool: Bash
    Steps: tmux -L aiwb-test -f /dev/null new-session -d -s aiwb-demo; tmux -L aiwb-test -f /dev/null has-session -t =aiwb-demo
    Expected: exit 0 on has-session
    Evidence: .sisyphus/evidence/task-3-tmux-isolated-smoke.txt

  Scenario: Dry-run produces commands only
    Tool: Bash
    Steps: uv run python -m ai_workbench --dry-run workspaces create --name demo --template triple-agent
    Expected: exit 0; printed lines start with 'tmux'
    Evidence: .sisyphus/evidence/task-3-dry-run.txt
  ```

  **Commit**: YES | Message: `feat(workbench): add tmux adapter with exact targeting` | Files: `ai_workbench/core/tmux_adapter.py`, `ai_workbench/cli.py`

- [ ] 4. Implement layout engine (focus resize + compact mode thresholds)

  **What to do**:
  - Implement `compute_stack_rows(total_rows, pane_count, active_index, note_open)` as per PRD, but:
    - derive header/footer/note rows from actual widget sizes (pass in constants)
    - enforce minima: active>=12, inactive>=3
    - distribute leftover rows to inactive panes up to +2 each
  - Implement responsive rules:
    - width >= 140: allow optional two-column (OFF by default)
    - 100-139: portrait stack
    - <100: compact mode: slim panes 2 lines, tabs abbreviated, notes as overlay only

  **Must NOT do**:
  - No auto focus switch on output.

  **Recommended Agent Profile**:
  - Category: `quick` — algorithm module + tests
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [8-11,16] | Blocked By: [1]

  **References**:
  - Focus resize spec (embedded): `.sisyphus/plans/2026-03-11-ai-workspace-tui-mvp.md`

  **Acceptance Criteria**:
  - [ ] Unit tests cover: small terminal (compact), normal (3 panes), many panes (>=6), note open/closed

  **QA Scenarios**:
  ```
  Scenario: Row allocation invariants
    Tool: Bash
    Steps: uv run pytest -q -k "layout_engine"
    Expected: exit 0
    Evidence: .sisyphus/evidence/task-4-layout-tests.txt

  Scenario: Compact mode threshold
    Tool: Bash
    Steps: uv run pytest -q -k "compact_mode"
    Expected: exit 0
    Evidence: .sisyphus/evidence/task-4-compact-tests.txt
  ```

  **Commit**: YES | Message: `feat(workbench): add focus-resize layout engine` | Files: `ai_workbench/core/layout_engine.py`, tests

- [ ] 5. Implement workspace service (CRUD + safe naming + lifecycle state machine)

  **What to do**:
  - Implement:
    - create workspace: generate UUID id; validate/display name; derive tmux session name `aiwb-<shortid>`
    - list/filter/search by name/tags/repo
    - pin/unpin
    - archive (soft)
    - active workspace switching and persistence
  - Implement workspace lifecycle state transitions exactly as PRD:
    - draft → starting → running
    - running → detached/degraded/missing
    - missing → restoring → running|degraded|error
    - archived terminal state
  - Never use raw user-provided name for tmux session name.

  **Must NOT do**:
  - No destructive deletes by default; archive only.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — state machine + persistence
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [8-13,16] | Blocked By: [2,3]

  **References**:
  - Workspace state machine (embedded): `.sisyphus/plans/2026-03-11-ai-workspace-tui-mvp.md`

  **Acceptance Criteria**:
  - [ ] Creating a workspace persists and assigns tmux session name `aiwb-...`
  - [ ] State transitions are unit-tested (invalid transitions rejected)

  **QA Scenarios**:
  ```
  Scenario: Workspace CRUD non-UI
    Tool: Bash
    Steps: uv run python -m ai_workbench workspaces create --name demo --template triple-agent; uv run python -m ai_workbench workspaces list --json
    Expected: list contains demo workspace with status starting|running
    Evidence: .sisyphus/evidence/task-5-crud.txt

  Scenario: Invalid transition rejected
    Tool: Bash
    Steps: uv run pytest -q -k "workspace_state_machine"
    Expected: exit 0
    Evidence: .sisyphus/evidence/task-5-state-tests.txt
  ```

  **Commit**: YES | Message: `feat(workbench): add workspace service and lifecycle states` | Files: `ai_workbench/core/workspace_service.py`

- [ ] 6. Implement health + preview snapshot pipeline (two-tier poller)

  **What to do**:
  - Implement `health_service`:
    - determine `running/detached/degraded/missing` by:
      - session exists? (has-session)
      - required roles present? (resolve via `@ai_role`)
      - pane dead? (`#{pane_dead}`)
  - Implement `preview_service`:
    - list-panes metadata tick every 1s
    - capture-pane tick every 2-3s only for visible panes (active + slim) OR when activity changed
    - detect output change by hashing captured tail
  - Update `PaneRuntime.last_output_at` when tail changes.
  - Keep capture size bounded: last 200 lines; render 1-3 lines depending on compact mode.

  **Must NOT do**:
  - Do not enable tmux hooks or alter user global monitor-activity.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — concurrency + performance
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [8-13,16] | Blocked By: [3,4,5]

  **References**:
  - tmux list-panes formats: https://github.com/tmux/tmux/wiki/Formats
  - tmux capture-pane: tmux man page `capture-pane` section

  **Acceptance Criteria**:
  - [ ] Dashboard refresh loop does not block UI (no noticeable input lag)
  - [ ] Preview updates when pane output changes

  **QA Scenarios**:
  ```
  Scenario: Snapshot pipeline updates timestamps
    Tool: Bash
    Steps: run isolated tmux session; write to a pane; run service-level test that observes last_output_at changes
    Expected: last_output_at increases after output
    Evidence: .sisyphus/evidence/task-6-preview-updates.txt

  Scenario: No UI freeze under polling
    Tool: Bash
    Steps: uv run pytest -q -k "poller_non_blocking"
    Expected: exit 0
    Evidence: .sisyphus/evidence/task-6-non-blocking.txt
  ```

  **Commit**: YES | Message: `feat(workbench): add health and preview snapshot services` | Files: `ai_workbench/core/preview_service.py`, `ai_workbench/core/health_service.py`

- [ ] 7. Implement restore service + restore modal actions (missing/degraded)

  **What to do**:
  - Implement restore semantics (MVP): deterministic recreate from stored PaneProfiles.
  - Restore options per PRD:
    - Recreate session and relaunch agents (default)
    - Open note only
    - Archive workspace
  - Only offer destructive tmux actions when the session is tagged as managed.
  - If session exists but roles mismatch (externally modified): classify as `degraded` and offer `recreate`.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — failure handling
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [12-13,16] | Blocked By: [2,3,5,6]

  **References**:
  - Restore UI spec (embedded): `.sisyphus/plans/2026-03-11-ai-workspace-tui-mvp.md`

  **Acceptance Criteria**:
  - [ ] When tmux session missing, workspace shows `missing` and restore modal can recreate session
  - [ ] When a pane is dead, workspace shows `degraded` and pane restart is available

  **QA Scenarios**:
  ```
  Scenario: Missing -> restore -> running
    Tool: Bash
    Steps: create workspace; kill its tmux session; open dashboard; trigger restore action
    Expected: session recreated; status running
    Evidence: .sisyphus/evidence/task-7-restore-missing.txt

  Scenario: Degraded -> restart pane
    Tool: Bash
    Steps: create workspace; kill one pane process; refresh; trigger restart for that role
    Expected: pane relaunches; degraded clears
    Evidence: .sisyphus/evidence/task-7-restore-degraded.txt
  ```

  **Commit**: YES | Message: `feat(workbench): add restore service and modal flow` | Files: `ai_workbench/core/restore_service.py`, `ai_workbench/tui/screens/restore_modal.py`

- [ ] 8. Implement Textual dashboard skeleton (tabs + search + preview stack + footer)

  **What to do**:
  - Create Textual App with:
    - top tab bar (workspaces + +)
    - search input below tabs
    - header summary lines (workspace/repo/branch/status + note summary)
    - main vertical preview stack
    - bottom footer with keybinding hints
  - Ensure portrait-first layout (Vertical containers); minimal CSS/TCSS for slim/active previews.
  - Implement `Focus Resize Mode` toggle `z`:
    - when ON: active preview gets majority height; others slim
    - when OFF: equal distribution

  **Recommended Agent Profile**:
  - Category: `visual-engineering` — Textual layout + CSS
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [9-13,16] | Blocked By: [4,5,6]

  **References**:
  - Textual layout guide: https://textual.textualize.io/guide/layout/
  - Textual reactivity: https://textual.textualize.io/guide/reactivity/

  **Acceptance Criteria**:
  - [ ] `uv run python -m ai_workbench` renders dashboard at size (100, 50) and (80, 24) without exceptions
  - [ ] `z` toggles focus resize and reflows preview stack

  **QA Scenarios**:
  ```
  Scenario: Dashboard renders in portrait sizes
    Tool: Bash
    Steps: uv run pytest -q -k "dashboard_renders"
    Expected: exit 0; pilot screenshots saved
    Evidence: .sisyphus/evidence/task-8-dashboard.svg

  Scenario: Focus resize toggle
    Tool: Bash
    Steps: pilot presses 'z' and asserts preview heights changed
    Expected: assertion passes
    Evidence: .sisyphus/evidence/task-8-focus-resize.txt
  ```

  **Commit**: YES | Message: `feat(workbench): add Textual dashboard skeleton` | Files: `ai_workbench/tui/*`

- [ ] 9. Implement pane preview widgets (active + slim) with status badges

  **What to do**:
  - Widget `PanePreview`:
    - renders 1st line: ROLE + lifecycle + activity + last seen
    - renders tail lines depending on mode: active shows ~N lines; slim shows 1 line (2 in non-compact)
  - Map lifecycle/activity per PRD:
    - lifecycle derived from tmux + exit code
    - activity derived from output-change heuristic (idle/streaming/busy/error)
  - Ensure previews do not render escape sequences by default.

  **Recommended Agent Profile**:
  - Category: `visual-engineering` — widget rendering polish
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [10-13,16] | Blocked By: [6,8]

  **Acceptance Criteria**:
  - [ ] Slim preview always shows role + status line even if no output
  - [ ] Error state shows exit code and highlights badge

  **QA Scenarios**:
  ```
  Scenario: Slim preview displays last line
    Tool: Bash
    Steps: simulate runtime with tail_preview; render widget in pilot; assert text contains expected line
    Expected: pass
    Evidence: .sisyphus/evidence/task-9-slim-preview.txt

  Scenario: Error badge
    Tool: Bash
    Steps: set runtime lifecycle crashed + exit code; render; assert badge contains ERROR
    Expected: pass
    Evidence: .sisyphus/evidence/task-9-error-badge.txt
  ```

  **Commit**: YES | Message: `feat(workbench): add pane preview widgets and status badges` | Files: `ai_workbench/tui/widgets/pane_preview.py`

- [ ] 10. Implement note drawer (toggle + edit + save) with structured template

  **What to do**:
  - Bottom drawer toggled by `m`.
  - View mode shows note; edit mode toggled by `e`.
  - Save via `ctrl+s`:
    - writes to `~/.ai-workbench/notes/<workspace_id>.md` atomically
  - If terminal too small (<100 width): show note as Modal overlay instead of drawer.
  - Seed new notes from template.

  **Recommended Agent Profile**:
  - Category: `visual-engineering` — Textual TextArea + UX
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [11-13,16] | Blocked By: [2,8]

  **References**:
  - Note template (embedded): `.sisyphus/plans/2026-03-11-ai-workspace-tui-mvp.md`
  - Textual input/actions: https://textual.textualize.io/guide/input/

  **Acceptance Criteria**:
  - [ ] Note persists and reloads across app restart
  - [ ] Compact mode uses modal overlay for notes

  **QA Scenarios**:
  ```
  Scenario: Edit and save note
    Tool: Bash
    Steps: pilot presses 'm', 'e', types text, presses 'ctrl+s', restarts app, reopens note
    Expected: saved text present
    Evidence: .sisyphus/evidence/task-10-note-save.txt

  Scenario: Compact note modal
    Tool: Bash
    Steps: run_test size=(80,24); press 'm'
    Expected: modal screen visible
    Evidence: .sisyphus/evidence/task-10-note-compact.txt
  ```

  **Commit**: YES | Message: `feat(workbench): add note drawer with save workflow` | Files: `ai_workbench/tui/widgets/note_drawer.py`, `ai_workbench/storage/note_repo.py`

- [ ] 11. Implement workspace create flow (templates + agent commands + safe validation)

  **What to do**:
  - Create screen/modal: name, project_path, template choice.
  - Templates (MVP):
    - triple-agent: roles {claude,codex,opencode}
    - research: roles {claude,shell,logs}
    - debug: roles {opencode,shell,logs}
    - writing: roles {claude,shell}
  - For each pane profile:
    - command default equals role name (e.g. `claude`), but configurable per workspace before launch
    - run via `bash -lc <command>`
  - Validate workspace display name:
    - allow spaces for display
    - generate internal id UUID + safe tmux session name `aiwb-<shortuuid>`

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — correctness + UX
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [12-13,16] | Blocked By: [2,3,5,8]

  **Acceptance Criteria**:
  - [ ] Creating a triple-agent workspace results in 3 tmux panes tagged by role
  - [ ] If a command is missing on PATH, pane is marked failed and error surfaced

  **QA Scenarios**:
  ```
  Scenario: Create workspace with template
    Tool: Bash
    Steps: run dashboard pilot to create triple-agent workspace; then query tmux list-panes and verify 3 panes
    Expected: 3 panes present, each has @ai_role set
    Evidence: .sisyphus/evidence/task-11-create-workspace.txt

  Scenario: Missing command
    Tool: Bash
    Steps: create workspace with role command "definitely-not-a-command"; refresh dashboard
    Expected: pane lifecycle failed; error badge visible
    Evidence: .sisyphus/evidence/task-11-missing-cmd.txt
  ```

  **Commit**: YES | Message: `feat(workbench): add workspace creation templates` | Files: `ai_workbench/tui/screens/create_workspace.py`, `ai_workbench/models/pane.py`

- [ ] 12. Implement attach behavior + portrait preset tmux layout on attach

  **What to do**:
  - Implement `Enter` action:
    - if `TMUX` env set: `tmux switch-client -t =session`
    - else: exec `tmux attach-session -t =session`
  - Before attaching, apply portrait preset to the workspace window:
    - `select-layout main-horizontal` (main on top)
    - swap focused role pane into main slot (`swap-pane`)
    - set `main-pane-height` based on current client height (70%)
  - Persist `active_role` and restore it on next dashboard run.
  - Show toast warning inside tmux: attaching will replace dashboard view.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — tmux layout + exec semantics
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [13,16] | Blocked By: [3,6,8,9]

  **References**:
  - tmux attach-session behavior: https://man7.org/linux/man-pages/man1/tmux.1.html

  **Acceptance Criteria**:
  - [ ] Outside tmux: `Enter` attaches successfully (app exits via exec)
  - [ ] Inside tmux: `Enter` switches client session
  - [ ] Focused role appears in main pane after attach

  **QA Scenarios**:
  ```
  Scenario: Attach outside tmux
    Tool: Bash
    Steps: run dashboard in a non-tmux shell; select workspace; press Enter
    Expected: process replaced by tmux attach; session visible
    Evidence: .sisyphus/evidence/task-12-attach-outside.txt

  Scenario: Switch-client inside tmux
    Tool: Bash
    Steps: start dashboard inside tmux; press Enter
    Expected: client session changes to target
    Evidence: .sisyphus/evidence/task-12-switch-client.txt
  ```

  **Commit**: YES | Message: `feat(workbench): attach to tmux with portrait preset layout` | Files: `ai_workbench/core/tmux_adapter.py`, `ai_workbench/tui/screens/dashboard.py`

- [ ] 13. Implement missing/degraded UI states + actions (restore, restart pane, archive)

  **What to do**:
  - Show workspace status badge (LIVE/DEGRADED/MISSING/ERROR/ARCHIVED).
  - Bind `r` to restore flow (when missing/degraded).
  - Bind `R` to restart focused pane (role) using restore service.
  - Bind `x` to archive.
  - Ensure actions are disabled when workspace archived.

  **Recommended Agent Profile**:
  - Category: `visual-engineering` — UI state fidelity
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [16] | Blocked By: [6,7,8,9]

  **Acceptance Criteria**:
  - [ ] Missing workspace shows restore CTA; restore recreates session
  - [ ] Degraded shows restart pane CTA; restart clears degraded

  **QA Scenarios**:
  ```
  Scenario: Restore button visible only for missing
    Tool: Bash
    Steps: pilot marks workspace missing; refresh UI
    Expected: restore action enabled
    Evidence: .sisyphus/evidence/task-13-restore-cta.txt

  Scenario: Restart pane action
    Tool: Bash
    Steps: simulate pane crashed; press 'R'
    Expected: tmux respawn/relaunch occurs; UI updates
    Evidence: .sisyphus/evidence/task-13-restart-pane.txt
  ```

  **Commit**: YES | Message: `feat(workbench): add degraded/missing actions and keybindings` | Files: `ai_workbench/tui/screens/dashboard.py`

- [ ] 14. Unit tests: persistence + state machine + layout engine

  **What to do**:
  - Add tests under `tests/workbench_tui/` (root test suite) for:
    - atomic persistence + backup recovery
    - workspace lifecycle transitions
    - layout engine invariants

  **Recommended Agent Profile**:
  - Category: `deep` — test coverage and edge cases
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: [16] | Blocked By: [2,4,5]

  **Acceptance Criteria**:
  - [ ] `uv run pytest -q tests/workbench_tui` exits 0

  **QA Scenarios**:
  ```
  Scenario: Run unit tests
    Tool: Bash
    Steps: uv run pytest -q tests/workbench_tui
    Expected: exit 0
    Evidence: .sisyphus/evidence/task-14-unit-tests.txt
  ```

  **Commit**: YES | Message: `test(workbench): add unit tests for persistence and layout` | Files: `tests/workbench_tui/*`

- [ ] 15. tmux adapter tests with fake tmux + optional integration tests

  **What to do**:
  - Implement fake tmux binary approach for unit tests:
    - place a test helper script in `tests/workbench_tui/fake_bin/tmux`
    - tests prepend PATH so adapter uses fake
    - fake prints canned outputs for `list-panes`, `has-session`, etc.
  - Add optional integration tests gated by env `AIWB_RUN_TMUX_IT=1`:
    - run isolated server: `tmux -L aiwb-test -f /dev/null ...`
    - create/destroy only tagged sessions

  **Recommended Agent Profile**:
  - Category: `deep` — determinism + hermetic tests
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: [16] | Blocked By: [3]

  **Acceptance Criteria**:
  - [ ] Default test run passes without a real tmux server
  - [ ] Integration tests pass when enabled and tmux is installed

  **QA Scenarios**:
  ```
  Scenario: Hermetic fake tmux tests
    Tool: Bash
    Steps: uv run pytest -q -k "fake_tmux"
    Expected: exit 0
    Evidence: .sisyphus/evidence/task-15-fake-tmux.txt

  Scenario: Optional real tmux integration
    Tool: Bash
    Steps: AIWB_RUN_TMUX_IT=1 uv run pytest -q -k "tmux_integration"
    Expected: exit 0 (when tmux available)
    Evidence: .sisyphus/evidence/task-15-tmux-it.txt
  ```

  **Commit**: YES | Message: `test(workbench): add fake tmux and optional integration tests` | Files: `tests/workbench_tui/*`

- [ ] 16. Textual pilot tests: keybindings + responsive behavior + notes

  **What to do**:
  - Use `App.run_test()` and `Pilot` to test:
    - Tab/Shift+Tab workspace switching
    - j/k pane focus switching
    - m/e/ctrl+s notes
    - z focus resize toggle
    - compact mode behavior at size=(80,24)
  - Save screenshot evidence via Textual snapshot or `save_screenshot`.

  **Recommended Agent Profile**:
  - Category: `deep` — UI testing
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: [] | Blocked By: [8-13]

  **References**:
  - Textual testing guide: https://textual.textualize.io/guide/testing/

  **Acceptance Criteria**:
  - [ ] `uv run pytest -q -k "pilot"` exits 0

  **QA Scenarios**:
  ```
  Scenario: Pilot keybinding coverage
    Tool: Bash
    Steps: uv run pytest -q -k "pilot"
    Expected: exit 0
    Evidence: .sisyphus/evidence/task-16-pilot.txt

  Scenario: Compact mode layout
    Tool: Bash
    Steps: uv run pytest -q -k "compact" 
    Expected: exit 0
    Evidence: .sisyphus/evidence/task-16-compact.txt
  ```

  **Commit**: YES | Message: `test(workbench): add Textual pilot tests for keybindings and layout` | Files: `tests/workbench_tui/test_dashboard_pilot.py`

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA (agent-executed) — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Use 6-10 small commits aligned with TODO groupings (scaffold, persistence, tmux adapter, layout, UI, restore, tests).
- Never commit `~/.ai-workbench` artifacts or `.sisyphus/evidence/*` to git.

## Success Criteria
- MVP acceptance list in PRD is met with automated verification where feasible.
- Tool is safe around user tmux environment (no accidental session kills; exact targeting; isolated integration tests).

## Appendix: MVP Specs (Embedded)

### A1) Workspace Lifecycle States
- `draft`, `starting`, `running`, `detached`, `degraded`, `missing`, `restoring`, `archived`, `error`

Allowed transitions:
```
draft -> starting -> running
starting -> error
running -> detached
detached -> running
running -> degraded
degraded -> restoring -> running
running/detached/degraded -> missing
missing -> restoring -> running | degraded | error
running/detached/error -> archived
```

### A2) Pane Lifecycle States
- `configured`, `launching`, `live`, `exited`, `crashed`, `restarting`, `failed`

Allowed transitions:
```
configured -> launching -> live
launching -> failed
live -> exited
live -> crashed
exited/crashed -> restarting -> live | failed
```

### A3) Pane Activity Badges (UI-only)
- `idle`, `streaming`, `busy`, `waiting`, `done`, `warn`, `error`

### A4) Keybindings (MVP)
- `Tab` / `Shift+Tab`: workspace tab switch
- `h` / `l`: prev/next workspace
- `j` / `k`: pane focus switch
- `Enter`: attach/switch tmux session
- `m`: toggle note drawer
- `e`: note edit mode (drawer)
- `Ctrl+S`: save note
- `/`: workspace search
- `n`: create workspace
- `r`: restore (missing/degraded)
- `R`: restart focused pane
- `x`: archive workspace
- `z`: toggle focus resize mode
- `?`: help
- `Esc`: close drawer/modal
- `q`: quit

### A5) Focus Resize Row Allocation (Reference Implementation)
Executor implements the same constraints and behavior:
- active pane: 65-75% of available body height, min 12 rows
- inactive panes: min 3 rows (or 2 rows in compact mode)
- leftover rows distributed across inactive panes (cap +2)

Pseudo-code (copy into `layout_engine.py` and adapt for header/footer/note sizes):
```python
def compute_stack_rows(total_rows: int, pane_count: int, active_index: int, note_open: bool):
    header = 4
    footer = 2
    note = 12 if note_open else 0

    body = total_rows - header - footer - note
    slim_min = 4 if body >= 28 else 3
    active_min = 12

    rows = [slim_min] * pane_count
    active_rows = int(body * 0.7)
    active_rows = max(active_min, active_rows)

    max_active = body - (pane_count - 1) * slim_min
    active_rows = min(active_rows, max_active)

    rows[active_index] = active_rows

    used = sum(rows)
    leftover = body - used

    i = 0
    while leftover > 0:
        if i != active_index and rows[i] < slim_min + 2:
            rows[i] += 1
            leftover -= 1
        i = (i + 1) % pane_count

    return rows
```

### A6) Note Template
```md
# Goal

# Current Status

# Blockers

# Next Actions
- 

# Related Files
- 

# Related Branch

```
