# VS Code-Friendly tmux Workspace Switching Keys

## TL;DR
> **Summary**: Replace unreliable VS Code-captured `Ctrl+1..9` tmux workspace switching with `Alt+1..9`, and add a single-key "go back to first screen" binding that returns to the saved home session when available.
> **Deliverables**: tmux keybinding swap + go-home binding + updated UI hint text + tests
> **Effort**: Short
> **Parallel**: YES - 2 waves
> **Critical Path**: Update tmux bindings -> update tests -> run workbench test suite

## Context
### Original Request
- VS Code integrated terminal captures `Ctrl+1`, `Ctrl+2`; change to something like `Tab+1`, `Tab+2`.
- Also: "first screen" navigation (how to go back after attaching into a workspace tmux).

### Interview Summary
- Literal `Tab+digit` is not safe in tmux root table because it breaks shell/editor tab-completion.
- Selected scheme: `Alt+Number`.
- Interpretation of "first screen": return to originating tmux session when switching from inside tmux; otherwise detach back to shell.

### Metis Review (gaps addressed)
- Add explicit "go home" binding that reads `@ai_workbench_home` (already written during inside-tmux switch) and falls back deterministically.
- Add tests that assert the exact `bind-key` emission to catch regressions without requiring real tmux.

## Work Objectives
### Core Objective
- Inside tmux (including VS Code integrated terminal), workspace switching does not depend on VS Code-intercepted `Ctrl+1..9`.

### Deliverables
- Update `ai_workbench/core/tmux_adapter.py` tmux root keybindings:
  - `M-1..M-9` (Alt+1..Alt+9) switch workspaces by index (1-based).
  - Remove `C-1..C-9` bindings.
  - Add `M-0` (Alt+0) go-home: switch to `@ai_workbench_home` if set, else detach.
- Update user-facing hint text in `ai_workbench/tui/screens/dashboard.py` attach notification to mention the new go-home key.
- Add/extend tests in `tests/workbench_tui/` to validate binding emission.

### Definition of Done (verifiable)
- `pytest tests/workbench_tui -v` passes.
- In a real tmux session launched by AIWB, `tmux list-keys -T root` contains `M-1..M-9` bindings pointing to `python -m ai_workbench cycle --index N`.
- In VS Code terminal attached to a workspace tmux session, pressing `Alt+1` switches workspaces (if VS Code passes the key through).

### Must Have
- No `Tab` keybinding changes (do not break completion).
- Go-home binding works even if home session is missing (deterministic fallback).

### Must NOT Have
- Do NOT change CLI `cycle --index` semantics (keep 1-based).
- Do NOT require manual config edits in tmux.conf.

## Verification Strategy
- Test decision: tests-after (pytest)
- Evidence: executor captures terminal output of `tmux list-keys` (if doing live verification) into `.sisyphus/evidence/`.

## Execution Strategy
### Parallel Execution Waves
Wave 1:
- tmux binding changes + UI hint update
- test additions for binding emission

Wave 2:
- full test run + optional live tmux smoke verification

### Dependency Matrix
- Binding tests depend on binding changes.
- Full suite depends on both.

## TODOs

- [ ] 1. Swap tmux workspace index bindings to Alt+Number

  **What to do**:
  - Edit `ai_workbench/core/tmux_adapter.py` in `TmuxAdapter.configure_workspace_bar()`.
  - Keep existing root-table bindings:
    - `C-Tab` -> `cycle --direction next`
    - `C-S-Tab` -> `cycle --direction prev`
    - `C-t` -> `quick-create`
    - `M-t` -> `add-pane`
  - Replace the loop that adds `C-1..C-9` bindings with `M-1..M-9`.
    - Each should run: `{python} -m ai_workbench cycle --index {i}`.
  - Add a new root-table binding: `M-0` (Alt+0) go-home.
    - Use this exact `run-shell` command (copy/paste):
      `home='#{@ai_workbench_home}'; [ -n "$home" ] && tmux switch-client -t "=$home" || tmux detach-client`

  **Must NOT do**:
  - Do NOT bind literal `Tab` or create a `Tab`-based chord.
  - Do NOT keep `C-1..C-9` (user wants away from Ctrl+digits).

  **Recommended Agent Profile**:
  - Category: `quick` - small change in one function with quoting care
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 2 | Blocked By: []

  **References**:
  - Binding definition: `ai_workbench/core/tmux_adapter.py:308` (`configure_workspace_bar`)
  - Home session option write + prefix return: `ai_workbench/core/tmux_adapter.py:267` (`attach_or_switch`)
  - Cycle CLI: `ai_workbench/cli.py:226` (`cmd_cycle`, 1-based index)

  **Acceptance Criteria**:
  - [ ] `pytest tests/workbench_tui/test_tmux_adapter_fake.py -v` passes (after updating tests in Task 3)
  - [ ] `grep -n "C-1" ai_workbench/core/tmux_adapter.py` finds no remaining `C-1..C-9` binding loop

  **QA Scenarios**:
  ```
  Scenario: Root keybindings contain M-1..M-9 and M-0
    Tool: Bash
    Steps: Start a workspace, then run `tmux list-keys -T root | grep -E 'M-[0-9]'`
    Expected: lines exist for M-1..M-9 cycle --index N and M-0 go-home
    Evidence: .sisyphus/evidence/task-1-tmux-list-keys.txt

  Scenario: Go-home with missing home detaches
    Tool: Bash
    Steps: Run `tmux list-keys -T root | grep -F "home='#{@ai_workbench_home}'"`
    Expected: binding contains the `detach-client` fallback
    Evidence: .sisyphus/evidence/task-1-go-home-detach.txt
  ```

  **Commit**: YES | Message: `fix(aiwb): use alt-number tmux workspace switching` | Files: `ai_workbench/core/tmux_adapter.py`

- [ ] 2. Update UI hint text to match the new tmux navigation

  **What to do**:
  - In `ai_workbench/tui/screens/dashboard.py` update the attach success notification text to mention:
    - Return to previous session: `Alt+0` (preferred) and `prefix+B` (fallback when home is set)
    - Keep mention of `prefix+L` if still relevant for "last session".

  **Must NOT do**:
  - Do NOT claim `Ctrl+1..9` works.

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [] | Blocked By: []

  **References**:
  - Attach notification: `ai_workbench/tui/screens/dashboard.py` (in `action_attach_workspace` notify block)

  **Acceptance Criteria**:
  - [ ] `pytest tests/workbench_tui/test_dashboard_pilot.py -v` passes

  **QA Scenarios**:
  ```
  Scenario: Attach notification mentions Alt+0
    Tool: Bash
    Steps: Run TUI, attach workspace, observe notification content (or assert in test)
    Expected: message includes Alt+0 guidance
    Evidence: .sisyphus/evidence/task-2-attach-notify.txt

  Scenario: No outdated Ctrl+1 hints
    Tool: Bash
    Steps: Search strings in dashboard screen file
    Expected: no text instructs Ctrl+1..9 for workspace switching
    Evidence: .sisyphus/evidence/task-2-no-ctrl-hints.txt
  ```

  **Commit**: YES | Message: `docs(aiwb): update tmux navigation hint` | Files: `ai_workbench/tui/screens/dashboard.py`

- [ ] 3. Add tests to assert tmux binding emission in dry-run mode

  **What to do**:
  - Extend `tests/workbench_tui/test_tmux_adapter_fake.py` (or add a new test file) to validate:
    - `configure_workspace_bar("aiwb-demo")` emits `bind-key -T root M-1 .. M-9` and does not emit `C-1..C-9`.
    - `M-0` binding exists.
  - Implementation approach (choose one; do not leave executor to decide):
    - Prefer: instantiate `TmuxAdapter(dry_run=True, print_tmux=True)` and call `configure_workspace_bar("aiwb-demo")`, then assert on `capsys.readouterr().out`.

  **Must NOT do**:
  - Do NOT require real tmux; keep unit-level.

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: [] | Blocked By: 1

  **References**:
  - Existing dry-run print test: `tests/workbench_tui/test_tmux_adapter_fake.py:28`
  - Binding function: `ai_workbench/core/tmux_adapter.py:308`

  **Acceptance Criteria**:
  - [ ] `pytest tests/workbench_tui/test_tmux_adapter_fake.py -v` passes
  - [ ] Output asserts include `bind-key -T root M-1 run-shell` ... `--index 1` and include `M-0`

  **QA Scenarios**:
  ```
  Scenario: Dry-run prints new bindings
    Tool: Bash
    Steps: pytest -q tests/workbench_tui/test_tmux_adapter_fake.py -k configure_workspace_bar
    Expected: tests assert M-1..M-9 present, C-1..C-9 absent
    Evidence: .sisyphus/evidence/task-3-pytest-output.txt

  Scenario: Go-home binding present
    Tool: Bash
    Steps: same test run
    Expected: M-0 binding detected
    Evidence: .sisyphus/evidence/task-3-go-home-binding.txt
  ```

  **Commit**: YES | Message: `test(aiwb): assert tmux keybinding emission` | Files: `tests/workbench_tui/test_tmux_adapter_fake.py`

- [ ] 4. Verification run + optional VS Code notes

  **What to do**:
  - Run: `pytest tests/workbench_tui -v`.
  - Optional (if VS Code still intercepts Alt+digits): document recommended VS Code settings in a short note (either existing doc location or `README` entry) with these keys:
    - `terminal.integrated.sendKeybindingsToShell`
    - `terminal.integrated.allowChords`
    - macOS: `terminal.integrated.macOptionIsMeta`

  - Recommended VS Code settings snippet (copy/paste into user settings, adjust to taste):
    ```json
    {
      "terminal.integrated.sendKeybindingsToShell": true,
      "terminal.integrated.allowChords": false,
      "terminal.integrated.macOptionIsMeta": true
    }
    ```

  **Recommended Agent Profile**:
  - Category: `unspecified-low`
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [] | Blocked By: 1,2,3

  **Acceptance Criteria**:
  - [ ] `pytest tests/workbench_tui -v` exits 0

  **QA Scenarios**:
  ```
  Scenario: Full suite passes
    Tool: Bash
    Steps: pytest tests/workbench_tui -v
    Expected: all pass (integration may skip)
    Evidence: .sisyphus/evidence/task-4-pytest-full.txt
  ```

  **Commit**: NO (unless docs were added)

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [ ] F1. Plan Compliance Audit - oracle
- [ ] F2. Code Quality Review - unspecified-high
- [ ] F3. Real Manual QA - unspecified-high
- [ ] F4. Scope Fidelity Check - deep

## Commit Strategy
- Prefer 3 small commits matching tasks 1/2/3; do not mix unrelated repo changes.

## Success Criteria
- VS Code integrated terminal can switch workspaces without Ctrl+digits.
- Clear, discoverable way to return to the prior screen/session.
