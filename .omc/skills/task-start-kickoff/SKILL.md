---
name: task-start-kickoff
description: Run a mandatory kickoff checklist before any non-trivial coding task in this repository.
triggers:
  - new task
  - start task
  - task kickoff
  - non-trivial change
  - task doc
argument-hint: "<task-intent>"
---

# Task Start Kickoff Skill

Canonical source:
- `docs/agent-skills/task-start-kickoff.md`

Use this file as OMC runtime wrapper of the canonical checklist.

## Purpose

Prevent scope drift and accidental regressions by forcing the repository kickoff checklist
before code edits.

## When to Activate

- A new coding task starts.
- The change is non-trivial (see `docs/2026-03-14-agent-개발-운영.md` section `5.5`).
- Work may touch protected contracts, multi-file changes, or cross-layer behavior.

## Workflow

1. Inspect working tree and ownership
   - Run `git status --short`.
   - Treat unrelated dirty files as user-owned changes.
2. Enforce one-agent-per-worktree
   - Confirm this worktree is not shared with another active coding agent.
   - If uncertain, create a dedicated worktree first.
3. Lock contract scope before edits
   - Read `docs/contracts/product-contract.md` and identify relevant contract IDs.
   - If no contract matches, record `Contracts To Preserve: None (reason)` and mark status as blocked until clarified.
4. Create or update task doc before edits (non-trivial tasks)
   - Use `docs/tasks/TASK_TEMPLATE.md`.
   - Fill at minimum: `Contracts To Preserve`, `Allowed Files`, `Verification Plan`.
5. Start execution discipline
   - Track work with a todo list.
   - Keep exactly one task `in_progress` at a time.
6. Close with evidence
   - Record `Verification Results` in the task doc after running commands.
   - If handing off, fill `Handoff` section with status, remaining TODOs, and scope changes.

## Stop Conditions

- Do not edit code if `Verification Plan` is missing.
- Do not edit code if contract scope is unresolved.
- Do not continue in a shared worktree.

## Required References

- `docs/2026-03-14-agent-개발-운영.md`
- `docs/contracts/product-contract.md`
- `docs/tasks/TASK_TEMPLATE.md`
- `docs/tasks/README.md`
- `AGENTS.md`

## Quick Output Template

Use this structure when reporting kickoff completion:

```md
Kickoff complete
- worktree isolation: confirmed
- contracts identified: C-...
- task doc: docs/tasks/TASK-YYYYMMDD-short-name.md
- allowed files: set
- verification plan: set
```

## Example

```bash
/oh-my-claudecode:task-start-kickoff retrieval-only regression follow-up
```
