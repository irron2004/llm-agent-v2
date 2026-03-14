# Task Start Kickoff (Cross-Agent Skill)

## Purpose

Use this kickoff checklist at the start of every non-trivial task to prevent scope drift,
contract breakage, and missing verification evidence.

Canonical rule:
- Keep the full workflow here.
- Keep runtime entry docs and local skill wrappers short and aligned to this file.

## Startup Read Order

Use the following read order when a runtime starts a task:
1. Repo entry document for the runtime:
   - Codex: `AGENTS.md`
   - Claude: `CLAUDE.md`
2. `docs/2026-03-14-agent-개발-운영.md`
   - Read section `0) 신규 agent 5분 온보딩 (필독)`
   - Read section `5.5 비사소 작업 판정 기준`
3. `docs/contracts/product-contract.md`
4. `docs/tasks/TASK_TEMPLATE.md`
5. `docs/tasks/README.md` if handoff or verification logging rules are needed

## Scope Boundary

The full kickoff is required before code edits on non-trivial tasks.

You may skip task doc creation for:
- tiny one-file typo or docs-only fixes,
- read-only investigation,
- command-only inspection that does not edit code.

If a tiny or exploratory task grows into a non-trivial code change, run the full kickoff before the first edit.

## Mandatory Checklist

1. Check working tree ownership
   - Run `git status --short`.
   - Treat unrelated dirty files as user-owned changes.
2. Enforce one-agent-per-worktree
   - Confirm this worktree is not shared by another active coding agent.
   - If uncertain, create a dedicated worktree first.
3. Lock contract scope before edits
   - Read `docs/contracts/product-contract.md`.
   - Identify `Contracts To Preserve`.
4. Prepare task doc before edits (non-trivial tasks)
   - Use `docs/tasks/TASK_TEMPLATE.md`.
   - Fill `Contracts To Preserve`, `Allowed Files`, `Verification Plan` first.
5. Execute with scope discipline
   - If scope grows, update task doc before editing new files.
6. Close with evidence
   - Record `Verification Results` after running commands.
   - If handed off, fill `Handoff`.

## Stop Conditions

- Do not edit code if `Verification Plan` is missing.
- Do not edit code if relevant contract scope is unresolved.
- Do not continue when worktree isolation is uncertain.

## Runtime Wrappers

- OMC: `.omc/skills/task-start-kickoff/SKILL.md`
- OpenCode / oh-my-opencode: `.opencode/skills/task-start-kickoff/SKILL.md`
- Claude: `.claude/skills/task-start-kickoff/SKILL.md`
- Codex: `.codex/skills/task-start-kickoff/SKILL.md`

If a runtime does not auto-load local skills, open this file and run the checklist manually.
