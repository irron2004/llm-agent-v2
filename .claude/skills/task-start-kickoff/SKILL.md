---
name: task-start-kickoff
description: Run the repository kickoff checklist before any non-trivial coding task in Claude.
triggers:
  - new task
  - task kickoff
  - non-trivial change
  - task doc
argument-hint: "<task-intent>"
---

# Task Start Kickoff (Claude)

Canonical source:
- `docs/agent-skills/task-start-kickoff.md`

Use this skill at the start of every non-trivial task.

## Startup Read Order

1. `CLAUDE.md`
2. `docs/agent-skills/task-start-kickoff.md`
3. Follow the canonical checklist there
4. If the task is tiny or read-only, use the `Scope Boundary` rules in the canonical source

## Stop Conditions

- Do not edit code if the canonical checklist says to stop.
