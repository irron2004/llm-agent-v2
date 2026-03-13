---
name: task-start-kickoff
description: Cross-agent kickoff checklist wrapper for Claude.
triggers:
  - new task
  - task kickoff
  - non-trivial change
argument-hint: "<task-intent>"
---

# Task Start Kickoff (Claude Wrapper)

Canonical source:
- `docs/agent-skills/task-start-kickoff.md`

Runtime usage:
- If your runtime can load local skills, run `task-start-kickoff`.
- If not, open the canonical file and execute its checklist manually before edits.
