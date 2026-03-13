---
name: task-start-kickoff
description: Cross-agent kickoff checklist wrapper for OpenCode/oh-my-opencode.
triggers:
  - new task
  - task kickoff
  - non-trivial change
argument-hint: "<task-intent>"
---

# Task Start Kickoff (OpenCode Wrapper)

Canonical source:
- `docs/agent-skills/task-start-kickoff.md`

Runtime usage:
- If your runtime can load local skills, run `task-start-kickoff`.
- If not, open the canonical file and execute its checklist manually before edits.
