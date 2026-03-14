# Task: agent startup entrypoints and kickoff discoverability

Status: Done
Owner: Codex
Branch or worktree: `86ewk6385-1차-PE-피드백-v2`
Created: 2026-03-13

## Goal

Strengthen repository startup entrypoints so that agents read the startup instruction docs
before non-trivial work, even when local skill auto-loading is inconsistent across runtimes.

## Why

The repo already had kickoff docs and local skill wrappers, but the startup path was weak in
Codex, Claude, and OpenCode because the runtime might not auto-load local skills.
The entry documents needed to force the same read order and the wrappers needed to be
self-contained if they do get loaded.

## Contracts To Preserve

- None

This is a docs-and-instructions change only. No runtime product behavior is intended to change.

## Contracts To Update

- None

## Allowed Files

- `AGENTS.md`
- `CLAUDE.md`
- `docs/agent-skills/task-start-kickoff.md`
- `.codex/skills/task-start-kickoff/SKILL.md`
- `.claude/skills/task-start-kickoff/SKILL.md`
- `.opencode/skills/task-start-kickoff/SKILL.md`
- `docs/2026-03-14-agent-개발-운영.md`
- `docs/tasks/TASK-20260313-onboarding-flow-sample.md`
- `docs/tasks/TASK-20260313-agent-startup-entrypoints.md`

## Out Of Scope

- No runtime plugin implementation
- No CI or pre-commit enforcement
- No backend or frontend code changes

## Risks

- Entry docs and runtime wrappers may drift if only one side is updated later
- Local skill auto-loading still depends on runtime support outside this repo

## Verification Plan

Manual verification for this docs-only task:

```bash
test -f AGENTS.md
test -f CLAUDE.md
test -f docs/2026-03-14-agent-개발-운영.md
test -f docs/agent-skills/task-start-kickoff.md
test -f .codex/skills/task-start-kickoff/SKILL.md
test -f .claude/skills/task-start-kickoff/SKILL.md
test -f .opencode/skills/task-start-kickoff/SKILL.md
test -f docs/tasks/TASK-20260313-onboarding-flow-sample.md
```

## Verification Results

- command: `test -f AGENTS.md`
  - result: pass
  - note: confirmed repo-level Codex/OpenCode entry document exists
- command: `test -f CLAUDE.md`
  - result: pass
  - note: confirmed Claude entry document exists
- command: `test -f docs/2026-03-14-agent-개발-운영.md`
  - result: pass
  - note: confirmed common startup guide exists
- command: `test -f docs/agent-skills/task-start-kickoff.md`
  - result: pass
  - note: confirmed canonical kickoff checklist exists
- command: `test -f .codex/skills/task-start-kickoff/SKILL.md`
  - result: pass
  - note: confirmed Codex runtime wrapper exists
- command: `test -f .claude/skills/task-start-kickoff/SKILL.md`
  - result: pass
  - note: confirmed Claude runtime wrapper exists
- command: `test -f .opencode/skills/task-start-kickoff/SKILL.md`
  - result: pass
  - note: confirmed OpenCode runtime wrapper exists
- command: `test -f docs/tasks/TASK-20260313-onboarding-flow-sample.md`
  - result: pass
  - note: confirmed onboarding sample doc exists for task doc reference

## Handoff

- Current status: done
- Last passing verification command and result: file existence checks passed
- Remaining TODOs (priority order):
  1. Decide whether to add runtime-specific config that truly auto-invokes the kickoff skill
  2. Consider CI or pre-commit guardrails for task doc presence on large diffs
- Whether `Allowed Files` changed and why: no
- Whether `Contracts To Update` is expected: no

## Change Log

- 2026-03-13: created task doc
- 2026-03-13: strengthened AGENTS and CLAUDE startup protocol
- 2026-03-13: made Codex/Claude/OpenCode kickoff wrappers self-contained
- 2026-03-13: clarified canonical startup read order in agent docs
- 2026-03-13: fixed onboarding sample path to existing frontend component

## Final Check

- [x] Diff stayed inside allowed files, or this doc was updated first
- [x] Protected contract IDs were re-checked
- [x] Verification commands were run, or blockers were recorded
- [x] Any contract changes were reflected in `product-contract.md`
- [x] Remaining risks and follow-ups were documented
