# Task Docs

Use a task document for any change that is more than a tiny one-file fix, especially when:
- multiple agents are working in parallel,
- the work touches both backend and frontend,
- the work changes agent routing, interrupt/resume, or response payloads,
- the work can affect an existing protected contract.

For the single source of truth on non-trivial task judgment, see
[`docs/2026-03-14-agent-개발-운영.md`](../2026-03-14-agent-개발-운영.md)
section `5.5 비사소 작업 판정 기준`.

Recommended workflow:
1. Create a task file from [`TASK_TEMPLATE.md`](./TASK_TEMPLATE.md).
2. Name it `TASK-YYYYMMDD-short-name.md`.
3. Record the allowed files before editing code.
4. List the contract IDs that must remain true.
5. Update the task file if the scope expands.
6. Run the verification plan and record `Verification Results`.
7. If handing off or resuming later, fill the `Handoff` section.
8. Close the task with final risks and follow-up items.

Task doc minimum sections:
- `Contracts To Preserve`
- `Allowed Files`
- `Verification Plan`
- `Verification Results`
- `Handoff` (required if handoff/resume happens)

Reference sample:
- [`docs/tasks/TASK-20260313-onboarding-flow-sample.md`](./TASK-20260313-onboarding-flow-sample.md)

Parallel work rule:
- Do not run multiple coding agents in the same worktree.
- Give each agent its own branch or `git worktree`.

Example:
```bash
git worktree add ../llm-agent-v2-task-agent-api -b task/agent-api-guardrails
git worktree add ../llm-agent-v2-task-chat-ui -b task/chat-ui-regression-fix
```

Minimal operator checklist:
- One agent per worktree.
- One task doc per change stream.
- One explicit list of protected contract IDs.
- One verification section with commands and outcomes.
- One handoff section when work is passed or resumed.
