# Task: <short title>

Status: Draft
Owner: <agent or person>
Branch or worktree: <branch name or worktree path>
Created: YYYY-MM-DD

## Goal

Describe the intended outcome in 2-4 lines.

## Why

Describe the user or system problem this task addresses.

## Contracts To Preserve

List contract IDs from [`docs/contracts/product-contract.md`](../contracts/product-contract.md).

- C-...
- C-...

## Contracts To Update

Leave this empty unless the task intentionally changes protected behavior.

- None

If not `None`, include:
- exact contract ID,
- why the behavior must change,
- what test updates will prove the new contract.

## Allowed Files

List the files or directories this task is expected to touch.

- `backend/...`
- `frontend/...`
- `tests/...`

## Out Of Scope

List nearby changes that should not be pulled into this task.

- No unrelated prompt rewrites
- No refactor of resume persistence

## Risks

List rollback or regression risks that need explicit checking.

- API response shape drift
- Hidden frontend prompt regression
- Interrupt/resume state mismatch

## Verification Plan

Write the exact commands to run before closing the task.

```bash
uv run pytest tests/api/test_agent_response_metadata_contract.py -v
uv run pytest tests/api/test_agent_retrieval_only.py -v
cd frontend && npm run test -- src/features/chat/__tests__/chat-page-device-panel.test.tsx
```

## Verification Results

Record actual execution outcomes after running the plan.

- command: `...`
  - result: pass | fail | skip
  - note: what behavior this verifies
  - evidence: (required if fail) key error lines or log snippet

## Handoff

Fill this section if work is handed off or resumed in another session.

- Current status: done / in-progress / blocked
- Last passing verification command and result
- Remaining TODOs (priority order)
- Whether `Allowed Files` changed and why
- Whether `Contracts To Update` is expected

## Change Log

Update this section as scope changes.

- YYYY-MM-DD: task created

## Final Check

Complete before marking the task done.

- [ ] Diff stayed inside allowed files, or this doc was updated first
- [ ] Protected contract IDs were re-checked
- [ ] Verification commands were run, or blockers were recorded
- [ ] Any contract changes were reflected in `product-contract.md`
- [ ] Remaining risks and follow-ups were documented
