# Task: onboarding flow sample (reference)

Status: Example
Owner: new-agent
Branch or worktree: task/onboarding-flow-sample
Created: 2026-03-13

## Goal

Show a complete task document lifecycle for new agents: scope setup,
contract protection, verification evidence, and handoff information.

## Why

New agents need one concrete example that demonstrates how to use
`TASK_TEMPLATE.md` end-to-end without guessing where to write results.

## Contracts To Preserve

- C-API-001
- C-UI-002

## Contracts To Update

- None

## Allowed Files

- `backend/api/routers/agent.py`
- `frontend/src/features/chat/components/DevicePanel.tsx`
- `tests/api/test_agent_response_metadata_contract.py`
- `frontend/src/features/chat/__tests__/chat-page-device-panel.test.tsx`
- `docs/tasks/TASK-20260313-onboarding-flow-sample.md`

## Out Of Scope

- No retrieval strategy refactor
- No interrupt/resume architecture changes
- No unrelated frontend layout updates

## Risks

- API metadata response shape drift
- Device selection command semantics drift (`0`, `1`, `2`)
- Test updates that accidentally weaken assertions

## Verification Plan

```bash
uv run pytest tests/api/test_agent_response_metadata_contract.py -v
cd frontend && npm run test -- src/features/chat/__tests__/chat-page-device-panel.test.tsx
```

## Verification Results

- command: `uv run pytest tests/api/test_agent_response_metadata_contract.py -v`
  - result: pass
  - note: validated required metadata keys are preserved
- command: `cd frontend && npm run test -- src/features/chat/__tests__/chat-page-device-panel.test.tsx`
  - result: pass
  - note: validated pending regeneration device flow behavior

## Handoff

- Current status: done
- Last passing verification command and result: both plan commands passed
- Remaining TODOs (priority order):
  1. Add project-specific sample for backend-only task
  2. Add project-specific sample for docs-only task
- Whether `Allowed Files` changed and why: no
- Whether `Contracts To Update` is expected: no

## Change Log

- 2026-03-13: added reference sample for onboarding

## Final Check

- [x] Diff stayed inside allowed files, or this doc was updated first
- [x] Protected contract IDs were re-checked
- [x] Verification commands were run, or blockers were recorded
- [x] Any contract changes were reflected in `product-contract.md`
- [x] Remaining risks and follow-ups were documented
