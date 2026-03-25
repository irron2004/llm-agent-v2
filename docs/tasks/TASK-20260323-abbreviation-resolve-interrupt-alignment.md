# Task: Align abbreviation resolve interrupt across backend and frontend

Status: Done
Owner: OpenCode (gpt-5.3-codex)
Branch or worktree: /home/hskim/work/llm-agent-v2
Created: 2026-03-23

## Goal

Implement end-to-end support for `abbreviation_resolve` interrupt/resume so abbreviation disambiguation decisions are captured in frontend UI and validated by backend before proceeding.

## Why

Current backend emits `abbreviation_resolve` interrupts, but frontend does not have a dedicated pending state/panel/resume payload path. This causes payload-shape mismatch and allows invalid resume input to advance incorrectly.

## Contracts To Preserve

- C-API-002
- C-UI-001
- C-UI-002

## Contracts To Update

- None

## Allowed Files

- `frontend/src/features/chat/hooks/use-chat-session.ts`
- `frontend/src/features/chat/pages/chat-page.tsx`
- `frontend/src/features/chat/components/index.ts`
- `frontend/src/features/chat/components/abbreviation-resolve-panel.tsx`
- `frontend/src/features/chat/__tests__/issue-flow-ui.test.tsx`
- `frontend/src/features/chat/__tests__/chat-page-device-panel.test.tsx`
- `frontend/src/features/chat/__tests__/helpers/render-chat-page.tsx`
- `backend/llm_infrastructure/llm/langgraph_agent.py`
- `backend/tests/test_abbreviation_resolve_node.py`
- `tests/api/test_agent_interrupt_resume_regression.py`
- `docs/tasks/TASK-20260323-abbreviation-resolve-interrupt-alignment.md`

## Out Of Scope

- Query expansion ranking algorithm changes
- New API endpoint introduction
- Broad UI redesign beyond abbreviation decision panel

## Risks

- Interrupt kind routing regression in chat hook
- Incorrect resume payload schema for abbreviation decisions
- Backend allowing malformed abbreviation selections
- Existing issue/device interruption UX regressions

## Verification Plan

```bash
cd frontend && npm run test -- src/features/chat/__tests__/issue-flow-ui.test.tsx
cd frontend && npm run test -- src/features/chat/__tests__/chat-page-device-panel.test.tsx
cd frontend && npm run build
cd backend && uv run pytest tests/test_abbreviation_resolve_node.py -v
uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v
uv run pytest tests/api/test_agent_response_metadata_contract.py -v
```

## Verification Results

- command: `cd frontend && npm run test -- src/features/chat/__tests__/issue-flow-ui.test.tsx`
  - result: pass (`8 passed`)
  - note: abbreviation interrupt resume payload path + existing issue-flow regressions verified.
- command: `cd frontend && npm run test -- src/features/chat/__tests__/chat-page-device-panel.test.tsx`
  - result: pass (`17 passed`)
  - note: existing device/issue panel regressions remain stable and abbreviation panel render/disable + submit interaction behavior verified (numbered button UI restyle reflected).
- command: `cd frontend && npm run build`
  - result: pass
  - note: production bundle build succeeded; pre-existing Vite chunk size warning unchanged.
- command: `cd backend && uv run pytest tests/test_abbreviation_resolve_node.py -v`
  - result: pass (`2 passed`)
  - note: malformed abbreviation decision re-prompts and non-ambiguous fast-path verified.
- command: `uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v`
  - result: pass (`5 passed`)
  - note: C-API-002 interrupt/resume continuity behavior preserved, including explicit abbreviation resume routing to HIL.
- command: `uv run pytest tests/api/test_agent_response_metadata_contract.py -v`
  - result: pass (`2 passed`)
  - note: C-API-001 metadata contract remains intact.

## Handoff

- Current status: done
- Last passing verification command and result
  - `cd backend && uv run pytest tests/test_abbreviation_resolve_node.py -v` -> pass
- Remaining TODOs (priority order)
  1. none
- Whether `Allowed Files` changed and why
  - initial scope
- Whether `Contracts To Update` is expected
  - no

## Change Log

- 2026-03-23: task created
- 2026-03-23: frontend `abbreviation_resolve` pending state + submission handler + panel wiring added.
- 2026-03-23: backend abbreviation decision guard changed to re-interrupt on malformed/empty decision payloads.
- 2026-03-23: regression tests added for abbreviation node and frontend resume payload path; full verification plan passed.
- 2026-03-23: backend selection validation refined to use interrupt option IDs instead of direct private expander internals (`_concepts`) for better compatibility.
- 2026-03-23: added frontend abbreviation panel submit interaction test and API regression test for abbreviation resume HIL routing.
- 2026-03-23: aligned abbreviation interrupt assistant prompt to prefer backend-provided instruction text for UX consistency.
- 2026-03-23: reopened task to restyle abbreviation resolve panel for UI consistency with existing numbered chat selection screens.
- 2026-03-23: abbreviation panel restyled to numbered button choices with stronger contrast and unified chat selection visual language.

## Final Check

- [x] Diff stayed inside allowed files, or this doc was updated first
- [x] Protected contract IDs were re-checked
- [x] Verification commands were run, or blockers were recorded
- [x] Any contract changes were reflected in `product-contract.md`
- [x] Remaining risks and follow-ups were documented
