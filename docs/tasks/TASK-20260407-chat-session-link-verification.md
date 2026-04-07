# Task: verify chat session link flow

Status: In Progress
Owner: Hephaestus
Branch or worktree: dev
Created: 2026-04-07

## Goal

Verify the completed chat `session` deep-link flow and feedback resolution links,
then close the gaps needed for safe sign-off. The focus is on preserving chat
revisit/back-navigation and making dissatisfied feedback items reviewable later.

## Why

The implementation is already in the tree, but it currently lacks dedicated
verification for the new URL/session behavior and feedback resolution link flow.
This task adds the missing evidence and any minimal fixes discovered during review.

## Contracts To Preserve

- C-UI-001
- C-UI-002

## Contracts To Update

- None

## Allowed Files

- `docs/tasks/TASK-20260407-chat-session-link-verification.md`
- `frontend/src/features/chat/pages/chat-page.tsx`
- `frontend/src/features/chat/api.ts`
- `frontend/src/features/chat/types.ts`
- `frontend/src/features/chat/__tests__/**`
- `frontend/src/features/feedback/pages/feedback-page.tsx`
- `frontend/src/features/feedback/**/*.test.tsx`
- `backend/api/routers/feedback.py`
- `backend/services/feedback_service.py`
- `backend/llm_infrastructure/elasticsearch/mappings.py`
- `backend/tests/**/*feedback*.py`

## Out Of Scope

- No unrelated router/import cleanup
- No repo-wide TypeScript config repair
- No unrelated backend startup fixes outside this feature

## Risks

- Chat URL sync could break existing chat-page behavior or clear unrelated query params
- Feedback resolution links could be stored/rendered inconsistently
- Missing tests could allow regressions in revisit/back-navigation flow

## Verification Plan

```bash
cd frontend && npm run test -- src/features/chat/__tests__/chat-page-device-panel.test.tsx
cd frontend && npm run test -- src/features/chat/__tests__/chat-request-payload.test.tsx
cd frontend && npm run build
cd frontend && npm run test -- <new chat/feedback tests>
uv run pytest backend/tests -k feedback -v
uv run pytest tests/api -k feedback -v
```

## Verification Results

- command: `cd frontend && npm run test -- src/features/chat/__tests__/chat-page-device-panel.test.tsx`
  - result: pass
  - note: existing chat-page regression suite still passes after session URL sync change
- command: `cd frontend && npm run test -- src/features/chat/__tests__/chat-request-payload.test.tsx`
  - result: pass
  - note: existing chat request payload behavior still passes
- command: `cd frontend && npm run build`
  - result: pass
  - note: production frontend bundle builds successfully
- command: `cd frontend && npx tsc --noEmit`
  - result: fail
  - note: repo-level TypeScript environment already fails in dependencies/test typing unrelated to this feature
  - evidence: `node_modules/@ant-design/... TS1259`, `src/features/chat/__tests__/helpers/render-chat-page.tsx`, `src/main.tsx`
- command: `uv run pytest backend/tests -k feedback -v`
  - result: skip
  - note: no backend tests matched `feedback`
- command: `uv run pytest tests/api -k feedback -v`
  - result: fail
  - note: API test environment is blocked by unrelated missing router import
  - evidence: `ImportError: cannot import name 'slang_dict' from 'backend.api.routers'`
- command: `cd frontend && npm run test -- src/features/chat/__tests__/chat-page-session-link.test.tsx src/features/feedback/pages/feedback-page.test.tsx src/features/chat/__tests__/chat-page-device-panel.test.tsx src/features/chat/__tests__/chat-request-payload.test.tsx`
  - result: pass
  - note: session URL sync, original chat link exposure, safe resolved-link rendering, and existing chat regressions all pass together (35 tests)
- command: `cd frontend && npm run build`
  - result: pass
  - note: frontend still builds after adding original chat links, safe href handling, and new tests
- command: `Playwright route QA via MCP on http://127.0.0.1:4173`
  - result: skip
  - note: blocked by missing local Chrome binary required by Playwright MCP in this environment
  - evidence: `Chromium distribution 'chrome' is not found at /opt/google/chrome/chrome`

## Handoff

- Current status: done
- Last passing verification command and result: `cd frontend && npm run build` → pass
- Remaining TODOs (priority order): complete browser QA once Playwright Chrome is installed; fix unrelated API test env (`slang_dict`) before backend-level end-to-end verification
- Whether `Allowed Files` changed and why: no
- Whether `Contracts To Update` is expected: no

## Change Log

- 2026-04-07: task created

## Final Check

- [x] Diff stayed inside allowed files, or this doc was updated first
- [x] Protected contract IDs were re-checked
- [x] Verification commands were run, or blockers were recorded
- [ ] Any contract changes were reflected in `product-contract.md`
- [x] Remaining risks and follow-ups were documented
