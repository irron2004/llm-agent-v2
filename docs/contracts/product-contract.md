# Product Contract

Purpose: protect user-visible behavior and high-risk integration points from accidental rollback when multiple agents work in parallel.

How to use this document:
- Read this file before substantial edits to backend agent flow, API response shapes, or chat UI behavior.
- Treat each contract item as protected behavior. Do not change it silently.
- If a task must change a contract item, update this file, update or add linked tests, and record the change in the active task document.
- If a behavior is important but not listed here yet, add it before refactoring it.

Change rule:
- Code change only: allowed when all linked contract items still hold.
- Contract change: requires explicit note in the task document under "Contracts to update" and matching test updates.

## Protected Contracts

| ID | Layer | Protected behavior | Verification source |
| --- | --- | --- | --- |
| C-API-001 | API | `POST /api/agent/run` and `POST /api/agent/run/stream` must keep the response metadata contract used by clients. Required metadata keys include `mq_mode`, `mq_used`, `mq_reason`, `route`, `st_gate`, `attempts`, `max_attempts`, `retry_strategy`, `guardrail_dropped_numeric`, `guardrail_dropped_anchor`, `guardrail_final_count`, `search_queries_final`, `search_queries_raw`, and `index_name`. | `uv run pytest tests/api/test_agent_response_metadata_contract.py -v` |
| C-API-002 | API | Interrupt/resume must preserve `thread_id` continuity and resume through a fresh agent instance using the shared checkpointer path rather than relying on mutated in-memory state from the first request. | `uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v` |
| C-API-003 | API | When `retrieval_only=true`, the agent must stop before answer generation, return an interrupt payload of type `retrieval_review`, expose retrieved documents, and set `metadata.response_mode` to `retrieval_only`. | `uv run pytest tests/api/test_agent_retrieval_only.py -v` |
| C-UI-001 | Frontend | The missing-device regeneration prompt must not be shown when the linked assistant message already contains a completed answer. | `cd frontend && npm run test -- src/features/chat/__tests__/chat-page-device-panel.test.tsx -t "does not show missing-device prompt when the linked assistant message already has an answer"` |
| C-UI-002 | Frontend | Pending regeneration device flow must keep the current command semantics: `1` loads the device list, `2` dismisses the panel, `0` cancels without sending, and selecting a device sends `[DeviceName] {originalQuery}` with the expected overrides payload. | `cd frontend && npm run test -- src/features/chat/__tests__/chat-page-device-panel.test.tsx` |

## Review Checklist

Before editing:
- Identify which contract IDs are in scope.
- Check whether the task preserves the contract or intentionally changes it.
- If intentionally changing it, document the reason in the active task file before editing code.

Before finishing:
- Re-read every touched contract ID.
- Run the linked verification commands or explain why they could not be run.
- Confirm that any new API field removals, prompt flow changes, interrupt semantics changes, and chat UI branching changes are reflected here.

## Candidate Areas To Add Next

These are important but not yet formalized enough in this contract file:
- Device selection interrupt/resume payload shape beyond the current regression coverage.
- Retrieval result ordering and expanded-doc behavior for canonical retrieval mode.
- Feedback and issue-flow chat UI behavior.
