# Unified Start-Work Entrypoint (All Remaining Tasks)

## Intent
- Use this as the single `/start-work` target to execute all remaining plan work in one run order.

## Single Task
- [ ] Execute all remaining plan scopes end-to-end in the sequence below, carrying forward shared outputs/evidence and marking completion only when each scope's verification criteria passes.

## Execution Order (locked)
1. `ui-autoparse-confirm-task-mode-2026-03-05.md`
2. `ui-chat-improvements-v2.md`
3. `chunk_v3_embed_ingest_plan.md`
4. `chapter-grouping-retrieval.md`
5. `agent-retrieval-followups-2026-03-04.md`
6. `before-after-regression-compare.md`
7. `paper-a-scope-implementation.md`
8. `paper-a-supervisor-review-plan.md`

## Source Archive
- Original split plans are moved under `.sisyphus/plans/.legacy/` and treated as source specifications.
- During execution, read each source file from `.legacy/` at the matching filename and apply it under this unified run.

## Non-Negotiable Rules
- Keep one active workstream at a time from the locked order.
- Reuse prior outputs instead of re-running completed validations unless dependencies changed.
- Write evidence in each original plan's expected evidence path.
- If a downstream task depends on upstream schema/API contracts, update downstream tasks immediately in the same run.

## Definition of Done
- All 8 source scopes report done with their own test/build/verification gates passing.
- No pending checkboxes remain in this file.
