# 2026-03-06 Plan Cleanup Report

## Scope
- Clean up completed plans and duplicate/overlapping plans across `.sisyphus/plans` and `.omc/plans`.

## Completed Plans (Execution Complete)
- `.sisyphus/plans/ui-autoparse-confirm-task-mode-2026-03-05.md`
  - Task checkboxes complete (`1..12`, `F1..F4` all checked).
- `.sisyphus/plans/agent-retrieval-stability-hardening.md`
  - Task checkboxes complete (`1..10`, `F1..F4` all checked).

## In-Progress Plans (Not Complete)
- `.sisyphus/plans/agent-retrieval-followups-2026-03-04.md`
  - Remaining: `14`, `F1..F4`.
- `.sisyphus/plans/before-after-regression-compare.md`
  - Remaining: `7`, `8`, `9`, `F1..F4`.

## Duplicate/Overlap Cleanup

### Exact Duplicate Content
- Duplicate pair (byte-identical, SHA256 short `5156b44c4eed`):
  - `.sisyphus/plans/paper-a-supervisor-review-plan.md`
  - `.omc/plans/paper-a-scope-implementation.md`

Decision:
- Canonical: `.sisyphus/plans/paper-a-supervisor-review-plan.md`
- Treat `.omc/plans/paper-a-scope-implementation.md` as duplicate reference copy.
- Archived full duplicate snapshot:
  - `.omc/plans/archive/paper-a-scope-implementation.duplicate-2026-03-06.md`
- Kept `.omc/plans/paper-a-scope-implementation.md` as a stub that points to canonical.

### Same Topic, Different Purpose (Not Exact Duplicate)
- `chunk_v3` plan pair:
  - `.sisyphus/plans/chunk_v3_embed_ingest_plan.md` (master implementation plan)
  - `.omc/plans/chunk_v3_embed_ingest_plan.md` (current CLI runbook)

Decision:
- Keep both, but use role split:
  - Planning/spec: `.sisyphus/plans/chunk_v3_embed_ingest_plan.md`
  - Execution runbook: `.omc/plans/chunk_v3_embed_ingest_plan.md`

## Unified Tracking Status
- `.sisyphus/plans/unified-todo.md` is the active merged tracker.
- Current summary in unified tracker:
  - UI guided selection: `0 open / 13 done`
  - Retrieval followups: `2 open / 13 done`
  - Retrieval stability hardening: `0 open / 11 done`
  - Before/after regression compare: `4 open / 6 done`

## Operational Note
- Remaining RF/RC evaluation tasks require live LLM backend availability.
- Resume order after LLM recovery: `RF-14 -> RF-Final -> RC-7/8/9 -> RC-Final`.
