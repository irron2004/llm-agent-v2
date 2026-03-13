# Task: multi-agent change safety protocol

Status: Done
Owner: Codex
Branch or worktree: `86ewk6385-1차-PE-피드백-v2`
Created: 2026-03-13

## Goal

Introduce a lightweight repository-level process to reduce accidental rollback when multiple agents work in parallel.
Add a product contract document, a reusable task template, and explicit AGENTS rules that force contract-aware edits.

## Why

Parallel agent work in a shared workspace can silently overwrite unfinished work or revert stable behavior.
The repo needed a single place to list protected behavior and a task-scoped checklist to keep edits inside an explicit boundary.

## Contracts To Preserve

- None

This task adds process documents and agent instructions. It does not intentionally change runtime behavior.

## Contracts To Update

- None

## Allowed Files

- `AGENTS.md`
- `docs/contracts/product-contract.md`
- `docs/tasks/README.md`
- `docs/tasks/TASK_TEMPLATE.md`
- `docs/tasks/TASK-20260313-multi-agent-change-safety.md`

## Out Of Scope

- No backend or frontend runtime behavior changes
- No new automation script or CI hook
- No rewrite of existing tests

## Risks

- Contract file could become stale if future tasks do not maintain it
- Operators may still launch multiple agents in the same worktree unless they follow the new protocol

## Verification Plan

Manual verification for this documentation-only change:

- Read all new docs and verify internal links and paths
- Confirm AGENTS instructions reference the new contract and task docs
- Confirm the contract items point to existing test files

## Change Log

- 2026-03-13: created task doc
- 2026-03-13: added product contract, task docs guide, and task template
- 2026-03-13: updated `AGENTS.md` with change safety protocol and contract-aware checklist

## Final Check

- [x] Diff stayed inside allowed files, or this doc was updated first
- [x] Protected contract IDs were re-checked
- [x] Verification commands were run, or blockers were recorded
- [x] Any contract changes were reflected in `product-contract.md`
- [x] Remaining risks and follow-ups were documented
