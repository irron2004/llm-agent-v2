# Start-Work Remaining (Prioritized, B4 Removed)

## Intent
- This file is the execution plan for the current cycle.
- It reflects user-priority updates on 2026-03-09.
- It is aligned with `.sisyphus/plans/start-work-all-remaining.md`.

## Scope Decisions (Locked)
- B4 (HTTP before/after quality gate pipeline) is removed from this cycle.
- C3 is treated as verify-first because rollout work already progressed.
- Category 3 work (agent retrieval Stage2/MQ follow-ups) is still required, but must be re-planned after C3 verification.
- Category 4 work (guided-selection REQ-2 robustness) is the highest priority.
- Category 7 bundle (cross-plan F* governance/audit tasks) is removed from this cycle.
- Category 5/6 streams (Paper A/Paper B) are deferred to a later separate request.

## Execution Order
- P0 (highest): Guided-selection REQ-2 robustness hardening.
- P1: C3 verification and evidence refresh.
- P2: Agent retrieval Stage2/MQ follow-up replan and required subset execution.

## TODOs

- [ ] 1. P0-1 Lock REQ-2 implementation contract and current-gap ledger

  **What to do**:
  1) Confirm current gaps still open from `ui-chat-improvements-v2` REQ-2:
     - common parse fallback utility
     - judge JSON mode enforcement
     - retry ceiling
     - translate/query_rewrite output normalization
     - prompt hardening
  2) Write a compact execution contract at:
     - `.sisyphus/evidence/start-work-remaining/p0/req2-gap-lock.md`
  3) Freeze this contract as the P0 source for tasks 2-3.

  **References**:
  - `.sisyphus/plans/.legacy/ui-chat-improvements-v2.md`
  - `.sisyphus/plans/.legacy/ui-autoparse-confirm-task-mode-2026-03-05.md`

  **Acceptance Criteria**:
  - [ ] Gap lock note exists at `.sisyphus/evidence/start-work-remaining/p0/req2-gap-lock.md`.
  - [ ] All five REQ-2 sub-gaps are explicitly listed as in-scope or already-resolved.

- [ ] 2. P0-2 Implement REQ-2 robustness hardening (highest priority)

  **What to do**:
  1) Introduce one shared parse-fallback utility and apply to:
     - `_parse_auto_parse_result`
     - `_parse_queries`
     - `_parse_route`
     - `_parse_needs_history_from_text`
  2) Enforce judge JSON mode where client path supports it (`response_format=json_object`).
  3) Add retry ceiling for parse failures with safe fallback defaults.
  4) Normalize translate/query_rewrite outputs (remove numbered/explanatory noise).
  5) Harden prompts to JSON-only in:
     - `backend/llm_infrastructure/llm/prompts/auto_parse_v1.yaml`
     - `backend/llm_infrastructure/llm/prompts/router_v1.yaml`

  **Target files**:
  - `backend/llm_infrastructure/llm/langgraph_agent.py`
  - `backend/llm_infrastructure/llm/prompts/auto_parse_v1.yaml`
  - `backend/llm_infrastructure/llm/prompts/router_v1.yaml`

  **Acceptance Criteria**:
  - [ ] REQ-2 code paths use shared fallback behavior.
  - [ ] Retry loop is bounded and cannot spin indefinitely.
  - [ ] Judge/parser behavior remains stable for GLM-like non-compliant outputs.

- [ ] 3. P0-3 Validate REQ-2 with tests and focused QA evidence

  **What to do**:
  1) Run backend tests covering guided flow and parser/judge behavior.
  2) Run frontend guided-selection tests.
  3) Store outputs under `.sisyphus/evidence/start-work-remaining/p0/`.

  **References**:
  - `tests/api/test_agent_autoparse_confirm_interrupt_resume.py`
  - `tests/api/test_agent_interrupt_resume_regression.py`
  - `frontend/src/features/chat/__tests__/guided-selection-panel.test.tsx`

  **Acceptance Criteria**:
  - [ ] Backend target tests pass.
  - [ ] Frontend guided-selection tests pass.
  - [ ] Evidence logs are written under `.sisyphus/evidence/start-work-remaining/p0/`.

- [ ] 4. C3-1 Re-verify chunk_v3 section rollout artifacts and index sync

  **What to do**:
  1) Verify latest chunk JSONL contains `section_chapter`, `section_number`, `chapter_source`, `chapter_ok`.
  2) Verify `chunk_v3_content` mapping still has section fields with correct types.
  3) Run sync verification for three models (`qwen3_emb_4b`, `bge_m3`, `jina_v5`).

  **References**:
  - `scripts/chunk_v3/run_chunking.py`
  - `scripts/chunk_v3/run_ingest.py`
  - `backend/llm_infrastructure/elasticsearch/mappings.py`

  **Acceptance Criteria**:
  - [ ] Section fields are present in sampled JSONL rows.
  - [ ] Mapping contract for section fields is valid.
  - [ ] Verify outputs for all 3 models report sync OK.

- [ ] 5. C3-2 Re-verify retrieval section-expansion behavior on current index state

  **What to do**:
  1) Re-run section-expansion test suite and smoke retrieval checks.
  2) Confirm expansion is gated by allowed `chapter_source` and respects caps.
  3) Capture updated C3 evidence under `.sisyphus/evidence/start-work-remaining/c3/`.

  **References**:
  - `backend/tests/test_section_expansion.py`
  - `backend/llm_infrastructure/retrieval/engines/es_search.py`
  - `backend/llm_infrastructure/llm/langgraph_agent.py`

  **Acceptance Criteria**:
  - [ ] Section-expansion tests pass.
  - [ ] Smoke checks show grouped section retrieval still works on current indices.
  - [ ] Evidence is refreshed under `.sisyphus/evidence/start-work-remaining/c3/`.

- [ ] 6. R3-1 Re-plan Stage2/MQ follow-ups with C3-aware constraints

  **What to do**:
  1) Re-scope remaining tasks from `agent-retrieval-followups` after C3 verification.
  2) Ensure the replan uses current assumptions:
     - canonical doc_type semantics
     - section-grouping metadata availability
     - no B4/before-after dependency
     - known C3 blockers (must be handled explicitly in E2 scope):
       - embed indices are 87 docs behind `chunk_v3_content` for all 3 models (vector parity not satisfied)
       - live section-expansion can fail when ES returns `_score=None` for sorted/filtered section fetch queries
  3) Write a short replan note under:
     - `.sisyphus/evidence/start-work-remaining/r3/replan.md`

  **References**:
  - `.sisyphus/plans/.legacy/agent-retrieval-followups-2026-03-04.md`
  - `.sisyphus/plans/.legacy/chapter-grouping-retrieval.md`

  **Acceptance Criteria**:
  - [ ] Replan note exists and explicitly lists included vs excluded tasks.
  - [ ] Replan excludes B4 and before/after gate tasks.

- [ ] 7. R3-2 Execute required Stage2/MQ core subset (without B4)

  **What to do**:
  1) Implement remaining core retrieval follow-ups:
     - Stage2 retrieval + early-page penalty in `retrieve_node`
     - Stage2 integration tests
     - sticky inheritance policy freeze + tests
     - eval JSONL schema normalization + validator sync
  2) Keep implementation strictly decoupled from before/after HTTP gate work.

  **Target files (expected)**:
  - `backend/llm_infrastructure/llm/langgraph_agent.py`
  - `tests/api/test_agent_stage2_retrieval.py`
  - `tests/api/test_agent_sticky_policy_followup_only.py`
  - `scripts/evaluation/validate_agent_eval_jsonl.py`

  **Acceptance Criteria**:
  - [ ] Stage2 core behavior and tests pass.
  - [ ] Sticky policy behavior is locked and tested.
  - [ ] Eval JSONL validator enforces canonical keys.

- [ ] 8. R3-3 Execute evaluator/doc sync subset (without B4)

  **What to do**:
  1) Add MQ mode sweep automation (`off|fallback|on`) and ablation output.
  2) Update retrieval follow-up docs to match final non-B4 scope.
  3) Write evidence to `.sisyphus/evidence/start-work-remaining/r3/`.

  **Acceptance Criteria**:
  - [ ] MQ sweep runs and emits per-mode outputs + ablation summary.
  - [ ] Docs reflect current Stage2/MQ scope and explicitly exclude B4.

## Removed / Deferred From This Cycle
- Removed now: all B4 tasks (HTTP gate, before/after raw generation, compare report).
- Removed now: cross-plan F1-F4 audit bundles.
- Deferred to future separate request: Paper A stream and Paper B stream tasks.

## Definition of Done (This Cycle)
- P0 tasks (1-3) complete.
- C3 verification tasks (4-5) complete with refreshed evidence.
- R3 tasks (6-8) complete and decoupled from B4.
- No B4, F*, Paper A, or Paper B tasks executed in this cycle.
