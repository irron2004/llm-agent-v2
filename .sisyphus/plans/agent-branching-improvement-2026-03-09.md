# Agent Branching + Issue Multi-Step Flow (2026-03-09)

## TL;DR
> **Summary**: Make guided selection `task_mode` drive routing + answer/judge templates, then add an issue-specific multi-step flow (list cases → select → detail → SOP link) with safe interrupt/resume contracts.
> **Deliverables**:
> - `task_mode`-first routing (skip router LLM when task_mode set)
> - Issue-specific prompts (answer + judge) + v2 language prompt parity (en/zh/ja)
> - Issue Step1/2/3 flow implemented via LangGraph interrupts/resume + frontend panels
> - Regression + new flow tests (backend + frontend)
> - ~~Repetition prevention (committed: `a0c9c59`)~~
> - Frontend bug fixes: device catalog error, skip recommend tag, re-search button restore
> **Effort**: Large
> **Parallel**: YES - 3 waves
> **Critical Path**: Interrupt/resume contract (BE+FE) → backend graph nodes/edges → frontend panels → tests

## Context
### Original Request
- Implement the design in `docs/2026-03-09-agent-분기-개선.md` (Phase A+B).

### Interview Summary
- User selected: Plan Phase A+B.
- Source-of-truth decisions come from `docs/2026-03-09-agent-분기-개선.md`.

### Repo Grounding (facts)
- Guided confirm sets `task_mode` + doc-type scope in `backend/llm_infrastructure/llm/langgraph_agent.py:3369` (`auto_parse_confirm_node`).
- Routing and templates are currently route-only (task_mode not consulted yet):
  - `route_node`: `backend/llm_infrastructure/llm/langgraph_agent.py:1046`
  - `answer_node`: `backend/llm_infrastructure/llm/langgraph_agent.py:2134`
  - `judge_node`: `backend/llm_infrastructure/llm/langgraph_agent.py:2224`
- PromptSpec includes issue prompt fields, but `load_prompt_spec()` currently gates issue prompt loading to `version == "v2"`: `backend/llm_infrastructure/llm/langgraph_agent.py:171`, `backend/llm_infrastructure/llm/langgraph_agent.py:250`
  - Plan decision: remove the v2-only gating (try-load issue prompts for v1 too) so issue flow works under default `prompt_spec_version=v1`.
- Graph assembly: `backend/services/agents/langgraph_rag_agent.py:458` (edge `expand_related → answer` at `backend/services/agents/langgraph_rag_agent.py:631`).
- Interrupt/resume transport + guided resume checkpoint guard exists in `backend/api/routers/agent.py` (`_validate_guided_resume_checkpoint`), and FE already treats issue interrupt types as guided resumes in `frontend/src/features/chat/hooks/use-chat-session.ts:584`.
- Frontend interrupt handling + guided panel:
  - `frontend/src/features/chat/hooks/use-chat-session.ts`
  - `frontend/src/features/chat/components/guided-selection-panel.tsx`
  - `frontend/src/features/chat/pages/chat-page.tsx`

### Metis Review (gaps addressed)
- New issue-step interrupts must NOT be treated as generic HIL resume; otherwise FE sends `ask_user_after_retrieve=true` and BE resumes the wrong graph family.
- Add nonce + validation per interrupt type; add `graph_version` guard for resume.
- Add regression tests to ensure legacy (no task_mode) behavior is unchanged.
- Remove `load_prompt_spec()` v2-only gating for issue prompts (default `prompt_spec_version=v1` must still support issue flow).
- SOP-id tokens extracted from issue detail must be treated as query hints (not assumed to equal Elasticsearch `doc_id`).

## Work Objectives
### Core Objective
- Guided confirm `task_mode` must control routing + prompting, and `task_mode="issue"` must support a structured, interactive issue workflow.

### Deliverables
- Phase A:
  - `route_node` uses task_mode map when task_mode exists; LLM router only when task_mode is unset.
  - Issue answer prompt(s) + issue judge prompt.
  - `answer_node` + `judge_node` use task_mode-first selection.
  - v2 language prompt file parity for answer prompts (setup/ts/general/issue: ko/en/zh/ja).
- Phase B:
  - Issue Step1 list cases + interrupt selection.
  - Issue Step2 show selected case detail + interrupt SOP confirm.
  - Issue Step3 fetch + answer SOP content.
  - Safe interrupt/resume contract: type + nonce + Pydantic validation + graph_version guard.

### Definition of Done (agent-verifiable)
- Backend tests pass:
  - `uv run pytest -v tests/api/test_agent_autoparse_confirm_interrupt_resume.py`
  - `uv run pytest -v backend/tests/test_answer_language_templates.py`
  - `uv run pytest -v tests/api/test_agent_issue_flow_interrupt_resume.py` (expanded in this work)
- Frontend tests pass:
  - `cd frontend && npm test`
- Manual QA (agent-executed) verifies:
  - Guided confirm → issue mode → list cases → select → detail → SOP answer.

### Must Have
- Resume safety: issue-step resumes must run in the SAME graph family that created the checkpoint.
- Stable selection ID: resume must use `doc_id` (not index-only) + nonce.
- Legacy behavior unchanged when task_mode is absent.

### Must NOT Have
- No new Route literal values (keep `Route = Literal["setup","ts","general"]`).
- No breaking changes to `/api/agent/run` request schema beyond additive interrupt resume decision types.
- No dependence on manual-only verification.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: tests-after (Pytest + Vitest).
- Evidence: `.sisyphus/evidence/` artifacts per task.

## Execution Strategy
### Parallel Execution Waves
Wave 1 (contracts + prompts): mostly landed; complete Tasks 2-3 (issue prompt loading policy + resume guard/idempotency tests) + confirm baseline tests
Wave 2 (backend flow): task_mode routing + issue prompts wiring + issue step graph nodes
Wave 3 (frontend + tests): FE panels + resume wiring + end-to-end tests + QA

### Dependency Matrix (high level)
- Contract + prompt availability blocks backend flow and frontend UI.
- Backend flow blocks frontend integration tests.

## TODOs
> Implementation + Test = ONE task.
> EVERY task includes QA scenarios and evidence output.

 - [x] 1. Add Issue + v2 Language Prompt Files (Backend)

  **What to do**:
  - Status: Already present in repo (as of 2026-03-10). Only update if prompt requirements change.
  - Add issue prompt YAMLs for BOTH `v1` and `v2` under `backend/llm_infrastructure/llm/prompts/` (repo default is `prompt_spec_version=v1`; runtime may override via env):
    - Issue list:
      - `backend/llm_infrastructure/llm/prompts/issue_ans_v1.yaml`
      - `backend/llm_infrastructure/llm/prompts/issue_ans_{en,zh,ja}_v1.yaml`
      - `backend/llm_infrastructure/llm/prompts/issue_ans_v2.yaml`
      - `backend/llm_infrastructure/llm/prompts/issue_ans_{en,zh,ja}_v2.yaml`
    - Issue detail:
      - `backend/llm_infrastructure/llm/prompts/issue_detail_ans_v1.yaml`
      - `backend/llm_infrastructure/llm/prompts/issue_detail_ans_{en,zh,ja}_v1.yaml`
      - `backend/llm_infrastructure/llm/prompts/issue_detail_ans_v2.yaml`
      - `backend/llm_infrastructure/llm/prompts/issue_detail_ans_{en,zh,ja}_v2.yaml`
    - v2 language parity for existing routes:
      - `backend/llm_infrastructure/llm/prompts/setup_ans_{en,zh,ja}_v2.yaml`
      - `backend/llm_infrastructure/llm/prompts/ts_ans_{en,zh,ja}_v2.yaml`
      - `backend/llm_infrastructure/llm/prompts/general_ans_{en,zh,ja}_v2.yaml`
  - Prompt requirements (decision-complete):
    - All *issue list* prompts must produce a numbered case list and MUST cite sources with `[N]` and a references section.
    - All *issue detail* prompts must output sections: Issue / Cause / Action / Status, and include 3-8 bullet excerpts from REFS.
    - For en/zh/ja prompts: MUST respond in that language; do not rely on dynamic language instructions.

  **Must NOT do**:
  - Do not rename existing v1 prompt files; only add v2 variants.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: prompt authoring + consistency across languages
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 4,5 | Blocked By: none

  **References**:
  - Loader naming convention: `backend/llm_infrastructure/llm/prompt_loader.py:41`
  - Existing v2 prompts to mirror: `backend/llm_infrastructure/llm/prompts/general_ans_v2.yaml`, `backend/llm_infrastructure/llm/prompts/setup_ans_v2.yaml`, `backend/llm_infrastructure/llm/prompts/ts_ans_v2.yaml`
  - Existing v1 language prompts to mirror: `backend/llm_infrastructure/llm/prompts/general_ans_en_v1.yaml`, `backend/llm_infrastructure/llm/prompts/setup_ans_zh_v1.yaml`
  - prompt spec version default: `backend/config/settings.py:55`

  **Acceptance Criteria**:
  - [ ] `uv run pytest -v backend/tests/test_answer_language_templates.py` passes after updating tests for v2 (Task 2)

  **QA Scenarios**:
  ```
  Scenario: Prompt files exist and load
    Tool: Bash
    Steps: uv run python -c "from backend.llm_infrastructure.llm.langgraph_agent import load_prompt_spec; load_prompt_spec('v2')"
    Expected: Exit code 0
    Evidence: .sisyphus/evidence/task-1-prompts-load.txt

  Scenario: Missing prompt fails loudly (negative)
    Tool: Bash
    Steps: uv run python -c "from backend.llm_infrastructure.llm.prompt_loader import load_prompt_template; load_prompt_template('issue_ans','v999')"
    Expected: Non-zero exit + FileNotFoundError mentioning available prompts
    Evidence: .sisyphus/evidence/task-1-prompts-missing.txt
  ```

  **Commit**: YES | Message: `feat(agent): add issue prompts and v2 language variants` | Files: `backend/llm_infrastructure/llm/prompts/issue*_v{1,2}.yaml`, `backend/llm_infrastructure/llm/prompts/*_ans_{en,zh,ja}_v2.yaml`

- [x] 2. Extend PromptSpec + v2 Language Template Coverage Tests (and remove v2-only gating for issue prompts)

  **What to do**:
  - Status: PromptSpec fields exist, but `load_prompt_spec()` currently gates issue prompts to `version == "v2"` only; adjust to avoid runtime/env ambiguity for issue flow.
  - Extend `PromptSpec` in `backend/llm_infrastructure/llm/langgraph_agent.py:171`:
    - Add optional fields:
      - `issue_ans`, `issue_ans_en`, `issue_ans_zh`, `issue_ans_ja`
      - `issue_detail_ans`, `issue_detail_ans_en`, `issue_detail_ans_zh`, `issue_detail_ans_ja`
      - `judge_issue_sys` (string; follow existing `judge_*_sys` pattern)
  - Extend `load_prompt_spec()` in `backend/llm_infrastructure/llm/langgraph_agent.py:250`:
    - `_try_load_prompt('issue_ans', version)` etc.
    - Keep all new prompt fields optional to avoid breaking older versions.
    - Policy change: try-load issue prompts for ANY version (v1/v2). Do not hard-gate issue prompt loading on v2.
  - Update and extend `backend/tests/test_answer_language_templates.py`:
    - Add a new test asserting `load_prompt_spec('v2')` includes `*_ans_{en,zh,ja}` for setup/ts/general.
    - Add tests asserting v2 spec includes issue prompts (non-None) when files exist.

  **Must NOT do**:
  - Do not change required PromptSpec fields; only add optional fields.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: contained edits + focused tests
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 4,5 | Blocked By: 1

  **References**:
  - PromptSpec + loader: `backend/llm_infrastructure/llm/langgraph_agent.py:171`, `backend/llm_infrastructure/llm/langgraph_agent.py:250`
  - Current v2-only gating to remove: `backend/llm_infrastructure/llm/langgraph_agent.py:278`
  - Existing language test patterns: `backend/tests/test_answer_language_templates.py:77`

  **Acceptance Criteria**:
  - [ ] `uv run pytest -v backend/tests/test_answer_language_templates.py` passes

  **QA Scenarios**:
  ```
  Scenario: v2 spec contains language variants
    Tool: Bash
    Steps: uv run pytest -v backend/tests/test_answer_language_templates.py
    Expected: All tests pass
    Evidence: .sisyphus/evidence/task-2-backend-promptspec-tests.txt

  Scenario: v1 remains compatible (issue prompts optional)
    Tool: Bash
    Steps: uv run python -c "from backend.llm_infrastructure.llm.langgraph_agent import load_prompt_spec; s=load_prompt_spec('v1'); assert getattr(s,'issue_ans',None) is not None"
    Expected: Exit code 0
    Evidence: .sisyphus/evidence/task-2-v1-compat.txt
  ```

  **Commit**: YES | Message: `feat(agent): extend PromptSpec for issue flow` | Files: `backend/llm_infrastructure/llm/langgraph_agent.py`, `backend/tests/test_answer_language_templates.py`

- [x] 3. Interrupt/Resume Contract for Issue Flow (BE + FE) with Graph Version Guard

  **What to do**:
  - Status: Base contract exists (decision models + graph_version/nonce guard + FE guided resume routing), but add single-use nonce semantics + pending-interrupt new-run guard + regression tests.
  - Backend (`backend/api/routers/agent.py`):
    - Add new resume decision models (Pydantic) analogous to `AutoParseConfirmDecision`:
      - `IssueCaseSelectionDecision`: `{type:"issue_case_selection", nonce:str, selected_doc_id:str}`
      - `IssueSopConfirmDecision`: `{type:"issue_sop_confirm", nonce:str, confirm:bool}`
    - Add helper `_is_guided_resume(req)` that returns true for these types and `auto_parse_confirm`.
    - Keep the existing dict-dispatch validation style (`_validated_guided_resume_decision`) for now; do not refactor to a discriminated-union schema unless it measurably improves error reporting.
    - On resume, BEFORE `agent._graph.invoke(...)`, load checkpoint state via `agent._graph.get_state(config)` and validate:
      - `graph_version` matches the current code constant (e.g., `AGENT_GRAPH_VERSION = "2026-03-09"`).
      - `pending_interrupt_nonce` in checkpoint equals decision.nonce.
    - New-run guard (non-resume requests): if `req.thread_id` is provided, `req.resume_decision` is null, and the checkpoint indicates an issue-step interrupt is pending (e.g., `pending_interrupt_nonce` is set and `issue_flow_step` is not null), reject with HTTP 409 and a deterministic recovery hint (client must resume or start a new thread).
    - Ensure resume chooses `_new_guided_confirm_agent` for guided resumes (including issue-step resumes).
      - Rationale/guardrail: guided resume checkpoints for this flow are created by the auto-parse-enabled graph; do not resume with a non-auto-parse graph variant (it may lack the same nodes/edges).
  - Backend nodes (state):
    - Ensure state includes and persists `graph_version` and `pending_interrupt_nonce` when an interrupt is raised.
  - Frontend (`frontend/src/features/chat/hooks/use-chat-session.ts`):
    - Extend interrupt kind recognition for `issue_case_selection` and `issue_sop_confirm`.
    - Treat them as guided resumes (NOT HIL): keep `ask_user_after_retrieve=false`.
    - Include `nonce` on resume_decision.
  - Tests:
    - Backend: add a test that resuming with wrong nonce returns HTTP 409 with deterministic message.
    - Backend: add a test that replaying the same resume_decision (same nonce) is rejected (HTTP 409) after successful apply (nonce is single-use).
    - Backend: add a test that sending a NEW non-resume request with an existing `thread_id` while a guided interrupt is pending is rejected (HTTP 409) with a recovery hint.
    - Frontend: add a unit test asserting issue-step resume payload does not set `ask_user_after_retrieve: true`.

  **Must NOT do**:
  - Do not change the existing guided_confirm interrupt type/schema (`auto_parse_confirm`).
  - Do not attempt to persist checkpoints across server restarts (MemorySaver stays in-memory).

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: cross-layer contract + edge-case-heavy resume logic
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5,6 | Blocked By: none

  **References**:
  - Resume selection logic: `backend/api/routers/agent.py:959`
  - Guided confirm decision validation: `backend/api/routers/agent.py:306`
  - FE resume routing: `frontend/src/features/chat/hooks/use-chat-session.ts:584`
  - Existing guided resume test: `frontend/src/features/chat/__tests__/chat-request-payload.test.tsx:89`

  **Acceptance Criteria**:
  - [ ] `uv run pytest -v tests/api/test_agent_issue_flow_interrupt_resume.py -k "wrong_nonce or replay or pending_interrupt or version"` passes
  - [ ] `cd frontend && npm test` passes

  **QA Scenarios**:
  ```
  Scenario: Resume with wrong nonce rejected
    Tool: Bash
    Steps: uv run pytest -v tests/api/test_agent_issue_flow_interrupt_resume.py -k wrong_nonce
    Expected: Test passes; API returns 409 with clear error string
    Evidence: .sisyphus/evidence/task-3-nonce-guard.txt

  Scenario: Frontend guided resume payload for issue step
    Tool: Bash
    Steps: cd frontend && npm test -- -t "issue-case-selection resume payload"
    Expected: Test passes; payload has ask_user_after_retrieve=false
    Evidence: .sisyphus/evidence/task-3-fe-payload.txt
  ```

  **Commit**: YES | Message: `feat(agent): add issue-flow resume contract` | Files: `backend/api/routers/agent.py`, `frontend/src/features/chat/hooks/use-chat-session.ts`, tests

- [x] 4. Task-Mode First Routing + Answer/Judge Selection (Phase A)

  **What to do**:
  - Backend (`backend/llm_infrastructure/llm/langgraph_agent.py`):
    - Update `route_node` (`:1046`) to:
      - If `state.task_mode` in {sop, issue, all}, set route using map `{sop:"setup", issue:"general", all:"general"}` and SKIP LLM router.
      - Else keep existing LLM router behavior.
    - Update `answer_node` (`:2134`) to pick templates using task_mode precedence:
      - task_mode==issue → use `issue_ans_*` (language-specific) for Step1 list (Phase A fallback path)
      - task_mode==sop → force `setup_ans_*`
      - else → existing route-based templates
      - Guardrail: if issue prompt is missing (`spec.issue_ans_* is None`), fallback to existing `general_ans_*` (do not crash).
    - Update `judge_node` (`:2224`) to:
      - task_mode==issue → use `judge_issue_sys`
      - else existing route-based judge system
  - Tests:
    - Add backend unit tests ensuring:
      - task_mode set → route_node does NOT call router LLM (use stub LLM that raises if router prompt invoked).
      - task_mode issue → answer_node uses issue template system prompt for each language.
      - Missing issue prompt falls back to general template.

  **Must NOT do**:
  - Do not remove legacy router-based routing when task_mode is unset.
  - Do not add new `Route` literals.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: localized code changes + tests
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 5 | Blocked By: 1,2

  **References**:
  - Current route_node: `backend/llm_infrastructure/llm/langgraph_agent.py:1046`
  - Current answer_node: `backend/llm_infrastructure/llm/langgraph_agent.py:2134`
  - Current judge_node: `backend/llm_infrastructure/llm/langgraph_agent.py:2224`
  - task_mode source: `backend/llm_infrastructure/llm/langgraph_agent.py:3369`

  **Acceptance Criteria**:
  - [ ] `uv run pytest -v tests/api/test_agent_autoparse_confirm_interrupt_resume.py` passes
  - [ ] New task_mode routing/templating tests pass

  **QA Scenarios**:
  ```
  Scenario: task_mode sop forces route=setup without router LLM
    Tool: Bash
    Steps: uv run pytest -v tests/api/test_agent_autoparse_confirm_interrupt_resume.py -k sop
    Expected: Pass
    Evidence: .sisyphus/evidence/task-4-sop-route.txt

  Scenario: task_mode issue uses issue answer template
    Tool: Bash
    Steps: uv run pytest -v backend/tests/test_answer_language_templates.py
    Expected: Pass
    Evidence: .sisyphus/evidence/task-4-issue-template.txt
  ```

  **Commit**: YES | Message: `feat(agent): make task_mode override routing and prompts` | Files: `backend/llm_infrastructure/llm/langgraph_agent.py`, tests

- [x] 5. Implement Issue Step1/2/3 Flow in LangGraph (Phase B)

  **What to do**:
  - Backend state additions (in `backend/llm_infrastructure/llm/langgraph_agent.py` AgentState):
    - `graph_version: str` (set once per thread)
    - `pending_interrupt_nonce: str | None`
    - `issue_flow_step: int | None` (1/2/3)
    - `issue_cases: list[dict] | None` (persisted list for selection validation)
    - `issue_selected_doc_id: str | None`
    - `issue_sop_confirmed: bool | None`
    - `issue_sop_candidates: list[str] | None`
  - Add new node helpers in `backend/llm_infrastructure/llm/langgraph_agent.py`:
    - Step 1 (prepare + interrupt split to persist nonce in checkpoint):
      - `issue_step1_prepare_node`: build a *persisted* `issue_cases` list from `display_docs` / `docs` (RetrievalResult list; use `metadata.title`/`metadata.doc_description` fallback to `doc_id`, and a deterministic snippet for `summary`) so resume validation can check `selected_doc_id`; generate `pending_interrupt_nonce`; return update with `issue_flow_step=1`, `issue_cases`, `pending_interrupt_nonce`.
      - `issue_step1_interrupt_node`: call `interrupt({type:"issue_case_selection", nonce, cases:issue_cases, instruction, question})`.
    - Step 1 resume apply:
      - `issue_case_selection_apply_node`: validate resume decision (nonce + doc_id exists in `issue_cases`), then:
        - set `issue_selected_doc_id`
        - set `issue_flow_step=2`
        - clear `pending_interrupt_nonce` (single-use nonce)
    - Step 2 (prepare + interrupt split):
    - `issue_step2_prepare_detail_node`:
      - Fetch richer content for selected doc_id via `SearchService.fetch_doc_chunks` if available (fallback to `answer_ref_json`).
      - Call LLM `issue_detail_ans_*` to generate detail text.
      - Compute `has_sop_ref` + `sop_hint` from content using deterministic regex:
        - `SOP_ID_RE = r"\bSOP[\s:_-]*[A-Za-z0-9][A-Za-z0-9._-]{2,}\b"`
        - Normalize matches by stripping whitespace and collapsing separators to `SOP-...`.
      - Generate new `pending_interrupt_nonce`; return update with detail text + nonce + `issue_sop_candidates` (list of extracted SOP ids, capped at 3).
      - `issue_step2_interrupt_sop_confirm_node`: call `interrupt({type:"issue_sop_confirm", nonce, prompt, has_sop_ref, sop_hint})`.
    - Step 2 resume apply:
      - `issue_sop_confirm_apply_node`: validate resume decision, then:
        - set `issue_sop_confirmed`
        - set `issue_flow_step=3`
        - clear `pending_interrupt_nonce` (single-use nonce)
    - `issue_step3_sop_answer_node`:
      - If `issue_sop_confirmed` is false: skip to finalize.
      - If true: run SOP retrieval restricted to SOP doc types (`expand_doc_type_selection(["sop"])`) with this deterministic policy:
        1) If `issue_sop_candidates` has values: build `sop_query` from the top candidate token (do NOT assume it matches Elasticsearch `doc_id`).
           - Example: `sop_query = f"{issue_sop_candidates[0]} SOP 절차"`
        2) Else: build `sop_query` as `f"{original_query} SOP 절차"`.
        3) Retrieve using `sop_query` with SOP doc-type restriction; do not set `selected_doc_ids` from regex tokens.
      - Use existing `setup_ans_*` prompt to answer; MUST include citations from SOP REFS.
    - `finalize_node`: clear issue-flow fields (`issue_flow_step`, `issue_cases`, `issue_selected_doc_id`, `issue_sop_confirmed`, `issue_sop_candidates`, `pending_interrupt_nonce`) so next user query starts clean.
      - Requirement: finalize/cleanup must run for ALL terminal outcomes (confirm=false, retrieval failure, LLM failure, validation failure) so stale issue-flow state never leaks into later turns.
  - Graph wiring (`backend/services/agents/langgraph_rag_agent.py:458`):
    - Insert conditional branch after `expand_related`:
      - If `task_mode=="issue"` and no active issue_flow_step → go to `issue_step1_list_cases`.
      - Else → existing `answer` node.
    - Wire issue-step nodes linearly with interrupts/resume.
    - Ensure `should_retry` short-circuits to `done` when issue_flow_step is active (avoid retry loops that re-trigger interrupts).
  - Tests:
    - Add `tests/api/test_agent_issue_flow_interrupt_resume.py` covering:
      - Happy path: Step1 interrupt → resume select doc_id → Step2 interrupt → resume confirm false → finalize.
      - SOP-confirm path: confirm true triggers Step3 and returns final answer.
      - Invalid selection doc_id rejected (400) and invalid nonce rejected (409).

  **Must NOT do**:
  - Do not reuse `retrieval_review` for issue selection (Phase B needs guided resume family).
  - Do not leave issue_flow state uncleared at end.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: new LangGraph branch + interrupts + safety
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 6 | Blocked By: 3,4

  **References**:
  - Interrupt pattern: `backend/llm_infrastructure/llm/langgraph_agent.py:3343` (guided confirm), `backend/llm_infrastructure/llm/langgraph_agent.py:1979` (retrieval_review)
  - Graph edges: `backend/services/agents/langgraph_rag_agent.py:631`
  - Fetch doc chunks API: `backend/services/es_search_service.py:367`

  **Acceptance Criteria**:
  - [ ] `uv run pytest -v tests/api/test_agent_issue_flow_interrupt_resume.py` passes
  - [ ] `uv run pytest -v tests/api/test_agent_autoparse_confirm_interrupt_resume.py` still passes

  **QA Scenarios**:
  ```
  Scenario: Issue flow happy path via API
    Tool: Bash
    Steps: uv run pytest -v tests/api/test_agent_issue_flow_interrupt_resume.py
    Expected: All pass
    Evidence: .sisyphus/evidence/task-5-issue-flow-api.txt

  Scenario: Issue flow does not enter retry loop
    Tool: Bash
    Steps: uv run pytest -v tests/api/test_agent_issue_flow_interrupt_resume.py -k no_retry
    Expected: Pass; graph ends without should_retry re-entering issue_step1
    Evidence: .sisyphus/evidence/task-5-no-retry.txt
  ```

  **Commit**: YES | Message: `feat(agent): add issue multi-step flow interrupts` | Files: `backend/llm_infrastructure/llm/langgraph_agent.py`, `backend/services/agents/langgraph_rag_agent.py`, `backend/api/routers/agent.py`, tests

- [x] 6. Frontend UI for Issue Case Selection + SOP Confirm (Phase B)

  **What to do**:
  - Types:
    - Extend `frontend/src/features/chat/types.ts`:
      - Add interrupt payload shapes for `issue_case_selection` and `issue_sop_confirm`.
      - Payload contract (must match backend interrupt payloads from Task 5):
        - `issue_case_selection`: `{type, nonce, question, instruction, cases:[{doc_id,title,summary}]}`
        - `issue_sop_confirm`: `{type, nonce, question, instruction, prompt, has_sop_ref, sop_hint}`
  - Hook (`frontend/src/features/chat/hooks/use-chat-session.ts`):
    - Confirm `resolveInterruptKind()` recognizes the new types (already present); focus on pending UI state + panels.
    - Add explicit pending state (do not reuse `pendingInterrupt`):
      - `pendingIssueCaseSelection: { threadId, question, instruction, nonce, cases } | null`
      - `pendingIssueSopConfirm: { threadId, question, instruction, nonce, prompt, hasSopRef, sopHint } | null`
    - In `handleAgentResponse`, when `kind === "issue_case_selection"`, set `pendingIssueCaseSelection` from payload and clear any other pending panels.
    - In `handleAgentResponse`, when `kind === "issue_sop_confirm"`, set `pendingIssueSopConfirm` from payload and clear any other pending panels.
    - Ensure sending a decision for these types is treated as guided resume (same as `auto_parse_confirm`).
    - Add submit helpers (mirror GuidedSelection patterns):
      - `submitIssueCaseSelection(selectedDocId: string)` -> `send({ text: `이슈 선택: ${selectedDocId}`, decisionOverride: {type:"issue_case_selection", nonce, selected_doc_id:selectedDocId} })`
      - `submitIssueSopConfirm(confirm: boolean)` -> `send({ text: `SOP 확인: ${confirm ? "예" : "아니오"}`, decisionOverride: {type:"issue_sop_confirm", nonce, confirm} })`
  - Components:
    - Add `frontend/src/features/chat/components/issue-case-selection-panel.tsx`:
      - Render case list (1..N), support click + numeric input, submit via `onSelect(selected_doc_id)`.
    - Add `frontend/src/features/chat/components/issue-sop-confirm-panel.tsx`:
      - Render prompt + Yes/No buttons, submit via `onConfirm(confirm)`.
    - Export via `frontend/src/features/chat/components/index.ts`.
  - Page wiring (`frontend/src/features/chat/pages/chat-page.tsx`):
    - Render the new panels when pending.
    - Disable `ChatInput` while issue panels are pending (same policy as guided selection).
  - Tests:
    - Add unit tests for panels (similar to `frontend/src/features/chat/__tests__/guided-selection-panel.test.tsx`).
    - Extend `frontend/src/features/chat/__tests__/chat-request-payload.test.tsx` with:
      - “issue-case-selection resume payload” test (ask_user_after_retrieve must remain false).
    - Add/extend tests to verify `handleAgentResponse` sets the correct pending panel state for each interrupt.

  **Must NOT do**:
  - Do not route issue-step resumes through the existing HIL `pendingInterrupt` path.

  **Recommended Agent Profile**:
  - Category: `visual-engineering` — Reason: new panels + input ergonomics
  - Skills: [`frontend-ui-ux`]

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 7 | Blocked By: 3,5

  **References**:
  - Guided panel pattern: `frontend/src/features/chat/components/guided-selection-panel.tsx`
  - Interrupt handling: `frontend/src/features/chat/hooks/use-chat-session.ts:157`
  - Payload tests baseline: `frontend/src/features/chat/__tests__/chat-request-payload.test.tsx:89`

  **Acceptance Criteria**:
  - [ ] `cd frontend && npm test` passes
  - [ ] Panel tests exist for both Step1 and Step2 panels.

  **QA Scenarios**:
  ```
  Scenario: Issue case selection panel submits correct decision
    Tool: Bash
    Steps: cd frontend && npm test -- -t "IssueCaseSelectionPanel"
    Expected: Pass
    Evidence: .sisyphus/evidence/task-6-fe-panel-tests.txt

  Scenario: Issue SOP confirm panel submits correct decision
    Tool: Bash
    Steps: cd frontend && npm test -- -t "IssueSopConfirmPanel"
    Expected: Pass
    Evidence: .sisyphus/evidence/task-6-fe-sop-confirm-tests.txt

  Scenario: Hook sends guided resume payload for issue selection
    Tool: Bash
    Steps: cd frontend && npm test -- -t "issue-case-selection resume payload"
    Expected: Pass
    Evidence: .sisyphus/evidence/task-6-fe-hook-payload.txt
  ```

  **Commit**: YES | Message: `feat(chat-ui): support issue flow interrupts` | Files: `frontend/src/features/chat/**/*`

- [x] 7. End-to-End UI Flow QA (Vitest + RTL) + Evidence Capture (no Playwright dependency)

  **What to do**:
  - Add ONE deterministic frontend test that drives the full UI flow (without real backend / ES):
    - Mock SSE transport (`connectSse`) so the hook receives a scripted sequence of `final` events whose `result` is an `AgentResponse`.
    - Scripted sequence must cover:
      1) initial send -> `interrupted=true` with `auto_parse_confirm` payload
      2) resume auto_parse_confirm (task_mode=issue) -> `interrupted=true` with `issue_case_selection` payload
      3) resume issue_case_selection -> `interrupted=true` with `issue_sop_confirm` payload
      4) resume issue_sop_confirm (confirm=false) -> `interrupted=false` final answer
    - Assert UI renders the correct panel at each step and the resume payload includes `nonce` and keeps `ask_user_after_retrieve=false`.
  - Save the test run output under `.sisyphus/evidence/`.

  **Must NOT do**:
  - Do not rely on real ES data or a running backend; the test must be deterministic via mocked SSE.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: cross-component flow test + SSE mocking
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: none | Blocked By: 6

  **References**:
  - SSE transport: `frontend/src/lib/sse.ts`
  - SSE event handling: `frontend/src/features/chat/hooks/use-chat-session.ts:720`
  - Guided confirm panel test patterns: `frontend/src/features/chat/__tests__/guided-selection-panel.test.tsx`
  - Suggested test location: `frontend/src/features/chat/__tests__/issue-flow-ui.test.tsx` (new)

  **Acceptance Criteria**:
  - [ ] A new Vitest test exists and passes that covers the full UI flow with mocked SSE.
  - [ ] Evidence log exists: `.sisyphus/evidence/task-7-issue-flow-ui.txt`

  **QA Scenarios**:
  ```
  Scenario: Guided confirm → issue flow panels (deterministic mocked SSE)
    Tool: Bash
    Steps: cd frontend && npm test -- -t "issue flow end-to-end"
    Expected: Pass; panels appear in order and final answer renders
    Evidence: .sisyphus/evidence/task-7-issue-flow-ui.txt

  Scenario: Invalid nonce decision rejected (negative)
    Tool: Bash
    Steps: uv run pytest -v tests/api/test_agent_issue_flow_interrupt_resume.py -k wrong_nonce
    Expected: Pass
    Evidence: .sisyphus/evidence/task-7-negative-nonce.txt
  ```

  **Commit**: NO | Message: - | Files: -

- [x] 8. Repetition Prevention (Already Committed: `a0c9c59`)

  **What was done**:
  - `repeat_penalty` 1.1→1.3, `repeat_last_n=256` in `backend/config/settings.py`
  - `repeat_last_n` passed to ollama engine native API in `backend/llm_infrastructure/llm/engines/ollama.py`
  - `_truncate_repetition` safety net in `answer_node` with WARNING log in `backend/llm_infrastructure/llm/langgraph_agent.py`

  **Commit**: DONE | `a0c9c59` — `fix(llm): strengthen repetition prevention with repeat_last_n and fallback truncation`

- [x] 9. Fix Device Catalog Loading Error (Backend Wiring + Frontend Confirmation)

  **What to do**:
  - Root cause to check first: the devices router is defined but not registered in the FastAPI app.
  - Backend: enable `devices` router registration so `GET /api/device-catalog` exists:
    - `backend/api/main.py:10` (currently commented router import that includes `devices`)
    - `backend/api/main.py:112` (currently commented `app.include_router(devices.router, prefix="/api")`)
    - Endpoint definition: `backend/api/routers/devices.py:133`
    - Expected final path: router defines `/device-catalog` and app mounts with `prefix="/api"` -> `/api/device-catalog`.
  - Frontend: keep current call path (`/api/device-catalog`) and ensure errors are surfaced gracefully.
  - Tests:
    - Add `tests/api/test_device_catalog_route_registered.py` to assert `/api/device-catalog` is registered on `create_app()`.

  **References**:
  - API call: `frontend/src/features/chat/api.ts:60-62` (`fetchDeviceCatalog()`)
  - Error display: `frontend/src/features/chat/pages/chat-page.tsx:110-119`
  - API router wiring: `backend/api/main.py:98`, `backend/api/main.py:112`
  - Device catalog endpoint: `backend/api/routers/devices.py:133`
  - Design doc: `docs/2026-03-09-agent-분기-개선.md` Section 8-1

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: single router include + targeted tests
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: none | Blocked By: none

  **Acceptance Criteria**:
  - [ ] Backend route is registered (no 404): `uv run pytest -v tests/api/test_device_catalog_route_registered.py` passes.
  - [ ] Frontend does not crash on device catalog fetch errors: `cd frontend && npm test -- -t "device catalog"` passes.

  **QA Scenarios**:
  ```
  Scenario: Device catalog route registered (no 404)
    Tool: Bash
    Steps: uv run pytest -v tests/api/test_device_catalog_route_registered.py
    Expected: Pass
    Evidence: .sisyphus/evidence/task-9-device-catalog-route.txt

  Scenario: Frontend device catalog fetch is resilient
    Tool: Bash
    Steps: cd frontend && npm test -- -t "device catalog"
    Expected: Pass
    Evidence: .sisyphus/evidence/task-9-fe-device-catalog.txt
  ```

  **Commit**: YES | Message: `fix(api): register devices router for device catalog` | Files: `backend/api/main.py`, `tests/api/test_device_catalog_route_registered.py`

- [x] 10. Remove Recommend Tag from Skip Button (Frontend)

  **What to do**:
  - In `guided-selection-panel.tsx`, ensure `__skip__` option does NOT display the `<Tag color="blue">Recommend</Tag>` badge.
  - Either filter out `recommended` flag for skip option in rendering, or ensure backend sends `recommended: false` for skip.
  - Tests:
    - Add a new test case to `frontend/src/features/chat/__tests__/guided-selection-panel.test.tsx`.

  **References**:
  - Skip option rendering: `frontend/src/features/chat/components/guided-selection-panel.tsx:322-343`
  - Manual skip button: `frontend/src/features/chat/components/guided-selection-panel.tsx:364-372`
  - Design doc: `docs/2026-03-09-agent-분기-개선.md` Section 8-2

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: small UI tweak + RTL test
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: none | Blocked By: none

  **Acceptance Criteria**:
  - [ ] Add a Vitest RTL test that renders `GuidedSelectionPanel` with an `__skip__` option marked recommended and asserts the Skip row does NOT render the `Recommend` tag.
  - [ ] Same test asserts at least one non-skip recommended option still renders the `Recommend` tag.
  - [ ] `cd frontend && npm test -- -t "skip.*recommend"` passes.

  **QA Scenarios**:
  ```
  Scenario: Skip option has no Recommend tag
    Tool: Bash
    Steps: cd frontend && npm test -- -t "skip.*recommend"
    Expected: Pass
    Evidence: .sisyphus/evidence/task-10-skip-tag.txt
  ```

  **Commit**: YES | Message: `fix(chat-ui): remove recommend badge from skip option` | Files: `frontend/src/features/chat/components/guided-selection-panel.tsx`, `frontend/src/features/chat/__tests__/guided-selection-panel.test.tsx`

- [x] 11. Restore Re-search Button (Frontend)

  **What to do**:
  - Investigate why "재검색" button is no longer visible after answer generation.
  - Check `message-item.tsx` for `onRegenerate` callback wiring and rendering conditions.
  - Likely regressed in commit `f55e3f7` (UX improvement). Restore the button rendering.
  - Tests:
    - Create `frontend/src/features/chat/__tests__/message-item.test.tsx` with a focused regenerate-button visibility + click test.

  **References**:
  - Regeneration handler: `frontend/src/features/chat/pages/chat-page.tsx:140-165` (`submitRegeneration`)
  - Message item: `frontend/src/features/chat/components/message-item.tsx:220-228, 320-331`
  - Right sidebar: `frontend/src/components/layout/right-sidebar.tsx`
  - Design doc: `docs/2026-03-09-agent-분기-개선.md` Section 8-3

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: small UI restore + RTL test
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: none | Blocked By: none

  **Acceptance Criteria**:
  - [ ] Add a Vitest RTL test that renders a post-answer assistant message and asserts the 재검색/regenerate control is visible.
  - [ ] Same test clicks the control and asserts `submitRegeneration` (or the provided handler) is invoked with the expected original query.
  - [ ] `cd frontend && npm test -- -t "재검색|re-search|regenerate"` passes.

  **QA Scenarios**:
  ```
  Scenario: Re-search button renders and triggers handler
    Tool: Bash
    Steps: cd frontend && npm test -- -t "재검색|re-search|regenerate"
    Expected: Pass
    Evidence: .sisyphus/evidence/task-11-research-button.txt
  ```

  **Commit**: YES | Message: `fix(chat-ui): restore re-search action after answer` | Files: `frontend/src/features/chat/components/message-item.tsx`, `frontend/src/features/chat/pages/chat-page.tsx`, tests

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real UI QA (agent-executed) — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Prefer 3 commits:
  1) (already landed) `feat(agent): add issue prompts and v2 language variants`
  2) `feat(agent): task_mode routing and issue multi-step interrupts`
  3) `feat(chat-ui): support issue flow interrupts and panels`

## Success Criteria
- `task_mode` affects routing + answer/judge behavior exactly as defined in `docs/2026-03-09-agent-분기-개선.md`.
- Issue Step1/2/3 flow works via interrupts/resume without graph mismatch or stale resume decisions.
- Tests and QA scenarios pass with recorded evidence.

## Appendix: Archived start-work-all-remaining.md (verbatim)

> Merged here on 2026-03-10 to reduce active plan files. Source checklist was fully marked complete. (The original file `start-work-all-remaining.md` has been deleted; references inside this code block are historical.)

### Appendix Errata (archived content)
- The archived block includes the historical statement "Active writable plan files ... are limited to `start-work-all-remaining.md` and `start-work-remaining.md`". Current active plan files in `.sisyphus/plans/` are:
  - `.sisyphus/plans/agent-branching-improvement-2026-03-09.md` (this plan)
  - `.sisyphus/plans/start-work-remaining.md`
  - `.sisyphus/plans/2026-03-10-122b-moe-quality-regression-mitigation.md`
- Keep the code block verbatim as archival evidence; treat any conflicts inside it as historical-only.

```markdown
# Unified Start-Work Plan (Manual Reconciliation)

## Intent
- This file is the only execution plan for /start-work.
- `.sisyphus/plans/.legacy/` is source spec storage, not a direct execution target.
- Active writable plan files in `.sisyphus/plans/` are limited to `start-work-all-remaining.md` and `start-work-remaining.md`.
- Every task below is reconciled against the current repository layout before merge.

## Reconciliation Method (One-by-One)
1. Read each legacy plan in order.
2. Compare pending TODOs with current files/tests/scripts.
3. Replace stale references with current equivalents.
4. Exclude completed scopes from active execution queue.
5. Run only the integrated backlog in this file.

## Process Safety Rules (Zombie Prevention)
- Forbidden in plan execution: &, nohup, disown, orphaned background loops.
- Long-running commands must use bounded execution (timeout with explicit seconds).
- If any worker process is spawned, cleanup is mandatory before moving to next task.
- Evidence paths are explicit per phase to avoid ambiguous reruns.
- Do not recreate legacy task files in `.sisyphus/plans/`; keep legacy specs only in `.sisyphus/plans/.legacy/`.
- Do not run wildcard restore/create flows against `.sisyphus/plans/*.md`.

## Legacy-by-Legacy Consistency Review

### 1) `.sisyphus/plans/.legacy/ui-autoparse-confirm-task-mode-2026-03-05.md`
- Status: completed.
- Decision: keep as historical evidence only; no active backlog.

### 2) `.sisyphus/plans/.legacy/ui-chat-improvements-v2.md`
- Status: partially complete.
- Decision: keep only REQ-2 robustness hardening in active backlog; all other REQ items remain closed.
- File consistency: `frontend/src/features/chat/components/guided-selection-panel.tsx`, `frontend/src/features/chat/hooks/use-chat-session.ts`, `frontend/src/features/chat/pages/chat-page.tsx`, `frontend/src/features/chat/__tests__/guided-selection-panel.test.tsx`.

### 3) `.sisyphus/plans/.legacy/chunk_v3_embed_ingest_plan.md`
- Status: mostly complete (C1/C2/C4 done), one operational rollout task remains.
- Scope alignment: keep section-rollout execution (C3) and remove stale symbol-style references.
- File consistency: `scripts/chunk_v3/run_chunking.py`, `scripts/chunk_v3/chunkers.py`, `scripts/chunk_v3/run_embedding.py`, `scripts/chunk_v3/run_ingest.py`, `scripts/chunk_v3/validate_vlm.py`, `backend/llm_infrastructure/elasticsearch/mappings.py`, `backend/domain/doc_type_mapping.py`, `scripts/evaluation/evaluate_sop_agent_page_hit.py`.

### 4) `.sisyphus/plans/.legacy/chapter-grouping-retrieval.md`
- Status: partially complete, operational reindex follow-up remains.
- Scope alignment: keep reindex + retrieval integration tasks only.
- File consistency: `scripts/chunk_v3/section_extractor.py`, `scripts/chunk_v3/chunkers.py`, `scripts/chunk_v3/run_ingest.py`, `backend/llm_infrastructure/retrieval/engines/es_search.py`, `backend/llm_infrastructure/llm/langgraph_agent.py`.

### 5) `.sisyphus/plans/.legacy/agent-retrieval-followups-2026-03-04.md`
- Status: partially complete.
- Scope alignment: keep required Stage2/MQ follow-ups, but remove B4-coupled before/after gate tasks from active execution.
- Mismatch corrected:
  - Legacy ref tests/api/test_agent_rrf_and_sticky_gates.py does not exist.
  - Current verification set: `tests/api/test_agent_stage2_retrieval.py`, `tests/api/test_agent_sticky_policy_followup_only.py`, `tests/api/test_agent_interrupt_resume_regression.py`, `backend/tests/test_sop_intent_heuristic.py`, `backend/tests/test_retrieve_node_sop_soft_boost.py`.

### 6) `.sisyphus/plans/.legacy/before-after-regression-compare.md`
- Status: deferred by user decision (2026-03-09).
- Scope alignment: remove from active backlog for now; execute only on later explicit request.
- File consistency: `scripts/evaluation/regression_compare_manifest.py`, `scripts/evaluation/evaluate_sop_agent_page_hit.py`, `scripts/evaluation/agent_eval_report.py`, `tests/evaluation/test_regression_compare_manifest.py`.

### 7) `.sisyphus/plans/.legacy/paper-a-scope-implementation.md`
- Status: deferred (lowest priority, separate future request).
- Scope alignment: exclude from current active execution queue.
- File consistency: `scripts/paper_a/build_corpus_meta.py`, `scripts/paper_a/build_shared_and_scope.py`, `scripts/paper_a/build_family_map.py`, `scripts/paper_a/build_eval_sets.py`, `scripts/paper_a/evaluate_paper_a.py`, `scripts/paper_a/retrieval_runner.py`, `backend/llm_infrastructure/retrieval/filters/scope_filter.py`, `docs/papers/20_paper_a_scope/paper_a_scope_spec.md`.

### 8) `.sisyphus/plans/.legacy/paper-a-supervisor-review-plan.md`
- Status: deferred (lowest priority, separate future request).
- Merge decision: keep as historical paper stream; do not execute in this cycle.
- File consistency: `docs/papers/20_paper_a_scope/evidence_mapping.md`, `docs/papers/20_paper_a_scope/review/`, `scripts/paper_a/`.

### 9) `.sisyphus/plans/.legacy/agent-retrieval-stability-hardening.md`
- Status: completed.
- Decision: no active backlog.

### 10) `.sisyphus/plans/.legacy/paper-b-stability.md`
- Status: deferred (lowest priority, separate future request).
- Decision: exclude from current active execution queue.

## Integrated Execution Backlog (Current-Code Synced)

## Phase A - Baseline Consistency Freeze (blocking)
- [x] A1. Create consolidated mismatch ledger at `.sisyphus/evidence/start-work/phase-a/consistency-ledger.md`.
- [x] A2. Confirm all execution references in this plan resolve to current files (or explicit "historical only").
- [x] A3. Freeze this plan as the single source and stop direct execution from `.sisyphus/plans/.legacy/`.

## Phase B - UI + Agent Retrieval Follow-ups
- [x] B1. Finish REQ-6 tabbed guided-selection UI and keep existing guided-confirm behavior stable.
  - Target files: `frontend/src/features/chat/components/guided-selection-panel.tsx`, `frontend/src/features/chat/pages/chat-page.tsx`, `frontend/src/features/chat/hooks/use-chat-session.ts`, `frontend/src/features/chat/__tests__/guided-selection-panel.test.tsx`.
- [x] B2. Complete retrieval follow-ups (stage2 + strict override + sticky policy + SOP soft boost).
  - Target files: `backend/llm_infrastructure/llm/langgraph_agent.py`, `backend/config/settings.py`, `backend/llm_infrastructure/retrieval/engines/es_search.py`, `scripts/evaluation/evaluate_sop_agent_page_hit.py`.
- [x] B3. Replace stale test references with current test suite and pass regressions.
  - Test files: `tests/api/test_agent_stage2_retrieval.py`, `tests/api/test_agent_sticky_policy_followup_only.py`, `tests/api/test_agent_interrupt_resume_regression.py`, `backend/tests/test_sop_intent_heuristic.py`, `backend/tests/test_retrieve_node_sop_soft_boost.py`.
- [x] B4. De-scope HTTP-mode before/after gate from this cycle (user-deferred).
  - Decision: keep historical evidence only; do not execute B4 unless explicitly re-requested.
  - Evidence (historical): `.sisyphus/evidence/start-work/phase-b/http_eval/`, `.sisyphus/evidence/start-work/phase-b/before_after/`.
- [x] B5. Top priority: finish REQ-2 model-format robustness hardening.
  - Scope: shared parse fallback util, judge JSON mode, retry cap, translate/query_rewrite normalization, prompt hardening.
  - Primary files: `backend/llm_infrastructure/llm/langgraph_agent.py`, `backend/llm_infrastructure/llm/prompts/auto_parse_v1.yaml`, `backend/llm_infrastructure/llm/prompts/router_v1.yaml`, `tests/api/test_agent_autoparse_confirm_interrupt_resume.py`.

## Phase C - Chunk v3 + Chapter Grouping Integration
- [x] C1. Finalize canonical doc_type + manifest + VLM coverage gates.
  - Target files: `scripts/chunk_v3/run_chunking.py`, `scripts/chunk_v3/chunkers.py`, `scripts/chunk_v3/validate_vlm.py`, `normalize.py`.
- [x] C2. Harden embedding/ingest/verify contracts and mapping safety.
  - Target files: `scripts/chunk_v3/run_embedding.py`, `scripts/chunk_v3/run_ingest.py`, `backend/llm_infrastructure/elasticsearch/mappings.py`.
- [x] C3. Re-verify section grouping rollout already in progress/executed.
  - Scope: validate section_* fields in regenerated JSONL, verify `chunk_v3_content` sync for 3 models, and confirm retrieval section-expansion behavior on current indices.
  - Target files/scripts: `scripts/chunk_v3/section_extractor.py`, `scripts/chunk_v3/run_chunking.py`, `scripts/chunk_v3/run_ingest.py`, `backend/llm_infrastructure/retrieval/engines/es_search.py`, `backend/llm_infrastructure/llm/langgraph_agent.py`.
- [x] C4. Run smoke/formal eval and update runbook alignment.
  - Target files: `scripts/chunk_v3/smoke_eval.py`, `scripts/chunk_v3/eval_sop_questionlist.py`, `.sisyphus/plans/.legacy/chunk_v3_embed_ingest_plan.md`, `docs/2026-03-07-3Model-Embedding-Eval-Report.md`.

## Phase D - Paper A Unified Stream (Implementation + Supervisor Verification)
- [x] D1. Lock eval schema/policy artifacts and resolve doc-data-code mismatches first.
  - Target files: `scripts/paper_a/rebuild_query_gold_master_splits.py`, `scripts/paper_a/validate_master_eval_jsonl.py`, `scripts/paper_a/build_shared_and_scope.py`, `docs/papers/20_paper_a_scope/evidence_mapping.md`.
- [x] D2. Complete evaluator parity and statistics pipeline.
  - Target files: `scripts/paper_a/evaluate_paper_a.py`, `scripts/paper_a/retrieval_runner.py`, `backend/llm_infrastructure/retrieval/filters/scope_filter.py`.
- [x] D3. Execute reproducible runs and produce reviewer-grade audit outputs.
  - Evidence: `.sisyphus/evidence/paper-a/runs/`, `docs/papers/20_paper_a_scope/evidence/2026-03-05_paper_a_run_index.md`, `docs/papers/20_paper_a_scope/review/reviewer_report.md`.
- [x] D4. Update manuscript-aligned docs using only evidence-backed claims.
  - Target files: `docs/papers/20_paper_a_scope/paper_a_scope.md`, `docs/papers/20_paper_a_scope/review/consistency_audit.md`, `docs/papers/20_paper_a_scope/review/`.

## Phase E - Agent Retrieval Follow-up Replan (C3-aware)
- [x] E1. Rebaseline Stage2/MQ follow-up plan against current C3 outputs.
  - Scope: align Stage2/MQ tasks to canonical doc_type, section grouping metadata availability, and current index state.
  - Reference plans: `.sisyphus/plans/.legacy/agent-retrieval-followups-2026-03-04.md`, `.sisyphus/plans/.legacy/chapter-grouping-retrieval.md`.
- [x] E2. Execute required retrieval follow-up subset (excluding B4/before-after gates).
  - Include: stage2 retrieve_node integration, integration tests, sticky policy freeze, eval JSONL normalization, docs sync.
  - Exclude: HTTP before/after quality gate and final before/after report tasks.

## Definition of Done
- All pending tasks in Phases A-E are complete.
- Deferred (not in active queue): B4 before/after gate, Paper A stream, Paper B stream, and cross-plan F* audit bundles.
- No stale reference remains unresolved in this plan.
- Every execution phase has evidence under `.sisyphus/evidence/start-work/` or canonical phase-specific evidence paths referenced above.
- No zombie process remains after each phase boundary.
```
