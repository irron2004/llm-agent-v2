# UI Guided Selection (Auto-Parse Confirm + Task-Mode Scoped Retrieval)

## TL;DR
> **Summary**: Add a guided confirmation step before retrieval/answer so users can confirm `answer language`, `device`, `equip_id`, and `task` (SOP vs Issue vs All); `task` forcibly scopes doc_type filtering to prevent SOP being crowded out by GCB.
> **Deliverables**: (1) Backend `auto_parse_confirm` LangGraph interrupt + resume merge + metadata, (2) Frontend `GuidedSelectionPanel` with numeric + click selection, (3) `task_mode`→doc_type strict scope enforcement, (4) Regression tests (backend + frontend) proving resume correctness and scoping.
> **Effort**: Medium
> **Parallel**: YES - 3 waves
> **Critical Path**: Backend resume routing (same graph/checkpointer) → FE guided panel + resume payload → scoped retrieval validation tests

## Context
### Original Request
- Based on `docs/2026-03-05-ui_개선.md`: after a user asks a question, run `auto_parse` first, then show sequential confirmation UI `language -> device -> equipment_id -> task` before retrieval/answer.
- Task selection is explicit and authoritative:
  - `SOP 조회`: search SOP only
  - `이슈조회`: search `myservice/gcb/ts` only
  - `전체조회`: keep existing behavior (no doc_type filter)

### Interview Summary
- No additional preferences provided; apply safe defaults.

### Repo-Grounded Findings
- Existing interrupt/resume patterns already exist:
  - `retrieval_review`: `backend/llm_infrastructure/llm/langgraph_agent.py` (`ask_user_after_retrieve_node`)
  - `device_selection`: `backend/llm_infrastructure/llm/langgraph_agent.py` (`device_selection_node`)
  - API returns interrupts via `__interrupt__` in `backend/api/routers/agent.py`.
- `auto_parse_node` is rule-based and cheap (no LLM call) and already outputs `parsed_query` including `device_names`, `equip_ids`, `detected_language`:
  - `backend/llm_infrastructure/llm/langgraph_agent.py` (`auto_parse_node`)
- Current API resume path always uses the HIL graph (`_new_hil_agent`) when `resume_decision` is present:
  - `backend/api/routers/agent.py` (`run_agent`, `run_agent_stream`)
  - Guided flow must not resume using a different graph/checkpointer.
- Doc type group expansion exists and should be reused for strict scoping:
  - `backend/domain/doc_type_mapping.py` (`expand_doc_type_selection`)

### Oracle Review (architecture)
- Insert a single new interrupt node `auto_parse_confirm` immediately after `auto_parse` and resume within the same compiled graph/checkpointer.
- Keep `task_mode` as an explicit enum and compute a dedicated doc_type scope override; do not hide this inside ambiguous logic.
- Gate new behavior behind a request flag for backward compatibility.

### Metis Review (gaps addressed)
- Resume routing risk: backend currently routes **all** resumes through `_new_hil_agent`; guided-confirm resumes MUST route to the auto-parse/guided graph (same checkpointer + node set) or checkpoints become incompatible.
- Backward compatibility risk: existing `retrieval_review` approve/deny resumes are often `bool | str` (not typed dict). Guided routing must trigger ONLY when `resume_decision` is a dict with `type == "auto_parse_confirm"`.
- Frontend resume risk: current resume requests tend to force `ask_user_after_retrieve=true` + `auto_parse=false`; guided resume must NOT force HIL flags.
- State divergence risk: any guided selections must update BOTH top-level keys and `parsed_query.*` keys, because downstream retrieval prefers `parsed_query` in several branches.
- UX scope: confirm should happen before retrieval/answer on every guided-enabled user question; keep `0` as a fast-path to accept recommendation/skip.

## Work Objectives
### Core Objective
- Add a guided confirmation UX that lets users explicitly control retrieval scope (SOP vs Issue vs All) and answer language before retrieval runs, while preserving existing interrupt/resume behaviors.

### Deliverables
- Backend:
  - New LangGraph interrupt type: `auto_parse_confirm`
  - New API request flag to enable the flow (default OFF)
  - Resume routing that guarantees the same graph/checkpointer is used for the interrupt thread_id
  - `task_mode`→doc_type strict scope enforcement using `expand_doc_type_selection`
  - Answer language override (`target_language`) that does NOT overwrite query `detected_language`
  - Response metadata fields for audit/debug
- Frontend:
  - `GuidedSelectionPanel` above chat input (no new modal for MVP)
  - Numeric input (`1..N`, `0`) and click selection support
  - Kind-aware resume payload behavior (do not force HIL flags on guided resume)
- Tests:
  - Backend API test proving interrupt/resume works and scoping is enforced
  - Frontend tests proving numeric step progression and correct payload formation

### Definition of Done (verifiable)
- [ ] New interrupt `auto_parse_confirm` is returned when enabled and includes option lists for all 4 steps.
- [ ] Resume applies user selections and completes retrieval/answer without falling into `retrieval_review`/`device_selection` unless explicitly enabled.
- [ ] `task_mode=sop` results in `selected_doc_types_strict=true` and doc_type scope limited to SOP variants.
- [ ] `task_mode=issue` results in doc_type scope limited to `myservice/gcb/ts` variants.
- [ ] `task_mode=all` applies no doc_type filter.
- [ ] Answer language selection affects answer prompt template selection without changing query-language detection/translation behavior.
- [ ] Backend + frontend test suites pass.

### Must Have
- Backward compatible defaults: existing clients see no new interrupt unless they opt in.
- Resume correctness: guided resume uses the same compiled graph/checkpointer that created the checkpoint.
- Explicit audit metadata:
  - `metadata.selected_task_mode`
  - `metadata.applied_doc_type_scope`
  - `metadata.selected_language_source`

### Must NOT Have
- Must NOT overload `detected_language` to represent answer language.
- Must NOT change retrieval ranking algorithms (this is scope control, not scoring changes).
- Must NOT require multiple backend interrupt hops for MVP (single interrupt payload; FE sequences steps locally).
- Must NOT break existing HIL interrupts/resume semantics (`retrieval_review`, `device_selection`, `human_review`).

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Backend: `pytest` (unit + API TestClient) with dependency overrides (fake SearchService + fake LLM).
- Frontend: `npm -C frontend test` (Vitest + RTL) for guided panel flow and payload.
- QA scenarios: scripted steps using API calls + UI tests; capture evidence under `.sisyphus/evidence/`.

## Execution Strategy
### Parallel Execution Waves
Wave 1 (Backend contract + graph resume foundations)
- T1 Add request/response contract fields + metadata scaffolding (backend + frontend types)
- T2 Implement guided agent variant + resume routing (same graph/checkpointer)
- T3 Implement `auto_parse_confirm` LangGraph node (payload + resume merge)
- T4 Implement task_mode→doc_type strict scope enforcement
- T5 Implement answer `target_language` override (template selection)

Wave 2 (Frontend guided panel + resume behavior)
- T6 Add `auto_parse_confirm` interrupt kind handling + pending state
- T7 Implement `GuidedSelectionPanel` (click + numeric)
- T8 Integrate panel into ChatPage and useChatSession (numeric intercept, resume payload)

Wave 3 (Tests + QA + docs)
- T9 Backend API tests for interrupt/resume + scoping + language override
- T10 Frontend tests for step progression + payload formation
- T11 End-to-end QA script run (local API) + evidence
- T12 Docs update / runbook notes (minimal)

### Dependency Matrix
- T2 blocks T3 (graph must support resume routing)
- T3 blocks T7/T8 (FE needs payload schema)
- T4 blocks T9/T11 (must validate scoping)
- T5 blocks T9/T11 (must validate language override)

### Agent Dispatch Summary
- Wave 1: backend-heavy (`deep`, `unspecified-high`)
- Wave 2: frontend-heavy (`visual-engineering`)
- Wave 3: test/QA (`deep`, `playwright` if doing browser E2E)

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task includes QA scenarios with concrete evidence outputs.

- [x] 1. Backend/Frontend Contract: Guided Flow Flag + Payload Schemas

  **What to do**:
  - Backend request contract (opt-in, default OFF):
    - Add `guided_confirm: bool = False` to `backend/api/routers/agent.py` `AgentRequest`.
    - Ensure `guided_confirm` is passed into the graph state for BOTH JSON and SSE endpoints:
      - `backend/api/routers/agent.py` (`chat_state` for auto-parse branch, `state_overrides` for override branch).
    - Ensure `guided_confirm` mode resets per new question (decision-complete):
      - For any non-resume call in guided flow, set `chat_state["auto_parse_confirmed"] = False` so the confirm node interrupts once per user question.
  - Interrupt payload schema (decision-complete; emitted by backend node in Task 3):
    - `type: "auto_parse_confirm"`
    - `question: <original user query>`
    - `instruction: <Korean instruction for numeric/click steps>`
    - `steps: ["language","device","equip_id","task"]`
    - `options.language: [{value,label,recommended}]` values in `{ko,en,zh,ja}`
    - `options.device: [{value,label,recommended,doc_count?}]` values are device names plus a sentinel `__skip__`
    - `options.equip_id: [{value,label,recommended}]` values are equip_id strings plus sentinels `__skip__`,`__manual__`
    - `options.task: [{value,label,recommended}]` values in `{sop,issue,all}`
    - `defaults: {target_language, device, equip_id, task_mode}` (values or null)
  - Resume decision schema (decision-complete; FE sends this object as `resume_decision`):
    - `{type:"auto_parse_confirm", target_language:"ko|en|zh|ja", selected_device?:string|null, selected_equip_id?:string|null, task_mode:"sop|issue|all"}`
  - Frontend types:
    - Extend `frontend/src/features/chat/types.ts` `AgentRequest` with `guided_confirm?: boolean`.
    - Extend `frontend/src/features/chat/types.ts` `AgentRequest` with `filter_equip_ids?: string[] | null` (needed for equip_id selection; backend already supports `filter_equip_ids`).
    - Add frontend union type `TaskMode = "sop" | "issue" | "all"` (guided flow only).

  **Must NOT do**:
  - Do not change existing interrupt payload shapes (`retrieval_review`, `device_selection`, `human_review`).
  - Do not make guided confirm the backend default (must remain opt-in via `guided_confirm`).

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: API contract touches both FE + BE
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 2-3,6-8 | Blocked By: none

  **References**:
  - Request model: `backend/api/routers/agent.py`
  - FE types: `frontend/src/features/chat/types.ts`

  **Acceptance Criteria**:
  - [ ] `guided_confirm` exists in backend `AgentRequest` and FE `AgentRequest` types.
  - [ ] A documented interrupt payload + resume decision schema exists in this plan and is implemented verbatim.

  **QA Scenarios**:
  ```
  Scenario: Contract fields present
    Tool: Bash
    Steps:
      python -c "import inspect; import backend.api.routers.agent as a; print('guided_confirm' in a.AgentRequest.model_fields)"
    Expected: prints True
    Evidence: .sisyphus/evidence/ui-guided-selection/task-1-contract.txt
  ```

  **Commit**: YES | Message: `feat(api): add guided_confirm contract fields` | Files: `backend/api/routers/agent.py`, `frontend/src/features/chat/types.ts`

- [x] 2. Backend: Resume Routing for `auto_parse_confirm` (Same Graph/Checkpointer)

  **What to do**:
  - Update both endpoints to route resumes by `resume_decision.type`:
    - JSON: `backend/api/routers/agent.py` `run_agent`
    - SSE: `backend/api/routers/agent.py` `run_agent_stream`
  - Decision-complete routing rules:
    1) If `is_resume` and `resume_decision` is a dict with `type == "auto_parse_confirm"`:
       - Validate/sanitize the resume payload BEFORE resuming the graph (so errors become clean 4xx, not 5xx):
         - Add a small Pydantic model in `backend/api/routers/agent.py` (decision-complete name): `AutoParseConfirmDecision`.
         - If invalid: return HTTP 400 with a message that includes which field is invalid.
      - Instantiate the SAME auto-parse/guided graph family used for the first call:
        - Add a new factory in `backend/api/routers/agent.py` (decision-complete name): `_new_guided_confirm_agent`.
        - It MUST pass `checkpointer=_checkpointer` so the checkpoint survives across requests.
        - It MUST set `auto_parse_enabled=True` and MUST NOT enable other interrupts (`ask_user_after_retrieve=False`, `ask_device_selection=False`).
        - It MUST pass `device_fetcher=_create_device_fetcher(search_service)` so the confirm payload can include real device candidates.
      - Resume with `agent._graph.invoke(Command(resume=req.resume_decision), config={"configurable": {"thread_id": tid}})`.
       - Before invoking, validate the checkpoint exists:
         - `state = agent._graph.get_state(config)`
         - If missing/empty, return HTTP 400 with the existing message style:
           - `No checkpoint for thread_id=... Server may have restarted.`
    2) Else (existing resume types): keep current behavior (use `_new_hil_agent` and existing logic).

  - Initial call routing (decision-complete; required for resume to work):
    - In both endpoints, when NOT resuming:
      - If `req.auto_parse is True` AND `req.guided_confirm is True` AND overrides are absent:
        - Use `_new_guided_confirm_agent` (NOT `_new_auto_parse_agent`).
      - Else keep current behavior.
  - Add explicit handling for process-local MemorySaver limitation:
    - Keep current 400 behavior (resume after restart fails clearly).

  **Must NOT do**:
  - Do not route all resumes through `_new_hil_agent` (breaks guided resume correctness).
  - Do not change `_checkpointer` implementation in this task (scope boundary).

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: resume correctness + regression risk
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 3,9 | Blocked By: 1

  **References**:
  - Current resume path: `backend/api/routers/agent.py`
  - Existing resume regression test pattern: `tests/api/test_agent_interrupt_resume_regression.py`

  **Acceptance Criteria**:
  - [ ] Resuming `auto_parse_confirm` does NOT instantiate `_new_hil_agent`.
  - [ ] Missing checkpoint returns HTTP 400 with clear error.

  **QA Scenarios**:
  ```
  Scenario: Resume routing is kind-aware
    Tool: Bash
    Steps: pytest -q tests/api/test_agent_interrupt_resume_regression.py
    Expected: pass
    Evidence: .sisyphus/evidence/ui-guided-selection/task-2-resume-routing.txt
  ```

  **Commit**: YES | Message: `fix(api): route auto_parse_confirm resume to auto-parse graph` | Files: `backend/api/routers/agent.py`

- [x] 3. Backend: Implement `auto_parse_confirm` Interrupt Node (One Interrupt, One Resume)

  **What to do**:
  - Add new LangGraph node in `backend/llm_infrastructure/llm/langgraph_agent.py`:
    - Function name (decision-complete): `auto_parse_confirm_node`
    - Signature: `def auto_parse_confirm_node(state: AgentState, *, device_fetcher: Any | None = None) -> Command[Literal["history_check"]]`
  - Insert node into the auto-parse enabled graph:
    - File: `backend/services/agents/langgraph_rag_agent.py`
    - In `LangGraphRAGAgent._build_graph` when `self.auto_parse_enabled`:
      - `builder.add_node("auto_parse_confirm", self._wrap_node("auto_parse_confirm", functools.partial(auto_parse_confirm_node, device_fetcher=self.device_fetcher)))`
      - Replace edge: `auto_parse -> history_check` with `auto_parse -> auto_parse_confirm -> history_check`.
  - Ensure the guided confirm node can show real device candidates (decision-complete):
    - Update `_new_guided_confirm_agent` in `backend/api/routers/agent.py` to pass `device_fetcher=_create_device_fetcher(search_service)`.
    - This fetcher MUST be called only inside `auto_parse_confirm_node` (when `guided_confirm=true`), not on every request.
  - Interrupt behavior (decision-complete):
    - Only interrupt when `state.get("guided_confirm") is True` AND `state.get("auto_parse_confirmed") is not True`.
    - `interrupt(payload)` MUST be the last statement (payload built only from already-persisted state).
    - On non-resume invocation, `auto_parse_confirmed` must be forced to `False` by API state_overrides (Task 1) so the node interrupts once per new question.
    - On resume, always set `auto_parse_confirmed=True` to prevent double-resume re-interrupt loops.
  - Payload building (decision-complete):
    - Use auto-parse outputs from `state.parsed_query` (preferred) + fallbacks (`auto_parsed_*`).
    - Language recommend: `detected_language` if in `{ko,en,zh,ja}` else `ko`.
    - Task recommend: always `issue`.
    - Device options: include recommend (if present), plus up to 8 devices from `device_fetcher()` if available, plus `__skip__`.
    - Equip_id options: include recommend (if present), plus `__skip__`, plus `__manual__`.
  - Resume merge rules (decision-complete):
    - Accept `resume_decision` dict only when `type=="auto_parse_confirm"`.
    - If `selected_device` is `null` or `__skip__`: clear device filter.
    - If `selected_equip_id` is `null` or `__skip__`: clear equip_id filter.
    - If `selected_equip_id` is present: normalize to uppercase; if invalid equip_id format, treat as skip.
    - Always set `auto_parse_confirmed=True`.
    - Set `target_language` (answer language) separate from `detected_language`.
    - Apply task_mode scoping via Task 4 mapping.
    - Return `Command(goto="history_check", update=<merged state>)`.

  **Must NOT do**:
  - Do not add multiple backend interrupts (language then device then equip_id then task). One payload only.
  - Do not change existing auto_parse extraction logic.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: graph node correctness + interrupt contract
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 9 | Blocked By: 2

  **References**:
  - Existing interrupt patterns: `backend/llm_infrastructure/llm/langgraph_agent.py` (`ask_user_after_retrieve_node`, `device_selection_node`)
  - Graph build site: `backend/services/agents/langgraph_rag_agent.py`
  - Device catalog fetcher: `backend/api/routers/agent.py` (`_create_device_fetcher`)

  **Acceptance Criteria**:
  - [ ] When `guided_confirm=true`, first `/api/agent/run` returns `interrupted=true` with `interrupt_payload.type == "auto_parse_confirm"`.
  - [ ] On resume, graph continues to completion and `interrupted=false`.

  **QA Scenarios**:
  ```
  Scenario: auto_parse_confirm interrupt appears only when enabled
    Tool: Bash
    Steps: pytest -q tests/api/test_agent_autoparse_confirm_interrupt_resume.py -k interrupt_gate
    Expected: pass
    Evidence: .sisyphus/evidence/ui-guided-selection/task-3-confirm-node-gate.txt
  ```

  **Commit**: YES | Message: `feat(agent): add auto_parse_confirm interrupt node` | Files: `backend/llm_infrastructure/llm/langgraph_agent.py`, `backend/services/agents/langgraph_rag_agent.py`

- [x] 4. Backend: Task Mode → Doc-Type Strict Scope Enforcement (+ Metadata)

  **What to do**:
- Implement mapping inside `auto_parse_confirm_node` (authoritative):
    - `task_mode="sop"` => set `selected_doc_types = expand_doc_type_selection(["sop"])` and `selected_doc_types_strict=True`
    - `task_mode="issue"` => set `selected_doc_types = expand_doc_type_selection(["myservice","gcb","ts"])` and `selected_doc_types_strict=True`
    - `task_mode="all"` => clear doc type filter (`selected_doc_types=[]`, `selected_doc_types_strict=False`)
  - Persist to BOTH locations for compatibility:
    - `state["parsed_query"]["selected_doc_types"]` + `state["parsed_query"]["doc_types_strict"]`
    - Top-level `state["selected_doc_types"]` + `state["selected_doc_types_strict"]`
  - Add response metadata so evaluator/debug can see what happened:
    - File: `backend/api/routers/agent.py` `_build_response_metadata`
    - Add keys:
      - `metadata.selected_task_mode = <task_mode or null>`
      - `metadata.applied_doc_type_scope = <list[str]>` (the expanded list)
      - `metadata.scope_by_task_override = <bool>` (true when task_mode != all)
  - Conflict/precedence rule (decision-complete):
    - If a request/resume also provides `filter_doc_types` (or preexisting `selected_doc_types_strict=true`), `task_mode` wins and overwrites doc_type scope.
    - When an overwrite happens, set `metadata.scope_by_task_override=true`.

  **Must NOT do**:
  - Do not add new doc_type group logic; reuse `backend/domain/doc_type_mapping.py`.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: cross-cutting state + metadata contract
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 9,11 | Blocked By: 3

  **References**:
  - Doc type expansion: `backend/domain/doc_type_mapping.py`
  - Retrieval filter usage: `backend/llm_infrastructure/llm/langgraph_agent.py` (`retrieve_node` uses `selected_doc_types_strict`)

  **Acceptance Criteria**:
  - [ ] For `task_mode=sop`, retriever calls receive `doc_types` containing SOP variants only.
  - [ ] For `task_mode=issue`, retriever calls receive `doc_types` containing only myservice/gcb/ts variants.
  - [ ] For `task_mode=all`, retriever calls receive `doc_types` as empty/None (no scoping).

  **QA Scenarios**:
  ```
  Scenario: task_mode forces doc_types scope
    Tool: Bash
    Steps: pytest -q tests/api/test_agent_autoparse_confirm_interrupt_resume.py -k task_mode_scope
    Expected: pass
    Evidence: .sisyphus/evidence/ui-guided-selection/task-4-task-scope.txt
  ```

  **Commit**: YES | Message: `feat(retrieval): enforce task_mode doc_type scope` | Files: `backend/llm_infrastructure/llm/langgraph_agent.py`, `backend/api/routers/agent.py`, `backend/domain/doc_type_mapping.py`

- [x] 5. Backend: Answer Language Override (`target_language`)

  **What to do**:
  - Store `target_language` in state from `auto_parse_confirm_node`.
  - Update `backend/llm_infrastructure/llm/langgraph_agent.py` `answer_node`:
    - Determine `answer_language = state.get("target_language") or state.get("detected_language") or "ko"`.
    - Use `answer_language` to select templates (ko/en/zh/ja) exactly as current logic does for `detected_language`.
    - Keep translate/routing behavior unchanged (still uses `detected_language` / query_en/query_ko).
  - Add response metadata:
    - `metadata.target_language = <answer_language>`
    - `metadata.selected_language_source = "user"|"auto_parse"|"default"` (decision-complete rules:
      - user if guided_confirm resume set it
      - auto_parse if derived from detected_language
      - default if fallback to ko)

  **Must NOT do**:
  - Do not overwrite `detected_language` with `target_language`.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: subtle language semantics + regressions
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 9,11 | Blocked By: 3

  **References**:
  - Answer templates test: `backend/tests/test_answer_language_templates.py`
  - Answer node: `backend/llm_infrastructure/llm/langgraph_agent.py` (`answer_node`)

  **Acceptance Criteria**:
  - [ ] `target_language` changes answer template selection while leaving translation/routing logic intact.

  **QA Scenarios**:
  ```
  Scenario: target_language affects answer templates
    Tool: Bash
    Steps: pytest -q backend/tests/test_answer_language_templates.py
    Expected: pass
    Evidence: .sisyphus/evidence/ui-guided-selection/task-5-target-language.txt
  ```

  **Commit**: YES | Message: `feat(agent): support target_language for answer generation` | Files: `backend/llm_infrastructure/llm/langgraph_agent.py`, `backend/api/routers/agent.py`

- [x] 6. Frontend: Add `auto_parse_confirm` Interrupt Kind + Pending Guided State

  **What to do**:
  - Extend interrupt kind union and resolver:
    - File: `frontend/src/features/chat/hooks/use-chat-session.ts`
    - Add kind: `"auto_parse_confirm"`
    - Update `resolveInterruptKind()` to map `payload.type === "auto_parse_confirm"`.
  - Add guided selection state (decision-complete):
    - Store `pendingGuidedSelection` separately from `pendingInterrupt` OR extend `pendingInterrupt` to carry options.
    - Expose from hook: `pendingGuidedSelection`, `submitGuidedSelectionNumber`, `submitGuidedSelectionFinal`.

  **Must NOT do**:
  - Do not break existing pendingReview/pendingDeviceSelection behavior.

  **Recommended Agent Profile**:
  - Category: `visual-engineering` — Reason: state + UX wiring
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 7-8,10 | Blocked By: 1,3

  **References**:
  - Hook: `frontend/src/features/chat/hooks/use-chat-session.ts`
  - Existing panel render patterns: `frontend/src/features/chat/pages/chat-page.tsx`

  **Acceptance Criteria**:
  - [ ] `auto_parse_confirm` interrupt yields a visible guided panel state.

  **QA Scenarios**:
  ```
  Scenario: auto_parse_confirm kind is recognized
    Tool: Bash
    Steps: npm -C frontend test
    Expected: guided kind unit tests pass
    Evidence: .sisyphus/evidence/ui-guided-selection/task-6-fe-kind.txt
  ```

  **Commit**: YES | Message: `feat(ui): handle auto_parse_confirm interrupt kind` | Files: `frontend/src/features/chat/hooks/use-chat-session.ts`, `frontend/src/features/chat/types.ts`

- [x] 7. Frontend: Implement `GuidedSelectionPanel` Component (Click + Manual Equip ID)

  **What to do**:
  - New component:
    - `frontend/src/features/chat/components/guided-selection-panel.tsx`
  - Decision-complete UX rules:
    - Render above input (same position as `DeviceSelectionPanel`).
    - Show step header + question + option list.
    - Option 1 is recommended (if any) and shows `(Recommend)`.
    - Provide buttons for each option; clicking advances step.
    - Equip ID step:
      - If user selects `__manual__`, show an inline text input + confirm button.
      - If user selects `__skip__`, clear equip id and advance.
    - Task step shows `SOP 조회`, `이슈조회`, `전체조회`.
    - Display a compact summary row of chosen selections.
  - Export component in `frontend/src/features/chat/components/index.ts` (or existing barrel export).

  **Must NOT do**:
  - No new modal for MVP.
  - Do not require user to type free-text in the chat input for equip_id.

  **Recommended Agent Profile**:
  - Category: `visual-engineering`
  - Skills: [`frontend-ui-ux`] (optional)

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 8,10 | Blocked By: 6

  **References**:
  - Existing panel style: `frontend/src/features/chat/components/device-selection-panel.tsx`

  **Acceptance Criteria**:
  - [ ] Panel supports click-based completion of all 4 steps.
  - [ ] Manual equip_id input path works.

  **QA Scenarios**:
  ```
  Scenario: GuidedSelectionPanel renders and progresses by click
    Tool: Bash
    Steps: npm -C frontend test
    Expected: guided panel tests pass
    Evidence: .sisyphus/evidence/ui-guided-selection/task-7-panel.txt
  ```

  **Commit**: YES | Message: `feat(ui): add GuidedSelectionPanel` | Files: `frontend/src/features/chat/components/guided-selection-panel.tsx`

- [x] 8. Frontend: Numeric Input Intercept + Resume Payload + Rendering Integration

  **What to do**:
  - Render panel:
    - File: `frontend/src/features/chat/pages/chat-page.tsx`
    - Add render block above `<InputArea>` similar to `DeviceSelectionPanel`.
  - Numeric intercept rules in `ChatPage.handleSend` (decision-complete):
    - If `pendingGuidedSelection` exists:
      - Accept tokens: `0` (use recommend or skip), `1..N` (choose option index)
      - On valid token: call hook handler (does NOT send message)
      - On invalid token: ignore (no send, no state change)
      - Return early.
  - When guided selection completes:
    - Hook sends ONE resume call via `send({ text: <summary>, decisionOverride: {type:"auto_parse_confirm", ...} })`.
    - Ensure resume payload does NOT force `ask_user_after_retrieve=true` for this kind.
  - Initial request:
    - When sending a new user query (not resuming), include `guided_confirm: true` in payload so backend returns `auto_parse_confirm` interrupt.

  **Must NOT do**:
  - Do not interpret numeric input as `resolveDecision()` approve/reject text for this guided flow.

  **Recommended Agent Profile**:
  - Category: `visual-engineering`
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 10 | Blocked By: 6-7

  **References**:
  - Existing numeric intercept example: `frontend/src/features/chat/pages/chat-page.tsx` (missing_device_parse flow)
  - Resume send path: `frontend/src/features/chat/hooks/use-chat-session.ts`

  **Acceptance Criteria**:
  - [ ] Typing `1`, `2`, ... progresses guided steps without sending a chat message.
  - [ ] Final resume call includes structured `resume_decision.type == "auto_parse_confirm"`.
  - [ ] Initial call includes `guided_confirm: true`.

  **QA Scenarios**:
  ```
  Scenario: Numeric-guided flow triggers a single resume payload
    Tool: Bash
    Steps: npm -C frontend test
    Expected: payload tests pass
    Evidence: .sisyphus/evidence/ui-guided-selection/task-8-numeric.txt
  ```

  **Commit**: YES | Message: `feat(ui): wire guided numeric resume flow` | Files: `frontend/src/features/chat/pages/chat-page.tsx`, `frontend/src/features/chat/hooks/use-chat-session.ts`

- [x] 9. Backend Tests: Guided Interrupt/Resume + Task Scope + Language (API-level)

  **What to do**:
  - Add new API test file:
    - `tests/api/test_agent_autoparse_confirm_interrupt_resume.py`
  - Test setup (decision-complete):
    - Override `dependencies.get_default_llm` with a stub `BaseLLM` returning `LLMResponse(text="general")`.
    - Override `dependencies.get_search_service` with a fake SearchService whose `search()` records arguments and returns predictable RetrievalResults with metadata including `doc_type`, `device_name`, `equip_id`, `page`.
    - Force request `mq_mode="off"` to avoid MQ nodes.
  - Required test cases (decision-complete):
    1) `interrupt_gate`: when `guided_confirm=false`, response is not interrupted.
    2) `interrupt_when_enabled`: when `guided_confirm=true`, response is interrupted with payload.type `auto_parse_confirm`.
    3) `resume_applies_task_mode_scope`: resume with `task_mode=sop` causes search() to receive `doc_types` SOP-only; resume with `task_mode=issue` scopes to myservice/gcb/ts.
    4) `target_language_metadata`: resume with `target_language=en` makes response metadata show `target_language=en`.
    5) `missing_checkpoint_400`: resume with unknown thread_id returns 400.

  **Must NOT do**:
  - Do not rely on ES/vLLM; tests must run with fakes.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: API resume + graph behavior
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 11 | Blocked By: 2-5

  **References**:
  - Existing resume regression: `tests/api/test_agent_interrupt_resume_regression.py`

  **Acceptance Criteria**:
  - [ ] `pytest -q tests/api/test_agent_autoparse_confirm_interrupt_resume.py` passes.

  **QA Scenarios**:
  ```
  Scenario: Guided interrupt/resume works with strict scoping
    Tool: Bash
    Steps: pytest -q tests/api/test_agent_autoparse_confirm_interrupt_resume.py
    Expected: pass
    Evidence: .sisyphus/evidence/ui-guided-selection/task-9-backend-tests.txt
  ```

  **Commit**: YES | Message: `test(api): cover auto_parse_confirm guided resume` | Files: `tests/api/test_agent_autoparse_confirm_interrupt_resume.py`

- [x] 10. Frontend Tests: Guided Panel Flow + Payload Formation

  **What to do**:
  - Add tests under `frontend/src/features/chat/__tests__/`:
    - `guided-selection-panel.test.tsx` for panel step progression by click.
    - Extend `chat-request-payload.test.tsx` to assert `guided_confirm: true` is sent on first call when feature enabled.
  - Decision-complete assertions:
    - Numeric handler selects option N and advances step.
    - On completion, hook calls API with `resume_decision: {type:"auto_parse_confirm", ...}`.

  **Recommended Agent Profile**:
  - Category: `visual-engineering`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 11 | Blocked By: 6-8

  **References**:
  - Existing payload tests: `frontend/src/features/chat/__tests__/chat-request-payload.test.tsx`
  - Existing numeric intercept tests: `frontend/src/features/chat/__tests__/chat-page-device-panel.test.tsx`

  **Acceptance Criteria**:
  - [ ] `npm -C frontend test` passes.

  **QA Scenarios**:
  ```
  Scenario: guided panel tests pass
    Tool: Bash
    Steps: npm -C frontend test
    Expected: pass
    Evidence: .sisyphus/evidence/ui-guided-selection/task-10-frontend-tests.txt
  ```

  **Commit**: YES | Message: `test(ui): add guided selection coverage` | Files: `frontend/src/features/chat/__tests__/...`

- [x] 11. QA: End-to-End Guided Flow Evidence (Local)

  **What to do**:
  - Run API locally (no need for vLLM if using HTTP path? For QA, use TestClient mode or stub LLM via env is not available; use docker vLLM if needed).
  - Produce evidence showing:
    - First call returns `auto_parse_confirm` interrupt.
    - Resume call returns final answer and includes metadata for `selected_task_mode`, `applied_doc_type_scope`, `target_language`.

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: none | Blocked By: 9-10

  **Acceptance Criteria**:
  - [ ] Evidence file exists with representative request/response JSON.

  **QA Scenarios**:
  ```
  Scenario: Manual QA via curl
    Tool: Bash
    Steps:
      curl -s -X POST http://127.0.0.1:8001/api/agent/run -H 'Content-Type: application/json' \
        -d '{"message":"SOP에서 SUPRA N power cal 절차?","guided_confirm":true,"auto_parse":true,"mq_mode":"off"}'
    Expected: interrupted=true and interrupt_payload.type=auto_parse_confirm
    Evidence: .sisyphus/evidence/ui-guided-selection/task-11-e2e.txt
  ```

  **Commit**: NO

- [x] 12. Docs: Update `docs/2026-03-05-ui_개선.md` With Final Contract + Runbook

  **What to do**:
  - Update to reflect the implemented contract:
    - `guided_confirm` opt-in behavior
    - Interrupt payload schema
    - Resume decision schema
    - Task_mode mapping is authoritative and how it maps to doc_type scope (mention `expand_doc_type_selection`).
    - Note: resume requires same-process MemorySaver; after restart resumes return 400.

  **Recommended Agent Profile**:
  - Category: `writing`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: none | Blocked By: 1-5

  **Acceptance Criteria**:
  - [ ] Doc contains the final JSON schema examples for interrupt + resume.

  **QA Scenarios**:
  ```
  Scenario: docs mention guided_confirm + task scoping
    Tool: Bash
    Steps: grep -n "guided_confirm\|auto_parse_confirm\|task_mode\|applied_doc_type_scope" -n docs/2026-03-05-ui_개선.md
    Expected: matches found
    Evidence: .sisyphus/evidence/ui-guided-selection/task-12-docs-grep.txt
  ```

  **Commit**: YES | Message: `docs(ui): document auto_parse_confirm guided flow` | Files: `docs/2026-03-05-ui_개선.md`

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [x] F1. Plan Compliance Audit — oracle
- [x] F2. Code Quality Review — unspecified-high
- [x] F3. Real QA Simulation — unspecified-high (+ playwright if UI)
- [x] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit 1: `feat(api): add auto_parse_confirm guided flow contract`
- Commit 2: `feat(agent): add auto_parse_confirm interrupt and task_mode scoping`
- Commit 3: `feat(ui): guided selection panel with numeric resume`
- Commit 4: `test: add guided flow regression coverage`

## Success Criteria
- Guided selection works end-to-end with one interrupt + one resume.
- SOP vs issue scoping is enforced deterministically and visible via metadata.
- Existing interrupt flows remain unchanged unless explicitly enabled.
