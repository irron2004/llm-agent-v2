# Issue Flow Fix: Summary-First + Confirm Gates + Loop

## TL;DR
> **Summary**: `task_mode=issue`에서 답변 없이 곧바로 사례 선택 interrupt가 뜨는 문제를 고치고, 문서에 정의된 "요약 답변 → yes/no → 선택 → 상세 → SOP yes/no → 다른 이슈 루프" 플로우로 재배선한다.
> **Deliverables**:
> - Backend LangGraph issue 플로우 재배선 + 새 guided interrupt `issue_confirm`
> - API resume_decision validation에 `issue_confirm` 추가
> - Frontend: interrupted 응답에서도 `answer` 유지 + `issue_confirm` 패널 추가 + 기존 issue 패널 흐름 업데이트
> - Backend/Frontend 테스트 추가·수정
> **Effort**: Medium
> **Parallel**: YES - 2 waves
> **Critical Path**: Backend graph → API validation → Frontend hook/panel → Tests

## Context
### Original Request
- `docs/2026-03-11-개선작업.md`의 `3. 이슈 플로우 수정`을 구현하기 위한 작업 계획 수립.

### Interview Summary
- 추가 질문 없이 문서에 명시된 플로우/변경 대상 파일을 그대로 구현한다.

### Metis Review (gaps addressed)
- Frontend가 interrupt 시 `res.answer`를 덮어쓰는 문제가 있어, 백엔드만 바꿔도 UX가 개선되지 않는 점을 계획에 반영했다.
- SOP 답변 노드가 `docs/display_docs`를 SOP로 덮어써 루프에서 케이스 목록이 흔들릴 수 있어, 원본 issue docs를 별도 상태로 보존하는 결정을 포함했다.
- `AGENT_GRAPH_VERSION` 상수가 2곳에 중복되어 있어 동시 bump를 계획에 포함했다.

## Work Objectives
### Core Objective
- `task_mode=issue`에서 "요약 답변을 먼저 보여주고" 사용자에게 다음 단계 진입 여부를 묻는 UX로 전환한다.

### Deliverables
- Backend graph changes:
  - `expand_related` 이후 `issue_step1_prepare` → `answer` → `issue_confirm(post_summary)` 순서
  - 상세 답변/선택 후 `issue_confirm(post_detail)`로 루프 제공
- New guided interrupt contract: `issue_confirm`
- API: `resume_decision` validation + nonce/graph_version guard 확장
- Frontend: `issue_confirm` 패널 + interrupt 응답에서도 `answer` 유지
- Tests:
  - Backend: issue interrupt 순서/loop를 검증하는 LangGraph 테스트
  - Frontend: hook 기반 issue-flow e2e 테스트 시나리오 업데이트 + request payload 테스트 추가

### Definition of Done (verifiable)
- Backend:
  - `cd backend && uv run pytest tests/ -q` 통과
  - 새 테스트 파일이 `issue_confirm → case_selection → sop_confirm → post_detail_confirm → case_selection(루프)` 순서를 검증
- Frontend:
  - `cd frontend && npm run test` 통과
  - `frontend/src/features/chat/__tests__/issue-flow-ui.test.tsx`가 새 interrupt 단계를 포함
  - `frontend/src/features/chat/__tests__/chat-request-payload.test.tsx`에 `issue_confirm` resume payload 검증 추가
  - `cd frontend && npm run build` 통과

### Must Have
- 요약 답변(`answer_node`)이 사용자에게 보이는 상태에서 yes/no 패널이 뜬다.
- 상세 답변(`issue_step2_prepare_detail_node` 결과)이 사용자에게 보이는 상태에서 SOP yes/no 패널이 뜬다.
- SOP 답변 후 또는 SOP 스킵 후 "다른 이슈도 볼까요?" yes/no 패널이 뜨고, yes 선택 시 동일 검색 결과(case list)로 되돌아간다.

### Must NOT Have
- retrieval/rerank 로직 변경(검색 품질 튜닝) 포함 금지
- 새 UI 디자인 시스템 도입/대규모 리팩터링 금지
- interrupt 타입을 free-text(사용자 입력 파싱)로 처리(반드시 guided decision + nonce)

## Verification Strategy
- Test decision: tests-after (Backend PyTest + Frontend Vitest)
- QA policy: 모든 TODO에 최소 2개 시나리오(정상/엣지) 포함
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.txt` (테스트 로그/핵심 assertion 캡처)

## Execution Strategy
### Parallel Execution Waves
Wave 1 (Backend contract + graph)
- Backend graph rewiring + new interrupt nodes
- API router guided decision validation 확장
- Backend unit tests

Wave 2 (Frontend UX + tests)
- Frontend hook: interrupted answer 보존 + 새 pending state
- Frontend panel + 페이지 렌더링
- Frontend tests 업데이트

### Dependency Matrix (high level)
- Backend `issue_confirm` contract 정의가 FE/API 작업을 블로킹
- FE에서 `res.answer` 보존 로직이 없으면 플로우 요구사항(요약/상세 답변 먼저 표시)을 만족할 수 없음

## TODOs

- [ ] 1. Define `issue_confirm` Guided Interrupt Contract (Backend+Frontend)

  **What to do**:
  - Contract를 단일 소스로 문서화(코드 상수/타입)한다.
  - Payload (interrupt_payload)와 resume_decision의 필드를 고정한다.
  - Stage를 명시하여 재사용(요약 이후 vs 상세 이후)을 구분한다.

  **Must NOT do**:
  - 기존 `issue_case_selection`/`issue_sop_confirm` payload 필드 변경

  **Recommended Agent Profile**:
  - Category: `unspecified-low` — Reason: 타입/상수 정의 중심
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 2-7 | Blocked By: none

  **References**:
  - Doc spec: `docs/2026-03-11-개선작업.md:79`
  - Existing payload shapes: `backend/llm_infrastructure/llm/langgraph_agent.py:3646`, `backend/llm_infrastructure/llm/langgraph_agent.py:3747`
  - FE kind resolution: `frontend/src/features/chat/hooks/use-chat-session.ts:175`
  - FE issue flow tests: `frontend/src/features/chat/__tests__/issue-flow-ui.test.tsx:52`

  **Acceptance Criteria**:
  - [ ] Payload fields are fixed to:
    - interrupt_payload: `{ type: "issue_confirm", nonce: string, stage: "post_summary"|"post_detail", question: string, instruction: string, prompt: string }`
    - resume_decision: `{ type: "issue_confirm", nonce: string, stage: "post_summary"|"post_detail", confirm: boolean }`

  **QA Scenarios**:
  ```
  Scenario: Contract compatibility check
    Tool: Bash
    Steps: run backend+frontend unit tests after implementation
    Expected: tests asserting resume payload shape pass
    Evidence: .sisyphus/evidence/task-1-issue-confirm-contract.txt

  Scenario: Stage mismatch handling
    Tool: Bash
    Steps: backend API should reject invalid stage value via Pydantic validation
    Expected: HTTP 400 with "Invalid ... resume_decision" message
    Evidence: .sisyphus/evidence/task-1-issue-confirm-contract-error.txt
  ```

  **Commit**: YES | Message: `fix(issue-flow): define issue_confirm interrupt contract` | Files: `backend/api/routers/agent.py`, `frontend/src/features/chat/types.ts`


- [ ] 2. Backend: Bump `AGENT_GRAPH_VERSION` Consistently

  **What to do**:
  - 아래 두 파일의 `AGENT_GRAPH_VERSION`을 동일 값(예: `2026-03-11`)으로 bump.
  - 그래프 상태에 저장되는 `graph_version`과 API guard 비교가 일치하도록 확인.

  **Must NOT do**:
  - 한쪽만 변경해서 resume guard가 409를 유발하는 상태로 남기지 않기

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 3-5 | Blocked By: none

  **References**:
  - `backend/llm_infrastructure/llm/langgraph_agent.py:36`
  - `backend/api/routers/agent.py:39`
  - Guard logic: `backend/api/routers/agent.py:472`

  **Acceptance Criteria**:
  - [ ] Both constants match exactly

  **QA Scenarios**:
  ```
  Scenario: Resume guard sanity
    Tool: Bash
    Steps: run backend tests touching resume guard if present; otherwise run full backend test suite
    Expected: no failures related to graph_version mismatch
    Evidence: .sisyphus/evidence/task-2-graph-version.txt

  Scenario: Old checkpoint rejection (smoke)
    Tool: Bash
    Steps: simulate resume with a checkpoint containing old graph_version (unit test-level mock)
    Expected: HTTP 409 raised by _validate_guided_resume_checkpoint
    Evidence: .sisyphus/evidence/task-2-graph-version-409.txt
  ```

  **Commit**: YES | Message: `chore(agent): bump graph version for issue flow redesign` | Files: `backend/llm_infrastructure/llm/langgraph_agent.py`, `backend/api/routers/agent.py`


- [ ] 3. Backend: Add `issue_confirm` Nodes + Preserve Issue Source Docs

  **What to do**:
  - `backend/llm_infrastructure/llm/langgraph_agent.py`에 아래를 추가/수정:
    - `AgentState`에 필드 추가:
      - `issue_source_docs: List[RetrievalResult]` (또는 동일 의미의 명확한 이름)
      - `issue_source_display_docs: List[RetrievalResult]` (UI 표시용 merge된 docs)
    - `issue_step1_prepare_node` 수정:
      - `display_docs/docs`로부터 `issue_cases`를 만들고, 동시에 source docs를 `issue_source_*`로 보존
    - 새 노드(이름 고정):
      - `issue_step1_interrupt_confirm_node` (post_summary)
      - `issue_step1_confirm_apply_node`
      - `issue_case_list_prepare_node` (loop 시 nonce 재발급 + `docs/display_docs`를 `issue_source_*`로 복구)
      - `issue_other_interrupt_confirm_node` (post_detail)
      - `issue_other_confirm_apply_node`
    - `__all__` export 리스트에 새 노드 이름을 추가(서비스 레이어 import에서 사용됨)
  - Confirm interrupt payload는 TODO 1의 contract를 사용.

  **Must NOT do**:
  - 기존 `issue_case_selection`/`issue_sop_confirm` interrupt payload 타입/필드 변경

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: LangGraph 상태/interrupt/resume 정확도 중요
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 4-5 | Blocked By: 1,2

  **References**:
  - Issue state fields: `backend/llm_infrastructure/llm/langgraph_agent.py:163`
  - Current step1 interrupt: `backend/llm_infrastructure/llm/langgraph_agent.py:3619`
  - SOP confirm interrupt: `backend/llm_infrastructure/llm/langgraph_agent.py:3734`
  - SOP answer overwrites docs: `backend/llm_infrastructure/llm/langgraph_agent.py:3814`
  - Node exports: `backend/llm_infrastructure/llm/langgraph_agent.py:4107`

  **Acceptance Criteria**:
  - [ ] `issue_step1_prepare_node`가 `issue_cases`와 `issue_source_*`를 채운다.
  - [ ] `issue_case_list_prepare_node`가 loop 진입 시 `pending_interrupt_nonce`를 새로 만들고, `docs/display_docs`를 `issue_source_*`로 복구한다.

  **QA Scenarios**:
  ```
  Scenario: SOP 이후에도 case list 복구
    Tool: Bash
    Steps: run new backend graph test (added in Task 5)
    Expected: loop에서 issue_case_selection interrupt가 정상 payload(cases)로 다시 뜬다
    Evidence: .sisyphus/evidence/task-3-backend-nodes.txt

  Scenario: No cases edge
    Tool: Bash
    Steps: run new backend test variant with empty docs
    Expected: no issue_confirm interrupt; graceful final answer returned
    Evidence: .sisyphus/evidence/task-3-backend-nodes-empty.txt
  ```

  **Commit**: YES | Message: `fix(issue-flow): add issue_confirm interrupts and preserve case docs` | Files: `backend/llm_infrastructure/llm/langgraph_agent.py`


- [ ] 4. Backend: Rewire LangGraph Flow to Match Doc Section 3

  **What to do**:
  - `backend/services/agents/langgraph_rag_agent.py`에서 그래프 배선을 문서 플로우로 맞춘다.
  - 새 노드들을 import + builder.add_node + edge wiring까지 포함해 컴파일 가능한 상태로 만든다.
  - Required flow (issue mode):
    - `expand_related` → `issue_step1_prepare` → `answer` → `issue_step1_interrupt_confirm`
      - confirm no → `issue_finalize`
      - confirm yes → `issue_case_list_prepare` → `issue_step1_interrupt` → `issue_case_selection_apply`
        → `issue_step2_prepare` → `issue_step2_interrupt` → `issue_sop_confirm_apply`
        → `issue_step3_sop_answer` → `issue_other_interrupt_confirm`
          - confirm yes → `issue_case_list_prepare` (loop)
          - confirm no → `issue_finalize`
  - `answer` 노드 이후 conditional edge를 추가해 issue 모드는 `judge`로 가지 않도록 한다.

  **Must NOT do**:
  - Verified-mode retry 정책/should_retry 로직 수정(필요 없음)

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: 그래프 edge 실수는 런타임에서만 터짐
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 5 | Blocked By: 3

  **References**:
  - Existing route-after-expand logic: `backend/services/agents/langgraph_rag_agent.py:688`
  - Existing issue wiring: `backend/services/agents/langgraph_rag_agent.py:700`

  **Acceptance Criteria**:
  - [ ] `task_mode=issue` 최초 진입 시 첫 interrupt는 `issue_confirm(post_summary)`이다.
  - [ ] `issue_confirm`에서 yes를 선택하면 그 다음 interrupt는 `issue_case_selection`이다.

  **QA Scenarios**:
  ```
  Scenario: First interrupt ordering
    Tool: Bash
    Steps: run new backend test (Task 5)
    Expected: first interrupt payload.type == "issue_confirm"
    Evidence: .sisyphus/evidence/task-4-backend-wiring.txt

  Scenario: Non-issue path unchanged
    Tool: Bash
    Steps: run backend test suite
    Expected: existing non-issue tests remain green
    Evidence: .sisyphus/evidence/task-4-backend-wiring-regression.txt
  ```

  **Commit**: YES | Message: `fix(issue-flow): rewire graph to summary-first confirm gates` | Files: `backend/services/agents/langgraph_rag_agent.py`


- [ ] 5. Backend: Add LangGraph Issue-Flow Interrupt Ordering Tests

  **What to do**:
  - 새 테스트 파일 추가(이름 고정): `backend/tests/test_issue_flow_interrupts.py`
  - 테스트는 실제 LangGraph를 실행하고 `__interrupt__` payload를 assert한다.
  - Approach:
    - `LangGraphRAGAgent(mode="verified")` 생성(체크포인터 필요)
    - `backend.services.agents.langgraph_rag_agent.retrieve_node` / `expand_related_docs_node` / `answer_node` 등을 monkeypatch로 stub
    - `thread_id` 고정 config로 initial invoke + resume invoke를 수행
  - 최소 시나리오:
    - initial → `issue_confirm(post_summary)` and `answer` non-empty
    - resume(confirm yes) → `issue_case_selection`
    - resume(case select) → `issue_sop_confirm` and `answer` == "detail"
    - resume(sop confirm false) → `issue_confirm(post_detail)`
    - resume(post_detail yes) → `issue_case_selection` (loop)

  **Must NOT do**:
  - 외부 ES/LLM 네트워크 호출

  **Recommended Agent Profile**:
  - Category: `unspecified-high`
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 6 | Blocked By: 4

  **References**:
  - Graph test pattern: `backend/tests/test_agent_graph_mq_bypass.py`
  - Expand node tests (doc fixture style): `backend/tests/test_expand_related_docs_node.py`

  **Acceptance Criteria**:
  - [ ] `cd backend && uv run pytest tests/test_issue_flow_interrupts.py -v` passes

  **QA Scenarios**:
  ```
  Scenario: Full loop path
    Tool: Bash
    Steps: cd backend && uv run pytest tests/test_issue_flow_interrupts.py -v
    Expected: assertions for interrupt ordering and answer preservation pass
    Evidence: .sisyphus/evidence/task-5-backend-tests.txt

  Scenario: Empty retrieval
    Tool: Bash
    Steps: run a second test in same file with empty docs
    Expected: flow ends without issue_confirm interrupts
    Evidence: .sisyphus/evidence/task-5-backend-tests-empty.txt
  ```

  **Commit**: YES | Message: `test(issue-flow): assert interrupt ordering and loop behavior` | Files: `backend/tests/test_issue_flow_interrupts.py`


- [ ] 6. Backend API: Validate `issue_confirm` Guided Resume Decisions

  **What to do**:
  - `backend/api/routers/agent.py`에 Pydantic decision model 추가:
    - `IssueConfirmDecision(type="issue_confirm", nonce, stage, confirm)`
  - 아래 로직 확장:
    - `_is_guided_resume`: `issue_confirm` 포함
    - `_validated_guided_resume_decision`: `issue_confirm` 분기 추가
    - (선택) `_validate_guided_resume_checkpoint`: `issue_confirm`은 nonce만 검증하면 충분(현 로직이 이미 nonce 검증)

  **Must NOT do**:
  - 기존 resume decision validation을 느슨하게(extra allow) 바꾸지 않기 (계속 `extra="forbid"` 유지)

  **Recommended Agent Profile**:
  - Category: `unspecified-low`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 7-8 | Blocked By: 1,2

  **References**:
  - Guided decision models: `backend/api/routers/agent.py:306`
  - Guided resume detection: `backend/api/routers/agent.py:413`
  - Nonce guard: `backend/api/routers/agent.py:472`

  **Acceptance Criteria**:
  - [ ] Invalid `issue_confirm` resume_decision (missing nonce/stage/confirm) returns HTTP 400

  **QA Scenarios**:
  ```
  Scenario: Validation happy path
    Tool: Bash
    Steps: run backend unit tests (Task 5) which perform resume
    Expected: no validation errors; resume works
    Evidence: .sisyphus/evidence/task-6-api-validation.txt

  Scenario: Validation failure
    Tool: Bash
    Steps: add a unit test at API layer OR extend backend graph test to call _validated_guided_resume_decision with bad payload
    Expected: HTTPException 400
    Evidence: .sisyphus/evidence/task-6-api-validation-error.txt
  ```

  **Commit**: YES | Message: `fix(api): validate issue_confirm resume decisions` | Files: `backend/api/routers/agent.py`


- [ ] 7. Frontend: Preserve `res.answer` for Interrupted Responses (Issue UX Fix)

  **What to do**:
  - `frontend/src/features/chat/hooks/use-chat-session.ts` 수정:
    - interrupted 처리 시 `res.answer`가 비어있지 않으면 assistant message `content`를 `res.answer`로 유지
    - 기존 동작(안내문으로 덮어쓰기)은 `res.answer`가 빈 경우만 적용
  - 적용 범위는 interrupt kind 전체에 적용하되, `res.answer` non-empty 조건으로 회귀 리스크를 줄인다.

  **Must NOT do**:
  - auto_parse_confirm의 guided 패널 UX를 깨지 않기 (`answer`가 비어있어야 정상)

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 9-10 | Blocked By: 1

  **References**:
  - Current overwrite logic: `frontend/src/features/chat/hooks/use-chat-session.ts:507`
  - Issue flow desired behavior: `docs/2026-03-11-개선작업.md:80`

  **Acceptance Criteria**:
  - [ ] SOP confirm interrupt 응답에서 `answer`(상세 답변)가 UI message에 남는다.

  **QA Scenarios**:
  ```
  Scenario: Hook test asserts answer preservation
    Tool: Bash
    Steps: cd frontend && npm run test -- src/features/chat/__tests__/issue-flow-ui.test.tsx
    Expected: test asserts assistant content equals provided answer when interrupted
    Evidence: .sisyphus/evidence/task-7-fe-answer-preserve.txt

  Scenario: Non-issue interrupt still shows prompt
    Tool: Bash
    Steps: run frontend test suite
    Expected: existing guided/device selection tests pass
    Evidence: .sisyphus/evidence/task-7-fe-answer-preserve-regression.txt
  ```

  **Commit**: YES | Message: `fix(chat): keep answer text on interrupted responses` | Files: `frontend/src/features/chat/hooks/use-chat-session.ts`


- [ ] 8. Frontend: Add `issue_confirm` Interrupt Handling + Confirm Panel

  **What to do**:
  - Types/contract:
    - `frontend/src/features/chat/types.ts`에 `IssueConfirmInterruptPayload` 타입 추가
  - Hook state:
    - `use-chat-session.ts`에 `InterruptKind`에 `"issue_confirm"` 추가
    - `resolveInterruptKind`에 매핑 추가
    - pending state 추가: `pendingIssueConfirm`
    - submit handler 추가: `submitIssueConfirm(confirm: boolean)` → resume_decision payload 생성
    - guided resume type 판별에 `issue_confirm` 포함
  - UI:
    - 새 컴포넌트 추가: `frontend/src/features/chat/components/issue-confirm-panel.tsx`
    - `frontend/src/features/chat/components/index.ts` export
    - `frontend/src/features/chat/pages/chat-page.tsx`에서 패널 렌더 + 입력 disable 조건에 포함

  **Must NOT do**:
  - 기존 `IssueSopConfirmPanel` props를 깨는 방식의 재사용/변경

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: hook state + guided resume 경로는 깨지기 쉬움
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 9-10 | Blocked By: 7

  **References**:
  - Existing panels: `frontend/src/features/chat/components/issue-sop-confirm-panel.tsx`, `frontend/src/features/chat/components/issue-case-selection-panel.tsx`
  - Panel wiring: `frontend/src/features/chat/pages/chat-page.tsx:523`
  - Existing guided resume logic: `frontend/src/features/chat/hooks/use-chat-session.ts:663`

  **Acceptance Criteria**:
  - [ ] `issue_confirm` interrupt가 오면 yes/no 패널이 표시되고, 클릭 시 `{type:"issue_confirm", nonce, stage, confirm}`로 resume 된다.

  **QA Scenarios**:
  ```
  Scenario: Panel click sends correct resume payload
    Tool: Bash
    Steps: cd frontend && npm run test -- src/features/chat/__tests__/chat-request-payload.test.tsx
    Expected: new test asserts resume_decision matches contract
    Evidence: .sisyphus/evidence/task-8-fe-issue-confirm.txt

  Scenario: Panel component unit test
    Tool: Bash
    Steps: cd frontend && npm run test -- src/features/chat/__tests__/issue-panels.test.tsx
    Expected: IssueConfirmPanel emits onConfirm(true/false)
    Evidence: .sisyphus/evidence/task-8-fe-issue-confirm-panel.txt
  ```

  **Commit**: YES | Message: `feat(issue-flow): add issue_confirm interrupt panel and hook state` | Files: `frontend/src/features/chat/hooks/use-chat-session.ts`, `frontend/src/features/chat/pages/chat-page.tsx`, `frontend/src/features/chat/components/issue-confirm-panel.tsx`, `frontend/src/features/chat/types.ts`


- [ ] 9. Frontend Tests: Update Issue Flow Hook Test for New Step

  **What to do**:
  - `frontend/src/features/chat/__tests__/issue-flow-ui.test.tsx` 시나리오를 다음으로 업데이트:
    - guided auto_parse_confirm 이후: `issue_confirm(post_summary)` interrupt가 먼저 온다 (answer non-empty 포함)
    - submitIssueConfirm(true) 호출 후 `issue_case_selection`으로 진행
    - 기존 `issue_sop_confirm` 이후에는 `issue_confirm(post_detail)`이 한 번 더 온다 (confirm false 케이스)
  - 테스트에서 `assistant message content`가 interrupted+answer non-empty일 때 answer를 유지하는지 assert 추가.

  **Must NOT do**:
  - 테스트를 느슨하게(expect.anything) 만들어 계약을 흐리게 하지 않기

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 10 | Blocked By: 8

  **References**:
  - Existing test: `frontend/src/features/chat/__tests__/issue-flow-ui.test.tsx:52`

  **Acceptance Criteria**:
  - [ ] `cd frontend && npm run test -- src/features/chat/__tests__/issue-flow-ui.test.tsx` passes

  **QA Scenarios**:
  ```
  Scenario: Updated issue flow sequence
    Tool: Bash
    Steps: cd frontend && npm run test -- src/features/chat/__tests__/issue-flow-ui.test.tsx
    Expected: connectSse called N times with expected resume_decisions
    Evidence: .sisyphus/evidence/task-9-fe-issue-flow-test.txt

  Scenario: Answer preservation assertion
    Tool: Bash
    Steps: same test asserts message content equals summary/detail answer during interrupts
    Expected: passes
    Evidence: .sisyphus/evidence/task-9-fe-issue-flow-test-answer.txt
  ```

  **Commit**: YES | Message: `test(issue-flow): update hook e2e for issue_confirm gates` | Files: `frontend/src/features/chat/__tests__/issue-flow-ui.test.tsx`


- [ ] 10. Frontend Tests: Add Request-Payload Coverage for `issue_confirm`

  **What to do**:
  - `frontend/src/features/chat/__tests__/chat-request-payload.test.tsx`에 케이스 추가:
    - interrupt_payload가 `issue_confirm`일 때 pendingIssueConfirm 생성
    - submitIssueConfirm(true/false) 호출 시 post payload에 `thread_id`, `ask_user_after_retrieve:false`, `resume_decision` 포함
    - `resume_decision.stage`와 `nonce`까지 정확히 assert

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: none | Blocked By: 8

  **References**:
  - Existing payload tests for issue decisions: `frontend/src/features/chat/__tests__/chat-request-payload.test.tsx:150`

  **Acceptance Criteria**:
  - [ ] `cd frontend && npm run test -- src/features/chat/__tests__/chat-request-payload.test.tsx` passes

  **QA Scenarios**:
  ```
  Scenario: Resume payload correctness
    Tool: Bash
    Steps: cd frontend && npm run test -- src/features/chat/__tests__/chat-request-payload.test.tsx
    Expected: new issue_confirm test passes
    Evidence: .sisyphus/evidence/task-10-fe-payload-test.txt

  Scenario: No regression in existing payload tests
    Tool: Bash
    Steps: run full frontend test suite
    Expected: all tests pass
    Evidence: .sisyphus/evidence/task-10-fe-payload-test-regression.txt
  ```

  **Commit**: YES | Message: `test(chat): cover issue_confirm resume payload` | Files: `frontend/src/features/chat/__tests__/chat-request-payload.test.tsx`


- [ ] 11. Prompt: Tighten Issue Summary Format to Include `[doc_id]` (Optional-but-Recommended)

  **What to do**:
  - `backend/llm_infrastructure/llm/prompts/issue_ans_v2.yaml`의 system 지시문을 최소 변경으로 강화:
    - 각 항목 첫 줄에 반드시 `[{doc_id}]`를 포함하도록 요구 (doc_id는 `ref_text`의 `doc_id` 토큰에서 가져오도록 안내)
    - 문서 예시(`docs/2026-03-11-개선작업.md:95`)와 같은 "원인/조치" 요약 키워드를 유지
    - 출력이 과도하게 길어지지 않도록(예: 케이스당 1-2줄) 제한을 추가
  - 기존 `issue_ans_v1.yaml`은 그대로 둔다(v2만 조정).

  **Must NOT do**:
  - 프롬프트 이름/버전 키 변경(로드 경로 깨짐)

  **Recommended Agent Profile**:
  - Category: `quick`
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: none | Blocked By: none

  **References**:
  - Prompt file: `backend/llm_infrastructure/llm/prompts/issue_ans_v2.yaml`
  - Desired format example: `docs/2026-03-11-개선작업.md:95`

  **Acceptance Criteria**:
  - [ ] Prompt explicitly requires `[{doc_id}]` in each list item

  **QA Scenarios**:
  ```
  Scenario: Prompt load regression
    Tool: Bash
    Steps: cd backend && uv run pytest tests/test_answer_language_templates.py -q
    Expected: prompt spec loads; tests remain green
    Evidence: .sisyphus/evidence/task-11-prompt.txt

  Scenario: Smoke run (no external calls)
    Tool: Bash
    Steps: rely on existing tests; no network-dependent validation
    Expected: no failures
    Evidence: .sisyphus/evidence/task-11-prompt-smoke.txt
  ```

  **Commit**: YES | Message: `chore(prompts): enforce doc_id in issue summary list` | Files: `backend/llm_infrastructure/llm/prompts/issue_ans_v2.yaml`


## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. UI/Flow QA via Tests — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Prefer 4-6 atomic commits aligned to TODO groupings (contract/version, backend nodes+wiring, backend tests, FE hook+panel, FE tests).
- No formatting-only commits.

## Success Criteria
- 문서의 "수정할 플로우"가 실제 동작(테스트로 검증)한다.
- interrupt 응답에서도 사용자에게 요약/상세 답변이 먼저 보인다.
- 루프(다른 이슈)에서 케이스 리스트가 안정적으로 재표시된다.
