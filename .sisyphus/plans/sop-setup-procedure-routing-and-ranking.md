# SOP/Setup 선택: 비절차 질문 라우팅 편향 + Setup 결과 노출(랭킹 다양성) 개선

## TL;DR
> **Summary**: SOP/Setup 선택 상태에서 비절차(조회) 질문이 절차 파이프라인으로 강제되는 편향을 제거하고, “절차조회” 결과에서 setup 문서가 sop에 묻히지 않도록 doc_type 다양성(쿼터) 정책을 추가한다.
> **Deliverables**:
> - Intent-gated SOP retrieval penalties/boosts (route/키워드 기반)
> - Inquiry(조회) 키워드 안전 오버라이드로 `route=general` 보장
> - SOP+Setup 동시 선택 시 top-k doc_type 다양성(최소 setup 노출) 정책
> - 회귀 방지 unit/API 테스트 + 기존 contract 테스트 통과
> **Effort**: Medium
> **Parallel**: YES — 3 waves
> **Critical Path**: (tests red) → retrieve_node gating → route override → doc_type quota → verification

## Context
### Original Request
- UI에서 “절차조회”를 선택하면 `sop`, `setup`이 함께 조회되지만 **SOP 절차 위주로만 조회되는 것 같다**.
- 동시에, SOP/Setup 문서를 선택한 상태에서 “work sheet/tool list/scope 조회” 같은 **비절차(정보 조회) 질문**이 절차 답변 형식으로 편향되는 문제가 있어 이를 개선하려 함.
- 기준 문서: `docs/tasks/TASK-20260326-non-procedural-query-routing.md`

### Interview Summary
- Scope: **라우팅 편향 + 랭킹(Setup 노출) 둘 다**
- Routing strategy: **B + 일부 A**
  - B: retrieval penalties/boost를 ‘절차 의도’일 때만 적용
  - 일부 A: inquiry 키워드가 명확하면 route를 general로 안전 오버라이드

### Key Code Facts (evidence-backed)
- Backend routing entry:
  - `backend/llm_infrastructure/llm/langgraph_agent.py:route_node()`
  - `task_mode`가 비어있으면 `_infer_task_mode_from_doc_types(state.selected_doc_types)`로 추론
- MQ generation:
  - `mq_node()`는 최종적으로 `route` 기반으로 `setup_mq/ts_mq/general_mq` 선택
- Retrieval bias root cause:
  - `retrieve_node()`에서 `sop_only_predicate = state.sop_intent True OR selected_doc_types ∩ sop_variants`
  - 이 플래그로 인해 `route="general"`이어도 SOP soft-boost / early-page penalty / scope-contents penalty 등이 적용될 수 있음
- Frontend request surface:
  - `frontend/src/features/chat/types.ts:AgentRequest.filter_doc_types` 존재
  - docType 선택 UI: `frontend/src/features/chat/components/device-selection-panel.tsx`

### Metis Review (gaps addressed)
- 최소 변경 권장: **3-way route 유지(`setup|ts|general`)**, `info` route 추가는 blast radius가 큼.
- 기존 테스트 일부가 현재 “route=general인데도 SOP boost 적용”을 전제하고 있어, 의도 변경 시 테스트 업데이트가 필수.
- API contract(C-API-001/002/003) 및 회귀 테스트를 우선 통과시키는 형태로 설계.

## Work Objectives
### Core Objective
1) SOP/Setup 선택 상태에서 **비절차 조회 질문이 절차 파이프라인으로 편향되지 않게** 하고, 2) “절차조회” 결과에서 **setup 문서가 최소한 일정 수 노출**되도록 한다.

### Deliverables
1. Backend logic changes
   - SOP 관련 retrieval penalties/boost를 **절차 의도(=route/setup 또는 procedure 키워드)**일 때만 적용
   - Inquiry 키워드(worksheet/tool list/scope/contents 등)에는 `route=general` 안전 오버라이드
   - SOP+Setup 동시 선택 시 top-k 결과에 **doc_type 다양성/쿼터** 적용
2. Tests
   - unit: retrieve penalties/boost gating, route override, doc_type quota
   - api: SOP/Setup 선택 + inquiry query가 `route=general`로 끝나는 회귀 테스트
   - 기존 contract tests 유지

### Definition of Done (verifiable)
- [ ] SOP/Setup 선택 + inquiry query(예: “tool list 보여줘”)에서:
  - metadata route = `general`
  - general MQ/answer 경로를 사용
  - scope/contents penalty 및 SOP-only boost가 적용되지 않음
- [ ] SOP/Setup 선택 + procedure query(예: “slot valve 교체 절차”)에서:
  - metadata route = `setup`
  - 기존 SOP 절차 부스트/패널티는 유지
- [ ] “절차조회” (SOP+Setup 동시 선택)에서 top-k에 **setup doc_type이 최소 N개 이상** 포함(가능할 때)
- [ ] Contract/Regression:
  - `uv run pytest tests/api/test_agent_response_metadata_contract.py -v`
  - `uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v`
  - `uv run pytest tests/api/test_agent_retrieval_only.py -v`
  - `cd backend && uv run pytest tests/ -v` (관련 테스트 포함)

### Defaults Applied (locked)
- Route taxonomy는 유지: `setup|ts|general` (no `info` route)
- Inquiry override는 **procedure wins** 규칙을 가진다:
  - query에 procedure intent가 있으면 inquiry 키워드가 있어도 `setup` 우선

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: tests-after (routing/heuristic 변경은 회귀 위험이 높으므로 테스트 우선 작성)
- Evidence files: `.sisyphus/evidence/task-{N}-{slug}.txt`

## Execution Strategy
### Parallel Execution Waves
Wave 1: 테스트/정의 고정 (Tasks 1–4)
Wave 2: backend 로직 변경 (Tasks 5–8)
Wave 3: quota + API regression + full verification (Tasks 9–12)

### Dependency Matrix
- Task 1 blocks everything (clean worktree)
- Task 2–4 block Task 5–8 (tests define behavior)
- Task 5–8 block Task 9–12

## TODOs

- [ ] 1. Create clean worktree + task doc

  **What to do**:
  - 새 브랜치/워크트리 생성 (예: `git worktree add ../llm-agent-v2-proc-route -b fix/procedure-routing-bias`)
  - `docs/tasks/TASK-20260326-non-procedural-query-routing.md`를 기반으로 새 task 문서 생성:
    - `docs/tasks/TASK-20260326-procedure-routing-and-setup-diversity.md`
    - Contracts To Preserve: `C-API-001`, `C-API-002`, `C-API-003`
    - Allowed Files는 아래 TODO의 파일들로 lock

  **Must NOT do**:
  - 기존 dirty worktree에서 작업하지 말 것

  **Recommended Agent Profile**:
  - Category: `unspecified-low`
  - Skills: [`task-start-kickoff`]

  **Parallelization**: Can Parallel: NO | Wave 1

  **Acceptance Criteria**:
  - [ ] 새 worktree에서 `git status --short` clean

  **QA Scenarios**:
  ```
  Scenario: Isolated worktree
    Tool: Bash
    Steps: git worktree add ...; git status --short
    Expected: clean status
    Evidence: .sisyphus/evidence/task-1-worktree.txt
  ```

  **Commit**: NO

- [ ] 2. Lock inquiry vs procedure intent keyword sets (decision complete)

  **What to do**:
  - 절차 의도 키워드(현행 `_PROCEDURE_KEYWORDS` 기반)와 inquiry 키워드(worksheet/tool list/scope/contents 등)를 명시적으로 확정
  - 충돌 규칙 확정: **procedure wins**

  **References**:
  - Task doc: `docs/tasks/TASK-20260326-non-procedural-query-routing.md` (키워드 후보)

  **Acceptance Criteria**:
  - [ ] 키워드 목록과 precedence 규칙이 task doc에 기록됨

  **QA Scenarios**:
  ```
  Scenario: Policy recorded
    Tool: Read
    Steps: open task doc and confirm keyword lists + precedence
    Expected: explicit lists present
    Evidence: .sisyphus/evidence/task-2-keywords.txt
  ```

  **Commit**: NO

- [ ] 3. Update existing unit tests that assume SOP penalties apply even when route=general

  **What to do**:
  - 아래 테스트들을 “절차 의도 있을 때만 SOP penalties/boost 적용”으로 재정의:
    - `backend/tests/test_retrieve_node_sop_soft_boost.py`
    - `backend/tests/test_retrieve_node_stage2_early_page_penalty.py`
  - 기존 assertion 중 `route="general"`인 케이스는:
    - (a) route를 `setup`으로 바꾸거나
    - (b) query를 procedure-intent로 바꿔서 의도를 맞춘다.

  **Acceptance Criteria**:
  - [ ] 변경 전후로 테스트가 “의도 기반 적용”을 정확히 검증

  **QA Scenarios**:
  ```
  Scenario: Tests reflect new contract
    Tool: Bash
    Steps: cd backend && uv run pytest tests/test_retrieve_node_sop_soft_boost.py -v
    Expected: pass
    Evidence: .sisyphus/evidence/task-3-test-update.txt
  ```

  **Commit**: YES | Message: `test(retrieve): gate SOP boosts by procedural intent` | Files: [backend/tests/test_retrieve_node_sop_soft_boost.py, backend/tests/test_retrieve_node_stage2_early_page_penalty.py]

- [ ] 4. Add new unit tests for inquiry override + scope penalty bypass

  **What to do**:
  - 신규 테스트 추가:
    - `backend/tests/test_route_node_inquiry_override.py`
      - router가 `setup`을 반환하더라도 inquiry 키워드면 최종 route=`general`
      - procedure 키워드 포함 시 route=`setup`
    - `backend/tests/test_sop_info_query_scope_penalty_bypass.py`
      - SOP selection + route=`general` + inquiry query에서 scope/contents penalty 미적용

  **Acceptance Criteria**:
  - [ ] 신규 테스트가 fail-first로 동작 후 구현으로 green

  **QA Scenarios**:
  ```
  Scenario: New tests fail before fix
    Tool: Bash
    Steps: cd backend && uv run pytest tests/test_route_node_inquiry_override.py -v
    Expected: fails until implementation is added
    Evidence: .sisyphus/evidence/task-4-red.txt
  ```

  **Commit**: YES | Message: `test(route): add inquiry override and scope-penalty bypass cases` | Files: [backend/tests/test_route_node_inquiry_override.py, backend/tests/test_sop_info_query_scope_penalty_bypass.py]

- [ ] 5. Implement retrieve_node: gate SOP penalties/boosts by intent (route/setup OR procedure keywords)

  **What to do**:
  - `backend/llm_infrastructure/llm/langgraph_agent.py:retrieve_node`에서:
    - `sop_only_predicate`를 “doc_type 선택”이 아니라 **의도 기반**으로 분리:
      - `is_procedural_intent = (route == 'setup') OR contains_procedure_keyword(query)`
      - `is_inquiry_intent = contains_inquiry_keyword(query)`
    - early-page penalty / sop soft boost / scope penalty는 `is_procedural_intent`일 때만 적용
    - 단, `is_inquiry_intent`일 때는 scope/contents penalty를 항상 bypass

  **Must NOT do**:
  - issue flow/ts flow 로직을 건드리지 말 것

  **Acceptance Criteria**:
  - [ ] Task 3/4 테스트 green

  **QA Scenarios**:
  ```
  Scenario: Inquiry query bypasses SOP penalties
    Tool: Bash
    Steps: cd backend && uv run pytest tests/test_sop_info_query_scope_penalty_bypass.py -v
    Expected: pass
    Evidence: .sisyphus/evidence/task-5-inquiry-bypass.txt
  ```

  **Commit**: YES | Message: `fix(retrieve): apply SOP penalties only for procedural intent` | Files: [backend/llm_infrastructure/llm/langgraph_agent.py]

- [ ] 6. Implement route_node: inquiry safe override (B + 일부 A)

  **What to do**:
  - `backend/llm_infrastructure/llm/langgraph_agent.py:route_node`에서 라우터 결과를 받은 뒤:
    - `route == 'setup'`이지만 inquiry 키워드가 강하면 `general`로 override
    - procedure 키워드가 있으면 override 금지(procedure wins)

  **Acceptance Criteria**:
  - [ ] `backend/tests/test_route_node_inquiry_override.py` green

  **Commit**: YES | Message: `fix(route): override setup to general for inquiry queries` | Files: [backend/llm_infrastructure/llm/langgraph_agent.py]

- [ ] 7. Implement SOP+Setup doc_type diversity quota for “절차조회” top-k

  **What to do**:
  - 적용 조건(locked): `selected_doc_types`가 SOP와 setup을 모두 포함할 때
  - 정책(locked default): top_k=10 기준
    - `min_setup = 3` (가능하면)
    - `min_sop = 3` (가능하면)
    - 나머지는 원래 score 순서로 fill
  - 구현 위치(권장): retrieve_node에서 `all_docs`를 최종 반환/축약하기 직전
    - doc metadata에서 `doc_type`/canonical group을 읽고 bucketize
    - canonicalization은 `backend/domain/doc_type_mapping.py`를 사용

  **Acceptance Criteria**:
  - [ ] 신규 unit test로 min_setup 보장 검증

  **QA Scenarios**:
  ```
  Scenario: Setup appears in top-k when available
    Tool: Bash
    Steps: cd backend && uv run pytest tests/test_doc_type_diversity_quota.py -v
    Expected: pass
    Evidence: .sisyphus/evidence/task-7-quota.txt
  ```

  **Commit**: YES | Message: `fix(retrieve): enforce SOP/setup diversity quota for procedure lookup` | Files: [backend/llm_infrastructure/llm/langgraph_agent.py, backend/domain/doc_type_mapping.py]

- [ ] 8. Add API regression test for SOP/Setup + inquiry query route and template path

  **What to do**:
  - `tests/api/test_agent_sop_info_query_routing_regression.py` 추가
  - `filter_doc_types=["sop","setup"]` + message="tool list 보여줘"
  - dependency_overrides로 router가 일부러 `setup` 반환하도록 만든 뒤에도 route가 `general`로 끝나는지 확인

  **Acceptance Criteria**:
  - [ ] 테스트가 CI에서 안정적으로 pass

  **Commit**: YES | Message: `test(api): guard SOP info-query routing against procedural bias` | Files: [tests/api/test_agent_sop_info_query_routing_regression.py]

- [ ] 9. Full verification sweep (contracts + backend tests)

  **What to do**:
  - 아래를 순서대로 실행하고 evidence에 저장:
    - `uv run pytest tests/api/test_agent_response_metadata_contract.py -v`
    - `uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v`
    - `uv run pytest tests/api/test_agent_retrieval_only.py -v`
    - `cd backend && uv run pytest tests/ -v`

  **Acceptance Criteria**:
  - [ ] 모두 pass

  **Commit**: NO

## Final Verification Wave (MANDATORY)
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Tests-first commits(Tasks 3–4) 후 behavior change commits(Tasks 5–8).
- Contract tests green 유지.

## Success Criteria
- SOP/Setup 선택 상황에서 “조회/목록/Scope/Contents” 질의가 절차 답변으로 오염되지 않음.
- 절차조회 top-k에서 setup 문서가 가능하면 최소 N개 노출.
- 모든 회귀/계약 테스트 통과.
