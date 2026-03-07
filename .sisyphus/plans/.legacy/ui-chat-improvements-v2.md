# UI/Chat Improvements v2 Plan

## TL;DR
> **Summary**: `docs/2026-03-06-Agent_개선.md`의 6개 개선 항목을 코드/테스트/검증까지 포함해 구현한다.
> **Deliverables**: guided selection UX 개선 + 모델 포맷 불이행 방어 + 완료 상태 resume 오류 방지 + 용어/매핑 정합화
> **Effort**: Medium
> **Parallel**: YES - 3 waves
> **Critical Path**: REQ-2 파싱/프롬프트 안정화 → REQ-3 resume 방지 → REQ-6 탭 UI

## Context
### Original Request
- `.omc/plans/ui-chat-improvements-v2.md`를 검토하고 `docs/2026-03-06-Agent_개선.md` 구현 목적에 맞게 계획을 수정

### Interview Summary
- REQ-5 정책 확정: **추가 문서유형 질문 없이**
  - 절차조회(sop): SOP+TS+SETUP 전체 검색
  - 이슈조회(issue): gcb+myservice 전체 검색

### Metis Review (gaps addressed)
- (추후 생성: 구현 에이전트가 실행 전 자동 점검)

## Work Objectives
### Core Objective
- guided selection 첫 질문 플로우를 사용자 혼란 없이 완료시키고, 모델 변경(GLM/vLLM) 시에도 파싱/라우팅/judge가 깨지지 않게 만든다.

### Deliverables
- REQ-1: `건너뛰기`는 어떤 조건에서도 추천 표시가 되지 않음
- REQ-2: 파싱 fallback 통일 + judge JSON 강제(가능시) + 재시도 상한 + translate/query_rewrite 정규화 + 프롬프트 강화
- REQ-3: 완료 상태에서 장비선택 resume 경로가 UI/서버 모두에서 차단됨
- REQ-4: UI 라벨 `SOP 조회` → `절차조회` (value `sop` 유지)
- REQ-5: task_mode별 고정 doc_type 문서군 적용(추가 질문 없음)
- REQ-6: 탭 기반 단계 UI + 진행 위치(`총 N개 중 X번째`) 표시 + 이전 단계 수정

### Definition of Done (verifiable)
- Backend: `pytest -q` 관련 테스트 통과
- Frontend: `cd frontend && npm test` 통과
- 수동 QA 시나리오(플랜 내 Playwright/브라우저 기반)에서 크래시/루프 없음

### Must NOT Have
- REQ-5에 문서유형 추가 질문(step) 도입 금지
- 요구사항에 없는 검색 로직(랭킹/쿼리 생성 알고리즘) 변경 금지

## Verification Strategy
- Test decision: tests-after (기존 테스트 인프라 사용)
- Evidence: 각 태스크별 .sisyphus/evidence/task-{N}-*.txt/json

## Execution Strategy
### Parallel Execution Waves
Wave 1: REQ-1/REQ-4/REQ-5 (backend 매핑/라벨 단순 변경) + REQ-3(backend 가드)
Wave 2: REQ-2(파싱/프롬프트/재시도 상한) + REQ-3(front 조건)
Wave 3: REQ-6(탭 UI 리팩토링) + 통합 테스트/QA

## TODOs

- [x] 1. REQ-1 `건너뛰기` recommended 항상 false로 고정

  **What to do**:
  - `backend/llm_infrastructure/llm/langgraph_agent.py`에서 `__skip__` 옵션의 `recommended`를 항상 `False`로 변경
  - device/equip_id 모두 동일 적용

  **References**:
  - Backend payload 생성: `backend/llm_infrastructure/llm/langgraph_agent.py` (device/equip_id options; 현재 3082-3101 부근)

  **Acceptance Criteria**:
  - [ ] `pytest -q tests/api/test_agent_autoparse_confirm_interrupt_resume.py`에서 guided payload의 skip recommended가 false로 검증되거나, 동일 커버리지의 테스트 추가/갱신

  **QA Scenarios**:
  ```
  Scenario: Parse 실패에서도 skip 비추천
    Tool: Bash
    Steps: 관련 테스트/스냅샷 실행
    Expected: __skip__ 항목 recommended=false
    Evidence: .sisyphus/evidence/task-01-skip-recommended.txt
  ```

- [x] 2. REQ-4 라벨 `SOP 조회` → `절차조회`

  **What to do**:
  - `task_options`의 label만 변경(value는 `sop` 유지)

  **References**:
  - `backend/llm_infrastructure/llm/langgraph_agent.py` task_options (3104-3108 부근)

  **Acceptance Criteria**:
  - [ ] 프론트 guided selection UI에 `절차조회`가 표시되고 선택 시 task_mode=`sop` 전달

  **QA Scenarios**:
  ```
  Scenario: 라벨 변경 반영
    Tool: Playwright
    Steps: 첫 질문에서 업무 옵션 확인
    Expected: SOP 조회 대신 절차조회 표시
    Evidence: .sisyphus/evidence/task-02-label.png
  ```

- [x] 3. REQ-5 task_mode별 고정 문서군 매핑 변경(추가 질문 없음)

  **What to do**:
  - `task_mode == "sop"`: `expand_doc_type_selection(["sop", "ts", "setup"])`
  - `task_mode == "issue"`: `expand_doc_type_selection(["gcb", "myservice"])`
  - `task_mode == "all"`: 기존대로 필터 없음
  - guided selection steps는 기존 4단계 유지

  **References**:
  - 현재 매핑: `backend/llm_infrastructure/llm/langgraph_agent.py` (3164-3176 부근)
  - helper: `backend/domain/doc_type_mapping.py:expand_doc_type_selection`

  **Acceptance Criteria**:
  - [ ] `절차조회` 선택 시 selected_doc_types에 SOP/ts/setup 그룹이 포함
  - [ ] `이슈조회` 선택 시 gcb/myservice만 포함(기존 ts 포함 제거)

  **QA Scenarios**:
  ```
  Scenario: task_mode별 doc_type 고정
    Tool: Bash
    Steps: 관련 테스트 실행 또는 재현 스크립트로 payload/응답 확인
    Expected: sop=3그룹, issue=2그룹
    Evidence: .sisyphus/evidence/task-03-doc-type-mapping.txt
  ```

- [x] 4. REQ-3 완료 상태 resume/장비선택 버튼 오류 방지 (backend + frontend)

  **What to do**:
  - Frontend: pending interrupt가 있을 때만 DeviceSelectionPanel/버튼 노출
  - Backend: pending 없는 resume_decision 요청은 4xx + 명확한 에러 코드/메시지

  **References**:
  - Backend interrupt 생성: `backend/llm_infrastructure/llm/langgraph_agent.py` device_selection_node (3285-3372 부근)
  - API: `backend/api/routers/agent.py`
  - UI 조건: `frontend/src/features/chat/pages/chat-page.tsx`
  - 핸들러: `frontend/src/features/chat/hooks/use-chat-session.ts`

  **Acceptance Criteria**:
  - [ ] 답변 완료 상태에서 장비선택 UI가 노출되지 않음
  - [ ] pending 없이 resume 호출 시 4xx + 에러 메시지

  **QA Scenarios**:
  ```
  Scenario: 완료 상태에서 resume 차단
    Tool: Playwright
    Steps: 대화 종료 후 장비선택 UI 노출 여부 확인
    Expected: 버튼/패널 미노출
    Evidence: .sisyphus/evidence/task-04-no-device-panel.png

  Scenario: 서버 resume 가드
    Tool: Bash
    Steps: pending 없는 상태로 resume API 호출
    Expected: 4xx + 명확한 에러 코드
    Evidence: .sisyphus/evidence/task-04-resume-guard.txt
  ```

- [x] 5. REQ-2 모델 포맷 불이행 방어(파싱 fallback 통일 + judge JSON 강제 + 재시도 상한 + 정규화 + 프롬프트 강화)

  **What to do**:
  - `_parse_auto_parse_result`, `_parse_queries`, `_parse_route`, `judge_node`, `_parse_needs_history_from_text`에 공통 fallback 유틸 적용
  - 가능한 LLM client 경로에서 `judge`는 `response_format=json_object` 사용
  - 파싱 실패 시 재시도 상한 도입(폭주 방지) + 안전 기본값
  - translate/query_rewrite 출력에서 분석문/번호형 설명 제거(정규화)
  - `backend/llm_infrastructure/llm/prompts/auto_parse_v1.yaml`, `backend/llm_infrastructure/llm/prompts/router_v1.yaml`에 JSON-only + markdown 금지 강화

  **Acceptance Criteria**:
  - [ ] GLM 계열로도 judge 파싱 실패가 나더라도 재시도 루프가 과도하게 반복되지 않음

- [ ] 6. REQ-6 탭 기반 guided selection UI 리팩토링

  **What to do**:
  - 기존 GuidedSelectionPanel을 탭 헤더 + 진행 상태로 리팩토링
  - 상단에 `총 N개 중 X번째` 항상 표시
  - 이전 단계 탭 클릭으로 수정 가능
  - 마지막 단계에서만 확인/진행 버튼 활성화(그 외 자동 다음 단계)

  **References**:
  - 컴포넌트: `frontend/src/features/chat/components/guided-selection-panel.tsx`
  - 타입: `frontend/src/features/chat/types.ts`
  - 테스트: `frontend/src/features/chat/__tests__/guided-selection-panel.test.tsx`

  **Acceptance Criteria**:
  - [ ] `npm test`에서 guided selection 관련 테스트 통과
  - [ ] 탭 UI에서 진행 위치가 항상 보임

## Final Verification Wave
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high (+ playwright)
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Atomic commits 추천:
  - `fix(chat): remove skip recommended + rename sop label`
  - `fix(chat): task_mode doc_type mapping + resume guard`
  - `fix(llm): robust parsing fallbacks + judge json mode`
  - `feat(chat): tabbed guided selection`

## Success Criteria
- REQ-1~REQ-6 수용 기준 충족 + 테스트 통과 + 수동 QA에서 크래시/무한루프 없음
