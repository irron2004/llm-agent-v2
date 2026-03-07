# Agent 개선 구현 계획 (2026-03-07)

**소스**: `docs/2026-03-06-Agent_개선.md`, `.omc/plans/ui-chat-improvements-v2.md`

---

## 현황 분석: 이미 완료된 항목

| REQ | 상태 | 근거 |
|-----|------|------|
| REQ-1: 건너뛰기 recommend 제거 | **완료** | `langgraph_agent.py:3237,3248-3253` — 이미 `"recommended": False` |
| REQ-4: SOP 조회 → 절차조회 | **완료** | `langgraph_agent.py:3258` — 이미 `"label": "절차조회"` |
| REQ-5: 고정 문서군 매핑 | **완료** | `langgraph_agent.py:3317-3325` — sop→[sop,ts,setup], issue→[gcb,myservice] |

---

## 남은 작업: 3개

### Task 1: REQ-2 — 모델 포맷 에러 방어 (중간 난이도)

**목표**: GLM-4.7-Flash 등 다른 모델에서도 파싱 에러 없이 동작

**현재 상태 분석**:
- `judge_node()` (line 2196): 이미 `response_format=json_object` 사용, `_parse_json_object_or_none` fallback 있음
- `_parse_json_object_or_none()` (line 510): JSON 직접 파싱 + regex 추출 지원
- 문제점: `_parse_auto_parse_result()`, `_parse_queries()`, `_parse_route()` 등은 각각 다른 파싱 로직 사용

**변경 사항**:

#### 1-1. 통일된 JSON 파싱 유틸리티 강화
- **파일**: `langgraph_agent.py:510-540` (`_parse_json_object_or_none`)
- **변경**: 코드블록 내 JSON 추출 (```json ... ```) 단계 추가
- fallback 체인: JSON 직접 → 코드블록 내 JSON → regex `\{.*\}` → None

#### 1-2. `_parse_auto_parse_result()` fallback 강화
- **파일**: `langgraph_agent.py:2421+`
- **변경**: 파싱 실패 시 빈 결과(`{}`) 반환 + 경고 로그, 에러 전파 차단

#### 1-3. `_parse_queries()` 결과 정규화
- **파일**: `langgraph_agent.py:482-647`
- **변경**: 분석문/번호형 설명 제거 정규화 (`_normalize_single_query_text` 활용)
- 쿼리 리스트에서 비어있는 항목 필터링

#### 1-4. `_parse_route()` fallback
- **파일**: `langgraph_agent.py:417-424`
- **변경**: 파싱 실패 시 기본 route `"general"` 반환

#### 1-5. 프롬프트 포맷 지시 강화
- **파일**: `prompts/auto_parse_v1.yaml`, `prompts/router_v1.yaml`
- **변경**: "JSON만 출력, 마크다운 코드블록 금지" 명시 추가

**검증 기준**:
- GLM-4.7-Flash로 10개 질문 → 파싱 에러 0건
- gpt-oss-20b에서 기존 동작 정상 유지
- judge 파싱 실패 시 무한 재시도 없이 3회 이내 종료

---

### Task 2: REQ-3 — 답변 완료 후 장비선택 버튼 에러 수정 (중간 난이도)

**목표**: 답변 완료 상태에서 장비선택/DeviceSelectionPanel이 노출되지 않도록

**현재 상태 분석**:
- 프론트: `chat-page.tsx:66-69` — `hasPendingGuidedSelectionInterrupt`, `hasPendingDeviceSelectionInterrupt` 조건으로 이미 guard 있음
- 백엔드: `agent.py:410-413` — `_resume_guard_detail()` 존재, pending 없는 resume에 에러 반환
- 문제: 정확한 에러 재현 조건 파악 필요

**변경 사항**:

#### 2-1. 프론트엔드 guard 강화
- **파일**: `chat-page.tsx:502-524`
- **변경**: `hasPendingDeviceSelectionInterrupt` 조건에 추가로 `isStreaming === false && !isCompleted` 체크
- 답변이 완료된(`isCompleted`) 상태에서는 패널 노출 차단

#### 2-2. 백엔드 resume guard 에러 코드 명확화
- **파일**: `agent.py:895-910`
- **변경**: pending 없는 resume 시 `400` 대신 `409 Conflict` + `RESUME_NO_PENDING_INTERRUPT` 코드
- 프론트에서 이 에러코드를 받으면 패널 자동 닫기

#### 2-3. 프론트엔드 에러 핸들링
- **파일**: `use-chat-session.ts:680-695`
- **변경**: resume 요청 실패 시 pending 상태 초기화 + 사용자에게 "선택 시간이 만료되었습니다" 메시지

**검증 기준**:
- 답변 완료 후 장비선택 패널이 노출되지 않음
- pending 없는 resume 요청 시 409 + 명확한 에러 메시지
- 네트워크 에러 시 crash 없이 에러 메시지 표시

---

### Task 3: REQ-6 — 탭 기반 스텝 인디케이터 UI 개선 (높은 난이도)

**목표**: 사용자가 전체 진행 상황을 한눈에 파악하고, 이전 단계를 수정할 수 있는 탭 UI

**현재 상태 분석**:
- `guided-selection-panel.tsx`: 이미 Ant Design `Tabs` 사용, 기본 탭 전환 동작 있음
- 이미 구현된 것:
  - 탭 헤더 표시 (언어/기기/설비/작업)
  - "총 N개 중 X번째" 진행 표시
  - 탭 클릭으로 이전 단계 이동
  - 선택 시 자동 다음 탭 전환
  - 마지막 단계에서 자동 제출
- 미구현:
  - 탭 헤더에 완료 상태 아이콘 (✓/●/○)
  - 탭 헤더에 선택된 값 미리보기
  - 미방문 탭 비활성화 (forward 클릭 방지)

**변경 사항**:

#### 3-1. 탭 헤더에 상태 아이콘 + 선택값 표시
- **파일**: `guided-selection-panel.tsx:255-265`
- **변경**: `items` 배열의 `label`을 JSX로 변환
  - 완료 탭: `✓ 언어: 한국어`
  - 현재 탭: `● 기기`
  - 미방문 탭: `○ 설비`

#### 3-2. 미방문 탭 비활성화
- **파일**: `guided-selection-panel.tsx:254`
- **변경**: `onChange` 핸들러에서 `idx > maxVisitedStep` 이면 클릭 무시
- 새 state: `maxVisitedStep` (가장 멀리 진행한 탭 인덱스)

#### 3-3. 타입 정의 추가 (선택사항)
- **파일**: `types.ts`
- **변경**: `GuidedStep` 인터페이스 추가 (REQ 문서에 정의된 대로)
- 현재 컴포넌트 내부에서만 사용하므로 별도 export는 불필요할 수 있음

**검증 기준**:
- 탭 헤더에 완료/현재/미방문 상태 아이콘 표시
- 완료된 탭에 선택값 미리보기 표시
- 미방문 탭 클릭 불가
- 기존 동작 (자동 전환, 이전 탭 수정, 자동 제출) 유지

---

## 실행 순서 및 의존관계

```
Phase 1 (백엔드, 병렬 가능):
  Task 1: REQ-2 포맷 방어 (파싱 유틸 + 프롬프트)
  Task 2: REQ-3 장비선택 에러 (백엔드 guard)

Phase 2 (프론트엔드, Phase 1과 병렬 가능):
  Task 2: REQ-3 프론트 guard (chat-page, use-chat-session)
  Task 3: REQ-6 탭 UI (guided-selection-panel)

Phase 3 (통합 테스트):
  - GLM 모델로 전체 플로우 테스트
  - gpt-oss-20b로 기존 플로우 회귀 테스트
  - 탭 UI 동작 확인 (4단계 완주, 이전 탭 수정, 자동 제출)
```

## 수정 파일 요약

| 파일 | Task | 변경 범위 |
|------|------|-----------|
| `backend/llm_infrastructure/llm/langgraph_agent.py` | 1 | 파싱 유틸 강화, fallback 체인 |
| `backend/llm_infrastructure/llm/prompts/auto_parse_v1.yaml` | 1 | 포맷 지시 강화 |
| `backend/llm_infrastructure/llm/prompts/router_v1.yaml` | 1 | 포맷 지시 강화 |
| `backend/api/routers/agent.py` | 2 | resume guard 에러코드 |
| `frontend/src/features/chat/pages/chat-page.tsx` | 2 | pending 조건 강화 |
| `frontend/src/features/chat/hooks/use-chat-session.ts` | 2 | 에러 핸들링 |
| `frontend/src/features/chat/components/guided-selection-panel.tsx` | 3 | 탭 상태 아이콘, 선택값 미리보기 |
| `frontend/src/features/chat/types.ts` | 3 | GuidedStep 타입 (선택) |
