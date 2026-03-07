# UI/Chat 개선 요구사항 (2026-03-06)

## 요약

첫 질문 플로우 및 모델 호환성 관련 6개 개선 사항.

---

## REQ-1: "건너뛰기" 버튼에서 Recommend 태그 제거

### 현상
- 장비(device) / 설비ID(equip_id) 선택 시, 파싱 결과가 없으면 "건너뛰기"에 `recommended: True`가 설정됨
- 사용자에게 "건너뛰기"가 추천처럼 보여 혼란 야기

### 원인
- `backend/llm_infrastructure/llm/langgraph_agent.py:3082-3084`
  ```python
  device_options.append(
      {"value": "__skip__", "label": "건너뛰기", "recommended": len(parsed_devices) == 0}
  )
  ```
- `backend/llm_infrastructure/llm/langgraph_agent.py:3095-3101`
  ```python
  equip_id_options.append({
      "value": "__skip__",
      "label": "건너뛰기",
      "recommended": len(parsed_equip_ids) == 0,
  })
  ```

### 변경 사항
- `"건너뛰기"`의 `recommended`를 항상 `False`로 설정
- 파싱 결과가 없을 때도 건너뛰기가 추천으로 표시되지 않도록 함

### 수정 대상
| 파일 | 라인 | 변경 |
|------|------|------|
| `backend/llm_infrastructure/llm/langgraph_agent.py` | 3082-3084 | `"recommended": False` |
| `backend/llm_infrastructure/llm/langgraph_agent.py` | 3095-3101 | `"recommended": False` |

### 검증
- 파싱 실패 시 "건너뛰기"에 Recommend 태그 미표시 확인
- 파싱 성공 시 첫 번째 장비에만 Recommend 태그 표시 확인

---

## REQ-2: 모델 변경 시 LLM 출력 포맷 에러 방어

### 현상
- 모델을 GLM-4.7-Flash 등으로 변경하면 기존 포맷(JSON)을 따르지 않아 파싱 에러 발생
- 특히 `auto_parse`, `judge`, `router`, `query generation` 단계에서 실패

### 원인
- LLM 출력 파싱이 regex 기반 JSON 추출에 의존 (`\{.*\}` 패턴)
- 모델별 출력 스타일 차이 (마크다운 래핑, 설명 추가, 다른 JSON 구조 등)
- 파싱 실패 시 fallback이 일관되지 않음

### 영향받는 파싱 함수
| 함수 | 파일:라인 | 역할 |
|------|-----------|------|
| `_parse_auto_parse_result()` | `langgraph_agent.py:2252-2328` | 장비/문서종류/언어 파싱 |
| `_parse_queries()` | `langgraph_agent.py:482-647` | 검색 쿼리 추출 |
| `_parse_route()` | `langgraph_agent.py:417-424` | 라우팅 결정 |
| `judge_node()` | `langgraph_agent.py:2006-2062` | 답변 충실도 판단 |
| `_parse_needs_history_from_text()` | `langgraph_agent.py:2683-2709` | 히스토리 사용 여부 |

### 변경 사항
1. **모든 파싱 함수에 일관된 fallback 체인 적용**:
   - Step 1: JSON 직접 파싱
   - Step 2: 코드블록 내 JSON 추출 (````json ... ``` `)
   - Step 3: Regex `\{.*\}` 추출
   - Step 4: 기본값 반환 + 경고 로그
2. **judge 응답은 response_format 기반 JSON 강제 + 재시도 제한**:
   - `judge_node()`에서 가능하면 `response_format=json_object` 사용
   - 파싱 실패 시 무한 재시도/폭주 방지: 최대 재시도 횟수 제한 + 안전한 기본값으로 종료
3. **translate/query rewrite 결과 정규화**:
   - 모델이 분석문/번호형 설명을 섞어 내보내도 downstream에서 깨지지 않도록 정규화
4. **프롬프트에 출력 포맷 강화**:
   - `prompts/auto_parse_v1.yaml`: JSON-only 출력 지시 강화
   - 마크다운 코드블록 금지 명시
   - `prompts/router_v1.yaml`: 포맷 지시 강화
5. **에러 발생 시 사용자에게 표시하지 않고 graceful degradation**

### 수정 대상
| 파일 | 변경 |
|------|------|
| `backend/llm_infrastructure/llm/langgraph_agent.py` | 파싱 함수들에 통일된 fallback 적용 |
| `backend/llm_infrastructure/llm/prompts/auto_parse_v1.yaml` | 포맷 지시 강화 |
| `backend/llm_infrastructure/llm/prompts/router_v1.yaml` | 포맷 지시 강화 |

### 검증
- GLM-4.7-Flash로 10개 질문 테스트 → 파싱 에러 0건
- 기존 모델(gpt-oss-20b)에서도 정상 동작 확인

---

## REQ-3: 답변 완료 후 장비선택 버튼 에러 수정

### 현상
- 대답이 끝난 후 장비선택 버튼이 표시될 때 에러 발생

### 원인 조사 필요
- `backend/llm_infrastructure/llm/langgraph_agent.py:3285-3372` — `device_selection_node()`
- `backend/api/routers/devices.py:133-165` — `/api/device-catalog` API
- `frontend/src/features/chat/pages/chat-page.tsx:506-517` — `DeviceSelectionPanel` 렌더링
- `frontend/src/features/chat/hooks/use-chat-session.ts:907-930` — `submitDeviceSelection()`

### 변경 사항
- 에러 재현 후 정확한 원인 파악 필요
- 우선 방어 변경(요구사항 문서 기준):
  1. 프론트: pending interrupt가 있을 때만 버튼/패널 노출
  2. 백엔드: pending 없는 resume 요청은 명확한 오류 코드/메시지 반환
  3. (부가) API 실패/예상치 못한 payload에 대한 에러 처리로 crash 방지

### 수정 대상
| 파일 | 변경 |
|------|------|
| `backend/llm_infrastructure/llm/langgraph_agent.py` | device_selection_node 에러 핸들링 |
| `frontend/src/features/chat/pages/chat-page.tsx` | pending interrupt 조건에 따른 패널 노출 제어 |
| `backend/api/routers/devices.py` | API 에러 응답 개선 |
| `backend/api/routers/agent.py` | pending 없는 resume 요청 오류 코드/메시지 |

### 검증
- 답변 완료 후 장비선택 패널 정상 표시
- pending interrupt 없는 완료 상태에서는 장비선택 버튼/패널이 노출되지 않음
- pending 없는 resume 요청 시 4xx + 명확한 에러 메시지 확인
- API 에러 시 사용자에게 에러 메시지 표시 (crash 방지)

---

## REQ-4: "SOP 조회" → "절차조회"로 라벨 변경

### 현상
- 첫 질문 시 태스크 선택 옵션에 "SOP 조회"로 표시됨
- 사용자 관점에서 "SOP"는 내부 용어 → "절차조회"가 더 직관적

### 원인
- `backend/llm_infrastructure/llm/langgraph_agent.py:3104-3108`
  ```python
  task_options = [
      {"value": "sop", "label": "SOP 조회", "recommended": False},
      {"value": "issue", "label": "이슈조회", "recommended": True},
      {"value": "all", "label": "전체조회", "recommended": False},
  ]
  ```

### 변경 사항
- `"SOP 조회"` → `"절차조회"`로 라벨 변경
- value는 `"sop"` 유지 (백엔드 로직 변경 없음)

### 수정 대상
| 파일 | 라인 | 변경 |
|------|------|------|
| `backend/llm_infrastructure/llm/langgraph_agent.py` | 3105 | `"label": "절차조회"` |

### 검증
- 프론트엔드에서 "절차조회" 라벨 표시 확인
- 선택 후 백엔드 task_mode="sop" 정상 전달 확인

---

## REQ-5: 업무 선택 기반 고정 문서군 검색 규칙으로 변경 (추가 문서유형 질문 없음)

### 현상
- 현재 task_mode 선택 후 바로 검색 진행
- 업무 의도와 실제 검색 문서군 매핑이 기대와 다름

### 현재 동작
- `task_mode == "sop"` → `expand_doc_type_selection(["sop"])` (SOP만 검색)
- `task_mode == "issue"` → `expand_doc_type_selection(["myservice", "gcb", "ts"])` (3종 전체 검색)
- 관련 코드: `langgraph_agent.py:3164-3176`

### 변경 사항
- 세부 문서유형 추가 질문은 만들지 않음
- 업무 선택 시 백엔드에서 고정 문서군으로 검색
  - `절차조회(sop)` 선택 시: `SOP + TS + SETUP` 전체 검색
  - `이슈조회(issue)` 선택 시: `gcb + myservice` 전체 검색
  - `전체조회(all)` 선택 시: 기존처럼 문서유형 필터 미적용

#### 백엔드 변경
1. `auto_parse_confirm_node()`의 `task_mode`별 매핑만 변경
2. `resume_auto_parse_confirm_node()`에서 task_mode 수신 시 고정 문서군 반영
3. 매핑 규칙:

| 업무 선택 | ES doc_types |
|-----------|-------------|
| 절차조회(`sop`) | `expand_doc_type_selection(["sop", "ts", "setup"])` |
| 이슈조회(`issue`) | `expand_doc_type_selection(["gcb", "myservice"])` |
| 전체조회(`all`) | 필터 없음 |

#### 프론트엔드 변경
1. 세부 문서종류 질문 step 추가 없음
2. task 선택 UI는 현재 구조 유지

### 수정 대상
| 파일 | 변경 |
|------|------|
| `backend/llm_infrastructure/llm/langgraph_agent.py` | task_mode별 doc_type 매핑 수정 |
| `backend/api/routers/agent.py` | task_mode 전달/검증 동작 확인 (스키마 변경 없음) |

### 검증
- "절차조회" 선택 시 SOP/TS/SETUP 문서군에서 모두 검색되는지 확인
- "이슈조회" 선택 시 gcb/myservice 문서군에서 모두 검색되는지 확인
- "전체조회" 선택 시 기존처럼 필터 없이 검색되는지 확인

---

## REQ-6: 탭 기반 스텝 인디케이터 (Guided Selection)

### 현상
- 현재 guided selection이 한 번에 하나씩 질문을 던지는 방식
- 사용자는 "질문이 몇 개나 남았지?" "언제까지 답해야 하지?" 라는 불안감을 느낌
- 전체 진행 상황을 알 수 없음

### 변경 사항

#### UI 구조
현재 (순차 질문):
```
Q1: 언어를 선택하세요 → [한국어] [영어] [일본어]
(선택 후)
Q2: 장비를 선택하세요 → [SUPRA N] [INTEGER plus] ...
(선택 후)
Q3: ...
```

변경 후 (탭 기반):
```
┌──────┬──────┬──────┬────────┐
│ 언어 │ 장비 │ 설비ID│ 업무   │
│ ●    │ ○    │ ○    │ ○      │
└──────┴──────┴──────┴────────┘
  [한국어]  [영어]  [일본어]

  → 선택 시 자동으로 다음 탭으로 이동
```

#### 동작 방식
1. 모든 step을 탭 헤더로 표시 (언어 / 장비 / 설비ID / 업무)
2. 현재 활성 탭 강조 (●), 미완료 탭은 비활성 (○), 완료 탭은 체크 (✓)
3. 옵션 선택 시 자동으로 다음 탭으로 전환
4. 이전 탭 클릭 시 되돌아가서 선택 변경 가능
5. 마지막 step 선택 완료 시 자동 제출 (또는 "확인" 버튼)
6. 각 탭 헤더에 선택된 값 미리보기 표시 (예: `언어 ✓ 한국어`)

#### 스텝 구성
- **절차조회 선택 시**: 언어 → 장비 → 설비ID → 업무(절차조회) 완료 후, 백엔드에서 SOP/TS/SETUP 전체 검색
- **이슈조회 선택 시**: 언어 → 장비 → 설비ID → 업무(이슈조회) 완료 후, 백엔드에서 gcb/myservice 전체 검색
- **전체조회 선택 시**: 언어 → 장비 → 설비ID → 업무(전체조회) 완료 후, 기존 검색 로직 수행

#### 프론트엔드 변경
1. `guided-selection-panel.tsx` — 전면 리팩토링
   - 탭 헤더 컴포넌트 추가
   - 스텝 상태 관리 (current, completed, pending)
   - 탭 클릭으로 이전 스텝 수정 가능
   - 선택 시 자동 다음 탭 전환 애니메이션
2. `types.ts` — GuidedSelection에 step 메타데이터 추가
   ```typescript
   interface GuidedStep {
     key: string;         // "language" | "device" | "equip_id" | "task"
     label: string;       // "언어" | "장비" | "설비ID" | "업무"
     selectedValue?: string;
     status: "pending" | "active" | "completed";
   }
   ```

#### 백엔드 변경
- 탭 UI 자체를 위해서는 별도 백엔드 변경 없음
- 단, REQ-5의 task_mode별 고정 문서군 매핑 변경은 백엔드에서 필요

### 수정 대상
| 파일 | 변경 |
|------|------|
| `frontend/src/features/chat/components/guided-selection-panel.tsx` | 탭 UI 전면 리팩토링 |
| `frontend/src/features/chat/types.ts` | GuidedStep 타입 추가 |
| `frontend/src/features/chat/hooks/use-chat-session.ts` | 탭 상태 관리 로직 |

### 검증
- 탭 헤더에 전체 step 수 표시 확인
- 선택 시 자동 다음 탭 전환 확인
- 이전 탭 클릭 시 선택 변경 가능 확인
- 모든 step 완료 시 자동 제출 확인
- 업무 선택 후 백엔드 검색 문서군이 규칙대로 적용되는지 확인

---

## 우선순위 및 의존관계

| 순서 | 항목 | 난이도 | 예상 소요 |
|------|------|--------|-----------|
| 1 | REQ-4: 라벨 변경 | 낮음 | 5분 |
| 2 | REQ-1: 건너뛰기 recommend 제거 | 낮음 | 10분 |
| 3 | REQ-2: 포맷 에러 방어 | 중간 | 1-2시간 |
| 4 | REQ-3: 장비선택 에러 수정 | 중간 | 에러 재현 필요 |
| 5 | REQ-5: 업무별 고정 문서군 매핑 변경 | 중간 | 1시간 |
| 6 | REQ-6: 탭 기반 스텝 인디케이터 | 높음 | 3-4시간 |

### 의존관계
- REQ-4 → REQ-5 (라벨 변경 후 task 의미를 기준으로 매핑 변경)
- REQ-5와 REQ-6은 병렬 진행 가능
- REQ-6은 guided-selection-panel.tsx 전면 리팩토링이므로, REQ-1도 같이 반영 권장
