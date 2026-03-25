# RAG 에이전트 플로우 가이드

> 작성일: 2026-03-25
> "사용자가 질문을 보내면 답변이 나오기까지 어떤 일이 일어나는가?"를 코드 위치와 함께 정리한 문서

---

## 한눈에 보는 전체 흐름

```
사용자가 채팅창에 질문 입력
  ↓
[FE] 장비/문서타입/호기 선택 (guided-selection-panel)
  ↓
[FE → BE] POST /api/agent/run 으로 요청 전송
  ↓
[BE] 요청 파싱 + 어떤 에이전트를 쓸지 결정
  ↓
[에이전트 내부 - 순서대로 실행]
  1. 질문 분류     (route)         — "이 질문이 SOP? TS? 일반?"
  2. 검색어 생성   (mq)            — "이 질문으로 뭘 검색할까?"
  3. 문서 검색     (retrieve)      — "ES에서 문서 가져오기"
  4. 문서 확장     (expand_related) — "관련 페이지 더 가져오기"
  5. 답변 생성     (answer)        — "검색 결과로 답변 작성"
  6. 답변 검증     (judge)         — "답변이 근거 있는가?"
  7. (검증 실패시) 재시도           — "검색어 바꿔서 다시"
  ↓
[BE → FE] 답변 + 참조문서 + 메타데이터 응답
  ↓
[FE] 왼쪽에 답변, 오른쪽에 문서 이미지 표시
```

---

## 모듈 0: 시작점 — 사용자 요청이 들어오는 곳

### 어디서 시작하나?

사용자가 채팅창에서 "전송"을 누르면:

1. **FE** (`frontend/src/features/chat/api.ts:34`)가 `POST /api/agent/run` 호출
2. **BE** (`backend/api/routers/agent.py:1011`)의 `run_agent()` 함수가 받음

### FE가 보내는 데이터

```
{
  message: "ETCH PM 절차 알려줘"     ← 사용자가 입력한 질문
  filter_devices: ["ETCH_A"]          ← 장비 선택 (선택한 경우만)
  filter_doc_types: ["sop"]           ← 문서타입 선택 (선택한 경우만)
  filter_equip_ids: ["EQ001"]         ← 호기 선택 (선택한 경우만)
  auto_parse: true                    ← LLM이 장비/문서타입 자동 파싱할지
  mode: "verified"                    ← 답변 검증+재시도 할지
  max_attempts: 3                     ← 최대 재시도 횟수
  mq_mode: null                       ← 멀티쿼리 모드 (off/fallback/on)
}
```

> 코드 위치: `AgentRequest` 클래스 — `agent.py:335-364`

### BE가 요청을 받으면 하는 일

**1단계: 필터 변환** (`_build_state_overrides`, agent.py:179-243)

FE에서 온 필터를 에이전트가 이해할 수 있는 형태로 바꾼다.

```
FE가 보낸 것                    →  에이전트가 쓰는 것
─────────────────────────────────────────────────────
filter_devices: ["ETCH_A"]      →  selected_devices: ["ETCH_A"]
filter_doc_types: ["sop"]       →  selected_doc_types: ["sop","pems"]  (그룹 확장)
                                   + task_mode: "sop"  (문서타입에서 추론)
filter_equip_ids: ["EQ001"]     →  selected_equip_ids: ["EQ001"]
message (질문 텍스트)            →  detected_language: "ko"  (언어 감지)
```

**2단계: 에이전트 선택** (agent.py:1046-1181)

필터가 있냐 없냐에 따라 다른 에이전트를 만든다.

| 상황 | 어떤 에이전트? | 설명 |
|------|--------------|------|
| 필터 없이 질문만 보냄 | auto_parse 에이전트 | LLM이 질문에서 장비/문서타입 자동 파싱 |
| 장비/문서타입 선택해서 보냄 | override 에이전트 | 자동 파싱 건너뛰고 바로 검색 시작 |
| 이전 대화 이어서 보냄 | resume 에이전트 | 이전 상태에서 이어서 실행 |

> **쉽게 말하면**: 사용자가 guided-selection-panel에서 장비를 골랐으면 "이미 알고 있으니 바로 검색해", 안 골랐으면 "LLM이 질문 분석부터 해"

---

## 모듈 1: Route — "이 질문이 뭔지 분류하기"

### 하는 일

사용자 질문을 3가지 중 하나로 분류한다:
- **setup**: 설치/교체/조립/설정 절차 질문 (예: "PM 절차 알려줘")
- **ts**: 알람/에러/트러블슈팅 질문 (예: "Interlock 발생 원인이 뭐야?")
- **general**: 위 두 개에 해당 안 되는 일반 질문

### 코드 위치
- 노드 함수: `langgraph_agent.py:1479` (`route_node`)
- LLM 프롬프트: `prompts/router_v2.yaml`

### 동작 순서

```
1. 이미 결정된 게 있나 확인
   - FE에서 "이슈조치" 선택했으면 → task_mode="issue" → route="general" 확정 (LLM 안 부름)
   - FE에서 "TSG" 선택했으면    → task_mode="ts"    → route="ts" 확정 (LLM 안 부름)

2. 아직 결정 안 됐으면, 문서타입에서 추론
   - SOP 문서만 선택됐으면  → task_mode="sop"   (route는 LLM이 결정)
   - TS 문서만 선택됐으면   → task_mode="ts"    → route="ts" 확정
   - 이슈 문서만 선택됐으면 → task_mode="issue" → route="general" 확정

3. 그래도 안 됐으면, LLM에게 물어봄
   - 영문 번역된 질문을 router_v2.yaml 프롬프트에 넣어서
   - LLM이 "setup" / "ts" / "general" 중 하나를 응답
   - 우선순위: ts > setup > general (둘 다 해당되면 ts)
```

### route와 task_mode의 차이

헷갈리기 쉬운 부분이라 명확히 정리:

- **route** = "어떤 프롬프트를 쓸까?" 결정
  - `"setup"` → setup용 멀티쿼리, setup용 답변 프롬프트
  - `"ts"` → ts용 멀티쿼리, ts용 답변 프롬프트
  - `"general"` → general용 프롬프트

- **task_mode** = "답변을 어떤 형식으로 만들까?" 결정
  - `"sop"` → SOP 절차형 답변 + 오른쪽에 문서 이미지 표시
  - `"issue"` → 이슈 목록 → 상세 조회 루프
  - `"ts"` → TS 답변 형식

> **예시**: SOP 질문 → task_mode="sop", route="setup" (둘 다 설정됨)
> **예시**: 이슈 질문 → task_mode="issue", route="general" (route는 항상 general)

### 이 단계의 출력

```python
# state에 저장되는 값
{"route": "setup", "parsed_query": {..., "route": "setup"}}
```

이 `route` 값이 다음 단계(멀티쿼리, 답변, 검증)에서 어떤 프롬프트를 쓸지 결정한다.

---

## 모듈 2: MQ (멀티쿼리 생성)

> TODO: 다음 워크스루에서 작성

---

## 모듈 3: ST Gate + ST MQ (보충 TS 쿼리)

> TODO: 다음 워크스루에서 작성

---

## 모듈 4: Retrieve (검색 + 리랭킹)

> TODO: 다음 워크스루에서 작성

---

## 모듈 5: Expand Related (문서 확장)

> TODO: 다음 워크스루에서 작성

---

## 모듈 6: Answer (답변 생성)

> TODO: 다음 워크스루에서 작성

---

## 모듈 7: Judge (답변 검증)

> TODO: 다음 워크스루에서 작성

---

## 모듈 8: Retry (재시도 전략)

> TODO: 다음 워크스루에서 작성
