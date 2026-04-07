# Task: auto-parse typo device normalization plan

Status: Draft
Owner: hskim
Branch or worktree: dev
Created: 2026-04-07

## Goal

오타가 포함된 장비명 질의(예: `SUPRAvvplus APC valve 교체 방법`)에서
`장비를 자동으로 파싱하지 못했습니다. 기기를 검색하시겠습니까?`
legacy 프롬프트가 표시되는 문제를 제거한다.

**핵심 결론 (2026-04-07 사용자 확인):**
> "중간에 장비를 묻는 플로우는 예전에 있던 flow야. 현재는 legacy로 빼놓은 기능이야."
> "기기를 선택하는 UI가 있어서, 중간에 따로 파싱하는 질문은 안하기로 했어."

현재 시스템 설계:
- 장비 선택 UI가 별도로 존재함
- 의도한 흐름: `UI 장비 선택 → 질의 → auto_parse → 검색 → 답변`
- 중간에 장비를 묻는 인터럽트 프롬프트는 **전면 비활성화** 대상

## Why

### 현재 관측된 문제

질의:

```text
SUPRAvvplus APC valve 교체 방법
```

현재 사용자 경험:

```text
장비를 자동으로 파싱하지 못했습니다. 기기를 검색하시겠습니까?
```

사용자 기대:

- `SUPRAvvplus`를 `SUPRA Vplus`로 복구하거나,
- 적어도 `SUPRA Vplus를 의미하나요?` 같은 확인형 UX로 이어져야 한다.

### 현재 코드 기준 확정 사실

1. `auto_parse`는 장비명 추출을 먼저 시도한다.
2. `SUPRAvvplus` 같은 오타는 현재 `auto_parse` 단계에서 안정적으로 복구되지 않는다.
3. `APC`는 component 성격이 강하고, 테스트상 **장비로 파싱돼도 유효 장비로 인정되지 않는 경우**가 있다.
4. 따라서 이 질의는 결과적으로 **유효한 auto-parsed device가 없음** 상태가 된다.
5. backend의 `_should_suggest_additional_device_search(...)`는
   "auto_parse는 수행됐지만 device는 못 찾았다"고 판단해
   추가 장비 검색 제안 플로우를 띄운다.

### 현재 문제의 본질

이 문제는 retrieval failure가 아니라,
**검색 이전 단계의 auto-parse / normalization / UX 설계 문제**다.

즉,

> 장비명 오타를 복구하지 못하고,
> component(APC)와 device를 구분만 한 채,
> 사용자에게 너무 이른 시점에 "장비를 못 찾았다"고 말하는 구조가 문제다.

## Contracts To Preserve

- C-API-001: 응답 metadata 구조 유지
- C-API-002: interrupt/resume 호환 유지
- C-UI-001: 기존 guided confirm 흐름의 의미 보존

## Contracts To Update

- None

## Allowed Files

- `backend/llm_infrastructure/llm/langgraph_agent.py`
- `backend/services/agents/langgraph_rag_agent.py`
- `backend/api/routers/agent.py`
- `backend/tests/test_auto_parse_node_events.py`
- `backend/tests/test_agent_additional_device_search_flag.py`
- `frontend/src/features/chat/hooks/use-chat-session.ts`
- `frontend/src/features/chat/pages/chat-page.tsx`
- `frontend/src/features/chat/types.ts`
- `docs/tasks/TASK-20260407-auto-parse-typo-device-normalization-plan.md`

## Out Of Scope

- retrieval ranking(B/F) 자체 수정
- answer/group selection 로직 수정
- expand_related_docs_node 수정
- reranker/ES scoring 조정
- React agent 구조 변경

## Root Cause Breakdown

### RC-1: auto_parse 단계에서 typo-tolerant device normalization이 없다

현재 retrieval 단계에서는 장비 alias/fuzzy match 로직이 존재하지만,
`auto_parse` 단계에서는 그 수준의 typo 복구가 선행되지 않는다.

결과:

- `SUPRAvvplus` → 장비 미검출
- 이후 backend는 "device를 못 찾음"으로 해석

> **[2026-04-07 검증 의견]** RC-1의 전제가 사실과 다를 수 있음.
> `_extract_devices_from_query`에는 이미 Phase 2 fuzzy matching
> (rapidfuzz WRatio, cutoff=82)이 구현되어 있고, 실제 production
> device_names(118개)로 테스트한 결과 오타 쿼리에서 device를 **정상 감지**함:
>
> ```python
> _extract_devices_from_query(device_names, "SUPRAvvplus APC valve 교체 방법")
> # → ["SUPRA_VPLUS"]  (Phase 2 fuzzy, WRatio=90.9)
>
> _extract_devices_from_query(device_names, "Suppravvplus apc 교체 방법")
> # → ["SUPRA_VPLUS"]  (Phase 2 fuzzy, WRatio=90.9)
> ```
>
> 따라서 backend의 device 감지 자체는 이미 작동하며,
> 프롬프트가 뜨는 진짜 원인은 RC-1이 아니라 아래 RC-4, RC-5일 가능성이 높음.

### RC-2: component와 device가 UX 레벨에서 명확히 분리되지 않는다

`APC`는 component이지만,
사용자 입장에서는 "왜 APC를 썼는데 장비를 못 찾았다고 하지?"라는 혼란이 생긴다.

즉, 시스템 내부적으로는 맞는 처리여도,
UX 메시지가 현재 실패 원인을 제대로 설명하지 못한다.

### RC-3: 실패 UX가 검색형으로만 열려 있다

현재는:

```text
장비를 자동으로 파싱하지 못했습니다. 기기를 검색하시겠습니까?
```

이 메시지는 사용자가 다시 장비를 입력하거나 검색하게 만든다.
하지만 이 케이스는 실제로는 "못 찾음"보다
**"거의 찾았는데 오타라서 확정 못 함"**에 가깝다.

따라서 검색형 UX보다,
**보정 후보 확인형 UX**가 더 맞다.

### RC-4: 프론트엔드 `fallbackSuggest` OR 로직이 백엔드 판정을 override한다 (신규)

> **[2026-04-07 코드 분석에서 발견]**

`use-chat-session.ts:948-950`에서:

```javascript
const shouldSuggestAdditionalDeviceSearch =
    typeof res.suggest_additional_device_search === "boolean"
    ? (res.suggest_additional_device_search || fallbackSuggest)  // ← OR 결합
    : fallbackSuggest;
```

백엔드가 `suggest_additional_device_search=false`를 보내도,
프론트엔드의 `fallbackSuggest`가 `true`면 override된다.

`fallbackSuggest`가 true가 되는 조건:

```javascript
fallbackSuggest = hasAutoParseResult && !hasParsedDeviceSignal
```

- `hasAutoParseResult=true`: `detected_language` 또는 `auto_parse` 이벤트가 있으면 true
- `hasParsedDeviceSignal=false`: SSE `final` 이벤트의 `res.auto_parse.devices`에
  device 정보가 누락되거나, `isEffectiveParsedDevice` 필터에 걸리면 false

이 경우 backend는 정상인데 frontend가 단독으로 프롬프트를 띄울 수 있다.

### RC-5: guided_confirm interrupt 응답에서 answer가 비어있다 (신규)

> **[2026-04-07 코드 분석에서 발견]**

guided_confirm interrupt가 발생하면:

```python
# agent.py:1601-1633
resp = AgentResponse(
    answer=result.get("answer", "") or "",  # ← 아직 answer 생성 전이므로 빈 문자열
    interrupted=True,
    ...
)
```

프론트엔드에서:

```javascript
const hasCompletedAnswer = typeof res.answer === "string" && res.answer.trim().length > 0;
// → false (answer가 비어있으므로)

const shouldShowMissingDevicePrompt =
    shouldSuggestAdditionalDeviceSearch && !hasCompletedAnswer;
// → RC-4에 의해 suggest=true이고, answer 비어있으므로 → 프롬프트 발동
```

RC-4 + RC-5가 결합되면, **백엔드가 device를 정상 감지했음에도**
프론트엔드에서 legacy 프롬프트가 표시될 수 있다.

## Desired Behavior

질의:

```text
SUPRAvvplus APC valve 교체 방법
```

원하는 동작 우선순위:

1. 내부적으로 `SUPRA Vplus`로 고신뢰 보정
2. 또는 사용자에게 `SUPRA Vplus를 의미하나요?` 제안
3. component `APC valve`는 유지
4. 이후 setup/SOP 검색으로 자연스럽게 이어짐

## Improvement Plan

### A. auto_parse 단계에 device fuzzy normalization 추가

> **[2026-04-07 검증 결과] 이미 구현되어 있음 — 추가 작업 불필요.**
>
> `_extract_devices_from_query`에 Phase 2 fuzzy matching
> (rapidfuzz WRatio, cutoff=82)이 이미 포함되어 있으며,
> `SUPRAvvplus` → `SUPRA_VPLUS` (score=90.9)로 정상 감지된다.
>
> 따라서 이 항목은 신규 구현이 아니라 **동작 확인 완료** 상태.
> 프롬프트가 뜨는 원인은 A가 아니라 E, F에 있다.

#### (참고) 기존 Phase 2 동작 흐름

```
_extract_devices_from_query(device_names, query)
  Phase 1: exact substring match (compact)     → 실패 (오타)
  Phase 2: token-level fuzzy (rapidfuzz WRatio) → "SUPRA_VPLUS" (score=90.9)
```

### B. component와 device를 분리 저장/표시 (2순위)

#### 방향

auto_parse 결과를 명시적으로 분리:

- `auto_parsed_devices`
- `auto_parsed_components`

#### 기대 효과

- APC는 잡혔지만 device는 못 잡은 상황을 더 정확히 설명 가능
- UX 메시지를 "장비를 못 찾음" 하나로 뭉개지 않게 됨

#### 주의

- API payload/metadata 필드 확장 시 contract 영향 검토 필요

### C. 추가 장비 검색 UX를 보정 후보 확인 UX로 전환 (3순위)

#### 현재 UX

```text
장비를 자동으로 파싱하지 못했습니다. 기기를 검색하시겠습니까?
```

#### 목표 UX

예시:

```text
장비명을 정확히 확정하지 못했습니다.
SUPRAvvplus를 SUPRA Vplus로 보정할까요?
```

또는:

```text
의도한 장비를 선택해주세요:
- SUPRA Vplus
- SUPRA V
- 직접 검색
```

#### 기대 효과

- 사용자 인지 부하 감소
- 실제 실패 원인(오타)을 더 잘 설명
- 검색 UX보다 빠르게 복구 가능

### D. 고신뢰 typo correction은 자동 적용

#### 방향

다음 조건을 모두 만족하면 사용자 확인 없이 내부 적용 가능:

- setup/procedure 계열 route
- fuzzy top1 score 높음
- top1-top2 격차 충분함
- component/문맥도 일치함

#### 기대 효과

- 가장 자연스러운 UX
- 장비 확인 인터럽트 자체 감소

#### 주의

- 잘못된 자동 보정은 큰 side effect
- 따라서 가장 보수적으로 적용해야 함

### E. 프론트엔드 `fallbackSuggest` OR 로직 수정 (신규, RC-4 대응)

> **[2026-04-07 코드 분석에서 도출]**

#### 현재 (문제)

```javascript
// use-chat-session.ts:948-950
shouldSuggest = res.suggest_additional_device_search || fallbackSuggest
//              ↑ false (백엔드 정상)           ↑ true → override!
```

백엔드가 명시적으로 `false`를 보내도 프론트엔드 fallback이 `true`로 덮어쓴다.

#### 수정안

```javascript
shouldSuggest = typeof res.suggest_additional_device_search === "boolean"
    ? res.suggest_additional_device_search   // 백엔드 판단 우선 (신뢰)
    : fallbackSuggest;                       // 백엔드 값 없을 때만 fallback
```

#### 기대 효과

- 백엔드가 device를 정상 감지했으면 프론트엔드가 override하지 않음
- fallback은 구버전 호환이나 streaming 미완료 시에만 동작

#### 주의

- 구버전 백엔드가 `suggest_additional_device_search` 필드를 보내지 않는 경우
  fallback이 여전히 필요하므로 완전 제거는 불가

### F. interrupt 응답에서 missing_device 프롬프트 억제 (신규, RC-5 대응)

> **[2026-04-07 코드 분석에서 도출]**

#### 현재 (문제)

guided_confirm interrupt 시 `answer=""`이므로 `!hasCompletedAnswer=true`.
이것이 RC-4의 `shouldSuggest=true`와 결합되면 프롬프트 발동.

#### 수정안

```javascript
const shouldShowMissingDevicePrompt =
    shouldSuggestAdditionalDeviceSearch &&
    !hasCompletedAnswer &&
    !res.interrupted;  // ← interrupt 응답에서는 표시 안 함
```

#### 기대 효과

- interrupt는 정상 흐름 (guided_confirm)이므로,
  이 시점에서 "장비를 못 찾았다" 프롬프트를 띄우는 것은 부적절
- interrupt resume 후 답변이 완성되면 그때 판단

## Recommended Rollout Order (최종)

> **[2026-04-07 사용자 확인 후 최종 결정]**
>
> 전체 `shouldShowMissingDevicePrompt` 흐름이 legacy이므로,
> E/F의 세부 로직 수정 대신 **프롬프트 자체를 전면 비활성화**한다.

1. **G** `shouldShowMissingDevicePrompt`를 항상 `false`로 설정 — legacy 흐름 전면 비활성화
2. **진단 확인** — 적용 후 프롬프트 소멸 여부 검증
3. (선택) 후속 정리: 관련 dead code 제거 (`pendingRegeneration`, `suggestedDevices` 등)

## Why This Order (최종)

- 사용자 확인: 중간 장비 질문 흐름 자체가 legacy이며, 장비 선택 UI가 이를 대체
- E/F는 legacy 로직을 "고치는" 것이므로 불필요. 전면 비활성화가 올바른 해법
- dead code 제거는 기능 확인 후 별도 정리 작업으로

## Risks

- 오타 보정이 잘못된 장비로 수렴할 수 있음
- APC 같은 component를 device로 오인하는 기존 규칙과 충돌 가능
- guided confirm / resume payload가 복잡해질 수 있음
- frontend가 새 보정 후보 UX를 제대로 렌더링하지 못할 수 있음

## Verification Plan

```bash
# 1. auto_parse typo query regression (기존)
cd backend && uv run pytest tests/test_auto_parse_node_events.py -v

# 2. additional device search suggestion behavior (기존)
cd backend && uv run pytest tests/test_agent_additional_device_search_flag.py -v

# 3. FE 진단 로그 (E+F 적용 전 — 원인 확인)
# use-chat-session.ts handleAgentResponse 내 console.log 추가:
#   res.auto_parse, res.suggest_additional_device_search,
#   fallbackSuggest, hasParsedDeviceSignal, hasCompletedAnswer, res.interrupted

# 4. E+F 적용 후 regression
cd frontend && npm run test

# 5. 실제 flow 검증 (E+F 적용 후)
# 쿼리: "SUPRAvvplus APC valve 교체 방법"
# 기대: missing_device 프롬프트 없이 guided_confirm 또는 검색 진행
```

## Verification Results

- command: `pending`
  - result: pending
  - note: implementation not started

## Handoff

- Current status: G 구현 진행 중 — legacy 프롬프트 전면 비활성화
- Remaining TODOs:
  1. ~~auto_parse typo normalization 설계 확정~~ → 이미 구현 확인됨 (Phase 2 fuzzy)
  2. ~~E/F 구현~~ → legacy 흐름 전면 비활성화(G)로 대체
  3. **G 구현**: `shouldShowMissingDevicePrompt`를 항상 `false`로 설정
  4. 적용 후 FE 테스트 및 실 쿼리 검증
  5. (후속 정리) 관련 dead code 제거
- Whether `Allowed Files` changed and why: no
- Whether `Contracts To Update` is expected: maybe (if API metadata fields expand)

## Change Log

- 2026-04-07: task created from typo-device auto-parse failure analysis
- 2026-04-07: RC-1 검증 — `_extract_devices_from_query` Phase 2 fuzzy matching이 이미 구현되어 있고, production device_names(118개)에서 오타 쿼리 정상 감지 확인 (WRatio=90.9). RC-4(FE fallbackSuggest OR override), RC-5(interrupt 시 answer 비어있음) 신규 발견. 우선순위 재조정: Plan A 불필요, Plan E+F 최우선
- 2026-04-07: 사용자 확인 — 중간 장비 질문 흐름 전체가 legacy. 장비 선택 UI가 별도 존재하므로 `shouldShowMissingDevicePrompt` 전면 비활성화(Plan G). E/F 불필요

## Final Check

- [ ] Diff stayed inside allowed files, or this doc was updated first
- [ ] Protected contract IDs were re-checked
- [ ] Verification commands were run, or blockers were recorded
- [ ] Any contract changes were reflected in `product-contract.md`
- [ ] Remaining risks and follow-ups were documented
