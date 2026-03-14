# Task: issue search summary-selection flow

Status: completed
Owner: hskim
Branch or worktree: `86ewk6385-1차-PE-피드백-v2` @ `/home/hskim/work/llm-agent-v2`
Created: 2026-03-13
Updated: 2026-03-14

## Goal

`task_mode=issue`에서 검색된 상위 10개 문서를 문서별 이슈 요약 목록으로 먼저 보여주고,
사용자가 원하는 문서를 선택했을 때 해당 문서 1건만 기준으로 상세 답변을 생성하는
summary → confirm → select → detail 플로우를 구현한다.

## Why

현재 이슈검색은 `myservice/gcb/ts` 범위 필터링까지만 실질적으로 동작하고,
사용자가 "검색된 이슈 사례를 비교한 뒤 관련 문서를 골라 상세히 본다"는 UX는 제공하지 못한다.
이 때문에 사용자는 여러 문서 중 무엇이 자신의 증상과 가까운지 판단하기 어렵고,
선택 기반 상세 조회도 할 수 없다.

## Problem Definition

### 현재 문제

- `issue` 모드에서도 검색 결과가 바로 `general_ans` 프롬프트 → 일반 answer 흐름으로 소모된다.
- 검색된 상위 문서 10건에 대한 문서별 요약 목록이 없다.
- 요약 답변 이후 `"상세히 보고싶은 문서가 있습니까?"` yes/no 확인 단계가 없다.
- yes 이후 1~10 문서 선택 UI와 resume 계약이 없다.
- 사용자가 특정 문서를 선택해도 그 문서 1건만 기준으로 상세 답변을 생성하는 live flow가 없다.
- 프론트에는 issue 전용 panel 테스트 (`issue-flow-ui.test.tsx`)가 있지만, 실제 hook/컴포넌트/API 계약 구현은 없다.

### 기존 작업물 현황 (구현 0%, 설계/테스트만 존재)

| 항목 | 상태 | 파일 |
|------|------|------|
| 설계 문서 | ✅ 존재 | `docs/2026-03-09-agent-분기-개선.md` |
| 요약 프롬프트 (ko/en/zh/ja) | ✅ 존재 | `prompts/issue_ans_v2.yaml` + `_en/_zh/_ja` |
| 상세 프롬프트 (ko/en/zh/ja) | ✅ 존재 | `prompts/issue_detail_ans_v2.yaml` + `_en/_zh/_ja` |
| 백엔드 플로우 테스트 | ✅ 작성됨 | `tests/test_issue_flow_interrupts.py` |
| 프론트 플로우 테스트 | ✅ 작성됨 | `chat/__tests__/issue-flow-ui.test.tsx` |
| resume 검증 테스트 | ✅ 작성됨 | `tests/test_agent_guided_resume_validation.py` |
| **PromptSpec 확장** | ❌ 미구현 | `langgraph_agent.py:166-191` — `issue_ans`, `issue_detail_ans` 필드 없음 |
| **load_prompt_spec** | ❌ 미구현 | `langgraph_agent.py:240-295` — issue 프롬프트 로딩 없음 |
| **answer_node 분기** | ❌ 미구현 | `langgraph_agent.py:2528` — `task_mode` 분기 없음 |
| **issue interrupt 노드** | ❌ 미구현 | `langgraph_agent.py` — `issue_confirm`, `issue_case_selection` 노드 없음 |
| **ISSUE_CASE_EMPTY_MESSAGE** | ❌ 미구현 | `langgraph_agent.py` — 상수 없음 |
| **그래프 배선** | ❌ 미구현 | `langgraph_rag_agent.py:460-699` — issue 분기 없음 |
| **API resume 검증** | ❌ 미구현 | `agent.py` — `_validated_guided_resume_decision` 함수 자체 없음 |
| **프론트 hook/컴포넌트** | ❌ 미구현 | `use-chat-session.ts` — issue pending state 없음 |

### 원하는 결과

1. `task_mode=issue`로 검색하면 상위 10개 문서를 가져온다.
2. 첫 답변은 10개 문서 각각에 대한 짧은 이슈 요약 목록이다.
3. 첫 답변 직후 `"상세히 보고싶은 문서가 있습니까?"` yes/no panel이 뜬다.
4. yes를 누르면 1~10 선택 버튼이 뜬다.
5. 사용자가 선택한 문서 1건만 근거로 상세 답변을 생성한다.
6. 상세 답변은 최소 `이슈 내용`과 `해결 방안` 두 섹션을 포함한다.

## Target UX

### Summary Step

- 상위 10개 문서에 대해 번호 목록을 생성한다.
- 각 항목은 아래 정보를 포함한다.
  - 번호
  - 문서 식별 정보 (`title` 또는 `doc_id`)
  - 이슈 한 줄 요약
  - 해결 방안 한 줄 요약

예시:

```text
1. [doc-1] LP1 Mapping Arm Open Alarm
   - 이슈: Mapping arm open alarm 반복 발생
   - 해결: 센서 점검 및 교체
2. [doc-2] RFID Read Failure
   - 이슈: RFID read 불가
   - 해결: GUI 설정 재적용
```

### Confirm Step

- summary answer 직후 guided interrupt를 띄운다.
- 문구: `상세히 보고싶은 문서가 있습니까?`
- 버튼: `Yes`, `No`

### Select Step

- yes 선택 시 1~10 선택 버튼을 띄운다.
- FE는 숫자를 보여주지만, BE resume payload는 `selected_doc_id`를 authoritative key로 사용한다.

### Detail Step

- 선택된 문서 1건만 answer context로 사용한다.
- 상세 답변 형식:
  - `## 이슈 내용`
  - `## 해결 방안`

## Naming Convention (테스트 기준 확정)

기존 테스트 코드에서 사용하는 이름을 canonical name으로 고정한다.
doc에서 사용하던 `issue_doc_select`는 **폐기**하고 `issue_case_selection`으로 통일한다.

| interrupt type | 용도 | 테스트 출처 |
|----------------|------|------------|
| `issue_confirm` | summary/detail 후 yes/no 확인 | `test_issue_flow_interrupts.py:122,166` |
| `issue_case_selection` | 1~10 문서 선택 | `test_issue_flow_interrupts.py:139,181` |
| `issue_sop_confirm` | SOP 연계 확인 (Phase 2) | `test_issue_flow_interrupts.py:152` |

| state 필드 | 테스트 기준 |
|-------------|------------|
| `PromptSpec.issue_ans` | `test_issue_flow_interrupts.py:76` |
| `PromptSpec.issue_detail_ans` | `test_issue_flow_interrupts.py:77` |
| `ISSUE_CASE_EMPTY_MESSAGE` | `test_issue_flow_interrupts.py:14,199` |

## Contracts To Preserve

- C-API-001
- C-API-002

## Contracts To Update

- None

## Allowed Files

- `docs/tasks/TASK-20260313-issue-search-summary-selection-flow.md`
- `backend/llm_infrastructure/llm/langgraph_agent.py`
- `backend/services/agents/langgraph_rag_agent.py`
- `backend/api/routers/agent.py`
- `backend/llm_infrastructure/llm/prompts/issue_ans*.yaml`
- `backend/llm_infrastructure/llm/prompts/issue_detail_ans*.yaml`
- `backend/tests/test_issue_flow_interrupts.py`
- `backend/tests/test_agent_guided_resume_validation.py`
- `tests/api/test_agent_autoparse_confirm_interrupt_resume.py`
- `tests/api/test_agent_response_metadata_contract.py`
- `tests/api/test_agent_interrupt_resume_regression.py`
- `frontend/src/features/chat/hooks/use-chat-session.ts`
- `frontend/src/features/chat/components/issue-confirm-panel.tsx`
- `frontend/src/features/chat/components/issue-case-selection-panel.tsx`
- `frontend/src/features/chat/components/index.ts`
- `frontend/src/features/chat/pages/chat-page.tsx`
- `frontend/src/features/chat/__tests__/issue-flow-ui.test.tsx`
- `frontend/src/features/chat/__tests__/chat-request-payload.test.tsx`

## Out Of Scope

- 검색 품질 튜닝, rerank 정책 변경, synonym 확장
- SOP 연동 (`issue_sop_confirm`) — 노드 스텁만 두고 Phase 2에서 구현
- task_mode radio UX 개편
- guided selection의 언어 단계 제거
- detail answer 이후 "다른 문서 더 보기" 반복 루프 (`post_detail` → loop) — Phase 2
- unrelated prompt/system-wide answer format refactor
- route_node LLM 분류 제거 (task_mode 기반 고정 라우팅) — 별도 task

## Risks

- interrupt/resume 상태 불일치로 기존 `auto_parse_confirm` guided flow가 깨질 수 있음
  - **대응**: 기존 interrupt/resume regression 테스트 통과 필수
- `selected_doc_id`와 1~10 표시 순번 매핑이 어긋날 수 있음
  - **대응**: `issue_case_selection` payload에 `cases` 배열로 `{index, doc_id, title, summary}` 전달, FE는 index 표시 + BE는 `doc_id`로 처리
- interrupted 응답에서 FE가 summary answer를 덮어써 UX가 깨질 수 있음
  - **대응**: interrupted여도 `res.answer`가 있으면 메시지 영역에 유지 (테스트 `issue-flow-ui.test.tsx:221` 에서 검증)
- detail answer가 선택 문서 외 다른 문서를 섞어 인용할 수 있음
  - **대응**: detail answer에 선택 문서 1건의 REFS만 전달
- metadata contract drift로 `C-API-001`이 깨질 수 있음

## Proposed Flow

```text
task_mode=issue
  → retrieve (top 10)
  → expand_related
  → answer_node (issue_ans 프롬프트 → 10건 요약 생성)
  → issue_confirm_node (stage="post_summary")
     ├─ No  → done (종료)
     └─ Yes → issue_case_selection_node (1~10 선택 버튼)
              → 사용자 선택
              → issue_detail_answer_node (선택 doc 1건 → issue_detail_ans)
              → done (종료)
```

Phase 2 확장 (이번 범위 아님):
```text
              → issue_detail_answer_node
              → issue_sop_confirm_node (SOP 확인?)
              → issue_confirm_node (stage="post_detail", 다른 문서 더 보기?)
                 └─ Yes → issue_case_selection_node (루프)
```

## Implementation Plan

### Step 1. PromptSpec 확장 + 프롬프트 로딩

**파일**: `backend/llm_infrastructure/llm/langgraph_agent.py`

1-1. `PromptSpec` dataclass (L166-191)에 필드 추가:
```python
# Issue flow prompts
issue_ans: Optional[PromptTemplate] = None
issue_detail_ans: Optional[PromptTemplate] = None
# Language variants
issue_ans_en: Optional[PromptTemplate] = None
issue_ans_zh: Optional[PromptTemplate] = None
issue_ans_ja: Optional[PromptTemplate] = None
issue_detail_ans_en: Optional[PromptTemplate] = None
issue_detail_ans_zh: Optional[PromptTemplate] = None
issue_detail_ans_ja: Optional[PromptTemplate] = None
```

1-2. `load_prompt_spec()` (L240-295)에서 로딩 추가:
```python
issue_ans = _try_load_prompt("issue_ans", version)
issue_detail_ans = _try_load_prompt("issue_detail_ans", version)
issue_ans_en = _try_load_prompt("issue_ans_en", version)
# ... (zh, ja도 동일)
```

1-3. `ISSUE_CASE_EMPTY_MESSAGE` 상수 정의:
```python
ISSUE_CASE_EMPTY_MESSAGE = "관련 이슈 사례를 찾지 못했습니다."
```

**검증**: `test_issue_flow_interrupts.py`의 `_make_spec()`이 `issue_ans=base, issue_detail_ans=base`로 생성 가능해야 함.

### Step 2. answer_node에서 task_mode="issue" 분기

**파일**: `backend/llm_infrastructure/llm/langgraph_agent.py` — `answer_node` (L2528)

현재 answer_node는 `route` 기반으로만 템플릿을 선택한다.
`task_mode="issue"` 분기를 **route 기반 선택 앞에** 삽입:

```python
def answer_node(state, *, llm, spec):
    task_mode = state.get("task_mode")

    # Issue mode: 검색 결과 0건이면 빈 메시지 반환 (interrupt 없이 종료)
    if task_mode == "issue":
        ref_items = state.get("answer_ref_json") or state.get("ref_json", [])
        if not ref_items:
            return {"answer": ISSUE_CASE_EMPTY_MESSAGE}
        # issue_ans 프롬프트로 요약 생성
        tmpl = _select_issue_template(spec, answer_language, detail=False)
        # ... 요약 생성 후 issue_top10_cases도 state에 저장
        return {"answer": summary_text, "issue_top10_cases": cases_metadata}

    # 기존 route 기반 로직 유지
    ...
```

**핵심**: 검색 결과 0건 → `ISSUE_CASE_EMPTY_MESSAGE` 반환, interrupt 없이 종료.

### Step 3. Issue interrupt 노드 구현

**파일**: `backend/llm_infrastructure/llm/langgraph_agent.py`

3-1. `issue_confirm_node`:
```python
def issue_confirm_node(state):
    """summary/detail 후 '상세히 보고싶은 문서가 있습니까?' yes/no"""
    stage = "post_summary"  # 또는 "post_detail" (Phase 2)
    nonce = str(uuid.uuid4())
    payload = {
        "type": "issue_confirm",
        "nonce": nonce,
        "stage": stage,
        "question": state["query"],
        "instruction": "summary confirm",
        "prompt": "상세히 보고싶은 문서가 있습니까?",
    }
    decision = interrupt(payload)
    # decision: {"type": "issue_confirm", "nonce": nonce, "stage": stage, "confirm": bool}
    if decision.get("confirm"):
        return Command(goto="issue_case_selection")
    return Command(goto="done")
```

3-2. `issue_case_selection_node`:
```python
def issue_case_selection_node(state):
    """1~10 문서 선택 버튼 표시"""
    cases = state.get("issue_top10_cases", [])
    nonce = str(uuid.uuid4())
    payload = {
        "type": "issue_case_selection",
        "nonce": nonce,
        "question": state["query"],
        "instruction": "case pick",
        "cases": cases,  # [{doc_id, title, summary}, ...]
    }
    decision = interrupt(payload)
    # decision: {"type": "issue_case_selection", "nonce": nonce, "selected_doc_id": "doc-1"}
    selected_doc_id = decision.get("selected_doc_id")
    return Command(
        goto="issue_detail_answer",
        update={"issue_selected_doc_id": selected_doc_id},
    )
```

3-3. `issue_detail_answer_node`:
```python
def issue_detail_answer_node(state, *, llm, spec):
    """선택된 문서 1건만 REFS로 상세 답변 생성"""
    selected_doc_id = state.get("issue_selected_doc_id")
    ref_items = [r for r in (state.get("ref_json") or [])
                 if r.get("doc_id") == selected_doc_id]
    # issue_detail_ans 프롬프트 사용
    tmpl = _select_issue_template(spec, answer_language, detail=True)
    # ... LLM 호출
    return {"answer": detail_text}
```

### Step 4. AgentState 확장

**파일**: `backend/llm_infrastructure/llm/langgraph_agent.py` — `AgentState` (L86-161)

```python
# Issue flow fields
issue_top10_cases: List[Dict[str, Any]]  # [{doc_id, title, summary}, ...]
issue_selected_doc_id: Optional[str]
```

### Step 5. 그래프 배선

**파일**: `backend/services/agents/langgraph_rag_agent.py` — `_build_graph` (L460-699)

`expand_related → answer` 이후에 조건부 분기 추가:

```python
# 기존
builder.add_edge("expand_related", "answer")

# answer 이후 조건 분기
def _after_answer(state):
    if state.get("task_mode") == "issue" and state.get("answer") != ISSUE_CASE_EMPTY_MESSAGE:
        return "issue_confirm"
    return "judge"  # 기존 경로

builder.add_conditional_edges("answer", _after_answer, {
    "issue_confirm": "issue_confirm",
    "judge": "judge",
})

# issue 전용 노드 등록 + 배선
builder.add_node("issue_confirm", ...)
builder.add_node("issue_case_selection", ...)
builder.add_node("issue_detail_answer", ...)
# issue_confirm → (Command로 issue_case_selection 또는 done)
# issue_case_selection → (Command로 issue_detail_answer)
builder.add_edge("issue_detail_answer", "done")
```

### Step 6. API resume 검증

**파일**: `backend/api/routers/agent.py`

`_validated_guided_resume_decision()` 함수 신규 추가:
- `issue_confirm`: `stage` 필드 필수 (`"post_summary"` | `"post_detail"`), `confirm` bool 필수
- `issue_case_selection`: `selected_doc_id` 필수
- `issue_sop_confirm`: Phase 2 (스텁만)
- 기존 `auto_parse_confirm` 검증은 그대로 유지

resume 엔드포인트에서 `resume_decision.type`이 위 타입이면 이 함수로 검증 후 `Command(resume=decision)` 전달.

### Step 7. 프론트엔드 hook + 컴포넌트

**파일**: `frontend/src/features/chat/hooks/use-chat-session.ts`

7-1. pending state 추가:
```typescript
pendingIssueConfirm: InterruptPayload | null
pendingIssueCaseSelection: InterruptPayload | null
// Phase 2: pendingIssueSopConfirm
```

7-2. submit 핸들러:
```typescript
submitIssueConfirm(confirm: boolean)     // → resume_decision: {type: "issue_confirm", ...}
submitIssueCaseSelection(docId: string)  // → resume_decision: {type: "issue_case_selection", ...}
```

7-3. interrupt_payload 라우팅: `type` 기반으로 적절한 pending state에 저장.
7-4. interrupted여도 `res.answer`가 있으면 메시지 영역에 유지.

**파일**: `frontend/src/features/chat/components/`

7-5. `IssueConfirmPanel`: "상세히 보고싶은 문서가 있습니까?" + Yes/No 버튼
7-6. `IssueCaseSelectionPanel`: 1~N 번호 버튼 목록 (cases 배열 기반)
7-7. `chat-page.tsx`에서 pending state에 따라 패널 렌더링

## Acceptance Criteria

- [x] `PromptSpec`에 `issue_ans`, `issue_detail_ans` 필드 존재하고 `load_prompt_spec("v2")`에서 로딩됨
- [x] `ISSUE_CASE_EMPTY_MESSAGE` 상수가 export 가능
- [x] `task_mode=issue` + 검색 결과 있음 → `answer`에 10건 요약 포함 + `interrupt_payload.type == "issue_confirm"` + `stage == "post_summary"`
- [x] `task_mode=issue` + 검색 결과 0건 → `answer == ISSUE_CASE_EMPTY_MESSAGE`, interrupt 없이 종료
- [x] `issue_confirm(confirm=false)` → 추가 interrupt 없이 종료
- [x] `issue_confirm(confirm=true)` → `interrupt_payload.type == "issue_case_selection"` + `cases` 배열 포함
- [x] `issue_case_selection(selected_doc_id=X)` → 선택 문서 1건만 REFS로 상세 답변 생성
- [x] 상세 답변에 최소 `이슈 내용`, `해결 방안` 섹션 포함
- [x] 기존 `auto_parse_confirm` guided flow 및 interrupt/resume regression 테스트 통과
- [x] `_validated_guided_resume_decision`에서 `issue_confirm` (stage 필수), `issue_case_selection` (selected_doc_id 필수) 검증
- [x] FE: summary answer가 interrupted 응답에서도 메시지 영역에 유지됨
- [x] FE: Yes/No 버튼과 1~N 선택 버튼이 정상 렌더링됨

## Implementation Order

```
Step 1 (PromptSpec + 로딩)
  → Step 2 (answer_node 분기)
  → Step 3 (interrupt 노드 3개)
  → Step 4 (AgentState 확장)
  → Step 5 (그래프 배선)
  → Step 6 (API resume 검증)
  → Step 7 (FE hook + 컴포넌트)
```

의존 관계:
- Step 1~4는 순서 무관하게 병렬 가능 (PromptSpec/State/노드는 독립)
- Step 5는 Step 1~4 완료 후 진행 (그래프에 노드를 연결하므로)
- Step 6은 Step 5 완료 후 (resume 타입이 그래프와 일치해야)
- Step 7은 Step 6 완료 후 (API 계약 확정 후 FE 구현)

## Verification Plan

```bash
# 1. Backend issue flow 테스트
cd backend && uv run pytest tests/test_issue_flow_interrupts.py -v

# 2. Resume validation 테스트
cd backend && uv run pytest tests/test_agent_guided_resume_validation.py -v

# 3. 기존 regression 테스트 (interrupt/resume 깨짐 방지)
uv run pytest tests/api/test_agent_response_metadata_contract.py -v
uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v
uv run pytest tests/api/test_agent_autoparse_confirm_interrupt_resume.py -v

# 4. Lint
cd backend && uv run ruff check .

# 5. Frontend
cd frontend && npm run test -- src/features/chat/__tests__/issue-flow-ui.test.tsx
cd frontend && npm run test -- src/features/chat/__tests__/chat-request-payload.test.tsx
cd frontend && npm run build
```

## Verification Results

- command: `uv run pytest backend/tests/test_issue_flow_interrupts.py -v`
  - result: pass
  - note: issue summary/confirm/case-selection/sop-confirm interrupt ordering and empty-result fallback verified
- command: `uv run pytest backend/tests/test_agent_guided_resume_validation.py -v`
  - result: pass
  - note: `_validated_guided_resume_decision` accepts/rejects issue resume payloads as expected
- command: `uv run pytest tests/api/test_agent_response_metadata_contract.py -v`
  - result: pass
  - note: C-API-001 metadata contract remains intact
- command: `uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v`
  - result: pass
  - note: C-API-002 interrupt/resume regression remains intact
- command: `uv run pytest tests/api/test_agent_autoparse_confirm_interrupt_resume.py -v`
  - result: pass
  - note: guided auto-parse resume path still works with new `task_mode=issue` interrupt continuation
- command: `cd frontend && npm run test -- src/features/chat/__tests__/issue-flow-ui.test.tsx`
  - result: pass
  - note: FE issue flow pending state and resume payload wiring verified
- command: `cd frontend && npm run test -- src/features/chat/__tests__/chat-request-payload.test.tsx`
  - result: pass
  - note: request payload contract for guided/issue resume paths verified
- command: `cd frontend && npm run build`
  - result: pass
  - note: production build succeeds (chunk-size warning only)
- command: `cd backend && uv run ruff check .`
  - result: fail
  - note: backend-wide pre-existing Ruff violations in `backend/api/routers/agent.py` (legacy typing/import style) surfaced; not introduced by this task and outside current scoped cleanup
- command: `uv run pytest backend/tests/test_issue_flow_interrupts.py -v`
  - result: pass
  - note: follow-up regression added for `device_selection` path (`selected_doc_types=myservice/gcb/ts`) verifying `task_mode=issue` and `route=general`
- command: `uv run pytest tests/api/test_agent_autoparse_confirm_interrupt_resume.py -v`
  - result: pass
  - note: follow-up regressions added for `filter_doc_types=myservice,gcb,ts` path verifying (1) issue-mode interrupt + metadata task_mode/route and (2) issue interrupt resume 가능(checkpointer 공유)
- command: `uv run pytest backend/tests/test_regeneration_filter_behavior.py -v`
  - result: pass
  - note: 기존 regeneration filter behavior 회귀 없음
- command: `uv run pytest tests/api/test_agent_response_metadata_contract.py -v`
  - result: pass
  - note: C-API-001 metadata contract 유지 확인(추가 재검증)
- command: `uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v`
  - result: pass
  - note: C-API-002 interrupt/resume contract 유지 확인(추가 재검증)

## Handoff

- Current status: completed
- Last passing verification command and result:
  - `uv run pytest tests/api/test_agent_autoparse_confirm_interrupt_resume.py -v` (pass)
- Remaining TODOs (priority order):
  1. (Optional follow-up) decide whether to normalize legacy Ruff violations in `backend/api/routers/agent.py` under a separate style-migration task
- Whether `Allowed Files` changed and why: no additional scope change during implementation
- Whether `Contracts To Update` is expected: no

## Change Log

- 2026-03-13: task created for issue search summary-selection-detail flow
- 2026-03-14: 기존 작업물 현황 추가 (테스트/프롬프트 존재 vs 구현 0% gap 정리), naming convention 확정 (`issue_doc_select` → `issue_case_selection`), implementation plan을 코드 위치 기반 7-step으로 구체화, acceptance criteria 체크리스트화, risk 대응 방안 추가
- 2026-03-14: follow-up bugfix — 사용자가 `myservice/gcb/ts` 범위를 선택했을 때 `task_mode`가 누락되어 setup 라우팅/프롬프트로 흐르던 경로 수정. `device_selection_node`와 API override 경로 모두에서 scope 기반 `task_mode=issue`를 강제하고, issue scope일 때 route를 `general`로 고정하는 회귀 테스트 추가.
- 2026-03-14: follow-up stabilization — `has_overrides` 경로에서 issue interrupt가 발생할 때 resume 불가하던(checkpointer 미사용) 문제 수정. `/api/agent/run` 및 `/api/agent/run/stream` override 경로에 shared checkpointer를 연결하고 resume 회귀 테스트 추가.

## Final Check

- [x] Diff stayed inside allowed files, or this doc was updated first
- [x] Protected contract IDs were re-checked
- [x] Verification commands were run, or blockers were recorded
- [x] Any contract changes were reflected in `product-contract.md`
- [x] Remaining risks and follow-ups were documented
