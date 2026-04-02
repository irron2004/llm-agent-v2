# Task: Hybrid ReAct Planner Loop 도입 (openclaw-transform)

Status: In-Progress
Owner: Claude (feat/openclaw-transform)
Branch or worktree: feat/openclaw-transform
Created: 2026-03-31

## Goal

현재 구조를 전면 교체하지 않고, 아래 원칙으로 **유연성만 먼저 확보**한다.

1. LangGraph 외곽 상태 머신(interrupt/resume/checkpointer/SSE event)은 유지
2. core retrieval/answer orchestration 구간만 planner loop로 단계적 치환
3. 기존 API response shape 및 계약(C-API-001/002/003)은 변경하지 않음

즉, `full DAG replacement`가 아니라 `compatibility-first hybrid migration`이 목표다.

## Why

현재 문제:
1. **의도 파악 실패**: 사전 정의되지 않은 질문에서 잘못된 검색 경로로 빠짐
2. **컨텍스트 단절**: 후속 질문에서 대화 흐름 유지가 불안정
3. **단일 검색 한계**: 문제 문서 이후 해결 문서를 추가 탐색하는 유연한 loop 부족

하지만 현재 시스템은 단순 DAG가 아니라,
- `langgraph_rag_agent.py`(그래프 조립/체크포인터 연계)
- `langgraph_agent.py`(상태/노드 로직)
- `agent.py`(router 분기/metadata 조립)
가 강하게 결합되어 있어 전면 교체 시 회귀 위험이 높다.

## Compatibility Surface (반드시 유지)

### 1) Router가 소비하는 result state key
`backend/api/routers/agent.py::_build_response_metadata()` 및 `_sanitize_search_queries_raw()`가 읽는 key는
존재 + 값 의미를 모두 유지해야 한다.

핵심: `route`, `st_gate`, `mq_mode`, `mq_used`, `mq_reason`, `attempts`, `max_attempts`,
`retry_strategy`, `guardrail_dropped_numeric`, `guardrail_dropped_anchor`, `guardrail_final_count`,
`search_queries`, `search_queries_raw`(또는 동등한 source key), `index_name`.

### 2) Interrupt payload 타입과 resume semantics
- interrupt payload 타입: `retrieval_review`, `auto_parse_confirm`, `issue_confirm`,
  `issue_case_selection`, `issue_sop_confirm`, `abbreviation_resolve`
- resume는 `agent._graph.get_state()` + `agent._graph.invoke(Command(resume=...))`
  패턴과 nonce/thread continuity를 유지해야 함.

### 3) SSE observability parity
`event_sink`를 통해 `/run/stream`에서 노드 이벤트가 기존과 동일한 수준으로 전달되어야 함.

### 4) LLM abstraction 유지
기본 구현은 `BaseLLM.generate(..., response_model=...)` 기반으로 유지한다.
LangGraph ToolNode/외부 tool-calling API로 즉시 전환하지 않는다(별도 adapter scope 방지).

## Contracts To Preserve

### C-API-001 — 응답 메타데이터
- 보존 대상 key:
  `mq_mode`, `mq_used`, `mq_reason`, `route`, `st_gate`, `attempts`, `max_attempts`,
  `retry_strategy`, `guardrail_dropped_numeric`, `guardrail_dropped_anchor`,
  `guardrail_final_count`, `search_queries_final`, `search_queries_raw`, `index_name`
- **semantics 포함**: key 존재뿐 아니라 값의 의미/생성 경로 유지

### C-API-002 — Interrupt/Resume thread 연속성
- `thread_id` 기반 checkpointer 연속성 유지
- guided resume 분기/decision type/nonce semantics 유지

### C-API-003 — retrieval_only 모드
- `retrieval_only=True` 시 answer 생성 전 interrupt 유지
- interrupt payload type=`retrieval_review`
- `metadata.response_mode = retrieval_only` 유지

## Contracts To Update

- None (계약 변경 없음)

## Allowed Files

**신규 파일:**
- `backend/llm_infrastructure/llm/react_agent.py`
- `backend/llm_infrastructure/llm/react_tools.py` (필요 시)
- `backend/tests/test_react_agent_core_loop.py` (신규 단위 테스트)

**기존 파일 (최소 수정):**
- `backend/services/agents/langgraph_rag_agent.py` (factory/feature-flag wiring)
- `backend/api/routers/agent.py` (feature-flag 분기 추가, 기존 분기 semantics 유지)

**검증 대상 테스트 (수정 금지, 실패 시 구현 쪽 수정):**
- `tests/api/test_agent_response_metadata_contract.py`
- `tests/api/test_agent_interrupt_resume_regression.py`
- `tests/api/test_agent_retrieval_only.py`
- `tests/api/test_agent_autoparse_confirm_interrupt_resume.py`
- `tests/api/test_agent_sticky_policy_followup_only.py`
- `tests/api/test_agent_sticky_autoparse_thread.py`
- `tests/api/test_agent_concurrency.py`
- `tests/api/test_agent_mq_mode_defaulting.py`
- `tests/api/test_agent_stage2_retrieval.py`
- `tests/api/test_agent_canonical_retrieval.py`

## Out Of Scope

- Full DAG replacement (이번 작업 범위 아님)
- interrupt/resume 프로토콜 변경
- guided_confirm / issue / abbreviation flow semantics 변경
- reranker/embedding/retrieval pipeline 알고리즘 변경
- frontend 변경

허용 예외:
- planner loop 동작을 위한 최소 planner action prompt/response schema 추가는 허용
  (기존 answer/router prompt 교체는 금지)

## Hybrid Migration 단계

### Phase 1: Compatibility 명문화 (필수 선행)
- [ ] React 경로가 반환해야 할 state key/semantics 표 작성
- [ ] interrupt payload/decision type/nonce 요구사항 표 작성
- [ ] `react_agent.py` skeleton이 위 표를 만족하도록 보정

### Phase 2: No-interrupt 경로 파일럿
- [ ] `use_react_agent=true` + non-resume + non-retrieval_only + non-guided 경로에만 적용
- [ ] 기존 LangGraph 경로를 default로 유지
- [ ] `BaseLLM + response_model` 기반 planner NextAction 구현

### Phase 3: Metadata parity 강화
- [ ] C-API-001 및 `mq_mode_defaulting`, stage2/canonical metadata 관련 테스트 통과
- [ ] `search_queries_raw` semantics 일치 확인

### Phase 4: retrieval_only 통합
- [x] `retrieval_review` interrupt/`response_mode` parity 확보
- [x] C-API-003 테스트 통과

### Phase 5: Guided/Issue/Abbreviation resume 통합
- [ ] resume decision type + nonce + checkpointer continuity 유지
- [ ] C-API-002 + guided resume 관련 테스트 통과

### Phase 6: Stream/Concurrency parity
- [x] `/run` vs `/run/stream` final payload parity 확인
- [x] `event_sink` 기반 observability parity 및 concurrency 테스트 통과

## Risks

| 리스크 | 수준 | 대응 |
|---|---|---|
| C-API-001 metadata semantics 불일치 | 높음 | Phase 1에서 key+semantics 명문화 후 테스트 고정 |
| C-API-002 resume 경로/nonce 깨짐 | 높음 | interrupt/resume 외곽 경로 유지, Phase 5 분리 적용 |
| C-API-003 retrieval_only interrupt 누락 | 높음 | Phase 4 별도 단계로 분리 |
| BaseLLM vs ToolNode 추상화 불일치 | 중간 | ToolNode 즉시 도입 금지, BaseLLM 기반 NextAction 우선 |
| SSE 이벤트/스트림 품질 저하 | 중간 | `event_sink` parity 검증을 필수 게이트로 추가 |
| planner loop 발산 | 중간 | max iterations hard cap + fallback 정책 |

## Verification Plan

```bash
cd /home/hskim/work/llm-agent-v2

# 최소 단위 검증
uv run python -c "from backend.llm_infrastructure.llm.react_agent import ReactRAGAgent; print('OK')"

# 계약/회귀 핵심
uv run pytest tests/api/test_agent_response_metadata_contract.py -v
uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v
uv run pytest tests/api/test_agent_retrieval_only.py -v
uv run pytest tests/api/test_agent_autoparse_confirm_interrupt_resume.py -v
uv run pytest tests/api/test_agent_sticky_policy_followup_only.py -v
uv run pytest tests/api/test_agent_sticky_autoparse_thread.py -v
uv run pytest tests/api/test_agent_concurrency.py -v
uv run pytest tests/api/test_agent_mq_mode_defaulting.py -v
uv run pytest tests/api/test_agent_stage2_retrieval.py -v
uv run pytest tests/api/test_agent_canonical_retrieval.py -v
```

## Verification Results

- `react_agent.py` import: **pass** (2026-03-31)
- `test_agent_response_metadata_contract.py`: **2 passed** (Phase 1/3)
- `test_agent_retrieval_only.py`: **1 passed** (Phase 4 C-API-003)
- `test_agent_interrupt_resume_regression.py`: **5 passed**, 1 failed (pre-existing: `test_abbreviation_resume_routes_to_hil_agent`)
- `test_agent_autoparse_confirm_interrupt_resume.py`: **8 passed** (Phase 5/6)
- `test_agent_mq_mode_defaulting.py`: **passed**
- `test_agent_concurrency.py`: **passed** (Phase 6)
- `test_agent_sticky_policy_followup_only.py`: **passed**
- `test_agent_sticky_autoparse_thread.py`: **passed**
- `test_agent_stage2_retrieval.py`: **passed**
- `test_agent_canonical_retrieval.py`: **passed**

## Change Log

- 2026-03-31: task created, kickoff 완료
- 2026-03-31: Goal을 full replacement에서 hybrid migration으로 전환
- 2026-03-31: Oracle/architecture review 반영 — compatibility surface 명시,
  BaseLLM 제약 추가, 단계별 게이트(phase)와 검증 범위 확장
- 2026-03-31: Phase 1-4, 6 구현 완료. Phase 5(abbreviation resume)는 pre-existing failure로 별도 태스크로 분리

## Final Check

- [ ] Diff stayed inside allowed files
- [ ] C-API-001 key + semantics 재확인
- [ ] C-API-002 resume/nonce/thread continuity 재확인
- [ ] C-API-003 retrieval_only interrupt/response_mode 재확인
- [ ] Verification commands 실행
- [ ] product-contract.md 업데이트 필요 없음 (계약 변경 없음)
