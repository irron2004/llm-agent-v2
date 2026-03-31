# Hybrid Migration Plan: ReAct Planner Loop 도입 로드맵

작성일: 2026-03-31
브랜치: feat/openclaw-transform
태스크: docs/tasks/TASK-20260331-openclaw-transform.md

---

## 배경

기존 LangGraph DAG(20+ 노드, 5800줄)는 다음 세 문제를 갖고 있다:

1. **의도 파악 실패** — 사전 정의되지 않은 질문 → 잘못된 검색 경로
2. **컨텍스트 단절** — 후속 질문에서 대화 흐름 유지 불안정
3. **단일 검색 한계** — 문제 문서 발견 후 해결 문서 추가 탐색 불가

하지만 현재 시스템은 단순 DAG가 아니라 세 레이어가 강하게 결합되어 있다:

```
agent.py (router 분기 / metadata 조립 / SSE streaming)
    └── langgraph_rag_agent.py (그래프 조립 / checkpointer 연계)
          └── langgraph_agent.py (상태 / 노드 / interrupt / retry)
```

→ **전면 교체 금지. Compatibility-first hybrid migration.**

---

## 핵심 원칙

```
1. LangGraph 외곽(interrupt/resume/checkpointer/SSE)은 건드리지 않는다
2. core retrieval/answer cluster만 planner loop로 단계적으로 교체한다
3. API response shape와 계약(C-API-001/002/003)은 변경하지 않는다
4. BaseLLM 기반 planner 유지 — ToolNode/외부 tool-calling API 즉시 전환 금지
5. 기존 LangGraphRAGAgent는 default. ReactRAGAgent는 feature-flag로 선택 활성화
```

---

## 단계별 계획

### Phase 1 — Compatibility 명문화 ✅ 진행 중
**목표**: ReactRAGAgent가 반환해야 할 state key/semantics를 코드보다 먼저 명문화

작업:
- [x] `_build_response_metadata()` / `_sanitize_search_queries_raw()` 분석
- [x] `docs/contracts/react-agent-state-contract.md` 작성 (state key semantics 표)
- [x] `react_agent.py` skeleton의 semantics 보정
  - `general_mq_list` 키 추가 → `_sanitize_search_queries_raw` fallback 활용
  - `guardrail_*` 누적 방식 보정

완료 기준: `react_agent.py` import OK + state contract 문서 존재

---

### Phase 2 — No-interrupt 경로 파일럿
**목표**: 가장 단순한 경로에서만 ReactRAGAgent를 활성화

적용 조건 (모두 만족 시만):
- `use_react_agent=true` (feature flag)
- `is_resume=False`
- `retrieval_only=False`
- `guided_confirm=False`
- `auto_parse=True` (auto_parse 경로만 먼저)

작업:
- [ ] `BaseLLM.generate(response_model=NextAction)` 기반 planner 구현
- [ ] `backend/api/routers/agent.py` feature-flag 분기 추가 (기존 분기 보존)
- [ ] `backend/services/agents/langgraph_rag_agent.py` ReactRAGAgent factory 추가

완료 기준: `test_agent_response_metadata_contract.py` 통과

---

### Phase 3 — Metadata parity 강화
**목표**: C-API-001 전체 key + semantics 일치

작업:
- [ ] `search_queries_raw` semantics 검증 (`general_mq_list` fallback 활용)
- [ ] `mq_mode_defaulting`, `stage2`, `canonical_retrieval` 관련 테스트 통과
- [ ] `selected_doc_types`, `task_mode`, `retrieval_stage2` 등 보조 키 채우기

완료 기준: 메타데이터 관련 테스트 10개 모두 통과

---

### Phase 4 — retrieval_only 통합
**목표**: C-API-003 interrupt 경로를 ReactRAGAgent에 추가

작업:
- [ ] search 완료 후 `retrieval_only=True` 시 `interrupt(payload)` 삽입
- [ ] interrupt payload: `{"type": "retrieval_review", "docs": ..., "response_mode": "retrieval_only"}`
- [ ] `test_agent_retrieval_only.py` 통과

완료 기준: C-API-003 테스트 통과

---

### Phase 5 — Guided/Issue/Abbreviation resume 통합
**목표**: C-API-002 전체 — resume decision type / nonce / checkpointer continuity

이 단계는 복잡도가 높아 별도 태스크로 분리 권장.

작업:
- [ ] guided_confirm resume 경로 분석
- [ ] nonce/decision type semantics 유지 방법 설계
- [ ] `test_agent_autoparse_confirm_interrupt_resume.py` 통과

완료 기준: C-API-002 + guided resume 테스트 통과

---

### Phase 6 — Stream/Concurrency parity
**목표**: `/run` vs `/run/stream` final payload 동일, 동시 요청 안정성

작업:
- [ ] `event_sink` 기반 SSE 이벤트 순서/내용 parity 검증
- [ ] `test_agent_concurrency.py` 통과
- [ ] A/B 답변 품질 비교 (실제 PE 질문 세트 기준)

완료 기준: 모든 계약 테스트 통과 + 품질 회귀 없음

---

## 현재 파일 상태

```
backend/llm_infrastructure/llm/
  react_agent.py          ← Phase 1 skeleton (import OK)
  langgraph_agent.py      ← 기존 유지 (5807줄, 변경 금지)

backend/services/agents/
  langgraph_rag_agent.py  ← 기존 유지 (Phase 2에서 factory 추가 예정)

backend/api/routers/
  agent.py                ← 기존 유지 (Phase 2에서 flag 분기 추가 예정)

docs/contracts/
  product-contract.md     ← 기존 유지
  react-agent-state-contract.md  ← Phase 1 신규 (아래)
```

---

## 참고: `_sanitize_search_queries_raw` fallback 우선순위

`agent.py`의 `_sanitize_search_queries_raw()`는 다음 키를 순서대로 병합한다:

```
search_queries_raw → raw_search_queries →
general_mq_list → setup_mq_list → ts_mq_list →
general_mq_ko_list → setup_mq_ko_list → ts_mq_ko_list
```

ReactRAGAgent는 `general_mq_list = search_queries_used`로 설정하면
기존 sanitize 로직을 그대로 재사용할 수 있다. (별도 key 신규 도입 불필요)

---

## Skill 및 Task 정의 시점

- **지금**: task doc + migration plan doc으로 충분. skill 정의 불필요.
- **Phase 2 파일럿 완료 후**: hybrid migration 패턴이 안정되면 skill 추출
- **전체 migration 완료 후**: task template 업데이트 (이번 작업 교훈 반영)
