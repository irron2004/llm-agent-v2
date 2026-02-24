# Agent 아키텍처 분석 및 디버깅 개선 보고서

> 작성일: 2026-02-24
> 브랜치: 86ewk6385-1차-PE-피드백-v2

---

## 1. 요약

현재 Agent 파이프라인은 **3개 파일(4,474줄)** 에 핵심 로직이 집중되어 있으며, `langgraph_agent.py` 단일 파일에 def 57개(노드 함수 19개 + 헬퍼 38개)가 존재한다. 이 구조는 디버깅 시 원인 추적이 어렵고, 개별 노드를 독립적으로 개선하기 힘들어 모델 개선에 과도한 리소스가 소요되는 원인이 된다.

---

## 2. 현재 구조

### 2.1 파일 구성 (3파일, 4,474줄)

| 파일 | 줄 수 | 책임 |
|------|-------|------|
| `llm_infrastructure/llm/langgraph_agent.py` | **2,918줄** | 노드 함수 19개 + 헬퍼 38개 + 타입/프롬프트 스펙 (def 57개) |
| `services/agents/langgraph_rag_agent.py` | **564줄** | 그래프 조립 + 노드 래핑 + 실행 |
| `api/routers/agent.py` | **992줄** | API 엔드포인트 + 요청/응답 모델 + 팩토리 |

### 2.2 그래프 파이프라인

요청 조건에 따라 4가지 에이전트 경로(auto_parse / override / HIL / 일반)로 분기되며, 그래프 내부에서도 조건부 엣지(history_check 분기, should_retry 분기 등)가 존재한다. 가장 일반적인 auto_parse 경로:

```
[요청] → auto_parse → history_check → [query_rewrite]
       → translate → route → mq → st_gate → st_mq
       → retrieve → expand_related → answer → judge
       → [retry_expand / retry_mq / refine_queries]  (verified 모드)
```

### 2.3 노드별 위치 (`langgraph_agent.py` 내)

| 순서 | 노드 | 라인 | 역할 |
|------|------|------|------|
| 1 | `auto_parse_node` | :2496 | 쿼리에서 장비/문서종류/equip_id 파싱 |
| 2 | `history_check_node` | :2393 | 후속 질문 여부 판별 (LLM + 룰 기반) |
| 3 | `query_rewrite_node` | :2454 | 대화 이력 기반 쿼리 재작성 |
| 4 | `translate_node` | :2631 | 쿼리 EN/KO 번역 |
| 5 | `route_node` | :814 | setup / ts / general 3분류 |
| 6 | `mq_node` | :831 | route별 멀티쿼리 생성 |
| 7 | `st_gate_node` | :905 | setup+ts 교차 게이트 판단 |
| 8 | `st_mq_node` | :973 | 최종 검색 쿼리 병합 |
| 9 | `retrieve_node` | :1083 | ES 검색 + rerank (265줄) |
| 10 | `expand_related_docs_node` | :1348 | 인접 페이지/청크 확장 (128줄) |
| 11 | `answer_node` | :1659 | LLM 답변 생성 |
| 12 | `judge_node` | :1740 | 답변 충실성 판정 |
| - | `should_retry` | :1776 | 재시도 전략 결정 (조건부 엣지) |
| - | `retry_expand_node` | :1818 | 문서 확장 범위 증가 |
| - | `retry_mq_node` | :1833 | MQ 재생성 |
| - | `refine_queries_node` | :1856 | 쿼리 정제 후 재검색 |

### 2.4 호출 흐름

```
[FE] POST /api/agent/run/stream
  → agent.py: run_agent_stream()
    → 4가지 분기 (auto_parse / override / HIL / 일반)
      → LangGraphRAGAgent.__init__()   # langgraph_rag_agent.py
        → _build_graph(mode)           # StateGraph 조립 (노드 등록 + 엣지 연결)
    → agent.run(query)
      → graph.invoke(state)            # LangGraph 실행
        → 각 노드 함수 순차 호출       # langgraph_agent.py에서 import된 함수들
  → SSE 스트리밍 응답
```

### 2.5 기존 부분 실행/단위 테스트 인프라

별도의 retrieval pipeline API와 일부 노드 단위 테스트가 이미 존재한다:

| 기능 | 위치 | 설명 |
|------|------|------|
| 단계별 retrieval 실행 | `retrieval_pipeline.py:66` | `steps=["route"]` 등으로 특정 단계까지만 실행 가능 |
| route-only 테스트 | `tests/api/test_retrieval_run_api.py:238` | route 단계만 실행하고 결과 검증 |
| expand 노드 단위 테스트 | `backend/tests/test_expand_related_docs_node.py` | expand_related_docs_node 고립 테스트 |
| st_mq 노드 단위 테스트 | `backend/tests/test_st_mq_bilingual_queries.py` | st_mq_node 고립 테스트 |

**단, 이 인프라는 agent 그래프(`/api/agent/run`)와 별개 경로로 동작하며, agent 그래프 자체의 노드 교체/부분 실행은 지원하지 않는다.**

---

## 3. 디버깅이 어려운 원인 분석

### 3.1 단일 파일 집중 (God Object)

`langgraph_agent.py` **2,918줄**에 def 57개가 모두 존재한다.

**영향:**
- 특정 노드를 수정할 때 관련 없는 코드를 탐색해야 함
- 함수 간 암묵적 의존 관계가 보이지 않음 (같은 파일 내 헬퍼 공유)
- git blame/diff에서 변경 이력이 파일 단위로 섞임

### 3.2 노드 간 상태 전달이 불투명

LangGraph의 `AgentState`(TypedDict)에 **60개 이상의 키**가 flat하게 존재한다. 각 노드가 어떤 키를 읽고 어떤 키를 쓰는지 명시적 계약이 없다.

```python
class AgentState(TypedDict, total=False):
    query: str
    route: str
    docs: list
    search_queries: list
    parsed_query: dict
    # ... 60+ keys
```

**영향:**
- 노드 A가 쓴 키를 노드 C가 읽는데, 중간 노드 B가 덮어쓰는 경우 추적 어려움
- 새로운 키를 추가할 때 기존 노드와 충돌 여부를 2,918줄 전체에서 확인해야 함
- 테스트 시 mock state 구성이 복잡함

### 3.3 노드 함수의 복합 책임

개별 노드가 여러 관심사를 동시에 처리한다.

**예: `retrieve_node` (265줄, :1083~:1347)**
- ES 검색 쿼리 실행
- equip_id 기반 문서 필터링
- doc_type 필터링
- rerank 호출
- top_k 슬라이싱
- all_docs 보존 (재생성용)
- display_docs 병합
- 로깅

하나의 검색 개선(예: equip_id 매칭 로직 변경)을 위해 265줄 전체를 이해해야 한다.

### 3.4 그래프 조립과 노드 구현의 분리

그래프 구조는 `langgraph_rag_agent.py`에, 노드 로직은 `langgraph_agent.py`에 있다.

**영향:**
- 노드 추가/제거 시 두 파일을 동시에 수정해야 함
- `functools.partial`로 의존성을 주입하기 때문에 IDE에서 노드의 실제 시그니처 추적이 어려움
- 그래프의 조건부 엣지(`should_retry`, `history_check` 분기)가 어떤 조건으로 동작하는지 파악하려면 다른 파일로 이동해야 함

### 3.5 API 레이어의 과도한 책임

`agent.py` (992줄)가 담당하는 것:
- 4가지 에이전트 생성 경로 (auto_parse / override / HIL / 일반)
- 동일 로직이 `/run`과 `/run/stream`에 **각각 복사**되어 있음
- 요청/응답 모델 정의 (12개 Pydantic 모델)
- 응답 변환 헬퍼 (7개)
- SSE 스트리밍 구현

`/run`과 `/run/stream`의 에이전트 생성 로직이 **거의 동일하게 2벌** 존재하여 하나를 수정하면 다른 것도 수정해야 한다.

---

## 4. 모델 개선에 미치는 영향

### 4.1 개선 사이클이 느린 이유

| 작업 | 현재 소요 리소스 | 원인 |
|------|-----------------|------|
| MQ 프롬프트 수정 | 높음 | `mq_node` 내부에 route별 분기 + 쿼리 파싱 + 중복 제거가 혼합 |
| 검색 로직 변경 | 높음 | `retrieve_node` 265줄에 필터/검색/rerank/슬라이싱이 일체 |
| 새로운 노드 추가 | 중간 | 2개 파일 동시 수정 + AgentState 키 충돌 확인 |
| 단일 노드 단위 테스트 | 중간 | 일부 노드(expand, st_mq)는 단위 테스트 존재. 그 외 노드는 mock state 구성이 필요 |
| 버그 원인 추적 | 높음 | 12단계 파이프라인에서 어느 노드가 문제인지 로그만으로 판별 어려움 |

### 4.2 실제 사례

**canonical graph 미사용 발견 과정:**
- FE의 `withCanonicalRetrievalDefault()`가 no-op(`{ ...payload }`)임을 확인하기까지 FE/BE 양쪽을 추적해야 했음
- 서버 기본값 `False`와 FE no-op이 결합되어, 코드상 canonical graph가 실행되는 경로가 없었음 (FE가 `use_canonical_retrieval: true`를 보내는 곳 없음, 서버 기본값 `False`)
- 이 사실을 발견하기까지 `agent.py` → `langgraph_rag_agent.py` → `api.ts` → `use-chat-session.ts` → `.env` 총 5개 파일을 추적함
- 참고: 프로덕션 로그/메트릭을 확인하지는 못했으므로, "코드상 실행 경로 없음"으로 판단

---

## 5. Agent 그래프에서의 실험 제약

### 5.1 범위 한정

이 섹션은 **agent 그래프(`/api/agent/run`) 경로**에 한정된 분석이다. 별도의 `retrieval_pipeline.py`(`/api/retrieval/run`)에서는 `steps=["route"]` 등으로 단계별 실행이 가능하고, 일부 노드(expand, st_mq 등)에는 단위 테스트가 존재한다. 그러나 이 인프라는 agent 그래프와 별개 경로이므로, **agent 그래프 자체의 노드 교체/A·B 비교 실험에는 활용할 수 없다.**

### 5.2 실험 시나리오: "MQ 전략을 바꿔서 비교하고 싶다"

**지금 agent 그래프에서 해야 하는 일:**
1. `langgraph_agent.py` 2,918줄에서 `mq_node` 찾기 (:831)
2. 함수 안에서 route별 분기(`setup_mq`, `ts_mq`, `general_mq`) 파악
3. 헬퍼 `_parse_queries`, `_dedupe_queries` 등 같은 파일 내 의존성 추적
4. 수정 후 테스트하려면 agent 파이프라인 전체를 돌려야 함
5. 비교 실험을 하려면 원본 코드를 백업하거나 git branch를 따야 함
6. 같은 쿼리로 "전략 A" vs "전략 B"를 나란히 돌리는 방법이 없음

**실험이 쉬운 구조라면:**
```python
# 노드를 갈아끼우기만 하면 됨
agent_a = build_graph(mq=mq_strategy_v1, retrieve=retrieve_default)
agent_b = build_graph(mq=mq_strategy_v2, retrieve=retrieve_default)

result_a = agent_a.run("PM 점검 주기는?")
result_b = agent_b.run("PM 점검 주기는?")
compare(result_a, result_b)
```

### 5.3 Agent 그래프에서 실험이 어려운 구조적 제약

| 제약 | 현재 상태 | 실험에 미치는 영향 |
|------|----------|-------------------|
| **노드 교체가 어려움** | `_build_graph()`에서 `functools.partial`로 하드코딩 | 노드 하나를 바꾸려면 클래스를 상속하거나 파일을 직접 수정해야 함 |
| **agent 그래프의 부분 실행 미지원** | `graph.invoke()` 한 번에 끝까지 실행 | agent 경로에서 "route까지만 돌려서 결과 보기"가 안 됨 (retrieval API에서는 가능) |
| **노드 단위 테스트 커버리지 부분적** | expand, st_mq 등 일부 노드만 단위 테스트 존재 | mq, route, answer 등 핵심 노드의 고립 테스트가 부족 |

### 5.4 다른 단계에서도 동일한 문제

| 실험 내용 | 현재 필요한 작업 | 이상적 작업 |
|-----------|-----------------|------------|
| retrieve에서 rerank 전략 변경 | `retrieve_node` 265줄 전체 파악 후 내부 수정 | `retrieve(reranker=new_reranker)` 로 교체 |
| route 분류 로직 비교 | `route_node` 수정 → 전체 파이프라인 실행 → 원복 → 다시 실행 | `agent_a = build(route=v1)`, `agent_b = build(route=v2)` 나란히 실행 |
| answer 프롬프트 A/B 테스트 | `answer_node` 내부의 프롬프트 선택 로직 수정 | `agent(answer_prompt=prompt_v2)` 로 주입 |
| judge 기준 완화 후 비교 | `judge_node` + `should_retry` 두 함수 동시 수정 | `agent(judge=lenient_judge)` 로 교체 |
| 특정 노드 건너뛰기 (예: translate 제거) | `_build_graph()` 320줄 내 엣지 수정 | `build_graph(skip=["translate"])` |

### 5.5 핵심 결론

> **agent 그래프(`/api/agent/run`) 경로에서는, 노드가 `_build_graph()`에 하드코딩되어 있어 "노드 하나를 교체해서 나란히 비교"하는 A/B 실험이 어렵다.**
>
> 별도의 `retrieval_pipeline.py`에서 단계별 실행이 가능하지만, 이는 agent 그래프와 별개 경로이므로 agent의 실제 동작을 검증하는 용도로는 한계가 있다. agent 그래프 수준의 실험은 여전히 "파일 수정 → 전체 실행 → 원복"의 수동 사이클을 요구한다.

### 5.6 실제 운영에서 겪은 검증 어려움

#### 문제 1: Agent 그래프에서 중간 결과 확인이 어려움

agent 경로(`/api/agent/run`)에서는 최종 답변만 확인할 수 있다. 답변 품질이 나빠졌을 때 파이프라인 중 **어느 단계에서 문제가 발생했는지** 파악하기 어렵다.

```
auto_parse → history_check → translate → route → mq → st_gate → st_mq
→ retrieve → expand_related → answer → judge

  ↑ 어디서 잘못되었는지 모름. 최종 답변만 보임.
```

예를 들어 답변이 부정확할 때, 원인이 될 수 있는 곳:
- `route`가 잘못 분류됨 (setup인데 general로 감)
- `mq`가 엉뚱한 쿼리를 생성함
- `retrieve`는 좋은 문서를 가져왔는데 `answer`가 잘못 합성함
- `retrieve` 자체가 관련 없는 문서를 가져옴

이 중 어떤 것인지 **최종 답변을 보고 추측**하거나, 서버 로그를 뒤져야 한다.

> 참고: `retrieval_pipeline.py`(`/api/retrieval/run`)에서는 `steps=["route"]`로 중간 단계를 확인할 수 있지만, 이는 agent 그래프와 별개 경로이므로 agent가 실제로 어떤 결과를 내는지와 다를 수 있다.

#### 문제 2: Search 페이지와 Chat 페이지의 검색 로직 불일치

중간 검색 결과를 확인하기 위해 Search 페이지를 만들었으나, **두 페이지의 검색 경로가 다르다.**

| | Chat 페이지 | Search 페이지 |
|---|---|---|
| 엔드포인트 | `/api/agent/run/stream` | `/api/retrieval/run` |
| 실행 코드 | `LangGraphRAGAgent._build_graph()` | `run_retrieval_pipeline()` |
| MQ 생성 | `mq_node` (LLM 호출) | 별도 파이프라인 |
| 문서 확장 | `expand_related_docs_node` | 없음 |
| 상태 관리 | LangGraph `AgentState` | 독립 dict |

Search 페이지에서 검색 결과가 좋아도 Chat 페이지에서는 다른 결과가 나올 수 있다. **Search 페이지가 Chat의 검증 도구로 동작하지 않는다.**

#### 문제 3: 의도한 경로가 아닌 다른 경로로 실행되어도 파악 어려움

그래프에는 조건부 분기가 여러 곳 존재한다:

```python
# history_check 후 분기
"query_rewrite" if s.get("needs_history") else "translate"

# judge 후 분기 (5가지 경로)
should_retry → "done" | "retry_expand" | "retry" | "retry_mq" | "human"
```

요청이 의도한 경로로 갔는지, 예상치 못한 경로로 갔는지를 **실행 후에 확인할 방법이 부족하다.**

#### 현재 로깅의 한계

`_wrap_node`가 각 노드 실행 시 로그를 남기고 있으나:

```
[langgraph] >>> auto_parse INPUT: {query: "PM 점검 주기는?", ...}
[langgraph] auto_parse (1.2s) 장비: Pump-A100 | 문서: SOP
[langgraph] >>> history_check INPUT: ...
[langgraph] history_check (0.8s) 독립 질문
...
```

- 개별 노드 로그는 있으나 **실행 경로 요약이 없다** (어떤 노드를 거쳤는지 한눈에 보이지 않음)
- 노드의 **입출력 state diff가 없다** (어떤 키가 변경되었는지 모름)
- FE SSE 이벤트로 전달되지만 **저장/재현이 안 된다** (한 번 지나가면 사라짐)

#### 필요한 것: 실행 흐름 전체 기록

모든 요청에 대해 다음이 기록되어야 한다:

```
[run:abc123] 실행 경로: auto_parse → history_check → translate → route → mq → st_gate → st_mq → retrieve → expand_related → answer → judge → done
[run:abc123] 총 소요: 8.3s

[run:abc123] auto_parse (1.2s)
  IN:  {query: "PM 점검 주기는?"}
  OUT: {parsed_query: {device: "Pump-A100", doc_type: "SOP"}, detected_language: "ko"}

[run:abc123] route (0.3s)
  IN:  {query_en: "What is PM inspection cycle?"}
  OUT: {route: "setup"}          ← 여기서 "general"로 갔으면 문제 발견

[run:abc123] mq (0.9s)
  IN:  {route: "setup", query_en: "..."}
  OUT: {setup_mq_list: ["PM inspection cycle pump", "preventive maintenance schedule"]}

[run:abc123] retrieve (2.1s)
  IN:  {search_queries: [...]}
  OUT: {docs: [20 items], all_docs: [50 items]}    ← 검색 품질 확인 지점

[run:abc123] answer (2.5s)
  IN:  {docs: [20 items], route: "setup"}
  OUT: {answer: "PM 예방 점검 주기는...(340자)"}

[run:abc123] judge (0.8s)
  OUT: {judge: {faithful: true}} → done
```

이 로그가 있으면:
- **실행 경로**가 의도대로인지 즉시 확인 가능
- **어느 단계에서 품질이 떨어지는지** 중간 결과를 보고 판단 가능
- Search 페이지 없이도 **retrieve 결과를 직접 확인** 가능
- 과거 실행을 **재현/비교** 가능

---

## 6. 개선 제안

### 6.1 단기 (현재 구조 유지하면서 개선)

| # | 제안 | 효과 | 난이도 |
|---|------|------|--------|
| S1 | **실행 경로 요약 로그** — `run()` 완료 시 거쳐간 노드 순서를 한 줄로 출력 | 의도한 경로로 실행되었는지 즉시 확인 | 낮음 |
| S2 | **`_wrap_node`에 state diff 로깅** — 노드 실행 전후 변경/추가된 키만 로깅 | 어떤 노드가 어떤 값을 바꿨는지 확인 | 낮음 |
| S3 | **`/run`과 `/run/stream` 에이전트 생성 로직 통합** — 공통 함수로 추출 | 수정 시 2벌 관리 제거 | 낮음 |
| S4 | **노드별 입출력 키 문서화** — 각 노드가 읽는/쓰는 AgentState 키를 docstring에 명시 | 디버깅 시 상태 추적 용이 | 낮음 |

### 6.2 중기 (파일 분리 + 교체 가능 구조)

| # | 제안 | 효과 | 난이도 |
|---|------|------|--------|
| M1 | **노드 함수를 개별 모듈로 분리** — `nodes/route.py`, `nodes/mq.py`, `nodes/retrieve.py` 등 | 노드별 독립 테스트, git 이력 분리, 코드 탐색 용이 | 중간 |
| M2 | **`retrieve_node` 265줄을 책임 분리** — 검색/필터/rerank/슬라이싱을 별도 함수로 | 검색 로직 단위 개선 가능 | 중간 |
| M3 | **AgentState 키를 노드 그룹별 Namespace로 정리** — `state["retrieve"]["docs"]` 형태 | 키 충돌 방지, 노드 간 계약 명확화 | 중간 |
| M4 | **노드 교체 가능한 `_build_graph()` 인터페이스** — 노드 함수를 dict로 받아 조립 | A/B 실험 시 코드 수정 없이 노드 교체 가능 | 중간 |
| M5 | **핵심 노드(mq, route, answer) 단위 테스트 추가** — 기존 expand/st_mq 테스트 패턴 활용 | 노드 수정 시 빠른 회귀 확인 | 중간 |

### 6.3 장기 (실험 인프라)

| # | 제안 | 효과 | 난이도 |
|---|------|------|--------|
| L1 | **Step-by-step 실행 모드** — agent 그래프에서 특정 노드까지만 실행하고 중간 state를 반환 | 디버깅 시 파이프라인 중간 지점 확인 | 높음 |
| L2 | **A/B 비교 실행 CLI** — 동일 쿼리에 대해 2개 그래프 설정을 나란히 실행하고 결과 비교 | 실험 사이클 "파일 수정→전체 실행→원복" 제거 | 높음 |
| L3 | **Search/Chat 검색 로직 통합** — 두 페이지가 동일한 retrieval 경로를 사용하도록 통합 | Search 페이지가 Chat의 검증 도구로 동작 가능 | 높음 |

---

## 7. 현재 세션에서 수행한 정리 작업

| 작업 | 상태 |
|------|------|
| `main.py` local backend 분기 제거 | 커밋 완료 |
| FE 미사용 라우터 7개 주석 처리 (chat, preprocessing, rerank, query_expansion, summarization, devices, retrieval_evaluation) | 커밋 완료 |
| canonical graph 전체 제거 (`_build_canonical_graph`, `_canonical_retrieve_node`, `use_canonical_retrieval` 파라미터) | 커밋 완료 |
| FE `withCanonicalRetrievalDefault()` no-op 함수 제거 | 커밋 완료 |
| canonical 관련 테스트 4개 삭제 + 6개 테스트 파일에서 참조 제거 | 커밋 완료 |
| 주석 처리된 라우터의 테스트 2개 삭제 (chat, preprocessing) | 커밋 완료 |

**정리 후 현재 그래프:** canonical graph가 제거되어 단일 그래프만 존재. 요청 조건에 따라 4가지 에이전트 경로(auto_parse / override / HIL / 일반)로 분기되며, 그래프 내부 조건부 엣지도 동작한다.

---

## 부록: 파일 간 의존 관계

```
frontend/.env (VITE_CHAT_PATH)
  → frontend/src/features/chat/api.ts (resolveChatPaths)
    → frontend/src/features/chat/hooks/use-chat-session.ts (send)

  → backend/api/routers/agent.py (run_agent / run_agent_stream)
    → _new_auto_parse_agent / _new_hil_agent (팩토리)
      → backend/services/agents/langgraph_rag_agent.py (LangGraphRAGAgent)
        → _build_graph() (그래프 조립)
          → backend/llm_infrastructure/llm/langgraph_agent.py (노드 함수 19개)
            → backend/services/search_service.py (ES 검색)
            → backend/llm_infrastructure/llm/base.py (LLM 호출)
            → backend/llm_infrastructure/llm/prompt_loader.py (프롬프트 로드)

  → backend/api/routers/retrieval.py (별도 retrieval API)
    → backend/services/retrieval_pipeline.py (단계별 실행 가능)
```
