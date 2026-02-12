# PE Agent 구조적 문제 진단 보고서

**작성일**: 2026-02-12
**대상 브랜치**: `86ew947y6-사내테스트-점검-리스트`
**목적**: 사내 테스트에서 반복 발생한 문제의 근본 원인을 코드 수준에서 추적·진단

---

## 1. 개요

### 1.1 배경

사내 테스트 과정에서 다음 세 가지 유형의 문제가 반복적으로 보고되었다.

1. **화면에서 설정한 검색 파라미터가 실제 쿼리에 반영되지 않음**
2. **동일 질문에 대해 답변 품질이 일관되지 않음**
3. **디버깅을 위해 수정한 코드가 관련 모듈에 일괄 적용되지 않거나, 모듈 간 충돌 발생**

이 문제들은 독립적으로 보이지만, 코드를 추적한 결과 공통된 구조적 원인에서 파생되는 것으로 진단되었다.

### 1.2 진단 요약

| 유형 | 증상 | 근본 원인 | 심각도 |
|------|------|----------|:------:|
| A. 설정값 미반영 | UI 파라미터가 검색에 적용되지 않음 | 전파 파이프라인 누락, 싱글톤 하드코딩 | 높음 |
| B. 답변 품질 편차 | 같은 질문에 다른 품질의 답변 | 컨텍스트 truncation 불일치, 상태 필드 혼재 | 높음 |
| C. 수정 미전파 | 코드 변경이 다른 모듈에 반영 안 됨 | 단일 거대 파일, 하드코딩 중복, 이중 필드 관리 | 중간 |

### 1.3 대상 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────────┐
│  Frontend (React)                                                │
│  search-config-panel → retrieval-test-context → use-retrieval-test│
└───────────────────────────┬──────────────────────────────────────┘
                            │ HTTP (search_override)
┌───────────────────────────▼──────────────────────────────────────┐
│  API Layer (FastAPI Routers)                                      │
│  agent.py (싱글톤 에이전트) │ search.py (ChatPipelineSearchRequest) │
└───────────────────────────┬──────────────────────────────────────┘
                            │ Python call
┌───────────────────────────▼──────────────────────────────────────┐
│  Service Layer                                                    │
│  langgraph_rag_agent.py (functools.partial로 노드 바인딩)          │
└───────────────────────────┬──────────────────────────────────────┘
                            │ State dict
┌───────────────────────────▼──────────────────────────────────────┐
│  Agent Nodes (langgraph_agent.py, 125KB)                          │
│  mq_node → retrieve_node → expand_node → answer_node → judge_node│
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. 유형 A: 설정값 미반영

### 2.1 전파 경로 추적 결과

UI에서 설정 가능한 10개 파라미터의 전파 상태를 프론트엔드 → API → 서비스 → 노드 → 검색 엔진 순서로 추적했다.

| # | 설정 | UI | Context | API 전송 | 백엔드 수신 | 검색 적용 | 판정 |
|:-:|------|:--:|:-------:|:--------:|:----------:|:--------:|:----:|
| 1 | denseWeight | O | O | O | O | O | 정상 |
| 2 | sparseWeight | O | O | O | O | O | 정상 |
| 3 | useRrf | O | O | O | O | O | 정상 |
| 4 | rrfK | O | O | O | O | O | 정상 |
| 5 | rerank | O | O | O | O | O | 정상 |
| 6 | rerankTopK | O | O | O | O | O | 정상 |
| 7 | size (top_k) | O | O | O | O | **X** | **미적용** |
| 8 | multiQuery | O | O | **X** | X | X | **미적용** |
| 9 | multiQueryN | O | O | **X** | X | X | **미적용** |
| 10 | fieldWeights | O | O | **X** | X | X | **미적용** |

**10개 파라미터 중 4개(40%)가 UI에만 존재하고, 실제 검색 동작에는 전혀 반영되지 않는다.**

### 2.2 원인 분석

#### 2.2.1 프론트엔드 Request Body 구성 누락

**파일**: `frontend/src/features/retrieval-test/hooks/use-retrieval-test.ts` (line 118-124)

프론트엔드가 API로 보내는 `search_override` 객체에 `multiQuery`, `multiQueryN`, `fieldWeights`가 포함되지 않는다.

```typescript
// use-retrieval-test.ts:118-124 — 실제 전송 객체
const requestBody: ChatPipelineSearchRequest = {
  query: question.question,
  search_override: {
    top_k: config.size,            // ✓ 전송
    use_rrf: config.useRrf,        // ✓ 전송
    rrf_k: config.rrfK,            // ✓ 전송
    rerank: config.rerank,         // ✓ 전송
    rerank_top_k: config.rerankTopK, // ✓ 전송
    // ✗ config.multiQuery    — 누락
    // ✗ config.multiQueryN   — 누락
    // ✗ config.fieldWeights  — 누락
  },
};
```

백엔드의 `ChatPipelineSearchRequest` 모델(`backend/api/routers/search.py` line 343-351)도 `search_override`를 `Optional[dict]`로 정의하고 있어, 해당 필드에 대한 검증이 없다.

```python
# search.py:343-351
class ChatPipelineSearchRequest(BaseModel):
    query: str = Field(..., description="검색 쿼리")
    search_override: Optional[dict] = Field(
        default=None,
        description="검색 파라미터 override (dense_weight, sparse_weight, rerank 등)"
    )
    selected_devices: Optional[List[str]] = None
    selected_doc_types: Optional[List[str]] = None
    # ✗ multi_query, multi_query_n, field_weights 필드 없음
```

**전파 단절 지점**: 프론트엔드 → API 경계. 타입 정의 수준에서 차단.

#### 2.2.2 싱글톤 에이전트의 하드코딩된 파라미터

**파일**: `backend/api/routers/agent.py` (line 359-396)

채팅 경로의 주 진입점인 HIL/auto-parse 에이전트가 전역 싱글톤으로 생성되며, `top_k`와 `retrieval_top_k`가 고정값이다.

```python
# agent.py:363-376 — HIL 에이전트 싱글톤
def _get_hil_agent(llm, search_service, prompt_spec) -> LangGraphRAGAgent:
    global _hil_agent
    if _hil_agent is None:
        _hil_agent = LangGraphRAGAgent(
            llm=llm,
            search_service=search_service,
            prompt_spec=prompt_spec,
            top_k=20,              # ← 고정값
            retrieval_top_k=100,   # ← 고정값
            mode="verified",
            ask_user_after_retrieve=True,
            ask_device_selection=True,
            ...
        )
    return _hil_agent
```

```python
# agent.py:383-396 — Auto-parse 에이전트 싱글톤 (동일 구조)
def _get_auto_parse_agent(llm, search_service, prompt_spec) -> LangGraphRAGAgent:
    global _auto_parse_agent
    if _auto_parse_agent is None:
        _auto_parse_agent = LangGraphRAGAgent(
            ...
            top_k=20,              # ← 동일한 고정값
            retrieval_top_k=100,   # ← 동일한 고정값
            ...
        )
    return _auto_parse_agent
```

사용자가 `AgentRequest.top_k`(`agent.py` line 436: `top_k: int = Field(20, ge=1, le=50)`)에 다른 값을 보내도, 이 싱글톤들은 한번 생성되면 재사용되므로 요청별 `top_k`가 무시된다.

`top_k`가 실제로 반영되는 경로는 다음 분기뿐이다:
- `has_overrides and not is_resume` 분기 (`agent.py` line 721)
- 일반 모드 분기 (`agent.py` line 768)

즉, **대부분의 채팅 요청은 HIL/auto-parse 싱글톤을 사용하므로, `top_k=20` 고정.**

#### 2.2.3 mq_node에 search_override 미전달

**파일**: `backend/services/agents/langgraph_rag_agent.py` (line 773)

`retrieve_only()` 메서드에서 Multi-Query 생성 노드를 호출할 때 `search_override`를 전달하지 않는다.

```python
# langgraph_rag_agent.py:773 — retrieve_only()
result = mq_node(state, llm=self.llm, spec=self.spec)
# ✗ search_override 미전달
```

`retrieve_node`에는 state를 통해 `search_override`가 전달되지만(`langgraph_rag_agent.py` line 785-791), MQ 생성 단계에서는 사용자가 설정한 `multiQuery`/`multiQueryN`이 무시된다.

그래프 빌더를 통한 호출에서도 동일하다:
```python
# langgraph_rag_agent.py:378 — functools.partial 바인딩
builder.add_node("mq", self._wrap_node("mq", functools.partial(
    mq_node, llm=self.llm, spec=self.spec
)))
# ✗ search_override 바인딩 없음
```

#### 2.2.4 fieldWeights — 백엔드에 개념 자체가 부재

프론트엔드에는 `fieldWeights` 관련 UI 컨트롤이 완성되어 있다:
- `search-config-panel.tsx` (line 231-294): 필드별 가중치 슬라이더, 활성/비활성 토글
- `retrieval-test-context.tsx`: `SearchConfig.fieldWeights: FieldConfig[]` 타입 정의

그러나 백엔드 코드 전체에서 `field_weights` 또는 `fieldWeights`를 참조하는 곳이 **0건**이다. Elasticsearch 쿼리 빌더, Retriever, SearchService 어디에도 필드별 가중치를 수신하거나 적용하는 로직이 없다.

**결론**: fieldWeights UI는 완전한 dead control이다.

---

## 3. 유형 B: 답변 품질 비일관성

### 3.1 컨텍스트 truncation 정책 불일치

#### 3.1.1 현상

동일 문서가 어떤 코드 경로를 거쳤느냐에 따라 **200자 또는 무제한**으로 잘려서 LLM에 전달된다. 이로 인해 같은 질문에 대해 답변에 포함되는 정보량이 비결정적이다.

#### 3.1.2 상수 정의

```python
# langgraph_agent.py:349-353
MAX_TOKENS_CLASSIFICATION = 256
MAX_TOKENS_JUDGE = 1024
MAX_TOKENS_ANSWER = 4096
MAX_REF_CHARS_REVIEW = 200       # 검색 결과 리뷰용 (실제 사용됨)
MAX_REF_CHARS_ANSWER = 1200      # 답변 생성용 (정의만 있고 미사용)
```

`MAX_REF_CHARS_ANSWER = 1200`은 답변 생성 시 문서를 적절한 길이로 제공하려는 의도로 정의되었으나, 어디서도 참조하지 않는다.

#### 3.1.3 경로별 truncation 동작

`results_to_ref_json()` 함수(`langgraph_agent.py` line 646-681)의 기본 `max_chars` 값이 200이다:

```python
# langgraph_agent.py:646-649
def results_to_ref_json(
    docs: List[RetrievalResult],
    *,
    max_chars: int | None = MAX_REF_CHARS_REVIEW,  # 기본값: 200자
    prefer_raw_text: bool = False,
) -> List[Dict[str, Any]]:
```

이 함수를 호출하는 세 가지 경로가 각각 다른 truncation을 적용한다:

| 경로 | 호출 위치 | max_chars 값 | 결과 |
|------|----------|:-----------:|------|
| `retrieve_node` → `ref_json` 생성 | line ~1716 | 200 (기본값) | 문서당 최대 200자 |
| `expand_related_docs_node` → `answer_ref_json` 생성 | line ~1840 | `max_ref_chars` 파라미터 | **호출부에서 미전달 → None → 무제한** |
| `answer_node` fallback → `ref_items` 재생성 | line 2085 | `None` (명시적) | 무제한 |

`expand_related_docs_node`는 `max_ref_chars: int | None = None` 시그니처(`langgraph_agent.py` line 1728)를 갖지만, 호출하는 곳(`langgraph_rag_agent.py` line 403-411, 795-799)에서 이 파라미터를 전달하지 않는다:

```python
# langgraph_rag_agent.py:403-411 — functools.partial 바인딩
builder.add_node(
    "expand_related",
    self._wrap_node("expand_related", functools.partial(
        expand_related_docs_node,
        page_fetcher=self.page_fetcher,
        doc_fetcher=self.doc_fetcher,
        # ✗ max_ref_chars 미전달
    )),
)
```

**결과**: `retrieve_node`가 만든 `ref_json`은 200자 제한이 적용되지만, `expand_related_docs_node`가 만든 `answer_ref_json`은 무제한이다. `answer_node`가 어느 것을 사용하느냐에 따라 LLM에 전달되는 정보량이 극적으로 달라진다.

### 3.2 문서 참조 상태 필드 5종 혼재

#### 3.2.1 현상

LLM에 전달할 참조 문서를 결정하는 데 5개의 AgentState 필드가 존재한다.

```python
# langgraph_agent.py:141-145 (AgentState TypedDict)
docs: List[RetrievalResult]           # retrieve_node가 설정
all_docs: List[RetrievalResult]       # rerank 전 전체 문서 (max 20)
display_docs: List[RetrievalResult]   # 화면 표시용 문서
ref_json: List[Dict[str, Any]]       # 200자 truncation 적용된 참조 JSON
answer_ref_json: List[Dict[str, Any]] # expand 후 생성된 참조 JSON
```

#### 3.2.2 노드별 접근 패턴

각 노드가 서로 다른 fallback 순서로 이 필드들에 접근한다:

```python
# answer_node (langgraph_agent.py:2081-2087)
ref_items = state.get("answer_ref_json")              # 1순위
if not ref_items:
    docs = state.get("display_docs") or state.get("docs") or []  # 2순위
    if docs and hasattr(docs[0], "doc_id"):
        ref_items = results_to_ref_json(docs, max_chars=None, prefer_raw_text=True)
    else:
        ref_items = state.get("ref_json", [])          # 3순위

# judge_node (langgraph_agent.py:2368)
ref_items = state.get("answer_ref_json") or state.get("ref_json", [])

# expand_related_docs_node (langgraph_agent.py:1735)
docs = state.get("docs", [])  # docs만 사용
```

**문제**: 같은 질문이라도 `expand_related_docs_node`를 거쳤는지 여부에 따라 `answer_node`가 받는 참조 문서의 내용과 길이가 달라진다.

- `expand`를 거친 경우: `answer_ref_json` 사용 → 무제한 길이
- `expand`를 거치지 않은 경우: `ref_json` fallback → 200자 제한

**추가 위험**: `answer_node`의 fallback에서 `display_docs`나 `docs`를 찾으면 `max_chars=None`으로 재변환하므로, 이전에 200자로 만들어둔 `ref_json`이 무시된다. 어떤 필드가 먼저 설정되었느냐에 따라 다른 결과가 나오는 비결정적 동작이다.

### 3.3 LLM 파라미터 미제어

#### 3.3.1 temperature 미지정

`_invoke_llm` 함수 시그니처(`langgraph_agent.py` line 359-360):

```python
def _invoke_llm(llm: BaseLLM, system: str, user: str, **kwargs: Any) -> str:
```

`**kwargs`를 통해 temperature를 전달할 수 있는 구조이지만, **모든 호출부에서 temperature를 명시하지 않는다.** 분류(결정적 판단이 필요), 쿼리 생성(다양성이 필요), 답변 생성(일관성이 필요), 판정(이진 판단이 필요) 등 성격이 다른 노드들이 모두 모델 기본 temperature로 동작한다.

#### 3.3.2 max_tokens 노드별 현황

| 노드 | max_tokens | 설정 방식 | 위치 |
|------|:----------:|----------|------|
| 분류 (route 등) | 256 | `MAX_TOKENS_CLASSIFICATION` 상수 | line 349, `_invoke_llm` 기본값 (line 366) |
| mq_node | 4096 | **인라인 하드코딩** | line 1273: `{"max_tokens": 4096}` |
| st_mq_node | 1024 | **인라인 하드코딩** | line 1442: `max_tokens=1024` |
| answer_node | 4096 | `MAX_TOKENS_ANSWER` 상수 | line 2121 |
| judge_node | 1024 | `MAX_TOKENS_JUDGE` 상수 | line 2407 |

`mq_node`의 `4096`은 `MAX_TOKENS_ANSWER` 상수와 동일한 값이지만, 상수를 참조하지 않고 직접 입력되어 있다. 상수를 수정해도 이 값은 바뀌지 않는다.

### 3.4 빈 검색결과 처리

```python
# langgraph_agent.py:690-691
def ref_json_to_text(ref_json: List[Dict[str, Any]]) -> str:
    if not ref_json:
        return "EMPTY"
```

검색 결과가 없으면 프롬프트의 `{ref_text}` placeholder에 `"EMPTY"` 문자열이 들어간다. 프롬프트들(`retrieval_ans_v1.yaml`, `retrieval_ans_en_v1.yaml`, `retrieval_ans_ja_v1.yaml`)은 구조화된 참조 형식(`[1] doc_id: content...`)을 기대하므로, `"EMPTY"`가 들어왔을 때 LLM의 동작이 비결정적이다.

### 3.5 언어 템플릿 fallback

```python
# langgraph_agent.py:2111-2116 (의사코드)
if detected_language == "en" and spec.retrieval_ans_en:
    template = spec.retrieval_ans_en
elif detected_language == "ja" and spec.retrieval_ans_ja:
    template = spec.retrieval_ans_ja
else:
    template = spec.retrieval_ans   # 한국어 fallback
```

영어/일본어 템플릿이 `_try_load_prompt()`(`prompt_loader.py` line 52-84)로 로드되므로, 해당 YAML 파일이 없으면 `None`이 된다. 이 경우 영어 질문에 한국어 프롬프트가 적용되어 LLM의 명령 해석이 불안정해질 수 있다.

### 3.6 히스토리 요약 truncation

```python
# chat_history_service.py:71-72
summary: str  # assistant만 사용: 답변 요약 (truncate 150자)
```

대화 히스토리 요약이 150자로 잘리며, 이 요약이 후속 질문의 컨텍스트로 사용될 때 중요한 정보가 손실될 수 있다.

---

## 4. 유형 C: 코드 수정이 전파되지 않는 구조적 문제

### 4.1 단일 거대 파일 (125KB)

`langgraph_agent.py`가 약 3,500줄(125KB)로, 다음 요소가 모두 한 파일에 존재한다:

| 요소 | 예시 |
|------|------|
| 상태 정의 | `AgentState` TypedDict (line 83-166) |
| 전역 상수 | `MAX_TOKENS_*`, `EXPAND_TOP_K` 등 (line 349-356) |
| 헬퍼 함수 | `_invoke_llm`, `results_to_ref_json`, `_is_garbage_query` 등 |
| 노드 함수 15+ | `mq_node`, `retrieve_node`, `answer_node`, `judge_node` 등 |
| 라우팅 로직 | `route_node`, `st_gate_node` 등 |

**영향**:
- 한 노드의 수정이 같은 파일 내 다른 노드의 state 접근 패턴에 영향을 줄 수 있음
- diff가 커서 코드 리뷰 시 변경 범위 파악이 어려움
- IDE의 자동완성·리팩토링 도구 성능 저하

### 4.2 하드코딩 값 중복

같은 의미의 값이 상수 정의와 별도로 인라인 하드코딩되어 있어, 상수를 수정해도 인라인 값은 그대로 남는다.

| 의도 | 상수 정의 | 인라인 하드코딩 |
|------|----------|---------------|
| 확장 검색 top_k | `EXPAND_TOP_K = 20` (line 356) | `retry_expand_node` line 2529: `20`, line 2534: `20` |
| MQ 생성 토큰 | `MAX_TOKENS_ANSWER = 4096` (line 351) | `mq_node` line 1273: `{"max_tokens": 4096}` |
| retrieve 기본 top_k | 서비스에서 `retrieval_top_k=100` 전달 | `retrieve_node` 시그니처 (line 1500): 기본값 `20` |
| retrieve 최종 top_k | 서비스에서 `final_top_k=20` 전달 | `retrieve_node` 시그니처 (line 1501): 기본값 `10` |

**시나리오**: `EXPAND_TOP_K`를 30으로 변경하면, line 356의 상수만 변경되고 line 2529, 2534의 직접 입력된 `20`은 그대로 남는다. 테스트에서 확인하지 않으면 발견이 어렵다.

### 4.3 레거시·신규 상태 필드 이중 관리

MQ(Multi-Query) 관련 상태 필드가 구조 변경 과정에서 이중화되었다.

**현재 필드 구조**:
```
신규 필드 (mq_node가 생성):
  ├── retrieval_mq_list       ← 영어 쿼리 목록
  └── retrieval_mq_ko_list    ← 한국어 쿼리 목록

레거시 필드 (하위 호환용 복사):
  ├── general_mq_list         ← retrieval_mq_list와 동일 값 복사
  ├── setup_mq_list           ← 빈 배열
  ├── ts_mq_list              ← 빈 배열
  ├── general_mq_ko_list      ← retrieval_mq_ko_list와 동일 값 복사
  ├── setup_mq_ko_list        ← 빈 배열
  └── ts_mq_ko_list           ← 빈 배열
```

**문제 1 — 읽기 불일치**: `refine_queries_node` (line 2586-2620)는 레거시 필드(`setup_mq_ko_list`, `ts_mq_ko_list`, `general_mq_ko_list`)를 읽는다. 신규 필드(`retrieval_mq_ko_list`)만 수정하면 이 노드에는 반영되지 않는다.

**문제 2 — 초기화 불일치**: `retry_mq_node` (line 2567-2572)는 레거시 필드 6개만 초기화하고, 신규 필드 2개(`retrieval_mq_list`, `retrieval_mq_ko_list`)는 건드리지 않는다. 재시도 시 이전 MQ 값이 남아 있을 수 있다.

```python
# langgraph_agent.py:1319-1329 — mq_node 반환값 (이중 기록)
return {
    "retrieval_mq_list": retrieval_mq_list,       # 신규 필드에 기록
    "retrieval_mq_ko_list": retrieval_mq_ko_list, # 신규 필드에 기록
    "general_mq_list": retrieval_mq_list,          # 레거시 필드에 복사
    "general_mq_ko_list": retrieval_mq_ko_list,    # 레거시 필드에 복사
    "setup_mq_list": [],                           # 레거시 필드 비우기
    "ts_mq_list": [],
    "setup_mq_ko_list": [],
    "ts_mq_ko_list": [],
}
```

### 4.4 동일 로직 3곳 중복

쿼리 추출 + 이중언어 MQ 생성 패턴이 3개 노드에서 독립적으로 구현되어 있다.

| 노드 | 위치 | 역할 |
|------|------|------|
| `mq_node` | line 1255-1301 | 최초 멀티쿼리 생성 |
| `st_mq_node` | line 1414-1465 | 특수 태스크용 멀티쿼리 생성 |
| `refine_queries_node` | line 2586-2620 | 재시도 시 쿼리 정제 |

세 곳 모두 다음 패턴을 반복한다:
1. `state.get("query_en") or state["query"]`로 영어 쿼리 추출
2. `state.get("query_ko") or state["query"]`로 한국어 쿼리 추출
3. 영어 MQ 생성: `system + "\n\n**IMPORTANT: Generate all queries in English.**"`
4. 한국어 MQ 생성: `system + "\n\n**중요: 모든 검색어를 반드시 한국어로 생성하세요...**"`
5. 병합 및 중복 제거

**위험**: 한 곳에서 버그를 수정하거나 로직을 개선해도 나머지 2곳에는 반영되지 않는다. 실제로 `_is_garbage_query` 필터가 `mq_node`에만 적용되고 나머지에는 누락된 사례가 이미 있었다 (Task 2 수정 과정에서 확인).

### 4.5 레이어 간 기본값 불일치

같은 파라미터의 기본값이 레이어마다 다르다.

```
레이어별 기본값 비교:

                        top_k (final)    retrieval_top_k
                        =============    ===============
API Layer (agent.py)         20               100
Service (rag_agent.py)       20               100
Node (langgraph_agent.py)    10                20       ← 불일치!
```

코드 근거:

```python
# agent.py:368-369 — API 레이어
top_k=20, retrieval_top_k=100

# langgraph_rag_agent.py:82 — 서비스 레이어
top_k: int = 20, retrieval_top_k: int = 100

# langgraph_agent.py:1500-1501 — 노드 레이어
def retrieve_node(state, *, retriever, reranker=None,
                  retrieval_top_k: int = 20,   # ← API/서비스와 다름
                  final_top_k: int = 10,        # ← API/서비스와 다름
) -> Dict[str, Any]:
```

정상 경로(서비스 레이어 → 노드)에서는 `functools.partial`로 올바른 값이 바인딩되지만:
```python
# langgraph_rag_agent.py:387-389
functools.partial(
    retrieve_node,
    retrieval_top_k=self.retrieval_top_k,  # 100
    final_top_k=self.top_k,                 # 20
)
```

**위험**: 노드를 직접 호출하는 코드(테스트, 디버깅, `retrieve_only()` 등)에서는 `functools.partial` 바인딩 없이 호출할 경우 `retrieval_top_k=20`, `final_top_k=10`이 적용된다. 디버깅 시 "왜 결과가 다른가?"를 추적하기 어렵다.

---

## 5. 문제 간 인과관계

```
┌─────────────────────────────────────────────────────────────┐
│  C. 구조적 문제 (근본 원인)                                    │
│                                                              │
│  C-1. 단일 거대 파일 (125KB)                                  │
│   └→ C-4. 동일 로직 3곳 중복 (분리 어려워 복사·붙여넣기)          │
│       └→ C-3. 레거시·신규 필드 이중화 (리팩토링 미완료)           │
│                                                              │
│  C-2. 하드코딩 값 중복                                        │
│   └→ C-5. 레이어 간 기본값 불일치                               │
└─────────────────────┬───────────────────────────────────────┘
                      │ 파생
┌─────────────────────▼───────────────────────────────────────┐
│  B. 답변 품질 편차                                            │
│                                                              │
│  C-3 → B-2. 문서 참조 필드 5종 혼재                             │
│         └→ B-1. truncation 불일치 (200자 vs 무제한)             │
│              └→ 같은 질문에 다른 품질의 답변                      │
│                                                              │
│  B-3. LLM 파라미터 미제어 (temperature, max_tokens)            │
│  B-4. 빈 검색결과 시 "EMPTY" 문자열                             │
│  B-5. 언어 템플릿 fallback                                    │
└─────────────────────┬───────────────────────────────────────┘
                      │ 파생
┌─────────────────────▼───────────────────────────────────────┐
│  A. 설정값 미반영                                              │
│                                                              │
│  A-1. Request 타입 누락 (multiQuery, fieldWeights)            │
│   └→ A-4. fieldWeights dead control                          │
│                                                              │
│  C-5 → A-2. 싱글톤 에이전트 하드코딩 (top_k 고정)               │
│  C-5 → A-3. mq_node에 search_override 미전달                  │
└─────────────────────────────────────────────────────────────┘
```

**핵심**: 구조적 문제(C)가 품질 편차(B)와 설정 미반영(A)의 근본 원인이다. C를 해결하지 않으면 A, B의 개별 수정이 새로운 불일치를 만들 수 있다.

---

## 6. 영향받는 파일 목록

| 파일 | 크기 | 관련 문제 |
|------|:----:|----------|
| `backend/llm_infrastructure/llm/langgraph_agent.py` | 125KB | B-1~5, C-1~5 |
| `backend/services/agents/langgraph_rag_agent.py` | — | A-3, C-5 |
| `backend/api/routers/agent.py` | 47KB | A-2 |
| `backend/api/routers/search.py` | 18KB | A-1 |
| `frontend/src/features/retrieval-test/hooks/use-retrieval-test.ts` | — | A-1 |
| `frontend/src/features/retrieval-test/context/retrieval-test-context.tsx` | — | A-1 |
| `frontend/src/features/retrieval-test/components/search-config-panel.tsx` | — | A-4 |
| `backend/llm_infrastructure/llm/prompts/*.yaml` | — | B-5 |
| `backend/llm_infrastructure/llm/prompt_loader.py` | — | B-5 |
| `backend/services/chat_history_service.py` | — | B-6 |

---

## 7. 부록: 발견된 문제 전체 목록

| ID | 문제 | 위치 (파일:라인) | 심각도 |
|----|------|----------------|:------:|
| A-1 | Request 타입에 multiQuery/fieldWeights 누락 | use-retrieval-test.ts:118, search.py:343 | 높음 |
| A-2 | 싱글톤 에이전트 top_k=20 하드코딩 | agent.py:368, 389 | 높음 |
| A-3 | mq_node에 search_override 미전달 | langgraph_rag_agent.py:773, 378 | 높음 |
| A-4 | fieldWeights dead control | search-config-panel.tsx:231-294 | 중간 |
| B-1 | truncation 불일치 (200자 vs 무제한) | langgraph_agent.py:649, 2085 | 높음 |
| B-2 | 문서 참조 필드 5종 혼재 | langgraph_agent.py:141-145, 2081-2087 | 높음 |
| B-3a | temperature 미지정 | langgraph_agent.py:359 (전체 호출) | 중간 |
| B-3b | max_tokens 인라인 하드코딩 | langgraph_agent.py:1273, 1442 | 중간 |
| B-4 | 빈 검색결과 시 "EMPTY" 반환 | langgraph_agent.py:690-691 | 낮음 |
| B-5 | 언어 템플릿 한국어 fallback | langgraph_agent.py:2111-2116 | 낮음 |
| B-6 | 히스토리 요약 150자 truncation | chat_history_service.py:71-72 | 낮음 |
| C-1 | 단일 파일 125KB (3,500줄) | langgraph_agent.py 전체 | 중간 |
| C-2 | 하드코딩 값 중복 | langgraph_agent.py:1273, 2529, 2534 | 중간 |
| C-3 | 레거시·신규 MQ 필드 이중화 | langgraph_agent.py:1319-1329, 2567-2572 | 중간 |
| C-4 | 이중언어 MQ 생성 로직 3곳 중복 | langgraph_agent.py:1255, 1414, 2586 | 중간 |
| C-5 | 레이어 간 기본값 불일치 (top_k) | langgraph_agent.py:1500-1501 vs agent.py:368 | 중간 |
| C-6 | MAX_REF_CHARS_ANSWER 상수 미사용 | langgraph_agent.py:353 | 낮음 |

---

## 8. 시스템 간 통신 구조 진단

### 8.1 전체 통신 흐름

```
┌─────────────────────────────────────────────────────────────────────┐
│  사용자 브라우저                                                      │
└──────┬──────────────────────────────┬───────────────────────────────┘
       │ REST (POST /api/agent/run)   │ SSE (POST /api/agent/run/stream)
       │ fetch + JSON                 │ fetch + ReadableStream
       ▼                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FastAPI (Uvicorn)                                                   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  API Routers                                                  │   │
│  │  agent.py (1,144줄) │ search.py │ conversations.py │ ...     │   │
│  └──────────┬───────────────────────────────────────────────────┘   │
│             │ Python 함수 호출 (동기)                                 │
│  ┌──────────▼───────────────────────────────────────────────────┐   │
│  │  Service Layer                                                │   │
│  │  LangGraphRAGAgent │ SearchService │ ChatHistoryService       │   │
│  └──────────┬───────────┬───────────────┬───────────────────────┘   │
│             │           │               │                           │
│  ┌──────────▼──┐ ┌──────▼───────┐ ┌────▼──────────────┐           │
│  │  vLLM       │ │ Elasticsearch│ │ TEI (Embedding)   │           │
│  │  httpx POST │ │ elasticsearch│ │ httpx POST        │           │
│  │  /v1/chat/  │ │ -py client   │ │ /embed            │           │
│  │  completions│ │              │ │                    │           │
│  └──────┬──────┘ └──────┬───────┘ └────┬──────────────┘           │
└─────────┼───────────────┼──────────────┼────────────────────────────┘
          │ HTTP          │ HTTP         │ HTTP
          ▼               ▼              ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐
   │ vLLM GPU │    │ ES 8.x   │    │ TEI GPU  │
   │ 서버     │    │ 클러스터  │    │ 서버     │
   └──────────┘    └──────────┘    └──────────┘
```

### 8.2 통신 프로토콜별 현황

| 구간 | 프로토콜 | 라이브러리 | 에러 처리 | 재시도 | 타임아웃 |
|------|---------|-----------|----------|:------:|---------|
| FE → BE (일반) | REST (JSON) | fetch / apiClient | 부분적 | X | X |
| FE → BE (스트리밍) | SSE (ReadableStream) | 커스텀 sse.ts | 부분적 | X | X |
| BE → vLLM | HTTP POST | httpx.Client | O (raise) | **X** | 고정값 |
| BE → Elasticsearch | HTTP | elasticsearch-py | 부분적 | **X** | **X** |
| BE → TEI | HTTP POST | httpx.Client | **X** | **X** | 고정값 |
| BE → MinIO | HTTP | minio-py | O | O | O |

**핵심 문제**: vLLM, Elasticsearch, TEI 세 가지 핵심 외부 서비스 모두 **재시도 로직이 없고**, vLLM과 TEI는 **요청별 타임아웃 조정이 불가**하다.

### 8.3 통신 구조의 문제점

#### 8.3.1 SSE 스트리밍 경로 결정이 URL 기반

**파일**: `frontend/src/features/chat/hooks/use-chat-session.ts` (line 574)

```typescript
const canStream = env.chatPath.endsWith("/run");
if (!canStream) { /* REST */ } else { /* SSE */ }
```

REST와 SSE 중 어느 것을 사용할지가 URL 경로의 접미사(`/run`)로 결정된다. 환경 변수 `VITE_CHAT_PATH`의 값에 따라 스트리밍 여부가 암묵적으로 바뀌는 취약한 구조다.

#### 8.3.2 SSE 이벤트 큐 오버플로 가능성

**파일**: `backend/api/routers/agent.py` (line ~910)

```python
queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)

def _enqueue(event):
    def _put():
        if not queue.full():
            queue.put_nowait(payload)  # 큐 가득 차면 이벤트 버림
    loop.call_soon_threadsafe(_put)
```

큐 크기가 256으로 고정되어 있고, 큐가 가득 차면 이벤트를 무시한다. 느린 클라이언트 연결에서 중요한 이벤트(최종 답변 등)가 유실될 수 있다.

#### 8.3.3 외부 서비스 장애 시 단일 실패점 (SPOF)

| 서비스 | 장애 시 영향 | 폴백 전략 | 판정 |
|--------|------------|----------|:----:|
| vLLM | 모든 LLM 호출 실패 → 답변 생성 불가 | 없음 | SPOF |
| Elasticsearch | 모든 검색·히스토리 실패 | 없음 | SPOF |
| TEI | 임베딩 실패 → dense 검색 불가 | 없음 | SPOF |
| MinIO | 이미지 로드 실패 | 로컬 폴백 가능 | 부분 SPOF |

vLLM이 일시적으로 응답하지 않아도 재시도 없이 즉시 `RuntimeError`가 발생한다:

```python
# llm/engines/vllm.py:76-85
except httpx.HTTPStatusError as exc:
    raise RuntimeError(f"vLLM request failed: status={exc.response.status_code}")
except httpx.HTTPError as exc:
    raise RuntimeError(f"vLLM request failed: error={exc}")
```

---

## 9. Frontend 구조 진단

### 9.1 전체 구조

```
frontend/src/
├── app/
│   ├── router.tsx          # 환경별 라우터 분기
│   ├── router.dev.tsx      # 개발 빌드 라우트 (7개 페이지)
│   └── router.prod.tsx     # 프로덕션 빌드 라우트 (2개 페이지)
├── components/
│   ├── layout/
│   │   ├── index.tsx       # 1,443줄 — 레이아웃 + 리뷰 패널 + 유틸리티
│   │   └── left-sidebar.tsx
│   └── global-search/
├── features/
│   ├── chat/               # 핵심 기능
│   │   ├── hooks/use-chat-session.ts  # 1,100줄 — 단일 훅
│   │   ├── pages/chat-page.tsx
│   │   ├── components/
│   │   └── types.ts        # 373줄
│   ├── batch-answer/
│   ├── retrieval-test/
│   └── ...
├── lib/
│   ├── api-client.ts       # 공용 REST 클라이언트
│   └── sse.ts              # SSE 핸들러
└── config/
    └── env.ts              # 환경 변수
```

### 9.2 발견된 문제

#### 9.2.1 거대 단일 파일 2개

| 파일 | 줄 수 | 포함 내용 |
|------|:-----:|----------|
| `layout/index.tsx` | 1,443 | Layout, ReviewPanel, RegeneratePanel, RetrievedDocs, ChatLogs, 유틸리티 함수 |
| `use-chat-session.ts` | 1,100 | 세션 관리, 메시지 처리, 스트리밍, 피드백, 분기, 인터럽트 |

두 파일 모두 여러 관심사를 혼합하고 있어 개별 기능 수정 시 부작용 추적이 어렵다.

#### 9.2.2 API 호출 패턴 3종 혼재

같은 프로젝트 내에서 서로 다른 API 호출 방식이 사용된다:

| 기능 | 패턴 | 파일 |
|------|------|------|
| Chat | `apiClient.post()` (공용 클라이언트) | chat/api.ts |
| Batch Answer | `apiClient.get()` + 수동 URL 빌드 | batch-answer/api.ts |
| Retrieval Test | **직접 `fetch()` 호출** (apiClient 미사용) | use-retrieval-test.ts:137-143 |

Retrieval Test가 `apiClient`를 사용하지 않고 `fetch()`를 직접 호출하므로, 공용 클라이언트에 추가된 인터셉터나 에러 처리가 적용되지 않는다.

#### 9.2.3 API 엔드포인트 분산

엔드포인트 경로가 중앙에서 관리되지 않고 각 feature 모듈에 직접 하드코딩되어 있다:

```
chat:             env.chatPath 또는 "/api/agent/run"
retrieval-test:   "/api/search/chat-pipeline"
batch-answer:     "/api/batch-answer/runs"
conversations:    "/api/conversations"
device-catalog:   "/api/device-catalog"
```

백엔드 URL이 변경되면 여러 파일을 찾아 수정해야 한다.

#### 9.2.4 상태 관리 파편화

3개의 Context가 Chat 기능 하나에 사용된다:

| Context | 역할 | 지속성 |
|---------|------|:------:|
| `ChatLogsContext` | 스트리밍 로그 저장 | 메모리만 (새로고침 시 소실) |
| `ChatHistoryContext` | 대화 히스토리 | ES 기반 |
| `ChatReviewContext` | 리뷰/재생성 UI 상태 | 메모리만 |

`ChatReviewContext`에서 핸들러 등록 패턴이 사용된다:

```typescript
// chat-review-context.tsx — 핸들러 등록 안티패턴
registerSubmitHandlers: (handlers: {
  submitReview: (...) => void;
  submitSearchQueries: (...) => void;
}) => void;
```

ChatPage가 Context에 핸들러를 등록하고, Layout이 Context에서 핸들러를 꺼내 호출하는 양방향 의존성이 발생한다.

#### 9.2.5 window 이벤트를 통한 암묵적 통신

```typescript
// left-sidebar.tsx:73
window.dispatchEvent(new CustomEvent("pe-agent:new-chat"));

// layout/index.tsx:78
window.addEventListener("pe-agent:new-chat", handleNewChat);
```

컴포넌트 간 통신을 React Context/Props 대신 `window.dispatchEvent`로 처리하여 데이터 흐름 추적이 어렵다.

#### 9.2.6 타입 중복

`DeviceInfo` 타입이 `use-chat-session.ts` (line 37-40)와 `types.ts` (line 47-50) 양쪽에 정의되어 있다. snake_case → camelCase 변환도 여러 곳에서 인라인으로 처리된다.

---

## 10. Backend 구조 진단

### 10.1 전체 구조

```
backend/
├── api/
│   ├── main.py             # FastAPI 앱 초기화
│   ├── dependencies.py     # DI (3가지 패턴 혼재)
│   └── routers/
│       ├── agent.py        # 1,144줄 — 에이전트 생성, 실행, 응답 변환
│       ├── search.py       # 검색 + ChatPipeline
│       ├── conversations.py # 대화 히스토리
│       ├── batch_answer.py  # 배치 답변
│       └── ...              # 14개 라우터
├── services/
│   ├── search_service.py   # God Object (검색 + 임베딩 + 리랭킹 + 쿼리 확장)
│   ├── agents/
│   │   └── langgraph_rag_agent.py  # 그래프 빌더 + 노드 바인딩
│   ├── chat_history_service.py
│   └── ...
├── llm_infrastructure/     # 핵심 RAG 인프라
│   ├── llm/                # LLM 엔진, 어댑터, 프롬프트
│   ├── retrieval/          # 검색 (dense, sparse, hybrid)
│   ├── embedding/          # 임베딩 (SentenceTransformer, TEI)
│   ├── reranking/          # 리랭킹
│   └── elasticsearch/      # ES 관리
└── config/
    └── settings.py         # 11개 Settings 클래스
```

### 10.2 발견된 문제

#### 10.2.1 의존성 주입 패턴 3종 혼재

같은 프로젝트에서 3가지 DI 패턴이 사용된다:

```python
# 패턴 1: @lru_cache (dependencies.py)
@lru_cache
def get_default_preprocessor() -> BasePreprocessor:
    return get_preprocessor(...)

# 패턴 2: 전역 변수 + setter (dependencies.py)
_search_service_instance: SearchService | None = None
def set_search_service(service: SearchService) -> None:
    global _search_service_instance
    _search_service_instance = service

# 패턴 3: 라우터 내부 팩토리 (agent.py)
_hil_agent: Optional[LangGraphRAGAgent] = None
def _get_hil_agent(llm, search_service, prompt_spec):
    global _hil_agent
    if _hil_agent is None:
        _hil_agent = LangGraphRAGAgent(...)
    return _hil_agent
```

어떤 서비스가 어디서 생성되는지 파악하려면 3가지 패턴을 모두 확인해야 한다.

#### 10.2.2 전역 싱글톤의 스레드 안전성 부재

`agent.py`의 에이전트 팩토리 함수에 lock이 없다:

```python
# agent.py:359-376 — 경쟁 조건 가능
def _get_hil_agent(llm, search_service, prompt_spec):
    global _hil_agent
    if _hil_agent is None:  # ← 두 요청이 동시에 통과 가능
        _hil_agent = LangGraphRAGAgent(...)
    return _hil_agent
```

동시 요청 시 두 스레드가 모두 `None` 체크를 통과하여 에이전트가 이중 생성될 수 있다.

또한 `MemorySaver` 체크포인터가 3개 에이전트에서 공유된다:

```python
# agent.py:41
_checkpointer: MemorySaver = MemorySaver()  # 모듈 레벨 전역

# 3개 에이전트가 동일 인스턴스 사용
_hil_agent = LangGraphRAGAgent(..., checkpointer=_checkpointer)
_auto_parse_agent = LangGraphRAGAgent(..., checkpointer=_checkpointer)
_device_selection_agent = LangGraphRAGAgent(..., checkpointer=_checkpointer)
```

thread_id 충돌 시 상태 오염 가능성이 있다.

#### 10.2.3 agent.py 라우터의 과도한 책임 (1,144줄)

하나의 라우터 파일이 다음 역할을 모두 담당한다:

| 기능 | 라인 범위 | 역할 |
|------|----------|------|
| ES 집계 쿼리 (Painless 스크립트) | 47-128 | 디바이스 목록 조회 |
| 응답 포매팅 유틸리티 | 130-212 | 요약, 참조 추출 |
| 히스토리 로딩/상태 빌드 | 215-356 | 대화 이력 관리 |
| 에이전트 팩토리 3개 | 359-415 | 싱글톤 생성 |
| Pydantic 모델 8개 | 434-521 | 요청/응답 스키마 |
| `/run` 엔드포인트 | 678-882 | 동기 실행 (200줄) |
| `/run/stream` 엔드포인트 | 886-1141 | SSE 스트리밍 (250줄) |

라우터가 비즈니스 로직, 데이터 모델, 인프라 접근을 모두 포함하는 God Object 상태다.

#### 10.2.4 라우터에서 직접 ES 쿼리 실행

```python
# agent.py:47-128 — 라우터가 직접 ES aggregation 실행
def _create_device_fetcher(search_service):
    es = search_service.es_engine.es
    index = search_service.es_engine.index_name
    agg_query = {
        "size": 0,
        "aggs": { "devices": { "terms": { "field": "device_name", ... } } }
    }
    result = es.search(index=index, body=agg_query)
```

라우터가 SearchService의 내부 구현(`es_engine.es`)에 직접 접근하여 Elasticsearch에 묶이는 결합이 발생한다.

#### 10.2.5 에러 핸들링의 무음 실패

```python
# agent.py — 히스토리 로딩 실패 시 빈 리스트 반환
def _load_chat_history_from_session(session_id, limit=5):
    try:
        ...
    except Exception as e:
        logger.warning("Failed to load history: %s", e)
        return []  # 무음 실패
```

히스토리 로딩 실패가 사용자에게 전달되지 않아, 이전 대화 맥락 없이 답변이 생성되는데도 사용자는 인지하지 못한다.

#### 10.2.6 SearchService God Object

`SearchService`가 너무 많은 책임을 갖고 있다:

```python
# search_service.py — __init__에 7가지 관심사
def __init__(
    self,
    corpus,           # 코퍼스 관리
    method,           # 검색 방법
    dense_weight,     # 하이브리드 검색 가중치
    multi_query_enabled, # 쿼리 확장
    rerank_enabled,   # 리랭킹
    rerank_method,    # 리랭킹 방법
    ...
)
```

코퍼스 관리, 임베딩, 검색, 쿼리 확장, 리랭킹이 한 클래스에 결합되어 있어, 리랭킹만 변경하고 싶어도 SearchService 전체를 이해해야 한다.

#### 10.2.7 순환 의존성

```
llm_infrastructure/llm/langgraph_agent.py
  └→ imports from: backend.services.search_service (서비스 레이어)

llm_infrastructure/elasticsearch/document.py
  └→ imports from: backend.services.ingest.document_ingest_service
```

인프라 레이어가 서비스 레이어를 import하는 역방향 의존이 존재한다. 클린 아키텍처 원칙에서 인프라는 서비스에 의존하지 않아야 한다.

---

## 11. LLM Infrastructure 구조 진단

### 11.1 전체 구조

```
llm_infrastructure/
├── llm/
│   ├── langgraph_agent.py     # 125KB — 에이전트 그래프 전체
│   ├── engines/
│   │   └── vllm.py            # vLLM 엔진 (httpx)
│   ├── adapters/
│   │   └── vllm.py            # vLLM 어댑터 (OpenAI 호환)
│   ├── prompts/               # YAML 프롬프트 템플릿
│   └── prompt_loader.py       # 프롬프트 로더
├── retrieval/
│   ├── base.py                # BaseRetriever (추상)
│   ├── registry.py            # 레지스트리 패턴
│   ├── adapters/
│   │   ├── dense.py           # Dense 검색
│   │   ├── hybrid.py          # Dense + Sparse + RRF
│   │   └── es_hybrid.py       # ES 네이티브 하이브리드
│   └── engines/
│       └── es_search.py       # ES 검색 엔진
├── embedding/
│   ├── base.py                # BaseEmbedder (추상)
│   ├── registry.py            # 레지스트리 패턴
│   └── adapters/
│       ├── sentence_transformer.py
│       └── tei.py             # TEI 클라이언트
├── reranking/
│   └── base.py                # BaseReranker (추상)
├── query_expansion/
│   └── adapters/
│       └── llm.py             # LLM 기반 쿼리 확장
└── elasticsearch/
    └── manager.py             # ES 인덱스 관리
```

### 11.2 잘 설계된 부분

레지스트리 패턴과 추상 기반 클래스는 잘 구현되어 있다:

- `BaseEmbedder` → `embed()`, `embed_batch()` 추상 메서드
- `BaseRetriever` → `retrieve(query, top_k)` 단일 인터페이스
- `BaseReranker` → `rerank(query, results, top_k)` 단일 인터페이스

하이브리드 검색 파이프라인도 구조적으로 깔끔하다:

```
Query → Embedding → ES Hybrid Search → RRF 또는 Script Score → 결과
                                        │
                                        ├─ RRF 실패 시 → Script Score 폴백
                                        └─ Device Boost 적용
```

### 11.3 발견된 문제

#### 11.3.1 vLLM 통신: 재시도·서킷브레이커 없음

```python
# llm/engines/vllm.py:70-85
try:
    resp = self._client.post(...)
    resp.raise_for_status()
except httpx.HTTPStatusError as exc:
    raise RuntimeError(f"vLLM request failed: status={exc.response.status_code}, body={detail}")
except httpx.HTTPError as exc:
    raise RuntimeError(f"vLLM request failed: base_url={self.base_url}, error={exc}")
```

- 429 (Rate Limit)와 503 (Service Unavailable) 구분 없이 동일하게 `RuntimeError` 발생
- 일시적 네트워크 오류에도 즉시 실패
- 재시도, 지수 백오프, 서킷브레이커 패턴 없음

#### 11.3.2 TEI 통신: 에러 처리 부재

```python
# embedding/adapters/tei.py:43-72
response = self._client.post("/embed", json=payload)
response.raise_for_status()  # 에러 시 바로 예외

# 소멸자에서 예외 무시
def __del__(self):
    try:
        self._client.close()
    except Exception:
        pass  # 무음 처리
```

`embed()` 및 `embed_batch()` 메서드에 try-except가 없어, TEI 서버 오류가 상위 레이어로 그대로 전파된다.

#### 11.3.3 쿼리 확장의 vLLM 하드 의존

```python
# query_expansion/adapters/llm.py:55-63
self._llm = get_llm("vllm", version="v1", base_url=..., model=...)
```

쿼리 확장이 `"vllm"` 문자열로 LLM 엔진을 직접 지정한다. 다른 LLM 백엔드로 교체하려면 이 코드를 수정해야 한다.

단, 쿼리 확장은 유일하게 graceful degradation이 구현되어 있다:
```python
except Exception as e:
    logger.warning(f"Query expansion failed: {e}. Returning original query only.")
    return ExpandedQueries(original_query=query, expanded_queries=[])
```

#### 11.3.4 ES 연결 관리

- `EsIndexManager`에 `health_check()` 메서드가 존재하지만(line 361-367), 실제로 호출되는 곳이 없음
- 연결 풀링 설정이 명시적으로 구성되지 않음
- ES 오류 시 즉시 예외 전파 (재시도 없음)

#### 11.3.5 타임아웃의 고정성

vLLM과 TEI 모두 클라이언트 초기화 시 타임아웃이 고정된다:

```python
# vllm.py:37
self._client = httpx.Client(base_url=base_url, timeout=timeout)

# tei.py:34
self._client = httpx.Client(base_url=base_url, timeout=timeout)
```

요청별로 타임아웃을 조정할 수 없다. 짧은 분류 요청(256 토큰)과 긴 답변 생성(4,096 토큰)에 동일한 타임아웃이 적용된다.

---

## 12. 전체 문제 요약 및 인과관계

### 12.1 레이어별 문제 요약

```
┌─ Frontend ───────────────────────────────────────────────────────┐
│  F-1. 거대 파일 2개 (layout 1,443줄, use-chat-session 1,100줄)    │
│  F-2. API 호출 패턴 3종 혼재 (apiClient, fetch, SSE)              │
│  F-3. 엔드포인트 하드코딩 분산                                     │
│  F-4. 상태 관리 파편화 (Context 3개 + 핸들러 등록 안티패턴)         │
│  F-5. window 이벤트 기반 암묵적 통신                               │
│  F-6. 타입 중복 (DeviceInfo 등)                                   │
└──────────────────────────────┬────────────────────────────────────┘
                               │
┌─ FE↔BE 통신 ─────────────────┼────────────────────────────────────┐
│  I-1. SSE 경로 결정이 URL 접미사 기반 (취약)                        │
│  I-2. SSE 이벤트 큐 오버플로 가능 (maxsize=256, best-effort drop)  │
│  I-3. 설정값 전파 파이프라인 누락 (A-1~4)                          │
└──────────────────────────────┼────────────────────────────────────┘
                               │
┌─ Backend ────────────────────┼────────────────────────────────────┐
│  B-API-1. DI 패턴 3종 혼재                                        │
│  B-API-2. agent.py 1,144줄 God Object                            │
│  B-API-3. 라우터에서 직접 ES 쿼리 실행                             │
│  B-API-4. 전역 싱글톤 스레드 안전성 부재                            │
│  B-API-5. MemorySaver 3개 에이전트 공유                            │
│  B-API-6. SearchService God Object                                │
│  B-API-7. 순환 의존성 (인프라 → 서비스)                            │
│  B-API-8. 에러 핸들링 무음 실패                                    │
└──────────────────────────────┼────────────────────────────────────┘
                               │
┌─ BE↔LLM 통신 ───────────────┼────────────────────────────────────┐
│  L-1. vLLM 재시도·서킷브레이커 없음                                │
│  L-2. TEI 에러 처리 부재                                          │
│  L-3. ES 연결 health check 미사용                                 │
│  L-4. 3대 SPOF (vLLM, ES, TEI)                                   │
│  L-5. 요청별 타임아웃 조정 불가                                    │
└──────────────────────────────┼────────────────────────────────────┘
                               │
┌─ LLM Infrastructure ────────┼────────────────────────────────────┐
│  기존 진단 (A-1~4, B-1~6, C-1~6)                                  │
│  langgraph_agent.py 125KB 단일 파일에 집중된 문제                   │
└──────────────────────────────────────────────────────────────────┘
```

### 12.2 전체 인과관계

```
구조적 근본 원인
├── 거대 단일 파일 3개 (langgraph_agent 125KB, agent.py 1,144줄, layout.tsx 1,443줄)
│   └→ 로직 중복, 관심사 혼합, 수정 미전파
├── 일관성 부재 (DI 3종, API 호출 3종, 에러 처리 3종)
│   └→ 디버깅 어려움, 새 기능 추가 시 어느 패턴을 따를지 모호
└── 외부 서비스 내결함성 부재 (재시도·폴백·서킷브레이커 없음)
    └→ 일시적 장애가 전체 시스템 장애로 확대

  → 이 근본 원인들이 결합하여:
    → 설정값 미반영 (전파 파이프라인의 각 레이어에서 개별적으로 누락)
    → 답변 품질 편차 (상태 필드 혼재, truncation 불일치)
    → 코드 수정 미전파 (동일 로직 중복, 레거시 필드 잔존)
```

### 12.3 발견된 전체 문제 수

| 카테고리 | 높음 | 중간 | 낮음 | 합계 |
|---------|:----:|:----:|:----:|:----:|
| A. 설정값 미반영 | 3 | 1 | 0 | 4 |
| B. 답변 품질 편차 | 2 | 2 | 3 | 7 |
| C. 코드 수정 미전파 | 0 | 5 | 1 | 6 |
| F. 프론트엔드 구조 | 2 | 3 | 1 | 6 |
| I. FE↔BE 통신 | 1 | 2 | 0 | 3 |
| B-API. 백엔드 구조 | 3 | 4 | 1 | 8 |
| L. LLM 인프라 통신 | 2 | 2 | 1 | 5 |
| **합계** | **13** | **19** | **7** | **39** |
