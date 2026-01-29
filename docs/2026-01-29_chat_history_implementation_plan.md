# RAG Agent 대화 히스토리 및 doc_lookup 구현 계획

> 작성일: 2026-01-29
> 관련 문서: `2026-01-29_chat_history_and_doc_lookup_proposal.md`
> 상태: **시니어 피드백 반영 완료**

---

## 1. 구현 개요

### 1.1 목표
1. 대화 히스토리 전달 기능 구현
2. doc_lookup 라우트 추가 (문서 직접 조회)
3. 기존 플로우와 통합

### 1.2 구현 범위 결정 사항

| 항목 | 선택 | 이유 |
|------|------|------|
| History 전달 방식 | **클라이언트 전달** (MVP) | 서버 세션 관리 복잡도 회피 |
| 요약 방식 | **user 원문 유지 + assistant truncate** | user 의도 보존, 토큰 절약 |
| Title 없는 문서 | **doc_id에서 추출 또는 content snippet** | 추가 조회 없이 처리 |
| doc_lookup 구현 | **fetch_doc_chunks 직접 사용** | 기존 retrieve_node 우회 |
| LLM 판단 범위 | **의도 판단만** (doc_id 생성 안함) | 보안/정확도 |
| 모호성 처리 | **가장 최근 doc_id 자동 선택** | UX 단순화, soft confirmation 고려 |

### 1.3 시니어 피드백 반영 사항

| 피드백 | 반영 내용 |
|--------|----------|
| doc_ids 우선순위 | `doc_ids`를 rank 순 정렬, 첫 번째가 가장 중요 |
| entities 필드 추가 | `auto_parse` 결과를 history에 포함 (P1에서 검토) |
| 실행 순서 조정 | chat 우회 → Rule doc_id → LLM 판단 → router |
| 모호성 처리 | doc_ids 2개 이상 시 가장 최근 1개로 자동 선택 |
| 측정 계획 | latency, 토큰, ES 호출 로깅 추가 (P2) |
| 접근 제어 | doc_id 검증 + fallback 필수 |

### 1.4 현재 코드 구조 확인 사항

| 항목 | 현재 코드 | 비고 |
|------|----------|------|
| 요청 필드명 | `message` (query 아님) | `AgentRequest.message` |
| auto_parse | 규칙 기반만 사용 | LLM fallback 없음 (별도 구현됨) |
| doc_id 조회 | `fetch_doc_chunks()` 존재 | `es_search_service.py` |
| LLM 호출 방식 | `_invoke_llm(llm, system, user)` | `llm.invoke()` 아님 |
| selected_doc_ids | AgentRequest에 이미 존재 | 재생성용 필드 |

### 1.4 작업 범위 (BE/FE 분리)

**Backend:**
- AgentState에 chat_history 추가
- AgentRequest/Response에 history 관련 필드 추가
- doc_lookup 라우트 및 노드 구현
- fetch_doc_chunks 기반 직접 조회

**Frontend (별도 작업 필요):**
- chat_history를 매 요청에 포함
- 응답의 summary, refs, doc_ids 저장
- 다음 요청 시 history로 전달

---

## 2. 단계별 구현 계획

### Phase 1: 기반 작업 (AgentState + API)

#### Task 1.1: AgentState에 chat_history 필드 추가

**파일:** `backend/llm_infrastructure/llm/langgraph_agent.py`

**필드명 규칙:**
- `user`: `content` 필드 사용 (원본 질문)
- `assistant`: `summary` 필드 사용 (요약) - content와 혼동 방지

```python
class ChatHistoryEntry(TypedDict, total=False):
    role: str  # "user" | "assistant"
    # user용 필드
    content: str  # user만 사용: 원본 질문 (그대로 유지)
    # assistant용 필드
    summary: str  # assistant만 사용: 답변 요약 (truncate 150자)
    refs: List[str]  # 참조 문서 title (rank 순)
    doc_ids: List[str]  # 참조 문서 ID (rank 순 - 첫 번째가 가장 중요)
    # P1에서 검토: entities 추가
    # entities: Optional[Dict[str, str]]  # {"equipment": "SUPRA XP", "doc_type": "ts"}

class AgentState(TypedDict, total=False):
    query: str
    chat_history: List[ChatHistoryEntry]  # NEW
    lookup_doc_ids: List[str]  # NEW: doc_lookup용 doc_id 목록
    lookup_source: str  # NEW: "query" | "history"
    # ... 기존 필드
```

**History 구조 설계 원칙:**
- `user.content`: 원문 그대로 유지 (짧고 의도가 담겨있음)
- `assistant.summary`: truncate 요약 (150자) - **content 아님 주의**
- `assistant.doc_ids`: rank 순 정렬 (첫 번째가 주요 근거)
- `entities`: P1에서 검토 후 추가 여부 결정

**작업 내용:**
- [ ] `ChatHistoryEntry` TypedDict 정의 (user: content, assistant: summary 분리)
- [ ] `AgentState`에 `chat_history`, `lookup_doc_ids`, `lookup_source` 필드 추가

---

#### Task 1.2: API 요청/응답에 history 추가

**파일:** `backend/api/routers/agent.py`

**요청 스키마 (현재 `message` 필드 사용 주의):**
```python
class AgentRequest(BaseModel):
    message: str = Field(..., description="사용자 질문")  # 기존 (query 아님!)
    chat_history: Optional[List[ChatHistoryEntry]] = None  # NEW
    # ... 기존 필드
```

**응답 스키마:**
```python
class AgentResponse(BaseModel):
    answer: str
    summary: Optional[str] = None  # NEW: 답변 요약 (다음 턴 history용)
    refs: Optional[List[str]] = None  # NEW: 참조 문서 title
    ref_doc_ids: Optional[List[str]] = None  # NEW: 참조 문서 ID
    # ... 기존 필드
```

**작업 내용:**
- [ ] `AgentRequest`에 `chat_history` 필드 추가
- [ ] `AgentResponse`에 `summary`, `refs`, `ref_doc_ids` 필드 추가
- [ ] 요청 → AgentState로 history 전달
- [ ] 응답 생성 시 summary, refs, ref_doc_ids 포함

**주의:** 필드명 `doc_ids` → `ref_doc_ids`로 변경 (기존 `selected_doc_ids`와 구분)

---

#### Task 1.3: 답변 요약 유틸 함수

**파일:** `backend/llm_infrastructure/llm/langgraph_agent.py`

```python
def _summarize_answer(answer: str, max_length: int = 150) -> str:
    """답변을 요약 (truncate 방식)"""
    if len(answer) <= max_length:
        return answer
    return answer[:max_length].rsplit(' ', 1)[0] + "..."
```

**작업 내용:**
- [ ] `_summarize_answer` 함수 구현
- [ ] 단어 중간 잘림 방지 (공백 기준)

---

### Phase 2: doc_lookup 라우트 구현

#### Task 2.1: doc_id 패턴 감지 (Rule 기반)

**파일:** `backend/llm_infrastructure/llm/langgraph_agent.py`

```python
import re

DOC_ID_PATTERNS = [
    r"(myservice|gcb|sop)\s*(\d+)",  # myservice 29392
    r"(myservice|gcb|sop)[-_](\d+)",  # myservice-29392, myservice_29392
]

def _extract_doc_id_from_query(query: str) -> Optional[tuple[str, str]]:
    """쿼리에서 doc_type과 doc_id 추출 (Rule 기반)"""
    query_lower = query.lower()
    for pattern in DOC_ID_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            doc_type, doc_id = match.groups()
            return (doc_type, doc_id)
    return None
```

**작업 내용:**
- [ ] `DOC_ID_PATTERNS` 정의
- [ ] `_extract_doc_id_from_query` 함수 구현
- [ ] 테스트 케이스 작성

---

#### Task 2.2: doc_lookup 의도 판단 (LLM 기반)

**파일:** `backend/llm_infrastructure/llm/langgraph_agent.py`

**주의:** 현재 코드는 `_invoke_llm(llm, system, user)` 방식 사용

```python
def _format_history_for_prompt(chat_history: List[ChatHistoryEntry]) -> str:
    """chat_history를 프롬프트용 텍스트로 변환

    주의: user는 content, assistant는 summary 필드 사용
    """
    lines = []
    for entry in chat_history[-5:]:  # 최근 5턴만
        role = entry.get("role", "")
        if role == "user":
            lines.append(f"User: {entry.get('content', '')}")  # user는 content
        elif role == "assistant":
            summary = entry.get("summary", "")  # assistant는 summary
            refs = entry.get("refs", [])
            lines.append(f"Assistant: {summary}")
            if refs:
                lines.append(f"  참조 문서: {', '.join(refs)}")
    return "\n".join(lines)


def _detect_doc_lookup_intent(
    llm: BaseLLM,
    query: str,
    chat_history: List[ChatHistoryEntry],
) -> dict:
    """LLM으로 doc_lookup 의도 판단 (_invoke_llm 사용)"""

    # history가 없으면 doc_lookup 불가
    if not chat_history:
        return {"is_doc_lookup": False}

    # history 포맷팅
    history_text = _format_history_for_prompt(chat_history)

    system = "사용자의 질문이 이전 대화에서 언급된 문서를 참조하는지 판단하세요."
    user = f"""이전 대화:
{history_text}

현재 질문: {query}

판단 기준:
- "그 문서", "아까", "위에서 말한", "더 자세히" 등의 표현
- 이전 답변의 특정 부분을 깊이 묻는 경우

이 질문이 이전 문서를 참조하면 "yes", 아니면 "no"로만 답하세요."""

    response = _invoke_llm(llm, system, user)
    is_doc_lookup = "yes" in response.lower()

    return {"is_doc_lookup": is_doc_lookup}
```

**작업 내용:**
- [ ] `_format_history_for_prompt` 헬퍼 함수
- [ ] `_detect_doc_lookup_intent` 함수 구현 (`_invoke_llm` 사용)
- [ ] 간단한 yes/no 응답 파싱 (JSON 불필요)

---

#### Task 2.3: doc_lookup_node 구현

**파일:** `backend/llm_infrastructure/llm/langgraph_agent.py`

**중요:** `selected_doc_ids`는 retrieve_node에서 쿼리 기반 검색 후 필터링하므로 부적합.
**→ `fetch_doc_chunks`로 직접 조회해야 함.**

```python
def doc_lookup_node(
    state: AgentState,
    *,
    doc_fetcher: Callable[[str], List[RetrievalResult]],  # fetch_doc_chunks
) -> Dict[str, Any]:
    """doc_id로 직접 문서 조회 (MQ 생략, retrieve 우회)"""

    doc_ids = state.get("lookup_doc_ids", [])

    if not doc_ids:
        # fallback: 기존 검색 플로우로
        logger.info("doc_lookup_node: no doc_ids, fallback to MQ")
        return {"route": "general"}  # fallback route

    # fetch_doc_chunks로 직접 조회
    all_docs: List[RetrievalResult] = []
    valid_doc_ids: List[str] = []

    for doc_id in doc_ids[:3]:  # 최대 3개 문서
        chunks = doc_fetcher(doc_id)
        if chunks:
            all_docs.extend(chunks)
            valid_doc_ids.append(doc_id)
        else:
            logger.warning("doc_lookup_node: doc_id=%s not found", doc_id)

    if not all_docs:
        # 모든 doc_id 검증 실패 → fallback
        logger.info("doc_lookup_node: all doc_ids invalid, fallback to MQ")
        return {"route": "general"}

    # docs 필드에 직접 설정 → answer_node로 바로 이동
    return {
        "docs": all_docs,
        "lookup_doc_ids": valid_doc_ids,
        "skip_retrieve": True,  # retrieve/rerank 생략 플래그
    }
```

**작업 내용:**
- [ ] `doc_lookup_node` 함수 구현
- [ ] `doc_fetcher` (fetch_doc_chunks) 의존성 주입
- [ ] doc_id 검증 (ES 존재 확인)
- [ ] fallback 처리 (검증 실패 시 기존 MQ 경로)

---

#### Task 2.4: Router에 doc_lookup 분기 추가

**파일:** `backend/llm_infrastructure/llm/langgraph_agent.py`

**주의:** 현재 LLM 호출 방식은 `_invoke_llm(llm, system, user)` 사용

```python
def route_node(state: AgentState, *, llm: BaseLLM) -> Dict[str, Any]:
    query = state["query"]
    chat_history = state.get("chat_history", [])

    # 1. Rule 기반: doc_id 패턴 체크 (LLM 호출 없음, 빠름)
    doc_info = _extract_doc_id_from_query(query)
    if doc_info:
        doc_type, doc_id = doc_info
        # doc_type + doc_id 조합으로 full_doc_id 생성
        full_doc_id = f"{doc_id}"  # 또는 f"{doc_type}_{doc_id}"
        return {
            "route": "doc_lookup",
            "lookup_doc_ids": [full_doc_id],
            "lookup_source": "query",
        }

    # 2. LLM 기반: 과거 문서 참조 의도 체크
    if chat_history:
        intent = _detect_doc_lookup_intent_llm(llm, query, chat_history)
        if intent.get("is_doc_lookup"):
            # history에서 doc_ids 추출
            prev_doc_ids = _get_doc_ids_from_history(chat_history)
            if prev_doc_ids:
                return {
                    "route": "doc_lookup",
                    "lookup_doc_ids": prev_doc_ids,
                    "lookup_source": "history",
                }

    # 3. 기존 라우팅 (setup/ts/general)
    return _existing_route_logic(state, llm=llm)


def _detect_doc_lookup_intent_llm(
    llm: BaseLLM,
    query: str,
    chat_history: List[ChatHistoryEntry],
) -> dict:
    """LLM으로 doc_lookup 의도 판단 (_invoke_llm 사용)"""

    history_text = _format_history_for_prompt(chat_history)

    system = "사용자의 질문이 이전 대화에서 언급된 문서를 참조하는지 판단하세요."
    user = f"""이전 대화:
{history_text}

현재 질문: {query}

이 질문이 이전 문서를 더 자세히 묻는 질문이면 "yes", 아니면 "no"로만 답하세요."""

    response = _invoke_llm(llm, system, user)
    is_doc_lookup = "yes" in response.lower()

    return {"is_doc_lookup": is_doc_lookup}


def _get_doc_ids_from_history(chat_history: List[ChatHistoryEntry]) -> List[str]:
    """history에서 가장 최근 assistant의 doc_ids 추출"""
    for entry in reversed(chat_history):
        if entry.get("role") == "assistant":
            doc_ids = entry.get("doc_ids", [])
            if doc_ids:
                return doc_ids[:3]  # 최대 3개
    return []
```

**작업 내용:**
- [ ] `route_node`에 doc_lookup 분기 추가
- [ ] `_detect_doc_lookup_intent_llm` 함수 (기존 `_invoke_llm` 사용)
- [ ] `_get_doc_ids_from_history` 헬퍼 함수
- [ ] `_format_history_for_prompt` 헬퍼 함수
- [ ] Route 타입에 "doc_lookup" 추가: `Route = Literal["setup", "ts", "general", "doc_lookup"]`

---

### Phase 3: 그래프 연결

#### Task 3.1: LangGraph에 doc_lookup 노드/엣지 추가

**파일:** `backend/services/agents/langgraph_rag_agent.py`

**핵심:** doc_lookup은 MQ/retrieve를 우회하고 바로 answer로 연결

```python
# Route 타입 업데이트 (langgraph_agent.py)
Route = Literal["setup", "ts", "general", "doc_lookup"]

# 노드 추가
builder.add_node(
    "doc_lookup",
    self._wrap_node(
        "doc_lookup",
        functools.partial(doc_lookup_node, doc_fetcher=self.doc_fetcher),
    ),
)

# 엣지 추가 (router → doc_lookup)
def route_after_router(state):
    route = state.get("route")
    if route == "doc_lookup":
        return "doc_lookup"
    if route == "chat":  # 기존 chat_answer 경로
        return "chat_answer"
    # ... 기존 분기 (setup/ts/general)

builder.add_conditional_edges("router", route_after_router, {
    "doc_lookup": "doc_lookup",
    "chat_answer": "chat_answer",
    "setup": "setup_mq",
    "ts": "ts_mq",
    "general": "general_mq",
})

# doc_lookup 이후 분기: 성공 → answer, 실패 → fallback
def route_after_doc_lookup(state):
    if state.get("docs"):  # 문서 조회 성공
        return "expand"  # 또는 "answer" (expand 필요 시)
    else:  # fallback
        return "general_mq"

builder.add_conditional_edges("doc_lookup", route_after_doc_lookup, {
    "expand": "expand_related_docs",
    "general_mq": "general_mq",  # fallback
})
```

**작업 내용:**
- [ ] Route 타입에 "doc_lookup" 추가
- [ ] `doc_lookup` 노드 등록 (doc_fetcher 주입)
- [ ] 조건부 엣지 추가 (router → doc_lookup)
- [ ] doc_lookup 이후 분기 (성공 → expand/answer, 실패 → fallback)

---

#### Task 3.2: Fallback 처리 상세

**doc_lookup 실패 시나리오:**

| 상황 | 처리 |
|------|------|
| doc_ids가 비어있음 | router에서 이미 걸러짐 (history 없음) |
| ES에 doc_id 없음 | doc_lookup_node에서 fallback route 반환 |
| 일부만 유효 | 유효한 것만 사용, 나머지 무시 |

**그래프 플로우:**

```
router
  │
  ├─ route=doc_lookup ─→ doc_lookup_node
  │                          │
  │                          ├─ docs 있음 → expand → answer
  │                          │
  │                          └─ docs 없음 → general_mq (fallback)
  │
  ├─ route=setup ─→ setup_mq → ...
  ├─ route=ts ─→ ts_mq → ...
  └─ route=general ─→ general_mq → ...
```

**작업 내용:**
- [ ] fallback 조건부 엣지 구현
- [ ] 로깅 추가 (doc_lookup 성공/실패)
- [ ] 테스트: doc_id 없는 경우 fallback 확인

---

### Phase 4: 통합 테스트

#### Task 4.1: 단위 테스트

**파일:** `backend/tests/test_doc_lookup.py`

```python
class TestDocLookup:
    def test_extract_doc_id_from_query(self):
        """myservice 29392 패턴 추출"""
        result = _extract_doc_id_from_query("myservice 29392 설명해줘")
        assert result == ("myservice", "29392")

    def test_detect_doc_lookup_intent(self):
        """LLM 기반 의도 감지"""
        history = [{"role": "assistant", "doc_ids": ["doc_001"]}]
        result = _detect_doc_lookup_intent(llm, "그 문서 더 자세히", history)
        assert result["is_doc_lookup"] == True

    def test_doc_lookup_fallback(self):
        """doc_id 검증 실패 시 fallback"""
        ...
```

**작업 내용:**
- [ ] Rule 기반 추출 테스트
- [ ] LLM 의도 감지 테스트
- [ ] Fallback 테스트
- [ ] 전체 플로우 통합 테스트

---

#### Task 4.2: E2E 테스트 시나리오

| 시나리오 | 입력 | 기대 결과 |
|----------|------|----------|
| 직접 문서 지정 | "myservice 29392 설명해줘" | doc_lookup → 해당 문서 조회 |
| 과거 문서 참조 | (history 있음) "그 문서 더 자세히" | doc_lookup → history doc_id 조회 |
| 일반 검색 | "SUPRA XP 센서 이상" | 기존 플로우 (MQ → 검색) |
| doc_id 없는 참조 | (history 없음) "그 문서 더 자세히" | fallback → 일반 검색 |

---

## 3. 구현 순서 및 우선순위 (시니어 피드백 반영)

### P0: 빠른 효과, 리스크 낮음 (먼저 완성)

```
1. API에 chat_history 필드 추가 (클라이언트 전달)
2. Rule 기반 doc_id 추출 ("myservice 29392" 패턴)
3. doc_lookup_node 구현 (fetch_doc_chunks 직접 조회)
4. 그래프 연결 (doc_lookup → expand → answer)
```

**P0 완료 시 효과:**
- "myservice 29392 설명해줘" 즉시 해결
- MQ 생성 LLM 호출 생략 → 토큰/latency 절약

### P1: 효과 크지만 설계 주의 필요

```
5. "그 문서에서..." 암묵 참조 감지 (LLM 판단)
6. history에서 doc_ids 추출 + 모호성 처리
7. entities 필드 활용 (장비/모듈 유지)
```

### P2: 운영/확장 (필요 시)

```
8. 서버 로딩 방식 (session_id 기반)
9. history 요약을 LLM 요약으로 전환 (조건부)
10. 측정 대시보드 (latency, 토큰, ES 호출)
```

### 실행 순서 (안전/효율 우선)

```
User Query
    │
    ▼
[1] chat 우회 (이미 구현됨)
    │
    ▼
[2] Rule: doc_id 패턴 체크 (LLM 0회, 빠름)
    ├─ 매칭 → doc_lookup 확정
    └─ 미매칭 ↓

[3] LLM: 암묵 참조 판단 (history 있을 때만)
    ├─ doc_lookup 필요 → history에서 doc_ids 추출
    └─ 불필요 ↓

[4] router (setup/ts/general)
    │
    ▼
[5] doc_lookup이면 → fetch_doc_chunks 직접 조회
    일반 검색이면 → MQ → retrieve → rerank
    │
    ▼
[6] answer
```

---

## 4. 체크리스트

### Phase 1: 기반 작업
- [ ] `ChatHistoryEntry` TypedDict 정의
- [ ] `AgentState`에 `chat_history`, `lookup_doc_ids` 필드 추가
- [ ] `AgentRequest`에 `chat_history` 필드 추가
- [ ] `AgentResponse`에 `summary`, `refs`, `ref_doc_ids` 필드 추가
- [ ] `_summarize_answer` 함수 구현 (truncate 방식)

### Phase 2: doc_lookup 로직 구현
- [ ] `DOC_ID_PATTERNS` 정의 (myservice, gcb, sop 등)
- [ ] `_extract_doc_id_from_query` 구현 (Rule 기반)
- [ ] `_format_history_for_prompt` 구현
- [ ] `_detect_doc_lookup_intent` 구현 (`_invoke_llm` 사용)
- [ ] `_get_doc_ids_from_history` 구현
- [ ] `doc_lookup_node` 구현 (`fetch_doc_chunks` 직접 사용)
- [ ] `route_node`에 doc_lookup 분기 추가

### Phase 3: 그래프 연결
- [ ] Route 타입에 "doc_lookup" 추가
- [ ] 그래프에 doc_lookup 노드 등록 (doc_fetcher 주입)
- [ ] router → doc_lookup 조건부 엣지 추가
- [ ] doc_lookup → expand/answer 또는 fallback 조건부 엣지 추가

### Phase 4: 테스트
- [ ] Rule 기반 doc_id 추출 테스트
- [ ] LLM 의도 감지 테스트
- [ ] doc_lookup_node fallback 테스트
- [ ] E2E 시나리오 검증
- [ ] 로깅 확인

---

## 5. 주의사항

1. **기존 플로우 영향 최소화**: doc_lookup은 새로운 분기로 추가, 기존 setup/ts/general 영향 없음
2. **Fallback 필수**: doc_id 검증 실패 시 항상 기존 검색(MQ)으로 fallback
3. **History 크기 제한**: 최근 5턴만 유지 (토큰 절약)
4. **테스트 우선**: 각 Task 완료 후 테스트 검증
5. **fetch_doc_chunks 사용**: retrieve_node 우회, doc_id로 직접 조회
6. **LLM 호출 방식**: `_invoke_llm(llm, system, user)` 사용 (llm.invoke 아님)
7. **필드명 주의**: AgentRequest는 `message` 필드 사용 (query 아님)

---

## 6. 구현 핵심 요약

| 항목 | 방식 |
|------|------|
| doc_id 직접 조회 | `fetch_doc_chunks()` 사용 |
| MQ 생략 | 그래프 엣지로 처리 (skip_mq 플래그 없음) |
| LLM 호출 | `_invoke_llm(llm, system, user)` |
| Fallback | 조건부 엣지로 general_mq로 분기 |
| History 저장 | 클라이언트 → 서버 전달 방식 |
