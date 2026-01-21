# Context Expansion 기능 설계 문서

## 1. 개요

### 1.1 목적
답변 생성 전에 선택된 문서의 관련 문서를 가져와 더 풍부한 컨텍스트로 답변을 생성한다.

### 1.2 배경
- 현재 RAG 시스템은 청크 단위로 문서를 검색하여 답변 생성
- 청크 단위 검색의 한계: 문맥 단절, 앞뒤 정보 누락
- 특히 정비 이력에서 순차적 절차나 연관 정보가 여러 페이지에 걸쳐 있는 경우 불완전한 답변 생성

### 1.3 해결 방안
검색된 문서의 doc_type에 따라 컨텍스트 확장 방식을 분기한다.
- gcb / myservice: 같은 doc_id의 모든 섹션(청크) 조회
- 그 외 doc_type: 선택 문서의 앞뒤 ±2 페이지 청크 조회
- 확장된 컨텍스트(상위 5개)로 답변을 생성하고, 답변 아래 문서 목록에는 확장 결과 전체 페이지를 표시한다

### 1.4 현재 반영 상태
**반영됨**
- doc_type 기반 확장 규칙 (gcb/myservice vs 기타)
- 확장 대상은 검색 결과 상위 5개로 제한
- expand_related 노드 추가 및 answer/judge에 확장 refs 사용
- ES helper (`fetch_doc_pages`, `fetch_doc_chunks`) 추가
- 확장 요약 로그를 백엔드/SSE로 기록
- 답변 이후 표시 문서 목록에 확장 대상(상위 5개) + 확장 페이지 이미지 모두 반영
- 대화 히스토리에 확장 페이지 목록 저장 (재로드 시 동일 출력)

**제거/보류**
- context_expansion_node (rerank 기반 확장, expansion_stats, fetch_surrounding_chunks) 제거: 요구사항과 불일치
- 상세 통계(expansion_stats) 로깅
- 확장 범위/최대 개수 설정을 config로 노출
- msearch 등 성능 최적화
- 테스트 코드 추가

---

## 2. 요구사항

### 2.1 기능 요구사항
| ID | 요구사항 | 우선순위 | 구현 상태 |
|----|----------|----------|----------|
| FR-01 | gcb/myservice가 아닌 문서에 대해 앞뒤 ±2 페이지 청크 조회 | 필수 | 적용 |
| FR-02 | gcb/myservice 문서는 같은 doc_id의 모든 섹션 조회 | 필수 | 적용 |
| FR-03 | 확장된 청크의 중복 제거 | 필수 | 적용 |
| FR-04 | 토큰 제한 고려한 청크 수 제한 | 필수 | 부분 적용 (답변 refs 길이 제한) |
| FR-05 | 확장 통계 로깅 (디버깅용) | 권장 | 부분 적용 (요약 로그) |
| FR-06 | 확장 대상은 검색 결과 상위 5개로 제한 | 필수 | 적용 |

### 2.2 비기능 요구사항
| ID | 요구사항 | 기준 | 구현 상태 |
|----|----------|------|----------|
| NFR-01 | ES 쿼리 응답 시간 | 500ms 이내 | 미측정 |
| NFR-02 | 최대 확장 청크 수 | 50개 이하 | 미적용 |
| NFR-03 | 확장 범위 설정 가능 | page_range 파라미터화 | 미적용 (상수 사용) |

---

## 3. 시스템 설계

### 3.1 현재 Agent 흐름

```
┌─────────┐   ┌────────┐   ┌─────────┐   ┌───────┐   ┌──────────┐
│  route  │ → │   mq   │ → │ st_gate │ → │ st_mq │ → │ retrieve │
└─────────┘   └────────┘   └─────────┘   └───────┘   └──────────┘
                                                           │
                                                           ▼
                                                    ┌───────────┐
                                                    │  ask_user │
                                                    └───────────┘
                                                           │
                                                           ▼
                                                  ┌──────────────┐
                                                  │ expand_related│
                                                  └──────────────┘
                                                           │
                                                           ▼
                                                    ┌──────────┐
                                                    │  answer  │
                                                    └──────────┘
                                                           │
                                                           ▼
                                                    ┌─────────┐
                                                    │  judge  │
                                                    └─────────┘
```

### 3.2 데이터 흐름

```
retrieve_node
    │
    ├── docs: List[RetrievalResult] (검색 결과)
    ├── ref_json: List[Dict] (리뷰용)
    │
    ▼
ask_user_after_retrieve_node
    │
    ├── (선택 문서가 있으면 docs 자체가 축소됨)
    │
    ▼
expand_related_docs_node
    │
    ├── docs: List[RetrievalResult] (확장된 문서로 교체)
    ├── display_docs: List[RetrievalResult] (UI 표시용: 확장 대상 상위 5개)
    ├── answer_ref_json: List[Dict] (답변/판정용 확장 참조)
    │
    ▼
answer_node / judge_node
    │
    └── answer_ref_json을 사용하여 답변/판정
```

---

## 4. 상세 설계

### 4.1 AgentState 확장

**파일**: `backend/llm_infrastructure/llm/langgraph_agent.py`

```python
class AgentState(TypedDict, total=False):
    # === 기존 필드 ===
    query: str
    route: Route
    setup_mq_list: List[str]
    ts_mq_list: List[str]
    general_mq_list: List[str]
    st_gate: Gate
    search_queries: List[str]
    available_devices: List[Dict[str, Any]]
    selected_devices: List[str]
    device_selection_skipped: bool
    docs: List[RetrievalResult]
    display_docs: List[RetrievalResult]
    ref_json: List[Dict[str, Any]]
    answer: str
    judge: Dict[str, Any]
    attempts: int
    max_attempts: int
    human_action: Optional[Dict[str, Any]]
    user_feedback: Optional[str]
    retrieval_confirmed: bool
    thread_id: Optional[str]

    # === 신규 필드 ===
    answer_ref_json: List[Dict[str, Any]]  # 답변/판정용 확장 참조
```

### 4.2 ES 검색 서비스 확장

**파일**: `backend/services/es_search_service.py`

#### 4.2.1 fetch_doc_pages

```python
def fetch_doc_pages(self, doc_id: str, pages: list[int]) -> list[RetrievalResult]:
    """doc_id + 특정 페이지 리스트로 청크 조회."""
```

#### 4.2.2 fetch_doc_chunks

```python
def fetch_doc_chunks(self, doc_id: str, max_chunks: int = 50) -> list[RetrievalResult]:
    """doc_id의 모든 청크 조회."""
```

### 4.3 expand_related_docs_node 구현

**파일**: `backend/llm_infrastructure/llm/langgraph_agent.py`

```python
def expand_related_docs_node(
    state: AgentState,
    *,
    page_fetcher,
    doc_fetcher,
    page_window: int = 2,
):
    docs = state.get("docs", [])
    expanded_docs = []

    for idx, doc in enumerate(docs):
        if idx >= 5:
            expanded_docs.append(doc)
            continue
        doc_type = normalize(doc.metadata.get("doc_type"))
        if doc_type in {"gcb", "myservice"}:
            related = doc_fetcher(doc.doc_id)
        else:
            page = doc.metadata.get("page")
            related = page_fetcher(doc.doc_id, pages_around(page, page_window))

        combined = combine_related_text(related)
        expanded_docs.append(as_answer_ref(doc, combined))

    display_docs = expanded_docs[:5]
    display_docs = merge_same_doc_id(display_docs, doc_type in {"gcb", "myservice"})
    return {
        "docs": expanded_docs,
        "display_docs": display_docs,
        "answer_ref_json": results_to_ref_json(display_docs, prefer_raw_text=True),
    }
```

핵심 포인트:
- doc_type은 소문자 정규화 후 비교
- gcb/myservice는 같은 doc_id의 전체 섹션을 결합
- 나머지는 page 기준 ±2
- 확장 결과는 `answer_ref_json`(답변/판정, 상위 5개)과 `display_docs`(UI 표시, 페이지 전체)로 분리 저장
- UI는 `expanded_pages` 기반으로 페이지 이미지를 모두 표시
- gcb/myservice는 같은 doc_id를 병합해 1건으로 표시
- 대화 히스토리 저장 시 doc_refs에 `pages`(expanded_pages)를 함께 저장

### 4.4 ask_user_after_retrieve_node 수정

**파일**: `backend/llm_infrastructure/llm/langgraph_agent.py`

Command의 goto를 `"answer"`에서 `"expand_related"`로 변경:

```python
return Command(
    goto="expand_related",
    update={...}
)
```

### 4.5 그래프 구성 수정

**파일**: `backend/services/agents/langgraph_rag_agent.py`

```python
builder.add_node("expand_related", expand_related_docs_node)
builder.add_edge("retrieve", "expand_related")
builder.add_edge("expand_related", "answer")
```

---

## 5. 설정 옵션

현재는 상수로 처리하며, 설정화는 보류한다.

| 항목 | 값 | 상태 |
|------|----|------|
| page_window | 2 | 상수 |
| doc_type 분기 | gcb/myservice | 상수 |
| answer ref 길이 | 1200 | 상수 |
| expand_top_k | 5 | 상수 |

---

## 6. 수정 파일 목록

| 파일 | 변경 내용 | 우선순위 |
|------|-----------|----------|
| `backend/services/es_search_service.py` | doc_id 기반 조회 helper 추가 | 1 |
| `backend/llm_infrastructure/llm/langgraph_agent.py` | answer_ref_json, expand_related_docs_node 추가 | 2 |
| `backend/services/agents/langgraph_rag_agent.py` | 그래프에 expand_related 노드 추가 | 3 |

---

## 7. 검증 계획

향후 테스트 추가:
- 단위 테스트: `fetch_doc_pages`, `fetch_doc_chunks`
- 통합 테스트: expand_related_docs_node 실행 경로
- E2E: 문서 선택 → 확장 적용 → 답변 생성 확인

---

## 8. 예상 효과

- 청크 단위 검색의 맥락 단절 문제 완화
- 정비 이력에서 연관 정보 누락 감소
- gcb/myservice 섹션형 문서의 일관된 컨텍스트 제공

---

## 9. 향후 개선 방향

1. msearch를 활용한 배치 조회 최적화
2. 확장 범위/최대 개수 설정화
3. 확장 컨텍스트 요약 추가
