# Task: Retrieval Pipeline Fixes, Group Scoring & Group-Level Retry

Status: In Progress
Owner: hskim
Branch: dev
Created: 2026-04-07

## Goal

APC position 검색 실패 문제를 해결하고, 검색 파이프라인 전반을 개선한다.
rerank 결과의 중복 등장 횟수(hit count)를 그룹 스코어링에 반영하여 관련성 판단을 강화한다.
답변 실패(unfaithful) 시 다음 relevant 그룹으로 재시도하는 group-level retry를 도입한다.

## Problems & Root Causes

### 1. APC 문서 검색 실패

**증상**: "supravvplus apc position 교체 방법" 검색 시 APC 문서 미검색 또는 검색되어도 답변 미생성.

**원인 3가지**:

| # | 원인 | 위치 |
|---|------|------|
| 1 | retrieve_node에서 디바이스 정규화 미적용 | `langgraph_agent.py` retrieve_node |
| 2 | ES 필터에서 공백 형태 디바이스명 누락 | `es_chunk_v3_search_service.py` |
| 3 | thinking 모델(gpt-oss:120b)에서 max_tokens=10이 너무 작아 빈 응답 | `langgraph_agent.py` _check_doc_relevance |

### 2. Router 하드코딩된 top-k 상수

**증상**: langgraph_rag_agent.py에서 top_k=50, retrieval_top_k=100으로 변경했지만 실제 21개만 검색됨.

**원인**: `api/routers/agent.py`의 `DEFAULT_RETRIEVAL_TOP_K=50`, `DEFAULT_FINAL_TOP_K=20`이 모든 agent 인스턴스 생성 시 오버라이드.

### 3. Docker Ollama 연결 실패

**증상**: Chat UI에서 "Connection refused" 에러.

**원인**: Docker 컨테이너 내부에서 `localhost:11435`는 컨테이너 자신을 가리킴. 호스트의 Ollama에 접근하려면 `host.docker.internal` 사용 필요.

### 4. 그룹 스코어링에 중복 정보 미반영

**증상**: 멀티쿼리로 여러 번 상위 랭크된 문서가 그룹 우선순위에 반영되지 않음.

**원인**: dedupe_by_doc_id가 중복을 단순 제거만 하고, 해당 문서가 rerank top-K에 몇 번 등장했는지(hit count) 정보를 버림.

## Changes

### Fix 1: retrieve_node 디바이스 정규화 추가

**File**: `backend/llm_infrastructure/llm/langgraph_agent.py` (retrieve_node 내부)

```python
# retrieve_node에서 검색 전 디바이스 정규화 추가
if selected_devices:
    _canonical_for_norm = selected_devices[0]
    queries = [_normalize_device_in_query(q, _canonical_for_norm) for q in queries]
```

기존에는 answer_node에서만 정규화했으나, 검색 쿼리 자체에도 적용해야 BM25가 올바른 토큰으로 매칭한다.

### Fix 2: ES 디바이스 필터에 공백 형태 추가

**File**: `backend/services/es_chunk_v3_search_service.py` (_normalize_device_names_to_v3)

```python
# 추가된 형태
space_form = name.replace("_", " ")     # "SUPRA VPLUS"
space_upper = space_form.upper()         # "SUPRA VPLUS" (대문자)
```

ES 인덱스에 `SUPRA_VPLUS`, `SUPRA Vplus`, `SUPRA VPLUS` 등 다양한 형태로 저장되어 있으므로, 필터에도 공백 형태를 포함해야 한다.

### Fix 3: thinking 모델 호환 max_tokens 증가

**File**: `backend/llm_infrastructure/llm/langgraph_agent.py` (_check_doc_relevance)

```python
# Before: max_tokens=10 (thinking 모델에서 reasoning에 토큰 소진 → content 빈 응답)
# After:
raw = _invoke_llm(llm, system, user, max_tokens=256, temperature=TEMP_CLASSIFICATION)
```

gpt-oss:120b(DeepSeek 기반)는 `thinking` 필드에서 reasoning 토큰을 소비한 후 `content`에 답변을 생성한다. max_tokens=10이면 reasoning만으로 소진되어 빈 응답 반환.

### Fix 4: Router DEFAULT 상수 업데이트

**File**: `backend/api/routers/agent.py` (line 45-46)

```python
# Before
DEFAULT_RETRIEVAL_TOP_K = 50
DEFAULT_FINAL_TOP_K = 20

# After
DEFAULT_RETRIEVAL_TOP_K = 100
DEFAULT_FINAL_TOP_K = 50
```

이 상수가 router 내 10개소의 LangGraphRAGAgent 인스턴스 생성에 사용됨.

### Fix 5: Docker Ollama 접속 URL

**File**: `.env.dev`

```bash
# Docker 컨테이너에서 호스트의 Ollama에 접근
OLLAMA_BASE_URL=http://host.docker.internal:11435
```

### Enhancement 1: rerank_hit_count 기반 그룹 스코어링

**File**: `backend/llm_infrastructure/llm/langgraph_agent.py`

**개념**: 멀티쿼리(EN/KO)가 동일 문서의 여러 페이지를 상위 랭크시키면, 이는 강한 관련성 신호다. 이 정보를 그룹 우선순위에 반영한다.

**변경 1 — dedupe 단계에서 hit count 보존** (retrieve_node):

```python
# dedupe 전에 각 base_doc_id별 등장 횟수 집계
hit_counts = Counter(_base_doc_id(doc.doc_id) for doc in docs)

# dedupe 시 대표 doc의 metadata에 주입
meta["rerank_hit_count"] = hit_counts.get(base, 1)
```

**변경 2 — ref_json에 hit count 전달** (results_to_ref_json):

```python
_rhc = d.metadata.get("rerank_hit_count")
if _rhc is not None:
    metadata["rerank_hit_count"] = int(_rhc)
```

**변경 3 — 그룹 스코어에 hit count 가산** (answer_node _group_query_score):

```python
# 기존: score = 키워드매칭(doc_id) + 키워드매칭(content)
# 변경: score += max(rerank_hit_count across refs in group)
max_hit = max(
    (int((r.get("metadata") or {}).get("rerank_hit_count", 1)) for r in grefs),
    default=1,
)
score += max_hit
```

**효과**: APC valve 문서가 rerank 50개 중 13페이지 등장 → hit_count=13 → 그룹 우선순위 대폭 상승.

### Enhancement 2: Group-Level Retry (답변 실패 시 다음 그룹 재시도)

**Files**: `langgraph_agent.py`, `langgraph_rag_agent.py`

**문제**: answer_node에서 그룹1로 답변 생성 → judge unfaithful → 재검색(retry_expand/retry_bump).
이미 relevance check 통과한 그룹2, 그룹3이 있는데 버리고 비용이 큰 재검색부터 시작함.

**변경 — AgentState 확장**:

```python
_relevant_groups: List[Tuple[str, List[Dict]]]  # relevance check 통과한 그룹 목록
_current_group_idx: int                          # 현재 사용 중인 그룹 인덱스
```

**변경 — answer_node 그룹 선택 로직**:

```python
# 첫 실행: relevance check 후 모든 통과 그룹 저장
relevant_groups = [(key, refs) for key, refs in groups if _check_doc_relevance(...)]
state["_relevant_groups"] = relevant_groups
state["_current_group_idx"] = 0

# group-level retry: relevance check 건너뛰고 다음 그룹 사용
if prev_relevant and current_group_idx < len(prev_relevant):
    ref_items = prev_relevant[current_group_idx][1]  # skip relevance check
```

**변경 — should_retry에서 그룹 우선 라우팅**:

```python
# 남은 그룹이 있으면 재검색 대신 다음 그룹으로
if current_idx + 1 < len(relevant_groups):
    return "retry_next_group"
# 그룹 소진 후에야 기존 retry 전략
```

**변경 — retry_next_group_node 추가**:

```python
def retry_next_group_node(state):
    return {"_current_group_idx": current_idx + 1}
```

**변경 — LangGraph 빌더 (langgraph_rag_agent.py)**:

```python
builder.add_node("retry_next_group", retry_next_group_node)
builder.add_edge("retry_next_group", "answer")  # 재검색 없이 answer로 복귀
builder.add_conditional_edges("judge", should_retry, {
    "retry_next_group": "retry_next_group",  # 새로 추가
    "retry_expand": "retry_expand",
    ...
})
```

**변경 후 retry 흐름**:

```
answer(그룹1) → judge → unfaithful
  → 그룹2 남음 → retry_next_group → answer(그룹2) → judge → unfaithful
    → 그룹3 남음 → retry_next_group → answer(그룹3) → judge → faithful
      → 그룹3의 ref로 보완 → done
    → 그룹 소진 → 기존 retry 전략 (retry_expand → retry_bump → retry_mq)
```

### Enhancement 3: answer_ref_json 항상 설정 (supplement 범위 수정)

**File**: `langgraph_agent.py` (answer_node return)

**문제**: fallback 선택 시 `answer_ref_json`을 설정하지 않아서, judge의 supplement가
`ref_json`(expand_related 전체 36개 문서, 10,403자)을 사용함.
답변에 사용한 1개 그룹의 ref가 아닌 전체 문서로 보완 답변을 재생성 → 불필요하게 느림.

**변경**:

```python
# Before: fallback이면 answer_ref_json 미설정
if route == "setup" and not is_fallback_selection:
    _result["answer_ref_json"] = ref_items

# After: 항상 설정 — 답변에 사용한 그룹의 ref만 supplement에 전달
if route == "setup":
    _result["answer_ref_json"] = ref_items
```

### Enhancement 4: 파이프라인 단계별 로깅

**File**: `backend/llm_infrastructure/llm/langgraph_agent.py`

파이프라인 각 단계에서 실제 문서 갯수를 추적하는 로그 체계:

```
retrieve_node: [1/4 rerank]         100 → 50 docs
retrieve_node: [2/4 group+dedupe]    50 → 18 docs (unique doc_ids=18), hit_counts=[...]
retrieve_node: [3/4 section_expand]  18 → 36 docs (+18)
retrieve_node: [4/4 done]           returning 36 docs (all_docs_for_regen: 100)
```

**Workflow trace (UI) 표시**:

```
[retrieve] pipeline: 100 retrieved → 50 reranked → 18 deduped → 36 expanded
[retrieve] rerank_hit_counts: apc_valve_eng=13, cln_pm_apc_valve=6, ...
[retrieve] top docs: #1 hits=13 p=12 s=0.0562 global_sop_supra_vplus_rep_pm_apc_valve_eng
```

**answer_node 그룹 로그**:

```
answer_node: setup doc_section_groups=12 groups=[apc_valve_eng::Work Procedure(hits=13,refs=3), ...]
```

## Pipeline Flow (변경 후)

```
[retrieve_node]
  1. ES 검색 (retrieval_top_k=100, 멀티쿼리 확장)
  2. Rerank (final_top_k=50)
  3. Score threshold 필터
  4. Group by base_doc_id + hit count 집계
  5. Dedupe (doc_id당 최고점 1개, hit_count 주입)
  6. Doc-type diversity quota
  7. Section expansion (인접 청크 추가)

[expand_related_docs_node]
  8. 관련 doc_type 확장 (expand_top_k=10)

[answer_node]
  9. Group by (doc_id, section_chapter)
 10. Group scoring: 키워드매칭 + rerank_hit_count
 11. 첫 실행: Relevance check (MAX_SETUP_DOC_TRIES=5) → 모든 relevant 그룹 저장
     Group retry: 저장된 다음 그룹 사용 (relevance check 생략)
 12. 선택된 그룹의 ref로 답변 생성

[judge_node]
 13. 선택된 그룹의 ref로 supplement (보완)
 14. faithful 판정
   → true: done
   → false + 남은 그룹 있음: retry_next_group → answer (다음 그룹)
   → false + 그룹 소진: 기존 retry (retry_expand → retry_bump → retry_mq)
```

## Commits

| Hash | Message |
|------|---------|
| `6a8f589` | feat(retrieval): tune top-k 100→50 and fix BM25 device tokenization |
| `7b747c4` | fix(retrieve): normalize device name in search queries + fix relevance check |
| `0b436fc` | fix(api): update DEFAULT_RETRIEVAL_TOP_K=100 and DEFAULT_FINAL_TOP_K=50 |
| (pending) | feat(retrieve): rerank hit count group scoring + pipeline logging |
| (pending) | feat(answer): group-level retry + answer_ref_json always set |

## Files Modified

| File | Changes |
|------|---------|
| `backend/llm_infrastructure/llm/langgraph_agent.py` | retrieve_node 디바이스 정규화, hit count 집계/주입, 그룹 스코어링, 파이프라인 로깅, group-level retry (`_relevant_groups`, `_current_group_idx`), `retry_next_group_node`, `should_retry` 그룹 우선, `answer_ref_json` 항상 설정 |
| `backend/services/agents/langgraph_rag_agent.py` | top_k=50, retrieval_top_k=100 기본값, `retry_next_group` 노드/엣지 추가 |
| `backend/services/es_chunk_v3_search_service.py` | 공백 형태 디바이스명 추가 |
| `backend/api/routers/agent.py` | DEFAULT_RETRIEVAL_TOP_K=100, DEFAULT_FINAL_TOP_K=50 |
| `.env.dev` | OLLAMA_BASE_URL=host.docker.internal |

## Verification

```
# APC position 검색 테스트 (Chat UI)
Query: "supravvplus apc position 교체 방법"
Result: APC valve 문서 #1 (hits=13, score=0.0562), 답변 정상 생성

# 파이프라인 로그 확인
[1/4 rerank] 100 → 50
[2/4 group+dedupe] 50 → 18 (hit_counts: apc_valve=13, cln_apc=6, ...)
[3/4 section_expand] 18 → 36
[4/4 done] returning 36 docs
```

## Risks

- rerank_hit_count 가산 가중치 조정 필요할 수 있음 (현재 1:1 가산)
  - 키워드 매칭 점수와 hit count 스케일이 다를 수 있음
  - 향후 가중치 파라미터화 고려
- thinking 모델에서 max_tokens=256이 모든 케이스에 충분한지 모니터링 필요
- section expansion이 dedupe 후에 실행되므로, 같은 doc_id의 다른 섹션이 expansion으로 다시 들어올 수 있음
- group-level retry 시 `_relevant_groups`가 state에 누적 저장됨
  - 재검색(retry_bump) 후에는 새로운 relevance check가 필요 → `_relevant_groups`가 초기화되어야 함
  - 현재 answer_node에서 `prev_relevant`가 없으면 새로 relevance check하므로 자연스럽게 처리됨

## Design Decisions

### 왜 expand를 그룹 선택 후로 옮기지 않았는가

expand_related는 ES에서 관련 청크를 미리 가져오는 것이고, LLM에는 들어가지 않는다.
실제 LLM에 들어가는 ref는 answer_node에서 선택된 그룹의 ref만이므로, 현재 구조(전체 expand → 그룹 선택)로 충분하다.

### supplement(보완)의 범위

supplement는 faithful=true가 된 그룹의 `answer_ref_json`만 사용한다.
목적: 답변에 사용한 문서(A) 대비 누락된 내용을 보완하는 것이지, 모든 검색 문서로 새 답변을 만드는 것이 아님.

## Change Log

- 2026-04-07: APC 검색 실패 3중 원인 분석 및 수정 (디바이스 정규화, ES 필터, max_tokens)
- 2026-04-07: Router DEFAULT 상수 업데이트 (100/50)
- 2026-04-07: Docker Ollama 연결 수정 (host.docker.internal)
- 2026-04-07: rerank_hit_count 그룹 스코어링 도입
- 2026-04-07: 파이프라인 단계별 로깅 체계 구축
- 2026-04-07: group-level retry 도입 (답변 실패 시 다음 relevant 그룹 재시도)
- 2026-04-07: answer_ref_json 항상 설정 (supplement가 답변에 사용한 그룹 ref만 사용)
