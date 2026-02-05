# Batch Answer Generation 기능 명세

## 개요

Retrieval Test에 등록된 질문들에 대해 **검색 결과를 그대로 사용해 답변만 생성**하는 기능.  
검색은 Retrieval Test에서 이미 수행된 것으로 간주하고, Batch Answer는 **Answer-only 모드**로 동작한다.

### 목적
- 검색 파라미터 변경에 따른 답변 품질 비교
- 답변 트렌드 분석 (이전 실행 결과와 비교)
- 수동 평가를 통한 품질 관리

---

## 핵심 원칙

### 검색 로직 동일성

**Retrieval Test는 Chat과 동일한 파이프라인을 실행한다.**

```
translate → mq → retrieve → expand
```

| 항목 | Chat | Retrieval Test |
|------|------|----------------|
| 파이프라인 | ✅ 동일 | ✅ 동일 |
| 파라미터 | 환경변수 (고정) | **UI에서 override 가능** |

**동작 방식**
- Retrieval Test는 Chat과 동일한 `LangGraphRAGAgent` 파이프라인 호출
- 단, `search()` 호출 시 override 파라미터 전달 가능:
  - `dense_weight`, `sparse_weight`
  - `use_rrf`, `rrf_k`
  - `rerank`, `rerank_top_k`
  - `top_k`
- MQ(Multi-Query)는 Chat과 동일하게 항상 적용 (별도 ON/OFF 없음)
- Chat은 환경변수 설정 그대로 사용
- Retrieval Test만 실험용 파라미터 조절 가능

### Answer-only 모드

Batch Answer Generation은 **검색 단계를 수행하지 않고**, Retrieval Test에서 저장된 문서 본문을 그대로 사용해 답변만 생성한다.

**동작 원칙**
- Retrieval Test 결과의 `search_results[].content`(본문) + 메타데이터(rank, doc_id, score, page)를 LLM 컨텍스트로 사용
- prompt/입력 구조는 Chat과 동일한 형식을 유지
- 검색/재검색/MQ/route/expand 노드는 실행하지 않음

### 검색 파라미터 반영

검색 파라미터는 **Retrieval Test 실행 단계**에서만 적용된다.
Batch Answer는 **검색 결과를 재사용**하므로, Batch 단계에서 검색 파라미터를 변경하지 않는다.

**필수 조건**
- Retrieval Test 결과에 **본문(content/raw_text)**을 함께 저장해야 함 (TICKET-000-2에서 구현)
- Batch Answer는 저장된 본문을 그대로 LLM에 전달

**content 저장 범위 (Retrieval Test)**
```
search_results[].content  - 청크 본문 전체 (LLM 컨텍스트용)
search_results[].snippet  - 짧은 스니펫 (UI 표시용)
```
- `content`는 ES 검색 결과의 `raw_text` 또는 `content` 필드에서 가져옴
- 저장 시 길이 제한: 최대 10,000자 (truncate)

---

## UI 설계

### 메인 화면

```
┌─────────────────────────────────────────────────────────────────────┐
│  Batch Answer Generation                                             │
├─────────────────────────────────────────────────────────────────────┤
│  [Retrieval Test 실행 선택]                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ ▼ 2024-02-03 15:30 (Dense:0.7, Rerank:ON, Size:20) - 50개   │    │
│  │   2024-02-03 14:00 (Dense:0.5, Rerank:ON, Size:10) - 50개   │    │
│  │   2024-02-02 18:00 (Dense:0.7, Rerank:OFF, Size:20) - 50개  │    │
│  └─────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│  [선택된 검색 파라미터] (source_config, 읽기 전용)                   │
│  Dense: 0.7  Sparse: 0.3  Rerank: ✓  Size: 20                       │
│                                                                      │
│  [답변 생성 실행]  진행: 12/50 ████████░░░░░░░░░░░░ 24%              │
├─────────────────────────────────────────────────────────────────────┤
│  [결과 테이블]                                                       │
│  ┌──────┬────────────────────────────┬────────┬────────┬─────────┐  │
│  │ #    │ 질문                       │ Hit@3  │ 답변   │ 평가    │  │
│  ├──────┼────────────────────────────┼────────┼────────┼─────────┤  │
│  │ 1    │ SUPRA XP PM 절차 알려줘    │ ✓ HIT  │ [보기] │ ⭐⭐⭐⭐☆│  │
│  │ 2    │ Ecolite 에러 E-001 원인    │ ✗ MISS │ [보기] │ ⭐⭐☆☆☆│  │
│  │ 3    │ Chuck 온도 관계            │ ✓ HIT  │ [보기] │ 미평가  │  │
│  └──────┴────────────────────────────┴────────┴────────┴─────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│  [집계]  Hit@3: 72% (36/50) | 평균 평점: 3.8/5 (평가됨: 40/50)      │
└─────────────────────────────────────────────────────────────────────┘
```

### [보기] 클릭 시 팝업 (답변 상세)

```
┌──────────────────────────────────────────────────────────────┐
│  질문: SUPRA XP PM 절차 알려줘                          [X]  │
├──────────────────────────────────────────────────────────────┤
│  [검색 문서]  |  [생성된 답변]  |  [Reasoning]               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  **SUPRA XP PM 절차**                                        │
│                                                              │
│  1. Chamber 점검                                             │
│     - CIP Chamber 청소 (ref: SOP-001 p.23)                   │
│     - Prism Source 확인 (ref: SOP-002 p.15)                  │
│                                                              │
│  2. Bottom Structure 점검                                    │
│     - ...                                                    │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  평가: ⭐⭐⭐⭐☆                              [평가 저장]     │
└──────────────────────────────────────────────────────────────┘
```

**탭 구성**
- **검색 문서**: 검색된 문서 목록 (rank, doc_id, score, 이미지)
- **생성된 답변**: LLM이 생성한 최종 답변
- **Reasoning**: LLM의 추론 과정 (**reasoning이 있는 경우에만 탭 표시**)

---

## 백엔드 API

### 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/batch-answer/runs` | 새 배치 실행 생성 (source_run_id 지정) |
| GET | `/api/batch-answer/runs` | 실행 이력 목록 조회 |
| GET | `/api/batch-answer/runs/{run_id}` | 실행 상세 + 결과 목록 |
| POST | `/api/batch-answer/runs/{run_id}/execute-next` | 다음 질문 실행 |
| DELETE | `/api/batch-answer/runs/{run_id}` | 실행 삭제 |
| PUT | `/api/batch-answer/results/{result_id}/rating` | 평가 저장 |

### API 상세

#### POST `/api/batch-answer/runs`
새로운 배치 실행 생성

**Request**
```json
{
  "name": "Dense0.7+Rerank 테스트",  // 선택
  "source_run_id": "retrieval_run_abc123",  // Retrieval Test 실행 결과 참조
  "question_ids": ["q1", "q2", "q3"]        // 선택: 특정 질문만 실행
}
```

**Response**
```json
{
  "run_id": "run_abc123",
  "source_run_id": "retrieval_run_abc123",
  "source_config": {
    "dense_weight": 0.7,
    "sparse_weight": 0.3,
    "rerank": true,
    "top_k": 20
  },
  "status": "pending",
  "question_count": 50,
  "completed_count": 0,
  "created_at": "2024-02-03T15:30:00Z"
}
```

#### POST `/api/batch-answer/runs/{run_id}/execute-next`
다음 미완료 질문에 대해 답변 생성 (Answer-only 모드: 저장된 검색 결과 사용)

**Response**
```json
{
  "result_id": "result_xyz789",
  "question_id": "q1",
  "question": "SUPRA XP PM 절차 알려줘",
  "answer": "**SUPRA XP PM 절차**\n\n1. Chamber 점검...",
  "reasoning": "The user asks about...",  // null일 수 있음
  "search_results": [
    {
      "rank": 1,
      "doc_id": "SOP-001",
      "score": 0.95,
      "title": "SUPRA XP PM SOP",
      "snippet": "Chamber 점검 절차...",
      "content": "전체 본문...",
      "page": 23,
      "expanded_pages": [22, 23, 24]
    },
    {
      "rank": 2,
      "doc_id": "SOP-002",
      "score": 0.87,
      "title": "Prism Source 점검",
      "snippet": "Prism Source 확인...",
      "content": "전체 본문...",
      "page": 15,
      "expanded_pages": [14, 15, 16]
    }
  ],
  "metrics": {
    "hit_at_1": true,
    "hit_at_3": true,
    "hit_at_5": true,
    "first_relevant_rank": 1
  },
  "progress": {
    "completed": 13,
    "total": 50
  }
}
```

#### PUT `/api/batch-answer/results/{result_id}/rating`
답변 평가 저장

**Request**
```json
{
  "rating": 4,
  "comment": "답변은 좋으나 일부 절차 누락"  // 선택
}
```

---

## 데이터 저장 (Elasticsearch)

### 인덱스 1: `batch_answer_runs_{env}_v1`
실행 이력 저장

| 필드 | 타입 | 설명 |
|------|------|------|
| run_id | keyword | 실행 ID (PK) |
| name | keyword | 실행 이름 (선택) |
| status | keyword | pending / running / completed / failed |
| source_run_id | keyword | Retrieval Test 실행 ID (FK) |
| source_config | object | 검색 파라미터 스냅샷 (Retrieval Test에서 복사) |
| source_config.dense_weight | float | Dense 가중치 |
| source_config.sparse_weight | float | Sparse 가중치 |
| source_config.use_rrf | boolean | RRF 사용 여부 |
| source_config.rrf_k | integer | RRF k 상수 |
| source_config.rerank | boolean | Rerank 사용 여부 |
| source_config.rerank_top_k | integer | Rerank 결과 수 |
| source_config.top_k | integer | 검색 결과 수 |
| question_count | integer | 총 질문 수 |
| completed_count | integer | 완료된 답변 수 |
| metrics | object | 집계 메트릭 |
| metrics.hit_at_1 | float | Hit@1 비율 |
| metrics.hit_at_3 | float | Hit@3 비율 |
| metrics.hit_at_5 | float | Hit@5 비율 |
| metrics.mrr | float | Mean Reciprocal Rank |
| metrics.avg_rating | float | 평균 평점 |
| metrics.rated_count | integer | 평가된 답변 수 |
| created_at | date | 생성 시각 |
| completed_at | date | 완료 시각 |

### 인덱스 2: `batch_answer_results_{env}_v1`
개별 답변 결과 저장

| 필드 | 타입 | 설명 |
|------|------|------|
| result_id | keyword | 결과 ID (PK) |
| run_id | keyword | 실행 ID (FK) |
| question_id | keyword | 질문 ID |
| question | text | 질문 내용 |
| ground_truth_doc_ids | keyword[] | 정답 문서 ID 목록 |
| answer | text | 생성된 답변 |
| reasoning | text | LLM 추론 과정 |
| search_results | nested | 검색 결과 목록 |
| search_results.content | text | 본문 (Retrieval Test에서 저장된 원문) |
| search_results.rank | integer | 순위 |
| search_results.doc_id | keyword | 문서 ID |
| search_results.score | float | 검색 점수 |
| search_results.title | keyword | 문서 제목 |
| search_results.snippet | text | 문서 스니펫 |
| search_results.page | integer | 페이지 번호 |
| search_results.expanded_pages | integer[] | 확장된 페이지 목록 |
| metrics | object | 개별 메트릭 |
| metrics.hit_at_1 | boolean | Hit@1 여부 |
| metrics.hit_at_3 | boolean | Hit@3 여부 |
| metrics.hit_at_5 | boolean | Hit@5 여부 |
| metrics.first_relevant_rank | integer | 첫 관련 문서 순위 |
| metrics.reciprocal_rank | float | Reciprocal Rank |
| rating | integer | 수동 평가 (1-5, null=미평가) |
| rating_comment | text | 평가 코멘트 |
| rated_at | date | 평가 시각 |
| executed_at | date | 실행 시각 |

---

## 실행 흐름

### 프론트엔드

```
1. Retrieval Test 실행 선택 (source_run_id)
2. source_config 표시 (읽기 전용)
3. [답변 생성] 클릭
4. POST /api/batch-answer/runs (source_run_id 포함) → run_id 획득
5. Loop (질문 수만큼 반복):
   a. POST /api/batch-answer/runs/{run_id}/execute-next
   b. 응답 수신 → 테이블에 결과 추가
   c. 진행률 업데이트
   d. 다음 질문 없으면 종료
6. 완료 상태 표시
```

### 백엔드 (execute-next 내부)

```python
async def execute_next(run_id: str):
    # 1. 실행 정보 및 config 조회
    run = get_run(run_id)
    source_run_id = run.source_run_id

    # 2. 다음 미완료 질문 조회
    question = get_next_pending_question(run_id)
    if not question:
        return {"status": "completed"}

    # 3. Retrieval Test 결과에서 문서 본문 로드
    search_results = get_retrieval_results(source_run_id, question.id)
    # search_results에는 content/raw_text 포함

    # 4. Answer-only 실행 (검색 스킵, 문서 본문만 사용)
    answer, reasoning = generate_answer_only(
        question=question.text,
        docs=search_results,  # content 포함
        prompt_spec=prompt_spec,  # Chat과 동일 포맷
    )

    # 5. Hit@K 메트릭 계산 (Retrieval Test 결과 기준)
    metrics = calculate_hit_metrics(
        retrieved_doc_ids=[d.doc_id for d in search_results],
        ground_truth_doc_ids=question.ground_truth_doc_ids
    )

    # 6. ES에 결과 저장
    save_result(run_id, question, answer, reasoning, search_results, metrics)

    # 7. 실행 이력 업데이트 (completed_count++)
    update_run_progress(run_id)

    return {
        "result_id": result_id,
        "question": question.text,
        "answer": answer,
        "metrics": metrics,
        ...
    }
```

---

## 향후 확장 가능

1. **실행 비교 뷰**: 같은 질문에 대해 여러 실행 결과 나란히 비교
2. **자동 평가**: LLM-as-Judge로 자동 평가 추가
3. **내보내기**: CSV/Excel 다운로드
4. **질문 필터**: 카테고리/난이도별 필터링 실행

---

## 파일 구조 (예상)

```
backend/
├── api/routers/
│   └── batch_answer.py           # API 엔드포인트
├── services/
│   └── batch_answer_service.py   # 비즈니스 로직
└── llm_infrastructure/elasticsearch/
    └── mappings.py               # ES 매핑 추가

frontend/src/features/
└── batch-answer/
    ├── pages/
    │   └── batch-answer-page.tsx
    ├── components/
    │   ├── run-history-select.tsx
    │   ├── config-panel.tsx
    │   ├── results-table.tsx
    │   └── answer-detail-modal.tsx
    ├── hooks/
    │   └── use-batch-answer.ts
    ├── api.ts
    └── types.ts
```

---

## 개발 티켓

### Phase 0: 선행 작업 (Retrieval Test 수정)

#### TICKET-000-1: Retrieval Test에서 Chat 파이프라인 호출
**우선순위**: P0 (필수, 선행)
**예상 작업량**: M

**배경**
- Retrieval Test가 Chat과 동일한 파이프라인(translate → mq → retrieve → expand)을 사용해야 함
- 단, search 파라미터는 override 가능해야 함

**작업 내용**
- [ ] Retrieval Test API에서 `LangGraphRAGAgent` 파이프라인 호출 (답변 생성 제외)
- [ ] search override 파라미터 전달 구조 구현
  - `dense_weight`, `sparse_weight`
  - `use_rrf`, `rrf_k`
  - `rerank`, `rerank_top_k`, `top_k`
- [ ] MQ는 Chat과 동일하게 항상 적용 (별도 ON/OFF 없음)
- [ ] 기존 `/api/search` 엔드포인트 수정 또는 새 엔드포인트 추가

**파일**
- `backend/api/routers/search.py` 또는 `retrieval_test.py` (수정)
- `backend/services/agents/langgraph_rag_agent.py` (수정: 검색만 실행하는 모드 추가)

---

#### TICKET-000-2: Retrieval Test 결과에 본문(content) 저장
**우선순위**: P0 (필수, 선행)
**예상 작업량**: M

**배경**
- 현재 Retrieval Test의 `SearchResult`에는 `snippet`만 있고 **본문(content)이 없음**
- Answer-only 모드를 위해 검색 결과에 문서 본문을 함께 저장해야 함

**작업 내용**
- [ ] 백엔드: 검색 API 응답에 `content` 필드 추가 (ES의 `raw_text` 또는 `content` 필드)
- [ ] 프론트엔드: `SearchResult` 타입에 `content` 필드 추가
- [ ] Retrieval Test 결과 저장 시 본문 포함
- [ ] 본문 길이 제한: 최대 10,000자 (truncate)

**파일**
- `backend/api/routers/retrieval_test.py` (또는 search.py)
- `frontend/src/features/retrieval-test/types/index.ts`

**타입 변경**
```typescript
export interface SearchResult {
  // 기존 필드...
  content: string;     // 추가: 문서 본문 (LLM 컨텍스트용, 필수)
}
```

**저장 구조**
- `content`: 청크 본문 전체 (LLM 컨텍스트로 사용)
- `snippet`: 짧은 스니펫 (UI 표시용, 기존)

---

### Phase 1: 백엔드 기반 (ES + Service)

#### TICKET-001: ES 인덱스 매핑 정의
**우선순위**: P0 (필수)
**예상 작업량**: S

**작업 내용**
- [ ] `batch_answer_runs_{env}_v1` 매핑 정의
- [ ] `batch_answer_results_{env}_v1` 매핑 정의
- [ ] `mappings.py`에 함수 추가
- [ ] 인덱스 생성 스크립트/마이그레이션

**파일**
- `backend/llm_infrastructure/elasticsearch/mappings.py`

---

#### TICKET-002: BatchAnswerService 구현
**우선순위**: P0 (필수)
**의존성**: TICKET-001
**예상 작업량**: M

**작업 내용**
- [ ] `BatchAnswerService` 클래스 생성
- [ ] `create_run()` - 새 실행 생성
- [ ] `get_run()` - 실행 정보 조회
- [ ] `list_runs()` - 실행 이력 목록
- [ ] `delete_run()` - 실행 삭제
- [ ] `get_next_pending_question()` - 다음 미완료 질문 조회
- [ ] `save_result()` - 개별 결과 저장
- [ ] `update_run_metrics()` - 집계 메트릭 업데이트
- [ ] `save_rating()` - 평가 저장

**파일**
- `backend/services/batch_answer_service.py`

---

### Phase 2: 백엔드 API

#### TICKET-003: Batch Answer API 엔드포인트
**우선순위**: P0 (필수)
**의존성**: TICKET-001, TICKET-002
**예상 작업량**: M

**작업 내용**
- [ ] `POST /api/batch-answer/runs` - 새 실행 생성
- [ ] `GET /api/batch-answer/runs` - 실행 이력 목록
- [ ] `GET /api/batch-answer/runs/{run_id}` - 실행 상세
- [ ] `DELETE /api/batch-answer/runs/{run_id}` - 실행 삭제
- [ ] `POST /api/batch-answer/runs/{run_id}/execute-next` - 다음 질문 실행
- [ ] `PUT /api/batch-answer/results/{result_id}/rating` - 평가 저장
- [ ] Request/Response 스키마 정의 (Pydantic)

**파일**
- `backend/api/routers/batch_answer.py`
- `backend/api/main.py` (라우터 등록)

---

#### TICKET-004: Answer-only 답변 생성 로직
**우선순위**: P0 (필수)
**의존성**: TICKET-003
**예상 작업량**: M

**작업 내용**
- [ ] `generate_answer_only()` 함수 구현
  - Retrieval Test 결과의 `content`를 LLM 컨텍스트로 사용
  - Chat의 answer 프롬프트와 동일한 형식 유지
- [ ] source_run_id에서 검색 결과 + 본문 로드
- [ ] Hit@K 메트릭 계산 (Retrieval Test 결과 기준)
- [ ] 결과 ES 저장
- [ ] 진행률 업데이트

**파일**
- `backend/api/routers/batch_answer.py`
- `backend/services/batch_answer_service.py`

---

### Phase 3: 프론트엔드 기반

#### TICKET-005: 타입 정의 및 API 클라이언트
**우선순위**: P0 (필수)
**예상 작업량**: S

**작업 내용**
- [ ] TypeScript 타입 정의 (Run, Result, Metrics)
- [ ] API 함수 구현
  - `createRun(sourceRunId)`
  - `listRuns()`
  - `getRun()`
  - `deleteRun()`
  - `executeNext()`
  - `saveRating()`

**파일**
- `frontend/src/features/batch-answer/types.ts`
- `frontend/src/features/batch-answer/api.ts`

---

#### TICKET-006: Retrieval Test 실행 선택 컴포넌트
**우선순위**: P1
**의존성**: TICKET-005
**예상 작업량**: S

**작업 내용**
- [ ] Retrieval Test 실행 이력 목록 조회
- [ ] 드롭다운으로 source_run 선택
- [ ] 선택된 실행의 파라미터/질문 수 표시

**파일**
- `frontend/src/features/batch-answer/components/source-run-select.tsx`

---

#### TICKET-007: 결과 테이블 컴포넌트
**우선순위**: P1
**의존성**: TICKET-005
**예상 작업량**: M

**작업 내용**
- [ ] 질문 목록 테이블
- [ ] Hit@3 표시 (HIT/MISS 뱃지)
- [ ] [보기] 버튼 → 팝업 열기
- [ ] 평가 별점 표시 (미평가 시 "미평가")
- [ ] 진행 중 상태 표시 (로딩)

**파일**
- `frontend/src/features/batch-answer/components/results-table.tsx`

---

#### TICKET-008: 답변 상세 팝업 (Modal)
**우선순위**: P1
**의존성**: TICKET-007
**예상 작업량**: M

**작업 내용**
- [ ] 팝업 레이아웃
- [ ] 탭: 검색 문서 / 생성된 답변 / Reasoning
- [ ] 검색 문서 탭: 문서 목록 + 이미지 (기존 컴포넌트 재사용)
- [ ] 답변 탭: 마크다운 렌더링
- [ ] Reasoning 탭: 텍스트 표시 (**reasoning이 있는 경우에만 탭 표시**)
- [ ] 평가 별점 입력 + 저장 버튼

**파일**
- `frontend/src/features/batch-answer/components/answer-detail-modal.tsx`

---

#### TICKET-009: 메인 페이지 통합
**우선순위**: P1
**의존성**: TICKET-006 ~ TICKET-008
**예상 작업량**: M

**작업 내용**
- [ ] 페이지 레이아웃 구성
- [ ] Retrieval Test 실행 선택 연동 (source_run_id)
- [ ] source_config 표시 (읽기 전용, 선택된 실행의 검색 파라미터)
- [ ] [답변 생성] 버튼 → 순차 실행 로직
- [ ] 진행률 표시
- [ ] 결과 테이블 연동
- [ ] 집계 메트릭 표시

**파일**
- `frontend/src/features/batch-answer/pages/batch-answer-page.tsx`
- `frontend/src/features/batch-answer/hooks/use-batch-answer.ts`

---

#### TICKET-010: 라우팅 및 네비게이션 추가
**우선순위**: P2
**의존성**: TICKET-009
**예상 작업량**: S

**작업 내용**
- [ ] `/batch-answer` 라우트 추가
- [ ] 사이드바/네비게이션에 메뉴 추가

**파일**
- `frontend/src/App.tsx` 또는 라우터 설정 파일
- `frontend/src/components/layout/sidebar.tsx`

---

### Phase 4: 테스트 및 검증

#### TICKET-011: 백엔드 테스트
**우선순위**: P2
**의존성**: TICKET-004
**예상 작업량**: M

**작업 내용**
- [ ] BatchAnswerService 유닛 테스트
- [ ] API 엔드포인트 통합 테스트
- [ ] Answer-only 로직 검증 테스트

**파일**
- `backend/tests/test_batch_answer.py`

---

#### TICKET-012: E2E 테스트
**우선순위**: P3
**의존성**: TICKET-010
**예상 작업량**: M

**작업 내용**
- [ ] Retrieval Test 실행 → Batch Answer 생성 → 평가 플로우 테스트
- [ ] 결과 비교 검증

---

## 티켓 요약

| 티켓 | 제목 | 우선순위 | 의존성 | 크기 |
|------|------|----------|--------|------|
| 000-1 | Retrieval Test에서 Chat 파이프라인 호출 | P0 (선행) | - | M |
| 000-2 | Retrieval Test에 content 저장 | P0 (선행) | 000-1 | M |
| 001 | ES 인덱스 매핑 | P0 | 000-2 | S |
| 002 | BatchAnswerService | P0 | 001 | M |
| 003 | API 엔드포인트 | P0 | 002 | M |
| 004 | Answer-only 답변 생성 | P0 | 003 | M |
| 005 | FE 타입/API | P0 | - | S |
| 006 | Retrieval Test 실행 선택 | P1 | 005 | S |
| 007 | 결과 테이블 | P1 | 005 | M |
| 008 | 답변 상세 팝업 | P1 | 007 | M |
| 009 | 메인 페이지 | P1 | 006-008 | M |
| 010 | 라우팅/네비게이션 | P2 | 009 | S |
| 011 | 백엔드 테스트 | P2 | 004 | M |
| 012 | E2E 테스트 | P3 | 010 | M |

**총 예상**: 선행 2개, 백엔드 4개, 프론트엔드 6개, 테스트 2개 = **14개 티켓**

---

## 개발 순서 (권장)

```
Week 1: 선행 + 백엔드
  000-1 → 000-2 → 001 → 002 → 003 → 004

Week 2: 프론트엔드
  005 → 006, 007 (병렬) → 008

Week 3: 통합 + 마무리
  009 → 010 → 011 → 012
```
