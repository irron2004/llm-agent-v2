# Retrieval 결과 평가 기능 구현 계획

## 개요

채팅 화면과 Search 페이지에서 검색된 문서들에 대해 관련성 점수(1-5점)를 평가하고, 평가 결과를 ES에 **쿼리 단위로** 저장하여 Retrieval Test 및 검색 파라미터 튜닝에 활용한다. 제출 버튼 기준으로 일괄 저장되며 미평가 문서는 저장에서 제외한다.

---

## 주요 결정 사항

| 항목 | 결정 | 비고 |
|------|------|------|
| 평가 점수 방식 | 1~5점 | 3점 이상 = relevant, 미만 = irrelevant |
| 저장 방식 | 제출 버튼 클릭 시 일괄 저장 | 별점 클릭은 로컬 상태만 변경 |
| 저장 단위 | 쿼리 단위 1건 | query_id 기준 upsert |
| 제출 조건 | 일부 평가만 해도 제출 가능 | 미평가 문서는 저장하지 않음 |
| 평가 대상 | `retrievedDocs` (표시된 문서만) | `allRetrievedDocs`(20개)는 평가 대상 아님 |
| UI 위치 | message-item.tsx의 Collapse 내부 | "확장 문서/참고 문서" 영역 |
| 우측 사이드바 | 평가 기능 없음 | 채팅 Collapse 영역에서만 평가 |
| 알림 | Antd `message.success` | "저장 되었습니다" 토스트 |
| Search 재현성 | search_params 저장 | field_weights, bm25Only, dense/sparse, size 등 |
| Search query_id | 검색 실행 시점 고정 | 제출 시 재사용 |

---

## 핵심 데이터 구조

**채팅 화면에서 평가:**
```json
{
  "query_id": "sess1:turn1",
  "source": "chat",
  "session_id": "sess1",
  "turn_id": 1,
  "query": "SUPRA XP 유지보수 방법은?",
  "relevant_docs": ["doc_001", "doc_003"],
  "irrelevant_docs": ["doc_002", "doc_004"],
  "doc_details": [
    { "doc_id": "doc_001", "doc_rank": 1, "doc_title": "SUPRA XP Manual", "relevance_score": 4, "retrieval_score": 0.875 },
    { "doc_id": "doc_002", "doc_rank": 2, "doc_title": "EFEM Guide", "relevance_score": 2, "retrieval_score": 0.721 }
  ],
  "filter_devices": ["SUPRA XP"],
  "filter_doc_types": ["manual"],
  "search_queries": ["SUPRA XP 유지보수", "maintenance procedure"],
  "ts": "2026-02-03T..."
}
```

**Search 페이지에서 평가:**
```json
{
  "query_id": "search:1706947200000",
  "source": "search",
  "query": "SUPRA 유지보수",
  "relevant_docs": ["doc_001", "doc_003"],
  "irrelevant_docs": ["doc_002", "doc_004"],
  "doc_details": [
    { "doc_id": "doc_001", "doc_rank": 1, "doc_title": "SUPRA XP Manual", "relevance_score": 4, "retrieval_score": 0.875 }
  ],
  "search_params": {
    "field_weights": "search_text^1.0,chunk_summary^0.7",
    "bm25_only": true,
    "dense_weight": 0.0,
    "sparse_weight": 1.0,
    "size": 20
  },
  "ts": "2026-02-03T..."
}
```

---

## 구현 범위

### 1. ES 평가 인덱스 (retrieval_evaluations)

**인덱스 네이밍**
- Index: `retrieval_evaluations_{env}_v{version}`
- Alias: `retrieval_evaluations_{env}_current`

**주요 필드**

| 필드 | 타입 | 용도 |
|------|------|------|
| query_id | keyword | PK (chat: `{session_id}:{turn_id}`, search: `search:{timestamp}`) |
| source | keyword | 출처 (`chat` 또는 `search`) |
| session_id | keyword | 세션 참조 (chat만 해당) |
| turn_id | integer | 턴 참조 (chat만 해당) |
| query | text (nori) | 원본 쿼리 |
| relevant_docs | keyword[] | 관련 문서 ID 목록 (relevance >= 3) |
| irrelevant_docs | keyword[] | 비관련 문서 ID 목록 (relevance < 3) |
| doc_details | nested | 각 문서별 상세 정보 (필수) |
| search_params | object | Search 전용 검색 파라미터 (재현성) |
| filter_devices | keyword[] | 검색 필터 (재현성) |
| filter_doc_types | keyword[] | 검색 필터 (재현성) |
| search_queries | text[] | Multi-query expansion 결과 |
| reviewer_name | keyword | 평가자명 |
| ts | date | 최종 평가 시각 |

**doc_details nested 구조 (필수)**

```json
{
  "doc_details": [
    {
      "doc_id": "doc_001",
      "doc_rank": 1,
      "doc_title": "SUPRA XP Manual",
      "relevance_score": 4,
      "retrieval_score": 0.875,
      "doc_snippet": "...",
      "chunk_id": "doc_001#0001",
      "page": 12
    }
  ]
}
```

---

### 2. 백엔드 API

**새로 생성할 파일**

#### `backend/services/retrieval_evaluation_service.py`

```python
class RetrievalEvaluationService:
    def save_evaluation(query_id, evaluation_data) -> str
        """쿼리 단위 평가 저장 (upsert)"""

    def get_evaluation(query_id) -> RetrievalEvaluation | None
        """쿼리별 평가 조회"""

    def list_evaluations(limit, offset, source=None) -> list[RetrievalEvaluation]
        """전체 평가 목록"""

    def export_for_retrieval_test(min_relevance=3) -> list[dict]
        """Retrieval test set 형식으로 내보내기"""
```

#### `backend/api/routers/retrieval_evaluation.py`

| Method | Endpoint | 용도 |
|--------|----------|------|
| POST | `/api/retrieval-evaluation/{query_id}` | 쿼리 단위 평가 저장 (제출, upsert) |
| GET | `/api/retrieval-evaluation/{query_id}` | 쿼리별 평가 조회 |
| GET | `/api/retrieval-evaluation` | 전체 평가 목록 |
| GET | `/api/retrieval-evaluation/export/json` | retrieval test set JSON 내보내기 |
| GET | `/api/retrieval-evaluation/export/csv` | retrieval test set CSV 내보내기 |
| GET | `/api/retrieval-evaluation/statistics` | 평가 통계 |

**query_id 규칙**
- chat: `{session_id}:{turn_id}`
- search: `search:{timestamp}` (검색 실행 시점에 생성)

**POST Request Body 예시:**
```json
{
  "source": "chat",
  "query": "SUPRA XP 유지보수 방법은?",
  "doc_details": [
    { "doc_id": "doc_001", "relevance_score": 4, "doc_title": "...", "doc_rank": 1, "retrieval_score": 0.875 },
    { "doc_id": "doc_002", "relevance_score": 2, "doc_title": "...", "doc_rank": 2, "retrieval_score": 0.721 }
  ],
  "session_id": "sess1",
  "turn_id": 1,
  "filter_devices": ["SUPRA XP"],
  "filter_doc_types": ["manual"],
  "search_queries": ["SUPRA XP 유지보수", "maintenance procedure"],
  "reviewer_name": "홍길동"
}
```

**파생 규칙**
- `relevant_docs/irrelevant_docs`는 `doc_details.relevance_score` 기준(min_relevance=3)으로 서버에서 자동 생성
- `doc_details`에 포함되지 않은 문서는 미평가로 간주되어 저장 대상에서 제외

**수정할 파일**
- `backend/llm_infrastructure/elasticsearch/mappings.py`: `get_retrieval_evaluation_mapping()` 추가
- `backend/api/main.py`: 라우터 등록

---

### 3. 프론트엔드 UI

#### 채팅 화면 (문서별 평가)

**새로 생성할 파일**
- `frontend/src/features/chat/components/doc-relevance-rating.tsx` - 개별 문서 별점
- `frontend/src/features/chat/components/retrieval-evaluation-form.tsx` - 제출 버튼 포함 폼

**UI 흐름:**
1. "문서 펼치기" 클릭 → 검색된 문서 목록 표시
2. 각 문서마다 별점(1-5) 선택 (로컬 상태에만 저장)
3. 하단 "제출" 버튼 클릭
4. API 호출 → 모든 문서 평가 한번에 저장
5. 성공 시 "저장 되었습니다" 토스트 알림 표시

**동작:**
- 3점 이상 → relevant_docs에 추가
- 3점 미만 → irrelevant_docs에 추가
- 미평가 문서는 저장하지 않음

**수정할 파일**
- `frontend/src/features/chat/components/message-item.tsx`: 평가 UI 통합
- `frontend/src/features/chat/api.ts`: API 함수 추가
- `frontend/src/features/chat/types.ts`: 타입 추가

#### Search 페이지 (검색 결과 평가)

**수정할 파일**
- `frontend/src/features/search/pages/search-page.tsx`

**UI 흐름:**
1. 검색어 입력 → 검색 결과 표시
2. 각 문서마다 별점(1-5) 선택 (로컬 상태)
3. 하단 "평가 제출" 버튼 클릭
4. API 호출 → 저장 (query_id는 `search:{timestamp}` 형식)
5. 성공 시 "저장 되었습니다" 토스트 알림

**참고:** Search 페이지는 session_id/turn_id가 없으므로 query_id를 `search:{timestamp}` 형식으로 **검색 실행 시점에 생성**하고 제출 시 재사용  
search_params에 field_weights, bm25Only, dense/sparse, size 등 검색 설정을 함께 저장

---

#### Retrieval Test 화면 (질문별 평가 확인 + 테스트)

**수정할 파일**
- `frontend/src/features/retrieval-test/` 하위 파일들

**공통 기능 (Chat/Search와 동일):**
1. 문서별 별점 평가 (1-5점) - 동일 컴포넌트 재사용
2. 제출 버튼 → 일괄 저장
3. "저장 되었습니다" 토스트 알림
4. 미평가 문서 제외

**추가 기능:**
1. **평가 목록 조회**: Chat/Search에서 저장된 질문들 목록 표시
2. **정답 추가 지정**: 검색 결과 외 문서도 정답으로 추가 가능
3. **문서 이동**: 정답 ↔ 오답 간 문서 이동 (별점 수정으로 구현)
4. **Retrieval 테스트 실행**: 저장된 질문들로 재검색 → NDCG/Recall 계산

**UI 흐름:**
1. 평가된 질문 목록에서 질문 선택
2. 정답/오답 문서 목록 + 각 문서별 별점 표시/수정
3. "문서 추가" → 문서 검색 팝업 → 정답으로 추가
4. "평가 제출" → 저장
5. "테스트 실행" → 재검색 → 메트릭 계산

---

## UI 레이아웃

### 채팅 화면 - 문서 평가

```
┌─────────────────────────────────────────────────────────────┐
│ ▼ 확장 문서/참고 문서 (5)                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. Document Title A                                 p.123   │
│    [Document image or snippet text]                         │
│    score: 0.875 (85%)                                       │
│    관련성: ★★★★☆                                            │
│                                                             │
│ 2. Document Title B                                 p.45    │
│    [Document image or snippet text]                         │
│    score: 0.721 (72%)                                       │
│    관련성: ★★☆☆☆                                            │
│                                                             │
│ 3. Document Title C                                 p.8     │
│    [Document image or snippet text]                         │
│    score: 0.654 (65%)                                       │
│    관련성: ☆☆☆☆☆  (미평가)                                  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                        [ 평가 제출 ]        │
└─────────────────────────────────────────────────────────────┘

→ "평가 제출" 클릭 시: "저장 되었습니다" 토스트 알림
```

### Search 페이지 - 검색 결과 평가

```
┌─────────────────────────────────────────────────────────────┐
│ [검색 설정 패널]              │  검색 결과 (총 20건)        │
│                              │                              │
│ 검색어: [SUPRA 유지보수    ] │  1. SUPRA XP Manual          │
│                              │     스코어: 0.875            │
│ ☑ 본문 텍스트    1.0        │     [snippet text...]        │
│ ☑ 청크 요약      0.7        │     관련성: ★★★★☆           │
│ ☐ 원본 콘텐츠    0.6        │                              │
│                              │  2. SUPRA Maintenance Guide  │
│                              │     스코어: 0.721            │
│                              │     [snippet text...]        │
│                              │     관련성: ★★☆☆☆           │
│                              │                              │
│                              │  ...                         │
│                              │                              │
│                              │  ─────────────────────────── │
│                              │              [ 평가 제출 ]   │
└─────────────────────────────────────────────────────────────┘

→ "평가 제출" 클릭 시: "저장 되었습니다" 토스트 알림
→ query_id: "search:1706947200000" (타임스탬프 기반)
```

### Retrieval Test 화면 - 질문별 평가

```
┌─────────────────────────────────────────────────────────────┐
│ Query: "SUPRA XP 유지보수 방법은?"                           │
├─────────────────────────────────────────────────────────────┤
│ ✅ 정답 문서 (2)                                             │
│   • doc_001 - SUPRA XP Manual p.45                          │
│   • doc_003 - SUPRA XP Maintenance Guide p.12               │
├─────────────────────────────────────────────────────────────┤
│ ❌ 오답 문서 (2)                                             │
│   • doc_002 - EFEM Installation Guide p.3                   │
│   • doc_004 - General Safety Manual p.1                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 구현 순서

### Phase 1: 백엔드 (ES + API)
1. `mappings.py` - query 단위 스키마 반영 (doc_details, relevant/irrelevant, search_params)
2. `retrieval_evaluation_service.py` - query_id 기준 upsert 로직
3. `retrieval_evaluation.py` - `{query_id}` 기반 API + export/json, export/csv, statistics
4. `main.py` - 라우터 등록/갱신

### Phase 2: 프론트엔드 - 채팅 화면 평가
1. `types.ts` - 타입 정의 추가
2. `api.ts` - API 함수 추가
3. `doc-relevance-rating.tsx` - 개별 문서 별점 컴포넌트
4. `retrieval-evaluation-form.tsx` - 제출 버튼 + 토스트 알림 폼
5. `message-item.tsx` - 평가 UI 통합

### Phase 3: 프론트엔드 - Search 페이지 평가
1. `search-page.tsx` - 평가 UI 통합 (Phase 2 컴포넌트 재사용)
2. query_id 자동 생성 로직 (`search:{timestamp}`) 및 search_params 저장

### Phase 4: 프론트엔드 - Retrieval Test 화면
1. 기존 화면 확인 및 연동 방안 검토
2. 질문별 정답/오답 표시 UI 구현

---

## Export 형식 (Retrieval Test Set)

```json
[
  {
    "query_id": "sess1:turn1",
    "query": "SUPRA XP 유지보수 방법은?",
    "relevant_docs": ["doc_001", "doc_003"],
    "irrelevant_docs": ["doc_002", "doc_004"]
  },
  {
    "query_id": "sess2:turn3",
    "query": "EFEM 에러 코드 E-101 해결 방법",
    "relevant_docs": ["doc_010"],
    "irrelevant_docs": ["doc_011", "doc_012"]
  }
]
```

이 형식은 NDCG, MAP, Recall@K 등 retrieval 메트릭 계산에 바로 사용 가능  
Export는 `doc_details`를 기준으로 min_relevance(기본 3)로 relevant/irrelevant을 분리해 생성

---

## 검증 방법

1. **백엔드 테스트**
   - POST로 문서 평가 저장
   - GET으로 쿼리별 평가 조회
   - Export API로 test set 다운로드

2. **프론트엔드 테스트**
   - 채팅 화면에서 별점 선택 후 "평가 제출" → 저장 확인
   - Search 페이지에서 별점 선택 후 "평가 제출" → 저장 확인
   - 새로고침 후 평가 점수 유지 확인
   - Retrieval Test 화면에서 정답/오답 확인

3. **통합 테스트**
   - 여러 질문 평가 후 export
   - Retrieval 메트릭 계산 스크립트로 검증

---

## 주요 파일 경로

**수정**
- `backend/llm_infrastructure/elasticsearch/mappings.py`
- `backend/api/main.py`
- `frontend/src/features/chat/components/message-item.tsx`
- `frontend/src/features/chat/api.ts`
- `frontend/src/features/chat/types.ts`
- `frontend/src/features/search/pages/search-page.tsx`

**신규 또는 수정/확장**
- `backend/services/retrieval_evaluation_service.py`
- `backend/api/routers/retrieval_evaluation.py`
- `frontend/src/features/chat/components/doc-relevance-rating.tsx`
- `frontend/src/features/chat/components/retrieval-evaluation-form.tsx`
