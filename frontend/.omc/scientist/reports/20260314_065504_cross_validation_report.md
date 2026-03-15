# 교차 검증 보고서: ES 인덱스 × 파이프라인 × 콘텐츠 품질

**작성일**: 2026-03-14  
**분석 대상**: Stage 1(ES 인덱스 구조) × Stage 2(이슈 검색 파이프라인) × Stage 3(doc_type별 콘텐츠 품질)  
**분석 범위**: 코드 직접 검증 포함 (langgraph_agent.py, es_chunk_v3_search_service.py, retrieval_pipeline.py)

---

[OBJECTIVE] Stage 1/2/3 연구 결과의 교차 검증, 인과 관계 연결, 종합 개선 우선순위 도출

[DATA] 
- Stage 1 인덱스 구조 사실: 8개
- Stage 2 파이프라인 병목: 6개 (P1~P6)
- Stage 3 콘텐츠 품질 이슈: 6개 (F1~F6)
- 코드 직접 검증: langgraph_agent.py (4,500+ lines), es_chunk_v3_search_service.py, retrieval_pipeline.py

---

## 1. Stage 간 모순 및 교차 검증 결과

### C1 [명확화 필요] P6 영향도 재분류

- **관련**: Stage 2-P6 (ts section 확장이 chapter_ok 의존) × Stage 3-F4 (ts chunk_v3 0건)
- **사실**: ts 문서가 chunk_v3에 적재되어 있지 않으므로(F4), expand_related_docs_node()의 chapter_ok 분기(코드 L2282-2293)는 현재 실행되지 않는 코드 경로다.
- **결론**: P6 영향도 = 현재 없음 / ts VLM 파싱 완료 후 High. F4 해결 시 P6가 활성화되므로 반드시 함께 처리.

### C2 [범위 한정] P1 영향도 세분화

- **관련**: Stage 2-P1 (issue 모드 general_mq 사용) × Stage 3-F4 (ts 0건)
- **사실**: P1의 즉각적 영향은 gcb/myservice 이슈로 제한된다. ts 이슈에서의 영향은 F4가 해결되기 전까지 잠재적.
- **결론**: P1의 우선순위는 유지되나, "ts 검색어 확장"보다 "gcb/myservice 이슈 키워드 개선"이 현재 실익.

### C3 [방향 수정] P4 영향 방향 오류

- **관련**: Stage 2-P4 (리랭킹 query_en → 한국어 문서 과소평가) × Stage 1-S1-D/F (gcb 영문 기반, nori 적용)
- **오류**: P4의 우려가 gcb에도 적용된다고 기술되었으나, gcb는 영문 콘텐츠이므로 query_en 리랭킹이 **유리**하게 작용한다.
- **수정**: P4 영향 범위 = myservice(한국어) 문서 과소평가로 한정. gcb의 실제 문제는 nori analyzer 미스매치(S1-F).

### C4 [마스킹 효과] P2 영향도 과대평가

- **관련**: Stage 2-P2 (MAX_ANSWER_REFS=5로 6~10번 사례 누락) × Stage 3-F1 (myservice 62.2% 빈 콘텐츠)
- **분석**: P2는 유효한 사례 10개가 존재한다는 전제를 가정한다. F1로 인해 실제 비어있지 않은 myservice 문서가 충분하지 않으면, MAX_ANSWER_REFS 증가의 실익이 없다.
- **결론**: IMP-2(빈 콘텐츠 정제) → IMP-5(REFS 수 증가) 순서가 논리적으로 강제된다.

### C5 [부분 반박] F5 코드 검증 결과

- **관련**: Stage 3-F5 (REFS에 doc_type/section_type 미포함) × 코드 L1281
- **검증**: langgraph_agent.py L1281에서 `section_type`과 `chapter`를 ref 텍스트 우선순위 계산에 포함한다. 따라서 F5는 부분 오류.
- **수정된 사실**: REFS에 `section_type`은 포함(코드상). `doc_type`은 미포함. 실제 문제는 section_type 값('cause', 'action', 'question', 'resolution' 등)의 의미를 LLM 프롬프트가 설명하지 않는 것.

---

## 2. 인과 관계 연결 체인 (데이터 품질 → 인덱스 구조 → 파이프라인 → 출력)

### CC-1 [HIGH] myservice 빈 콘텐츠 → 검색 오염 → 답변 품질 저하
```
F1: myservice 62.2% 빈 콘텐츠 (데이터 품질)
  → ES 인덱스에 빈 문서 적재 (인덱스 구조)
  → BM25/Dense 검색에서 빈 문서 반환 (파이프라인)
  → _filter_noisy_results() 부분 필터링만 (파이프라인)
  → answer_node REFS에 빈/짧은 텍스트 → LLM 답변 품질 저하 (출력)
```

### CC-2 [HIGH] issue 모드 라우팅 고정 → 검색어 품질 저하 → 관련 사례 누락
```
task_mode='issue' 설정 (API 입력)
  → route_node: route='general' 강제 반환 [코드 L1304-1308] (파이프라인)
  → mq_node: general_mq 프롬프트 사용 (이슈 특화 키워드 없음) (파이프라인)
  → gcb/myservice 검색어에 이슈 맥락 부족 → 검색 재현율 저하 (파이프라인)
  → issue_top10_cases 품질 저하 → 사용자가 관련 사례 선택 어려움 (출력)
```

### CC-3 [MEDIUM] gcb 분리 청킹 + nori 미스매치 → 영문 이슈 검색 정밀도 저하
```
F2: gcb Q/Resolution 별도 ES 문서 (데이터 구조)
  + S1-F: nori analyzer(한국어) → 영문 토큰화 부정확 (인덱스 구조)
  → BM25 검색에서 gcb 청크 점수 과소평가 (파이프라인)
  → Q만 검색, Resolution 누락 (또는 반대) → 절반 정보만 REFS 포함 (파이프라인)
  → LLM이 원인 없이 해결방안만, 또는 해결방안 없이 이슈만 제시 (출력)
```

### CC-4 [CRITICAL] ts VLM 파싱 미완 → chunk_v3 공백 → ts 사례 검색 불가
```
F4: ts PDF VLM 파싱 미완료 (운영 상태)
  → chunk_v3에 ts 문서 0건 (인덱스 상태)
  → issue 모드 doc_type 필터 ts 포함이나 검색 결과 없음 (파이프라인)
  → 가장 풍부한 기술 사례 정보인 ts 완전 누락 (출력)
[비고: 다른 모든 개선의 최종 효과는 F4 해결에 의존]
```

### CC-5 [MEDIUM] REFS 메타 부재 → LLM 문서 구분 불가 → 답변 구조 혼란
```
F5(수정): doc_type REFS 미포함, section_type 의미 미설명 (데이터/파이프라인)
  → LLM이 gcb question/resolution을 같은 문서로 병합 실패 (파이프라인)
  → 답변에서 원인-결과 구조 붕괴 또는 gcb 영문 내용 부정확 인용 (출력)
```

---

## 3. 종합 개선 우선순위 (ROI = Impact / Effort × 10)

| 순위 | ID | ROI | Impact | Effort | 분류 | 제목 |
|------|-----|-----|--------|--------|------|------|
| 1 | IMP-3 | 30.0 | 3 | 1 | quick-win | REFS 포맷에 doc_type + section_type 레이블 명시 |
| 2 | IMP-5 | 30.0 | 3 | 1 | quick-win | MAX_ANSWER_REFS 이슈 모드 한정 증가 (5→10) |
| 3 | IMP-1 | 20.0 | 4 | 2 | quick-win | issue 모드 전용 MQ 프롬프트 추가 |
| 4 | IMP-2 | 20.0 | 4 | 2 | quick-win | myservice 빈 콘텐츠 인제스트 필터링 강화 |
| 5 | IMP-4 | 10.0 | 3 | 3 | short-term | gcb 인덱스에 english analyzer 추가 |
| 6 | IMP-6 | 10.0 | 3 | 3 | short-term | gcb Q+Resolution 동일 doc_id 묶음 청킹 |
| 7 | IMP-7 | 10.0 | 5 | 5 | long-term | ts VLM 파싱 완료 및 chunk_v3 적재 |

**주의**: IMP-5는 IMP-2 선행 필수 (빈 문서 10개가 REFS로 가면 역효과).

---

## 4. 실행 가능한 개선안 5~7개 (코드 변경 위치 명시)

### [IMP-1] Quick-Win: issue 모드 전용 MQ 프롬프트 추가
**예상 효과**: issue 검색 재현율 +15~25%  
**변경 파일**:
- `backend/llm_infrastructure/llm/langgraph_agent.py`
  - `route_node()` L1304-1308: `route='general'` 강제를 `route='issue'` 또는 조건부로 변경
  - `mq_node()` L1376-1381: `elif route == 'issue':` 분기 추가, `spec.issue_mq` 호출
  - `PromptSpec` 데이터클래스: `issue_mq: Optional[PromptTemplate] = None` 필드 추가
- `backend/llm_infrastructure/llm/prompts/issue_mq_v2.yaml` (신규 생성)
  - 이슈 키워드 특화: 증상, 알람코드, 에러, 원인, 조치, 재발방지 등
- `backend/llm_infrastructure/llm/langgraph_agent.py:load_prompt_spec()` L253-294: issue_mq 로딩 추가

### [IMP-2] Quick-Win: myservice 빈 콘텐츠 인제스트 필터링 강화
**예상 효과**: REFS 노이즈 -62%, 답변 품질 직접 개선  
**변경 파일**:
- `backend/services/ingest/txt_parser.py`: 섹션 파싱 시 최소 길이(10자) 미만 섹션 스킵
- `backend/llm_infrastructure/text_quality.py`: `is_noisy_chunk()` — 짧은 콘텐츠 임계값 상향 (현재 미검증)
- `backend/services/es_ingest_service.py`: 인제스트 직전 빈 content 문서 필터링 로직 추가
- **운영**: 기존 myservice 문서 재인제스트 필요 (ES 기존 빈 문서 삭제)

### [IMP-3] Quick-Win: REFS 포맷에 doc_type + section_type 의미 레이블 추가
**예상 효과**: LLM gcb Q/Resolution 쌍 인식 개선, 답변 구조 개선  
**변경 파일**:
- `backend/llm_infrastructure/llm/langgraph_agent.py`
  - `_build_ref_text()` 또는 ref 포맷 생성 구간 L1278-1284: `meta.get('doc_type', '')`를 REF 헤더에 추가
  - 예: `[doc_type=gcb][section=question]` 헤더 형식
- `backend/llm_infrastructure/llm/prompts/issue_ans_v*.yaml`: 
  - 시스템 프롬프트에 "gcb 문서의 question/resolution은 같은 이슈의 쌍" 설명 추가
  - section_type 값 사전: `status=현상, action=조치, cause=원인, result=결과`

### [IMP-4] Short-term: gcb 인덱스에 english analyzer 추가
**예상 효과**: gcb BM25 검색 정밀도 +20~30%  
**변경 파일**:
- `backend/llm_infrastructure/elasticsearch/` (ES 인덱스 매핑 정의 파일):
  - `search_text` 필드에 `fields.english` sub-field 추가 (standard/english analyzer)
- `backend/llm_infrastructure/retrieval/engines/es_search.py`:
  - `_build_text_query()`: gcb doc_type일 때 `search_text.english` 필드 부스팅
- **운영**: ES 인덱스 재생성 + gcb 문서 재인제스트

### [IMP-5] Quick-Win: MAX_ANSWER_REFS 이슈 모드 한정 증가
**예상 효과**: issue 사례 제시 수 +100% (5→10)  
**변경 파일**:
- `backend/llm_infrastructure/llm/langgraph_agent.py`
  - L1103: `MAX_ANSWER_REFS = 5` 유지
  - L2694 issue 분기: `MAX_ISSUE_REFS = 10` 별도 상수 사용
  - `MAX_ISSUE_REFS = 10` 상수 L1103 근방에 추가
- **전제조건**: IMP-2 먼저 적용 필수

### [IMP-6] Short-term: gcb Q+Resolution 병합 청킹 또는 expand 전략
**예상 효과**: gcb 정보 완결성 ~100% (현재 84.1%)  
**변경 파일 (옵션 A - 파서 수정)**:
- `backend/services/ingest/gcb_parser.py`: question + resolution을 단일 청크로 병합
**변경 파일 (옵션 B - 런타임 확장)**:
- `backend/llm_infrastructure/llm/langgraph_agent.py:expand_related_docs_node()` L2228:
  - gcb 문서 검색 시 동일 doc_id의 section_type pair(question↔resolution)를 항상 같이 fetch

### [IMP-7] Long-term: ts VLM 파싱 완료 및 chunk_v3 적재
**예상 효과**: ts 사례 검색 가능 (현재 0건 → 전체), 이슈 모드 최대 효과  
**변경 파일**:
- VLM 파싱 파이프라인 (별도 스크립트/서비스)
- `backend/services/ingest/document_ingest_service.py`: ts PDF 처리 경로
- L4 정규화(PM 주소 마스킹) 활성화 여부 결정 (ts 적재와 함께)
- **활성화 효과**: P6(chapter_ok expand) 자동 활성화, IMP-1 ts 이슈 검색 효과 실현

---

## 5. Quick-win vs Long-term 분류 및 실행 순서

### Quick-win (1~2주, 코드 변경만)
실행 순서가 중요:
1. **IMP-3** (1일): REFS 포맷 레이블 추가 — 가장 안전한 변경, 부작용 없음
2. **IMP-2** (3~5일): myservice 빈 콘텐츠 필터링 + 재인제스트
3. **IMP-1** (3~5일): issue_mq 프롬프트 생성 + route_node/mq_node 수정
4. **IMP-5** (1일): MAX_ISSUE_REFS 증가 — IMP-2 적용 후

### Short-term (2~4주, ES 인덱스 재생성 포함)
5. **IMP-6** (1~2주): gcb Q+Resolution 병합 전략 선택 및 구현
6. **IMP-4** (2~3주): gcb english analyzer 추가 + ES 재생성

### Long-term (4주~, 인프라 작업)
7. **IMP-7** (4주+): ts VLM 파싱 완료 — 모든 개선의 최종 승수 효과

---

## 6. 제한사항 (Limitations)

[LIMITATION] 
- 이 분석은 ES 인덱스에 직접 쿼리하지 않고 코드 및 Stage 1/2/3 보고서를 기반으로 교차 검증함. 실제 문서 수/비율은 라이브 ES 접근 시 달라질 수 있음.
- 영향도 수치(+15~25% 등)는 코드 구조 분석 기반 추정값이며, A/B 테스트로 검증 필요.
- IMP-5의 MAX_ISSUE_REFS 증가는 LLM 컨텍스트 윈도우 제한을 고려해야 함 (10개 × 평균 REF 크기 확인 필요).
- S1-G(L4/L5 미적용)의 영향은 실제 PM 주소/알람코드 기반 쿼리 빈도에 따라 달라짐 — 현재 분석에서 개선안에 미포함.

---

Figure: /home/hskim/work/llm-agent-v2/frontend/.omc/scientist/figures/cross_validation_matrix.png
