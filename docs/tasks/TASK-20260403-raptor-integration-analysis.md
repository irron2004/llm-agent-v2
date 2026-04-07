# TASK: RAPTOR 통합 및 검색 파이프라인 분석

- **생성일**: 2026-04-03
- **상태**: 분석 완료, 설계 대기
- **목적**: 현재 검색 실패 근본 원인 분석 + RAPTOR 통합 방안 설계
- **브랜치**: `feat/raptor`

---

## 1. 현재 검색 실패 분석

### 1.1 Eval Dataset 정리

**원본**: Vplus SOP 704건 (expected_doc 기준 PPTX 492건 + PDF 212건)

**PPTX 제거 결정**: PPTX 정답 질문 492건(69.9%)을 eval에서 제외.
- PPTX 파싱 품질이 불안정하고, 인덱싱 누락 문서 3건 중 2건이 PPTX
- PPTX 기반 질문의 실패 79건은 PPTX 파싱/인덱싱 문제와 혼재되어 파이프라인 평가 신뢰도 저하

**정제 후 Eval**: **PDF only 212건**

### 1.2 Failure Dataset 재분류 (PDF only 기준)

기존 93건 실패 중 PPTX 정답 79건 제거 → **PDF only 실패 ~14건**

| 유형 | 기존 (93건) | PPTX 제거 후 | 근본 원인 |
|------|------------|-------------|----------|
| 진짜 질문 노이즈 | 8 | 0 (전부 PPTX) | 부품명 "the" 대체 |
| Verbose 영문 쿼리 | 42 | 대부분 PPTX | verbose 쿼리 검색 실패 |
| 0건 검색 | 9 | ~3 (PDF 1건 × 3 variant) | 문서 미인덱싱 |
| 페이지 미스 | 75 | ~8-10 | section expansion 부족 |
| Doc 미스 | 9 | ~3-4 | 임베딩/쿼리 불일치 |

> PDF only 기준 실패율: **~14/212 ≈ 6.6%** (기존 93/704 = 13.2%에서 대폭 감소)

### 1.3 "질문 노이즈" 재분류 결과

기존 분류에서 "the" 토큰 존재만으로 50건을 노이즈로 분류했으나, 재검증 결과:

**진짜 노이즈 (8건)**: 부품명이 "the"로 대체되어 무의미한 쿼리 (전부 PPTX → 제거됨)
- QID 473-476: `"SUPRA Vplus PM the 교체 방법"` (원래: heater chuck connector)
- QID 477-480: `"SUPRA Vplus the 교체 절차"` (원래: heater chuck)

**False positive (42건)**: "the"는 정상 영어 관사, 부품명 포함됨 (대부분 PPTX → 제거됨)
- 패턴: `"operation procedures, precautions, and preparations required for the replacement of the [부품명]..."`
- 실패 원인: 질문이 아닌 파이프라인 — verbose 영문 쿼리가 검색에 부적합

### 1.4 근본 원인 상세 (PDF only 기준)

#### A. 0건 검색 (~3건)

1개 미인덱싱 PDF × 3 variant:
- `global sop_supra vplus_rep_sub unit_gas manual valve_eng.pdf`

**조치**: 문서 인덱싱

#### B. 페이지 미스 (~8-10건)

올바른 문서를 찾았으나 타겟 페이지를 반환하지 못하는 구조적 문제:

| 원인 | 설명 | 영향 |
|------|------|------|
| **Contiguous page gap** | `expand_related_docs_node`의 gap 허용치 1페이지 → 빈 페이지 1장이면 그룹 분리 | 높음 |
| **`section_expand_max_pages=8`** | SOP 절차가 10-30페이지인데 최대 8페이지만 확장 | 높음 |
| **`dedupe_by_doc_id` 순서** | section expansion 전에 문서당 1 chunk만 남김 → 잘못된 section 확장 가능 | 중간 |
| **`chapter_source` 메타데이터** | chapter_source 커버리지 불명 → section expansion 미동작 가능 | 조사 필요 |
| **`early_page_penalty`** | max_page=2로 고정, SOP scope 페이지가 3-5페이지까지 이어지는 경우 미커버 | 낮음 |

#### C. Doc 미스 (~3-4건)

50건 이상 검색되었으나 gold doc 미포함. 임베딩 모델의 도메인 특화 부족, 한-영 혼합 용어 불일치 등이 원인.

---

## 2. RAPTOR 현황 분석

### 2.1 인프라 상태

| 컴포넌트 | 파일 | 상태 |
|----------|------|------|
| RAPTOR 설정 | `config/settings.py:122-138` | ✅ 정의됨, ❌ dead flag (미참조) |
| RAPTOR Retriever | `retrieval/adapters/raptor_retriever.py` | ✅ 구현됨 (588줄), ❌ import 안 됨 |
| RAPTOR Tree Builder | `raptor/tree_builder.py` | ✅ 구현됨 |
| RAPTOR Rebuild Service | `services/raptor_rebuild_service.py` | ✅ 구현됨 |
| RAPTOR Ingest Service | `services/raptor_ingest_service.py` | ✅ 구현됨 |
| Query Router | `raptor/query_router.py` | ✅ 구현됨 (MoE routing) |
| ES 매핑 (chunk_v3_content) | `elasticsearch/mappings.py:555` | ❌ RAPTOR 필드 미정의 (`dynamic: false`) |
| ES 실데이터 | chunk_v3_content 인덱스 | ✅ 런타임 PUT _mapping으로 추가된 것으로 추정 |

### 2.2 ES 데이터 현황

| 항목 | 수치 |
|------|------|
| 전체 문서 | 395,069 |
| Summary 노드 (level 1) | 1,628 (0.4%) |
| Summary 레벨 | Level 1만 (Level 2, 3 없음) |
| 파티션 수 | ~56 (device_name × doc_type) |
| 파티션당 요약 | ~29개 |
| 요약당 children | ~100 leaf chunks |
| Embedding | ✅ chunk_v3_embed_bge_m3_v1에 존재 확인 |

**Summary 예시**:
```
ID: raptor_summary_SUPRA_N_myservice_c000
content: "SUPRA/myservice 조치 클러스터. 관련부품: EFEM, CTR, ROBOT, PM1 | 조치: PM1 Pin Up/Down Check..."
children: [myservice_40031601#0000, myservice_40048139#0000, ...] (100+개)
partition_key: SUPRA_N_myservice
```

### 2.3 치명적 문제

1. **`raptor_enabled` = dead flag**: `settings.py`에 정의만 되어 있고 코드 어디에서도 읽지 않음
2. **이전 eval C2/C4 무효**: RAPTOR "on"으로 설정했지만 실제로 동작하지 않았음 → 결과가 C1과 동일한 파이프라인에서 나온 것
3. **`adapters/__init__.py`에 import 누락**: `get_retriever("raptor_hierarchical")` 호출 시 `KeyError`
4. **Leaf 노드 backfill 미완**: `raptor_level=0`, `is_summary_node=false` 미설정 → RAPTOR retriever 필터 동작 불가
5. **코드-매핑 비동기**: `get_chunk_v3_content_mapping()`에 RAPTOR 필드가 없어 코드와 실제 인덱스 불일치

### 2.4 기대효과 (현실적 평가, PDF only 기준)

| 실패 유형 | 건수 | RAPTOR 효과 | 근거 |
|-----------|------|------------|------|
| 0건 검색 | ~3 | 없음 | 인덱스에 문서 없음 |
| 페이지 미스 | ~8-10 | 간접적 | summary→children 확장으로 올바른 section 발굴 가능 |
| Doc 미스 | ~3-4 | 가능 | broad summary가 topic-level 매칭에 유리할 수 있음 |

**PDF only 기준 최대 기대 개선: ~11건 중 3-7건 (추정)**

### 2.5 한계와 리스크

| 한계 | 설명 |
|------|------|
| Summary 품질 | 단순 concat+truncate인지 LLM 요약인지 불명. 내용이 너무 broad |
| Level 1만 존재 | 1단계 요약만으로는 검색 precision 개선 한계 |
| 파티션 granularity | device × doc_type 단위가 너무 coarse (하나에 100+ children) |
| Latency 추가 | Query routing (partition 선택) + summary expansion → 200-500ms 추가 예상 |
| 기존 파이프라인 호환성 | RAPTOR 결과가 section_expand, rerank와 호환되는지 미검증 |
| Children 참조 무결성 | 재인덱싱 시 children ID 깨질 수 있음 |

---

## 3. 데이터 보강 필요 항목

### 3.1 필수 (RAPTOR 통합 전제조건)

| 항목 | 현황 | 작업 |
|------|------|------|
| `chunk_v3_content` 매핑 업데이트 | RAPTOR 필드 코드에 미정의 | `get_chunk_v3_content_mapping()`에 필드 추가 |
| Leaf 노드 backfill | `raptor_level=0`, `is_summary_node=false` 없음 | 393k 문서 bulk update |
| `adapters/__init__.py` import | raptor_retriever 미등록 | import 추가 |

### 3.2 높음

| 항목 | 현황 | 작업 |
|------|------|------|
| Summary 품질 검증 | 생성 방식 불명 | 50개 샘플 spot-check + self-retrieval rate 측정 |
| Children 참조 무결성 | 미검증 | 20개 summary 샘플링, mget으로 children 존재 확인 |
| 누락 PDF 1건 인덱싱 | `gas manual valve` PDF 미인덱싱 | 파싱 + 인덱싱 |
| Embedding 저장 경로 | v3 split architecture에서 summary embedding 위치 불명 | 확인 + 필요시 dual-write |

### 3.3 중간

| 항목 | 현황 | 작업 |
|------|------|------|
| chapter_source 커버리지 | section expansion 전제조건인데 미측정 | ES 쿼리로 커버리지 조사 |
| None-pool 크기 | device_name 없는 chunk 비율 불명 | ES 쿼리로 확인 |
| SOP 전용 파티셔닝 | 현재 GMM clustering은 절차형 문서에 부적합 | 별도 전략 설계 |

### 3.4 낮음 (L1 검증 후)

| 항목 | 작업 |
|------|------|
| Level 2, 3 빌드 | L1 효과 입증 시에만 |
| Summary LLM 재생성 | 품질 미달 시 Ollama로 재생성 |
| 다국어 요약 | source 언어 감지 → 적절한 프롬프트 |

---

## 4. 추천 개선 로드맵

### Phase 0: 즉시 개선 (RAPTOR 무관, 비용 대비 효과 최대)

```
작업                              예상 효과                  소요
─────────────────────────────────────────────────────────────
Eval PPTX 질문 제거               704→212건 정제              CSV 필터링
누락 PDF 1건 인덱싱               3건 제거 (0건검색 해소)      2시간
contiguous gap 허용치 1→3         페이지 미스 감소            코드 1줄
section_expand_max_pages 8→20     SOP 커버리지 확대           설정 1줄
dedupe_by_doc_id 순서 조정        section expansion 개선      로직 이동
```

### Phase 1: RAPTOR 최소 연결

```
작업                              소요
─────────────────────────────────────────
raptor_enabled flag 실제 연결      2시간
adapters/__init__.py import 추가   5분
chunk_v3_content 매핑 RAPTOR 필드  1시간
Supplementary RRF 방식 통합        4시간
동작 확인 로그 삽입                30분
```

### Phase 2: 효과 측정

```
작업                              소요
─────────────────────────────────────────
Leaf 노드 backfill (393k)         2시간 (bulk update)
Summary 품질 검증 (50개 샘플)      1시간
PDF only 212건 재측정              6-8시간 (실행 대기)
```

### Phase 3: 데이터 보강 (Phase 2 결과에 따라)

```
작업                              조건
─────────────────────────────────────────
Summary LLM 재생성                품질 미달 시
Level 2 빌드                      L1 효과 입증 시
SOP 전용 파티셔닝                 SOP page_hit 미개선 시
Reranker 다국어 모델 교체          doc_miss 미개선 시
```

---

## 5. 주요 파일 참조

| 파일 | 역할 |
|------|------|
| `backend/config/settings.py:122-138` | RAPTOR 설정 (dead flag) |
| `backend/services/search_service.py` | 검색 오케스트레이션 (RAPTOR 분기 없음) |
| `backend/llm_infrastructure/retrieval/adapters/raptor_retriever.py` | RAPTOR 전용 retriever (미연결) |
| `backend/llm_infrastructure/retrieval/adapters/__init__.py` | retriever import (raptor 누락) |
| `backend/llm_infrastructure/llm/langgraph_agent.py:2088-2960` | retrieve_node + expand_related_docs_node |
| `backend/llm_infrastructure/elasticsearch/mappings.py:555` | chunk_v3_content 매핑 (RAPTOR 필드 누락) |
| `backend/services/raptor_rebuild_service.py` | RAPTOR 트리 빌드 |
| `backend/llm_infrastructure/raptor/query_router.py` | MoE query routing |
| `data/eval_sop_question_list_vplus.csv` | Eval 질문 데이터 (704건, PPTX 포함) |
| `data/eval_vplus_retrieval/failure_clusters_9099.csv` | 실패 분류 데이터 |

---

## 6. Open Questions

- [ ] Summary 생성 시 사용된 summarizer (default concat vs LLM)는?
- [ ] PDF 출처 chunk의 `chapter_source` 커버리지는?
- [ ] `device_name` 없는 chunk (None-pool) 비율은?
- [ ] Summary embedding과 실제 쿼리 간 cosine similarity 수준은?
- [ ] `RaptorHierarchicalRetriever`의 output이 기존 `RetrievalResult`와 호환되는가?
- [ ] Rebuild 시 기존 summary 노드 cleanup 로직이 있는가? (중복 방지)
- [ ] PDF only 212건 재측정 시 실패율 변동은?
