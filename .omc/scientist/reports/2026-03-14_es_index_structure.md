# ES 인덱스 데이터 구조 조사 보고서
## ts, myservice, gcb doc_type

날짜: 2026-03-14

---

## [OBJECTIVE]
chunk_v3 ES 인덱스에서 ts, myservice, gcb doc_type의 데이터 형태, 매핑, 청킹 방식, 콘텐츠 구조를 코드 근거와 함께 파악한다.

---

## [DATA] 조사 파일 목록

| 파일 경로 | 역할 |
|---|---|
| backend/domain/doc_type_mapping.py | doc_type 그룹 매핑 |
| backend/llm_infrastructure/elasticsearch/mappings.py | ES 인덱스 매핑 정의 |
| backend/llm_infrastructure/elasticsearch/document.py | EsChunkDocument 스키마 |
| backend/services/ingest/txt_parser.py | myservice txt 파서 |
| backend/services/ingest/gcb_parser.py | gcb txt 파서 |
| backend/services/ingest/document_ingest_service.py | PDF 인제스트 서비스 (ts용) |
| backend/services/ingest/metadata_extractor.py | 챕터/메타 추출기 |
| backend/services/es_ingest_service.py | ES 인제스트 오케스트레이션 |
| backend/services/es_chunk_v3_search_service.py | chunk_v3 검색 서비스 |
| backend/llm_infrastructure/preprocessing/normalize_engine/domain.py | L3~L5 정규화 |
| backend/config/presets/retrieval_hybrid_rrf_v1.yaml | 검색 프리셋 |

---

## [FINDING 1] doc_type 그룹 및 variant 확장 매핑

**파일:** `backend/domain/doc_type_mapping.py:9-50`

5개 canonical 그룹으로 정규화되며, ts와 gcb 그룹의 variant 목록:

### ts 그룹 (10개 variant)
```
"문제 해결 가이드", "Trouble Shooting Guide", "trouble shooting",
"trouble shooting guide", "trouble shooting guilde", "troubleshooting",
"Guide", "ts", "t/s", "Troubleshooting Guide", "trouble_shooting_guide"
```
- `expand_doc_type_selection(["ts"])` → 위 10개 값 전부 반환
- ES 필터 쿼리에서 `doc_type` 필드에 이 10개 값으로 terms 필터 적용

### gcb 그룹 (2개 variant)
```
"gcb", "maintenance"
```
- 단 2개 variant. "maintenance" 포함이 주목할 점.

### myservice 그룹 (1개 variant)
```
"myservice"
```
- 단일 canonical 값. variant 없음.

**[STAT:n]** 그룹 수: 5개 (myservice, SOP, ts, setup, gcb)
**[STAT:n]** ts variant 수: 10개 / gcb variant 수: 2개 / myservice variant 수: 1개

---

## [FINDING 2] chunk_v3 ES 인덱스 매핑 구조

**파일:** `backend/llm_infrastructure/elasticsearch/mappings.py:555-596` (`get_chunk_v3_content_mapping`)

### chunk_v3_content 인덱스 필드 전체 목록

| 필드명 | 타입 | analyzer | 설명 |
|---|---|---|---|
| chunk_id | keyword | - | Primary key |
| doc_id | keyword | - | 문서 식별자 |
| page | integer | - | 페이지 번호 |
| lang | keyword | - | 언어 코드 (ko/en) |
| content | text | nori | 청크 원문 (BM25 검색용) |
| search_text | text | nori | content + title + tags 조합 |
| chunk_summary | text | - (index:True) | LLM 생성 요약 |
| chunk_keywords | keyword + text | standard | 청크 키워드 |
| doc_type | keyword | - | **ts / gcb / myservice 등** |
| device_name | keyword | - | 장비명 (필터용) |
| equip_id | keyword | - | 설비 ID |
| tenant_id | keyword | - | 멀티테넌시 |
| project_id | keyword | - | 프로젝트 |
| chapter | keyword | - | 챕터/섹션 제목 |
| section_chapter | keyword | - | 섹션 챕터 |
| section_number | integer | - | 섹션 번호 |
| chapter_source | keyword | - | 챕터 출처 (rule/llm/carry_forward) |
| chapter_ok | boolean | - | 챕터 유효 여부 |
| content_hash | keyword | - | 중복 방지 해시 |
| chunk_version | keyword | - | 청크 버전 |
| pipeline_version | keyword | - | 파이프라인 버전 |
| created_at | date | - | 생성 시각 |
| extra_meta | object (disabled) | - | 비정형 추가 메타 |

### chunk_v3_embed 인덱스 필드 (벡터 전용 분리 인덱스)

| 필드명 | 설명 |
|---|---|
| chunk_id | content 인덱스와 join key |
| doc_id | 문서 식별자 |
| content_hash | 중복 방지 |
| doc_type | 필터용 복사 |
| device_name | 필터용 복사 |
| equip_id | 필터용 복사 |
| lang | 필터용 복사 |
| tenant_id | 필터용 복사 |
| project_id | 필터용 복사 |
| chapter | 필터용 복사 |
| embedding | dense_vector, cosine similarity, HNSW |

**[STAT:n]** content 인덱스 필드 수: 23개 / embed 인덱스 필드 수: 11개

---

## [FINDING 3] doc_type별 콘텐츠 구조 및 청킹 방식

### 3-1. ts (Troubleshooting Guide) — PDF 기반

**파일:** `backend/services/ingest/document_ingest_service.py:410-420`

- **입력 형식:** PDF (VLM 또는 DeepDoc 파서)
- **섹션 분리 패턴:** `^[A-Z]-\d+\.\s*` (예: "A-1. 증상 설명")
- **청킹 단위:** 페이지별 섹션 (page_by_page, VLM 기준)
- **정규화 레벨:** L3 (`preprocess_semiconductor_domain`) — PM 모듈 마스킹, 도메인 synonym, 과학 표기 정규화
- **chapter 패턴:** `^([A-Z]-\d+(?:\.\d+)*\.?)\s*(.+)$` (metadata_extractor.py:72-74)
- **content 형식:** 자유 텍스트 (트러블슈팅 절차, 증상, 원인, 조치 기술)
- **`chapter` 필드:** carry-forward 방식. 첫 번째 `A-1.` 형태 헤딩 감지 후 하위 페이지에 전파

### 3-2. myservice (정비 보고서) — TXT 구조화 형식

**파일:** `backend/services/ingest/txt_parser.py:23-65`, `es_ingest_service.py:609-674`

- **입력 형식:** 구조화된 .txt 파일 (`>>>> [meta]` JSON + `[status]`/`[action]`/`[cause]`/`[result]` 섹션)
- **`[meta]` 주요 필드 (JSON):**
  - `Order No.` → `order_no` (extra_meta 저장 가능)
  - `Model Name` → `device_name`
  - `Equip. NO` → fallback `device_name`
  - `Title` → `doc_description`
- **섹션 분리:** status, action, cause, result — 각각 **별도 ES 문서**로 인덱싱
- **`chapter` 필드:** 섹션 이름 그대로 (`"status"`, `"action"`, `"cause"`, `"result"`)
- **content 형식:** 현장 정비 로그 (한국어 혼용, 장비 작동 상태/조치/원인/결과)
- **LLM 요약:** 섹션별 `chunk_summary` + `chunk_keywords` 생성 (선택적)
- **정규화:** L3 적용
- **`doc_id` 규칙:** `"myservice_{title}"` (파일명 기반)
- **언어:** `lang="ko"` (기본)

**myservice txt 파일 형식 예시:**
```
>>>> [meta]
{"Order No.": "WO-12345", "Model Name": "SUPRA XP", "Title": "PM 후 VACUUM ERROR"}

[status]
PM 완료 후 VACUUM ERROR 발생. PM2 pump down 불가.

[action]
PM2 ForeLine valve 교체 조치.

[cause]
ForeLine valve O-ring 마모로 leak 발생.

[result]
조치 후 VACUUM spec 정상 복귀.
```

### 3-3. gcb (Global Customer Bulletin) — TXT 구조화 형식

**파일:** `backend/services/ingest/gcb_parser.py:22-69`, `es_ingest_service.py:757-827`

- **입력 형식:** 구조화된 .txt 파일 (마크다운 유사 형식)
- **헤더 형식:** `### GCB {번호} | Status: {상태} | Model: {모델명} | Req:`
- **섹션 분리:** question, resolution — 각각 **별도 ES 문서**
- **메타 필드:**
  - `gcb_number` → common_meta (extra_meta에 저장 가능)
  - `model` → `device_name` (gcb_parser.py:56-57)
  - `title` → `doc_description`
  - Tags (`**Tags**: model=...; req=...; os=...; patch=...; module=...`) → key-value 파싱
- **`chapter` 필드:** `"question"` / `"resolution"`
- **content 형식:** 질문(문제/원인/초기 가설) + 확정 조치 (영문/한문 혼용)
- **embedding용 full_text 구조:**
  ```
  Title: {title}

  [question]
  {question text}

  [resolution]
  {resolution text}
  ```
- **언어:** `lang="en"` (기본)
- **`doc_id` 규칙:** 파일명 stem (예: `gcb_344`)

**gcb txt 파일 형식 예시:**
```
### GCB 344 | Status: Close | Model: TERA21 | Req:
**Title**: Chamber pressure instability after preventive maintenance

**Question (원인/문의/초기 가설)**:
After PM on PM2, chamber pressure shows instability...

**Confirmed Resolution (최종 확정 조치)**:
Replaced APC valve O-ring and recalibrated...

**Tags**: model=TERA21; req=; os=N/A; patch=N/A; module=CHAMBER
```

---

## [FINDING 4] 인덱스 Analyzer 설정

**파일:** `backend/llm_infrastructure/elasticsearch/mappings.py:241-254`

```yaml
analysis:
  analyzer:
    nori:
      type: custom
      tokenizer: nori_tokenizer
      filter: [nori_readingform, lowercase]
```

- `content` 필드: `nori` analyzer (한국어 형태소 분석)
- `search_text` 필드: `nori` analyzer
- `chunk_keywords` 서브필드 `.text`: `standard` analyzer (영문 키워드용)
- `embedding` 필드: cosine similarity, HNSW (m=16, ef_construction=100)

---

## [FINDING 5] 정규화 레벨 (L3~L5) — 모든 doc_type 공통 적용

**파일:** `backend/llm_infrastructure/preprocessing/normalize_engine/domain.py:22-77` (L3)

인제스트 파이프라인 Step 6에서 `get_normalizer(level="L3")` 고정 적용:

| 레벨 | 처리 내용 |
|---|---|
| L3 | Unicode NFKC, 대시 정규화, 도메인 synonym (FDC, vtis 등), PM 모듈 마스킹 (pm2 → PM), 과학 표기 정규화 |
| L4 | L3 + 모듈/알람 추출 → `[MODULE PM-2] [ALARM timeout_alarm]` 헤더 토큰 생성 |
| L5 | L4 + L5_VARIANT_MAP 동의어 사전, 단위 표기 표준화 (mTorr, sccm), 범위 표기 `[RANGE lo..hi]` |

**현재 인제스트 파이프라인은 L3 고정** (document_ingest_service.py:240).

---

## [FINDING 6] 검색 시 doc_type 필터링 방식

**파일:** `backend/services/es_chunk_v3_search_service.py:25-45`

```python
def _normalize_doc_types_to_v3_groups(doc_types):
    # variant → group_name(lowercase)로 변환
    # "Trouble Shooting Guide" → "ts"
    # "SOP/Manual" → "sop"
```

- 검색 시 variant 값이 들어와도 canonical group name으로 정규화 후 ES terms 필터 적용
- v3 인덱스의 `doc_type` 필드는 **lowercase canonical group name** 저장: `"ts"`, `"gcb"`, `"myservice"`, `"sop"`, `"setup"`

---

## [FINDING 7] 검색 프리셋 구성

**파일:** `backend/config/presets/retrieval_hybrid_rrf_v1.yaml`

기본 프리셋 (hybrid_rrf_v1):
- 임베딩: `bge_base` (768dims)
- 검색: dense 70% + BM25 30%, RRF k=60
- top_k: 50, final_top_k: 10
- multi_query: disabled
- reranker: disabled

full_pipeline 프리셋:
- 임베딩: `bge_large`
- 검색: hybrid + RRF
- multi_query: 4 variants + original
- reranker: cross-encoder enabled

---

## [LIMITATION]

1. **실제 데이터 미확인**: ES 인스턴스에 직접 접근하지 않았으므로 실제 인덱스에 저장된 `doc_type` 값이 canonical인지 variant인지 런타임 확인 불가.
2. **ts 후속 청킹 미확인**: ts PDF는 페이지 단위 섹션 분리 후 fixed_size chunker 추가 적용 여부를 `es_ingest_service.ingest_pdf` 전체 코드로 확인 필요.
3. **extra_meta 필터링 불가**: `gcb_number`, `order_no` 등 doc_type 고유 메타는 `extra_meta` (object, enabled=False)에 저장될 경우 ES 필터링/집계 불가.
4. **L3 고정 제약**: ts의 경우 L4/L5의 구조화 헤더 토큰([MODULE], [ALARM])이 미적용이므로, PM 주소/알람 기반 정밀 검색이 제한됨.
