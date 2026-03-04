# 2026-03-04 문서 유형별 Chunking/Embedding 적재 계획

## 1) 목적
- `chunk_v3` 파이프라인에서 문서 유형별 최적 chunking 전략을 정의한다.
- 임베딩 모델별 성능 비교가 가능하도록 적재 구조를 표준화한다.
- 과거 이슈였던 `page 누락`을 재발 방지하기 위한 검증/로그 태스크를 포함한다.

## 2) 데이터 소스 현황 (2026-03-04 기준)
루트: `/home/llm-share/datasets/pe_agent_data/pe_preprocess_data`

| 소스 | 경로 | 파일 수 | 포맷 | 비고 |
|---|---|---:|---|---|
| SOP | `sop_pdfs/` | 507 | pdf(335), pptx(172) | 신규 ppt 포함, VLM 필요 |
| Trouble Shooting | `ts_pdfs/` | 57 | pdf | VLM 필요 |
| Setup Manual | `set_up_manual/` | 14 | pdf | VLM 필요 |
| MyService | `myservice_txt/` | 99,031 | txt | `[meta/status/action/cause/result]` 구조 |
| GCB Raw | `gcb_raw/20260126/scraped_gcb.json` | 1 (json array) | json | 16,354건 케이스 |
| (참고) 기존 GCB 텍스트 | `gcb/` | 6,115 | txt | 이번 계획에서는 raw 우선 |

## 3) 전체 처리 순서 (수정 반영)
문서별 공통 처리 순서는 아래로 고정한다.

1. `VLM 필요 문서` 먼저 파싱
2. 문서 유형별 chunking
3. 임베딩 모델들로 embedding 생성
4. DB(ES) 적재

즉, 작업 순서는 항상 `VLM -> Chunk -> Embedding(s) -> DB`를 따른다.

## 4) 인덱스/저장 구조 (멀티 모델 평가용)
평가 목적 기준 권장 구조:

- `chunk_v3_content`
  - 원문 chunk + 메타데이터 저장
  - BM25/필터 검색 공용
- `chunk_v3_embed_{model}`
  - `chunk_id + embedding`만 저장
  - 모델별 독립 kNN 평가

장점:
- 원문 중복 저장 최소화
- 모델 추가/삭제가 인덱스 단위로 독립
- 차원이 다른 모델(768/1024/1536 등) 병행 가능

## 5) 문서 유형별 Chunking 전략

### A. SOP (`sop_pdfs/`: pdf + pptx)  [VLM 필요]
입력:
- PDF 페이지
- PPTX 슬라이드

전략:
- 파싱 단위: `page`(pdf), `slide`(pptx)
- chunk 경계: `절차 step/소제목` 우선
- 기본 길이: 350~700 tokens, overlap 60~100
- 표/체크리스트는 한 블록으로 유지 (분해 금지)
- 이미지 캡션/주의사항은 직전 step chunk에 병합

메타:
- `doc_type=sop`, `source_format=pdf|pptx`, `page_no|slide_no`, `section_title`, `language`

### B. Trouble Shooting (`ts_pdfs/`)  [VLM 필요]
전략:
- 파싱 단위: page
- `증상/원인/조치/결과` 패턴 기반 section chunk
- 오류코드/알람명은 chunk 선두에 보존
- 기본 길이: 300~650 tokens, overlap 50~80
- 로그 덤프는 별도 chunk(`chunk_subtype=log_excerpt`)로 분리

메타:
- `doc_type=trouble_shooting`, `alarm_code`, `model_name`, `page_no`

### C. Setup Manual (`set_up_manual/`)  [VLM 필요]
전략:
- 파싱 단위: page
- `장비/유닛/절차 단계` 중심으로 heading-aware chunking
- 조립/설정 순서는 순서 보존(`sequence_no`) 필수
- 기본 길이: 350~750 tokens, overlap 80
- 도해+본문 혼합 페이지는 설명문과 figure 설명을 묶어서 chunk 생성

메타:
- `doc_type=set_up_manual`, `chapter`, `page_no`, `equipment`

### D. MyService (`myservice_txt/`)  [텍스트 직처리]
입력 구조(관측):
- `[meta]` JSON
- `[status]`, `[action]`, `[cause]`, `[result]`

전략:
- 1차 분리: section 단위 (`status/action/cause/result`)
- 2차 분리: section이 길면 문장/항목 경계로 추가 split
- 권장:
  - `status`: 200~450 tokens
  - `action`: 300~700 tokens
  - `cause/result`: 80~250 tokens (가능하면 단일 chunk 유지)
- 기존처럼 section별 embedding 유지(질문 유형별 매칭 강화)

메타:
- `doc_type=myservice`
- `order_no`, `equip_id`, `model_name`, `reception_date`
- `section` (`status|action|cause|result`)
- `completeness`, `sections_present.*`

### E. GCB Raw (`gcb_raw/20260126/scraped_gcb.json`)  [JSON 직처리]
입력 구조(관측 키):
- `GCB_number`, `Status`, `Title`, `Model Name`, `Content`, `Request_Item2`, `Equip_ID`

전략:
- 레코드 단위 ingest 후 `Content`를 section-aware split
- 우선 분리 키워드:
  - `Description`, `Cause`, `Result`, `Background`, `Request`, 날짜/작성자 패턴
- 2계층 chunk 권장:
  - `summary chunk`: `Title + Request_Item2 + Status + Model Name`
  - `detail chunks`: 대화/로그 본문
- 기본 길이: 300~700 tokens, overlap 60
- 긴 타임라인 본문은 날짜 경계로 분할

메타:
- `doc_type=gcb`, `gcb_number`, `status`, `request_type`, `equip_id`, `model_name`

## 6) VLM 파싱 운영 기준
실행 기준(현재 compose):
- `docker compose --profile with-vlm up -d vlm`
- 준비 확인: `curl http://localhost:8004/v1/models`

파싱 산출물 표준 필드:
- `source_path`, `source_format`, `page_or_slide_no`
- `parsed_text`, `parse_status`, `parse_latency_ms`
- `token_count`, `char_count`

## 7) Page 누락 이슈 대응 (원인 가설 + 태스크)

### 7.1 주요 원인 가설
1. VLM 파싱 실패/타임아웃 페이지가 재시도 없이 누락
2. 빈 텍스트 필터(`min_chars`)에 의해 표/이미지 중심 페이지가 삭제
3. dedup(`content_hash`)가 연속 페이지를 중복으로 오인
4. 병렬 처리 시 page index 매핑 손실(순서/키 충돌)
5. PDF/PPT 페이지 수 집계와 파싱 결과 집계 기준 불일치

### 7.2 필수 로그/검증
문서 단위 로그:
- `expected_pages_or_slides`
- `parsed_pages_or_slides`
- `missing_pages` (배열)
- `coverage_ratio = parsed/expected`

페이지 단위 로그:
- `doc_id`, `page_no|slide_no`, `parse_status`
- `char_count`, `token_count`, `has_table`, `has_image`
- `latency_ms`, `error_type`, `retry_count`

단계별 카운트 로그:
- `raw_input_count -> parsed_unit_count -> chunk_count -> embedding_count -> indexed_count`

가드레일:
- `coverage_ratio < 0.98`이면 해당 문서 적재 중단 + 재처리 큐 이동
- `missing_pages`가 1장 이상이면 경고 이벤트 기록

## 8) 실행 태스크 (우선순위/순서)
1. VLM 파서에 `문서/페이지 커버리지 로그` 추가
2. SOP/TS/Setup(PDF/PPTX) VLM 파싱 실행
3. MyService section parser 정규화(`status/action/cause/result`)
4. GCB raw JSON 파서 구현 및 chunking 규칙 반영
5. 문서 타입별 chunker 파라미터 적용
6. 모델별 embedding 생성 (`chunk_v3_embed_{model}`)
7. `chunk_v3_content` + embed 인덱스 적재
8. 모델별 검색 평가(Recall@k, MRR, nDCG)
9. 누락 페이지 리포트 + 재처리 배치 실행

## 9) 완료 기준 (DoD)
- 모든 입력 문서에 대해 `expected vs parsed` 리포트 생성
- `missing_pages=0` 또는 재처리 근거 로그 보유
- 문서 타입별 chunk 품질 샘플 검수 완료
- 최소 2개 이상 embedding 모델 비교 리포트(Recall@k, MRR) 확보
- 인덱스 적재 건수와 로컬 산출물 건수 일치 검증 완료
