# 데이터 사양서

**과제명**: 반도체 PE 트러블슈팅을 위한 RAG 기반 지능형 에이전트 시스템 구축

| 항목 | 내용 |
|------|------|
| 작성일 | 2026-04-08 |
| 작성자 | PE RAG Agent 개발팀 |
| 버전 | v2.2 |

---

## PART 1. 데이터 사양서

### 1. 데이터 정보

| No. | 구분 | 데이터명 | 데이터유형 | 세부항목 | 수집설비/시스템 | 보유기간 | 형식 | 용량 | 식별키 |
|-----|------|----------|------------|---------|----------------|---------|------|------|--------|
| 1 | 운영 로그 데이터 | myservice (장비 유지보수 로그) | 비정형 텍스트 | `[meta]` JSON + `status/action/cause/result` 4개 섹션 유지보수 이력 | 사내 문서관리시스템(DMS) | 2019.03 ~ 현재 | TXT | 원본 99,031건 / chunk_v3 329,206건 | doc_id, device_name, equip_id |
| 2 | 기술 협의 기록 | gcb | 비정형 텍스트·JSON | 본사/현장 기술 협의 이력. raw JSON 기준 summary/detail/timeline chunking | 사내 문서관리시스템(DMS), 수집 JSON | 2019 ~ 현재 | JSON, TXT | raw 16,354건 / chunk_v3 49,021건 | gcb_number, doc_id, device_name, equip_id |
| 3 | 기술 문서 | trouble-shooting guide | 비정형 텍스트 | 증상·원인·점검·복구 절차 중심 트러블슈팅 문서 | 장비 제조사 제공 | 제조사 발행일 ~ 현재 | PDF | 원본 57건 / chunk_v3 760건 | doc_id, device_name, chapter |
| 4 | 기술 문서 | setup-manual | 비정형 텍스트 | 설치 절차, 초기 셋업, 장비/유닛별 절차 설명 | 장비 제조사 제공 | 제조사 발행일 ~ 현재 | PDF, PPTX | 원본 14건 / chunk_v3 3,583건 | doc_id, device_name, chapter |
| 5 | 기술 문서 | SOP | 비정형 텍스트 | 표준 작업 절차, 안전 수칙, 점검 항목, 작업 순서 | 사내 DMS / 장비 제조사 | 제조사 발행일 ~ 현재 | PDF, PPTX | 원본 507건 / chunk_v3 13,116건 | doc_id, device_name, chapter |
| 6 | 기술 보고서 | PEMS | 비정형 텍스트 | 공정 엔지니어링 분석·검증 보고서 (corpus 적재 완료, 라우팅 미연결 — 향후 확장 대상) | 사내 연구팀 | 2017 ~ 현재 | PDF | 원본 8건 / chunk_v3 87건 | doc_id, device_name |

> 주: 위 표는 **원본 수집 단위**와 **현재 chunk_v3 retrieval corpus 활용량**을 함께 적었다. raw 파일 수와 인덱싱 후 chunk 수는 서로 다른 단계의 집계다.

### 2. 데이터 품질 정보

| No. | 결측률 | 이상치율 | 라벨 보유 여부 |
|-----|--------|---------|--------------|
| 1 | `empty` 문서는 존재하나 retrieval 적재 기준 빈 콘텐츠 0% | cause/status가 짧고 섹션 누락 가능 | 부분 보유 — section 구조, Activity Type, Equip_ID |
| 2 | 정량 결측률 별도 집계 없음 | 영문 timeline 노이즈, parser 구조 불일치 가능 | 부분 보유 — status, request_type, gcb_number |
| 3 | 정량 결측률 별도 집계 없음 | PDF/VLM 파싱 품질, 알람·표 구조 보존 이슈 가능 | 부분 보유 — symptom/cause/action 구조 내재 |
| 4 | 정량 결측률 별도 집계 없음 | PDF/PPTX 혼합 페이지 구조 손실 가능 | 미보유 — 절차 안내용 |
| 5 | 정량 결측률 별도 집계 없음 | 표지/목차 page bias, OCR/VLM 품질 편차 가능 | 미보유 — 절차 안내용 |
| 6 | 정량 결측률 별도 집계 없음 | 자유 형식 보고서 특성상 문맥 손실 가능 | 부분 보유 — 분석/비교 근거 포함 |

> ※ 저장소 기준으로 **확정 가능한 값만 기재**하였다. 숫자형 결측률·이상치율 통계가 별도로 산출되지 않은 항목은 정성 설명으로 대신했다.

### 3. AI 활용 계획

| AI 활용 목적 | 적용공정 | 예측 항목 | 데이터 유형 | 제출데이터 매칭 |
|-------------|----------|----------|------------|---------------|
| 장비 트러블슈팅 지원: 증상 입력 시 `myservice`, `gcb`, `ts`를 통합 검색·요약하여 원인 후보, 우선 점검 항목, 조치 방향을 제시하는 지능형 진단 지원 | 반도체 PE 장비 유지보수 (BM/PM, 반복 장애 분석) | Hit@k, MRR, Recall@k, Contamination@k, 답변 만족도, 사례 선택 적합성 | 비정형 텍스트 (운영 로그, 협의 기록, 트러블슈팅 가이드) | 1, 2, 3 |
| 절차 안내 지원: 설치·셋업·정비 질의에 대해 setup-manual/SOP 기반 단계별 절차, 주의사항, 확인 포인트를 즉시 제시 | 신규 장비 설치, 이설, PM 절차 수행 | page hit, 절차 문서 매칭 정확도, page bias 감소, 절차 응답 일관성 | 비정형 텍스트 (설치 매뉴얼, SOP) | 4, 5 |
| 엔지니어 의사결정 지원: retrieved evidence 기반으로 점검 순서, 설정 확인 포인트, 조정 필요 항목의 근거를 제시하고 향후 제어 추천형 에이전트로 고도화 가능한 기반 확보 | 장비 유지보수 전반, 미해결 이슈 재발 분석 | 근거 문서 일치도, 유사 사례 매칭, 후속 피드백 점수, 해결 여부 추적, 제어 지원 확장성 | 비정형 텍스트 + 운영 피드백 | 1, 2, 3, 4, 5, 6 |

> 현재 구현 범위는 **문서 검색·근거 제시·의사결정 지원**이다. 제어 파라미터를 장비에 직접 적용하는 closed-loop 제어는 저장소 구현 범위에 포함되지 않는다.

---

## PART 2. AI학습데이터 사양서

### 1. AI 모델명

**반도체 PE 트러블슈팅 RAG 기반 의사결정 지원 에이전트 (Semiconductor PE Troubleshooting RAG Agent)**

#### 1) 문제 정의

**○ 문제 개요**

반도체 PE 현장에서는 장애 발생 시 유지보수 로그, 협의 이력, 트러블슈팅 가이드, SOP/Setup 문서를 수동 탐색해야 하므로 원인 파악과 조치 결정에 시간이 많이 든다. 현재 구현은 LangGraph 기반 RAG 에이전트가 질의에 맞는 문서를 검색하고 issue-summary / case-selection / detail-answer / procedure-guidance 흐름으로 근거를 제시해 엔지니어 판단을 지원하는 구조다. 본 과제는 이 기반을 바탕으로 **근거 기반 진단·절차 안내·제어 의사결정 지원**을 단계적으로 고도화하는 것을 목표로 한다.

**○ 공정 개요**

대상 영역은 반도체 PE 장비 유지보수와 절차 안내다. BM/PM, 장비 설치·이설, 반복 장애 재발 분석, 고장 진단 가이드 탐색이 주요 시나리오다. 제품 UX는 `이슈조회(issue)`와 `절차조회(sop)` 두 흐름으로 나뉘며, issue는 `myservice + gcb + ts`, procedure는 `setup + SOP`를 중심으로 동작한다.

**○ 데이터 개요**

현재 구현에서 실제 활용되는 데이터는 4개 층으로 구분되며, 향후 학습형 고도화와 운영형 피드백 루프의 기반으로 확장 가능하다.

1. **원본 문서 코퍼스**: MyService TXT, GCB raw JSON, TS/Setup/SOP/PEMS PDF·PPTX
2. **chunk_v3 retrieval corpus**: 문서 유형별 chunking 후 Elasticsearch에 적재되는 검색용 corpus (`all_chunks.jsonl` 기준 395,773 chunks)
3. **평가 데이터셋**: `experiments/`, `scripts/evaluation/`, `docs/papers/20_paper_a_scope/evidence/*`에서 관리하는 Hit@k/MRR/Contamination 평가셋과 결과
4. **운영 피드백 데이터**: `chat_turns`의 간단 만족도와 `feedback` index의 상세 점수(accuracy/completeness/relevance) 및 export 경로

저장소에는 feedback export와 future fine-tuning data collection 경로가 이미 마련되어 있다. 다만 RLHF 학습 파이프라인이나 retriever fine-tuning 실행 코드는 현재 범위에 포함되지 않으며, 향후 고도화 단계에서 연계 가능한 형태로 정리되어 있다.

**○ 기대 효과**

현재 저장소에서 직접 검증·관리하는 효과는 다음과 같다.

| No. | 기대효과 | 정량적/운영 지표 |
|-----|---------|------------------|
| 1 | 검색 품질 고도화 | Hit@k, MRR, Recall@k, NDCG, page hit |
| 2 | 장비 간 문서 혼입 통제 | Raw/Adjusted Contamination@k, ContamExist@k, ScopeAccuracy |
| 3 | 절차 문서 접근성 향상 | page bias 감소, SOP/setup section expansion 검증 |
| 4 | 운영 피드백 폐루프 확보 | 만족/불만족, 상세 3점 척도, 해결 여부 추적 |
| 5 | 엔지니어 의사결정 지원 고도화 | 사례 비교, 절차 제시, 점검/조정 근거 제공 |

#### 2) AI 모델 학습데이터 설명

**○ 데이터 유형:** 반도체 장비 유지보수 비정형 문서 6종 + 검색용 chunk corpus + 평가셋 + 운영 피드백 데이터

**○ 데이터 개수 및 크기:**

- 원본 문서 소스(2026-03-04 snapshot): MyService 99,031 TXT, GCB raw 16,354 entries, TS 57 PDF, Setup 14 PDF, SOP 507 PDF/PPTX, PEMS 8 PDF
- chunk_v3 retrieval corpus(`all_chunks.jsonl` 집계 기준): **395,773 chunks**
  - myservice 329,206 / gcb 49,021 / sop 13,116 / setup 3,583 / ts 760 / pems 87
- Paper A scope-routing 실험 코퍼스: **578 parsed docs**, 27 devices, 419 topics
- 운영 피드백 데이터: `chat_turns` + `feedback_{env}_current` 인덱스에 저장되며, 건수는 운영 시점별 변동

**○ 수집 기간:** 문서 코퍼스는 2017년~현재 범위의 사내/제조사 문서를 포함하며, 운영 대화/피드백 데이터는 PoC 운영 시점 이후 누적된다.

**○ 수집 장소:** 사내 문서관리시스템(DMS), 장비 제조사 제공 문서, 사내 연구팀 보고서, 운영 채팅/피드백 인터페이스

**○ 수집 대상 설비 :**

**[데이터 수집 대상설비 정보]**

| 설비 번호 | 설비명 | 제원 및 성능 | 제조사 |
|----------|--------|-------------|--------|
| ES-001 | Elasticsearch 검색 엔진 | Nori analyzer 기반 BM25 + dense vector 검색. v2 alias, v3 split index 구조 사용 | Elastic |
| LLM-001 | LLM 추론 서버 (Ollama / vLLM) | Ollama 기본 `qwen2.5:14b`, vLLM 대체 경로 지원 | Ollama / 자체 운영 |
| VLM-001 | 문서 파싱용 VLM/DeepDoc 파이프라인 | PDF/PPTX → Markdown 파싱, page/heading 기반 section 추출 | 자체 운영 + OpenAI-compatible VLM |
| ST-001 | MinIO 오브젝트 스토리지 | 문서 페이지 이미지 저장용 S3 호환 저장소 | MinIO |

**[데이터 수집 내역]**

| 연도 | 1월 | 2월 | 3월 | 4월 | 5월 | 6월 | 7월 | 8월 | 9월 | 10월 | 11월 | 12월 | 계 |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|------|------|----|
| 2019 |  |  | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | myservice 운영 로그 축적 시작 |
| 2020 | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | myservice 로그 지속 수집 |
| 2021 | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | myservice 로그 지속 수집 |
| 2022 | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | myservice 로그 지속 수집 |
| 2023 | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | myservice 로그 지속 수집 |
| 2024 | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | 유지보수 로그 + 기술 문서 누적 |
| 2025 | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |  |  | GCB raw/TS/Setup/SOP 코퍼스 정제 및 retrieval 실험 데이터 구축 |
| 2026 |  |  | ● |  |  |  |  |  |  |  |  |  | Paper A 실험 코퍼스 정리, feedback/resolution 추적 경로 운영 |
| 계 |  |  |  |  |  |  |  |  |  |  |  |  | 문서 코퍼스 + 평가셋 + 운영 피드백 데이터 지속 축적 |

**[학습 데이터 요약]**

| 데이터 구분 | 주요 변수 | 데이터 종류 | 수집 주기 | 수집 방식 | 데이터 크기(기간/샘플 수) |
|------------|----------|------------|----------|----------|--------------------------|
| 원본 문서 | source_file, doc_type, device_name, equip_id, page/chapter | 비정형 문서 + 메타데이터 | 비정기 (문서 등록/정제 시) | DMS/제조사 문서 수집 → VLM/텍스트 파싱 | 문서 6종, raw source 기준 수만 건 |
| retrieval corpus | chunk_id, doc_id, content, search_text, embedding, chapter | 텍스트 chunk + 메타데이터 + 벡터 | 인덱싱 시 자동 생성 | chunk_v3 파이프라인(`run_chunking.py` → `run_embedding.py` → `run_ingest.py`) | `all_chunks.jsonl` 기준 395,773 chunks |
| 평가 데이터 | query, expected_doc/pages, scope labels, contamination labels | 질의-정답/평가셋 | 실험 셋 구성 시 | `experiments/`, `scripts/paper_a/`, `docs/papers/.../evidence` | Paper A 578 parsed-doc corpus 기반 평가셋 |
| 대화 이력 | session_id, turn_id, user_text, assistant_text, metadata | 대화 로그 | 운영 중 상시 | `/api/agent/run`, `/api/conversations/*` 경로 저장 | 운영 시점에 따라 변동 |
| 상세 피드백 | accuracy, completeness, relevance, rating, comment, reviewer_name, resolved | 상세 평가 데이터 | 운영 중 상시 | `/api/feedback/*` 저장 및 JSON/CSV export | 운영 시점에 따라 변동 |

#### 3) 변수 세부 명세

| 데이터 구분 | 변수명 | 변수 설명 | 단위 | 데이터 종류 | 데이터 타입 | 수집 주기 | 사용 여부(Input/Target) | 설명 |
|------------|--------|----------|------|------------|------------|----------|------------------------|------|
| retrieval corpus | query | 엔지니어 질의문 | 텍스트 | 비정형 텍스트 | text | 요청 시 | Input | issue/sop route 및 retrieval의 시작 입력 |
| retrieval corpus | content | chunk 본문 텍스트 | 텍스트 | 비정형 텍스트 | text | 인덱싱 시 | Input | 답변 근거로 사용되는 본문 |
| retrieval corpus | search_text | 검색 최적화 텍스트 | 텍스트 | 비정형 텍스트 | text | 인덱싱 시 | Input | BM25 검색용 통합 텍스트 |
| retrieval corpus | embedding | dense retrieval 벡터 | 768d/1024d | 수치 벡터 | dense_vector / numpy array | 인덱싱 시 | Input | embedding model별 별도 embed index 구성 |
| retrieval corpus | doc_type | canonical 문서 그룹 | — | 범주형 | keyword | 인덱싱 시 | Input | `myservice`, `gcb`, `ts`, `setup`, `sop` (활성 라우팅 대상). `pems`는 corpus에 적재되어 있으나 현재 라우팅 미포함 |
| retrieval corpus | device_name | 장비명 | — | 범주형 | keyword | 인덱싱 시 | Input | 장비 scope/filtering 및 contamination 평가에 사용 |
| retrieval corpus | equip_id | 설비 호기 식별자 | — | 식별자 | keyword | 인덱싱 시 | Input | 특정 설비 이력 필터링 |
| retrieval corpus | chapter / section_chapter | 챕터/섹션 정보 | — | 범주형 | keyword | 인덱싱 시 | Input | section/page expansion 및 절차 문서 탐색에 사용 |
| feedback | accuracy / completeness / relevance | 상세 피드백 3점 척도 | 1~5 | 정수형 평점 | integer | 피드백 제출 시 | Target(운영평가) | 상세 사용자 평가 저장 |
| feedback | rating / avg_score | 단순 만족도 및 평균 점수 | up/down, 1~5 | 범주형 + 수치 | keyword / float | 피드백 제출 시 | Target(운영평가) | 대화별 만족/불만족 및 요약 지표 |
| feedback | resolved / resolved_link | 해결 여부 및 재확인 링크 | boolean / URL path | 상태값 | boolean / keyword | 해소 처리 시 | Target(운영관리) | 불만족 피드백의 후속 조치 추적 |

#### 4) 전처리 데이터 명세

**[전처리 후 데이터 변수 정보]**

| Feature Name | 설명 | 단위 | 데이터 타입 | Scaling/Encoding | 데이터 종류 | 학습 | 검증 | 테스트 |
|-------------|------|------|------------|------------------|------------|------|------|--------|
| content | 전처리 및 chunking 후 본문 텍스트 | 텍스트 | text | normalize(L3) + doc_type별 chunking | 비정형 텍스트 | — | — | — |
| search_text | BM25 검색용 통합 텍스트 | 텍스트 | text | 메타/제목/section-aware 조합 | 비정형 텍스트 | — | — | — |
| embedding | embedding model별 dense vector | 모델별 dims | dense_vector / np.ndarray | L2 normalize | 수치 벡터 | — | — | — |
| device_name | 장비명 표준화 값 | 범주형 | keyword | 공백/underscore/uppercase 정규화 | 메타데이터 | — | — | — |
| doc_type | canonical doc type | 범주형 | keyword | `myservice/gcb/ts/setup/sop` group mapping (`pems`는 corpus 적재만, 라우팅 미포함) | 메타데이터 | — | — | — |
| chapter / section metadata | 절차/섹션 연속성 정보 | 범주형/정수 | keyword / integer | heading carry-forward / section extraction | 메타데이터 | — | — | — |

> 본 저장소는 **운영 retrieval corpus**와 **실험 평가셋** 중심이다. 일반적인 70/20/10 학습 분할 기반 supervised training pipeline은 현재 구현 범위가 아니다.

#### 5) AI 모델 설명(AI Model summary)

| 항목 | 내용 |
|------|------|
| 모델명 | 반도체 PE 트러블슈팅 RAG 기반 의사결정 지원 에이전트 |
| 알고리즘 | LangGraph + Elasticsearch hybrid retrieval(BM25 + dense + RRF) + 선택적 reranker + issue/sop route 분기 |
| 입력 변수 | 질의문, 선택적 장비/doc_type 조건, indexed chunk(`content`, `search_text`, `device_name`, `doc_type`, `chapter`), embedding |
| 출력 변수 | 근거 문서 목록, issue summary / case selection / detail answer / 절차 안내, 운영 피드백 및 해결 여부 추적 |
| 학습 목표 | 현재 단계는 **retrieval corpus 구축, 검색 평가, 운영 피드백 수집 체계 확보**에 집중한다. 향후에는 feedback export와 평가셋을 활용한 fine-tuning·정책 고도화로 확장 가능하도록 설계되어 있다 |
| 성능지표 | Hit@k, Recall@k, MRR, NDCG, latency, feedback statistics, Paper A evidence의 Contamination@k / ScopeAccuracy, 향후 제어 지원 적합도 평가 확장 가능 |
| 프레임워크 | FastAPI, LangGraph, Elasticsearch 8.x, React, Ollama / vLLM, VLM parser/DeepDoc, MinIO |
| 학습 환경 | 현 단계의 인덱싱/실험은 Python 스크립트 기반이며, 추론은 Ollama 또는 vLLM 경로를 지원한다. 구조상 향후 supervised training 및 feedback-driven optimization으로 확장 가능하다 |
| 학습 조건 (하이퍼파라미터) | 설정 기준 search service 기본값은 dense_weight=0.7, sparse_weight=0.3, RRF k=60, temperature=0.7, max_tokens=30000. 실제 agent path는 일부 값을 override할 수 있음 |
| 개발 환경 | Python, FastAPI, LangGraph, Elasticsearch, SentenceTransformer 계열 embedding, React/TypeScript, Docker Compose |
| 모델 기타사항 | feedback index와 export는 구현되어 있고, 이를 기반으로 향후 RLHF·보상모델·제어 지원형 고도화로 확장할 수 있다. 현재 구현은 **근거 기반 검색·요약·의사결정 지원 시스템**이다 |

---

> 본 문서는 `docs/papers/20_paper_a_scope/data_specification.md`의 기존 초안을 **현재 저장소 구현 기준으로 정정**한 버전이다. raw 데이터 보유기간·운영 시작 시점 같은 운영 정보는 내부 기준에 따라 추가 조정할 수 있으나, 검색 파이프라인·chunking·평가·피드백 관련 설명은 모두 현행 repo evidence와 snapshot 문서(예: 2026-03-04, 2026-03-14 기준)에 맞춰 작성했다.
