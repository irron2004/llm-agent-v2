# 문서 유형(doc_type)별 데이터 특성 및 Retrieval 고려사항

## 1. 개요

본 시스템은 반도체 PE(Process Engineering) 장비 문서를 검색·분석하는 RAG 에이전트이다. Elasticsearch(Nori 분석기) 기반 BM25 + Dense(BGE-M3) Hybrid Retrieval을 사용한다.

문서는 5가지 doc_type으로 분류되며, 각각 원본 형태·청킹 방식·검색 특성이 다르다.

현재 제품은 doc_type을 그대로 개별 route로 나누기보다, 상위 수준에서 아래 2개 업무 모드로 먼저 묶어 사용한다.

- `이슈조회(issue)`: `myservice + gcb + ts`
- `절차조회(sop)`: `setup + SOP`

이 상위 구분은 UX 관점에서는 단순하고 유효하지만, retrieval 관점에서는 같은 그룹 안의 문서 형태가 매우 다르다.
특히 `issue` 내부의 `myservice`, `gcb`, `ts`는 문서 구조와 검색 실패 패턴이 뚜렷하게 다르기 때문에,
retrieval 개선은 `issue` 전체를 한 덩어리로 보기보다 doc_type-aware profile로 접근하는 편이 더 적합하다.

### 1.1 현재 제품의 canonical doc_type 그룹

코드에서 사용하는 canonical group은 아래와 같다.

| canonical group | 대표 raw 값 | 현재 제품 의미 |
|-----------------|-------------|----------------|
| myservice | `myservice` | 현장 서비스/정비 이력 |
| gcb | `gcb`, `maintenance` | 글로벌 이슈 케이스/분석/커뮤니케이션 |
| ts | `Trouble Shooting Guide`, `ts`, `Guide` | 트러블슈팅 가이드 |
| setup | `Installation Manual`, `setup`, `Manual`, `set_up_manual` | 설치/셋업 매뉴얼 |
| SOP | `SOP`, `Global SOP`, `SOP/Manual`, `generic` | 표준 작업 절차 문서 |

실제 retrieval 설계에서는 raw `doc_type`보다 이 canonical group 단위로 정책을 잡는 것이 안전하다.

| doc_type | 원본 형태 | 문서 수 | 총 chunk 수 | 문서당 평균 chunk | chunk 평균 길이(chars) |
|----------|----------|---------|------------|-------------------|----------------------|
| myservice | 구조화 TXT (JSON 메타 + 4섹션) | 84,526 | 329,206 | 3.9 | 148 |
| gcb | 비구조 JSON (영문 커뮤니케이션 원문) | 13,848 | 49,021 | 3.5 | 919 |
| sop | PDF → VLM 파싱 Markdown | 384 | 13,116 | 34.2 | 879 |
| ts | PDF → VLM 파싱 Markdown | 79 | 760 | 9.6 | 748 |
| setup_manual | PDF → VLM 파싱 Markdown | 14 | (sop와 동일 파이프라인) | — | — |

---

## 2. doc_type별 상세 특성

### 2.1 myservice (현장 정비 이력)

**원본**: 현장 엔지니어가 작성한 정비 이력 리포트. JSON 메타데이터 + 4개 텍스트 섹션으로 구성.

**메타데이터**:
- `Model Name`: 장비 모델명 (예: SUPRA N)
- `Equip_ID`: 장비 호기 (예: 4ESPE101)
- `Order No.`, `Activity Type`, `Country`, `Reception Date`

**4개 섹션** (각각 별도 chunk로 분리됨):
| 섹션 | 설명 | chunk 수 | max_tokens | 내용 예시 |
|------|------|----------|------------|-----------|
| status | 증상/현상 기술 | 78,953 | 450 | "PM1 APC Position Abnormal 발생" |
| cause | 원인 분석 | 81,524 | 250 | "APC Sensor Cable 접촉 불량 확인" |
| action | 조치 내용 | 81,336 | 700 | "APC Sensor Cable 재결선, Calibration 수행" |
| result | 결과 | 87,393 | 250 | "정상 동작 확인, 모니터링 중" |

**청킹 방식**:
- 섹션별 `_word_window_split(max_tokens, overlap)` → content에 `[status]`, `[action]` 등 섹션 태그 prefix
- `search_text` = title + cause + status + 해당 섹션 텍스트 (검색 시 증상+원인 context 제공)

**Retrieval 특성**:
- chunk가 **매우 짧음** (median 58자) → 단독으로 의미 파악 어려움
- 동일 doc_id의 4개 섹션이 분리되어 있어 **섹션 병합**이 필요할 수 있음
- 전체 chunk의 **83.7%** (329K/393K)를 차지 → 검색 시 myservice에 편향됨
- `equip_id` 필드로 특정 호기 필터링 가능
- 한국어 + 영어 혼용, 반도체 전문 약어 다수

**Retrieval 과제**:
- myservice 편향: top-50 검색 결과의 83~100%가 myservice → 다른 doc_type이 밀림
- 짧은 chunk: 검색 의미 부족, 관련성 판단 어려움
- 동일 이슈의 여러 호기 이력이 중복 검색됨

---

### 2.2 gcb (Global Communication Board)

**원본**: 장비 이슈에 대한 글로벌 기술 지원 커뮤니케이션 쓰레드. 영문 원문이 많음.

**메타데이터**:
- `GCB_number`: GCB 고유 번호 (예: 68351)
- `Model Name`: 장비 모델
- `Equip_ID`: 장비 호기
- `Status`: Open/Close
- `Request_Item2`: 요청 유형
- `Title`: GCB 제목

**청킹 구조** (summary + detail 2-tier):

| chunk_tier | chapter 값 | 내용 | chunk 수 |
|------------|-----------|------|----------|
| summary | summary | `[GCB {번호}] {제목}\nRequest: {유형}\nStatus: {상태}\nModel: {모델}` | 16,228 |
| detail | detail | 본문 (커뮤니케이션 원문) | 16,485 |
| detail | timeline | 날짜 기반 이력 분할 | 15,259 |
| detail | description/cause/result/background | 섹션 헤더 기반 분할 | ~1,000 |

**청킹 방식**:
- summary: GCB 메타 정보를 조합한 짧은 헤더 chunk (avg ~100자)
- detail: `_split_gcb_sections()` → 섹션 헤더(description/cause/result/background) 기반 분할
- timeline: 날짜 패턴 감지 + 220단어 초과 시 자동 분할
- 각 섹션을 `_word_window_split(700, 60)`으로 재분할

**Retrieval 특성**:
- **영문 원문**이 대부분 → 한국어 질문으로 영문 content 검색 시 cross-lingual 문제
- summary chunk는 메타 정보만 포함, detail chunk에 실질적 기술 내용
- 커뮤니케이션 쓰레드 형식 → "Dear All, ..." 등 비기술적 문장 다수
- chunk 길이 편차 큼 (median 815자, max 926K자 — 일부 비정상 chunk 존재)
- timeline 섹션에 시간순 이슈 이력 포함

**Retrieval 과제**:
- 한국어 질문 ↔ 영문 content 매칭 (cross-lingual retrieval)
- summary chunk 검색 시 실질적 답변 정보 부족, detail chunk까지 확장 필요
- 커뮤니케이션 노이즈 (인사말, CC 목록 등) 가 BM25 점수 왜곡
- 핵심 원인/조치가 긴 쓰레드 중간에 묻혀 있음

---

### 2.3 sop (Standard Operating Procedure)

**원본**: 장비별 정비 절차서 PDF. VLM(Vision-Language Model)으로 파싱하여 Markdown으로 변환.

**메타데이터**:
- `device_name`: 장비 모델 (예: SUPRA N)
- `chapter`: 토픽/모듈 (예: "pin motor", "slot valve")
- `doc_id`: `global_sop_{device}_{work_type}_{topic}` 형식

**청킹 구조**:
- PDF 페이지 단위 → VLM 마크다운 변환 → 페이지별 chunk
- 282종의 chapter (토픽) 존재
- 문서당 평균 34.2 chunk (페이지 수에 비례)

**내용 구조** (일반적인 SOP 문서):
```
1. Safety (안전 수칙)
2. Safety Label (안전 라벨)
3. Required Tools & Materials (필요 공구)
4. Worker Location (작업자 위치)
5. Procedure (절차) ← 핵심 내용
6. Checksheet (점검표)
```

**Retrieval 특성**:
- chunk 길이 적절 (avg 879자, median 773자)
- 마크다운 포맷 (표, 리스트, 헤더) → BM25에서 구조적 토큰 노이즈
- 특정 토크 스펙, 파트 넘버 등 **정확한 수치** 포함
- `device_name`으로 장비 필터링 가능, `chapter`로 모듈 필터링 가능
- 문서 수 적음 (384개) → chunk 수도 13K로 전체의 3.3%

**Retrieval 과제**:
- 동일 topic이 여러 장비에 존재 (shared document) → 장비 scope 필터링 중요
- VLM 파싱 품질에 따라 표/도면 정보 누락 가능
- 안전 수칙, 공구 목록 등 반복 내용이 여러 문서에 존재 → 검색 시 noise
- 실제 절차(Procedure)가 중간~후반 페이지에 위치 → 앞쪽 페이지가 먼저 검색됨

---

### 2.4 ts (Trouble Shooting Guide)

**원본**: 장비별 트러블슈팅 가이드 PDF. VLM으로 파싱하여 Markdown으로 변환.

**메타데이터**:
- `device_name`: 장비 모델 또는 "ALL" (공통 가이드)
- `chapter`: 알람/증상 이름 (예: "ALL_FFU Abnormal", "leak rate over")
- `doc_id`: `{device}_all_trouble_shooting_guide_trace_{alarm}` 형식

**청킹 구조**:
- PDF 페이지 단위 → VLM 마크다운 변환 → 페이지별 chunk
- 71종의 chapter (알람/증상)
- 문서당 평균 9.6 chunk

**내용 구조** (일반적인 TS 문서):
```
1. 알람/증상 정의 (Trace Description)
2. 가능한 원인 (Possible Causes)
3. 점검 절차 (Check Procedure)  ← 핵심
4. 복구 절차 (Recovery Procedure) ← 핵심
5. 플로우 차트 (Decision Tree)
```

**Retrieval 특성**:
- 진단 절차형 문서 → **순차적 단계(step-by-step)** 가 핵심
- 알람 이름이 검색 키워드로 직결 (예: "FFU Abnormal" → 해당 TS 문서)
- chunk 수 매우 적음 (760개, 전체의 0.2%) → 검색 시 score 불리
- 영문+한국어 이중 언어 (가이드 본문 영문, 설명 한국어)
- 일부 `ALL` 장비 공통 TS는 여러 장비에 적용 가능

**Retrieval 과제**:
- 절대 수량 부족 → hybrid retrieval에서 myservice에 밀림
- 알람 코드/이름의 정확한 매칭이 중요 (BM25 유리)
- 플로우 차트/결정 트리가 VLM에서 텍스트로 완전 변환 안 될 수 있음
- 여러 페이지에 걸친 절차 → 단일 chunk로는 완전한 답변 불가

---

### 2.5 setup_manual (설비 매뉴얼)

**원본**: 장비 Setup/Operation 매뉴얼 PDF. VLM 파싱.

**규모**: 14개 문서 (매우 소량)

**특성**: sop와 동일한 파이프라인으로 처리. 장비 설치·초기 설정·운영 절차 포함. chunk 수가 매우 적어 검색에서 거의 등장하지 않음.

---

## 3. 검색 파이프라인 현황

```
Query → [Translate EN/KO] → [Multi-Query 확장 (3쿼리)]
     → [BM25 + Dense Hybrid (RRF)] → top-50 후보
     → [Cross-encoder Reranking] → top-20
     → [Section Expansion] → 최종 답변용 refs
```

### 3.1 현재 인덱스 구조

```
ES Index: chunk_v3_content (BM25, Nori 분석기)
ES Index: chunk_v3_embed_{model}_v1 (Dense 벡터)
```

- 두 인덱스 분리 저장 (content/vector)
- RRF(Reciprocal Rank Fusion)로 BM25 + Dense 결합
- Reranking: Cross-encoder 모델

### 3.2 필터링

- `device_name` 필터: 사용자가 장비 선택 시 적용
- `doc_type` 필터: task_mode에 따라 자동 적용 (sop→SOP만, ts→TS만)
- `equip_id` 필터: 특정 호기 지정 시 적용

### 3.3 현재 코드에 이미 반영된 doc_type별 확장 차이

현재 agent는 retrieval 이후의 expansion 단계에서 doc_type별로 다른 확장 전략을 일부 사용하고 있다.

- `myservice`, `gcb`, `pems`
  - 같은 `doc_id`의 관련 chunk를 다시 fetch하는 문서 단위 확장에 더 가깝다
- `SOP`, `setup`, manual 계열
  - 현재 hit page 기준 전후 page window를 확장하는 방식에 더 가깝다

이 점은 중요한 시사점을 준다.
즉, 현재 코드도 이미 “모든 문서를 같은 후처리 전략으로 다루면 안 된다”는 전제를 일부 채택하고 있다.
향후 retrieval 개선은 이 차이를 answer 단계의 우연한 후처리가 아니라, query generation, first-stage retrieval, reranking, ref selection까지 확장해서 보는 것이 맞다.

---

## 4. doc_type별 Retrieval 난이도 요약

| doc_type | 수량 | 검색 유리한 점 | 검색 불리한 점 | 난이도 |
|----------|------|---------------|---------------|--------|
| sop | 중 (13K chunks) | chunk 길이 적절, 구조적 | shared doc 구분, 위치 편향 | 중 |
| ts | 소 (760 chunks) | 알람명 매칭 직관적 | 수량 부족, myservice에 밀림 | 중-하 |
| myservice | 대 (329K chunks) | equip_id 필터 가능 | 극단적 짧은 chunk, 압도적 수량 | 상 |
| gcb | 대 (49K chunks) | 상세한 이슈 분석 | 영문 원문, 커뮤니케이션 노이즈 | 상 |
| setup_manual | 소 (14 docs) | sop와 유사 구조 | 수량 부족 | 하 |

---

## 5. 핵심 개선 과제

### 5.1 myservice 편향 해소
- 전체 chunk의 84%를 차지 → top-50 검색 시 myservice가 지배적
- doc_type별 retrieval quota 또는 boosting 필요

### 5.2 cross-lingual 매칭 (gcb)
- 한국어 질문 → 영문 gcb content 매칭
- 영문 multi-query 생성으로 일부 완화 중이나 불충분

### 5.3 짧은 chunk 문제 (myservice)
- median 58자 chunk → BM25/Dense 모두에서 의미 부족
- search_text에 title+cause+status를 붙여 보강 중이나, 근본적으로 동일 doc_id 섹션 병합 고려 필요

### 5.4 doc_type별 scoring 차별화
- 현재 모든 doc_type이 동일한 BM25 + Dense scoring
- doc_type별 특성에 맞는 가중치/boosting 전략 필요

### 5.5 section expansion 효과
- retrieve 후 동일 doc_id의 다른 섹션/페이지를 추가 fetch하는 로직 있음
- myservice: 4섹션 병합, gcb: summary→detail 확장, sop: 전후 페이지 확장

---

## 6. doc_type별 Retrieval Profile 제안

아래 표는 전문가가 retrieval profile을 별도로 설계할 때 바로 참고할 수 있는 요약이다.

| canonical group | 우선 검색 단위 | 핵심 query 신호 | 우선 확장 단위 | 주요 리스크 | 우선 검토 전략 |
|-----------------|----------------|------------------|----------------|-------------|----------------|
| myservice | chunk → doc bundle | alarm, symptom, equip_id, device exact match | same `doc_id` section merge | 대량 편중, 짧은 chunk, 중복 사례 | diversity cap, doc merge, exact match 강화 |
| gcb | chunk + doc bundle | issue keyword, error phrase, bilingual query | same `doc_id` summary/detail/timeline 묶음 | 영문 검색 손실, summary-only hit, 커뮤니케이션 노이즈 | bilingual MQ, section weighting, english field 검토 |
| ts | chapter/alarm anchor | symptom, alarm code, cause, check/reset/mitigation | 문서 또는 관련 절차 page 묶음 | corpus 소량, top-k 경쟁 열세 | low-volume boost, ts-aware rerank, issue용 MQ 강화 |
| setup | page/section | step, module, part, operation phrase | page window / section continuity | 앞페이지 편향, 주변 문맥 손실 | page-window, section-aware rerank |
| SOP | page/heading | work procedure, workflow, checklist, precaution | page window / heading block | 절차보다 반복 안전 문구가 먼저 걸림 | heading cue boost, procedure section 우선화 |

추가로 중요한 설계 판단은 아래와 같다.

- `myservice`
  - retrieval recall보다 post-retrieve diversity 제어가 더 큰 병목이 될 가능성이 높다.
- `gcb`
  - 1건의 정보량이 크므로, hit 자체보다 올바른 section을 가져오는 것이 더 중요하다.
- `ts`
  - 전체 수량은 적지만 hit되면 가치가 높아서, 일반 corpus와 동일 prior를 쓰면 손해를 보기 쉽다.
- `setup` / `SOP`
  - 절차형 문서라서 doc-level fetch보다 page/heading continuity가 더 중요하다.

---

## 7. 전문가에게 전달할 핵심 질문

retrieval 전문가에게는 아래 질문을 우선 던지는 것이 좋다.

1. `myservice` 편중을 줄이면서 `gcb`, `ts`를 안정적으로 top-k에 포함시키는 diversity/quota 전략은 무엇인가
2. `myservice`와 `gcb`는 chunk retrieval 이후 어느 단계에서 same-`doc_id` merge를 하는 것이 가장 효과적인가
3. `gcb` 영문 원문에 대해 현재 Nori 기반 BM25의 손실을 줄이기 위한 analyzer/sub-field 설계는 무엇이 적절한가
4. `ts`처럼 low-volume but high-value corpus를 hybrid retrieval에서 살리기 위한 boost/rerank prior는 무엇이 좋은가
5. `setup`과 `SOP`는 page/heading 중심 retrieval을 어떻게 분리 최적화할 수 있는가
6. query generation 자체를 `issue` 공통이 아니라 `myservice/gcb/ts`별 profile로 분기하는 것이 실제 성능 차이를 만들 가능성이 큰가

---

## 8. 전문가 전달용 한 줄 요약

- 현재 제품은 상위 수준에서 `issue(myservice/gcb/ts)` vs `sop(setup/SOP)`로 나뉜다.
- 하지만 retrieval 관점에서는 `myservice`, `gcb`, `ts`의 문서 구조와 실패 패턴이 매우 다르다.
- `myservice`는 대량·짧은 로그라 diversity와 section merge가 핵심이다.
- `gcb`는 영문 장문 케이스라 bilingual retrieval과 doc-level section 조합이 핵심이다.
- `ts`는 소량이지만 구조화된 절차형 문서라 low-volume boost와 ts-aware profile이 중요하다.
- `setup`과 `SOP`는 manual/절차 문서라 page-window와 heading-aware retrieval이 더 적합하다.

---

## 9. 평가 데이터

| 데이터셋 | 건수 | 설명 |
|----------|------|------|
| `data/eval_chatflow_unified.jsonl` | 1,206 | 통합 eval (sop 764, issue 298, ts 144) |
| `data/golden_set/queries_v2.jsonl` | 20 | 수동 golden set |
| `data/eval_sop_question_list_박진우_변형.csv` | 79 | PE 엔지니어 작성 SOP 질문 |
| `data/paper_a/eval/query_gold_master_v0_7_with_implicit.jsonl` | 1,206 | 전건 gold_doc_ids 포함 |

평가 지표: `doc_hit@5`, `doc_hit@10`, `route_accuracy`

평가 스크립트:
```bash
# 데이터셋 생성
python scripts/evaluation/generate_chatflow_eval_dataset.py

# 평가 실행 (direct mode, ES 필요)
python scripts/evaluation/evaluate_chatflow_unified.py \
  --input data/eval_chatflow_unified.jsonl \
  --out-dir data/eval_results/chatflow_$(date +%Y%m%d) \
  --mode direct --limit 20
```
