# Multi-Query(MQ) 전략 보고서

## 1. ES 데이터 현황

### 1.1 문서 종류별 통계

| doc_type 그룹 | chunks | unique docs | lang | 검색 방식 |
|---|---|---|---|---|
| myservice | 324,482 | 83,719 | ko | hybrid (vector + nori BM25) |
| SOP | 11,452 | 339 | ko | hybrid |
| gcb | 7,264 | 3,639 | **en** | hybrid |
| setup (set_up_manual 계열) | 2,045 | ~100 | ko | hybrid |
| ts (trouble_shooting_guide 계열) | 688 | 77 | ko | hybrid |

> 기준: 보고서 내 통계는 `myservice/SOP/ts/setup/gcb` 그룹 기준으로 정리

### 1.2 myservice 문서 구조 (전체의 94%)

하나의 myservice 문서(doc_id)는 **4개 chapter**로 구성:

```
doc_id: 40015778 (장비 정비 기록 1건)
├── status  : "Run中 TM Robot Communication Alarm, Controller Error Code: 6"
├── cause   : "TM Robot Controller Reset → Servo Power On Fail"
├── action  : "TM Robot Controller Reset → Servo Power On → GUI Home OK → Wafer Clear"
└── result  : "Buzz5000 Alarm Check → Realtime Error Exists, Axis 2: Clamp Error"
```

**content 길이 분포 (100개 샘플 분석)**:

| chapter | median | mean | <50자 비율 | 50-200자 | >200자 |
|---|---|---|---|---|---|
| status | **36자** | 45자 | **79%** | 20% | 1% |
| cause | ~30자 | ~35자 | ~80% | ~18% | ~2% |
| action | **176자** | 249자 | 17% | 43% | **40%** |
| result | ~50자 | ~80자 | ~50% | ~35% | ~15% |

**핵심**: status/cause는 대부분 50자 미만의 **단문 메모** 형식.

### 1.3 myservice 용어 특성

```
status 상위 키워드: PM1, PM2, PM3, FCIP, CH1, CH2, alarm, power drop, LP1, Leak, TM Robot
cause  상위 키워드: root cause, customer request, FCIP, power drop, STD alarm, vacuum leak
action 상위 키워드: leak check, backup, FCIP replacement, aging test, calibration, cleaning
```

**특징**:
- 한/영 혼용: "-. PM1 CH2 Source Power 3800W→3815W수준 캘 요청"
- 약어 다수: PM, CH, FCIP, LP, TM, CTC, BM, SV
- 수치 포함: 3800W, 4400W, Error Code 6, Code 252
- 줄바꿈 구분: "-." 또는 "->" 패턴

### 1.4 SOP 문서 구조

```
doc_id: global_sop_supra_n_series_all_efem_controller  ← doc_id 자체가 검색 키워드
content: "Global SOP_SUPRA N series_ALL_EFEM_CONTROLLER ... Scope: EFEM CONTROLLER 관련 작업 절차..."
chapter: "2. Safety Label"
page: 1~40
```

- doc_id에 **장비명 + 대상 부품**이 포함 (검색에 유리)
- content는 마크다운 형식, 400-700자/chunk
- page 기반 구조 (목차 → 안전 → 절차)

### 1.5 GCB 문서 구조

```
doc_id: GCB_10118
├── question  : "timeout alarm during undocking of load port 2..."  (영문)
└── resolution: "resolved by updating software to version se-ver4.32.1..."  (영문)
```

- **전부 영문** (lang: en)
- Q&A 쌍 (문서당 정확히 2 chunks)
- 기술 문의/해결 사례 기록

### 1.6 Trouble Shooting Guide

```
doc_id: supra_n_all_trouble_shooting_guide_trace_microwave_abnormal  ← 알람명 포함
content: "Troubleshooting Guide - PM WAFERS EXCEED TIME LIMIT..."
```

- doc_id에 **알람/증상명**이 포함
- PDF 기반, VLM으로 추출된 마크다운
- 특정 알람별 대응 절차

---

## 2. 현재 MQ 프롬프트의 문제점

### 2.1 general_mq: "overview, summary" 키워드 추가 전략의 비효과

```yaml
Q3: Add document structure keywords (overview, summary, table of contents)
```

**문제**: myservice는 "overview", "summary" 같은 단어가 없음. 대부분 "-. 3800W→3815W 캘 요청" 같은 단문.
이런 확장은 BM25 스코어를 희석시키고, 벡터 검색에서도 의미가 달라짐.

### 2.2 ts_mq: 하드웨어 키워드 추가가 너무 일반적

```yaml
Q2: Add hardware keywords (PLC, sensor, valve, circuit, signal)
```

**문제**: 실제 데이터의 키워드는 `PM1`, `CH2`, `FCIP`, `TM Robot` 등 매우 구체적. "PLC, sensor, valve"는 너무 일반적이어서 노이즈만 증가.

### 2.3 프롬프트 언어 규칙과 실행 로직의 충돌

프롬프트에는 아래 규칙이 명시됨:
```yaml
- Use the same language as the input question
```

**문제**: 실제 실행 코드는 EN/KO 이중 MQ를 생성하는데, 프롬프트 지시는 단일 언어를 요구함.
이 충돌로 모델 출력이 불안정해지고(메타 문구/placeholder/중복 query), 검색 성공률이 흔들릴 수 있음.

### 2.4 myservice 단문에 긴 쿼리가 비효과적

myservice status/cause의 median이 36자인데, MQ가 "PM 점검 주기 preventive maintenance 예방 보전 정기 점검 주기 스케줄" 같은 긴 쿼리를 생성하면:
- BM25: 불일치 토큰이 많아 점수 하락
- Vector: 의미는 비슷할 수 있지만, 짧은 원문 대비 과도한 확장

### 2.5 chapter 구조를 활용하지 않음

사용자 의도에 따라 검색해야 할 chapter가 다름:
- "어떤 에러야?" → status
- "원인이 뭐야?" → cause
- "어떻게 조치했어?" → action
- "결과가 어떻게 됐어?" → result

현재 MQ는 chapter를 전혀 고려하지 않음.

### 2.6 동일 질문 raw1/raw2 사례로 본 Multi-Query 문제

동일 질문:
`fcip source 교체 후 power cal하는 방법`

아래는 실제 `search_queries` 비교에서 확인된 핵심 차이:

| 관측 포인트 | raw1 | raw2 | 문제 유형 | 검색 영향 |
|---|---|---|---|---|
| Query #1 품질 | `We can ask: "What model and firmware version..."` | `How to perform power cal after replacing the` | 메타 문장 유입 + 미완성 문장 | BM25/벡터 모두 잡음 증가 |
| Query #2~#3 성격 | 확인 질문형 문장 포함 (`What are...`, `Which...`) | 확인 질문형 문장 포함 | 검색 쿼리와 대화용 질문 미분리 | 문서 매칭보다 질의응답형 문장 검색됨 |
| 수치/조건 | `5W 3A` | `12V 50Hz` | 사용자 질문에 없는 파라미터 환각 삽입 | 잘못된 조건으로 recall/precision 저하 |
| 한국어 쿼리 | 핵심 질문 + 변형 포함 | 핵심 질문 + 변형 포함 | 일부는 유효하나 중복 높음 | 쿼리 다양성 대비 실제 정보량 낮음 |
| 전체 일관성 | 같은 질문인데 쿼리 의미축이 크게 변동 | 같은 질문인데 쿼리 의미축이 크게 변동 | 비결정성(출력 흔들림) | 재현성 저하, 답변 품질 편차 발생 |

정리하면, 이 케이스의 MQ 문제는 **"확장 부족"보다 "확장 오염"**이 더 큼:

1. 검색용 키워드 대신 대화형 문장(clarifying question)이 MQ에 섞임  
2. 원문에 없는 수치/스펙을 생성해 잘못된 필터링을 유도  
3. 문장 완결성이 깨진 query(절단 문장)가 생성됨  
4. 6개를 만들지만 실질적으로는 유사 query 반복으로 다양성이 부족  

즉, 동일 질문의 성공/실패 편차는 retrieval 파이프라인 이전에 **MQ 품질 게이트 부재**에서 먼저 발생하고 있음.

---

## 3. MQ 전략 제안

### 3.1 전략 1: 짧은 키워드 쿼리 + 원문 용어 보존

myservice 데이터 특성에 맞게, 긴 자연어 확장 대신 **핵심 키워드 조합**으로 쿼리 생성.

```
현재: "SUPRA N TM Robot Communication Alarm troubleshooting root cause checklist"
제안: "TM Robot Communication Alarm Controller Error"
```

**원칙**:
- 원문에 실제로 등장하는 용어를 최대한 유지
- 동의어 확장보다 **에러코드, 부품명, 수치** 유지가 중요
- 한 쿼리당 키워드 3-5개로 제한

### 3.2 전략 2: 이중 언어 쿼리 생성 (한/영)

GCB가 전부 영문이므로, **모든 MQ에 영문 쿼리 1개 이상** 포함:

```
사용자: "SUPRA N timeout alarm 원인"
MQ1 (ko): "SUPRA N timeout alarm"
MQ2 (en): "SUPRA N timeout alarm cause resolution"
MQ3 (ko): "SUPRA N timeout 알람 원인 조치"
```

현재 st_mq에서 EN/KO 쿼리를 조합하는 로직이 있지만, MQ 생성 단계에서부터 이중 언어를 의식하면 더 효과적.

### 3.3 전략 3: chapter-aware 쿼리 전략

사용자 질문의 의도에 따라 chapter 키워드를 쿼리에 포함하거나, chapter별 가중치 차별화:

| 사용자 의도 | 우선 chapter | 쿼리 전략 |
|---|---|---|
| 증상/현상 문의 | status | 에러코드, 알람명 그대로 |
| 원인 분석 | cause, status | "원인", "root cause" + 증상 |
| 조치 방법 | action | "교체", "replacement", "calibration" + 부품명 |
| 결과 확인 | result | "완료", "release", SOP 참조 |
| 절차 문의 | SOP action 챕터 | 부품명 + "절차", "procedure" |

### 3.4 전략 4: doc_type별 분리 쿼리

현재는 모든 doc_type에 동일한 쿼리를 사용. doc_type별 특성이 다르므로 분리 전략:

```
사용자: "SUPRA N EFEM robot alarm"

MQ for myservice: "SUPRA N EFEM Robot alarm"       ← 짧고 직접적
MQ for SOP:       "SUPRA N EFEM robot"              ← SOP doc_id 패턴 매칭
MQ for GCB:       "SUPRA N EFEM robot alarm cause"  ← 영문, Q&A 매칭
MQ for TS:        "EFEM robot alarm troubleshooting" ← TS doc_id 패턴 매칭
```

### 3.5 전략 5: 약어/정식명 병렬 쿼리

데이터에 약어와 정식명이 혼재:

```
데이터 실제 용어: "FCIP", "SV", "PM", "CH", "LP", "TM", "CTC"
```

MQ에서 약어와 정식명을 **동일 쿼리 내에 병기**하면 BM25+Vector 모두 유리:

```
"FCIP (Front Chamber Interface Panel) replacement"
"SV (Slot Valve) 교체"
```

### 3.6 전략 6: 수치/코드 보존 원칙

현재 MQ 프롬프트에 명시되지 않은 중요 원칙:

```
원문: "Error Code 252", "3800W", "version 2.50.14"
```

에러 코드, 전력값, 버전 번호 등은 **정확히 보존**해야 함. 이것이 가장 강력한 검색 단서.

---

## 4. 구현 우선순위 제안 (검색 성공률 기준)

평가 기준은 **검색 성공률**(예: `retrieved_docs >= 1`, `REFS 비어있지 않은 응답 비율`) 개선 폭을 최우선으로 둔다.

| 순위 | 전략 | 검색 성공률 개선 기대 | 난이도 |
|---|---|---|---|
| 1 | 짧은 키워드 쿼리 (3.1) | 매우 높음 (myservice 대다수 구간 직접 개선) | 낮음 (프롬프트 수정) |
| 2 | 수치/코드 보존 (3.6) | 높음 (정확 매칭 실패 구간 즉시 개선) | 낮음 (프롬프트 수정) |
| 3 | 이중 언어 쿼리 일관화 (3.2) | 높음 (GCB/영문 문서 검색 성공률 개선) | 낮음 (프롬프트 규칙 정합화) |
| 4 | 약어/정식명 병렬 (3.5) | 중간 (recall 보강) | 중간 (약어 사전 필요) |
| 5 | chapter-aware 쿼리 (3.3) | 중간 (의도별 precision 개선) | 중간 (라우팅/가중치 로직 보강) |
| 6 | doc_type별 분리 쿼리 (3.4) | 중간~높음 (장기 최적화) | 높음 (아키텍처 변경) |

---

## 5. 프롬프트 개선안 (즉시 적용 가능)

### 5.1 general_mq 개선안

```
현재:
  Q1: Reorganize the question into most intuitive search terms
  Q2: Expand abbreviations or use alternative expressions
  Q3: Add document structure keywords (overview, summary, table of contents)

개선:
  Q1: Extract core keywords from the question (device name, component, symptom/action).
      Keep error codes, part numbers, and numeric values exactly as-is.
  Q2: Create a short English keyword query (for GCB documents which are in English).
      Focus on the technical terms, not full sentences.
  Q3: Add ONE relevant action keyword based on intent:
      - Symptom/status inquiry → add "alarm" or "error"
      - How-to/procedure → add "replacement" or "calibration" or "절차"
      - Cause analysis → add "root cause" or "원인"
```

### 5.2 ts_mq 개선안

```
현재:
  Q2: Add hardware keywords (PLC, sensor, valve, circuit, signal)

개선:
  Q2: Keep the exact error code/alarm name. Add the specific module name
      (PM1, CH2, TM Robot, EFEM, LP) mentioned in the question.
      Do NOT add generic hardware terms.
  Q3: Create an English query with: device + alarm/error + "cause resolution"
      (for matching GCB English documents)
```

### 5.3 st_mq 개선안

st_mq에서 최종 쿼리를 선별할 때:

```
추가 규칙:
  - At least one query must be in English (for GCB matching)
  - Each query should be under 10 words/tokens (short queries match short myservice content better)
  - Preserve all numeric values, error codes, and version numbers exactly
  - Do NOT add generic expansion words like "overview", "summary", "guide", "checklist"
```
