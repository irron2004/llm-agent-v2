# Paper D — Keyword Query Log

> 작성일: 2026-04-16  
> 목적: 센서명/키워드로 ES를 조회했을 때 나온 문서들을 모아두고, 실제 관련 문서인지 사람이 판정할 수 있게 정리한 작업용 문서

---

## 1. 이 문서의 목적

이 문서는 다음 문제를 해결하기 위해 만든다.

1. 어떤 센서/키워드로 검색했는지 기록
2. 어떤 문서들이 hit 되었는지 모아두기
3. 사람이 보고 **관련 / 부분관련 / 무관** 으로 판정하기
4. 판정 결과를 기반으로 validated vocabulary를 만들기

즉, 이 문서는 **검색 결과 요약본**이 아니라,
Paper D의 **후보 문서 검토 작업장**이다.

---

## 2. 기록 규칙

각 검색 결과는 아래 단위로 기록한다.

- `sensor`: 기준 센서명
- `query_keyword`: 실제로 사용한 검색어
- `variant_type`: `exact / spaced / token_combo / single_token / corpus_phrase`
- `doc_id`
- `doc_type`
- `chapter`
- `summary`
- `relevance`
- `notes`

### relevance 값 정의
- `relevant`: 해당 센서 또는 센서군 문제를 직접 다룸
- `partial`: 같은 subsystem/증상 맥락이지만 직접적이지 않음
- `irrelevant`: 단어만 겹치고 실제로는 무관
- `unreviewed`: 아직 사람이 검토하지 않음

---

## 3. relevance 판정 기준

### 3.1 relevant
다음 중 하나 이상이면 `relevant`

- 센서명이 직접 언급됨
- 해당 센서군(APC, Temp, EPD 등)의 이상을 직접 설명함
- cause/action/status가 해당 센서 문제와 직접 연결됨
- 같은 failure mode 분석에서 핵심 센서로 등장함

### 3.2 partial
다음이면 `partial`

- 같은 chamber/subsystem 문맥에 있으나 직접 센서 설명은 아님
- 같은 troubleshooting 흐름에 포함되지만 다른 센서가 중심임
- 센서 자체보다 주변 조치/환경 정보가 중심임

### 3.3 irrelevant
다음이면 `irrelevant`

- 키워드 토큰만 우연히 겹침
- 일반 단어(`Pressure`, `Position`, `Valve`) 때문에 과도하게 잡힘
- 문서 내용이 실제 센서군과 무관함

---

## 4. 키워드별 조회 로그

> **ES 원문 검증일: 2026-04-16** (Claude가 `rag_chunks_dev_current`에서 doc_id별 원문을 직접 조회하여 판정)

### 4.1 APC_Position

| sensor | query_keyword | variant_type | doc_id | doc_type | chapter | summary | relevance | notes |
|---|---|---|---|---|---|---|---|---|
| APC_Position | APC_Position | exact | 40146514 | myservice | cause | 원인 미상 APC Position 상승 | **relevant** | ✅ 원문 확인: "원인 미상 APC Position 상승". status/action/cause/result 4개 chunk 모두 APC Position 직접 다룸. FAST purge 후 trend 상승 → APC Auto Learn/Tune으로 조치. SOP 참조: 없음 |
| APC_Position | APC Position | spaced | 40063372 | myservice | action | PDB toggle switch off → APC Position/Pin 복구 | **relevant** | ✅ 원문 확인: "APC Position 구동 시 동작 안됨", "PDB 확인 시 Toggle Switch Off됨". APC 데이터 reading 불가 → D-net reset → PDB switch on으로 복구. APC와 직접 연결 |
| APC_Position | Position | single_token | 40044000 | myservice | cause | TEACHING POSITION NO GOOD | **irrelevant** | ✅ 원문 확인: "LP2 EFEM ROBOT WAFER SENSOR ALARM" → robot teaching position 문제. APC와 **무관**. "Position" 토큰이 우연히 겹친 사례 |

### 4.2 APC_Pressure

| sensor | query_keyword | variant_type | doc_id | doc_type | chapter | summary | relevance | notes |
|---|---|---|---|---|---|---|---|---|
| APC_Pressure | APC_Pressure | exact | 40036448 | myservice | cause | APC PRESSURE HUNTING | **relevant** | ✅ 원문 확인: "APC PRESSURE HUNTING", "APC 위아래 반대로 달려있어서 원복해도 현상동일". A급 APC 교체 → parameter setting → monitoring. SOP: "Global SOP_SUPRA N_REP_PM_APC VALVE" |
| APC_Pressure | APC Pressure | spaced | 40117956 | myservice | status | 2800 APC PRESSURE E3 FLAG | **relevant** | ✅ 원문 확인: pressure difference between 2800 and 1U, APC auto learn, noise filter 추가. APC pressure 직접 다룸 |
| APC_Pressure | Pressure | single_token | 40048192 | myservice | action | 9.0 Step Pumping Time 증가 | **partial** | ✅ 원문 확인: recipe 비교 중 Pressure 1350 등장하지만 주제는 "pumping time 증가". APC pressure 특정이 아닌 recipe 조건 비교. Pressure 토큰이 넓게 잡힌 사례 |

### 4.3 Temp1

| sensor | query_keyword | variant_type | doc_id | doc_type | chapter | summary | relevance | notes |
|---|---|---|---|---|---|---|---|---|
| Temp1 | Temp1 | exact | 40042585 | myservice | status | PM2 Temp1 trigger temperature differential FDC out of Spec | **relevant** | ✅ 원문 확인: "Customer called PM2 Temp1 trigger temperature differential FDC out of Spec". heater chuck 성능 저하 → chuck 교체 → auto tune → wet clean → pump down → leak check. Temp1 센서 이상이 직접 트리거 |
| Temp1 | Temp | single_token | 40050384 | myservice | status | Power Supply fail → PMC communication error, Temp CTR off | **partial** | ✅ 원문 확인: "Pin,Temp,CTR등 전원 off 상태". 주 원인은 power supply 불량(output 2V). Temp는 영향받은 부품 중 하나일 뿐, Temp1 센서 자체 이상은 아님 |

### 4.4 Temp2

| sensor | query_keyword | variant_type | doc_id | doc_type | chapter | summary | relevance | notes |
|---|---|---|---|---|---|---|---|---|
| Temp2 | Temp2 | exact | 40086884 | myservice | status | intermittent Temp2 interlock alarm | **relevant** | (원문 미조회 — 이번 batch에 포함 안 됨. summary 기준 relevant 유지, 추후 원문 확인 필요) |
| Temp2 | Temp | single_token | 40050384 | myservice | status | Power Supply fail → Temp CTR off | **partial** | ✅ 위 40050384와 동일. Temp2 특정이 아닌 전원 계열 장애 |

### 4.5 EPD_Monitor1

| sensor | query_keyword | variant_type | doc_id | doc_type | chapter | summary | relevance | notes |
|---|---|---|---|---|---|---|---|---|
| EPD_Monitor1 | EPD Monitor1 | spaced | 40086313 | myservice | status | PM2 CH1 EPD MONITOR 값이 불량(9수준) | **relevant** | ✅ 원문 확인: "PM2 CH1 EPD MONITOR 값 불량 (9수준)". EPD Monitor 직접 등장 |
| EPD_Monitor1 | Monitor | single_token | TBD | TBD | TBD | TBD | unreviewed | 단독 Monitor는 노이즈 가능성 높음 — 조회 보류 |

### 4.6 SourcePwr1_Reflect

| sensor | query_keyword | variant_type | doc_id | doc_type | chapter | summary | relevance | notes |
|---|---|---|---|---|---|---|---|---|
| SourcePwr1_Reflect | SourcePwr1 Reflect | spaced | TBD | TBD | TBD | TBD | unreviewed | exact/spaced 재조회 필요 |
| SourcePwr1_Reflect | Source | single_token | TBD | TBD | TBD | TBD | unreviewed | Source 단독 hit는 사람 검토 필수 |
| SourcePwr1_Reflect | Reflect | single_token | TBD | TBD | TBD | TBD | unreviewed | "reflected power" 문맥 확인 필요 |

### 4.7 Pressure (generic token)

| sensor | query_keyword | variant_type | doc_id | doc_type | chapter | summary | relevance | notes |
|---|---|---|---|---|---|---|---|---|
| Pressure | Pressure | single_token | 40048192 | myservice | action | 9.0 Step Pumping Time 증가, recipe 비교 중 Pressure 등장 | **partial** | ✅ 원문 확인: 주제는 pumping time, Pressure는 recipe 파라미터 중 하나. 너무 넓게 잡히는 대표 사례 |

---

## 5. 다음 작업

### 5.1 즉시 해야 할 일
1. exact / spaced / token_combo / single_token 결과를 계속 표에 누적
2. 사람이 relevance를 채워 넣기
3. `relevant` / `partial` 결과에서 반복 표현 추출

### 5.2 이후 연결 작업
1. validated vocabulary 문서 생성
2. lexical expansion 규칙 보정
3. maintenance case ↔ document grounding 실험으로 연결

---

## 6. 한 줄 요약

> 이 문서는 “어떤 키워드로 검색했을 때 어떤 문서가 나왔는지”와 “그 문서가 실제 관련 문서인지 아닌지”를 기록하는 작업용 문서이며, 이후 Paper D의 validated vocabulary와 retrieval baseline 설계의 근거가 된다.
