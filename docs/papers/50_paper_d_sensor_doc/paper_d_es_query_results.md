# Paper D — ES 센서명 조회 결과 정리

> 작성일: 2026-04-14  
> 목적: SUPRA Vplus 장비의 센서명을 기준으로 `myservice`, `gcb` 문서를 조회한 결과를 Paper D의 데이터 시작점 관점에서 정리한다.

---

## 1. 조회 목적

이번 조회의 목적은 다음과 같다.

1. 실제 센서명이 정비 문서(`myservice`, `gcb`)와 직접 연결되는지 확인
2. 어떤 센서군부터 Paper D를 시작하는 것이 현실적인지 판단
3. raw sensor name이 바로 문서에서 검색되는지, 아니면 문서용 표현 사전이 필요한지 확인

즉, 이 조회는 Paper D의 첫 번째 데이터 적합성 점검이다.

---

## 2. 조회 조건

### Elasticsearch 대상
- ES 호스트: `http://localhost:8002`
- alias: `rag_chunks_dev_current`
- 실제 인덱스: `rag_chunks_dev_v2`

### 필터
- `device_name = SUPRA Vplus`
- `doc_type in [myservice, gcb]`

### 센서 목록
- 중복 제거 후 고유 센서명 **62개**

### 구현상 주의점
- 이 인덱스는 `doc_type.keyword`가 아니라 **`doc_type` 필드 자체**로 검색해야 했다.
- `device_name.keyword`가 아니라 **`device_name` 필드**로 필터링해야 했다.

---

## 3. 전체 결과 요약

### 전체 통계
- 고유 센서명 수: **62**
- hit 있음: **7**
- hit 없음: **55**

즉,

> 센서명 그대로는 문서와 직접 연결되지 않는 경우가 대부분이었고,
> 일부 센서만 의미 있게 연결되었다.

---

## 4. 실제로 잘 잡힌 센서들

| 센서명 | hit 수 | 해석 |
|---|---:|---|
| `Pressure` | 1305 | 너무 일반적이라 넓게 잡힘 |
| `Temp2` | 242 | 유의미 |
| `Temp1` | 80 | 유의미 |
| `APC_Pressure` | 58 | 유의미 |
| `APC_Position` | 25 | 유의미 |
| `Process_Run_Time` | 2 | 제한적 |
| `Gas4_Pressure` | 1 | 매우 제한적 |

### 핵심적으로 쓸 만한 센서군
현재 바로 출발할 수 있는 센서군은 다음과 같다.

1. `APC_Position`
2. `APC_Pressure`
3. `Temp1`
4. `Temp2`

특히 **APC_Position / APC_Pressure** 가 가장 유망하다.

---

## 5. 대표 검색 결과

### 5.1 APC_Position
- hit 수: **25**

대표 샘플:
- `doc_type = myservice`
- `chapter = cause`
- 요약: *The root cause of the APC position increase remains unknown.*
- 키워드: `APC Position drift`

의미:
- 문서에서 `APC Position`이 직접 언급됨
- 원인/조치 문맥과 연결됨
- Paper D의 sensor-event ↔ maintenance case 연결 시작점으로 적합

추가 검증 메모:
- `APC_Position`의 exact / spaced query는 실제 관련 문서를 잘 찾았다.
- 반면 `Position` single-token으로 잡힌 `40044000`은 원문 검증 결과 **robot teaching position** 문서로 확인되어 `irrelevant`였다.
- 따라서 `Position` 같은 일반 토큰은 후보 수집용으로는 쓸 수 있지만, relevance 판정 없이는 위험하다.

### 5.2 APC_Pressure
- hit 수: **58**

대표 샘플:
- `doc_type = myservice`
- `chapter = cause`
- 요약: *The root cause was pressure hunting of the APC valve.*
- 키워드: `pressure hunting`, `APC`

의미:
- failure mode 문맥과 직접 연결됨
- APC pressure 관련 troubleshooting entry point로 적합

추가 검증 메모:
- `40036448`은 status/action/cause/result chunk가 비교적 잘 갖춰져 있고,
  SOP 참조(`Global SOP_SUPRA N_REP_PM_APC VALVE`)도 등장해 **Paper D pilot의 gold case 후보**로 볼 수 있다.

### 5.3 Temp2
- hit 수: **242**

대표 샘플:
- `doc_type = myservice`
- `chapter = status`
- 요약: *intermittent Temp2 interlock alarm ...*

의미:
- alarm / status 문맥에서 잘 잡힘
- interlock, chamber, o-ring 등 maintenance context가 붙음

추가 검증 메모:
- `40086884`는 현재 summary 기준으로 relevant로 유지하고 있으나,
  full content 원문 확인은 아직 남아 있다.

### 5.4 Temp1
- hit 수: **80**

대표 샘플:
- `doc_type = myservice`
- `chapter = status`
- 요약: *PM2 Temp1 trigger temperature differential out of specification.*

의미:
- temperature differential / spec out 문맥과 연결됨

추가 검증 메모:
- `40042585`는 원문 확인 결과 `Temp1 trigger temperature differential FDC out of spec → chuck 교체 → 정상화` 흐름이 확인되어,
  **증상 → 원인/조치 → 결과**가 이어지는 또 하나의 gold case 후보다.

---

## 6. 안 잡힌 센서들

다음 계열은 센서명 그대로는 거의 검색되지 않았다.

### 거의 안 잡힌 센서군
- `APC_SetPoint`
- `EPD_*`
- `Gas1~Gas6_*` 대부분
- `SourcePwr*`
- `Temp1_Set`, `Temp2_Set`
- `Recipe_Step_Num`

이것은 “센서가 중요하지 않다”는 뜻이 아니라,

> 문서에서는 raw sensor name 그대로 쓰지 않고,
> 더 현장식 표현 / 사람 중심 표현 / symptom 표현으로 기록할 가능성이 크다

는 뜻이다.

예시:
- `APC_SetPoint` → `pressure setting`, `set point`
- `SourcePwr1_Reflect` → `reflected power`, `RF reflect`
- `EPD_Monitor1` → `EPD monitor`, `monitor value`, `spec out`
- `Recipe_Step_Num` → `recipe step`, `process step`

---

## 7. Paper D 관점에서의 해석

이번 결과는 Paper D의 실제 시작점 선택에 매우 중요하다.

### 지금 확인된 것

#### 바로 연결 가능한 sensor family
- APC
- Temp

#### 바로 연결 안 되는 sensor family
- EPD
- Gas
- SourcePwr
- Recipe step

즉, Paper D를 실제 데이터로 시작할 때는:

1. **문서와 직접 연결되는 센서군부터 시작**하고
2. 나머지는 **sensor name → document term mapping**을 만든 뒤 확장

하는 방식이 맞다.

### 원문 검증으로 추가 확인된 점

1. **single-token false positive가 실제로 존재한다**
   - `Position` → `40044000` 사례는 robot teaching position 문서로 확인됨
   - 따라서 single-token은 후보 수집용으로만 쓰고, 반드시 사람 검토가 필요하다.

2. **gold case 후보가 보인다**
   - `40036448` (`APC_Pressure`): pressure hunting + APC valve + SOP 참조
   - `40042585` (`Temp1`): Temp1 FDC out of spec → chuck 교체 → 정상화

3. **summary만으로는 불충분한 문서가 있다**
   - `Temp2`의 대표 사례는 아직 full content 검증이 필요하다.
   - 따라서 query log에서 `relevant / partial / irrelevant / unreviewed` 구분이 중요하다.

---

## 8. 추천 다음 단계

### Step 1
`APC_Position`, `APC_Pressure`, `Temp1`, `Temp2` 중심으로 pilot set 시작

### Step 2
안 잡히는 센서들에 대해 lexical expansion + corpus-derived synonym 사전 구축

### Step 3
그 사전으로 `myservice`, `gcb` 재조회

### Step 4
조회 결과를 바탕으로 sensor event ↔ maintenance case ↔ document 연결 구조 생성

---

## 9. 한 줄 결론

> `rag_chunks_dev_v2`에서 `SUPRA Vplus + myservice/gcb`를 조회한 결과, **APC_Position, APC_Pressure, Temp1, Temp2**가 가장 유효하게 문서와 연결되었고, 나머지 센서들은 raw sensor name보다 **문서용 표현 사전**을 만든 뒤 다시 검색해야 할 가능성이 높다.
