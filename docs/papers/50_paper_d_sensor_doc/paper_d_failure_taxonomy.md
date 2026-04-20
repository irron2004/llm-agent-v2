# Paper D — Failure Taxonomy (문서 기반 이벤트 정의)

> 추출일: 2026-04-20
> 소스: ES `rag_chunks_dev_current`, SUPRA Vplus, myservice+gcb, cause+status chapters
> 방법: APC 294 chunks, EPD 346 chunks에서 failure 표현 추출 후 규칙 기반 분류

---

## 1. 접근 방식

**시계열이 아닌 문서에서 먼저 이벤트를 정의한다.**

이유:
- 문서가 이미 "정비 가능한 언어"로 되어 있음
- retrieval 목적지가 문서이므로, 문서의 용어로 이벤트를 정의해야 검색이 맞음
- 시계열 패턴은 이 이벤트 정의에 맞춰서 나중에 매핑

---

## 2. APC Failure Taxonomy

### 2.1 확정 이벤트 (cause+status 합산 기준, 빈도 5건 이상)

| 이벤트 ID | 이벤트명 | cause | status | 합계 | 센서 패턴 (추정) |
|-----------|---------|------:|-------:|-----:|----------------|
| `APC_FAIL` | APC 고장 (일반) | 25 | 11 | **36** | 값 급변 또는 무응답 |
| `APC_ANGLE_DRIFT` | APC Angle 상승/불량 | 12 | 31 | **43** | APC_Position 점진적 상승 |
| `APC_TREND_ABNORMAL` | APC Trend 불량 | 7 | 22 | **29** | APC_Position trend 변화 |
| `APC_HUNTING` | APC Hunting (일반) | 8 | 9 | **17** | APC_Position/Pressure 진동 |
| `APC_PRESSURE_HUNTING` | Pressure Hunting | 1 | 8 | **9** | APC_Pressure 진동 |
| `APC_CALIBRATION` | Calibration / Auto Learn | 1 | 12 | **13** | 정상 복귀 후 재발 패턴 |
| `APC_DISPERSION` | 산포 불량 | 6 | 5 | **11** | APC_Position 분산 증가 |
| `APC_VALVE_FAIL` | Valve Fail | 6 | 3 | **9** | 급격한 동작 불능 |
| `APC_INSPECTION` | 점검 / 확인 | 1 | 9 | **10** | (이벤트라기보다 조치) |

### 2.2 희소 이벤트 (빈도 5건 미만)

| 이벤트 ID | 합계 | 비고 |
|-----------|-----:|------|
| `APC_POSITION_DRIFT` | 3 | ANGLE_DRIFT와 통합 가능 |
| `APC_PRESSURE_DRIFT` | 3 | PRESSURE_HUNTING과 구분 필요 |
| `APC_DRIFT` (일반) | 6 | ANGLE 또는 POSITION으로 세분화 가능 |
| `APC_COMM_FAIL` | 2 | 통신 오류 (D-net) |
| `APC_SCREW_DAMAGE` | 2 | 물리적 손상 |
| `APC_MISALIGNMENT` | 0* | "틀어짐" — OTHER에 포함됨 |

### 2.3 APC_OTHER (재분류 필요: 101건)

cause 59 + status 42 = 101건이 OTHER로 분류됨.
이들은 "APC"를 포함하지만 위 패턴에 명확히 매칭되지 않은 것.
→ **수동 검토 후 기존 이벤트로 재분류하거나 신규 이벤트 정의 필요**

---

## 3. EPD Failure Taxonomy

### 3.1 확정 이벤트 (빈도 5건 이상)

| 이벤트 ID | 이벤트명 | cause | status | 합계 | 센서 패턴 (추정) |
|-----------|---------|------:|-------:|-----:|----------------|
| `EPD_CALIBRATION_NG` | Calibration 불량 | 8 | 38 | **46** | EPD_Monitor 값 범위 이탈 |
| `EPD_DELAY` | EPD Delay | 18 | 29 | **47** | EPD 응답 지연 |
| `EPD_FAIL` | EPD 고장 (일반) | 16 | 11 | **27** | 값 무응답 또는 급변 |
| `EPD_VALUE_ABNORMAL` | EPD 값 불량 | 9 | 20 | **29** | EPD_Monitor 비정상 수준 |
| `EPD_PEAK_LOW` | Peak 값 저하 | 4 | 11 | **15** | EPD peak voltage 감소 |
| `EPD_CABLE_FAIL` | Cable 불량 | 8 | 3 | **11** | 간헐적 값 변동 / 노이즈 |
| `EPD_GLASS_LEAK` | Glass Leak | 5 | 6 | **11** | (하드웨어, 센서 간접) |
| `EPD_HUNTING` | EPD Hunting/노이즈 | 6 | 5 | **11** | EPD_Monitor 진동 |
| `EPD_SENSOR_FAIL` | Sensor 자체 고장 | 6 | 1 | **7** | 값 고정 또는 무응답 |
| `EPD_TREND_LOW` | Trend 저하 | 3 | 6 | **9** | EPD_Monitor 점진적 하락 |
| `EPD_TREND_ABNORMAL` | Trend 이상 (일반) | 0 | 6 | **6** | EPD trend 변화 |
| `EPD_INTERLOCK` | Interlock 발생 | 1 | 4 | **5** | EPD 값이 한계 초과 |

### 3.2 희소 이벤트

| 이벤트 ID | 합계 | 비고 |
|-----------|-----:|------|
| `EPD_CURVE_SHIFT` | 1 | EPD 파형 자체 변형 |
| `EPD_FAN_FAIL` | 1 | Fan 고장 |
| `EPD_PEAK_HIGH` | 1 | Peak 과다 |
| `EPD_PEAK_INTERLOCK` | 2 | Peak에 의한 interlock |

### 3.3 EPD_OTHER (재분류 필요: 117건)

cause 47 + status 70 = 117건. 수동 검토 필요.

---

## 4. 1편 논문용 최소 이벤트 세트

실험에서 사용할 이벤트는 **빈도 10건 이상**으로 제한하는 것이 안전.

### APC (7개)

| # | 이벤트 | 빈도 | 센서 패턴 매핑 |
|---|--------|-----:|---------------|
| 1 | `APC_ANGLE_DRIFT` | 43 | Position 점진적 상승 |
| 2 | `APC_FAIL` | 36 | 값 급변 / 무응답 |
| 3 | `APC_TREND_ABNORMAL` | 29 | Trend 변화 |
| 4 | `APC_HUNTING` | 17 | Position/Pressure 진동 |
| 5 | `APC_CALIBRATION` | 13 | 정상 복귀 후 재발 |
| 6 | `APC_DISPERSION` | 11 | Position 분산 증가 |
| 7 | `APC_PRESSURE_HUNTING` | 9* | Pressure 진동 |

### EPD (8개)

| # | 이벤트 | 빈도 | 센서 패턴 매핑 |
|---|--------|-----:|---------------|
| 1 | `EPD_DELAY` | 47 | 응답 지연 |
| 2 | `EPD_CALIBRATION_NG` | 46 | 값 범위 이탈 |
| 3 | `EPD_VALUE_ABNORMAL` | 29 | 비정상 수준 |
| 4 | `EPD_FAIL` | 27 | 무응답 / 급변 |
| 5 | `EPD_PEAK_LOW` | 15 | Peak 감소 |
| 6 | `EPD_CABLE_FAIL` | 11 | 간헐 노이즈 |
| 7 | `EPD_GLASS_LEAK` | 11 | (하드웨어) |
| 8 | `EPD_HUNTING` | 11 | 진동 |

**총 15개 이벤트** — 이것이 Paper D 1편의 event vocabulary.

---

## 5. UNKNOWN 처리 전략

기존 이벤트에 해당하지 않는 이상:

```
if 정상 아님 AND 15개 이벤트 어디에도 안 맞음:
  → UNKNOWN_ANOMALY
  → retrieval은 하되 confidence 낮게 출력
  → "가장 유사한 사례는 X이지만 확신도가 낮습니다"
```

이것 자체가 **open-set retrieval** 문제로 논문 Discussion에서 다룰 수 있음.

---

## 6. 다음 단계

- [ ] APC_OTHER 101건 수동 검토 → 기존 이벤트로 재분류 또는 신규 정의
- [ ] EPD_OTHER 117건 수동 검토
- [ ] 각 이벤트의 **센서 패턴 규칙** 구체화 (threshold, duration, condition)
- [ ] 시계열 데이터에서 실제 패턴 확인

---

## Related Documents

- `evidence/paper_d_failure_expressions_raw.json` — 원본 텍스트
- `evidence/paper_d_failure_taxonomy.json` — 분류 결과 JSON
- `paper_d_keyword_query_log.md` — 키워드 검색 로그
- `paper_d_full_sensor_doc_scan.md` — 전수 스캔 결과
