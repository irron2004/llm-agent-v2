# Paper D — APC Position 이상 진단 에이전트 아키텍처

> 작성일: 2026-04-14
> RAAD-LLM 리뷰를 바탕으로 설계한 실무형 에이전트 구조

---

## 1. 전체 파이프라인

```
센서 스트림
  → A. 센서 전처리 + 컨텍스트 결합
  → B. Eventizer (숫자 → 정비 가능한 증상)
  → C. 고장모드 후보 검색
  → D. 문서/정비이력 검색 (3종 Retriever)
  → E. 근거기반 답변 생성
  → F. 안전 가드
```

---

## 2. RAAD-LLM에서 가져올 것 / 버릴 것

### 가져올 것
- 장비별·recipe별·step별 adaptive baseline
- 센서 window → 텍스트/증상 변환 방식
- 도메인 규칙을 이용한 false positive 억제
- operator knowledge + maintenance knowledge 결합

### 버릴 것
- z-score CSV retrieval 중심의 RAG
- 변수별 독립 처리만으로 끝내는 구조
- "anomaly / non-anomaly"만 출력하는 답변 방식
- 문서 전체를 프롬프트에 다 붙여 넣는 방식

---

## 3. 각 단계 상세

### A. 센서 전처리 + 컨텍스트 결합

**입력 데이터**:
| 필드 | 설명 |
|------|------|
| equipment_id, chamber_id | 장비 식별 |
| recipe_id, recipe_step, mode | 운전 조건 |
| timestamp | 시각 |
| apc_position_actual, apc_position_setpoint | APC 위치 |
| chamber_pressure | 챔버 압력 |
| throttle/actuator current | 액추에이터 전류 |
| valve open/close state | 밸브 상태 |
| 최근 calibration 일자 | 교정 이력 |
| 최근 PM/교체 이력 | 정비 이력 |
| 알람 코드 / 인터락 상태 | 알람 정보 |

**파생 Feature**:
| Feature | 계산 |
|---------|------|
| tracking error | actual - setpoint |
| absolute error 지속시간 | threshold 초과 연속 구간 |
| saturation 비율 | 0%/100% 근처 고정 비율 |
| stuck score | 값 변화 거의 없음 + setpoint 변화 존재 |
| response lag | step change 후 반응 지연 |
| oscillation score | 주기적 진동 패턴 |
| pressure coupling score | APC-pressure 상관 |
| actuator inconsistency | 명령 vs 실제 불일치 |

### B. Eventizer: 숫자 → 정비 가능한 증상

**핵심**: anomaly가 아니라 **symptom** 출력.
문서는 anomaly가 아닌 symptom / failure mode / action에 연결됨.

**출력 JSON 예시**:
```json
{
  "component": "APC",
  "symptoms": [
    {"name": "position_tracking_failure", "score": 0.93},
    {"name": "high_saturation", "score": 0.81},
    {"name": "pressure_oscillation", "score": 0.74}
  ],
  "context": {
    "recipe_step": "pressure_stabilization",
    "mode": "auto",
    "recent_calibration_overdue_days": 18
  },
  "evidence_features": {
    "mean_abs_error": 31.2,
    "error_duration_sec": 42,
    "stuck_score": 0.88,
    "oscillation_hz": 0.17
  }
}
```

### C. 고장모드 후보 검색

문서 검색 전 중간층: component + symptom + context → failure mode

**고장모드 후보 예시**:
- APC calibration drift
- valve sticking
- actuator feedback mismatch
- pneumatic issue
- sensor wiring / encoder fault
- controller tuning instability

**규칙 엔진 예시**:
| 조건 | 우선 고장모드 |
|------|--------------|
| tracking_failure + saturation + overdue_calibration | calibration drift |
| tracking_failure + no actuator current change | actuator/control path |
| oscillation + pressure coupling | tuning or sticking |

### D. 문서 저장 및 검색

**문서 구조화 스키마**:
```json
{
  "doc_id": "SOP_APC_013",
  "section_id": "4.2.1",
  "component": "APC",
  "symptoms": ["position mismatch", "stuck", "unstable control"],
  "failure_modes": ["calibration drift", "valve sticking"],
  "actions": [
    "manual jog test",
    "feedback check",
    "recalibration",
    "mechanical inspection"
  ],
  "safety_level": "medium",
  "applicable_recipe_steps": ["pressure_stabilization", "pumpdown"],
  "text": "..."
}
```

**3종 문서 소스**:
1. Setup Manual
2. SOP / Maintenance Procedure
3. 과거 정비 이력 / Trouble Ticket / 작업 결과

**3종 Retriever**:
| Retriever | 역할 |
|-----------|------|
| Fault-mode retriever | APC + position_tracking_failure + auto mode + pressure_stabilization |
| Document retriever | BM25 + vector + metadata filter |
| Case retriever | 과거 동일 장비/유사 recipe/유사 symptom 사례 |

**최종 rank score**:
```
final_score = 0.35 * keyword + 0.35 * dense + 0.20 * metadata + 0.10 * case_similarity
```

**Metadata filter (강하게 적용)**:
- component = APC
- equipment family match
- recipe step match
- chamber type match
- safety criticality
- revision 최신성

### E. 답변 LLM — 근거 편집기

**입력**: 이벤트 JSON + top-k 문서 section + top-k 정비 사례

**출력 형식**:
1. 가능성 높은 원인 1~3개
2. 근거 문서 section
3. 즉시 점검 순서
4. 안전 주의
5. 확신도
6. 추가 확인 센서

**금지 규칙**:
- 검색된 문서에 없는 조치는 새로 만들지 말 것
- shutdown/override/reset 제안은 SOP 근거가 있을 때만
- 확신도 낮으면 "추가 점검 필요"
- 근거 section 없으면 일반론으로 단정하지 말 것

---

## 4. APC Position 이상 — End-to-End 추론 예시

### 현재 상태
- setpoint: 45% → 62%
- actual: 89~92% 부근에서 고정
- 40초 이상 추종 실패
- pressure ripple 증가
- 최근 calibration overdue
- actuator current 변화 미약

### Step 1: Eventizer 출력
```json
{
  "component": "APC",
  "symptoms": [
    "position_tracking_failure",
    "high_saturation",
    "pressure_oscillation"
  ],
  "candidate_failure_modes": [
    "calibration_drift",
    "valve_sticking",
    "actuator_feedback_issue"
  ]
}
```

### Step 2: 검색 결과
- SOP: APC calibration check
- Manual: APC actuator feedback verification
- Case log: "position stuck near high open after PM due to linkage friction"

### Step 3: 최종 답변
```
가장 가능성 높은 원인은 APC calibration drift입니다.

근거:
- SOP_APC_013 4.2.1: setpoint 추종 실패와 고정 saturation 시 calibration 확인 우선
- MANUAL_APC_007 3.4: feedback mismatch 발생 시 actuator feedback consistency 점검
- CASE_2025_11_03: 유사 증상에서 linkage sticking 확인

권장 점검 순서:
1) manual jog test 수행
2) actual vs feedback consistency 확인
3) calibration procedure 실행
4) 개선 없으면 valve linkage/mechanical sticking 점검

안전 주의:
- auto pressure control 불안정 상태에서 recipe 지속 수행 금지
- shutdown/override는 SOP_APC_013 안전 절차 준수 후 실행

확신도: 0.78
추가 확인 센서: actuator current, chamber pressure ripple, feedback raw signal
```

---

## 5. 실패 포인트

이 프로젝트가 보통 망가지는 지점:
1. 문서 chunk를 너무 거칠게 잘라서 step 순서가 깨짐
2. 센서 이상을 symptom이 아니라 raw z-score로만 표현함
3. maintenance log 정규화가 안 돼서 weak label이 안 생김
4. equipment/chamber/recipe metadata를 검색에 안 씀
5. LLM이 문서 없는 조치를 지어냄
6. safety-critical action에 guardrail이 없음

---

## 6. 왜 이 구조가 RAAD-LLM보다 이 문제에 더 맞는가

RAAD-LLM은 anomaly detection에는 유용하지만:
- **"문서를 정확히 골라 근거를 붙여 설명하는 문제"**를 정면으로 풀지 않음
- frozen LLM, 수동 재구성된 domain context, z-score retrieval, 정적 데이터셋, 변수 독립 처리에 무게
- RAAD-LLMv2는 더 scalable하지만 성능 trade-off 존재

**결론**: RAAD-LLM을 detector/eventizer로 축소 사용하고, 문서 연결은 별도 retriever에 맡기는 구조가 더 자연스러움.
