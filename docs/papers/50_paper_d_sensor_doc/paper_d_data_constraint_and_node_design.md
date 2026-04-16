# Paper D — 데이터 제약사항과 노드 설계

> 작성일: 2026-04-16
> 목적: 시간 매칭 불가 제약 확인 후, semantic retrieval 기반으로 전환된 방향과 센서 이상 노드 정의 방법을 정리

---

## 1. 핵심 제약사항

### 1.1 데이터 상황

| 항목 | 상태 |
|------|------|
| 센서 시계열 데이터 | ✅ 존재 (SUPRA Vplus) |
| 이상 라벨 | ❌ 없음 |
| 정비 이력 범위 | 전세계 SUPRA Vplus 정비 이력 (특정 장비/시점에 한정되지 않음) |

### 1.2 시간 기반 매칭이 안 되는 이유

- 정비 로그가 **글로벌**: 수집된 센서의 특정 장비가 아닌, 전세계 모든 장비의 정비 이력
- 센서 episode가 발생한 시점과 관련된 정비 로그를 **시간만으로 찾을 수 없음**
- 라벨 부재: 어떤 센서 패턴이 어떤 정비와 연결되는지 **ground truth 없음**

### 1.3 이전 계획에서 바뀐 것

| 항목 | 이전 계획 | 수정 후 |
|------|----------|--------|
| 연결 방법 | 시간 매칭 (±1일/±3일) | **의미(semantic) 매칭** |
| 정비 이력 역할 | 같은 장비의 event log | **전세계 knowledge base** |
| Pilot set 구축 | 시간 기반 gold/silver/weak | **패턴 기반 pseudo-labeling** |
| 핵심 novelty | temporal uncertainty alignment | **sensor pattern → maintenance text cross-modal retrieval** |

---

## 2. 수정된 연결 구조

```
이전:
  이 장비에서 T시점에 APC 이상 → T±3일 내 이 장비의 정비 로그 찾기

수정:
  APC position이 이런 패턴일 때 → 전세계 정비 이력에서 이 패턴과 관련된 사례 찾기
```

즉, 정비 이력은 **특정 시점의 이벤트 로그**가 아니라 **"이런 문제가 생기면 이렇게 했다"는 인류의 경험치 데이터베이스**로 본다.

---

## 3. 센서 이상 노드 정의

### 3.1 노드에 뭘 넣는가

노드는 **사람이 읽을 수 있는 설명 + 수치 feature + inter-sensor context** 조합.

```json
{
  "sensor": "APC_Position",
  "pattern": "sustained_high",

  "features": {
    "mean_value": 91.2,
    "deviation_from_normal": 36.2,
    "duration_sec": 42,
    "trend": "increasing",
    "saturation": 0.88,
    "oscillation_score": 0.17
  },

  "inter_sensor": {
    "pressure_stable": false,
    "actuator_current_normal": true
  },

  "metadata": {
    "equipment": "SUPRA Vplus",
    "recipe_step": "pressure_stabilization"
  }
}
```

### 3.2 왜 "APC position 상승"만으로는 부족한가

| "APC position 상승"만 있으면 | 수치 feature가 있으면 |
|---------------------------|-------------------|
| 얼마나 올랐는지 모름 | deviation = 36.2% |
| 얼마나 오래 지속됐는지 모름 | duration = 42초 |
| 다른 센서 상태 모름 | pressure_stable = false |
| 같은 "상승"이라도 원인 구분 불가 | saturation + oscillation 조합으로 구분 가능 |

### 3.3 이상 유형(pattern) 분류

초기에 **5~10개**로 시작:

| pattern | 정의 | 센서 예시 |
|---------|------|----------|
| `sustained_high` | setpoint 대비 높은 값 유지 | APC position 90%+ 고정 |
| `sustained_low` | setpoint 대비 낮은 값 유지 | |
| `drift_up` | 점진적 상승 | APC trend 서서히 상승 |
| `drift_down` | 점진적 하락 | |
| `oscillation` | 주기적 진동 | pressure hunting |
| `spike` | 급격한 단발성 변화 | |
| `stuck` | 값 변화 없음 (setpoint은 변함) | tracking failure |
| `step_response_delay` | setpoint 변경 후 반응 지연 | |
| `out_of_range` | 정상 범위 이탈 | FDC spec out |

### 3.4 같은 센서, 다른 pattern → 다른 정비 사례

이것이 노드 설계의 핵심 가치:

```
APC_Position + sustained_high + pressure_unstable
  → "APC PRESSURE HUNTING" (doc 40036448)
  → action: APC valve replacement

APC_Position + drift_up + pressure_stable  
  → "APC Position 상승 원인 미상" (doc 40146514)
  → action: APC Auto Learn/Tune

APC_Position + stuck + actuator_current_low
  → APC communication or mechanical issue
  → action: PDB check, D-net reset
```

수치 feature 조합이 다르면 **같은 센서라도 다른 정비 사례와 연결**된다.

---

## 4. 노드가 정비 로그와 연결되는 방법

### 4.1 전체 흐름

```
Step 1: 센서 시계열에서 이상 구간 검출
  - 규칙 기반 (threshold, duration, pattern matching)
  - 라벨 불필요

Step 2: 이상 구간 → 노드 생성
  - pattern 분류 (sustained_high, drift, oscillation 등)
  - 수치 feature 계산
  - inter-sensor context 추가

Step 3: 노드 → 텍스트 query 변환 (Eventizer)
  "APC Position sustained high deviation 36%,
   42 seconds duration, pressure unstable"

Step 4: 텍스트 query → 정비 이력 검색 (Retrieval)
  - BM25 / dense / hybrid
  - 전세계 정비 이력에서 유사 사례 검색

Step 5: 검색된 사례 + SOP/Manual → Grounded Diagnosis
```

### 4.2 핵심: 시간이 아니라 패턴으로 연결

```
시간 기반: "이 장비에서 4월 15일에 이상 → 4월 16일 정비 로그"
패턴 기반: "이런 패턴의 이상 → 전세계에서 이런 패턴일 때 어떻게 했는지"
```

---

## 5. Pilot Set 구축 전략 (수정)

시간 매칭이 안 되므로 **3가지 대안**:

### 방법 1: 규칙 기반 Pseudo-Labeling (추천, 가장 빠름)

```
1. 센서에서 pattern 자동 추출 (규칙 기반)
2. pattern → symptom keyword 변환
3. keyword로 정비 이력 ES 검색
4. 검색 결과 상위 N개 = silver label
5. 사람이 상위 건만 검토 → gold label 확보
```

### 방법 2: 역방향 매칭

```
1. 정비 이력에서 명확한 사례 먼저 선택 (APC pressure hunting, position drift 등)
2. 해당 사례의 증상 설명 → 센서 패턴 정의
3. 센서 시계열에서 그 패턴에 해당하는 구간 탐색
```

### 방법 3: 전문가 소량 매칭

```
1. 센서 패턴 10~20개를 뽑아서
2. 사람(본인)이 "이건 이런 정비 사례와 관련 있다"를 판정
3. 소량이지만 gold quality
```

### 추천 순서

**방법 1로 시작** → silver label로 baseline 실험 → **방법 3으로 gold 확보** → 논문용 평가

---

## 6. 이 전환이 논문에 미치는 영향

### 약해지는 것
- "temporal uncertainty-aware alignment"이 핵심 novelty로 쓰기 어려워짐

### 강해지는 것
- **"sensor pattern → maintenance knowledge base cross-modal retrieval"** — 더 범용적
- **"라벨 없이 동작하는 unsupervised/weakly-supervised retrieval"** — 더 현실적
- **"전세계 정비 이력을 knowledge base로 활용"** — 더 스케일러블

### 수정된 핵심 thesis 문장

> 본 연구는 반도체 장비의 센서 이상 패턴을 구조화된 이벤트 노드로 변환하고,
> 이 노드의 수치적 특성과 텍스트 표현을 이용하여 글로벌 정비 knowledge base에서
> 유사 사례를 검색하며, 검색된 사례와 절차 문서를 근거로 진단을 생성하는
> cross-modal retrieval 프레임워크를 제안한다.

---

## 7. 다음 행동

| 우선순위 | 행동 | 필요한 것 |
|---------|------|----------|
| **1** | 센서 시계열 데이터 접근 확인 | 파일 경로 / DB 접속 |
| **2** | APC 센서에서 pattern 5개 규칙 정의 + 자동 추출 | 시계열 데이터 |
| **3** | 추출된 pattern → keyword → ES 검색 | 규칙 + ES |
| **4** | 검색 결과 상위 건 검토 → silver label | 사람 검토 |

---

## Related Documents

- `paper_d_interim_report.md` — 중간 보고서 (시간 기반 전제)
- `paper_d_algorithm_design.md` — 6모듈 파이프라인 (temporal alignment 모듈 수정 필요)
- `paper_d_es_query_results.md` — 1차 ES 조회 결과
- `paper_d_full_sensor_doc_scan.md` — 전수 스캔 결과
- `paper_d_keyword_query_log.md` — 키워드별 원문 검증 로그
