# Paper D — 이상 노드 정의 전략

> 작성일: 2026-04-16
> 목적: "센서 시계열에서 이상 노드를 어떻게 정의하는가"라는 핵심 문제에 대한 접근 전략 정리

---

## 1. 핵심 문제 인식

Paper D에서 가장 어려운 문제는 retrieval이 아니라 **그 이전 단계**이다.

> 센서 시계열에서 "이건 이상이다"를 정의하고 노드로 남기는 것 자체가 이상 탐지이다.

그런데 **라벨이 없다.** 이상인지 아닌지를 알려주는 ground truth가 존재하지 않는다.

---

## 2. 가능한 접근법 3가지

### 방법 1: 규칙 기반 (baseline)

기존에 경험한 방법 — DTW로 base pattern을 잡고 거리로 이상 판정.

```
정상 패턴과의 거리가 threshold 이상 → 이상 노드 생성
```

| 장점 | 단점 |
|------|------|
| 빠르고 해석 가능 | threshold를 사람이 정해야 함 |
| 바로 실행 가능 | 새로운 유형의 이상을 못 잡을 수 있음 |

**논문에서의 위치**: baseline / comparison method

### 방법 2: 비지도 이상 탐지 (보조)

Anomaly Transformer, autoencoder 등으로 "정상과 다른 구간"을 자동 추출.

```
reconstruction error가 높은 구간 → 이상 노드 후보
```

| 장점 | 단점 |
|------|------|
| 라벨 불필요 | "왜 이상인지" 설명이 약함 |
| 새로운 유형도 잡을 수 있음 | 학습 비용 |

**논문에서의 위치**: eventizer 내부의 선택적 모듈

### 방법 3: 정비 이력 역방향 정의 (추천 — 당신의 데이터에 가장 적합)

**정비 이력에 있는 증상 설명을 먼저 보고, 그 증상에 해당하는 센서 패턴을 역으로 정의하는 것.**

```
정비 이력: "APC position drift"      → 센서에서 drift 패턴 규칙 정의
정비 이력: "pressure hunting"        → 센서에서 oscillation 패턴 규칙 정의
정비 이력: "Temp1 FDC out of spec"   → 센서에서 out_of_range 패턴 규칙 정의
```

| 장점 | 단점 |
|------|------|
| **정비 이력이 곧 라벨 역할** | 정비 이력에 없는 새 유형은 못 잡음 |
| 당신의 데이터 구조와 맞음 | failure mode 카탈로그를 먼저 만들어야 함 |
| 논문 스토리가 깔끔 | |

**논문에서의 위치**: 핵심 contribution

---

## 3. 왜 방법 3이 가장 적합한가

### 3.1 라벨이 없지만 정비 이력이 있다

정비 이력의 증상 설명이 "어떤 종류의 이상을 찾아야 하는지"를 알려준다.
이건 **직접 라벨**은 아니지만 **무엇을 탐지해야 하는지의 가이드**이다.

### 3.2 글로벌 정비 이력의 역할

전세계 정비 이력은 "이런 이상이 존재한다"는 **카탈로그** 역할을 한다.
즉, failure mode의 목록을 정비 이력에서 뽑을 수 있다.

### 3.3 논문 스토리

> "정비 이력에서 발견되는 failure mode를 기반으로 센서 이상 패턴을 정의하고,
> 새로운 센서 데이터에서 그 패턴이 감지되면 관련 정비 사례와 문서를 retrieval한다"

이 구조는:
- 이상 탐지 전체를 새로 푸는 게 아님
- 정비 이력을 "패턴 정의의 근거"로 활용
- retrieval 중심은 유지

---

## 4. 실행 순서

```
Step 1: 정비 이력에서 failure mode 카탈로그 추출
  - ES에서 chapter=cause인 문서 전체 수집
  - 반복 등장하는 failure mode 분류
  - 예: "APC position drift", "pressure hunting", "Temp FDC out of spec"

Step 2: 각 failure mode → 센서 패턴 규칙 정의
  - "position drift" → sustained_high + increasing trend
  - "pressure hunting" → oscillation + high frequency
  - "Temp out of spec" → out_of_range
  - DTW, 통계량, correlation 등을 조합하여 규칙 작성

Step 3: 센서 시계열에서 규칙으로 이상 구간 추출
  - 각 구간 = 이상 노드
  - 노드에 pattern, 수치 feature, inter-sensor context 포함

Step 4: 이상 노드 → 정비 이력 검색 (retrieval)
  - 노드의 feature로 query 생성
  - 전세계 정비 이력에서 유사 사례 검색

Step 5: 평가
  - 검색된 사례가 실제로 해당 패턴과 관련 있는지 사람이 판정
```

---

## 5. 논문 기여 관점

| 기여 | 내용 |
|------|------|
| **Failure-mode-driven anomaly definition** | 정비 이력에서 역방향으로 센서 패턴을 정의하는 방법론. 라벨 없이 "무엇을 찾아야 하는지"를 정비 이력에서 유도 |
| **Sensor pattern → maintenance case retrieval** | 정의된 패턴으로 관련 사례를 찾는 cross-modal retrieval |
| **Closed-loop 구조** | 정비 이력 → 패턴 정의 → 센서 탐지 → 정비 이력 검색. 정비 이력이 시작이자 끝 |

### 논문화 가능한 이유

이 접근은 단순히 "규칙을 만들어서 탐지했다"가 아니라:

1. **정비 이력이라는 외부 지식**을 이상 정의에 활용한다는 점
2. **정의한 노드로 다시 정비 이력을 검색**한다는 순환 구조
3. 기존 이상 탐지 연구와 다르게 **"탐지 → 문서 연결 → 진단"까지 한 프레임**

에서 기여가 생긴다.

---

## 6. 이전 문서와의 관계

| 이전 문서 | 이 문서와의 관계 |
|-----------|--------------|
| `paper_d_data_constraint_and_node_design.md` | 노드 구조(JSON schema, pattern 9종)를 정의함. 이 문서는 **"어떤 전략으로 그 노드를 만드는가"**를 다룸 |
| `paper_d_algorithm_design.md` | Eventizer 모듈 설계. 이 문서는 Eventizer의 **입력(failure mode 카탈로그)을 어디서 가져오는가**를 다룸 |
| `paper_d_map_and_graph_framing.md` | "지도 만들기"가 핵심. 이 문서는 그 지도의 **노드를 만드는 구체적 방법**을 다룸 |

---

## 7. 가장 급한 다음 행동

**Step 1: 정비 이력에서 failure mode 카탈로그 추출**

ES에서 `device_name=SUPRA Vplus`, `chapter=cause`인 문서를 전수 수집하고,
반복 등장하는 failure mode를 분류하면 된다.

이 카탈로그가 곧 **"어떤 이상 노드를 정의해야 하는지"의 목록**이 된다.

---

## Related Documents

- `paper_d_data_constraint_and_node_design.md` — 노드 구조 + 데이터 제약
- `paper_d_algorithm_design.md` — 6모듈 파이프라인
- `paper_d_map_and_graph_framing.md` — 지도/길 찾기 구분
- `paper_d_interim_report.md` — 중간 보고서
