---
date: 2026-04-20
type: daily-log
topics: [research-direction, scope-refinement, paper-strategy, agent-review]
status: completed
---

# 2026-04-20 — 최종 전략 확정: 두 Agent 검토 종합

> **Canonical daily decision note:** 2026-04-20 시점의 Paper D 최종 scope freeze와
> 1편 논문 기준 전략은 이 문서를 기준으로 본다. 같은 날짜의 다른 daily note는
> 아이디어 발전 과정과 중간 검토 기록으로 유지한다.

## 개요

두 개의 Agent 검토 결과를 종합하여 **1편 논문의 현실적이고 학술적으로 강한 방향**을 확정했습니다.

---

## Agent 검토 요약

### 첫 번째 Agent (박사논문 로드맵)

**강점:**
- RAAD-LLM보다 논문다운 방향 제시
- Cross-modal representation learning의 학술적 가치 인정
- Hard negative mining, hierarchy-aware loss 등 세밀한 방법론 제안

**위험:**
- Scope가 너무 큼 (anomaly + retrieval + grounding + action)
- 4개 loss 함수 동시 사용 → ablation 부담
- "top-tier conference 가능" 평가는 낙관적 (데이터/라벨 미확보)

### 두 번째 Agent (현실적 조정)

**핵심 지적:**
1. **Anomaly detection을 메인으로 가져오면 논문이 커진다**
   - 이전 합의는 "retrieval 중심"이었음
   - Detection은 전처리/보조 태스크로 위치 변경 필요

2. **"Rule-grounded"는 위험한 표현**
   - 리뷰어의 질문: "rule이 틀리면?", "rule engineering인가?"
   - → "rule-initialized" 또는 "event-informed"로 수정

3. **4개 loss를 한 번에 쓰면 위험**
   - Lambda 튜닝, ablation 부담
   - → 처음에는 L_align만, hierarchy-aware weight로 통합

4. **Baseline 20개+는 산만**
   - → 5~7개로 축소

**핵심 평가:**
> "hierarchy-aware hard negative와 contamination 평가 지표는 꼭 가져가라"

---

## 최종 판단

### 1편 논문 방향: **Retrieval 중심으로 Scope 축소**

| 항목 | 이전 (과도한 scope) | 수정 후 (집중된 scope) |
|------|---------------------|------------------------|
| **핵심 문제** | Anomaly + Retrieval + Grounding + Action | **Sensor-to-Maintenance Retrieval** |
| **제목** | Rule-Grounded Cross-Modal Alignment... | **Temporal- and Hierarchy-Aware Cross-Modal Retrieval of Maintenance Evidence from Sensor Events** |
| **Anomaly Detection** | 메인 contribution | 전처리/보조 태스크로 위치 변경 |
| **Loss 함수** | 4개 (L_align + L_anom + L_cause + L_device) | **1개** (hierarchy-aware contrastive) |
| **Baseline** | 20개+ | **5~7개** |

---

## 최종 1편 기여 3가지

### Contribution 1: Weakly-supervised sensor event abstraction

```
센서 윈도우를 정비 검색에 적합한 이벤트 표현으로 변환

- Rule-initialized (not rule-grounded)
- Noisy but useful intermediate representation
- Contrastive alignment를 통해 보정
```

**안 좋은 표현:** "rule-grounded sensor representation"
**좋은 표현:** "rule-initialized sensor event representation"

### Contribution 2: Cross-modal retrieval framework

```
센서 이벤트와 정비 문서를 정렬하는 retrieval framework

- Bi-encoder architecture
- Contrastive learning for alignment
- Weakly-supervised (no gold labels needed)
```

### Contribution 3: Hierarchy-aware hard negative mining

```
장비/모듈/부품 계층을 반영한 hard negative mining

- Same equipment family, different cause
- Same module, different part  
- Shared SOP but wrong context

→ Wrong-device contamination 감소
```

---

## Loss 함수 설계 (단순화)

### 수정 전 (과도함)

```
L = λ₁L_align + λ₂L_anom + λ₃L_cause + λ₄L_device
```

### 수정 후 (깔끔함)

```
L = -log[ exp(sim(q,d⁺)/τ) / (exp(sim(q,d⁺)/τ) + Σⱼ wⱼ exp(sim(q,dⱼ⁻)/τ)) ]

where wⱼ is hierarchy-aware weight:
- 완전히 다른 장비: 낮은 weight
- 같은 장비 family, 다른 cause: 높은 weight
- 같은 module, 다른 part: 더 높은 weight
- Shared SOP but wrong context: 높은 weight
```

**이점:**
- 별도의 L_device, L_cause, L_hierarchy 불필요
- 단일 contrastive loss 안에 hierarchy 녹임
- Ablation이 깔끔해짐 (weighting만 제거하면 됨)

---

## 모델 구조 (1편용 최소화)

```
[Sensor Window]
    ↓
[Eventizer] — Rule/stat-based event tokens
    - sustained high/low
    - sudden spike
    - oscillation
    - drift
    - correlation break
    ↓
[Sensor Encoder] — z_s = f_s(event)
    ↓
[Contrastive Learning]
    - sim(z_s, z_t)
    - hierarchy-aware hard negatives
    ↓
[Text Encoder] — z_t = f_t(doc chunk)
    ↓
[Maintenance Evidence Retrieval]
    - Top-k relevant documents
    - Root cause ranking
    - Evidence chunks
```

---

## 평가 지표 (핵심만)

### Retrieval 성능
- Recall@k
- MRR
- nDCG@k
- Cause Hit@k

### Contamination 성능
- Wrong-equipment@k
- Wrong-module@k
- Wrong-cause@k

### Expert Validation
- Top-5 결과 중 실제로 유용한 문서 비율
- 정비사가 판단한 relevance score
- Retrieved evidence가 원인 판단에 도움이 되는지

---

## Baseline (5~7개)

1. **BM25**
   - 센서 이벤트를 텍스트 query로 변환 → 정비 로그 검색

2. **Dense text retrieval**
   - Sentence embedding으로 검색

3. **Metadata filtering + BM25**
   - 장비/모듈 필터 후 BM25

4. **Naive bi-encoder**
   - Hierarchy-aware hard negative 없이 학습

5. **Proposed model (full)**
   - Hierarchy-aware hard negative 포함

6. **Ablation: No eventizer**
   - Raw statistical summary만 사용

7. **Ablation: No hierarchy weighting**
   - Hard negative는 있지만 hierarchy weight 없음

---

## 박사논문 전체 로드맵

| 편 | 주제 | 핵심 | 시기 |
|---|---|---|---|
| **1편** | Temporal- and Hierarchy-Aware Retrieval | Weak supervision + hierarchy-aware hard negatives | Y1-Q2 |
| **2편** | Fleet-wide Contamination Reduction | Equipment family, shared SOP, global log | Y2 |
| **3편** | Online Adaptation / Drift-Aware | 시간에 따른 pattern 변화 대응 | Y3 |
| **4편 (선택)** | Text-Grounded Anomaly Interpretation | Retrieved evidence 기반 anomaly scoring | Y4 |

---

## Agent 간 용어 차이 정리

두 agent가 다른 말을 한 것처럼 보였지만, 실제로는 동일한 방법론을 다른 용어로 표현한 것:

| Claude (Agent 2) | Hephaestus (Agent 1) | 실제 |
|-------------------|---------------------|------|
| "graph가 아니라 structured retrieval" | "graph-based state representation" | **같은 말** — 용어 정밀도 차이 |
| "LLM은 해석기" | "LLM은 해석기" | **동일** |
| "수치 기반 시스템이 retrieval" | "수치 기반 시스템이 retrieval" | **동일** |

핵심 합의: **수치 기반 엔진이 retrieval하고, LLM은 결과를 해석/정리한다**.

---

## Anomaly Detection 위치 (확정)

| 선택 | 설명 | 채택 |
|------|------|------|
| **A (채택)** | 외부 detector / 운영 시스템이 abnormal window를 준다고 가정 | ✅ 1편 기본 |
| B | Rule-based eventizer 사용, "새로운 detector"라고 주장 안 함 | 보조 |
| C | Retrieval confidence를 anomaly score로 보조 사용 | 보조 실험 |

**1편에서 anomaly detection은 메인 contribution이 아니다.**

---

## Scope Lock

> **이 scope 결정은 최종이다. 더 이상 방향을 바꾸지 않는다.**
> 추가 아이디어는 2편/3편으로 분리한다.

---

## 다음 행동

### 즉시 (이번 주)
- [ ] 센서 시계열 데이터 파일 위치 확인
- [ ] 샘플 데이터 로드하여 컬럼 구조 파악
- [ ] Eventizer 규칙 5개 구현 (sustained_high, oscillation, drift, spike, stuck)

### 단기 (2~4주)
- [ ] Pseudo-labeling 파이프라인 구축
- [ ] BM25 baseline 구현
- [ ] Hierarchy-aware hard negative 샘플링 구현
- [ ] Bi-encoder 학습 (contrastive loss only)

### 중기 (1~2개월)
- [ ] Pilot set 50~100개 구축
- [ ] 5~7개 baseline 비교 실험
- [ ] 1편 논문 초안 작성

---

## 핵심 문장 (논문 초록용)

> 기존 연구는 시계열 이상을 텍스트로 설명해 LLM이 판정하게 했다면, 본 연구는 rule-initialized sensor event와 정비 텍스트를 동일 의미공간에 정렬하여 **정비 근거 검색**을 수행한다. 특히 장비/모듈/부품 계층을 반영한 hierarchy-aware hard negative mining을 통해 **wrong-device contamination**을 감소시킨다.

---

## 관련 문서

- [[./2026-04-20--node-definition-strategy|이전: 노드 정의 전략]]
- [[../paper_d_data_constraint_and_node_design|데이터 제약사항과 노드 설계]]
- [[../paper_d_paper_strategy|논문 전략]]
- [[../paper_d_research_proposal|박사 연구계획서]]

---

**Written**: 2026-04-20  
**Status**: Final strategy confirmed, ready for implementation
