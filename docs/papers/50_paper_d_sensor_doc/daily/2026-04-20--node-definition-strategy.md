---
date: 2026-04-20
type: daily-log
topics: [node-definition, cross-modal-alignment, pattern-detection, research-direction]
status: completed
---

# 2026-04-20 — 노드 정의 전략 확정 및 Cross-Modal Alignment 방향 설정

## 오늘의 목표
1. [x] 시계열 데이터 라벨 부재 문제 해결 방안 검토
2. [x] 노드 정의 방법론 확정
3. [x] Cross-modal alignment 연구 방향 설정
4. [x] Daily log 폴터 구조 생성

## 완료한 것

### 1. 데이터 제약사항 재확인
- [x] 시간 기반 매칭 불가 확인
  - 정비 로그가 글로벌 (전세계 SUPRA Vplus)
  - 특정 장비의 시계열과 시간 매칭 어려움
- [x] 라벨(Gold Link) 부재 확인
- [x] 의미(semantic) 매칭으로 전환 결정

### 2. 노드 설계 전략 확정
**핵심 결정**: "시계열에 노드를 정의하는 것 자체가 이상 탐지"

노드 정의:
```json
{
  "sensor": "APC_Position",
  "pattern": "sustained_high",
  "features": {
    "mean_value": 91.2,
    "deviation_from_normal": 36.2,
    "duration_sec": 42,
    "trend": "increasing",
    "saturation": 0.88
  },
  "inter_sensor": {
    "pressure_stable": false,
    "actuator_current_normal": true
  }
}
```

초기 패턴 5~10개:
- `sustained_high/low`
- `drift_up/down`
- `oscillation`
- `spike`
- `stuck`
- `step_response_delay`
- `out_of_range`

### 3. Cross-Modal Alignment 방향 설정
**제안된 연구 제목**: 
> "Weakly-Supervised Cross-Modal Alignment of Sensor Events and Maintenance Knowledge for Industrial Diagnosis"

핵심 아키텍처 (4블록):
1. **Sensor Event Extractor** — Rule 기반 이벤트 추출
2. **Rule-grounded Event Encoder** — 이벤트 임베딩
3. **Maintenance Text Encoder** — 정비 로그 chunking
4. **Cross-modal Alignment + Retrieval** — 공동 임베딩 공간

Loss 함수:
- L_align (contrastive)
- L_hard_negative (hard negative mining)
- L_hierarchy (equipment/module/part 계층)
- L_anom (anomaly detection multitask)

### 4. RAAD-LLM과의 차별화
| 항목 | RAAD-LLM | Paper D 방향 |
|------|----------|--------------|
| 핵심 | 통계 이상탐지 → LLM 판정 | 센서 이벤트 ↔ 정비 텍스트 정렬 |
| 구조 | frozen LLM + prompt | representation learning |
| novelty | domain knowledge + LLM | cross-modal alignment |

## 핵심 발견/아이디어

> **"Rule 기반 센서 이벤트 표현과 정비 텍스트의 공동 임베딩을 통한 산업 이상탐지 및 정비 근거 검색"**

- 기존 연구가 시계열 이상을 텍스트로 설명해 LLM이 판정하게 했다면, 본 연구는 룰 기반 센서 이벤트와 정비 텍스트를 동일 의미공간에 정렬하여 이상탐지와 정비 근거 검색을 동시에 수행한다.

- 시간 매칭이 안 되는 제약이 오히려 **더 범용적인 접근** (global knowledge base 활용)으로 전환됨

## 장애물/문제

1. **시계열 데이터 접근**: 아직 실제 APC 센서 데이터 파일 위치 확인 필요
   - `/home/llm-share/datasets/` 또는 FDC DB 가능성
   - 확인 필요: 컬럼 구조, 샘플링 주기, 기간

2. **Rule의 robustness**: rule 기반 이벤트 추출이 잘못되면 전체가 잘못됨
   - 해결책: rule uncertainty를 explicit하게 모델링하거나, 점진적으로 neural event extractor로 대체

3. **Evaluation**: 라벨이 없어 traditional metric (Cause Hit@K) 계산 어려움
   - 대안: expert evaluation (소량), retrieval diversity, confidence calibration

## 다음 단계

### 우선순위 1: 시계열 데이터 확보
- [ ] APC_Position, APC_Pressure, APC_SetPoint 데이터 파일 위치 확인
- [ ] 샘플 데이터 로드하여 컬럼 구조 파악
- [ ] 샘플링 주기 확인 (1초? 1분?)

### 우선순위 2: Pattern Detector 구현
- [ ] `sustained_high` 규칙 구체화 (threshold, duration, stability 파라미터)
- [ ] `oscillation` 규칙 구현 (FFT 기반)
- [ ] Window sliding 로직 구현

### 우선순위 3: Pseudo-Labeling 파이프라인
- [ ] Pattern → 텍스트 쿼리 변환 (Eventizer)
- [ ] ES 검색 연동 (BM25)
- [ ] Silver label 생성 및 검토

## 관련 문서

- [[../paper_d_data_constraint_and_node_design|데이터 제약사항과 노드 설계]]
- [[../paper_d_research_proposal|박사 연구계획서]]
- [[../paper_d_architecture|APC 에이전트 아키텍처]]
- [[../paper_d_interim_report|중간 진행 보고서]]

## 메모

- Oracle 분석 결과: Multi-scale hybrid interval nodes + self-supervised embeddings + clustering 권장
- Explore agent: 시계열 데이터는 repo에 없고 외부 경로에 존재
- Daily log 폴터 생성 완료: `docs/papers/50_paper_d_sensor_doc/daily/`

---
**Written**: 2026-04-20  
**Status**: Day's work completed, ready for data acquisition phase
