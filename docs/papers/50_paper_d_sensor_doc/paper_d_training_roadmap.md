# Paper D — 학습 로드맵 (Training Roadmap)

> 작성일: 2026-04-14

센서-문서 연결 에이전트의 학습은 **LLM fine-tuning이 1순위가 아님**.
아래 A→B→C→D 순서가 가장 효율적.

---

## 단계 A: 학습 없이 먼저 붙이기 (Rule-Based Eventizer + Hybrid RAG)

### 방법
1. 센서 이벤트를 **규칙 또는 간단한 모델**로 텍스트화
   ```
   APC position high saturation
   setpoint tracking failure
   pressure oscillation after APC movement
   ```
2. 이 텍스트 + 장비 메타정보로 문서를 **hybrid search**

### 기대 효과
- 현장 PoC로 충분히 유효
- RAG 자체가 외부 지식 검색을 전제로 하므로, 이 단계만으로도 동작

### 필요 리소스
- 센서 이상 판정 규칙 (threshold, duration, correlation)
- 장비/부품 메타데이터
- 문서 인덱스 (기존 llm-agent-v2 인프라 활용 가능)

---

## 단계 B: Sensor-Document Contrastive Retriever 학습

### 핵심: 센서 패턴과 문서를 같은 임베딩 공간에 정렬

### 학습 데이터 구성
| 구성요소 | 설명 |
|----------|------|
| 입력 | 장애 직전 30초~5분 센서 window |
| 정답 문서 | 실제 정비 때 사용된 SOP/manual section/정비기록 |
| Positive pair | (sensor_window, 해결에 사용된 문서 chunk) |
| Negative pair | 비슷하지만 틀린 문서, 같은 부품이지만 다른 증상 문서 |

### 모델 아키텍처
```
E_s(sensor_window)  →  시계열 인코더  →  shared embedding space
E_d(document_chunk) →  문서 인코더    →  shared embedding space
```

- **Contrastive loss**: 해당 센서 이벤트와 맞는 문서가 가깝게
- CLaSP, LaSTR와 유사: "자연어 → 시계열 검색"을 뒤집어서 **"시계열 이벤트 → 문서 검색"**

### Hard Negative Mining
- 같은 장비/부품이지만 다른 증상의 문서
- 비슷한 센서 패턴이지만 다른 원인의 사례
- 시간적으로 가까운 비장애 구간

---

## 단계 C: Multi-Task 학습 (이벤트 분류 + Retrieval)

### Retrieval loss만으로는 부족 → 이상 분류를 같이 학습

### Loss 구조
```
L_total = L_anomaly + L_fault_type + L_retrieval
```

| Loss | 역할 |
|------|------|
| L_anomaly | 이상/정상 분류 |
| L_fault_type | 이상 유형 분류 (stuck, drift, slow response, oscillation, sensor fault) |
| L_retrieval | 문서 retrieval ranking |

### 효과
- "비슷한 파형"이 아닌 **"정비 의미가 맞는 파형"**을 더 잘 학습
- 비지도 이상탐지 베이스 → 도메인 지식 결합으로 고도화

---

## 단계 D: 답변 스타일 SFT (마지막)

### LLM fine-tuning이 필요하다면 제일 마지막

### 학습 데이터
내부 QA 예시를 모아서 아래 **스타일만** 학습:
- 근거 문서 인용
- 원인 후보 정렬
- 작업 순서 정리
- 확신도/안전 경고 표현

### 주의
- **문서 retrieval 없이 LLM만 fine-tune하는 방식은 비추**
- 장비 상태와 문서가 계속 변하므로 retrieval 기반이어야 업데이트 가능
- 출처 제시도 retrieval 없이는 불가

---

## 단계별 의존성 요약

```
A (Rule-based PoC)
  ↓ 데이터 축적
B (Contrastive Retriever)
  ↓ 라벨 확보
C (Multi-Task: 분류 + Retrieval)
  ↓ QA 데이터 확보
D (Answer Style SFT)
```

각 단계는 독립적으로도 가치가 있으며, 데이터가 쌓일수록 다음 단계로 진행.
