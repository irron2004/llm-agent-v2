# Paper D — Sensor-Document Linking Agent: Ideation

> 작성일: 2026-04-14
> 상태: 아이디어 정리 완료, 추가 서베이 및 PoC 설계 필요

---

## 1. 문제 정의

반도체 PE(Process Engineering) 환경에서:

- **센서 데이터**: APC position, pressure, valve current, actuator 상태 등 실시간 수집
- **문서 데이터**: Setup Manual, SOP, 정비 이력, TSG 등
- **Gap**: 센서 이상이 감지되어도, 어떤 문서가 해당 이상에 맞는지 수동으로 찾아야 함

**목표**: 센서 이상 패턴 → 관련 문서 자동 검색 → 근거 기반 트러블슈팅 답변 생성

---

## 2. 접근 방식: 4단계 파이프라인

end-to-end 학습보다, **센서 이상을 이벤트/증상으로 변환 → 문서 검색 → 근거 기반 답변** 구조가 현실적.

### Stage 1: 센서 이벤트화 (Sensor → Event)

센서 원시 시계열을 진단 가능한 **구조화된 이벤트**로 변환.

**추출 대상**:
- 장비/챔버/레시피 step/운전 모드
- setpoint 대비 actual 오차 크기 및 지속시간
- 포화 여부 (0%/100% 근처 고정)
- step response 지연
- 진동/oscillation 패턴
- 관련 센서 상관관계 (pressure, actuator current, valve state)
- 최근 PM/교정 이력

**이벤트 예시**:
```
APC actual position이 92% 부근에 고정,
setpoint는 45→60%로 변했지만 35초 이상 추종 실패,
pressure oscillation 증가,
마지막 calibration 78일 경과
```

**구현 전략**:
- Phase 1: 규칙 기반 (threshold + duration + correlation)
- Phase 2: 비지도 이상탐지 (Anomaly Transformer)
- Phase 3: 도메인 지식 결합 (RAAD-LLM 방식)

### Stage 2: 문서 구조화 (Document → Structured Knowledge)

flat chunking(500자)보다 **의미 단위 구조화**가 효과적.

**구조화 단위**:
| 필드 | 설명 |
|------|------|
| 절차 step | SOP 내 개별 단계 |
| 증상 (symptom) | 어떤 이상 현상 |
| 원인 (cause) | 왜 발생하는지 |
| 조치 (action) | 무엇을 해야 하는지 |
| 안전 주의사항 | 위험 요소 |
| 관련 부품/컴포넌트 | APC, actuator, controller 등 |
| 적용 조건 | recipe step, mode, chamber type |

**관련 연구**:
- **SOPRAG**: Entity / Causal / Flow graph 구조로 SOP 검색 개선
- **GraphRAG**: entity/relation/community summary로 넓은 문맥 질문에 강함
- **ManuRAG**: 표/수식/이미지 포함 멀티모달 제조 문서 처리

### Stage 3: 하이브리드 검색 (Hybrid + Metadata + Graph Retrieval)

단일 vector search가 아닌 복합 검색:

| 검색 방식 | 역할 |
|-----------|------|
| BM25/키워드 | APC, position mismatch, calibration, stuck, fault code 등 exact match |
| Dense embedding | 의미적으로 유사한 증상 검색 |
| Metadata filter | 장비 타입, 챔버, 부품, 운전 모드, step |
| Graph/Ontology traversal | APC → pressure control → valve sticking → calibration SOP |

**Ontology 구조**:
```
sensor → component → symptom → failure mode → action → document
```

이 공통 ontology 없이는 embedding이 좋아도 retrieval이 자주 빗나감.

### Stage 4: 근거 기반 답변 생성 (Evidence-Grounded Generation)

LLM은 "생성기"가 아닌 **"근거 편집기"**로 사용.

**답변 형식**:
1. 가능성 높은 원인 1~3개
2. 근거 문서 + 해당 section/step
3. 즉시 점검 순서
4. 위험도 / 장비 정지 필요 여부
5. 확신도 (confidence)
6. 추가 필요한 센서/로그

**원칙**: 문서를 근거로 요약/정리하게 해야 하며, 검색 근거가 약하면 "추가 점검 필요"로 낮춰 답하도록.

---

## 3. APC Position 이상 — End-to-End 예시

### 입력
> APC position 센서 값이 이상하다. 관련 문서를 연결해서 답해라.

### 파이프라인 실행

**Step 1 — 센서 분석**:
- setpoint 대비 tracking error 30% 초과
- 40초 지속
- pressure oscillation 동반
- valve current 낮음
- 최근 calibration overdue

**Step 2 — 이벤트 요약**:
```
APC position tracking failure with high saturation and pressure oscillation
during pressure stabilization
```

**Step 3 — 검색 조건**:
- component = APC
- symptom = position mismatch / stuck / oscillation
- mode = automatic pressure control
- related actions = calibration / actuator inspection / valve cleaning

**Step 4 — 문서 검색 결과**:
- Setup Manual: APC calibration section
- SOP: "APC stuck or unstable position" 절차
- 정비 이력: 동일 증상 과거 사례

**Step 5 — 최종 답변**:
- **원인**: calibration drift, valve sticking, actuator issue
- **우선 점검**: manual jog test → feedback consistency 확인 → calibration SOP 수행
- **근거**: 문서 section, 과거 유사 정비 case
- **확신도**: 중간~높음

---

## 4. 실무 적용 핵심 3가지

### 4.1 공통 Ontology 구축
```
sensor → component → symptom → failure mode → action → document
```
- 센서와 문서를 연결하는 공통 어휘 체계
- 없으면 embedding만으로는 retrieval 정확도 부족
- SOPRAG/GraphRAG가 구조를 강조하는 이유

### 4.2 정비 이력으로 Weak Label 생성
- "어떤 센서 패턴 뒤에 어떤 SOP/정비가 실제로 사용됐는가"
- 최고의 학습 데이터 = 실제 정비에서 사용된 (센서패턴, 문서) pair
- 시계열-언어 정렬 연구의 핵심도 이런 paired supervision

### 4.3 답변에 근거 문서 + Confidence 필수
- 정비/운전 조치에서 hallucination은 위험
- 검색 근거가 약하면 confidence를 낮춰서 표현
- RAG의 provenance + 외부 지식 업데이트 철학과 일치

---

## 5. 관련 연구 목록

| 논문/시스템 | 핵심 아이디어 | 관련성 |
|-------------|---------------|--------|
| **RAAD-LLM** | LLM + RAG 기반 적응형 이상탐지, 도메인 지식 결합 | Stage 1 이벤트화 |
| **CLaSP** | 시계열-자연어 공통 임베딩 (contrastive) | Stage B 학습 |
| **LaSTR** | 시계열-언어 검색 정렬 | Stage B 학습 |
| **SOPRAG** | SOP 구조 기반 RAG (Entity/Causal/Flow graph) | Stage 2 문서 구조화 |
| **GraphRAG** | Entity/Relation/Community summary 기반 검색 | Stage 3 검색 |
| **ManuRAG** | 멀티모달 제조 문서 RAG | Stage 2 문서 구조화 |
| **Anomaly Transformer** | 비지도 시계열 이상탐지 | Stage 1 이벤트화 |

---

## 6. 한 줄 요약

> **"센서 시계열을 텍스트/이벤트 공간으로 올리고, 그 이벤트를 키로 문서를 검색하는 RAG agent"**
>
> 처음엔 rule-based eventizer + hybrid RAG로 시작하고,
> 데이터가 쌓이면 sensor-window ↔ document-chunk contrastive retriever를 학습.
