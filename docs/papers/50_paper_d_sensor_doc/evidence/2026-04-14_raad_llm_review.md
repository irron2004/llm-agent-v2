# RAAD-LLM 논문 리뷰

> 논문: RAAD-LLM: Adaptive Anomaly Detection Using LLMs and RAG Integration
> 리뷰일: 2026-04-14
> Paper D (Sensor-Document Linking Agent) 관점에서의 실무 적용성 포함

---

## 1. 기본 정보

- **제목**: RAAD-LLM: Adaptive Anomaly Detection Using LLMs and RAG Integration
- **위치**: arXiv (기존 AAD-LLM의 확장판, substantial text overlap 주석 있음)
- **핵심 아이디어**: LLM + RAG를 결합한 적응형 시계열 이상탐지
- **Novelty 주의**: 저자 스스로 AAD-LLM 확장판이라 설명, arXiv에 substantial text overlap admin note 존재

---

## 2. 핵심 구조

- **중심 모델**: frozen Meta Llama 3.1 8B
- **센서 처리**: SPC로 정상 구간 잡기 → window 단위 분할 → baseline window와 current window의 통계값을 텍스트 템플릿에 주입
- **RAG**: CSV 기반 지식베이스에서 z-score 비교 정보를 끌어옴
- **최종 판정**: LLM 출력에 binarization rule 적용 → anomaly/non-anomaly 결정
- **Prompt 형식**: "설명하지 말고 deviation 유무만 출력하라" — yes/no 분류

**해석**: LLM이 복잡한 시계열 표현을 학습하는 모델이라기보다, 규칙과 컨텍스트를 소비하는 언어 인터페이스로 쓰이고 있음.

---

## 3. 좋은 점

### 3.1 적응형 baseline 업데이트
- 첫 window를 정상 baseline으로 두고, 현재 window가 비정상이 아니면 baseline에 합쳐 normal 정의를 갱신
- concept drift를 정교하게 푸는 건 아니지만, 현장 데이터가 조금씩 변하는 상황에 현실적

### 3.2 도메인 지식 활용
- maintenance log와 operator knowledge를 prompt에 녹이는 방식
- 제조/정비 맥락에 잘 맞음

### 3.3 성능 개선
| 데이터셋 | 모델 | Accuracy | F1 |
|----------|------|----------|-----|
| Use-case | AAD-LLM | 0.71 | 0.77 |
| Use-case | **RAAD-LLM** | **0.89** | **0.92** |
| SKAB | AAD-LLM | 0.58 | 0.56 |
| SKAB | **RAAD-LLM** | **0.72** | **0.74** |

---

## 4. 아쉬운 점

### 4.1 RAG의 폭이 좁음
- 기본 RAG는 문서 chunk나 SOP section 검색이 아니라, **CSV 기반 z-score comparison retrieval** 중심
- 제목만 보면 "LLM + RAG 기반 산업 문서 에이전트"처럼 보이지만, 실제 구현은 수치 비교 보조기에 가까움
- **RAAD-LLMv2**: LlamaIndex로 domain context를 vector store에서 가져오도록 확장했지만, 성능은 오히려 use-case와 SKAB 둘 다 RAAD-LLM보다 낮음

### 4.2 수작업 의존성
- raw domain context를 사람이 다시 정리해 prompt에 맞게 재구성 필요 (논문 자인정)
- "no training/no fine-tuning"은 맞지만, feature 선택, context engineering, rule 설계, binarization 설계에 사람 손이 많이 들어감
- 모델 가중치를 안 바꿨다고 해서 시스템 통합 비용까지 낮은 건 아님

### 4.3 평가 엄밀성
- 내부 use-case: screen pack failure 두 번의 downtime event에서 65시간 semi-labeled run-to-failure 데이터
- 580개 센서 중 일부 공정 변수만 모델 입력으로 사용
- Use-case baseline: "모든 관측을 positive class로 예측하는 모델"
- SKAB 비교: 평균이 아니라 best 5 model runs의 평균으로 제시 → 전체 분산/안정성 보고 부족

### 4.4 오탐 경향
- SKAB에서 F1 0.74로 상위권, MAR 11.43%로 놓치는 비율 낮음
- 그러나 **FAR 42.05%** — false alarm 부담 큼
- "safety-critical 환경에서 MAR이 중요"라고 해석하지만, 실제 운영팀 입장에서 잦은 오탐도 비용
- **"놓치지 않는 쪽"으로 강하게 치우친 detector**

### 4.5 다변량 처리
- multivariate anomaly detection을 말하지만, 실제로는 **univariate로 쪼개 각 변수별 처리 → 프롬프트와 correlation-based binarization으로 결합**
- "LLM이 센서 간 동적 상호작용을 end-to-end로 학습했다"기보다 독립 처리 + 규칙 기반 결합

---

## 5. Paper D 관점 평가

### 직접 도움 되는 부분
- 센서 → 증상/event 텍스트화 방식
- 도메인 규칙 반영 방식
- 정상 기준의 적응적 업데이트

### 부족한 부분
- 논문 자체는 "deviation 여부만 출력" 설계
- 기본 RAG도 **문서 retrieval이 아니라 z-score lookup 중심**
- 정비 절차를 근거 문서와 함께 설명하는 agent로는 부족

### Paper D에서의 활용 방안
> RAAD-LLM을 **앞단 anomaly/event detector**로만 쓰고,
> 뒤에 별도의 **문서 RAG + maintenance log retrieval + citation-based answerer**를 붙이는 구조

---

## 6. 최종 평가

| 항목 | 평가 |
|------|------|
| PoC 가치 | 높음 — 산업 현장 감각 있음 |
| 핵심 발상 | "LLM을 시계열 학습기가 아닌 도메인 지식 소비 엔진으로" — 현실적 |
| 새로움 | 약함 — AAD-LLM 확장, substantial overlap |
| 평가 엄밀성 | 보통 이하 — baseline 약함, best-5 average |
| Paper D 해결력 | 부분적 — 앞단 이벤트화만 참고 가능 |

**한 줄 요약**: 센서-문서 에이전트의 완성형 해법이 아니라, **앞단 이벤트화 모듈의 아이디어 참고용** 논문.
