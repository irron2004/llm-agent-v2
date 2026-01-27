# RAG 에이전트 성능 진단 보고서

## 개요

RAGAS(RAG Assessment) 프레임워크 기반 에이전트 성능 평가 결과 및 개선 방안

---

## 1. 평가 지표 현황 분석

### 1.1 지표별 결과

| 지표 | 평균 점수 | 평가 |
|------|----------|------|
| **Answer Relevancy** | ~0.79 | ✅ 양호 |
| **Faithfulness** | ~0.36 | ❌ 취약 (가장 심각) |
| **Context Precision** | ~0.58 | ⚠️ 개선 필요 |

### 1.2 분포 분석

- **Answer Relevancy**: 0.7-0.9 구간에 집중 (안정적)
- **Faithfulness**: 0에 가까운 샘플 다수, 편차 큼
  - 15개 샘플 중 **5개(~33%)가 Faithfulness = 0**
- **Context Precision**: 0인 경우 3건 (검색 완전 실패)

### 1.3 핵심 문제

**근거 충실도(Faithfulness)**가 가장 취약한 지표로, 답변이 제공된 문맥과 불일치하는 할루시네이션 문제가 심각함.

---

## 2. 문제 사례 및 오류 패턴

### 2.1 관련 문맥 미검색 (Context 미포함)

**증상**: 검색된 문맥이 질문과 전혀 무관하거나 부재

**사례**:
- 샘플 #1: "SUPRA V에서 SUPRA Np로 개조된 설비의 호기명?" → Context Precision 0, Faithfulness 0
- 샘플 #10, #13: EPA404 점검 이력 → 지식베이스에 없음

**영향**: Faithfulness 0점으로 직결

### 2.2 쿼리 부정확 및 부분적 문맥 미활용

**증상**: 복잡한 질문에 대해 불완전한 처리

**사례**:
- 샘플 #2: "SUPRA Np Issue를 GCB 기반 정리" → Context Precision 1.0이지만 **Answer Relevancy 0.51** (가장 낮음)
- 샘플 #11: 관련 문맥 있음(Context 1.0)에도 Faithfulness 0.4

**원인**: 추가 하위 질의 생성 없이 단순 요약에 그침

### 2.3 할루시네이션 및 근거 불일치

**증상**: 문맥에 없는 내용을 답변에 포함

**사례**:
- 샘플 #7: Screw 토크 스펙 질문 → 문헌에 없는 수치 단정적 언급 (Faithfulness 0, Relevancy 0.85)
- 샘플 #14: APC Sensor 부품 번호 → 문맥과 불일치하는 번호 제시

**패턴**: 수치/고유명사 답변에서 특히 발생

### 2.4 Route 분류 및 처리 흐름 오류

**증상**: ST_Gate/MQ 분기에서 오판

**추정 원인**:
- 모호한 질문에 대해 `need_st` 미발동
- 불필요한 세분화로 문맥 분산
- 잘못된 route 분류로 답변 형식/검색 우선순위 어긋남

---

## 3. 에이전트 구조 진단 및 개선안

### 3.1 질의 라우팅(Route) 및 다중 쿼리(ST_Gate) 흐름 개선

#### 현재 구조
```
질문 → Route(setup/ts/general) → MQ → ST_Gate(need_st/no_st) → ST_MQ → 검색
```

#### 개선 방향

**1. Route 분류기 정확도 향상**
- 경계 사례 처리 규칙 보강
- 예: "로그 이력 정리" → TS vs General 명확화

**2. ST_Gate 판단 로직 정교화**
- 모호/광범위 질문 → `need_st` 임계 기준 조정
- 키워드 기반 규칙 추가로 LLM 판단 보완

### 3.2 문맥 검색(Retrieval) 정확도 개선

#### 개선 방향

**1. 검색 쿼리 생성 점검**
- MQ 프롬프트 조정
- 도메인 용어 사전/키워드 보강

**2. 지식베이스 커버리지 확인**
- 답변 불가 영역 명확화
- "해당 정보 없음" 정책 일관화

**3. 필터링/재검색 전략**
- Re-ranker 임계치 조정
- refine_queries_node 활용 강화

### 3.3 답변 생성 단계 통제 강화

#### 개선 방향

**1. 근거 사용 강제**
- "제시된 문맥에 없는 내용은 추측하지 말 것" 명시
- 수치/용어는 문맥에서 인용 요구

**2. Judge 검증 활용**
- faithful=false 시 Refine 또는 "근거 부족" 답변 유도

**3. "No Answer" 전략 일관화**
- Context Precision 낮을 때 추론 포기 정책
- 임계 점수 설정

---

## 4. 평가 방법 및 RAGAS 구성 개선

### 4.1 평가 지표 보강

- **골든셋 활용**: 정답과의 텍스트 유사도 (BLEU/ROUGE)
- **Context Recall**: 실제 필요한 문서 검색 여부 평가
- **GPT-4 채점**: 답변 정확도 직접 평가

### 4.2 LLM 평가 안정성 개선

**문제점**:
- LLM Timeout으로 Faithfulness NaN 발생
- "LLM returned 1 generation instead of requested 3" 경고

**개선안**:
- max_tokens 조절, 프롬프트 길이 축소
- 안정적인 평가 전용 모델 사용 (GPT-4)
- strictness 조정

### 4.3 정성적 분석 병행

- 오류 유형별 자동 태깅
- 규칙 기반 또는 LLM 기반 오류 분류

---

## 5. 코드 기반 추가 분석 (담당자 의견)

### 5.1 실제 코드와의 비교

| 보고서 언급 | 실제 코드 현황 | 일치 여부 |
|------------|---------------|----------|
| ST_Gate 존재 | `st_gate_v1.yaml` - need_st/no_st 판단 | ✅ |
| Judge 노드 | `judge_node()` - faithful/issues/hint 반환 | ✅ |
| 재시도 전략 | 3단계 재시도 구현됨 | ✅ |
| Refine queries | `refine_queries_node()` - hint 활용 | ✅ |

### 5.2 현재 재시도 전략 (이미 구현됨)

```python
# langgraph_agent.py:1377-1409
def should_retry(state: AgentState) -> Literal["done", "retry", "retry_expand", "retry_mq", "human"]:
    """
    1st unfaithful (attempt 0→1): retry_expand - 문서 확장 (5→10)
    2nd unfaithful (attempt 1→2): retry - 쿼리 정제
    3rd unfaithful (attempt 2→3): retry_mq - MQ 전체 재생성
    """
```

**의견**: 3단계 재시도 전략이 이미 잘 구현되어 있음. 보고서의 권장사항과 일치.

### 5.3 Judge 프롬프트 현황

```python
# langgraph_agent.py:128-153
DEFAULT_JUDGE_SETUP = """
# 역할
설치/세팅 답변이 질문과 검색 증거에 충실한지 판정한다.
...
"""
```

**문제점**: Judge 프롬프트가 코드 내 상수로 정의됨 (YAML 외부화 안됨)

**권장**: `prompts/judge_setup_v1.yaml`, `judge_ts_v1.yaml`, `judge_general_v1.yaml`로 분리

### 5.4 ST_Gate 프롬프트 분석

```yaml
# st_gate_v1.yaml 현재 규칙
- Ambiguous/broad scope/abbreviations only/multiple modules/no validation criteria → need_st
- Very specific (module/numeric values/buttons/steps included) → no_st
```

**개선 필요**:
- "GCB 기반 정리" 같은 광범위 요청에 대한 명시적 규칙 추가
- Few-shot 예시 추가

### 5.5 우선순위별 개선 권장사항

#### 즉시 적용 (Low Effort, High Impact)

| 순위 | 조치 | 파일 | 영향 |
|------|------|------|------|
| 1 | **답변 프롬프트에 근거 제한 강화** | `*_ans_*.yaml` | Faithfulness 향상 |
| 2 | **ST_Gate Few-shot 예시 추가** | `st_gate_v1.yaml` | 복잡 질문 처리 개선 |
| 3 | **Judge 프롬프트 YAML 외부화** | 새 파일 생성 | 유지보수성 |

#### 단기 개선 (Medium Effort)

| 순위 | 조치 | 영향 |
|------|------|------|
| 4 | **No Answer 정책 일관화** | 할루시네이션 감소 |
| 5 | **MQ 프롬프트 도메인 용어 보강** | Context Precision 향상 |
| 6 | **refine_queries 활용도 강화** | 재시도 효과 개선 |

#### 장기 고려 (High Effort)

| 순위 | 조치 | 영향 |
|------|------|------|
| 7 | **골든셋 기반 평가 체계** | 정확한 성능 측정 |
| 8 | **임베딩 기반 라우팅** | 분류 일관성 |
| 9 | **지식베이스 커버리지 확장** | 답변 가능 범위 확대 |

---

## 6. 결론

### 6.1 핵심 발견

1. **Faithfulness(0.36)가 가장 취약** - 답변이 문맥에 근거하지 않는 할루시네이션 문제
2. **Context Precision(0.58) 불안정** - 검색 품질 개선 필요
3. **Answer Relevancy(0.79)는 상대적으로 양호** - 질문 의도 파악은 됨

### 6.2 근본 원인

- 질문 세분화(ST_Gate) 판단 미흡
- 답변 생성 시 근거 제한 약함
- 문맥 미검색 시 "No Answer" 정책 비일관

### 6.3 권장 조치 요약

```
1단계: 프롬프트 개선
├── 답변 프롬프트에 "문맥 외 추측 금지" 명시
├── ST_Gate에 Few-shot 예시 추가
└── Judge 프롬프트 YAML 외부화

2단계: 로직 개선
├── No Answer 정책 일관화
├── refine_queries 활용 강화
└── 도메인 용어 사전 구축

3단계: 평가 체계 구축
├── 골든셋 기반 정확도 평가
└── 오류 유형별 자동 분류
```

### 6.4 기대 효과

- **Faithfulness**: 0.36 → 0.6+ 목표 (할루시네이션 감소)
- **Context Precision**: 0.58 → 0.7+ 목표 (검색 품질 향상)
- 전체적인 RAG 시스템 신뢰성 향상
