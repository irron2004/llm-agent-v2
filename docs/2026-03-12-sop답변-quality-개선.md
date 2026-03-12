# SOP 답변 Quality 개선

## 목적

SOP 질문에 대한 현재 답변은 문서 검색 자체보다, 답변 구조와 절차 중심성이 부족한 문제가 더 크다.  
이 문서는 `sop_only` 평가 결과를 기준으로 현재 문제를 정리하고, `Work Procedure` / `Workflow` 중심 답변으로 바꾸기 위한 개선 방향을 정의한다.

기준 데이터:
- 검색+답변 preview: `.sisyphus/evidence/2026-03-11_sop_filter_eval/sop_only_results.jsonl`
- 답변 품질 요약 리포트: `.omc/scientist/reports/2026-03-12_sop_answer_quality_report.md`

---

## 현재 상태 요약

### 1. 검색은 대체로 충분히 좋다

- raw 기준 `doc_hit = 75/79`
- 이 중 4건(`idx 75~78`)은 실제 검색 miss라기보다 `gold_doc` 문자열 정규화 차이로 인한 평가 false negative에 가깝다.
- retrieved doc를 보면 `global_sop_supra_xp_sw_all_sw_installation_setting`가 정확히 잡혀 있다.

해석:
- SOP-only 조건에서는 문서 검색 자체가 핵심 병목이 아니다.
- 현재 사용자가 느끼는 품질 문제는 주로 답변 생성 단계에서 발생한다.

### 2. 답변은 "문서 요약형"으로 흐르고 있다

직접 확인한 preview 기준으로, 많은 답변이 다음 특징을 가진다.

- 안전/보호구/PPE를 길게 먼저 설명함
- 준비 항목과 일반 설명이 앞쪽을 차지함
- 실제 작업 단계가 뒤로 밀리거나, 절차보다 개요처럼 보임
- `Work Procedure` / `Workflow`를 최우선으로 재구성한 느낌이 약함

대표 예:
- `idx 0`: EFEM PIO SENSOR BOARD 교체
- `idx 1`: EFEM Controller 교체
- `idx 2`: EDA PC 교체

이 답변들은 틀렸다고 보긴 어렵지만, 사용자가 기대하는 "실제 작업 순서 중심 답변"과는 거리가 있다.

### 3. 형식 품질이 불안정하다

확인된 문제:
- 이모지 번호 사용: `1️⃣`, `2️⃣`
- 테이블/불릿/평문 혼용
- 인용 형식 혼용: `[1]`, ``, `【[1]】`
- 오타/형식 오류:
  - `EFEM -> EFIM`
  - `0️⃣`으로 절차 시작
  - `[…]` placeholder 출력
  - markdown bold 깨짐

해석:
- 현재 문제는 "정답 문서를 못 찾는다"보다 "답변 출력이 제품 수준으로 정제되어 있지 않다"에 가깝다.

---

## 왜 현재 답변이 절차 중심으로 안 보이는가

### 1. 한국어 `setup_ans_v2`가 실제 v2가 아니다

현재 한국어 setup 프롬프트:
- [setup_ans_v2.yaml](/home/hskim/work/llm-agent-v2/backend/llm_infrastructure/llm/prompts/setup_ans_v2.yaml)

현 상태:
- `setup_ans_v2`가 사실상 `setup_ans_v1`과 동일
- 영어/일본어/중국어 v2처럼 구조화된 section 강제가 없음

즉, 한국어 답변은 아직도 아래 수준의 느슨한 지시만 받고 있다.
- 단계별 절차로 답변
- [1] 형식 인용
- 참고문헌 추가

이 정도로는 모델이 `Work Procedure`를 중심으로 답변을 구성하도록 강제할 수 없다.

### 2. `Work Procedure` / `Workflow` 우선순위가 한국어 프롬프트에 없다

영어 v2에는 아래 규칙이 있다.
- `Work Procedure`
- `Workflow`
- procedural sections
를 우선 사용

하지만 한국어 v2에는 이 우선순위가 없다.

결과:
- 모델이 REFS 전체를 보고
- Safety / Scope / Purpose / Checklist / Procedure를 임의 비중으로 섞는다.

### 3. 답변 구조가 고정되지 않았다

현재는 모델이 매 질문마다 아래를 제멋대로 고른다.
- 테이블
- 이모지 절차
- 불릿
- 일반 설명

사용자 기대는 "절차를 물으면 작업 순서를 자세히"인데,
현재 출력은 "SOP 관련 정보를 그럴듯하게 정리"하는 쪽에 가깝다.

### 4. REFS 자체의 우선 정렬도 절차 중심이 아니다

현재 답변 단계에서는 REFS 전체를 한 번에 받아 생성한다.
따라서 프롬프트가 약하면 모델은 절차보다 다음을 먼저 소비하기 쉽다.

- cover/overview
- safety
- scope
- general note

즉, retrieval이 괜찮아도 answer prompt가 약하면 절차 중심 답변이 나오지 않는다.

### 5. REFS에 noise가 많다 — 절차 청크의 집중도 문제

현재 answer_node는 20개 청크를 컨텍스트로 전달한다.

문제:
- 절차와 무관한 cover/scope 청크가 상당 부분을 차지
- 모델이 절차 청크에 집중하기 어렵게 만드는 noise 역할
- 1,200자 truncation으로 인해 긴 절차가 잘릴 수 있음
- 쿼리당 unique doc 평균 3.81개 — 같은 문서의 비절차 청크가 반복 노출

top_k를 줄이면 noise 감소 → 절차 집중도 향상이 기대됨.

### 6. temperature=0.5로 답변 변동성이 높다

동일 질문에 대해 답변 포맷이 매번 달라지는 원인 중 하나.
절차검색처럼 정형화된 출력이 필요한 route에서는 temperature를 낮추는 것이 유리하다.

---

## 목표 상태

절차검색 질문에 대해서는 답변이 아래처럼 보여야 한다.

1. 작업 목적: 1~2줄
2. 사전 준비: 짧게
3. 작업 절차: 가장 길고 상세하게
4. 작업 후 확인: 있으면 추가
5. 주의사항: 문서에 있을 때만 추가
6. 참고문헌: `[1]`, `[2]` 형식만

핵심 원칙:
- 답변의 중심은 항상 `작업 절차`
- `Work Procedure` / `Workflow` / `Procedure` / `Replacement` / `Adjustment` / `Setting` section을 우선 사용
- safety/pre-check는 짧고 앞쪽에
- background/scope 설명은 최소화

---

## 개선 방안

### P1. 한국어 `setup_ans_v2`를 실제 v2로 교체

대상 파일:
- [setup_ans_v2.yaml](/home/hskim/work/llm-agent-v2/backend/llm_infrastructure/llm/prompts/setup_ans_v2.yaml)

반영할 핵심 규칙:
- REFS의 `Work Procedure`, `Workflow`, `작업 절차`, `절차`, `교체`, `설치`, `조정`, `Setting` section을 최우선 사용
- 절차는 반드시 `1. 2. 3.` 형식
- 이모지 금지
- 인용은 반드시 `[숫자]` 형식만 허용
- 서로 다른 장비 문서는 절대 섞지 않음

### P2. 한국어 답변 포맷 고정

권장 구조:

```md
### 작업 목적
- 1~2줄

### 사전 준비
- 필요 공구/부품/안전 확인

### 작업 절차
1. ...
2. ...
3. ...

### 작업 후 확인
- ...

### 주의사항
- ...

### 참고문헌
[1] ...
[2] ...
```

규칙:
- `작업 절차` 섹션은 항상 포함
- 가장 긴 섹션이어야 함
- 절차가 없으면 "문서에 상세 절차가 명시되지 않았다"를 명확히 답변

### P3. 절차검색 질문에 대한 추가 규칙

다음 질문은 procedure-first 규칙을 강하게 적용한다.
- 교체
- 설치
- 조정
- 셋업
- setting
- replacement
- install
- adjust
- work procedure
- workflow

절차검색 모드에서는:
- 목적/배경 설명은 2~3줄 이하
- 실제 단계 수를 충분히 상세하게 기술
- 문서에 있는 순서를 최대한 유지

### P4. 절차 관련 REFS를 앞쪽으로 재정렬

답변 전 REFS 정리 단계에서 아래를 우선순위로 재배치하는 방안 검토:

우선순위 예시:
1. `work procedure`
2. `workflow`
3. `replacement`
4. `adjustment`
5. `setting`
6. `caution` / `warning`
7. `scope` / `overview`

효과:
- 모델이 먼저 읽는 REFS가 절차 중심이 됨
- prompt만으로 부족한 부분을 일부 보완 가능

### P5. answer 컨텍스트 축소 (top_k 20→8)

20개 청크를 모두 전달하면 noise가 많다.
- 89.2%가 rank 1에서 hit — 상위 8개면 충분
- cover/scope 등 비절차 청크가 줄어 절차 집중도 향상
- 1,200자 truncation 문제도 청크 수가 줄면 완화 가능

### P6. setup route temperature 하향 (0.5→0.1~0.2)

절차검색은 정형화된 출력이 필요.
- temperature=0.5는 매번 다른 포맷을 유발하는 원인 중 하나
- 0.1~0.2로 낮추면 포맷 일관성 향상 + 평가 재현성 확보
- general route는 현행 0.5 유지 가능

### P7. 답변 후 경량 검증 추가

후처리 또는 validator에서 아래를 감지:
- `0️⃣` 시작
- `EFIM` 같은 오타
- `[…]` placeholder
- `【】` 인용 형식
- markdown bold 깨짐

검출 시:
- 경량 rewrite
- 또는 재생성

---

## 우선순위

### 1차 (프롬프트 구조 개선)
- P1. 한국어 `setup_ans_v2` 교체
- P2. 답변 포맷 고정

### 2차 (답변 품질 심화)
- P3. 절차검색 질문에 대한 procedure-first 규칙 강화
- P4. REFS 절차 우선 재정렬
- P5. top_k 축소 (20→8) — noise 감소로 절차 집중도 향상
- P6. temperature 하향 (setup route) — 포맷 일관성

### 3차 (안정화)
- P7. 출력 validator 추가

---

## 수용 기준

아래가 충족되면 개선 성공으로 본다.

### 답변 품질
1. 절차검색 질문에서 답변 본문 중심이 `작업 절차`여야 한다.
2. 답변 시작 200자 안에 절차 섹션 또는 첫 단계가 등장해야 한다.
3. 한국어 답변은 `1. 2. 3.` 형식만 사용한다.
4. `[1]` 외 인용 형식(`【】`)은 나오지 않아야 한다.
5. `0️⃣`, `[…]`, `EFIM` 같은 명백한 출력 오류가 없어야 한다.
6. 같은 유형 질문에 대해 답변 구조가 거의 동일해야 한다.

---

## 재평가 계획

개선 후 아래 질문군으로 비교한다.

우선 샘플:
- `EFEM PIO SENSOR BOARD`
- `EFEM Controller`
- `EDA PC`
- `Manometer Adjust`
- `PM Pirani Gauge`
- `SW Install / Device Net Setting`

확인 항목:
- 절차 섹션이 답변 중심인가
- safety/pre-check가 과도하게 길지 않은가
- 인용 형식이 `[N]`로 고정됐는가
- 이모지/테이블 혼용이 줄었는가

---

## 평가 스크립트 수정 사항

재평가 전에 아래 eval 스크립트 버그를 먼저 수정해야 정확한 비교가 가능하다.

### 1. `_normalize()` — `&` 문자 미처리

현재 4건의 false negative 발생. gold_doc에 `&`가 포함되면 ES doc_id와 불일치.

```python
# 수정안
import re
def _normalize(name):
    n = name.lower().strip()
    n = re.sub(r'\s*&\s*', '_', n)
    n = n.replace(' ', '_').replace('-', '_').replace('.pdf', '')
    n = re.sub(r'_+', '_', n).strip('_')
    return n
```

적용: `run_sop_filter_eval.py`, `run_sop_retrieval_eval.py`

### 2. `_check_hit()` — first-match-wins 로직

현재 첫 번째 매칭 청크의 페이지만 확인하고 종료. 커버 페이지가 top-1이면 정답 페이지가 top-20 내에 있어도 miss 판정.

수정 방향: 매칭 문서의 **모든** 청크 중 정답 페이지가 있으면 hit (best-page-for-doc)

### 3. `answer_preview` 200자 제한

현재 답변을 200자로 잘라서 저장. 전체 답변 품질 분석이 불가능.

수정: 전체 답변 저장 또는 최소 2,000자로 확대.

---

## 결론

현재 SOP 답변 문제의 핵심은 retrieval보다 answer prompt와 출력 구조다.

즉,
- 지금은 `SOP 요약형 답변`
- 사용자가 원하는 것은 `SOP 절차 추출형 답변`

따라서 다음 개선의 핵심은 한국어 setup 프롬프트를 실제 v2 수준으로 올리고,
절차검색 질문에 대해 `Work Procedure` / `Workflow` 중심으로 답변하도록 강하게 통제하는 것이다.

---

## 부록: 응답 지연 분석

참고용으로 기록. 품질 개선과 별개로 진행 가능.

| 노드 | 중앙값 | 비율 |
|------|--------|------|
| answer (LLM) | 35.0s | 66.5% |
| judge (LLM) | 16.5s | 31.4% |
| route (LLM) | 1.4s | 2.7% |
| retrieve + expand (ES) | 162ms | 0.3% |

- ES 검색은 전체의 0.3% — 검색 최적화로는 지연 개선 불가
- judge_node는 `mode="base"`에서 결과에 영향 없이 실행 (스킵 시 -31%)
- answer_node TTFT가 높은 건 24,000자(20 docs × 1,200자) 입력 때문
- judge 스킵 + top_k 축소 병행 시 57s → 31s 가능
