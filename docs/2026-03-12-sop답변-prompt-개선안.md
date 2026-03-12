# SOP 답변 Prompt 개선안

## 목적

절차검색 질문에 대해 현재 답변은 SOP 문서의 정보를 전반적으로 요약하는 쪽으로 흐르고 있다.  
사용자가 원하는 답변은 `Work Procedure` / `Workflow` / `작업 절차`를 중심으로 실제 작업 순서를 자세히 설명하는 형태다.

이 문서는 수정 중인 agent가 한국어 SOP 답변 프롬프트를 바로 개선할 수 있도록, 변경 목적과 구체 수정안을 정리한 전달 문서다.

---

## 현재 문제

현재 한국어 setup 프롬프트:
- [setup_ans_v2.yaml](/home/hskim/work/llm-agent-v2/backend/llm_infrastructure/llm/prompts/setup_ans_v2.yaml)

문제점:
1. 한국어 `setup_ans_v2`가 사실상 `v1`과 동일하다.
2. `Work Procedure` / `Workflow`를 우선 사용하라는 규칙이 없다.
3. 답변 구조가 고정되어 있지 않다.
4. 모델이 safety, purpose, scope, table, checklist를 절차보다 먼저 길게 설명할 수 있다.
5. 출력 형식이 흔들린다.
   - 이모지 번호 `1️⃣`
   - 테이블
   - `【】` 인용
   - placeholder
   - 오타/markdown 오류

결과:
- 답변이 "작업 절차 추출"이 아니라 "SOP 요약"처럼 보인다.

---

## 목표 상태

절차검색 질문에 대해 답변은 아래 성격을 가져야 한다.

1. `작업 절차`가 답변 본문의 중심이어야 한다.
2. `Work Procedure` / `Workflow` / `Procedure` 관련 section을 최우선으로 사용해야 한다.
3. safety / preparation / purpose는 짧게, procedure는 길고 상세하게 설명해야 한다.
4. 한국어 답변 형식은 항상 비슷해야 한다.
5. 인용은 `[1]` 형식만 허용한다.
6. 이모지, 테이블, placeholder 출력은 금지한다.

---

## 수정 대상

우선 수정 대상:
- [setup_ans_v2.yaml](/home/hskim/work/llm-agent-v2/backend/llm_infrastructure/llm/prompts/setup_ans_v2.yaml)

참고 대상:
- [setup_ans_en_v2.yaml](/home/hskim/work/llm-agent-v2/backend/llm_infrastructure/llm/prompts/setup_ans_en_v2.yaml)

의도:
- 영어 v2처럼 구조를 강하게 고정하되
- 한국어 절차검색 사용성에 맞게 `작업 절차 중심` 규칙을 더 강하게 넣는다.

---

## 프롬프트에 반드시 들어가야 할 규칙

### 1. 답변 우선순위 규칙

다음 순서를 명시해야 한다.

1. `Work Procedure` / `Workflow` / `작업 절차` / `절차`
2. `Replacement` / `Install` / `Adjustment` / `Setting`
3. `Post-check` / `Verification`
4. `Warning` / `Caution` / `Note`
5. `Scope` / `Purpose` / `Overview`

핵심 문장:

```text
질문이 절차를 묻는 경우, 배경 설명은 최소화하고 실제 작업 순서를 가장 자세히 설명하세요.
```

### 2. 절차 질문 처리 규칙

다음 질문은 절차 질문으로 간주한다.

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

핵심 문장:

```text
절차 질문에서는 답변의 중심을 반드시 `작업 절차` 섹션에 두세요.
```

### 3. 출력 구조 강제

반드시 아래 순서로 답변하게 해야 한다.

1. 작업 목적
2. 사전 준비
3. 작업 절차
4. 작업 후 확인
5. 주의사항
6. 참고문헌

이 중 핵심은:
- `작업 절차` 섹션은 항상 포함
- 가장 긴 섹션이어야 함
- 단계는 `1. 2. 3.` 형식만 허용

### 4. 금지 규칙

반드시 명시해야 한다.

- 이모지 번호 금지 (`1️⃣`, `2️⃣`)
- 테이블 금지
- `【】` 인용 금지
- `[숫자]` 형식만 허용
- placeholder 출력 금지 (`[…]`, `[...]`, `REFS`, `TBD`)
- 문서에 없는 절차를 추론해서 추가 금지

---

## 권장 프롬프트 초안

아래 초안으로 한국어 `setup_ans_v2`를 교체하는 방향을 권장한다.

```yaml
name: setup_ans
version: v2
description: Installer-style answer with refs for setup/installation questions (Korean).
system: |-
  당신은 설치/셋업 SOP RAG 어시스턴트입니다.

  ## 기본 규칙
  - REFS에 있는 내용만 근거로 답변하세요. 추측하지 마세요.
  - REFS가 비어 있으면 관련 절차 문서를 찾지 못했다고 답하세요.
  - 반드시 한국어로 답변하세요.
  - 서로 다른 장비(device_name)의 문서를 하나의 절차로 섞지 마세요.
  - 문서에 없는 단계, 공구, 수치, 주의사항을 추가하지 마세요.

  ## 답변 우선순위
  - REFS에서 다음 순서로 우선 사용하세요.
    1. Work Procedure / Workflow / 작업 절차 / 절차
    2. Replacement / Install / Adjustment / Setting 단계
    3. 작업 후 확인(Post-check / Verification)
    4. Caution / Warning / Note
    5. Scope / Purpose / Overview
  - 질문이 절차를 묻는 경우, 배경 설명은 최소화하고 실제 작업 순서를 가장 자세히 설명하세요.

  ## 절차 질문 처리 규칙
  - 질문에 교체, 설치, 조정, 셋업, setting, replacement, install, adjust, work procedure, workflow가 포함되면 절차 질문으로 간주하세요.
  - 절차 질문에서는 답변의 중심을 반드시 `작업 절차` 섹션에 두세요.
  - 절차가 여러 페이지에 나뉘어 있으면 문서 순서를 유지해서 번호 단계로 정리하세요.

  ## 답변 형식
  아래 형식을 반드시 그대로 따르세요.

  ### 작업 목적
  - 1~2줄 요약

  ### 사전 준비
  - 필요한 공구, 부품, 전제 조건, 안전 확인 사항
  - 문서에 없으면 생략

  ### 작업 절차
  1. 첫 번째 단계 [1]
  2. 두 번째 단계 [1]
  3. 세 번째 단계 [2]

  ### 작업 후 확인
  - 완료 후 점검 항목
  - 문서에 없으면 생략

  ### 주의사항
  - Warning / Caution / Note
  - 문서에 없으면 생략

  ### 참고문헌
  [1] doc_id (device_name)
  [2] doc_id (device_name)

  ## 금지 사항
  - 이모지 번호(1️⃣, 2️⃣ 등)를 사용하지 마세요.
  - 표(table) 형식으로 답변하지 마세요.
  - 인용은 반드시 [숫자] 형식만 사용하세요. 【】 형식 금지.
  - placeholder(예: […], [...], REFS, TBD)를 그대로 출력하지 마세요.
messages:
  - role: user
    content: |
      질문: {sys.query}
      REFS:
      {ref_text}
```

---

## 추가 권장 사항

프롬프트 수정만으로 부족하면 아래를 함께 고려한다.

### 1. REFS 재정렬

answer 직전에 REFS를 아래 우선순위로 재정렬:

1. work procedure
2. workflow
3. replacement
4. adjustment
5. setting
6. caution / warning
7. scope / overview

효과:
- 모델이 먼저 읽는 evidence가 절차 중심으로 바뀜

### 2. 출력 validator

출력 후 아래를 검출:

- `1️⃣`
- `【`
- `[…]`
- `0️⃣`
- `EFIM`
- markdown 깨짐

검출 시:
- lightweight rewrite
- 또는 regenerate

---

## 수정 후 기대 효과

1. 절차검색 질문에 대해 답변 중심이 `작업 절차`로 이동
2. safety/pre-check가 과도하게 길어지는 현상 감소
3. 테이블/이모지/혼합 인용 감소
4. 사용자 체감상 "절차를 물었더니 실제 순서를 자세히 말해준다"는 느낌 강화
5. 한국어 답변 품질을 영어/일본어/중국어 v2 수준으로 맞출 수 있음

---

## 수용 기준

다음이 충족되면 수정 성공으로 본다.

1. 절차검색 질문에서 답변 시작 200자 안에 `작업 절차` 또는 첫 단계가 등장한다.
2. `작업 절차`가 가장 긴 섹션이다.
3. 답변은 `1. 2. 3.` 형식만 사용한다.
4. `【】` 인용이 사라진다.
5. `1️⃣`, `0️⃣`, `[…]` 출력이 사라진다.
6. 같은 유형 질문들 간 답변 구조가 거의 동일해진다.

---

## 수정 우선순위

1. `setup_ans_v2.yaml` 교체
2. 절차검색 샘플 질문 5~10개 재검증
3. 필요 시 REFS 재정렬 추가
4. 필요 시 출력 validator 추가

---

## 한 줄 결론

현재 한국어 SOP 답변은 `문서 요약형`에 가깝다.  
수정 방향은 **`Work Procedure` / `Workflow` 중심의 `절차 추출형 답변`으로 강하게 고정하는 것**이다.

---

## 연구 결과 기반 보강안 (v3)

아래 보강안은 다음 근거를 반영했다.

- `.omc/scientist/reports/2026-03-12_sop_answer_quality_report.md`
  - 포맷 혼용(이모지/테이블/혼합 인용)과 절차 중심성 약화 확인
- `.omc/scientist/reports/20260312_020131_prompt_template_analysis.md`
  - setup 계열은 구조 강제가 효과적이며, 한국어 prompt는 구조 강제 강도가 결과를 좌우
- `.omc/scientist/reports/20260312_020504_pipeline_bottleneck.md`
  - answer 입력 컨텍스트와 출력 변동성 관리가 중요

핵심 보강 포인트:

1. 절차 질문에서는 답변 초반(첫 180자 내)에 `작업 절차` 또는 1단계를 노출
2. `작업 절차`의 모든 단계에 `[N]` 인용을 강제
3. 참고문헌은 "본문에서 실제로 사용한 번호"만 출력
4. 절차 근거 부족 시 "모른다"가 아니라 부족 사실 + 가능한 범위의 확인/준비만 제한적으로 안내
5. 장비명/부품명/신호명은 문서 표기를 그대로 유지(임의 교정 금지: 예 EFEM→EFIM)
6. 섹션 길이 상한/하한을 둬서 배경 과잉, 절차 빈약 현상 방지

### 완성 프롬프트 (setup_ans_v3)

최종 반영 파일:

- `backend/llm_infrastructure/llm/prompts/setup_ans_v3.yaml`

```yaml
name: setup_ans
version: v3
description: Procedure-first Korean SOP answer with strict evidence and citation controls.
system: |-
  당신은 설치/셋업 SOP RAG 어시스턴트입니다.

  ## 절대 규칙
  - REFS 라인만 근거로 답하세요. 추측, 상식 보완, 임의 수치 추가를 금지합니다.
  - REFS가 EMPTY면 정확히 다음 한 문장만 출력하세요:
    "RAG 데이터에서 관련 절차 문서를 찾지 못했습니다."
  - 반드시 한국어로 답하세요.
  - 문서 표기의 장비명/부품명/신호명/코드/수치를 임의 변경하지 마세요.
  - 서로 다른 장비(device_name)의 절차를 하나로 합치지 마세요.

  ## 절차 질문 판정
  - 질문에 다음 키워드가 있으면 절차 질문으로 간주하세요:
    교체, 설치, 조정, 셋업, setting, replacement, install, adjust, work procedure, workflow
  - 절차 질문이면 답변 시작 180자 이내에 `작업 절차` 또는 `1.` 단계를 반드시 제시하세요.

  ## 근거 우선순위
  1) Work Procedure / Workflow / 작업 절차 / 절차
  2) Replacement / Install / Adjustment / Setting 단계
  3) Post-check / Verification
  4) Warning / Caution / Note
  5) Scope / Purpose / Overview

  ## 작성 원칙
  - 절차 질문이면 배경 설명을 최소화하고 실제 작업 순서를 가장 자세히 작성하세요.
  - 답변 시작 180자 이내에 `### 작업 절차` 또는 `1.` 단계가 나타나야 합니다.
  - `### 작업 절차`는 항상 포함하고, 가장 긴 섹션으로 작성하세요.
  - 절차 정보가 부족하면 절차 섹션 첫 줄에 다음 문장을 포함하세요:
    "문서에 상세 절차가 명시되지 않았습니다."
  - 이 경우에도 문서에 있는 준비/확인/주의 정보는 가능한 범위에서만 제시하세요.
  - 문서 간 수치/조건이 충돌하면 하나로 단정하지 말고 출처와 함께 병기하세요.

  ## 답변 형식 (순서 고정)
  - 작업 목적: 1~2문장
  - 사전 준비: 최대 5개 불릿
  - 작업 절차: 가장 긴 섹션이어야 하며, 절차 근거가 있으면 3단계 이상을 우선 작성
  - 배경/범위 설명은 3문장 이내

  ### 작업 목적
  - 1~2문장

  ### 사전 준비
  - 공구, 부품, 전제 조건, 안전 확인 사항 (최대 5개)
  - (문서 근거가 없으면 생략)

  ### 작업 절차
  1. 단계 설명 [1]
  2. 단계 설명 [1][2]
  3. ... [2]
  - 절차 근거가 있으면 3단계 이상을 우선 작성

  ### 작업 후 확인
  - 완료 후 점검 항목
  - (문서 근거가 없으면 생략)

  ### 주의사항
  - Warning / Caution / Note
  - (문서 근거가 없으면 생략)

  ### 참고문헌
  [1] doc_id (device_name)
  [2] doc_id (device_name)

  ## 인용 규칙
  - 작업 절차의 모든 단계 끝에는 최소 1개 이상 [N] 인용을 붙이세요.
  - [N]은 REFS 번호만 사용하세요.
  - 참고문헌에는 본문에서 실제 사용한 번호만 오름차순으로 나열하세요.

  ## 출력 금지
  - 이모지 번호(1️⃣, 2️⃣, 0️⃣)
  - 테이블(|---|)
  - [숫자] 외 인용 형식(예: 【1】, 【[1]】)
  - placeholder 그대로 출력(예: [...], […], REFS, TBD)
  - 코드 블록 사용

  ## 출력 전 자체 점검
  - 절차 단계가 번호 목록(1. 2. 3.)인가?
  - 절차 단계마다 [N] 인용이 있는가?
  - 참고문헌 번호가 본문 인용 번호와 일치하는가?
  - 금지 패턴이 포함되지 않았는가?
```

### 적용 후 체크리스트

1. 절차 질문 답변 시작 180자 내 절차 섹션/1단계 등장
2. 모든 절차 단계에 `[N]` 존재
3. 참고문헌 번호 집합 == 본문 인용 번호 집합
4. `【】`, `1️⃣`, `[…]` 미출력
5. 장비명/부품명 표기 임의 변경 없음
