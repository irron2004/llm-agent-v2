# Baseline Simulation (Pre-Patch)

작성일: 2026-04-21
대상: `ReactRAGAgent` (현재 main 코드 + 현재 `general_ans_v2` / `ts_ans_v2` 프롬프트)
방식: Claude(claude-opus-4-7) 가 Ollama 역할을 **구조 수준**에서 시뮬레이션.
  - 실제 generation 대신, 현재 프롬프트의 제약에서 **답변이 가질 수밖에 없는 구조적 속성** 을 예측.
  - 예측 근거: `backend/llm_infrastructure/llm/prompts/*.yaml` + `react_agent.py` 의 route/doc_type 분기 로직.

---

## 0. Simulation 공통 모델

각 샘플에 대해 내 agent가 거칠 경로:

```
preprocess (auto_parse + translate + route 분류)
  → plan  (LLM, NextAction: search/search_solution/answer/followup)
  → search (retrieve_node, rerank OFF 기본값)
  → [plan loop까지 최대 6회]
  → answer (prompt: route 기반 setup_ans_v2 / ts_ans_v2 / general_ans_v2)
  → judge
```

### 답변 프롬프트 선택 규칙 (react_agent.py `_infer_route` + `preprocess_node`)
- `route == "setup"` → `setup_ans_v2` (SOP 절차)
- `route == "ts"` → `ts_ans_v2` (트러블슈팅)
- `route == "general"` → `general_ans_v2` (기본)

### 현재 프롬프트의 결정적 제약 (`general_ans_v2` / `ts_ans_v2` 동일 구조)
- L1. `금지: 마크다운 테이블(|---|)`
- L2. `**절차 질문에만** 아래 Markdown 템플릿을 **반드시** 따르세요` → `# 제목 / ## 준비/안전 / ## 작업 절차 / ## 복구/확인 / ## 주의사항 / ## 참고문헌`
- L3. `REFS가 비어있거나 질문과 관련성이 낮으면 ... 1~3개의 확인 질문을 하세요`
- L4. `[1]` 인용 형식만 명시
- L5. `금지: 이모지 번호(1️⃣ 등)` — 이모지 헤더(`## 📊`)와 구분 필요

### 현재 Ollama 샘플링 기본값
- `temperature = 0.7`
- `repeat_penalty = 1.3` (높음)
- `RAG_RERANK_ENABLED = False`

---

## 1. Sample-by-Sample Simulation

각 샘플에 대해:
- **REF**: 참조 서비스 답변의 핵심 속성 (줄 수, 테이블, 인용 패턴, 구조)
- **SIM**: 내 agent 가 현재 프롬프트 하에서 생성 가능한 답변의 예상 속성
- **Δ(divergence)**: 재현 불가 요소 + 이유

### S01 — `SUPRA Vm의 Slit door screw part number를알려줘` (spec_inquiry/s, 9줄)

- REF: 인트로 1문장으로 Part Number 제시(`**5900-1000227 (M6*50)**[3]`) + 불릿 3개(재질·연관 PN·토크) + `[1,3]` 다중 인용 + 요약 1줄 + GCB 꼬리말.
- SIM (route=`general`, `general_ans_v2`):
  - Part Number 가 REFS 에 있으면 제시 가능. ✓
  - 인용 형식: `[1]` 만 허용 → `[1,3]` 불허. ✗
  - 템플릿 강제 없음(`절차 질문`이 아니므로) → 자유 형식 가능. ✓
  - "SUPRA Vm" 은 질문 내 장비라 사용 가능. ✓ (`USER QUESTION에 없는 새 식별자 금지` 에 저촉 안 됨)
  - GCB 꼬리말은 프롬프트에 없어서 **자동 생성되지 않음**. ✗
- Δ: 다중 인용 불가 · GCB 꼬리말 누락 · 토크/재질 추가 권고(세 번째 불릿) 는 REFS에 있으면 가능하나 프롬프트가 "REFS 외 일반 주의사항 금지"여서 누락 위험.

### S02 — `What is the part number for the slit door screw on the SUPRA Vm?` (spec_inquiry/s, 11줄, **영문**)

- REF: 영어 답변. `The provided reference materials do not contain information...` 로 시작 + 유사 Part 2개 제시 + GCB 에스컬레이션 안내.
- SIM (route=`general`): 내 prompt 는 `반드시 한국어로 답변` 강제. 영문 질문에도 **한국어 답변 생성 → 언어 불일치**. ✗✗✗
  - (참고: `general_ans_en_v2.yaml` 존재하지만 `react_agent.py` 가 language detection 결과를 answer prompt 선택에 연결하는지 별도 확인 필요 → 현재 `load_prompt_spec` 기본은 한국어 YAML)
- Δ: 언어 mismatch 치명적. 영어 질문 처리 경로가 react_agent 에 결합되지 않음(C-API-001 범위 밖).

### S03 — `TM모듈에서 ROBOT PART NUMBER를 모두 알려줘` (spec_inquiry/m, 23줄, **테이블 포함**)

- REF: "없음" 선언 + `| 모듈명 | Part Number | Maker | Spec | 비고 |` **표 2개** (대체 정보: SANKYO EFEM ROBOT 4건) + GCB 꼬리말.
- SIM (route=`general`):
  - REFS 가 부족하다고 판단하면 `RAG 데이터에서 관련 정보를 찾지 못했습니다.` + `1~3 확인 질문` **강제** → 참조와 완전히 다른 동선. ✗✗
  - 테이블은 **금지**. ✗
  - "SANKYO, EFEM ROBOT 대체 정보" 는 질문에 없는 식별자(SANKYO) 도입 → 프롬프트가 **금지**. ✗
- Δ: 답변 형식 · 내용 · 인용 모두 재현 불가. 이 샘플은 Phase A 만으로 완전 재현 어려움.

### S04 — `SR8250 TM Robot Communication Alarm Error 6 ...` (alarm_trouble/m, 59줄, 표 O)

- REF: `### 1. 증상 정리 → ### 2. 주요 원인 → ### 3. 조치 → ### 4. 체크리스트 표 → ### 5. 참고` 유사 5섹션 + 표 + `[1,3,5,9]` 다중 인용.
- SIM (route=`ts`, `ts_ans_v2`):
  - "절차 질문" 이 아니므로 5섹션 템플릿 강제는 **안 됨** → planner 가 자유롭게 섹션 구성 가능. 🟡
  - 하지만 프롬프트는 `조회 vs 절차` 2분법만 제시 → 트러블슈팅 진단 구조(원인→조치→검증) 명시 가이드 없음. 🟡
  - 표 금지 → `### 4. 체크리스트 표` 재현 불가. ✗
  - 다중 인용 불가. ✗
- Δ: 구조는 일부 맞출 수 있지만 표/다중 인용 누락.

### S05 — `EFEM LL1 Teaching y target -240... Jaw align issue` (alarm_trouble/m, 46줄)

- REF: 본문이 공정 설명 + 원인 + 체크 + 조치 순서.
- SIM (route=`ts`):
  - 장비명(EFEM) 은 질문 내 있음 → 사용 가능. ✓
  - `절차 질문 templates` 강제 여부 모호 — "조치" 가 있으니 planner 가 절차로 분류할 수도 있음 → 5섹션 고정 템플릿에 **억지로 맞출 위험**. ✗
  - 다중 인용, 표 없음(REF 에도 표 없어 영향 작음).
- Δ: 절차 템플릿 오적용 위험.

### S06 — `SUpra설비 모든 채널 EPD Communication alarm` (alarm_trouble/l, 70줄, 표 O)

- REF: `### 1. 증상 → ### 2. 원인 → ### 3. 점검 → ### 4. 조치 → ### 5. 체크리스트(표)` + GCB 꼬리말.
- SIM (route=`ts`, `ts_ans_v2`):
  - 표 불가. ✗
  - 다중 인용 불가. ✗
  - "Supra" 오타 정규화 불확실 (device cache 매칭 실패 가능 → 검색 품질 저하 가능성). 🟡
- Δ: 표/인용/섹션 세분화 모두 제약.

### S07 — `INTEGER plus Chamber open interlock alarm` (alarm_trouble/l, 78줄, 키워드 블록 O)

- REF: 긴 원인/조치 구조 + **참고 문서 핵심 키워드** 블록 마지막 + GCB 꼬리말.
- SIM (route=`ts`):
  - 핵심 키워드 블록은 프롬프트에 **정의 없음** → 자동 생성 안 됨. ✗
  - GCB 꼬리말 없음. ✗
  - 다중 인용 `[1,5,7,12]` 불가. ✗
- Δ: 꼬리 섹션 전부 누락.

### S08 — `INTEGER plus Ball Screw 교체 이력 모두 알려줘` (history_lookup/m, 48줄, 표 O)

- REF: `| No. | 문서명 | Order No. | Equip | 작업일 | 내역 |` **이력 조회 테이블 인트로** 포맷.
- SIM (route=`ts`, keywords "교체"→TS expansion) 또는 `general`:
  - 표 금지 → 이력 테이블 포맷 **원천 불가**. ✗✗
  - planner 가 어떤 action 을 택하든 최종 answer 가 서술형으로 떨어짐.
- Δ: 이 카테고리는 Phase A 로도 치명적 miss.

### S09 — `supra n wafer broken 이력 표로 정리해줘` (history_lookup/m, 58줄, 표 O)

- REF: 질문에 "표로" 명시 + 테이블 포맷 답변.
- SIM: 프롬프트가 테이블 **전면 금지**. 질문이 명시적으로 "표" 를 요구해도 LLM 이 프롬프트에 묶여 서술형 출력. ✗✗✗
- Δ: 사용자 요청 직접 위반. 현재 프롬프트의 가장 가시적인 문제.

### S10 — `질문별 문서 위치 및 목적 안내 (프롬프트 삽입)` (location_inquiry/s, q_len=1611, 18줄)

- REF: 사용자가 표 형식의 "안내 요청" 을 주자 18줄 짧게 요약.
- SIM: 긴 질문을 planner 가 followup 으로 오탐할 위험 낮음(indicator 없음). 정상 search 루프 돌 것. 표 요구 → 위반. ✗
- Δ: 표 금지로 형식 mismatch.

### S11 — `GENEVA설비 vac Sol valve exhaust port 연결` (location_inquiry/m, 45줄, 키워드 블록 O)

- REF: 위치/연결 안내 + 핵심 키워드 블록.
- SIM (route=`general`): 핵심 키워드 블록 미생성. ✗
- Δ: 꼬리 섹션 누락.

### S12 — `mySITE setup service order 내 maintenance history` (procedure/m, 22줄)

- REF: 짧은 안내 + GCB 꼬리말.
- SIM (route=`setup`, `setup_ans_v2` — 본 task 수정 범위 밖이지만 동일 구조일 것): 템플릿 강제 여부가 관건. "절차" 키워드 매칭으로 template 적용 → 오히려 22줄보다 길게 나올 가능성. 🟡
- Δ: 과다 답변 위험.

### S13 — `INTEGER plus safety controller/module 교체 이유` (procedure/m, 32줄, 키워드 블록 O)

- REF: 이유 설명 + 원인 분석 + 핵심 키워드 블록.
- SIM (route=`ts` (교체→TS expansion) 또는 `setup`): 
  - 절차 템플릿 강제 적용 시 "준비/안전 → 작업절차 → 복구/확인" 으로 오인 가능. ✗
  - 핵심 키워드 블록 미생성. ✗
- Δ: 답변 의도 mismatch (이유 설명 질문인데 절차 답변).

### S14 — `SUPRA N TM ROBOT 교체 순서 + 3개월 이슈` (procedure/l, 104줄, 표 O, 키워드 O)

- REF: 긴 절차 안내 + 이슈 이력 테이블 + 핵심 키워드.
- SIM (route=`setup` or `ts`):
  - 긴 답변은 가능 (템플릿 강제가 오히려 분량 채움). 🟡
  - 이슈 이력 테이블 불가. ✗
  - 핵심 키워드 블록 미생성. ✗
- Δ: 절차 부분은 부분 재현, 이력/키워드 불가.

### S15 — `FCIP Leak 발생시 조치방법` (troubleshoot_diag/l, 63줄)

- REF: `### 1. 원인 → ### 2. 조치 → ### 3. 사례` + GCB 꼬리말.
- SIM (route=`ts`, `ts_ans_v2`):
  - 3섹션 구조 재현 가능성 있음 (템플릿 강제 안 됨 if planner 가 절차로 분류 안 함).
  - 다중 인용 불가. ✗
  - GCB 꼬리말 없음. ✗
- Δ: 구조는 맞출 수 있으나 꼬리말과 인용 diff.

### S16 — `reflow formic acid leak 조치방법` (troubleshoot_diag/l, 79줄, 표 O, 키워드 O)

- REF: 원인→조치→검증→표→키워드 + GCB 꼬리말.
- SIM (route=`ts`): 표/키워드 모두 불가. ✗✗
- Δ: 꼬리 섹션 전체 miss.

### S17 — `SUPRA XP Toxic Gas Turn On 방법` (troubleshoot_diag/xl, 142줄, 표 O, 키워드 O)

- REF: 긴 절차 + 안전 경고 + 표 + 키워드 블록.
- SIM (route=`setup` or `ts`):
  - 절차 템플릿 적용 시 분량은 가능하나 "안전" 섹션에 REFS 없는 일반 주의사항(정전기, 전원차단 등) 추가 금지 **프롬프트로 억제**되어 **오히려 참조보다 빈약**해질 수 있음. ✗
  - 표/키워드 불가. ✗
- Δ: 치명적인 분량·구조 diff.

### S18 — `체크리스트로 만들라고` (list_lookup/m, 34자, 46줄, 표 O)

- REF: 체크리스트 표 형식 답변.
- SIM: followup indicator `만들어줘` 감지 → `followup_node` 로 분기 → 이전 답변 재포맷만 시도. 표 금지로 체크리스트 표 불가. ✗✗
- Δ: followup 재포맷 + 표 금지 이중 miss.

### S19 — `FFU 转速spec` (short_followup/s, 10자, 18줄, 표 O)

- REF: 짧은 스펙 표.
- SIM: 질문 길이 ≤ 10자 → `_is_followup_query` 가 followup 후보로 보지만 indicator 없음 → 정상 search. `转速`(중국어) 가 preprocess 에서 tokenize 됨. 표 금지. ✗
- Δ: 중국어 혼용 → language detection 분기 불확실 + 표 금지.

### S20 — `SUPRA Q Robot 통신은 어떤 통신을 해?` (general/m, 47줄, 키워드 O)

- REF: 통신 프로토콜 설명 + 핵심 키워드 블록.
- SIM (route=`general`): 키워드 블록 미생성. ✗ 나머지는 재현 가능.
- Δ: 꼬리 섹션 diff.

---

## 2. 집계: Divergence 카테고리별 miss count (n=20)

| Divergence Cause | 영향 샘플 | 비율 |
|---|---|---:|
| 테이블 재현 불가 (프롬프트 금지) | 3, 4, 6, 8, 9, 10, 14, 16, 17, 18, 19 | **11 / 20 (55%)** |
| 다중 인용 `[1,3,5]` 불가 | 1, 3, 4, 5, 6, 7, 8, 15, 16, 17 | 10 / 20 (50%) |
| GCB 꼬리말 누락 | 1, 3, 6, 7, 15, 16 | 6 / 20 (30%) (REF 에서 꼬리말 있는 샘플 중) |
| 핵심 키워드 블록 누락 | 7, 11, 13, 14, 16, 17, 20 | 7 / 20 (35%) |
| "찾지 못했습니다" 조기 종료 + 확인 질문 강제 | 3 | 1 / 20 |
| 절차 템플릿 오적용 위험 | 5, 12, 13, 14 | 4 / 20 |
| "USER QUESTION 밖 식별자 금지" 로 대체 정보 제시 불가 | 3 (SANKYO 언급) | 1 / 20 |
| 언어 mismatch (영어→한국어) | 2 | 1 / 20 (english samples 는 전체의 약 5%) |
| followup 재포맷만 (신규 탐색 X) | 18 | 1 / 20 |

총 **20건 중 19건** 이 최소 1개의 구조적 재현 불가 항목을 포함. 치명 급(표+사용자 직접 요청 위반): **S09, S18** — 2건은 프롬프트 수정만으로 즉시 해결 가능.

---

## 3. Phase A 패치로 해결되는 Divergence (예상)

| Divergence | Phase A 패치 | 해결 예상 |
|---|---|---|
| 테이블 재현 불가 | `금지: 마크다운 테이블` 제거 + "질문 유형 맞는 표 사용 권장" | ✅ 11 / 20 |
| 다중 인용 불가 | `[1], [1,3,5], [1~19]` 모두 허용 명시 | ✅ 10 / 20 |
| 핵심 키워드 블록 누락 | "답변 마지막에 `### 참고 문서 핵심 키워드` 블록을 REFS 기반으로 작성" 권장 추가 | ✅ 7 / 20 |
| 절차 템플릿 오적용 | "5섹션 템플릿은 **예시**이며 강제 아님, 질문 유형에 맞게 적응" | ✅ 4 / 20 |
| "찾지 못했습니다" 조기 종료 | "REFS 가 질문과 직접 일치하지 않아도 유사 정보가 있으면 제시하고 부재를 명시" | ✅ 1 / 20 |
| GCB 꼬리말 | 외부 서비스 고유 꼬리말. 내 agent 는 의도적으로 **추가하지 않음** (PII 포함) | ❌ design decision |
| 언어 mismatch | `react_agent.py` 의 prompt selection by language 필요 | ❌ Phase B |
| followup 재포맷만 | `followup_node` 재설계 | ❌ Phase B |
| "USER QUESTION 밖 식별자 금지" | 완화하면 hallucination 위험 → 유지 | ❌ design decision |

**예상 커버리지**: Phase A 로 ~85% (17/20) 의 주요 구조 diff 해소. 3건(영어 질문, followup 신규 검색, GCB 꼬리말) 은 Phase B 이관 혹은 의도적 비재현.

---

## 4. 패치 타깃 (다음 단계)

1. `backend/llm_infrastructure/llm/prompts/general_ans_v2.yaml`
2. `backend/llm_infrastructure/llm/prompts/ts_ans_v2.yaml`

두 파일 모두 동일한 변경 철학을 적용. `setup_ans_v2` 는 본 task 범위 밖(절차 질문은 일부만 포함).
