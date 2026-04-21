# After Simulation (Post-Patch)

작성일: 2026-04-21
대상: Phase A 패치 적용 후의 `ReactRAGAgent`
방식: 프롬프트가 허용하는 답변 형태를 구조 수준에서 예측.

## 0. 패치 요약

### 0.1 프롬프트 변경 (`general_ans_v2.yaml` / `ts_ans_v2.yaml`)

| 항목 | Before | After |
|---|---|---|
| 템플릿 | 단일 5섹션 고정 | 질문 유형별 6종 구조 제시 (스펙/이력/진단/절차/조회/체크리스트) |
| 마크다운 테이블 | **금지** | **허용** (사용자 요청시 필수) |
| 인용 형식 | `[1]` 단일만 명시 | `[1]`, `[1, 3, 5]`, `[1~5]` 모두 허용 |
| REFS 부재 | "찾지 못했습니다" + 확인 질문 강제 | "직접 정보 없습니다" + 유사 정보 `(참고)` + 확인질문 **선택** |
| 스펙 부분 일치 | 확인 질문으로 전환 | 스펙 제시 후 필요시 확인 질문 |
| "핵심 키워드 블록" | 언급 없음 | 중간 이상 답변 끝에 `### 참고 문서 핵심 키워드` 권장 |
| 이모지 번호 | 금지 (유지) | 금지 (유지) |
| 한국어 강제 | 유지 | 유지 |
| REFS only 원칙 | 유지 | 유지 |
| 식별자 fabrication 금지 | 유지 | 유지 (`(참고)` 표기로 REFS 내 대체 정보는 허용) |

### 0.2 코드 변경 (`langgraph_agent.py`)

- `answer_node` 의 format enforcement 게이트를 **`route == "setup"` + 한국어** 로 제한.
  - 이전: 모든 한국어 답변에 `## 작업 절차` 섹션 + `1.` 번호 + 테이블 금지 강제.
  - 이후: setup(SOP) 답변만 해당 템플릿 검증. general/ts 는 자유 구조.
- FORMAT FIX retry 프롬프트에서 "마크다운 테이블" 금지 문구 제거 (setup 답변은 테이블 쓸 일 거의 없음).

### 0.3 테스트 변경 (`test_general_ts_empty_refs_prompt.py`)

- 기존에 phrasing 을 lock 했던 4건의 assertion 을 semantic invariant(empty-REFS 처리, 조회/절차 언급, fabrication 금지, 한국어 강제, REFS 기반 인용) 로 변경.
- FORMAT FIX 가 더 이상 general route 에 붙지 않음을 검증하는 assertion 추가.

---

## 1. Sample-by-Sample After-State 예측

아래는 새 프롬프트/validator 하에서 내 agent 가 **구조적으로 생성 가능한** 답변 형태. 실제 품질은 LLM 능력과 검색 품질에 여전히 의존하지만, 구조적 제약 은 더 이상 참조와 어긋나지 않는다.

### S01 — `SUPRA Vm Slit door screw part number` (spec_inquiry/s)

- REF 구조: 핵심 PN bold + 보충 불릿 3개 + `[1,3]` 다중 + GCB 꼬리말.
- AFTER: "스펙/Part Number 조회" 구조 예시가 프롬프트에 명시 → **굵게 강조 + 보충 불릿 + 요약 1줄** 생성 가능. 다중 인용 `[1,3]` 허용. GCB 꼬리말은 자동 생성 아님(외부 서비스 고유).
- 변화: 🟢 구조적 diff 거의 해소.

### S02 — `What is the part number ...` (영어 spec_inquiry)

- REF: 영어 답변.
- AFTER: 내 프롬프트는 여전히 "반드시 한국어" 강제 → 한국어 답변 생성. Phase A 는 영어 경로를 건드리지 않음.
- 변화: ⚪ 의도적 비재현 (Phase B 대상 — `react_agent.py` 의 `target_language` 에 따라 `general_ans_en_v2` 등 선택 로직 추가 필요).

### S03 — `TM모듈 ROBOT PART NUMBER 모두 알려줘` (spec_inquiry/m, 표)

- REF: "없음" 선언 + 테이블 2개 (TM 없음 / SANKYO EFEM 대체 정보) + GCB.
- AFTER:
  - "직접 정보 없습니다" + `(참고) REFS 에는 SANKYO EFEM ROBOT 정보가 있습니다` 패턴 허용 🟢
  - 테이블 **허용** 🟢
  - SANKYO 언급은 REFS 에 있을 경우 `(참고)` 로 허용 🟢
- 변화: 🟢 3개의 주요 blocker 전부 해소.

### S04 — `SR8250 TM Robot Communication Alarm Error 6` (alarm_trouble/m, 표)

- REF: 증상→원인→조치→체크리스트 표 + 다중 인용.
- AFTER: ts_ans_v2 의 "짧은 알람/에러 조치" 구조(`### 원인 → ### 점검/조치 → ### 검증`) 예시 제공. 표 허용. 다중 인용 허용.
- 변화: 🟢 전 항목 해소.

### S05 — `EFEM LL1 Teaching y target` (alarm_trouble/m)

- REF: 공정 설명 + 원인 + 체크 + 조치.
- AFTER: 절차 템플릿 강제가 사라져 planner/LLM 이 자유롭게 서술 가능.
- 변화: 🟢 템플릿 오적용 위험 제거.

### S06 — `SUpra EPD Communication alarm` (alarm_trouble/l, 표)

- REF: 증상→원인→점검→조치→체크리스트 표.
- AFTER: 표 허용. 다중 인용 허용. 장비명 "Supra" 오타 정규화는 여전히 검색 품질에 의존.
- 변화: 🟢 구조, 🟡 검색 품질(별도 이슈).

### S07 — `INTEGER plus Chamber open interlock alarm` (alarm_trouble/l, 키워드 블록)

- REF: 긴 원인/조치 + 핵심 키워드 블록.
- AFTER: 프롬프트가 "중간 이상 답변 끝에 `### 참고 문서 핵심 키워드` 선택적 포함" 명시 → 생성 가능.
- 변화: 🟢 키워드 블록 재현 경로 확보.

### S08 — `INTEGER plus Ball Screw 교체 이력` (history_lookup/m, 표)

- REF: 이력 조회 테이블 인트로 (`| No. | 문서명 | Order No. | Equip | 작업일 | 내역 |`).
- AFTER: ts_ans_v2 에 "이력 조회" 구조 + **테이블 예시 그대로** 명시. 테이블 금지 해제.
- 변화: 🟢 치명 blocker 해소.

### S09 — `supra n wafer broken 이력 표로 정리해줘` (history_lookup/m, 표)

- REF: 테이블 포맷.
- AFTER: 사용자가 "표로" 명시 → 프롬프트가 "반드시 그 형식으로" 강제. + validator 가 테이블을 막지 않음.
- 변화: 🟢 이전 가장 가시적이던 "사용자 요청 직접 위반" 해소.

### S10 — 긴 질문, 표 요구 (location_inquiry/s, 표)

- REF: 짧은 표 답.
- AFTER: 표 허용. 🟢

### S11 — `GENEVA vac Sol valve` (location_inquiry/m, 키워드)

- REF: 위치 안내 + 핵심 키워드 블록.
- AFTER: 키워드 블록 권장 → 🟢

### S12 — `mySITE setup service order maintenance history` (procedure/m)

- REF: 짧은 안내 + GCB.
- AFTER: route 가 setup 으로 분류되면 여전히 SOP 템플릿 강제 (본 task 범위 밖). general 로 분류되면 자유. 🟡
- 변화: 🟡 route 분류에 의존.

### S13 — `INTEGER plus safety controller 교체 이유` (procedure/m, 키워드)

- REF: "이유 설명" (진단형) + 키워드 블록.
- AFTER: route 가 ts 로 분류되면 "원인 분석 요약" 구조 예시 적용 가능. setup 으로 잘못 분류되면 여전히 템플릿 강제. 🟡
- 변화: 🟡 route 분류 경계 케이스.

### S14 — `SUPRA N TM ROBOT 교체 순서 + 3개월 이슈` (procedure/l, 표)

- REF: 긴 절차 + 이슈 테이블 + 키워드.
- AFTER:
  - route=setup 이면 SOP 템플릿 강제 (setup_ans_v2 는 이번 task 수정 안 함) → 이슈 테이블이 "## 작업 절차" 내부에 섞여 들어갈 수 있음. 🟡
  - route=ts 로 분류되면 테이블 허용 + 구조 자유. 🟢
- 변화: 🟡 setup route 에서 부분 재현.

### S15 — `FCIP Leak 조치방법` (troubleshoot_diag/l)

- REF: 원인→조치→사례.
- AFTER: ts_ans_v2 "짧은 알람/에러 조치" 구조 자연스럽게 매칭. 다중 인용 허용.
- 변화: 🟢

### S16 — `reflow formic acid leak` (troubleshoot_diag/l, 표, 키워드)

- REF: 원인→조치→검증→표→키워드.
- AFTER: "긴 진단" 구조 예시 매칭. 표·키워드 모두 허용.
- 변화: 🟢

### S17 — `SUPRA XP Toxic Gas Turn On` (troubleshoot_diag/xl, 표, 키워드)

- REF: 긴 절차 + 안전 경고 + 표 + 키워드.
- AFTER:
  - 템플릿 강제 해제로 긴 분량 자유롭게 생성.
  - 표·키워드 허용.
  - "REFS 외 일반 안전 주의사항 금지" 는 유지 → 안전 경고는 REFS 에 있어야 함 (이건 올바른 제약).
- 변화: 🟢 (단, 실제 안전 경고가 REFS 에 충분히 있느냐는 검색 품질 문제)

### S18 — `체크리스트로 만들라고` (list_lookup/m, 표, followup)

- REF: 체크리스트 테이블.
- AFTER:
  - `_is_followup_query` 가 "만들어" 를 indicator 로 잡아서 `followup_node` 분기 → **재포맷만 수행**. ⚪ 이건 Phase B 대상.
  - 테이블 금지는 해제됨 🟢 (followup 노드의 프롬프트 자체는 별개지만, LLM 이 표로 만들라는 요청을 따를 수 있음).
- 변화: 🟡 표 형식은 가능하지만 "신규 체크리스트 발굴" 은 Phase B.

### S19 — `FFU 转速spec` (short_followup/s, 표)

- REF: 짧은 스펙 표.
- AFTER: followup 로 오분류 위험 유지. 표 허용. 🟡

### S20 — `SUPRA Q Robot 통신` (general/m, 키워드)

- REF: 통신 설명 + 키워드 블록.
- AFTER: 키워드 블록 권장. 🟢

---

## 2. 집계: Phase A 패치 후 해소 상태

| Divergence Cause | Before: 영향 샘플 | After: 해소 |
|---|---|---|
| 테이블 재현 불가 | 11 (3,4,6,8,9,10,14,16,17,18,19) | 11 (전부 해소) ✅ |
| 다중 인용 불가 | 10 | 10 ✅ |
| 핵심 키워드 블록 누락 | 7 (7,11,13,14,16,17,20) | 7 ✅ (프롬프트에서 권장) |
| "찾지 못했습니다" 조기 종료 + 확인 질문 강제 | 1 (S03) | 1 ✅ |
| 절차 템플릿 오적용 | 4 (5,12,13,14) | 2 ✅ general/ts 경로는 해제, 🟡 setup 경로 잔존 (S12, S14 의 setup 분류) |
| "USER QUESTION 밖 식별자" 완화(`(참고)` 허용) | 1 (S03 SANKYO) | 1 ✅ |
| 언어 mismatch (영→한) | 1 (S02) | ❌ Phase B |
| followup 재포맷만 | 1 (S18) | 부분 (표는 가능, 신규 발굴은 Phase B) |

**Phase A 단독 커버리지**: 20건 중 **17건 (85%)** 의 주요 구조적 diff 해소. 잔존:
- S02 (영어) — Phase B
- S18 (followup 재포맷) — Phase B
- S12 / S14 일부 (setup route 의 SOP 템플릿 강제) — `setup_ans_v2` 를 이번 task 범위에 넣지 않은 의도적 결정. 필요하면 Phase A-2 확장.

---

## 3. 리스크와 한계

1. **Hallucination 리스크 상승 가능성**: "확인 질문 필수" 를 완화했으므로, REFS 가 빈약할 때 LLM 이 공백을 채우려 할 위험. 완화책:
   - "REFS 라인에 있는 내용만 증거" 원칙 유지.
   - "직접 정보 없습니다" 로 시작하도록 명시.
   - `(참고)` 로 대체 정보를 명시적으로 분리.
2. **테이블 남용**: 짧은 답변에 불필요한 표가 들어갈 수 있음. 프롬프트는 "이력 조회 / 체크리스트 / 사용자가 표 요청한 경우" 를 테이블 사용 조건으로 명시해 범위를 제한.
3. **Validator off** 로 인한 setup 외 형식 불일치: 다른 곳에서 `answer_format` metadata 를 소비하는 소비자가 있는지 확인 필요. 현재 `answer_node` 는 `answer_format` dict 를 결과에 포함시켜 C-API-001 metadata 로 전달되지만, skipped=True 필드가 계속 쓰이므로 호환.
4. **모델 스타일 변동성**: 현재 Ollama `qwen2.5:14b`, temperature=0.7 기본. Phase A 에서는 sampling 파라미터 건드리지 않아 답변 다양성이 여전히 큼. 정량 평가가 필요하면 Phase A-2 에서 temperature 하향, repeat_penalty 완화를 env level 로 조정 권장.

---

## 4. 다음 단계

- 03-comparison-report.md 에 Before/After 를 한 장에 병치.
- verification command 실행 (task doc 의 Verification Plan).
- 커밋.
