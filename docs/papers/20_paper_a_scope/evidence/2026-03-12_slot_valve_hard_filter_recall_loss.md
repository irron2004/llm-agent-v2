# Evidence: Hard Device Filter Recall Loss — SLOT VALVE 교체 케이스

> Date: 2026-03-12
> Related: F6 (Recall-Contamination Trade-off), Section 1 Introduction
> Query: "ZEDIUS XP 설비의 SLOT VALVE 교체 작업"

---

## 1. 현상 요약

사용자가 ZEDIUS XP의 SLOT VALVE 교체 절차를 질의했으나, 시스템이 RF Generator 교체 SOP만 반환하여 답변이 15단계 중 3단계만 포함된 불완전한 결과를 생성했다.

- **쿼리**: "ZEDIUS XP 설비의 SLOT VALVE 교체 작업"
- **기대 문서**: SLOT VALVE 교체 SOP (15단계 절차)
- **실제 반환**: `global_sop_supra_xp_all_rack_rf_generator` (RF Generator 교체 SOP) — top-10 전부 동일 문서의 다른 청크
- **judge 결과**: `faithful: false`, `max_attempts_reached`

## 2. 원인 분석

### Hard device filter의 recall 희생 패턴

1. Auto-parse가 "ZEDIUS XP"를 정상 추출 → device_name 필터 적용
2. 필터가 정상 작동하여 ZEDIUS XP 문서만 후보군에 남음
3. **SLOT VALVE SOP가 인덱스에 없거나 device_name 매핑 불일치**로 필터에서 탈락
4. 같은 장비의 다른 SOP(RF Generator)만 살아남아 top-k 독점 → **과집중(over-concentration)**
5. LLM이 RF Generator SOP에서 SLOT VALVE 절차를 찾을 수 없어 3단계만 억지 생성

### Paper A Failure Taxonomy 매핑

| Code | 해당 여부 | 설명 |
|------|----------|------|
| F6 | **Primary** | device 필터가 contamination은 제거하지만 gold doc도 함께 배제 |
| F5 | Secondary | RF Generator SOP가 의미적 유사도로 상위 랭킹 독점 |

## 3. 검색 결과 상세

```
retrieved_docs (10건): 전부 global_sop_supra_xp_all_rack_rf_generator
- chunk #0014 (p15): RF Generator Removal — score 5.6%
- chunk #0015 (p16): RF Generator Removal — score 5.1%
- chunk #0010 (p11): Flow Chart 시작 — score 0% (chapter expansion)
- chunk #0011 (p12): Rack Door Open — score 0% (chapter expansion)
- ... (이하 동일 문서 반복)

expanded_pages: 11~21 (RF Generator 교체 15단계 절차)
search_queries_final:
  - "ZEDIUS XP 설비의 SLOT VALVE 교체 작업"
  - "ZEDIUS XP 설비 SLOT VALVE 교체 설치 절차"
  - "ZEDIUS XP 설비 SLOT VALVE 교체 체크리스트"
```

## 4. 생성 답변의 문제점

```
answer (요약):
1. 랙 도어 개방, 키를 Maint 모드로 전환 [1]
2. RF Generator 전원 차단 [1]
3. (SLOT VALVE 교체 작업은 별도 매뉴얼에 따름) [1]
→ "문서에 상세 절차가 명시되지 않았습니다"
```

- RF Generator SOP의 초반 단계만 발췌하여 SLOT VALVE 답변에 억지 매핑
- 실제 SLOT VALVE 절차(15단계)는 전혀 반영되지 않음
- judge가 3회 retry 후 `max_attempts_reached`로 포기

## 5. Paper A 관점에서의 시사점

### Introduction 인용 근거

> "Aggressive filtering sacrifices recall for shared procedures—standard operating procedures (SOPs) and troubleshooting guides applicable across multiple equipment types—and fails entirely when the equipment type cannot be automatically parsed from query text."

이 케이스는 **device 필터가 정상 작동하더라도** 대상 문서가 인덱스에서 해당 device로 매핑되어 있지 않으면 recall이 0이 되는 문제를 실증한다.

### 개선 방향 (Paper A 시스템 매핑)

| 시스템 | 기대 효과 |
|--------|----------|
| B4 (현재) | device 필터로 contamination 감소, 그러나 이 케이스에서 recall = 0 |
| P1 (scope routing) | shared SOP 예외 처리 — SLOT VALVE SOP가 shared로 분류되면 필터 우회 가능 |
| P2-P4 (router) | implicit query에서도 올바른 device 후보 추론 가능 |

## 6. eval run 대비 현황

- **이전 v3 run**: `task_mode_sop doc_hit@10 = 7/79` (장비 필터 경로 편향)
- **최신 코드 (alias expansion)**: `task_mode_sop doc_hit@10 = 68/79` (+77.2%p 회복)
- alias expansion이 device_name 변형(SUPRA XP ↔ ZEDIUS XP 등)을 해소하여 대부분 회복
- 그러나 **인덱스에 문서 자체가 없는 케이스**(이 SLOT VALVE 사례)는 alias로도 해결 불가

---

## 7. 결론

이 케이스는 Paper A의 F6 failure mode (Recall-Contamination Trade-off)의 전형적 사례이며, hard device filtering만으로는 해결할 수 없는 recall loss를 보여준다. scope_level routing(P1)이나 router-based scope expansion(P2-P4)이 필요한 근거가 된다.
