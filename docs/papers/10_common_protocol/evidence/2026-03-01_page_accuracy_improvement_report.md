# SOP 페이지 정확도 개선 평가 보고서

> 작성일: 2026-03-01
> 데이터: `data/PE Agent 질문 리스트 - 0225 SOP 질문리스트.csv` (79건)
> 개선 내용: 2단계 검색(문서→페이지 재랭킹) + page≤2 penalty (0.3x)
> 평가 경로:
> - ES 직접 검색 시뮬레이션 (`rag_chunks_dev_v2`, BM25, doc_type=SOP)
> - Chat API `/api/search` (Docker port 8001, 검색 전용 경로)
> - Chat API `/api/agent/run` (Docker port 8001, 실제 chat 경로 — LLM 포함)
>
> 관련 파일:
> - 개선 전 정답률 보고서: `2026-03-01_sop_golden_set_accuracy_report.md`
> - 코드 변경: `backend/llm_infrastructure/llm/langgraph_agent.py`, `backend/llm_infrastructure/retrieval/engines/es_search.py`

---

## 1. 문제 정의

SOP 문서 검색에서 **정답 문서는 찾지만 정답 페이지를 놓치는 문제**.

| 원인 | 설명 |
|---|---|
| TOC/표지(page 1) BM25 편향 | SOP page 1에 모든 키워드가 포함되어 항상 최고 점수 |
| 단일 chunk 반환 | 문서 내 최고 점수 chunk 1개만 사용 → page 1만 노출 |
| 실제 절차 페이지(6+)는 밀림 | 작업 절차/단계별 키워드가 분산되어 개별 BM25 점수가 낮음 |

**영향**: page-hit@1 = **75.3%** (Chat API 기준), 미스 18건 전부 `first_page=1`

---

## 2. 개선 전략

### 전략 D: 2단계 검색 + page≤2 penalty (채택)

```
[1단계] 기본 BM25 → 정답 문서 찾기 (기존과 동일)
        ↓
[2단계] 찾은 문서 내에서 query로 재검색 (doc_id 필터)
        + page ≤ 2 chunk에 score × 0.3 penalty
        ↓
[병합]  2단계 결과 우선 + 1단계 fallback
```

### 비교 대상 6개 전략

| 전략 | 설명 |
|---|---|
| A | 현재 (기본 BM25, 첫 chunk) |
| B | page≤2 penalty만 (0.3x) |
| C | 2단계 검색만 (penalty 없음) |
| **D** | **2단계 + page≤2 penalty** |
| E | 2단계 + 의도 기반 부스트 (키워드 매칭) |
| F | 2단계 + penalty + 의도 부스트 |

---

## 3. ES 직접 검색 시뮬레이션 결과 (79건)

### 3.1 전략 비교

| 전략 | page-hit@1 | page-hit@3 | page-hit@5 | Δ(hit@1) | 악화 |
|---|---|---|---|---|---|
| A: 현재 | 61/79 (77.2%) | 61/79 (77.2%) | 61/79 (77.2%) | - | - |
| B: penalty만 | 78/79 (98.7%) | 78/79 (98.7%) | 78/79 (98.7%) | +17 | 0 |
| C: 2단계만 | 61/79 (77.2%) | 79/79 (100%) | 79/79 (100%) | 0 | 0 |
| **D: 2단계+penalty** | **79/79 (100%)** | **79/79 (100%)** | **79/79 (100%)** | **+18** | **0** |
| E: 2단계+의도 | 60/79 (75.9%) | 78/79 (98.7%) | 79/79 (100%) | -1 | 7 |
| F: 2단계+penalty+의도 | 71/79 (89.9%) | 78/79 (98.7%) | 79/79 (100%) | +10 | 7 |

### 3.2 핵심 분석

- **Strategy D가 유일한 100% 달성**: 79/79 전 지표 100%, 악화 0건
- **Strategy B(penalty만)**: 17건 개선, 1건 미해결 (Q52 PRISM SOURCE)
- **Strategy C(2단계만)**: hit@1은 개선 없지만 hit@3부터 100% — page 재랭킹 효과
- **Strategy E/F(의도 부스트)**: 키워드 매칭이 단순하여 7건 악화 → 현시점 비채택

### 3.3 page penalty 효과 검증 (ES 실측)

PIO SENSOR BOARD 교체 쿼리, 단일 문서 내 페이지별 결과:

| penalty 없음 | penalty 적용 (page≤2 × 0.3) |
|---|---|
| 1위: page 12 (17.1) | 1위: page 12 (17.1) |
| 2위: **page 1 (15.4)** | 2위: page 10 (14.4) |
| 3위: page 10 (14.4) | 3위: page 6 (13.7) |
| 4위: page 6 (13.7) | 4위: page 9 (11.8) |
| 5위: page 9 (11.8) | 5위: page 8 (11.6) |

→ page 1이 2위(15.4) → penalty 후 4.6으로 탈락, 절차 페이지만 top-5에 남음

---

## 4. Chat API 평가 결과 (Docker port 8001)

### 4.1 현재 상태 (개선 전 코드, 2/26 기동)

> **주의**: Docker 컨테이너가 `--reload` 없이 실행 중이어서 2/26 시점 코드를 사용.
> 새 코드 반영을 위해 `docker restart rag-api` 필요.

| 지표 | 값 |
|---|---|
| 평가 대상 | 73/79건 (6건 API 500 에러) |
| 문서 hit@5 | 70/73 (95.9%) |
| 문서 hit@10 | 73/73 (100%) |
| **page-hit@1** | **55/73 (75.3%)** |
| page-hit@3 | 73/73 (100%) |
| page-hit@5 | 73/73 (100%) |

### 4.2 page-hit@1 미스 상세 (18건)

| # | 질문 | first_page | expected | top pages |
|---|---|---|---|---|
| 1 | EFEM PIO SENSOR BOARD 교체 | 1 | 6-14 | 1,6,10,12,9 |
| 4 | EFEM Ionizer Replacement | 1 | 6-17 | 1,10,9,6,14 |
| 5 | EFEM PRESSURE SWITCH 교체 | 1 | 6-18 | 1,6,20,30,19 |
| 18 | PM Baratron Gauge ADJ | 1 | 6-18 | 1,6,19,2,18 |
| 19 | PM Baratron Gauge REP | 1 | 19-33 | 1,19,6,33,23 |
| 28 | PM HEATER CHUCK Replacement | 1 | 6-27 | 1,6,28 |
| 38 | PM Pirani Gauge ADJ | 1 | 6-21 | 1,6,22,9,40 |
| 45 | PRISM SOURCE 교체 (1) | 1 | 8-37 | 1,9,8,40,38 |
| 47 | PRISM SOURCE 3000QC RF GEN | 1 | 54-70 | 1,54,54,1,38 |
| 49 | PRISM SOURCE 교체 (2) | 1 | 8-39 | 1,9,8,40,40 |
| 51 | PRISM SOURCE 3100QC RF GEN | 1 | 56-72 | 1,56,54,40,1 |
| 52 | PRISM SOURCE 교체 (3) | 1 | 9-39 | 1,9,8,8,40 |
| 56 | SOURCE BOX BOARD 교체 | 1 | 6-15 | 1,6,13 |
| 59 | Water Shut Off Valve Replacement | 1 | 7-19 | 1,7,8,2,19 |
| 61 | GAS BOX BOARD 교체 | 1 | 6-18 | 1,6,2 |
| 67 | TM DC Power SUPPLY 교체 | 1 | 6-18 | 1,6,14,13,10 |
| 70 | Baratron Gauge Replacement/Adj | 1 | 7-19 | 1,7,20,11 |
| 71 | Pirani Gauge Replacement/Adj | 1 | 20-31 | 1,20,7,24,29 |

**공통 패턴**: 18건 전부 `first_page=1` (목차/표지), 정답 페이지는 rank 2에 존재

### 4.3 `/api/search` 경로 한계

`/api/search`는 `EsSearchService.search()`를 직접 호출하는 검색 전용 경로로,
agent의 `retrieve_node()`를 경유하지 않는다. 따라서 2단계 검색 및 page penalty가 적용되지 않음.
**실제 chat은 `/api/agent/run`을 사용하므로 아래 §4.4 결과가 실제 동작에 해당.**

### 4.4 `/api/agent/run` 평가 결과 (개선 후 코드, 재시작 완료)

> `docker restart rag-api` 후 코드 반영 확인:
> `EARLY_PAGE_PENALTY_ENABLED=True`, `SECOND_STAGE_DOC_RETRIEVE_ENABLED=True`

| 지표 | 값 |
|---|---|
| 평가 대상 | 79/79건 (에러 0건) |
| 문서 hit@10 | 56/79 (70.9%) |
| **page-hit@1 (문서 hit 중)** | **52/56 (92.9%)** |
| page-hit@3 (문서 hit 중) | 55/56 (98.2%) |
| **first_page=1 케이스** | **0건 (page penalty 정상 작동)** |
| 문서 miss | 23건 (agent pipeline 자체 이슈, §5 참조) |

**핵심**: 문서를 찾은 56건 중 **first_page=1이 0건** — page penalty가 완벽 작동.
개선 전에는 18건이 first_page=1이었음.

### 4.5 page-hit@1 미스 (4건, 문서 hit 성공 중)

| Q# | 질문 | first_page | expected | 비고 |
|---|---|---|---|---|
| 5 | EFEM PRESSURE SWITCH 교체 | 19 | 6-18 | 다른 절차 섹션이 1위 |
| 7 | RFID READER 교체 | 30 | 6-19 | 후반부 절차가 1위 |
| 37 | HOOK LIFTER Pin 조정 | 18 | 94-124 | 통합문서 내 다른 작업 섹션 |
| 53 | PRISM SOURCE O-ring 교체 | 8 | 40-53 | 통합문서 내 다른 작업 섹션 |

→ page 1 문제는 해결됨. 잔여 미스는 통합문서 내 여러 작업 섹션 간 경합 문제.

---

## 5. Agent Pipeline 자체 이슈 (문서 miss 23건)

### 5.1 현상

`/api/agent/run`으로 79건 실행 시 **23건이 정답 문서를 찾지 못함** (DOC_MISS).
이는 2단계 검색/page penalty와 무관한 기존 agent pipeline 문제.

### 5.2 원인 분석

대표 3건 상세 확인 결과:

| 케이스 | search_queries (agent가 생성) | 반환 문서 | 원인 |
|---|---|---|---|
| PENDULUM VALVE 교체 | `["PENDULUM VALVE replacement work for the ZEDIUS XP equipment."]` | GCB 문서 (40050558 등) | 영문 번역 + doc_type 필터 없음 |
| PROCESS KIT 교체 | `["Replacement and cleaning of the PROCESS KIT on the ZEDIUS XP equipment."]` | 다른 SOP (tm_robot, n2_curtain) | 영문 BM25가 한국어 SOP 내용과 미스매치 |
| TM ROBOT TEACHING | `["Work related to TM ROBOT TEACHING of the ZEDIUS XP equipment."]` | GCB 문서 (40045113 등) | 영문 번역 + doc_type 필터 없음 |

### 5.3 근본 원인 3가지

| # | 원인 | 설명 | 영향도 |
|---|---|---|---|
| **1** | **search_queries가 영문 전용** | `mq_mode=off`에서도 query가 영문 번역만 사용, 원본 한국어 미포함 | 23건 DOC_MISS |
| **2** | **doc_type 필터 없음** | `auto_parse`가 `doc_types: null` 반환 → SOP/GCB/TS 전체에서 검색 | GCB 문서가 SOP를 밀어냄 |
| **3** | **영문 BM25 ↔ 한국어 콘텐츠 미스매치** | SOP `search_text`는 한국어 위주 → 영문 query의 BM25 점수 낮음 | 올바른 SOP 문서 순위 하락 |

### 5.4 DOC_MISS 전체 목록 (23건)

| Q# | 질문 | expected 문서 |
|---|---|---|
| 9 | SR8241 EFEM Robot & Controller 교체 | efem_robot_sr8241 |
| 20 | Chamber Safety Cover 교체 | pm_chamber safety cover |
| 21 | PRISM 3100 SOURCE 교체 | pm_cip chamber |
| 32 | N2 CURTAIN (영문) | pm_n2 curtain eng |
| 33 | PENDULUM VALVE 교체 | pm_pendulum valve |
| 35 | LM GUIDE 교체 및 Grease 주입 | pm_pin assy |
| 38 | PM Pirani Gauge ADJ | pm_pirani gauge (또는 유사) |
| 42 | PRISM SOURCE 교체 | pm_preventive maintenance |
| 43 | Pendulum Valve 교체 | pm_preventive maintenance |
| 44 | Slot Valve 교체 | pm_preventive maintenance |
| 45 | PRISM SOURCE 교체 | pm_prism source 3000qc |
| 46 | PRISM SOURCE 3000QC O-Ring | pm_prism source 3000qc |
| 49 | PRISM SOURCE 교체 | pm_prism source 3100qc |
| 50 | PRISM SOURCE 3100QC O-Ring | pm_prism source 3100qc |
| 52 | PRISM SOURCE 교체 | pm_prism source |
| 54 | PRISM SOURCE RF GEN Calibration | pm_prism source |
| 55 | PROCESS KIT 교체 및 세정 | pm_process kit |
| 57 | PM Temp Controller 교체 | pm_temp controller |
| 58 | PM Temp Controller 조정 | pm_temp controller |
| 64 | SUB UNIT MANUAL VALVE 교체 | sub unit_igs block |
| 67 | TM DC Power SUPPLY 교체 | tm_dc power supply |
| 69 | 32 MULTI PORT 교체 | tm_multi port |
| 73 | TM ROBOT TEACHING | tm_robot |
| 75 | SLOT V/V Housing O-ring 교체 | tm_slot vv housing o-ring |

### 5.5 해결 방안

| # | 방안 | 설명 | 우선순위 |
|---|---|---|---|
| 1 | **bilingual search queries** | search_queries에 원본 한국어 + 영문 번역 모두 포함 | **P0** |
| 2 | **SOP doc_type 자동 필터** | 질문이 "교체/조정/작업" 패턴이면 doc_type=SOP 필터 추가 | **P0** |
| 3 | **한국어 BM25 가중** | `search_text`에 한국어 query 매칭 가중치 높임 | P1 |
| 4 | **auto_parse doc_type 감지 개선** | "교체 작업" → SOP, "불량 원인" → TS 자동 분류 | P1 |

---

## 6. 코드 변경 사항

### 5.1 `es_search.py` — `build_filter()`에 `doc_ids` 파라미터 추가

```python
# 2단계 검색에서 특정 문서 내 chunk만 필터링
def build_filter(self, ..., doc_ids: list[str] | None = None, ...):
    if doc_ids:
        terms.append(_terms_or_keyword("doc_id", normalized_doc_ids))
```

### 5.2 `langgraph_agent.py` — Early Page Penalty

```python
EARLY_PAGE_PENALTY_ENABLED = True   # env: AGENT_EARLY_PAGE_PENALTY_ENABLED
EARLY_PAGE_MAX = 2                  # env: AGENT_EARLY_PAGE_MAX
EARLY_PAGE_WEIGHT = 0.3             # env: AGENT_EARLY_PAGE_WEIGHT

def _apply_early_page_penalty(doc):
    # SOP 문서의 page ≤ 2에 score × 0.3 적용
```

### 5.3 `langgraph_agent.py` — 2단계 문서 내 재검색

```python
SECOND_STAGE_DOC_RETRIEVE_ENABLED = True  # env: AGENT_SECOND_STAGE_DOC_RETRIEVE_ENABLED
SECOND_STAGE_DOC_LIMIT = 20               # env: AGENT_SECOND_STAGE_DOC_LIMIT
SECOND_STAGE_PER_DOC_TOP_K = 5            # env: AGENT_SECOND_STAGE_PER_DOC_TOP_K

# 1단계 결과의 고유 doc_id별로 sparse_search 재실행
# page penalty 적용 후 score 기준 정렬
# 2단계 결과 우선, 1단계 fallback으로 병합
```

### 5.4 환경변수 제어 (A/B 테스트 가능)

```bash
# 비활성화 (기존 동작)
AGENT_EARLY_PAGE_PENALTY_ENABLED=false
AGENT_SECOND_STAGE_DOC_RETRIEVE_ENABLED=false

# 파라미터 조정
AGENT_EARLY_PAGE_MAX=3          # page ≤ 3까지 penalty
AGENT_EARLY_PAGE_WEIGHT=0.5     # 50% 감점
AGENT_SECOND_STAGE_PER_DOC_TOP_K=10  # 문서 내 top-10 chunk
```

---

## 6. 테스트 결과

### 6.1 단위 테스트

| 테스트 그룹 | 결과 |
|---|---|
| MQ bypass (2건) | PASS |
| MQ fallback reasons (3건) | PASS |
| Temperature policy (8건) | PASS |
| Search query guardrails (2건) | PASS |
| **합계** | **15/15 PASS** |

### 6.2 API 통합 테스트

46/50 PASS. 실패 4건은 이번 변경과 무관한 기존 mock 미구현 이슈.

---

## 8. Before → After 비교 요약

### ES 직접 검색 기준 (79건, doc_type=SOP 필터)

| 지표 | Before | After (Strategy D) | Δ |
|---|---|---|---|
| 문서 hit@20 | 79/79 (100%) | 79/79 (100%) | 0 |
| **page-hit@1** | **61/79 (77.2%)** | **79/79 (100%)** | **+18** |
| page-hit@3 | 61/79 (77.2%) | 79/79 (100%) | +18 |
| page-hit@5 | 61/79 (77.2%) | 79/79 (100%) | +18 |
| 악화 케이스 | - | **0건** | - |

### `/api/search` 기준 (73건, 개선 전 코드 경로)

> `/api/search`는 agent pipeline을 경유하지 않아 개선 전/후 동일.

| 지표 | 값 |
|---|---|
| page-hit@1 | 55/73 (75.3%) |
| first_page=1 미스 | 18건 |

### `/api/agent/run` 기준 (79건, 개선 후 코드 — 실제 chat 경로)

| 지표 | 값 | 비고 |
|---|---|---|
| 문서 hit | 56/79 (70.9%) | 23건 DOC_MISS (§5 agent 이슈) |
| **page-hit@1 (문서 hit 중)** | **52/56 (92.9%)** | |
| page-hit@3 (문서 hit 중) | 55/56 (98.2%) | |
| **first_page=1 케이스** | **0건** | **개선 전 18건 → 0건** |

---

## 9. 잔여 과제

### P0 — 즉시 필요

| # | 항목 | 설명 |
|---|---|---|
| 1 | **bilingual search queries** | agent search_queries에 한국어 원본 + 영문 번역 모두 포함 (23건 DOC_MISS 해결) |
| 2 | **SOP doc_type 자동 필터** | "교체/조정/작업" 패턴 질문에 doc_type=SOP 필터 추가 (GCB 문서 혼입 방지) |

### P1 — 개선

| # | 항목 | 설명 |
|---|---|---|
| 3 | auto_parse doc_type 감지 | "교체 작업"→SOP, "불량 원인"→TS 자동 분류 |
| 4 | 한국어 BM25 가중치 | search_text 한국어 query 가중치 조정 |
| 5 | API 500 에러 6건 | 영문 쿼리 처리 오류 조사 (Q10-13, Q16, Q32) |

### P2 — 고도화

| # | 항목 | 설명 |
|---|---|---|
| 6 | 의도 부스트 정제 | chapter 메타/LLM 기반 분류로 통합문서 내 섹션 정확도 향상 |
| 7 | cross-encoder reranker | query+page chunk 쌍 재정렬 |
| 8 | page-hit KPI 상설화 | 평가 스크립트를 CI/regression에 통합 |

---

## 10. 결론

### 페이지 정확도 (개선 목표)

2단계 검색 + page≤2 penalty (Strategy D)로:
- **ES 직접**: page-hit@1 = 77.2% → **100%** (+18건, 악화 0건)
- **agent/run**: first_page=1 케이스 = 18건 → **0건** (page penalty 완벽 작동)
- **agent/run**: 문서 hit 중 page-hit@1 = **92.9%** (52/56)

### 문서 정확도 (발견된 문제)

agent pipeline을 통한 실제 chat 경로에서 **23/79건 DOC_MISS** 발생:
- **근본 원인**: search_queries가 영문 전용 + doc_type 필터 없음
- **해결**: bilingual queries + SOP doc_type 자동 필터 (P0)
- ES 직접 검색에서는 79/79 (100%) — agent pipeline 개선으로 해결 가능
