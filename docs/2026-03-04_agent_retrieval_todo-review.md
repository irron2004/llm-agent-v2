# Agent Retrieval TODO + 실행 계획 리뷰 (2026-03-04)

> `docs/2026-03-02_agent_retrieval_todo.md` 및 `.sisyphus/plans/agent-retrieval-todo-2026-03-02.md`에 대한 리뷰.
> 2026-03-01 `/api/agent/run` 79건 SOP 평가 + bilingual 수정 실험 데이터를 근거로 작성.
> 2026-03-04 코드 구현 완료 후 검토 결과 추가.

---

## 1. TODO 문서 변경 확인

이전 리뷰 피드백이 모두 반영됨:

| 피드백 | 반영 위치 |
|--------|-----------|
| T0 티켓 추가 | 작업 분해 T0-1~T0-3 |
| Phase 0 baseline env var 조합 명시 | Phase 0 작업 #3 + `env_manifest.json` 산출물 |
| Phase 5 → 5a/5b 분리 | Phase 5a(판정 개선, 2일) + Phase 5b(KPI MVP, 1일) |
| 구현-note 섹션 추가 | RRF, SOP intent, Sticky, Eval harness 각각 |

구현-note들이 코드 위치(`es_search.py`, `rrf.py`)와 설정 키(`AGENT_SOP_INTENT_MODE`, `AGENT_SOP_SOFT_BOOST_FACTOR`)까지 명시되어 있어 실행 가이드로 바로 사용 가능한 수준.

---

## 2. 코드 구현 검토 (2026-03-04)

### 테스트 현황: 17 passed, 1 failed

```
PASSED:  test_app_rrf_merge (4), test_es_search_engine_app_rrf (3),
         test_es_search_service_rrf_metadata (2), test_sop_intent_heuristic (3/4),
         test_agent_sticky_policy_followup_only (3),
         test_agent_rrf_and_sticky_gates (2)
FAILED:  test_hard_sets_sop_doc_type_but_does_not_override_strict_filter_doc_types
```

### 2-A. RRF 직접 구현 — 완료, 문제 없음

| 파일 | 상태 | 내용 |
|------|------|------|
| `backend/llm_infrastructure/retrieval/rrf.py` | 완료 | 순수 RRF merge. 1-based rank, dedupe(`chunk_id`→`page`→`doc_id`), 5단계 deterministic tie-break |
| `es_search.py` (`_hybrid_search_rrf`) | 완료 | dense/sparse 별도 호출 → `rrf_merge_hits()` → source hit 선택(rank 기반) → metadata 주입 |
| `es_hybrid.py` | 완료 | `use_rrf=True` 기본값, `rrf_k=60` |
| RRF debug metadata | 완료 | `metadata.retrieval_meta.rrf.*` (rrf_dense_rank, rrf_sparse_rank, rrf_score, rrf_k) |

- Window sizing: `min(max(top_k * 2, 50), 200)` — 합리적
- ES native RRF 경로 완전 제거됨
- 테스트 9개 모두 통과, 10회 반복 결정성 검증됨

### 2-B. Stage2 retrieval — 제거됨, 복원 필요

이전에 `langgraph_agent.py`에 있던 코드가 **완전히 제거**됨:
- `_apply_early_page_penalty()` (SOP page≤2 score×0.3)
- 2nd stage doc-specific search (doc_id 필터로 문서내 재검색)
- `AGENT_EARLY_PAGE_PENALTY_ENABLED`, `AGENT_SECOND_STAGE_DOC_RETRIEVE_ENABLED` 환경변수

sisyphus plan Task 0에서 "stage2 not implemented; N/A"로 문서화되었으나, **이전 실험에서 stage2 + page penalty가 ES 시뮬레이션 기준 page-hit@1 100%를 달성**했고, agent/run에서도 first_page=1 케이스를 0으로 줄인 핵심 기능.

**복원 필요 사항**:
1. `_apply_early_page_penalty()` — SOP page≤2 penalty (TOC/표지 BM25 편향 제거)
2. 2nd stage doc-specific search — 1st stage에서 찾은 doc_id로 문서 내 page 재검색
3. Stage2 병합에 RRF 적용 — EN/KO 쿼리 결과의 score 스케일 비대칭 해소 (TODO Step 3)
4. 환경변수 복원 — `AGENT_EARLY_PAGE_PENALTY_ENABLED`, `AGENT_SECOND_STAGE_DOC_RETRIEVE_ENABLED`

**근거 (2026-03-01 실험 데이터)**:
- ES 시뮬레이션: Strategy D (2-stage + page≤2 penalty) → page-hit@1/3/5 모두 100%
- agent/run (stage2 적용 시): first_page=1 케이스 18→0건
- agent/run (stage2 없을 때): BM25가 TOC 페이지(page 1-2)를 항상 1위로 반환

### 2-C. SOP intent — 구현 완료, 버그 1건

**구현 상태**:
- `_is_sop_intent()`: SOP 후보 패턴(`교체|점검|절차|작업|calibration|teaching|adjust|replacement|...`) + TS 배제 패턴(`불량|원인|알람|고장|error|troubleshoot|...`) — 정상 동작
- `auto_parse_node`: hard mode → `expand_doc_type_selection(["SOP"])`, soft mode → `sop_intent=True` 플래그만 설정
- `retrieve_node`: soft mode에서 SOP doc_type 문서에 `score *= sop_soft_boost_factor` 적용
- 설정: `sop_intent_mode=soft` (기본), `sop_soft_boost_factor=1.05`

**버그: `strict_doc_type_override` 시 기존 선택이 덮어씌워짐**

`auto_parse_node` (line 2764-2786) 흐름:
```python
# 1. strict 감지 → SOP intent 스킵 (정상)
strict_doc_type_override = bool(state.get("selected_doc_types_strict")) or ...
# 2. 쿼리에서 doc_type 키워드 탐지 → 없으면 []
detected_doc_types = _extract_doc_types_from_query(query)  # → []
# 3. needs_history 없음 → prev도 []
prev_doc_types = [] if not allow_history_inheritance else ...
# 4. 최종: 빈 배열로 덮어씌움
doc_types = detected_doc_types if detected_doc_types else prev_doc_types  # → []
```

사용자가 `selected_doc_types=["Trouble Shooting Guide"]` + `selected_doc_types_strict=True`로 명시 선택했는데 `auto_parse_node`가 `[]`로 덮어씌움.

**수정 방향**: `strict_doc_type_override=True`일 때 기존 `state.get("selected_doc_types")`를 보존:
```python
if strict_doc_type_override:
    doc_types = list(state.get("selected_doc_types") or [])
else:
    doc_types = detected_doc_types if detected_doc_types else prev_doc_types
```

### 2-D. Sticky 정책 — 구현 완료, 문제 없음

- Graph 순서: `START → history_check → (query_rewrite | passthrough) → auto_parse` — `needs_history`가 auto_parse보다 먼저 결정됨
- `needs_history=True`일 때만 `prev_devices/prev_doc_types/prev_equip_ids` 상속 허용
- `needs_history=False`이면 모두 빈 배열
- 테스트 3건 통과 (상속 허용 / 차단 / thread_id 없음)

### 2-E. 평가 하네스 — 구현 완료, 보완 2건

| 파일 | 역할 |
|------|------|
| `evaluate_sop_agent_page_hit.py` | 배치 실행, JSONL 출력, `--use-testclient`, `--repeat N` |
| `validate_agent_eval_jsonl.py` | JSONL 스키마 검증 (required key/type/enum) |
| `agent_eval_report.py` | before/after 비교 리포트 + regression 감지 + Jaccard@k |
| `_fixtures/good.jsonl`, `bad.jsonl` | validator 테스트 데이터 |

JSONL 스키마 필드: `q_id`, `question`, `expected_doc`, `expected_pages`, `search_queries`, `docs[{doc_id,page,score,rank}]`, `result`, `trace{route,auto_parse,mq_mode,retry_count}`, `env_manifest{git_sha,index_name,timestamp}`

**보완 1: JSONL에 `answer` 필드 미저장**
- TODO 요구사항에는 답변(answer) 저장이 포함되어 있음.
- 현재 evaluator는 `answer`를 읽어서 판정에 사용하지만, JSONL row에는 기록하지 않음.
- 영향: `ANSWER_MISS` 사후 분석 시 원문 답변 재현이 어려움.
- 조치: JSONL 스키마에 `answer` 추가 + validator 동기화.

**보완 2: `--use-testclient`는 실플로우 품질검증이 아님**
- testclient 모드에서 `_new_auto_parse_agent`, `_new_hil_agent`, `LangGraphRAGAgent`를 fake로 교체함.
- 따라서 이 모드는 `/api/agent/run` 스키마/하네스 검증에는 유효하지만, 실제 검색/답변 품질 검증 근거로는 제한적.
- 조치: 문서에 용도 명시
  - `--use-testclient`: 스키마/도구 검증
  - 실품질(정확도/일관성): 실제 서버 HTTP 모드로 측정

### 2-F. MQ 정량 분석 자동화 — 코드 미완료

- TODO(P1-5) 요구: `mq_mode=off|fallback|on` 동일 조건 비교 자동화.
- 현재 evaluator는 `mq_mode` sweep 실행/집계를 직접 지원하지 않음.
- 영향: MQ 효과를 재현 가능한 방식으로 비교하기 어려움.
- 조치: evaluator에 mode sweep 옵션 추가 + `mq_ablation_report.md` 자동 생성.

### 2-G. boost factor 1.05 — 미조정

이전 리뷰(2-1)에서 1.05가 너무 작다고 지적한 상태. 코드에 아직 1.05가 기본값. RRF score 범위(0.01~0.05)에서 5% boost는 rank 1개 gap(≈0.00027)도 넘기 어려움. 1.2~1.5 상향 또는 rank 보정 방식 검토 필요.

---

## 3. Sisyphus 실행 계획 리뷰

### 잘 된 부분

- **4-Wave 병렬 구조**: TODO의 권장 실행 순서와 일치하며, 독립 작업이 병렬로 배치됨.
- **Task 0, 1 완료 ([x])**: RRF merge 유틸리티와 discovery가 끝나 Wave 2 즉시 착수 가능.
- **Tests-first 접근 (Task 2)**: sticky 정책의 행동 명세를 코드보다 선행 — 구현 후 테스트가 바로 통과하는지 확인 가능.
- **Decision-complete 표시**: dedupe key 우선순위, tie-break 규칙, window size 공식 등이 확정되어 구현 시 판단 지연 없음.
- **`--use-testclient` 모드 (Task 8)**: 실제 ES/vLLM 없이 검증 가능, agent-executable 원칙에 부합.
- **Commit strategy 3개**: RRF / 정책 / 하네스로 묶어 리뷰 단위로 적절.

### 확인 필요 사항

#### 3-1. SOP soft boost factor 1.05가 너무 작을 가능성

Task 6에서 `sop_soft_boost_factor: float = 1.05` (기본값).

RRF 적용 후 score는 `sum(1/(k+rank))`이므로 절대값이 0.01~0.05 범위. 여기에 ×1.05는 0.0005~0.0025 차이.

실험 데이터에서 GCB가 SOP보다 1~2 rank 위에 오는 케이스가 있었는데, RRF에서 rank 1개 차이의 score gap은:
```
1/(60+1) - 1/(60+2) ≈ 0.00027   (k=60 기준)
```

1.05 boost로는 이 gap을 넘기 어려움.

**제안**:
- 기본값을 **1.2~1.5** 정도로 올리거나
- Score 곱셈 대신 **rank 보정** (SOP 문서의 rank를 N만큼 올림)이 더 효과적일 수 있음
- 어느 쪽이든 Phase 3 검증에서 `soft_boost_factor` 값별 비교 실험을 포함해야 함

#### 3-2. Stage2 retrieval 제거됨 + Task에 없음

TODO 문서의 1) 개선방법 Step 3:
> "stage2(문서내 재검색) 결과도 동일한 RRF 병합 규칙을 재사용해 EN/KO 점수 스케일 비대칭을 제거한다."

구현-note에도 반영됨. 하지만 sisyphus plan의 **Task 4는 `es_search.py`의 `hybrid_search`만 다루고**, `langgraph_agent.py`의 stage2 병합 경로는 커버하지 않음.

**코드 검토 결과**: stage2 코드(`_apply_early_page_penalty`, 2nd stage doc-specific search, `AGENT_EARLY_PAGE_PENALTY_ENABLED`, `AGENT_SECOND_STAGE_DOC_RETRIEVE_ENABLED`)가 **완전히 제거**된 상태. sisyphus plan Task 0에서 "stage2 not implemented; N/A"로 문서화.

그러나 2026-03-01 실험에서 stage2 + page penalty가 page-hit@1 핵심 기능이었음:
- ES 시뮬레이션: Strategy D (2-stage + page≤2 penalty) → page-hit@1/3/5 모두 100%
- agent/run: first_page=1 케이스 18→0건
- stage2 없으면 BM25가 TOC 페이지(page 1-2)를 항상 1위로 반환

**제안**:
- **Stage2 retrieval 코드 복원** (page penalty + doc-specific 재검색)
- Stage2 병합에 RRF 적용하여 EN/KO score 스케일 비대칭 해소
- **Task 4b: stage2 retrieval 복원 + RRF 통합** 별도 추가 (Wave 2, Blocked By: T1, T4)

#### 3-3. Critical Path 정의가 부정확

TL;DR의 critical path:
```
App-level RRF → Sticky gating → SOP intent policy → Eval harness
```

이는 선형 의존처럼 보이지만, 실제 dependency matrix:
```
T1 → T4/T5 → T8   (RRF 체인)
T2 → T6/T7 → T8   (정책 체인, RRF와 독립)
T3 → T8            (스키마 체인)
```

Sticky(T7)와 SOP intent(T6)는 RRF(T4)에 의존하지 않으므로 **Wave 2와 Wave 3가 실제로는 병렬 가능**. TL;DR의 선형 표기가 불필요한 직렬화를 암시할 수 있음.

**제안**: Critical path를 두 갈래로 명시:
```
Critical Path A: T1 → T4 → T8 (RRF → Eval)
Critical Path B: T2 → T6/T7 → T8 (정책 → Eval)
Merge point: T8 (양쪽 완료 후)
```

#### 3-4. Bilingual 코드 수정의 위치가 불명확

이전 세션에서 `_prepare_retrieve_node`에 `query_ko` 추가 수정을 적용했다 (`langgraph_rag_agent.py`). 이 수정이 RRF의 bilingual score 비대칭 해소의 전제 조건인데, plan에는 포함되어 있지 않음.

**확인 필요**:
- bilingual 수정이 이미 코드에 반영된 상태 → plan에 전제 조건으로 명시
- 아직 미반영 → 별도 task로 추가 (Task 0.5 또는 Task 4의 선행 조건)

#### 3-5. Scope 표기가 부정확

Interview summary: "implement **P0 + P1**"

실제 포함 범위:
- P0: 1) RRF ✓, 2) SOP intent ✓, 3) Sticky ✓
- P1: 4) Eval harness ✓, **5) MQ 분석 ✗, 6) Follow-up 판정 ✗**

의도적 제외라면 scope를 **"P0 전체 + P1-4(eval harness)"**로 명시하는 게 정확.

#### 3-6. JSONL 스키마 필드명이 내부에서 충돌

Sisyphus plan Task 3의 스키마 정의는:
- `trace.retry_count`

하지만 Task 8의 수집 항목은:
- `metadata.attempts`

동일 의미 필드가 이름만 다르면 validator/리포터에서 파싱 분기가 생기고, before/after 비교 자동화가 불안정해진다.

**제안**:
- 스키마를 `trace.retry_count`로 고정하고
- evaluator에서 `metadata.attempts -> trace.retry_count`로 정규화 매핑
- validator는 `retry_count`만 required로 검사

#### 3-7. Sticky 상속 범위(`doc_type` vs `devices/equip_ids`) 명시 필요

현재 Task 7은 follow-up 시 `prev_devices/prev_doc_types/prev_equip_ids`를 모두 상속하도록 정의되어 있다.

하지만 대화 의도상 "새 세션에서 doc_type은 가져오지 않는다"가 핵심이었고, devices/equip_ids까지 상속할지 여부는 명시적 합의가 필요하다.

**제안**:
- Plan에 sticky 범위를 명확히 선언:
  - 옵션 A: `doc_type`만 follow-up 상속
  - 옵션 B: `doc_type + devices + equip_ids` 모두 상속
- 현재 요구가 A라면 Task 7/Task 2 테스트를 `doc_type-only` 기준으로 수정

#### 3-8. MQ 효과 분석이 실행 TODO로 분리되지 않음

현재 plan은 P1에서 evaluator(Task 8)까지는 포함하지만, MQ on/off 효과 측정(정확도/일관성 비교)이 별도 task로 구조화되어 있지 않다.

이 항목은 이전 대화에서 우선 검증 항목으로 합의된 내용이므로, "구현 완료 후 성능 검증" 단계에서 누락되면 결과 해석이 불완전해진다.

**제안**:
- **Task 8b: MQ on/off 비교 실험** 추가
  - 입력: 동일 질문셋, 동일 seed/설정
  - 출력: doc-hit/page-hit/Jaccard@k/답변 품질 요약
  - 산출물: `mq_ablation_report.md`

---

## 4. 리스크 요약

| 리스크 | 구분 | 영향 | 대응 |
|--------|------|------|------|
| **Stage2 retrieval 제거됨** | 코드 | page-hit@1 하락 (TOC 편향 미제거) | 복원 + RRF 적용 (2-B) |
| **SOP strict override 버그** | 코드 | 사용자 명시 doc_type 선택이 무시됨 | auto_parse_node 수정 (2-C) |
| SOP boost factor 부족 | 코드 | GCB 혼입 미해결 | 기본값 1.2~1.5 상향 (2-G) |
| Stage2 RRF task 누락 | Plan | stage2 병합에 EN/KO 스케일 비대칭 | Task 4b 추가 (3-2) |
| Critical path 오해 | Plan | 불필요한 직렬화로 일정 지연 | 두 갈래 병렬 명시 (3-3) |
| Bilingual 전제 조건 | Plan | RRF 효과 미검증 | 코드 반영 상태 확인 (3-4) |
| Scope 혼동 | Plan | MQ/follow-up 미구현 기대 불일치 | scope 표기 수정 (3-5) |
| JSONL 필드명 충돌 | Plan | validator/리포터 파싱 불일치 | `trace.retry_count` 단일화 (3-6) |
| Sticky 범위 불명확 | Plan | 세션 동작 기대와 구현 불일치 | doc_type-only vs full 명시 (3-7) |
| MQ 분석 누락 | Plan | 개선 효과 원인 분리 불가 | Task 8b 추가 (3-8) |
| JSONL answer 미저장 | 코드 | ANSWER_MISS 원인 재현 어려움 | `answer` 필드 추가 + validator 동기화 (2-E) |
| testclient 모드 오해 | 코드/운영 | 실품질 검증 근거 왜곡 가능 | 용도 분리 명시 + HTTP 모드 병행 (2-E) |

---

## 5. 결론

P0 + P1-4 구현이 대부분 완료됨. RRF, Sticky, 평가 하네스는 정상 동작 확인.

**즉시 수정 필요** (코드 버그):
1. **Stage2 retrieval 복원** — page-hit@1의 핵심 기능. `_apply_early_page_penalty` + 2nd stage doc-specific search + stage2 RRF 적용 (2-B)
2. **SOP strict override 버그 수정** — `strict_doc_type_override=True`일 때 기존 `selected_doc_types` 보존 → 테스트 1건 통과 (2-C)
3. **SOP boost factor 상향** — 1.05 → 1.2~1.5 (2-G)
4. **평가 JSONL에 `answer` 저장** — validator와 함께 스키마 확장 (2-E)
5. **`--use-testclient` 용도 분리 명시** — 스키마 검증 전용, 실품질은 HTTP 모드 기준 (2-E)
6. **MQ mode sweep 자동화 추가** — `off|fallback|on` 비교 리포트 자동 생성 (2-F)

**Plan 수정 권장**:
4. Stage2 RRF task 추가 (3-2)
5. Scope 표기 정정 (3-5)
6. JSONL 필드명 단일화 (3-6)
7. Sticky 상속 범위 명시 (3-7)
8. MQ on/off 분석 task 추가 (3-8)

**착수 전 확인 사항**:
9. Bilingual 코드 반영 상태 확인 (3-4)
10. Critical path 두 갈래 명시 (3-3)

---

## 6. 전체 문서 기반 문제-해결-결과 분석 (2026-02-26 ~ 03-04)

> 11개 문서를 날짜순으로 읽고, 각 시점에 **(1) 어떤 문제가 정의되었는지 → (2) 어떤 개선/실험을 했는지 → (3) 어떤 결과/해결이 있었는지 → (4) 추가로 확인이 필요한지**를 한 흐름으로 정리.

### 6-1. 날짜순 타임라인

| 날짜 | 문제 정의 | 개선/실험(조치) | 결과/해결 상태 | 추가 확인 필요 |
|------|-----------|----------------|---------------|---------------|
| **02-26** | **동일 쿼리인데 run마다 다른 문서가 나옴(검색 비일관성)**. `deterministic=false`에서 top-5 doc_id가 5회 실행 중 4개 패턴으로 흔들림. | `deterministic` 플래그 on/off 비교로 재현. 원인 후보를 **MQ(LLM multi-query expansion) sampling 변동성**으로 규정. | `deterministic=true`에서는 5회 모두 동일 top-5로 안정화(1개 패턴). 즉 **불안정의 "주요 레버"가 MQ 경로**라는 결론. | (1) 운영 기본값을 deterministic/skip MQ로 둘지 정책 확정 필요. (2) 변동의 "경로"가 바뀌는지(혼합 doc_id scheme) 추가 로깅 필요. |
| **02-28** | 위 문제들이 "과거 이슈"가 아니라 **실사용 대화 데이터에서도 재현**됨. 반복 질문군에서 문서 집합 Jaccard가 매우 낮고(예: APC position 계열 0.046~0.080), doc_refs 누락 턴도 존재. 또한 `/api/agent/run`에서 동일 질문 3회 실행 시 `st_gate` 분기와 top5가 run마다 달라짐. | 실데이터(`/api/conversations`) 100세션 분석 + `/api/retrieval/run`, `/api/agent/run` 실시간 재현. **MQ 쿼리에 근거 불명 수치(토크/압력 등)가 섞이는 현상**, retry tier(확장→쿼리정제→MQ재생성) 구조로 변동성이 누적될 수 있음을 정리. | 결론: "아직 재현된다". 해결보다는 **원인 구조화/계측/정책 제안** 단계. 제안: 운영 기본 경로 결정성 확보, search_queries 가드레일, route/st_gate/attempt/retry_strategy/search_queries 저장 등. | (1) 제안한 메타데이터 저장이 실제로 적용/수집되는지 확인 필요. (2) `mq_mode` 계약(off/fallback/on)과 실제 런타임 동작이 일치하는지 회귀 검증 필요. |
| **03-01** (SOP baseline) | SOP 79문항 기준으로 **문서는 찾는데 "정답 페이지"를 놓치는 구조적 문제**가 핵심으로 드러남: page-hit@1 67.1%, page-hit@10 77.2%인데 문서 hit@10은 100%. 원인은 **목차/표지(page 1) BM25 편향**. 또 운영 서버가 `rag_synth` 합성 인덱스를 쓰면 실제 SOP 성능이 반영되지 않는 설정 이슈도 지적. | ES 직접 검색 기반으로 정답률/오류 케이스 분해: (1) page 1로 쏠림 18건, (2) Top-5 밖 3건(유사 문서 경합: PRISM SOURCE, Pirani/Pressure Gauge 등). 운영 인덱스 전환 필요성도 명시. | "문서정확도는 충분, 페이지정확도가 병목"이라는 문제정의 완료. 인덱스 미스매치는 **운영환경 레벨 P0** 이슈로 정리. | (1) 운영 인덱스가 실제로 `rag_chunks`로 고정됐는지 확인 필요. (2) Top-5 밖 3건은 reranker/문서 그룹핑/메타 강화 등 후속 실험 필요. |
| **03-01** (페이지 정확도 개선) | 위 SOP baseline의 핵심 문제(목차 1등) 때문에 **정답 페이지 안내 실패**가 발생. | **2단계 검색(문서→문서내 재검색) + page≤2 penalty(0.3x)** 전략(Strategy D)을 설계/비교. ES 시뮬레이션에서 Strategy D가 page-hit@1/3/5 모두 100% (79/79) 달성. | 페이지 "1등이 목차인 현상"은 **해결 방향이 매우 명확**해짐(2-stage+penalty가 결정적). 다만 `/api/agent/run`에서는 문서 hit 자체가 56/79로 낮게 나와(DOC_MISS 23건) **페이지 개선만으로는 완성되지 않음**도 함께 발견. | (1) agent 파이프라인 DOC_MISS 원인(영문 전용 검색어, doc_type 필터 부재)을 동시에 해결해야 최종 KPI가 올라감. (2) 통합문서 내 "다른 섹션이 1위"로 뜨는 잔여 4건류는 rerank/섹션 메타가 필요. |
| **03-01** (agent/run 전/후) | 동일한 페이지정확도 개선을 `/api/agent/run` 기준으로 전/후 비교했을 때, **page-hit@1은 크게 개선되지만(doc-hit 일부 하락)** trade-off가 존재. | env 토글로 early page penalty + stage2 doc-local 검색 적용 전/후 비교. page-hit@1: 15/79(19.0%) → 39/79(49.4%). doc-hit@10: 57/79(72.2%) → 56/79(70.9%)로 소폭 하락. | 페이지정확도는 개선(특히 @1), 문서정확도는 일부 하락 → **stage1/stage2 병합 규칙/컷오프 보정 필요**라는 결론. | (1) stage2 결과를 어떻게 병합해야 doc-hit 저하를 막는지(가중/우선순위/정렬 규칙) 추가 실험 필요. |
| **03-01** (Bilingual 누락 버그) | `/api/search`는 한국어 원문으로 검색해서 잘 맞는데, `/api/agent/run`은 번역 후 **영문 query만 넣어 한국어 SOP를 놓치는 버그**가 확인됨. 동일 인덱스인데 doc hit@10이 100%(search) vs 70.9%(agent/run)로 크게 차이. | `_prepare_retrieve_node`에서 `query_en`만 쓰던 로직을 수정해 **EN+KO를 모두 `search_queries`에 넣도록** 변경. | 문서 hit@10: 56/79(70.9%) → 60/79(75.9%)로 개선(DOC_MISS 23→19). 하지만 **page-hit@1은 하락**(EN/KO 각각 다른 chunk 반환 + 점수스케일 차이로 병합 순서가 흔들림). SOP intent/doc_type 자동 세팅(필터/부스트)이 "추가 발견"으로 기록. | (1) bilingual을 "그냥 2개 쿼리 합치기"가 아니라 **RRF 등 안정적 병합으로** 바꿔야 함. (2) SOP intent/doc_type 자동 세팅(필터/부스트)이 실제로 DOC_MISS를 얼마나 줄이는지 검증 필요. |
| **03-02** (테스트 요약) | 핵심 문제 4개를 재정리: (1) 목차 1등, (2) 한국어인데 영어로만 검색, (3) GCB 과다로 SOP 밀림, (4) hybrid search가 RRF 실패 시 script_score fallback되어 의미적으로 맞는 문서가 후보에서 빠짐. | (1) 2-stage + penalty로 목차 1등 문제 해결(목차 1등 18→0, page-hit@1 19%→49%), (2) bilingual로 DOC_MISS 23→19 개선. 다음 할 일로 **앱 레벨 RRF**, **SOP 자동 인식**, **평가 자동화** 제시. | "페이지 1등 목차"는 해결 방향 확정, "영문전용 검색"은 부분 개선. 하지만 GCB 혼입, script_score fallback, RRF 필요성이 다음 단계로 확정. | (1) ES 버전 의존을 끊는 app-level RRF가 실제로 fallback을 0으로 만드는지 확인 필요. (2) doc_type 자동 인식이 오탐 없이 작동하는지(precision/recall) 측정 필요. |
| **03-02** (TODO/계획 수립) | 이제 문제를 "개별 버그"가 아니라 **운영 `/api/agent/run` 기준으로 안정화**해야 한다는 목표를 명시. | P0: app-level RRF, SOP intent 규칙, doc_type sticky 정책. P1: `/api/agent/run` 평가 하네스(JSONL), MQ 효과 정량분석, follow-up 판정 개선, KPI 상설화까지 단계/티켓/완료기준을 설계. | 실행 계획이 "검증 원칙(항상 agent/run)"과 "리포트 포맷"까지 고정되어 **재현 가능한 개선 루프**를 만들려는 방향으로 정리됨. | **(중요)** 당시 문서엔 stage2가 "미구현(N/A)"로 적혀 있는데, 03-01 실험 문맥에서는 stage2가 핵심이었음 → **stage2 코드의 존재/삭제 여부를 반드시 확인**해야 함. |
| **03-04** (TODO 리뷰/구현 검토) | P0/P1 일부가 실제 구현된 뒤, **새로운 상태의 문제점/누락**이 드러남. | 구현 검토 결과: (A) app-level RRF 구현 완료(테스트 대부분 통과), (B) SOP intent/Sticky/평가 하네스 구현 완료, (C) **Stage2 retrieval 코드에서 제거되어 복원 필요**, (D) SOP intent strict override 시 사용자 선택 doc_type이 빈 배열로 덮이는 버그 1건, (E) SOP soft boost factor 1.05가 너무 작을 가능성, (F) eval JSONL에 answer 미저장, (G) MQ mode sweep 자동화 미완료. | "RRF/정책/하네스"는 전반적으로 진척됐지만, **page-hit을 결정적으로 올리던 stage2가 빠져있으면 회귀 가능성이 큼**이라는 강한 경고가 핵심 결론. | (1) stage2 복원 + stage2 병합에도 RRF 적용 필요. (2) strict override 버그 수정 후 재평가 필요. (3) MQ on/off/fallback을 자동으로 비교하는 실험 파이프가 아직 부족. |

### 6-2. 핵심 인과 관계

#### A. 문서 정확도 흔들림의 뿌리: MQ 비결정성과 경로 분기

- 02-26에 **deterministic=false에서 동일 쿼리가 다른 문서로 흔들리는 현상**을 재현했고, 원인을 MQ(LLM query expansion)로 좁힘.
- 02-28에는 실사용 대화에서도 반복질문 Jaccard가 낮고, agent/run에서 st_gate 분기 + MQ 쿼리 "근거 없는 수치 삽입" 등으로 흔들림이 커질 수 있음을 확인.

#### B. 페이지 정확도 병목: 목차/표지 BM25 편향

- 03-01 SOP 평가에서 **문서는 맞는데 페이지가 틀리는** 현상을 구조 문제(TOC page 1 키워드 집중)로 규정.
- 같은 날 2-stage + early page penalty로 "목차 1등"을 사실상 제거할 수 있음을 실험적으로 보여줌(ES 시뮬레이션 100%, agent/run에서도 first_page=1 0건).

#### C. agent/run DOC_MISS의 직접 원인: 영문 전용 검색 + doc_type 필터 부재

- 03-01 page 개선을 agent/run에 넣어도 DOC_MISS가 크게 남는 이유가 **search_queries가 영문 전용**이었고, SOP인데 doc_type 필터가 없어 GCB가 섞여 정답 문서를 밀어냈기 때문으로 정리.
- bilingual 버그 수정으로 doc miss는 줄었지만, EN/KO 병합이 점수스케일 문제로 page-hit@1을 떨어뜨리는 부작용이 생겨 **병합 알고리즘(RRF)이 다음 병목**으로 넘어감.

### 6-3. 추가 확인이 필요한 항목 (우선순위별)

#### P0급 — 지금 상태를 결정적으로 좌우

**1. Stage2 retrieval(문서내 재검색 + early page penalty) 복원 여부 확인**
- 03-01 실험에서는 page-hit@1을 끌어올리고 first_page=1을 0으로 만드는 핵심 장치였는데, 03-04 리뷰에선 해당 코드가 "완전히 제거"되었다고 확인됨.
- 현재 코드에 stage2가 없으므로, page-hit 관련 KPI가 **회귀**했을 가능성이 큼.

**2. Bilingual 병합 로직 안정화(RRF 적용 범위 확인 포함)**
- bilingual로 doc-hit은 개선됐지만 page-hit@1이 하락했고, 원인이 EN/KO 결과 점수 스케일 차이 + 단순 점수 병합.
- 03-04에는 app-level RRF가 구현되었으므로: **(a) EN/KO 병합에도 RRF가 적용되는지**, **(b) stage2 병합에도 동일 규칙이 적용되는지**가 핵심 확인 포인트.

**3. SOP intent + doc_type 필터/부스트가 "정말로 GCB 혼입을 줄이는지"**
- 03-01~03-02에서 GCB가 SOP를 밀어내는 문제가 반복 언급되었고, 03-02 TODO에서 SOP intent(hard/soft)로 해결하려 했음.
- 03-04 리뷰에서 SOP intent는 구현되었지만, **strict override에서 사용자가 선택한 doc_type이 빈 배열로 덮이는 버그** 발견됨(운영에서 치명적).

**4. 운영 인덱스 설정 고정 여부**
- SOP 성능이 제대로 나오려면 `rag_synth` 같은 합성 인덱스가 아니라 SOP가 들어있는 운영 인덱스(`rag_chunks...`)로 고정되어야 한다는 지적 있음.

#### P1급 — 원인 분리/재발 방지에 필수

**5. MQ mode(off/fallback/on) 효과를 "자동으로" 비교하는 실험 파이프**
- 02-26, 02-28에서 MQ가 불안정의 핵심 변수로 지목됐고, 03-02 TODO에서 MQ 효과 정량 분석을 계획했지만, 03-04 리뷰에선 **자동화가 미완료**로 남아 있음.

**6. 평가 JSONL에 `answer` 저장 + 필드명 충돌 정리**
- 03-02 TODO는 answer까지 저장해야 사후 분석이 가능하다고 요구했고, 03-04 리뷰는 실제 구현에서 answer 저장이 빠졌다고 지적. 또한 `retry_count` 필드명 충돌 가능성도 언급됨.

**7. 통합문서 내 섹션 경합("문서는 맞는데 다른 섹션이 1위") 해결**
- page 1 문제는 penalty로 해결됐지만, 같은 문서 내에서 다른 절차 섹션이 1위가 되는 잔여 미스들이 남아 있음. chapter 메타, reranker, 섹션 단위 재랭킹 등이 필요할 수 있음.

### 6-4. 검증 체크리스트 (다음 확인)

문서들에서 제안된 방향을 "검증 질문" 형태로 정리:

**Q1. 현재 코드에 Stage2가 존재하나?**
- 존재한다면: stage2 결과 병합에도 RRF가 적용되나?
- 없다면: 복원 후 page-hit@1 / first_page=1 / doc-hit 변화가 03-01 실험과 유사하게 재현되나?

**Q2. bilingual on 상태에서 page-hit@1 하락이 RRF로 해소되는가?**
- "EN+KO 단순 병합" vs "RRF 병합(혹은 KO-only stage2)" A/B로 page-hit@1과 doc-hit@10을 같이 봐야 함.

**Q3. SOP intent가 실제로 GCB contamination을 줄이는가?**
- soft/hard 모드별로 DOC_MISS와 "정답 문서가 SOP인데 GCB가 상위" 비율을 비교.
- 특히 strict_doc_type_override 버그를 먼저 고친 뒤 측정해야 함.

**Q4. MQ mode sweep이 가능한가?**
- 동일 질문셋/동일 인덱스/동일 env에서 `mq_mode=off|fallback|on`을 반복 실행해 doc/page + Jaccard@k + 비용/지연을 자동 리포트로 뽑을 수 있나?

### 6-5. 현재 상태 종합

```
문제 1 (비결정성)     ███████████████████████████████████████████ 해결됨
문제 2 (페이지 정확도) ████████████████████████░░░░░░░░░░░░░░░░░░ stage2 제거 → 복원 필요
문제 3 (영어 전용)     ████████████████████░░░░░░░░░░░░░░░░░░░░░ bilingual 적용, RRF로 부작용 해소 필요
문제 4 (GCB + RRF)    ████████████████████████████░░░░░░░░░░░░░ 구현 완료, 실환경 검증 필요
```

### 6-6. 전체 평가

**잘 된 점**: 문제를 체계적으로 진단하고 데이터로 검증함.
- 02-26 비결정성 발견 → 02-28 통합 진단 → 03-01 golden set 기준선 → 6가지 전략 비교 실험 → agent/run 실경로 검증 → 버그 발견 → 해결책 설계 → 코드 구현까지 논리적 흐름.
- 각 단계마다 실험 데이터(79건 SOP 질문셋)를 기반으로 의사결정.
- 검증 원칙(`/api/agent/run` 필수, ES 직접조회는 보조)이 실험 과정에서 자연스럽게 확립됨.

**아쉬운 점**: 각 해결책이 개별적으로는 효과가 있었지만, **전체를 조합한 최종 검증이 빠져 있음**.
- Stage2 retrieval이 제거된 상태에서 RRF만으로 page-hit@1이 어떻게 되는지 모름
- RRF + bilingual + SOP intent + stage2를 모두 켠 상태의 79건 평가가 없음
- 해결책들 간의 상호작용(bilingual이 doc-hit↑ + page-hit↓, RRF가 이를 보완, stage2가 page↑) 통합 검증 미수행

### 6-7. 최종 판정

| 항목 | 판정 | 근거 |
|------|------|------|
| 문제 정의의 정확성 | **정확** | 비결정성, page 정확도, bilingual 누락, doc_type 혼입 이슈가 모두 실데이터로 재현됨 |
| 해결 접근의 타당성 | **타당** | deterministic/mq_mode 분리, stage2+penalty, bilingual 보완, RRF, SOP intent, sticky가 원인과 직접 대응 |
| 실험 설계의 품질 | **양호** | 79건 질문셋, ES 시뮬레이션 + `/api/agent/run` 실경로 비교, before/after 지표 축적 |
| 개선 결과 | **부분 성공** | 개별 개선은 유의미(예: page-hit@1 개선, DOC_MISS 감소)하나 일부 trade-off/회귀 존재 |
| 최종 해결 상태 | **부분 해결** | 핵심 잔여 과제(stage2 복원, strict override 버그, MQ sweep 자동화, 통합 full eval) 미완료 |

### 6-8. 다음 단계

**즉시**: stage2 복원 + strict override 버그 수정 + boost factor 상향
**이후**: 모든 개선사항을 동시 적용한 상태에서 79건 full evaluation 실행 → 최종 before/after 확정
