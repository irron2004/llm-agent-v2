# Task: SOP penalty gate fix — route=setup 시 sop_only_predicate 활성화

Status: In-progress
Owner: hskim
Branch or worktree: dev
Created: 2026-04-06

## Goal

retrieve_node의 penalty 로직(early_page_penalty, scope_penalty, procedure_boost)이 route=setup일 때도 정상 작동하도록 `sop_only_predicate` 게이트 조건을 수정한다.

## Why

### 증상
- "ZEDIUS XP 설비의 EFEM PIO SENSOR BOARD 교체 방법" 같은 SOP 절차 질문에서 page 1(개요/목차)이 top-1으로 검색됨
- expand_related_docs_node가 page 1 기준으로 확장 → Work Procedure(page 6-14) 미포함
- LLM이 개요만 보고 "절차가 명시되지 않았습니다" 응답

### 근본 원인 분석

**penalty 로직은 코드에 존재하고 기본 활성화 상태이나, 게이트 조건이 충족되지 않아 작동하지 않았음.**

`sop_only_predicate` (langgraph_agent.py:2419) 결정 조건:

| 조건 | 상태 | 이유 |
|------|------|------|
| `state.get("sop_intent") is True` | 항상 False | `sop_intent`를 설정하는 코드가 전체 codebase에 없음 |
| `selected_doc_types ∩ sop_variants` | False (API 직접호출 시) | auto_parse_confirm_node의 interrupt가 스킵되어 task_mode 미선택 → selected_doc_types=[] |
| `route == "setup"` | **체크하지 않음** (수정 전) | route_node가 setup으로 정확히 판단했지만 sop_only_predicate 계산에 미반영 |

**결과:** `sop_only_predicate=False` → early_page_penalty, scope_penalty, procedure_boost 전부 스킵

### 로그 증거 (수정 전)

```
auto_parse_node: detected_doc_types=[]
route OUTPUT: selected_doc_types=[], doc_type_selection_skipped=True
retrieve_node: intent gating — sop_pred=False, procedural=True, inquiry=False, route=setup
```

- `sop_pred=False`: 모든 penalty 비활성
- `procedural=True`: 쿼리에 "교체", "방법" 키워드 있음 (정상)
- `route=setup`: route 판단은 정확 (정상)

## 수정 내용

### 파일: `backend/llm_infrastructure/llm/langgraph_agent.py`

**변경 위치:** line 2419-2422

```python
# 수정 전
sop_only_predicate = bool(
    state.get("sop_intent") is True
    or bool(selected_doc_types_normalized.intersection(sop_variants))
)

# 수정 후
sop_only_predicate = bool(
    state.get("sop_intent") is True
    or route == "setup"
    or bool(selected_doc_types_normalized.intersection(sop_variants))
)
```

**효과:** route가 `setup`이면 `sop_only_predicate=True` → 다음 penalty 로직 활성화:

| Penalty | 조건 | 효과 |
|---------|------|------|
| early_page_penalty | page ≤ 2 | score × 0.3 |
| scope_penalty | scope/TOC markers 또는 page ≤ 1 | score × 0.25 |
| procedure_boost | section_chapter에 "work procedure" 등 | score × 1.4 |
| sop_soft_boost | doc_type이 SOP | score × 1.3 |

## 관련 배경 조사 (동일 세션에서 진행)

### 1. 약어 확장 env var prefix 오류
- `RAG_ABBREVIATION_EXPAND_ENABLED=false`가 무시됨 → `AgentSettings`의 prefix는 `AGENT_`
- **수정:** `.env`에서 `AGENT_ABBREVIATION_EXPAND_ENABLED=false`로 변경

### 2. reranker 테스트
- rerank ON 시 cross-encoder가 완전히 다른 문서(FFU)를 top으로 밀어올림
- PIO SENSOR BOARD 문서가 완전 배제됨 → **rerank OFF 유지**

### 3. expand_related_docs_node 구조 확인
- `_combine_related_text`가 같은 챕터의 청크를 하나로 합침 (의도대로 동작)
- 문제는 확장 로직이 아니라 **입력 단계(검색 결과)에서 이미 잘못된 페이지가 top-1**인 것

## Contracts To Preserve

- (해당 문서 미존재 시 N/A)

## Contracts To Update

- None

## Allowed Files

- `backend/llm_infrastructure/llm/langgraph_agent.py`
- `frontend/src/features/chat/components/workflow-trace.tsx`

## Out Of Scope

- ES 검색 알고리즘 변경
- reranker 튜닝
- expand_related_docs_node 로직 변경
- 프론트엔드 변경

## Risks

- route가 setup이 아닌 다른 경로(general, issue 등)에서의 SOP 검색에는 영향 없음
- setup route의 비-SOP 문서(PEMS, 매뉴얼 등)에도 penalty가 적용될 수 있으나, penalty 대상은 page ≤ 2 / scope markers이므로 정상 문서에는 영향 미미

## Verification Plan

```bash
# 1. API 재시작 후 테스트 쿼리 실행
curl -s -X POST http://localhost:8011/api/agent/run \
  -H "Content-Type: application/json" \
  -d '{"message": "ZEDIUS XP 설비의 EFEM PIO SENSOR BOARD 교체 방법"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
docs=d.get('retrieved_docs',[])
print(f'docs={len(docs)}')
for i,r in enumerate(docs):
    m=r.get('metadata',{})
    print(f'  [{i}] page={m.get(\"page\",\"?\")} ch=\"{m.get(\"section_chapter\",\"\")}\"')
print(f'answer: {d.get(\"answer\",\"\")[:200]}')
"

# 2. 로그에서 sop_pred=True 확인
docker logs rag-api-dev --since=60s 2>&1 | grep "intent gating"
# 기대: sop_pred=True, procedural=True

# 3. 로그에서 penalty 적용 확인
docker logs rag-api-dev --since=60s 2>&1 | grep "procedure_boost.*scope_penalty"
# 기대: scope_penalty > 0
```

## Verification Results

- command: `curl -s -X POST http://localhost:8011/api/agent/run -d '{"message":"ZEDIUS XP 설비의 EFEM PIO SENSOR BOARD 교체 방법"}'`
  - result: pass
  - note: sop_pred=True, scope_penalty 72개 적용, top-1이 page 10(Flow Chart)으로 변경
  - evidence: `retrieve_node: intent gating — sop_pred=True, procedural=True, inquiry=False, route=setup`
  - evidence: `retrieve_node: procedure_boost=3 scope_penalty=72`
  - evidence: `answer_node: selected global_sop_supra_xp_all_efem_pio_sensor_board pages=[11,12,13]`
  - evidence: answer 893자 — "EFEM PIO SENSOR BOARD 교체 ## 작업 절차 ..." (이전: "절차가 명시되지 않았습니다")

## Handoff

- Current status: in-progress (코드 수정 완료, 검증 대기)
- Remaining TODOs:
  1. API 재시작 후 sop_pred=True 확인
  2. penalty 적용 후 검색 품질 개선 확인
  3. retrieval test 전체 재실행하여 정확도 비교

## Change Log

- 2026-04-06: task 생성, sop_only_predicate에 route=="setup" 조건 추가
- 2026-04-06: 약어 확장 env var prefix 수정 (RAG_ → AGENT_)
- 2026-04-06: rerank ON 테스트 → 악화 확인 → OFF 유지

## Final Check

- [x] Diff stayed inside allowed files
- [ ] Protected contract IDs were re-checked
- [ ] Verification commands were run, or blockers were recorded
- [ ] Any contract changes were reflected in `product-contract.md`
- [ ] Remaining risks and follow-ups were documented
