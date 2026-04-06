# Task: Setup route 그룹 선택 및 judge supplement 범위 개선

Status: Draft
Owner: hskim
Branch or worktree: dev
Created: 2026-04-06

## Goal

setup route의 답변 품질과 속도를 개선한다:
1. 그룹 선택 시 refs가 부족한 그룹을 즉시 채택하지 않고 더 나은 그룹을 탐색
2. relevance 판정에서 빈 응답을 False로 처리
3. judge supplement가 선택된 그룹만 참조하도록 범위 제한
4. 위 변경사항을 추적할 수 있는 디버그 로그 보강

## Why

### 사례: APC Valve 교체 질문 (~400s 소요)

**질문:** "SUPRA V+ APC Valve 교체 작업 방법"

**검색 결과 그룹:**
```
group 0: "supra_series_pm_bottom_structure / APC Valve" → refs=1
group 1: "supra_vplus_apc_valve_eng / Flow Chart"       → refs=5
group 2: "supra_vplus_apc_valve_eng / Work Procedure"   → refs=3
group 3: "omnis_plus_apc_product_report / Work Proc"    → refs=2
```

**현재 동작:**
1. `answer_node`: group 0 relevance check → True → 즉시 선택 (refs=1)
2. refs=1(2545자)로 답변 생성 → 정보 부족 → format retry 2회
3. `judge_node`: supplement 모드 → **전체 검색 결과**(ref_json)에서 보충
4. 다른 기기(omnis_plus) 문서까지 참조하여 답변에 혼합
5. 총 소요 시간: ~400s

**기대 동작 (수정 후):**
1. `answer_node`: group 0 → relevant=True but refs=1 < 3 → fallback 저장, 계속 탐색
2. group 1 → relevant=True, refs=5 ≥ 3 → 선택
3. 충분한 컨텍스트로 1회에 답변 생성 성공
4. `judge_node`: supplement 시 선택된 그룹(group 1)만 참조
5. 총 소요 시간: ~30-50s (예상)

### 근본 원인 3가지

#### 원인 1: 그룹 선택 시 refs 수 미고려

`langgraph_agent.py` line ~3795-3797:
```python
if relevant:
    selected_refs = group_refs
    break  # ← refs=1이어도 즉시 선택
```

첫 번째 relevant=True 그룹을 무조건 선택한다. refs=1인 그룹이 선택되면:
- LLM에 전달되는 컨텍스트가 부족
- format validation 실패 → retry 2회 (각 ~50-100s)
- 그래도 부족하면 judge supplement으로 보충 (~100s 추가)

#### 원인 2: relevance 빈 응답 시 True 처리

`langgraph_agent.py` line ~1327-1331 (`_check_doc_relevance`):
```python
if not raw_stripped:
    logger.info("_check_doc_relevance: raw empty → default True")
    return True  # ← 판정 불가인데 관련있다고 간주
```

LLM이 빈 응답을 반환하면 "관련 있음"으로 처리한다.
- 실제로는 판정 불가 상태
- 부적합한 그룹이 통과될 수 있음
- 다음 그룹의 체크 기회가 사라짐

#### 원인 3: judge supplement가 전체 검색 결과 참조

setup route의 answer_node return (line ~3935, ~3994):
```python
return {
    "answer": answer,
    "reasoning": reasoning,
    "answer_format": ...,
    # answer_ref_json 없음!
}
```

issue route는 `"answer_ref_json": answer_refs`를 명시적으로 return하지만,
setup route는 `answer_ref_json`을 return하지 않는다.

judge_node (line ~4266):
```python
ref_items = state.get("answer_ref_json") or state.get("ref_json", [])
```

`answer_ref_json`이 없으므로 `ref_json`(전체 검색 결과)으로 fallback →
다른 그룹, 심지어 다른 기기의 문서까지 supplement 범위에 포함됨.

## 수정 내용

### 수정 1: Approach A — refs 부족 시 계속 탐색

**파일:** `langgraph_agent.py` line ~3778-3836

```python
# 수정 전
if relevant:
    selected_refs = group_refs
    break

# 수정 후
MIN_REFS_FOR_ACCEPT = 3

if relevant:
    if len(group_refs) >= MIN_REFS_FOR_ACCEPT:
        selected_refs = group_refs
        break
    elif fallback_refs is None:
        fallback_refs = group_refs  # 후보로 저장, 계속 탐색

# 최종 선택 우선순위:
# 1순위: selected_refs (refs >= 3인 relevant 그룹)
# 2순위: fallback_refs (refs < 3이지만 relevant)
# 3순위: doc_groups[0] (전부 부적합)
```

**비용:** relevance check LLM 호출 1~4회 추가 (각 ~1-2s)
**절약:** format retry + judge supplement 회피 시 ~200-350s 절약

### 수정 2: 빈 응답 시 False 처리

**파일:** `langgraph_agent.py` line ~1327-1331

```python
# 수정 전
if not raw_stripped:
    return True  # 판정 불가 → 관련있다고 간주

# 수정 후
if not raw_stripped:
    return False  # 판정 불가 → skip하고 다음 그룹 체크
```

### 수정 3: setup route return에 answer_ref_json 추가

**파일:** `langgraph_agent.py` line ~3935, ~3994 (두 return 경로)

```python
# 수정 전
return {
    "answer": answer,
    "reasoning": reasoning,
    "answer_format": ...,
}

# 수정 후
return {
    "answer": answer,
    "reasoning": reasoning,
    "answer_format": ...,
    "answer_ref_json": ref_items,  # 선택된 그룹만 전달
}
```

**효과:** judge supplement가 선택된 그룹 범위 내에서만 보충 →
다른 기기/다른 문서의 내용이 답변에 혼입되는 것을 방지

### 수정 4: 디버그 로그 보강

**파일:** `langgraph_agent.py` (answer_node `_events`)

- 그룹별 refs 수 표시
- fallback/direct 선택 여부 표시
- `_check_doc_relevance` 빈 응답 발생 시 이벤트 추가

## Contracts To Preserve

- (해당 문서 미존재 시 N/A)

## Contracts To Update

- None

## Allowed Files

- `backend/llm_infrastructure/llm/langgraph_agent.py`

## Out Of Scope

- ES 검색 알고리즘 변경
- reranker 튜닝
- expand_related_docs_node 로직 변경
- judge_node의 supplement 프롬프트 변경
- 프론트엔드 변경

## Risks

- `MIN_REFS_FOR_ACCEPT=3` 임계값이 모든 쿼리에 적합하지 않을 수 있음
  → 검증 후 조정 가능, AgentSettings로 추출 고려
- 빈 응답 False 전환 시, 모든 그룹이 빈 응답이면 전부 skip → 첫 번째 그룹 fallback 동작
  → 기존과 동일한 fallback 경로이므로 안전
- `answer_ref_json` 추가 시, supplement 범위가 좁아져 보충 내용이 줄어들 수 있음
  → 의도된 동작 (선택된 그룹 내에서만 보충)

## 코드 리뷰 결과 (2026-04-06)

Critic + Architect 에이전트 병렬 리뷰 수행. 코드 참조(line 번호, 변수명, 흐름) 모두 정확 확인됨.

### 발견된 문제점

#### [높음] 수정 1 + 수정 3 상호작용: fallback 시 supplement 효과 감소

fallback 그룹(refs < 3)이 선택된 상태에서 `answer_ref_json`을 좁히면,
judge supplement도 1-2개 ref만 참조 → 보충 효과가 거의 없어짐.

**대응:** fallback 선택 시에는 `answer_ref_json`을 설정하지 않아 judge가 전체
`ref_json`을 참조하도록 분기. 정상 선택(refs >= MIN_REFS_FOR_ACCEPT) 시에만 좁은 범위 적용.

```python
# return 시 분기
if is_fallback_selection:
    # answer_ref_json 미설정 → judge가 ref_json(전체) 참조
    pass
else:
    result["answer_ref_json"] = ref_items  # 선택된 그룹만
```

#### [높음] 수정 2: LLM 서버 장애 시 불필요한 대기

빈 응답 → False 시, LLM이 전부 빈 응답이면 MAX_SETUP_DOC_TRIES=5회 호출을
모두 기다린 후 결국 첫 번째 그룹 fallback. 현재(빈 응답 → True → 즉시 선택)보다
최악의 경우 더 느려질 수 있음.

**대응:** 그룹 선택 루프에 연속 빈 응답 카운터 추가. 2-3회 연속 빈 응답 시 early-exit.

```python
consecutive_empty = 0
for i, (group_key, group_refs) in enumerate(doc_groups[:MAX_SETUP_DOC_TRIES]):
    relevant = _check_doc_relevance(...)
    if relevant is None:  # 빈 응답 구분
        consecutive_empty += 1
        if consecutive_empty >= 2:
            logger.warning("answer_node: %d consecutive empty responses, breaking", consecutive_empty)
            break
    else:
        consecutive_empty = 0
    ...
```

※ 이를 위해 `_check_doc_relevance`가 빈 응답 시 `None`을 반환하도록 변경하거나,
별도 플래그를 두는 방안 검토 필요.

#### [중간] `doc_groups` 변수 스코프 버그 (기존)

line 3824의 `if route == "setup" and doc_groups:`에서, `ref_items`가 빈 경우
`doc_groups`가 정의되지 않아 `NameError` 가능.

**대응:** line 3755 이전에 `doc_groups: list = []` 초기화 추가.

#### [중간] `display_docs`와 `answer_ref_json` 불일치

프론트엔드 `display_docs`에는 모든 그룹 문서가 표시되지만, 답변은 선택된 그룹만 참조.
인용 번호(`[1]`, `[2]`...)가 display_docs 순서와 일치하지 않을 수 있음.

→ 프론트엔드 변경은 Out of Scope이므로 후속 과제로 기록.

#### [낮음] `_check_doc_relevance` docstring-코드 불일치

현재 docstring (line 1311-1314): "빈 응답 시 1회 재시도하고, 여전히 빈 응답이면 False로 처리"
현재 코드 body (line 1327-1331): 재시도 없이 즉시 `return True`

**대응:** 수정 2 적용 시 docstring도 "빈 응답 시 False 반환"으로 정리.
또는 docstring에 맞게 1회 ���시도 로직을 실제 구현.

#### [낮음] 수정 1 구현 코드 구체성 부족

문서에서 pseudo-code 수준만 제시. `fallback_refs` 초기화 위치, 루프 후 3단계 분기의
실제 적용 코드를 보강하면 구현 시 판단 여지가 줄어듦.

### 수정 간 의존성

| 조합 | 관계 | 비고 |
|------|------|------|
| 수정 1 → 수정 3 | 의존 | ref_items에 올바른 그룹이 담겨야 answer_ref_json이 의미 있음 |
| 수정 1 + 수정 2 | 시너지 | 빈 응답 skip + refs 부족 탐색이 결합되어 더 나은 그룹 발견 확률 증가 |
| 수정 1 + 수정 2 | 전멸 리스크 | LLM 장애 시 모든 그룹 False + fallback 미저장 → doc_groups[0] fallback (안전하나 느림) |
| 수정 4 | 독립 | 다른 수정과 무관, 순서 무관 |

### 검증 계획 보완 권고

- `time curl ...`로 소요 시간 정량 비교 추가
- 수정 2 빈 응답은 간헐적이므로 로그 모니터링으로 충분 (unit test 이상적이나 현 인프라 제약)
- judge supplement 범위 확인을 위해 answer_ref_json 설정 여부를 debug events에 포함

## Verification Plan

```bash
# 1. APC Valve 쿼리로 그룹 선택 동작 확인
curl -s -X POST http://localhost:8011/api/agent/run \
  -H "Content-Type: application/json" \
  -d '{"message": "SUPRA V+ APC Valve 교체 작업 방법"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'answer_chars={len(d.get(\"answer\",\"\"))}')
print(f'answer: {d.get(\"answer\",\"\")[:300]}')
"

# 2. 로그에서 그룹 선택 과정 확인
docker logs rag-api-dev --since=120s 2>&1 | grep -E "relevance check|selected group|fallback|supplement"
# 기대: refs >= 3인 그룹이 선택됨, 또는 fallback 사용 표시

# 3. judge supplement 범위 확인
docker logs rag-api-dev --since=120s 2>&1 | grep "judge_node"
# 기대: supplement가 선택된 그룹 범위 내에서만 동작

# 4. retrieval test 전체 실행
cd backend && python scripts/run_eval_matrix.py --test retrieval_test_bjw
```

## Verification Results

### 정적 검증 (2026-04-06)

- **Python syntax**: `ast.parse` 통과
- **ruff check**: 변경 라인 범위에 새 에러 0건 (기존 pre-existing I001/F401/E501만 존재)
- **Architect 검증**: 6개 US 전체 PASS

### Architect 엣지 케이스 검증

| 시나리오 | 결과 | 비고 |
|----------|------|------|
| `doc_groups` 비어있음 (ref_items=[]) | 안전 | `if route == "setup" and ref_items:` 가드로 스킵 |
| 전 그룹 None (LLM 장애) | 안전 | consecutive_empty=2에서 early-exit → doc_groups[0] fallback |
| 단일 그룹 | 안전 | relevance check 스킵, is_fallback_selection=False 유지 |
| non-setup route | 안전 | doc_groups=[], is_fallback_selection=False 초기화 |
| `is_fallback_selection` 미초기화 | 불가 | line 3758에서 무조건 초기화 |
| issue route answer_ref_json | 영향 없음 | line 3727에서 독립적으로 설정 |

### 런타임 검증 (대기)

- API 재시작 후 APC Valve 쿼리 테스트 필요
- retrieval test 전체 실행하여 regression 확인 필요

## Handoff

- Current status: implemented (코드 수정 완료, 정적 검증 통과, 런타임 검증 대기)
- Remaining TODOs:
  1. API 재시작 후 APC valve 쿼리 테스트
  2. retrieval test 전체 실행하여 regression 확인

## Change Log

- 2026-04-06: task 생성 — 사례 분석, 근본 원인 3가지 식별, 수정 방향 설계
- 2026-04-06: Critic + Architect 리뷰 반영 — 6개 문제점 식별, 수정 간 의존성·상호작용 분석 추가
- 2026-04-06: 수정 1~4 구현 완료 + Architect 검증 PASS

## Final Check

- [x] Diff stayed inside allowed files
- [x] Protected contract IDs were re-checked
- [x] Verification commands were run, or blockers were recorded
- [ ] Any contract changes were reflected in `product-contract.md`
- [x] Remaining risks and follow-ups were documented
