# Task: Setup route relevance check — 하위 컴포넌트 쿼리 시 전 그룹 거부 문제

Status: In-Progress (F, B, D 구현 완료 — 검증 대기)
Owner: hskim
Branch or worktree: feat/retrieval-logic-개선
Created: 2026-04-07

## Goal

setup route에서 하위 컴포넌트/부품명으로 질문할 때 relevance check가 전 그룹을 거부하여 "찾지 못했습니다" 답변이 반복되는 문제를 해결한다.

부차적으로, 장비명 오타("SUPRAvvplus" 등)가 쿼리 텍스트에 그대로 남아 그룹 정렬 정확도가 떨어지는 문제도 함께 개선한다.

## Why

### 현재 코드 기준 확정 사실 / 가설 구분

#### 확정 사실

1. **retrieval은 완전 실패가 아니다.**
   - APC 관련 단서를 포함한 PDB 문서가 후보군에 실제로 들어온다.
2. **실패 지점은 answer node의 group selection 경로다.**
   - setup route에서 `(doc_id, section_chapter)` 그룹화 → relevance check → fallback 순으로 진행된다.
3. **judge는 APC 관련 단서가 전체 REFS에 존재함을 인식한다.**
   - 즉, 검색 0건 문제가 아니라 “잘못된 그룹 선택” 문제다.
4. **fallback selection은 `answer_ref_json`을 남기지 않는다.**
   - 이 때문에 setup route 후속 보강 품질이 약해진다.

#### 아직 가설인 부분

1. `_check_doc_relevance` 프롬프트만 바꾸면 케이스가 충분히 해결될 것이다.
2. 장비명 오타 정규화만으로 APC 케이스가 안정적으로 해결될 것이다.
3. not-found supplement 허용이 부작용 없이 안전망이 될 것이다.

즉, **확정 사실은 group selection failure**이고,
아래 수정안(A/B/C/D)은 그 실패를 푸는 해결 가설이다.

### 재현 사례

**질문:** `"SUPRAvvplus APC 교체 방법"` (의도: supra vplus의 APC 교체)

**실제 동작 (3회 반복 실패, 총 ~170s 소요):**

```
abbreviation_resolve (2ms) → NO-OP (device명 미처리)
route (1.5s) → setup
retrieve (82ms) → 26개 문서 검색 ✅
  top5: global_sop_supra_vplus_adj_all_power_turn_on_off (page 9-13)
  하위: global_sop_supra_vplus_rep_sub_unit_pdb_en (APC 내용 포함)
expand_related (25ms) → 10개 문서 확장

answer 노드 — 그룹 선택:
  Group1: all_power_turn_on_off::10. Work Procedure (7 refs)
  Group2: rep_sub_unit_pdb_en::10. Work Procedure (3 refs)

  ┌─ relevance check(Group1) → "no" (power on/off ≠ APC 교체)
  ├─ relevance check(Group2) → "no" (PDB 교체 ≠ APC 교체) ← 핵심 실패 지점
  └─ 전부 실패 → fallback to Group1 (is_fallback_selection=True)

  answer LLM이 Group1의 REFS(power on/off)를 받음
  → APC 교체 정보 없음 → "관련 절차 문서를 찾지 못했습니다" (29자)

judge: "불충실" (전체 ref_json에서 APC 발견)
  → "찾지 못했습니다" 포함 → supplement 스킵
  → retry_expand → 같은 실패 반복
  → retry_bump + refine_queries → 같은 실패 반복
  → done (max_attempts 도달)
```

**기대 동작:**
1. "SUPRAvvplus" → "supra vplus"로 쿼리 정규화
2. PDB 그룹이 APC를 sub-step으로 포함한다고 relevance check 통과
3. PDB 그룹의 REFS로 APC 교체 절차 답변 생성

### 근본 원인 3가지

#### RC-1: `_check_doc_relevance`가 하위 절차(sub-step) 매칭에 약함 — ⚠️ 보류

> **보류 사유:** relevance check가 PDB ≠ APC로 판정한 것은 **정확한 동작**이다.
> APC 교체를 물어봤는데 PDB 문서로 답하면 그것도 오답이다.
> 이 케이스에서는 "APC 교체" 전용 SOP가 인덱스에 존재하지 않는 것이 근본 문제이며,
> relevance check를 느슨하게 만드는 것은 무관한 문서가 통과하는 부작용을 유발할 수 있다.
>
> RC-1에 대한 수정(수정 A)은 별도 검증 후 재판단한다.

`langgraph_agent.py` line 1307-1336

PDB 교체 문서 안에 APC 토글(SW15)이 sub-step으로 존재하지만, relevance check LLM이
"APC 교체"와 "PDB 교체"를 서로 다른 주제로 판정하여 `False` 반환.

현재 프롬프트:
```
"Compare the SPECIFIC subject in the question (model number, part name,
procedure type) against what the document actually describes."
```

이 판정은 **프롬프트 지시에 충실한 결과**다. 문서의 주제(PDB 교체)와 질문의 주제(APC 교체)는
실제로 다르며, PDB 문서 안의 APC 토글은 "APC 교체 방법"에 대한 답이 아니라
PDB 교체 절차의 한 단계일 뿐이다.

**핵심 쟁점:** sub-step 매칭을 허용하면 이 케이스는 통과하지만,
"APC 교체"를 물어봤는데 "PDB 교체" 절차를 답변으로 제공하는 것이 올바른 동작인지는 별도 판단 필요.

#### RC-2 (High): 전 그룹 relevance 실패 시 재검색 경로가 없음

**두 가지 하위 문제가 결합되어 있다:**

**RC-2a: answer 노드에 "관련 그룹 없음 → 재검색" early-exit 경로가 없음**

현재 LangGraph 엣지 구조:
```
answer → judge → should_retry → retry_expand / retry_bump
```

answer 노드가 "전 그룹 relevance 실패"를 감지해도, 재검색을 직접 요청할 수 없다.
무조건 답변 생성 → judge 판정을 거쳐야만 retry로 갈 수 있다.

현재 흐름 (전 그룹 실패 시):
```
answer: 전 그룹 "no" → 어쨌든 첫 번째 그룹으로 답변 생성 (~30s) → "찾지 못했습니다"
judge: "불충실" 판정 (~20-50s) ← 당연한 결과를 판정하느라 시간 소모
retry_expand: expand 더 가져옴 → answer 노드 재진입 → 같은 실패 반복 (~50s)
retry_bump: 쿼리 재구성 → 재검색 → answer 노드 재진입 → 같은 실패 반복 (~80s)
총: ~170s, 전부 낭비
```

바람직한 흐름:
```
answer: 전 그룹 "no" → "관련 그룹 없음" 시그널 → 답변 생성 스킵 → 바로 재검색
```

**RC-2b: fallback 그룹 선택 기준이 쿼리 관련성을 반영 못함**

`langgraph_agent.py` line 3840-3847

전 그룹 relevance check 실패 시 첫 번째 그룹(`power_turn_on_off`)으로 fallback.
이 그룹은 APC와 무관하므로 answer LLM이 정직하게 "찾지 못했습니다" 생성.

그룹 정렬(`_group_query_score`, line 3767-3771)이 doc_id 문자열에서만 토큰 매칭:
```python
score = sum(2 for tok in _q_tokens if tok in gkey_lower)
```
`"apc"` 토큰이 두 그룹의 `doc_id`/group key에 직접 나타나지 않으면 동점이 되고,
그 경우 기존 검색 순서(retrieve scoring)의 영향을 그대로 받는다.
본문에 `APC`가 있어도 group ordering에는 반영되지 않는다.

#### RC-3 (High): fallback answer가 setup 보강/복구 경로를 약화시킴

`langgraph_agent.py` line 4317

```python
if route == "setup" and answer and "찾지 못했습니다" not in answer:
    supplemented_answer, judge = _supplement_setup_answer(...)
```

`"찾지 못했습니다"`가 포함되면 supplement를 스킵한다. 이 때문에 judge가 전체 REFS에서 APC 단서를 찾더라도,
선택 그룹 실패를 뒤집는 후속 경로가 약해진다. 다만 이것은 **1차 원인이라기보다 실패를 고착화하는 증폭 요인**이다.

### 부차 원인: 쿼리 텍스트 미정규화

`abbreviation_expander.py`는 반도체 공정 용어(`semicon_word.json`)만 처리.
장비명 추출은 retrieve 내부에서 `device_aliases.json` + fuzzy matching으로 별도 수행.

결과적으로 `query_for_prompt`에 "SUPRAvvplus"가 그대로 남아:
- 그룹 정렬 시 토큰 "supravvplus"가 doc_id의 "supra_vplus"와 매칭 안 됨
- relevance check 프롬프트에 오타가 그대로 전달되어 LLM 혼란 가능

## 수정 방향

### 우선순위 재정렬 근거

이 케이스의 핵심은 "검색 실패"가 아니라 **group selection failure + 실패 시 비효율적 재시도**다.

1. **E: 전 그룹 실패 시 answer 노드 early-exit → 바로 재검색**
   - 가장 직접적. 낭비되는 ~170s를 구조적으로 제거
2. **B: 그룹 정렬에 본문 단서 반영**
   - fallback이 발생하더라도 올바른 그룹이 선택되게 함
3. **C: fallback/not-found 보강 경로를 안전망으로 허용**
   - E/B 실패 시 복구 경로 확보
4. **D: 장비명 오타 정규화**
   - 전체 정렬/프롬프트 품질을 보조적으로 개선
5. **A: relevance check sub-step 매칭 완화** — ⚠️ 보류
   - RC-1 보류와 동일 사유

### 수정 E: 전 그룹 relevance 실패 시 answer early-exit → 바로 재검색

**파일:** `langgraph_agent.py` answer_node (line ~3840-3847) + LangGraph 엣지 (`langgraph_rag_agent.py`)

현재 전 그룹 실패 시:
```python
# line 3840-3847
else:
    ref_items = doc_groups[0][1]          # 첫 번째 그룹으로 fallback
    is_fallback_selection = True
    # → 이후 답변 생성 (~30s) → judge (~20-50s) → retry
```

개선 — answer 노드에서 early-exit:
```python
# answer_node 내부 — 전 그룹 실패 감지 시
if selected_refs is None and fallback_refs is None:
    logger.info("answer_node: no relevant group found, signaling re-retrieve")
    return {
        "answer": "",
        "answer_skip_reason": "no_relevant_group",
        "_events": [*_answer_events, "[answer] early-exit: no relevant group"],
    }
```

LangGraph 엣지 — answer 이후 조건부 분기:
```python
# answer → judge 사이에 라우터 추가
def after_answer(state):
    if state.get("answer_skip_reason") == "no_relevant_group":
        return "retry_bump"   # judge 스킵, 바로 쿼리 재구성 + 재검색
    return "judge"

builder.add_conditional_edges("answer", after_answer, {
    "retry_bump": "retry_bump",
    "judge": "judge",
})
```

**비용:** LangGraph 엣지 1개 추가, answer_node에 early-exit 분기 5줄.
**효과:**
- 불필요한 답변 생성 (~30s) + judge 판정 (~20-50s) 스킵
- 바로 쿼리 재구성 → 재검색으로 진입
- 전 그룹 실패가 반복되면 max_attempts에서 종료 (무한 루프 방지)

**주의:**
- `retry_bump`는 `attempts`를 증가시키므로 max_attempts 도달 시 자연 종료
- early-exit 시 `answer`가 빈 문자열이므로, 최종 응답 경로에서 빈 답변 처리 필요
- 기존 `retry_expand` (expand 확장만) vs `retry_bump` (쿼리 재구성 + 재검색) 중
  전 그룹 실패는 문서 부족이 아니라 쿼리 문제이므로 `retry_bump`가 적합

### 수정 A: `_check_doc_relevance` 프롬프트 — sub-step 매칭 허용 — ⚠️ 보류

> **보류 사유 (RC-1과 동일):** relevance check가 PDB ≠ APC로 판정한 것은 정확한 동작이다.
> APC 교체를 물어봤는데 PDB 교체 절차를 답변으로 제공하면 그것도 오답이다.
> sub-step 매칭을 허용하면 무관한 문서가 통과하는 부작용이 있을 수 있으므로,
> 별도 케이스 수집 후 "어느 수준까지 허용할지" 기준을 정한 뒤 재판단한다.

**파일:** `langgraph_agent.py` `_check_doc_relevance()` (line 1320-1326)

개선안 (참고용, 미적용):
```python
system = (
    "You are a relevance checker. Determine if the document contains "
    "procedure/work steps that can answer the user's question.\n"
    "Compare the SPECIFIC subject in the question (model number, part name, "
    "procedure type) against what the document actually describes.\n"
    "A document is relevant if it directly describes the queried procedure "
    "OR contains steps for the queried component as a SUB-STEP of a larger procedure.\n"
    "Reply ONLY 'yes' or 'no'."
)
```

**열린 질문:**
- sub-step 매칭을 허용하면, "APC 교체"를 물어봤을 때 "PDB 교체 (APC 토글 포함)" 문서가
  통과하는 게 바람직한 동작인가? → 전용 SOP가 없는 경우에만 허용? 항상 허용?
- 어디까지가 "관련 있는 sub-step"이고 어디부터가 "주제가 다른 문서"인가?

### 수정 B: `_group_query_score` — 본문 키워드 매칭 추가

**파일:** `langgraph_agent.py` `_group_query_score()` (line 3767-3771)

현재:
```python
def _group_query_score(item):
    gkey, grefs = item
    gkey_lower = gkey.lower()
    score = sum(2 for tok in _q_tokens if tok in gkey_lower)
    return (-score, 0)
```

개선:
```python
def _group_query_score(item):
    gkey, grefs = item
    gkey_lower = gkey.lower()
    score = sum(2 for tok in _q_tokens if tok in gkey_lower)
    # 본문 키워드 매칭 (상위 3개 ref의 content 샘플)
    content_sample = " ".join(
        str(r.get("content", ""))[:500].lower() for r in grefs[:3]
    )
    score += sum(1 for tok in _q_tokens if tok in content_sample)
    return (-score, 0)
```

**비용:** 추가 LLM 호출 없음. 문자열 연산만 증가 (무시 가능).
**효과:** `"APC"`가 본문에 있는 PDB 그룹이 `power_turn_on_off`보다 먼저 체크될 가능성이 높아진다.

**왜 우선순위 1인가:** 현재 관측 실패는 “정답 그룹을 못 찾음” 이전에
“잘못된 그룹이 먼저 relevance check되고, 전부 실패하면 첫 그룹으로 fallback 되는 구조”이기 때문이다.

### 수정 C: "찾지 못했습니다" 답변에도 supplement 허용

**파일:** `langgraph_agent.py` `judge_node()` (line 4317)

현재:
```python
if route == "setup" and answer and "찾지 못했습니다" not in answer:
    supplemented_answer, judge = _supplement_setup_answer(...)
```

개선:
```python
if route == "setup" and answer:
    # ref가 있으면 "찾지 못했습니다" 답변도 보완 시도
    if "찾지 못했습니다" not in answer or bool(ref_items):
        supplemented_answer, judge = _supplement_setup_answer(...)
```

**비용:** fallback "not found" 시 supplement LLM 호출 1회 추가 (~10-20s).
**효과:** judge가 전체 ref_json에서 APC 관련 내용을 찾아 보완 답변 생성 가능.
수정 A/B가 완전히 작동하면 이 경로에 도달하지 않으므로 안전망 역할.

**주의:** 이 변경은 원인을 해결한다기보다 fallback 실패를 덜 치명적으로 만든다.

### 수정 D: abbreviation_resolve에서 device name 정규화

**파일:** `abbreviation_expander.py` 또는 `langgraph_rag_agent.py` abbreviation_resolve 노드

retrieve 내부의 device name 추출 결과(fuzzy matching 완료)를 `query_for_prompt`에
반영하는 방안. 두 가지 접근:

**접근 1:** abbreviation_resolve 노드에서 device_aliases.json 로드하여 쿼리 텍스트 정규화
```
"SUPRAvvplus APC 교체 방법" → "supra vplus APC 교체 방법"
```

**접근 2:** retrieve 노드의 device name 추출 결과를 state에 저장 → answer 노드에서 query_for_prompt 재구성
```python
# retrieve에서
state["detected_device_name"] = "supra vplus"
# answer에서
query_for_prompt = query.replace("SUPRAvvplus", state["detected_device_name"])
```

접근 2가 기존 fuzzy matching 로직을 재활용하므로 중복 구현 없이 깔끔하다.
**→ 접근 2로 구현 완료** (`_normalize_device_in_query` 헬퍼, answer_node에서 호출).

**접근 1 향후 가능성:** 현재 접근 2는 answer 단계 이후만 혜택을 받는다. 만약 검색 쿼리
자체의 품질이 device name 오타로 인해 저하되는 케이스가 추가로 발견되면, 접근 1
(abbreviation 단계에서 device_aliases.json 로드 + 쿼리 정규화)을 도입하여 retrieve
단계의 BM25/dense 검색까지 커버하는 것을 검토할 수 있다. 다만 현재는 수정 F(doc_id
keyword boost)가 retrieve 단계를 보완하므로 접근 1의 필요성은 낮다.

**주의:** 이 케이스의 직접 원인은 device filter 실패가 아니라 group selection failure다.
따라서 D는 독립 해결책이라기보다 F/B 성능을 끌어올리는 보조 수단으로 보는 편이 맞다.

**비용:** state 필드 추가 불필요 (기존 parsed_query.selected_devices 재사용).
**효과:** 그룹 정렬, relevance check, answer 프롬프트 모두에서 정규화된 장비명 사용.

### 수정 F: retrieval 단계에서 doc_id/section_chapter 키워드 부스트 (신규)

> **배경:** 현재 BM25 검색은 `search_text^1.0`, `chunk_summary^0.7`, `chunk_keywords^0.8`만
> 점수 필드로 사용한다. `doc_id`와 `section_chapter`는 **필터로만** 사용되며 점수에 반영되지 않는다.
>
> 그런데 SOP/setup/TS 문서의 doc_id는 문서 주제를 직접 인코딩한다:
> ```
> supra_vplus_apc_valve_eng         → APC Valve 문서
> supra_vplus_adj_all_power_turn_on_off → 전원 ON/OFF 문서
> supra_vplus_rep_sub_unit_pdb_en   → PDB 교체 문서
> ```
> doc_id에 쿼리 키워드가 있다는 건 "이 문서가 바로 그 주제"라는 **가장 강한 신호**인데,
> 현재 검색 점수에 전혀 반영되지 않고 있다.

**파일:** `backend/llm_infrastructure/retrieval/engines/es_search.py` + 어댑터/서비스 계층

**구현 방안:** 기존 `device_boost` should-clause 패턴을 활용

현재 device_boost 구현 (es_search.py line 631-662):
```python
# device_name에 대해 should clause로 점수 부스트
"should": [
    {"term": {"device_name": {"value": "supra vplus", "boost": 2.0}}}
]
```

동일 패턴으로 doc_id 키워드 부스트 추가:
```python
# 쿼리에서 핵심 토큰 추출 (device_name, 불용어 제외)
# 예: "SUPRAvvplus APC 교체 방법" → ["apc"]
#
# doc_id는 keyword 타입이므로 wildcard 사용:
"should": [
    {"wildcard": {"doc_id": {"value": "*apc*", "boost": 3.0}}}
]
```

**적용 조건:** route=setup 또는 sop_pred=True일 때만 (일반 쿼리에는 미적용)

**이 케이스에 적용하면:**
```
쿼리: "SUPRAvvplus APC 교체 방법"  →  핵심 토큰: "apc"

검색 결과 점수 변화:
  supra_vplus_apc_valve_eng          → "apc" in doc_id ✅ → boost 3.0 → 1위!
  supra_vplus_adj_all_power_turn_on_off → "apc" 없음 ❌ → 변화 없음
  supra_vplus_rep_sub_unit_pdb_en      → "apc" 없음 ❌ → 변화 없음
```

**비용:** ES 쿼리에 should clause 1-2개 추가. wildcard on keyword field는 카디널리티 낮아 성능 영향 미미.
**효과:** APC valve 전용 문서가 retrieval 단계에서 바로 1위. group selection 문제에 도달하기 전에 해결.

**리스크:**
- 일반적인 토큰(2글자 이하, 불용어)이 과도하게 매칭될 수 있음
  → 핵심 토큰만 선별 (device_name/불용어 제외, 3글자 이상)
- non-SOP 문서에서 오작동
  → route=setup 또는 sop_pred=True 조건부 적용

**장점 (다른 수정 대비):**
- group selection(answer 단계) 패치가 아니라 **retrieval 단계에서 근본 해결**
- 기존 `device_boost` 패턴과 동일한 구현 방식으로 검증된 접근
- APC 전용 SOP가 존재하는 경우, 그 문서가 확실하게 1위로 올라옴

## 우선순위 및 의존성

| 순서 | 수정 | 상태 | 영향도 | 난이도 | 비고 |
|------|------|------|--------|--------|------|
| 1 | F: doc_id 키워드 부스트 (retrieval) | 구현 대기 | Critical | 중간 (ES should clause + 토큰 추출) | 가장 근본적. 전용 SOP가 있으면 retrieval에서 바로 1위 |
| 2 | E: answer early-exit → 재검색 | 구현 대기 | Critical | 중간 (answer 5줄 + 엣지 변경) | 전 그룹 실패 시 ~170s 낭비를 구조적으로 제거 |
| 3 | B: 그룹 정렬 본문 매칭 | 구현 대기 | High | 낮음 (3줄) | fallback 발생 시 올바른 그룹 선택 확률을 높임 |
| 4 | D: device name 정규화 | 구현 대기 | Medium | 중간 (~15줄) | query 품질 개선, F/B 효과 증폭 |
| 5 | C: not-found supplement | 구현 대기 | Medium | 낮음 (1줄 조건 변경) | 최종 안전망 |
| — | A: relevance 프롬프트 | ⚠️ 보류 | 미정 | 낮음 | RC-1 보류 — sub-step 매칭 허용 범위에 대한 기준 정립 필요 |

의존성:
- **F는 독립이며 가장 근본적.** retrieval 단계에서 해결하므로 answer 단계 수정(E/B/C)에 도달하기 전에 문제를 제거
- F가 완전히 작동하면 E/B/C는 도달하지 않음 (안전망으로 유지)
- F가 불완전한 경우 (전용 SOP가 없는 쿼리): E → B → C 순으로 방어
- D → F, B: D가 쿼리를 정규화하면 F의 토큰 추출, B의 토큰 매칭 모두 정확도 상승
- A는 별도 판단 후 합류 가능

## 기존 관련 Task와의 관계

[TASK-20260406-setup-group-selection-improvements.md](TASK-20260406-setup-group-selection-improvements.md):
- **이전 Task:** refs=1인 그룹이 너무 빨리 선택되는 문제 → MIN_REFS_FOR_ACCEPT=3 도입
- **이번 Task:** 모든 그룹이 relevance check에서 거부되는 문제 → 다른 실패 모드
- 이전 Task의 수정(fallback 시 answer_ref_json 미설정)은 이번 문제의 직접 원인이 아님.
  이번 문제는 **group ordering + relevance check + first-group fallback** 조합이 원인.
- 이전 Task의 수정 3(answer_ref_json 설정)은 이번 수정 C(supplement 허용)와 시너지.

## 최종 진단 문장

> setup route에서 APC 같은 컴포넌트명으로 질의할 때,
> 전용 SOP(`supra_vplus_apc_valve_eng`)가 인덱스에 존재하더라도
> **retrieval이 doc_id 키워드를 점수에 반영하지 않아** 해당 문서가 상위에 오지 못하고,
> answer node의 **group ordering + relevance check + first-group fallback** 조합으로
> 올바른 그룹을 선택하지 못해 실패가 반복된다.
>
> 즉, **retrieval scoring 부족 + group selection failure**의 복합 문제다.
> 가장 근본적 해결은 retrieval 단계에서 doc_id 키워드 부스트(수정 F)이며,
> answer 단계 수정(E/B/C)은 전용 SOP가 없는 경우의 방어선이다.

## Contracts To Preserve

- C-API-001: 응답 metadata 구조 변경 없음
- C-API-002: interrupt/resume 영향 없음

## Contracts To Update

- None

## Allowed Files

- `backend/llm_infrastructure/llm/langgraph_agent.py` (수정 A, B, C)
- `backend/services/agents/langgraph_rag_agent.py` (수정 D 접근 2 시)
- `backend/llm_infrastructure/query_expansion/abbreviation_expander.py` (수정 D 접근 1 시)

## Out Of Scope

- ES 검색 알고리즘, reranker 튜닝
- expand_related_docs_node 로직 변경
- answer/judge 프롬프트 대규모 개편
- 프론트엔드 변경
- retrieve scoring penalty/boost 조정

## Risks

- 수정 A: relevance check가 너무 관대해지면 무관한 그룹이 통과할 수 있음
  → 키워드 힌트를 매칭된 경우에만 추가하여 범위 제한
- 수정 B: content 샘플링이 큰 그룹에서 성능 저하 우려
  → grefs[:3]으로 제한, 각 500자 한도
- 수정 C: "not found" + supplement 조합이 hallucination 유발 가능
  → ref_items가 있을 때만 supplement 허용
- 수정 D: device name 추출이 잘못되면 쿼리가 오히려 훼손됨
  → fuzzy match 점수 임계값(82) 유지, 확신 높은 경우만 적용

## Verification Plan

```bash
# 1. 핵심 재현 케이스 — 오타 + 하위 컴포넌트
curl -s -X POST http://localhost:8001/api/agent/run \
  -H "Content-Type: application/json" \
  -d '{"message": "SUPRAvvplus APC 교체 방법"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'answer_chars={len(d.get(\"answer\",\"\"))}')
print(f'answer: {d.get(\"answer\",\"\")[:300]}')
print(f'attempts={d.get(\"metadata\",{}).get(\"attempts\",\"?\")}')
"
# 기대: answer_chars > 100, attempts=0 (1회에 성공)

# 2. 정상 케이스 regression — 직접 매칭 쿼리
curl -s -X POST http://localhost:8001/api/agent/run \
  -H "Content-Type: application/json" \
  -d '{"message": "SUPRA Vplus APC Valve 교체 작업 방법"}' | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'answer_chars={len(d.get(\"answer\",\"\"))}')
"
# 기대: 기존과 동일 수준의 답변 품질

# 3. 로그 확인 — relevance check 동작
docker logs rag-api-dev --since=120s 2>&1 | \
  grep -E "relevance check|selected group|fallback|sub-step|keyword"

# 4. 기존 테스트
cd backend && uv run pytest tests/ -v -k "test_abbreviation or test_expand"
```

## Verification Results

(구현 후 기록)

## Handoff

- Current status: Draft (분석 완료, 구현 대기)
- Remaining TODOs:
  1. 수정 F 구현 (doc_id 키워드 부스트 — ES should clause)
  2. 수정 E 구현 (answer early-exit + LangGraph 엣지 분기)
  3. 수정 B 구현 (그룹 정렬 본문 매칭)
  4. 수정 D 설계 결정 (접근 1 vs 2) + 구현
  5. 수정 C 구현 (not-found supplement 허용)
  6. 통합 테스트 (재현 케이스 + regression)
  7. ⚠️ 수정 A 재판단 — sub-step 매칭 허용 기준 케이스 수집 후 결정

## Change Log

- 2026-04-07: task 생성 — 사례 분석, 근본 원인 3가지 + 부차 원인 1가지 식별, 수정 방향 4가지 설계
- 2026-04-07: RC-1 / 수정 A 보류 — relevance check의 PDB≠APC 판정은 정확한 동작. sub-step 매칭 허용 범위 기준 미정립
- 2026-04-07: RC-2 재정의 — "fallback 그룹 오선택" → "전 그룹 실패 시 재검색 경로 부재(RC-2a) + fallback 정렬 기준 부족(RC-2b)". 수정 E(early-exit → 재검색) 추가
- 2026-04-07: 수정 F 추가 — doc_id 키워드 부스트(retrieval 단계). ES BM25 검색에서 doc_id가 점수에 미반영되는 문제 발견. 가장 근본적 해결로 1순위 격상. 최종 진단을 "retrieval scoring 부족 + group selection failure 복합 문제"로 수정
- 2026-04-07: 수정 F+B 구현 완료 — `feat/retrieval-logic-개선` 브랜치. F: es_search.py에 doc_id wildcard should-clause 추가 (boost=3.0), adapter/engine 4계층 파라미터 전달. B: langgraph_agent.py _group_query_score에 content sampling 추가. 기존 테스트 38건 전체 통과
- 2026-04-07: 수정 D 구현 완료 (접근 2) — `_normalize_device_in_query` 헬퍼 추가, answer_node에서 parsed_query.selected_devices 재활용하여 query_for_prompt 정규화. 접근 1은 향후 필요 시 검토 가능성 문서에 명시

## Final Check

- [ ] Diff stayed inside allowed files, or this doc was updated first
- [ ] Protected contract IDs were re-checked
- [ ] Verification commands were run, or blockers were recorded
- [ ] Any contract changes were reflected in `product-contract.md`
- [ ] Remaining risks and follow-ups were documented
