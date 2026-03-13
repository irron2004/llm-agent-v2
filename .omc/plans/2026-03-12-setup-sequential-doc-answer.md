# Plan: Setup Route 순차 문서 적합성 판정 + 답변 생성

## 요구사항
SOP+setup route에서 모든 문서를 한번에 REFS로 전달하는 대신:
1. 문서를 1개씩 적합성 판정 (가벼운 LLM 호출)
2. 적합한 문서를 찾으면 해당 문서로만 답변 생성
3. 부적합하면 다음 문서로 넘어감

## 현재 흐름
```
expand_related → answer(모든 docs를 REFS로) → judge → retry
```
문제: 관련 없는 문서가 REFS에 포함되면 LLM이 억지 답변 생성

## 제안 흐름 (setup route only)
```
expand_related → relevance_check(doc #1) → 적합?
  → yes → answer(doc #1만 REFS로) → 완료
  → no  → relevance_check(doc #2) → 적합?
    → yes → answer(doc #2만 REFS로) → 완료
    → no  → ... (docs 소진까지)
  → 전부 부적합 → "관련 절차 문서를 찾지 못했습니다"
```

## 비용 비교
| 방식 | 최선 (1번째 적합) | 최악 (3개 다 부적합) |
|------|-------------------|---------------------|
| 현재 (전체 REFS) | answer 1회 + judge 1회 | answer 3회 + judge 3회 (retry) |
| 새 방식 | check 1회 + answer 1회 | check 3회 (가벼움) |

적합성 체크: max_tokens ~100, 답변 생성: max_tokens ~2048 → 체크가 ~20배 저렴

## 구현 계획

### Step 1: 적합성 판정 헬퍼 함수 추가
- **파일**: `langgraph_agent.py`
- **위치**: `_prioritize_setup_answer_refs` 근처 (line ~1069)
- **함수**: `_check_doc_relevance(query, doc_ref_text, *, llm, spec) -> bool`
- **프롬프트**:
  ```
  질문: {query}
  문서: {doc_ref_text}
  이 문서에 질문에 답할 수 있는 절차/작업 정보가 포함되어 있습니까?
  "yes" 또는 "no"로만 답하세요.
  ```
- max_tokens=10, temperature=0
- "yes" 포함이면 True, 아니면 False

### Step 2: ref_items를 doc_id별로 그룹핑하는 헬퍼 함수 추가
- **함수**: `_group_refs_by_doc_id(ref_items: List[dict]) -> List[Tuple[str, List[dict]]]`
- 같은 `doc_id`의 ref_items를 묶어서 `[(doc_id, [refs]), ...]` 반환
- 원래 rank 순서 유지 (첫 번째 그룹 = 가장 높은 점수)

### Step 3: answer_node에 setup 적합성 판정 루프 추가
- **파일**: `langgraph_agent.py`
- **위치**: `answer_node` (line 2492) 내부, route == "setup" 분기
- **상수**: `MAX_SETUP_DOC_TRIES = 3` (최대 3개 문서까지 시도)
- **로직**:
  ```python
  if route == "setup" and len(ref_items) > 0:
      doc_groups = _group_refs_by_doc_id(ref_items)

      # 적합한 문서 찾기
      selected_refs = None
      for i, (doc_id, group_refs) in enumerate(doc_groups[:MAX_SETUP_DOC_TRIES]):
          group_text = ref_json_to_text(group_refs)
          relevant = _check_doc_relevance(query_for_prompt, group_text, llm=llm, spec=spec)
          logger.info("answer_node: setup doc %d/%d (%s) relevant=%s",
                      i+1, len(doc_groups), doc_id, relevant)
          if relevant:
              selected_refs = group_refs
              break

      if selected_refs is None:
          # 전부 부적합 → 첫 번째 문서로 fallback (기존 동작)
          logger.info("answer_node: no relevant doc found, using first doc as fallback")
          selected_refs = doc_groups[0][1] if doc_groups else ref_items

      # 선택된 문서로만 ref_text 생성
      ref_items = selected_refs
      ref_text = ref_json_to_text(ref_items)

  # 이후 기존 답변 생성 로직 그대로
  ```

### Step 4: 로깅
- 각 doc 적합성 판정: `"answer_node: setup doc %d/%d (%s) relevant=%s (%.1fs)"`
- 최종 선택: `"answer_node: setup selected doc=%s after %d checks"`

## 수정 대상
- `backend/llm_infrastructure/llm/langgraph_agent.py` — 유일한 수정 파일

## 영향 범위
- `route == "setup"`일 때만 동작 변경
- ts, general route는 기존 로직 그대로
- 그래프 구조(edge) 변경 없음
- 기존 judge → retry 루프도 그대로 유지 (적합 문서 선택 후에도 judge가 unfaithful이면 retry)

## Acceptance Criteria
1. setup route에서 doc_id가 2개 이상이면 적합성 판정 루프 실행
2. 적합한 문서가 발견되면 해당 문서만으로 답변 생성 (LLM answer 호출 1회)
3. 적합성 판정은 문서당 max_tokens=10으로 가벼움
4. 전부 부적합이면 첫 번째 문서로 fallback
5. ts/general route는 기존 동작과 동일
6. 로그에서 적합성 판정 결과와 선택된 문서 확인 가능
7. MAX_SETUP_DOC_TRIES=3으로 최대 체크 횟수 제한

## 리스크 & 대응
| 리스크 | 대응 |
|--------|------|
| 적합성 판정이 잘못되면 좋은 문서를 건너뜀 | fallback으로 첫 번째 문서 사용 + 기존 judge retry 루프 유지 |
| LLM이 "yes"/"no" 외 다른 응답 | "yes" 포함 여부로 판단 (관대한 파싱) |
| 적합성 판정 LLM 호출 지연 | max_tokens=10으로 매우 빠름 (~1초 이내) |
| doc_groups가 1개면 불필요한 체크 | doc_groups가 1개이면 적합성 판정 생략, 바로 답변 생성 |
