# ReactRAGAgent State Contract

이 문서는 `ReactRAGAgent.run()`이 반환해야 할 state dict의
key 존재 여부 + 값 semantics를 명문화한다.

소비처: `backend/api/routers/agent.py`의
`_build_response_metadata()` / `_sanitize_search_queries_raw()`

---

## 필수 state keys (C-API-001)

| key | 타입 | 의미 | ReactRAGAgent 생성 방법 |
|---|---|---|---|
| `route` | `str \| None` | 질문 라우팅 결과: `"setup"`, `"ts"`, `"general"` | `_infer_route(state)` 또는 planner 결정 |
| `st_gate` | `str \| None` | ST 게이트 결과: `"need_st"`, `"no_st"` | 단순화: `"no_st"` 고정 (Phase 2 이후 개선 가능) |
| `mq_used` | `bool` | Multi-query 검색 사용 여부 | 검색 횟수 > 1이면 `True` |
| `mq_reason` | `str \| None` | mq_used 이유 | `"react loop: N search(es)"` |
| `mq_mode` | `str` | 요청에서 주입됨 (`state_overrides`) | router가 `state_overrides["mq_mode"]`로 주입 |
| `attempts` | `int` | 답변 생성 시도 횟수 | 현재는 항상 0 (retry 미구현) |
| `retry_strategy` | `str \| None` | 마지막 retry 전략 | `None` (retry 미구현) |
| `guardrail_dropped_numeric` | `int` | 숫자 필터로 제거된 쿼리 수 | retrieve_node 결과에서 누적 |
| `guardrail_dropped_anchor` | `int` | anchor 필터로 제거된 쿼리 수 | retrieve_node 결과에서 누적 |
| `guardrail_final_count` | `int` | 최종 수집 문서 수 | `len(collected_docs)` |
| `search_queries` | `list[str]` | 최종 검색에 사용된 쿼리 | `search_queries_used` |
| `answer` | `str` | 생성된 답변 | `answer_node` 출력 |
| `display_docs` | `list` | UI 표시용 문서 | `_merge_display_docs(collected_docs)` |
| `docs` | `list` | 내부 검색 결과 | `collected_docs` |

---

## `search_queries_raw` semantics

`agent.py`의 `_sanitize_search_queries_raw(result)`는 다음 키를 **우선순위 순서**로 병합한다:

```
search_queries_raw → raw_search_queries →
general_mq_list → setup_mq_list → ts_mq_list →
general_mq_ko_list → setup_mq_ko_list → ts_mq_ko_list
```

**ReactRAGAgent 구현 방법:**
`result["general_mq_list"] = search_queries_used`

→ 별도 key 추가 없이 기존 sanitize 로직을 그대로 재사용.

**semantics 주의:**
- `search_queries_raw`는 guardrail 적용 *이전* 원본 후보 쿼리여야 한다.
- ReactRAGAgent에서는 planner가 결정한 쿼리 = 최종 쿼리이므로,
  `general_mq_list = search_queries_used`로 설정하는 것이 의미상 동등하다.
- sanitize는 길이 120자 초과 제거, 최대 5개 제한, 공백/중복 제거를 적용한다.

---

## interrupt payload (C-API-002 / C-API-003)

현재 Phase 1에서 ReactRAGAgent는 interrupt를 지원하지 않는다.
feature-flag 조건: `is_resume=False`, `retrieval_only=False`, `guided_confirm=False`

Phase 4 이후 추가 예정:
```python
# retrieval_only 시 (C-API-003)
interrupt({
    "type": "retrieval_review",
    "docs": display_docs,
    "response_mode": "retrieval_only",
})
```

---

## `mq_mode` 주입 방식

`mq_mode`는 `agent.py`에서 request 파라미터로 받아 `state_overrides`에 주입된다:

```python
# agent.py
effective_mq_mode = req.mq_mode or agent_settings.mq_mode_default
state_overrides["mq_mode"] = effective_mq_mode
```

ReactRAGAgent는 `state_overrides`를 통해 이 값을 받으므로 별도 처리 불필요.
단, `run()` 초기 state에서 `mq_mode: "react"` 기본값을 설정하여
`state_overrides` 없을 때도 key가 존재하도록 한다.

---

## 보조 keys (`_build_response_metadata` 추가 소비)

| key | 타입 | ReactRAGAgent 처리 |
|---|---|---|
| `detected_language` | `str \| None` | `translate_node` 결과에서 채워짐 |
| `target_language` | `str \| None` | `state_overrides`로 주입 |
| `task_mode` | `str \| None` | 현재 미사용 (`None` 허용) |
| `selected_doc_types` | `list[str]` | `parsed_query`에서 읽어서 채우기 |
| `retrieval_stage2` | `dict \| None` | 현재 미사용 (`None` 허용) |
| `selected_device` | `str \| None` | 현재 미사용 (`None` 허용) |

---

## 검증 기준

```bash
# import 확인
uv run python -c "from backend.llm_infrastructure.llm.react_agent import ReactRAGAgent; print('OK')"

# C-API-001 계약 테스트
uv run pytest tests/api/test_agent_response_metadata_contract.py -v
```
