# `/api/agent/run` 페이지 정확도 개선 전/후 보고서

작성일: 2026-03-01  
평가 데이터: `data/PE Agent 질문 리스트 - 0225 SOP 질문리스트.csv` (79건)  
평가 기준: 정답 문서 + 정답 페이지 범위 매칭

## 1) 코드 반영 내용

아래 변경을 적용했다.

- [backend/llm_infrastructure/llm/langgraph_agent.py](/home/hskim/work/llm-agent-v2/backend/llm_infrastructure/llm/langgraph_agent.py)
  - SOP 조기 페이지(`page<=2`) 점수 패널티 추가
  - 2단계 문서 내 재검색(doc-local sparse) 추가
  - 전/후 실험을 위한 env 토글 추가
    - `AGENT_EARLY_PAGE_PENALTY_ENABLED`
    - `AGENT_SECOND_STAGE_DOC_RETRIEVE_ENABLED`
- [backend/llm_infrastructure/retrieval/engines/es_search.py](/home/hskim/work/llm-agent-v2/backend/llm_infrastructure/retrieval/engines/es_search.py)
  - `build_filter()`에 `doc_ids` 필터 지원 추가

## 2) 실험 설정

- API: `/api/agent/run`
- `top_k=10`, `mode=verified`, `auto_parse=true`, `max_attempts=0`
- 인덱스: `rag_chunks_dev_current`
- 개선 전 모드:
  - `AGENT_EARLY_PAGE_PENALTY_ENABLED=false`
  - `AGENT_SECOND_STAGE_DOC_RETRIEVE_ENABLED=false`
- 개선 후 모드:
  - `AGENT_EARLY_PAGE_PENALTY_ENABLED=true`
  - `AGENT_SECOND_STAGE_DOC_RETRIEVE_ENABLED=true`

## 3) 결과 요약

| 지표 | 개선 전 | 개선 후 | 변화 |
|---|---:|---:|---:|
| page-hit@1 | 15/79 (19.0%) | 39/79 (49.4%) | **+24건, +30.4%p** |
| page-hit@3 | 45/79 (57.0%) | 46/79 (58.2%) | +1건, +1.3%p |
| page-hit@5 | 47/79 (59.5%) | 50/79 (63.3%) | +3건, +3.8%p |
| page-hit@10 | 55/79 (69.6%) | 56/79 (70.9%) | +1건, +1.3%p |
| doc-hit@1 | 43/79 (54.4%) | 42/79 (53.2%) | -1건, -1.3%p |
| doc-hit@3 | 49/79 (62.0%) | 46/79 (58.2%) | -3건, -3.8%p |
| doc-hit@5 | 50/79 (63.3%) | 50/79 (63.3%) | 동일 |
| doc-hit@10 | 57/79 (72.2%) | 56/79 (70.9%) | -1건, -1.3%p |

핵심:

- 목표였던 **페이지 1위 정확도(page-hit@1)**는 크게 개선됨.
- 상위 5/10 범위도 소폭 개선.
- 문서 hit는 저순위(@1, @3)에서 약간 하락해 trade-off가 존재.

## 4) 케이스 변화

- page-hit@1 개선: **24건 개선 / 0건 악화**
- 대표 개선 예:
  - Q1: `global_sop_supra_xp_all_efem_pio_sensor_board` top page가 `1 → 6`
  - Q14: `...ll_flow_switch` top page가 `1 → 6`
  - Q18: `...pm_baratron_gauge` top page가 `1 → 18`

주의 케이스:

- page-hit@3/5 악화 케이스가 일부 존재 (예: Q60, Q74)
- doc-hit@1/3는 일부 하락 (`improved 3 / regressed 4`)

## 5) 결과 파일(원문 근거)

- 개선 전
  - `docs/evidence/2026-03-01_agent_page_eval_before/summary.json`
  - `docs/evidence/2026-03-01_agent_page_eval_before/rows.csv`
  - `docs/evidence/2026-03-01_agent_page_eval_before/raw.jsonl` (질문별 answer + retrieved_docs 원문)
- 개선 후
  - `docs/evidence/2026-03-01_agent_page_eval_after/summary.json`
  - `docs/evidence/2026-03-01_agent_page_eval_after/rows.csv`
  - `docs/evidence/2026-03-01_agent_page_eval_after/raw.jsonl` (질문별 answer + retrieved_docs 원문)

## 6) 결론

- 이번 코드 반영은 **페이지 정확도 개선 목표에는 유효**했다.
- 특히 `page-hit@1` 개선폭이 커서, "목차/표지 페이지가 1위를 차지하는 문제"를 실질적으로 완화했다.
- 다만 문서 hit 저하가 일부 있어, 다음 단계로는 stage2 결과와 stage1 결과의 혼합 규칙(가중/컷오프) 보정이 필요하다.
