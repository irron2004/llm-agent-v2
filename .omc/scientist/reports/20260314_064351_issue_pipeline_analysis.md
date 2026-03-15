# 이슈 검색 파이프라인 분석 보고서

생성일시: 20260314_064351
분석 대상: task_mode=issue 시 전체 검색~답변 파이프라인

---

[OBJECTIVE] task_mode=issue 시 검색 쿼리 생성 -> 검색 -> 리랭킹 -> 답변 생성까지의
전체 파이프라인을 분석하여 병목 및 품질 저하 포인트를 식별한다.

[DATA]
- 분석 파일: langgraph_agent.py (~4500 lines), langgraph_rag_agent.py (~700 lines)
- 프롬프트: issue_ans_v2.yaml, issue_detail_ans_v2.yaml, general_mq_v2.yaml, ts_mq_v2.yaml
- 설정: retrieval_hybrid_rrf_v1.yaml, retrieval_full_pipeline.yaml, presets.py
- 도메인: doc_type_mapping.py, es_search.py, search_service.py

---

## 1. route_node — issue 모드 라우팅

코드 근거: langgraph_agent.py:1303-1308
  task_mode == 'issue' -> route='general' (LLM 호출 없이 하드코딩)

[FINDING 1-A] issue 모드는 LLM 라우팅 호출 없이 route='general'로 확정된다.
[STAT:n] 라우팅 LLM 호출 0회 (다른 모드 대비 1회 절감)
- 장점: 지연 없음
- 단점: ts(트러블슈팅) 계열 쿼리임에도 ts_mq 프롬프트 대신 general_mq 프롬프트가 사용됨

---

## 2. mq_node — MQ 생성: general_mq 프롬프트 사용

general_mq_v2.yaml:
  system: 'You are a search query generator for GENERAL documentation retrieval.'
  Line 1 = original query (no rewrite)
  Lines 2-3 = 'distinct retrieval variants for broader coverage'
  이슈/트러블슈팅 특화 키워드 지시 없음

ts_mq_v2.yaml (비교):
  system: 'You are a search query generator for TROUBLESHOOTING retrieval.'
  'Add cause/diagnosis/action keywords (root cause, check, reset, mitigation, checklist)'
  'Add module/hardware terms only when relevant (sensor, valve, controller, PLC)'

[FINDING 2-A] issue 모드는 route='general'이므로 general_mq 프롬프트로 MQ를 생성한다.
ts_mq 대비 원인/조치/하드웨어 키워드 확장 지시가 없어 이슈 검색 쿼리 품질이 범용 수준에 머문다.
[STAT:effect_size] ts_mq: 도메인 키워드 4개 클래스 명시 (원인/진단/조치/하드웨어)
[STAT:effect_size] general_mq: 도메인 키워드 0개 클래스 명시
[STAT:n] MQ 생성: EN 3개 + KO 3개 = 최대 6개 쿼리 (bilingual, mq_node)
[LIMITATION] 프롬프트 품질 차이의 실제 검색 recall 영향은 A/B 실험 없이 정량화 불가

---

## 3. selected_doc_types — issue 모드 필터 설정

코드 근거: langgraph_agent.py:4217-4219
  task_mode='issue' 시:
    selected_doc_types = expand_doc_type_selection(['myservice', 'gcb', 'ts'])
    selected_doc_types_strict = True

expand_doc_type_selection 결과 (doc_type_mapping.py):
  'myservice' -> ['myservice']                           (1개)
  'gcb'       -> ['gcb', 'maintenance']                  (2개)
  'ts'        -> ['문제 해결 가이드', 'Trouble Shooting Guide', 'trouble shooting',
                  'trouble shooting guide', 'trouble shooting guilde', 'troubleshooting',
                  'Guide', 'ts', 't/s', 'Troubleshooting Guide', 'trouble_shooting_guide']
                                                          (11개)
  합계: 14개 variant

ES 반영 방식 (es_search.py:598-623):
  doc_types 리스트 -> _terms_or_keyword('doc_type', normalized)
  -> bool.filter (must 절) 에 hard filter로 적용
  -> doc_types_strict=True: expand 없이 이미 확장된 14개 variant를 그대로 전달

[FINDING 3-A] doc_type 필터는 ES must/filter 절의 hard filter로 동작한다.
[FINDING 3-B] selected_doc_types_strict=True이므로 extra expand 없이 14개 variant가 ES에 전달된다.
[STAT:n] 필터 variant 수: 14개

---

## 4. retrieve_node — 검색 설정 및 top_k

설정값 (langgraph_rag_agent.py:81-99):
  top_k = 20              (final_top_k: rerank 후 반환 수)
  retrieval_top_k = 50    (초기 후보 수)
  candidate_k = max(50, 20*2, 20) = 50  (per-query ES 검색 수)
  최대 6 queries x 50 = 300 docs -> dedup -> sort -> top 50

reranker 활성화 (search_service.py:77-103):
  self.rerank_enabled = rag_settings.rerank_enabled  (환경변수 의존)
  self.reranker = _build_reranker() if self.rerank_enabled else None
  -> LangGraphRAGAgent는 search_service.reranker를 그대로 상속

reranker 쿼리 (langgraph_agent.py:2145-2156):
  rerank_query = query_en if query_en else original_query
  docs = reranker.rerank(rerank_query, all_docs, top_k=final_top_k)

[FINDING 4-A] issue 모드 전용 리랭커 설정이 없다. SOP/TS/general과 동일한 리랭커를 공유한다.
[FINDING 4-B] reranker가 영어 쿼리(query_en) 기반으로 동작한다.
이슈 문서는 주로 한국어이므로 영어 기반 cross-encoder의 점수 정확도가 낮을 수 있다.
[STAT:n] 분석 근거: langgraph_agent.py:2145-2156
[LIMITATION] 리랭킹 품질 차이는 cross-encoder 모델의 다국어 지원 여부에 따라 달라짐

---

## 5. SOP 전용 스코어 조정 — issue 모드 미적용

코드 근거: langgraph_agent.py:1954-2088
  sop_only_predicate = (sop_intent=True) OR (selected_doc_types가 SOP variants 포함)
  issue 모드: selected_doc_types = myservice + gcb + ts -> SOP 미포함
  -> sop_only_predicate = False

비활성화되는 최적화 (issue 모드):
  1) early_page_penalty (앞 페이지 패널티)
  2) sop_soft_boost (SOP 문서 점수 부스트)
  3) procedure_boost (절차 섹션 부스트)
  4) scope_penalty (목차/TOC 페이지 패널티)

[FINDING 5-A] SOP 전용 스코어 조정 4가지가 issue 모드에서 모두 비활성화된다.
이는 의도된 설계이나, issue 문서(myservice/gcb/ts)에 대한 유사 품질 조정이 전혀 없다.
[STAT:n] 비활성화된 스코어 조정 로직: 4개

---

## 6. expand_related_docs_node — issue 문서 확장 전략

코드 근거: langgraph_agent.py:2228-2422
  DOC_TYPES_SAME_DOC = {'gcb', 'myservice', 'pems'}

  doc_type in DOC_TYPES_SAME_DOC (gcb, myservice):
    -> doc_fetcher(doc.doc_id)  # 동일 doc의 모든 청크 fetch (전체 문서 확장)

  doc_type = ts (chapter_ok=True, section_chapter 존재):
    -> section_fetcher(doc_id, section_chapter)  # 챕터 단위 섹션 확장
    -> 연속 페이지 그룹만 유지, max_pages=20

  ts (chapter_ok=False 또는 section_chapter 없음):
    -> page_fetcher(doc.doc_id, pages)  # ±2 페이지 윈도우 확장

expand_top_k (확장 적용 대상 문서 수):
  기본: EXPAND_TOP_K = 10
  retry_expand_node 후: 20

[FINDING 6-A] expand_related_docs_node는 task_mode를 체크하지 않는다.
issue 전용 확장 로직 없이 doc_type 기반 범용 로직만 동작한다.
[FINDING 6-B] ts 문서에서 chapter_ok 메타데이터가 없으면 ±2 페이지 윈도우로 fallback된다.
이슈 문서의 chapter_ok 충실도가 확장 품질을 결정한다.
[STAT:n] expand_top_k 기본값: 10 (docs 기준)

---

## 7. answer_node — issue 답변 생성

코드 근거: langgraph_agent.py:2687-2725
  MAX_ANSWER_REFS = 5
  task_mode='issue': ref_items[:MAX_ANSWER_REFS] -> issue_ans_v2 프롬프트
  _build_issue_cases(ref_items, max_cases=10) -> issue_top10_cases

issue_ans_v2 시스템 프롬프트:
  '이슈 분석 RAG 어시스턴트'
  'REFS 라인만 증거로 사용 (추측 금지)'
  '가능한 사례를 번호 목록으로 정리, 각 사례에 핵심 내용과 근거 포함'
  '[N] 형식으로 출처 인용'
  -> 구조적 섹션 강제 없음 (detail_ans와 달리 ## 이슈 내용 / ## 해결 방안 없음)

[FINDING 7-A] MAX_ANSWER_REFS=5로 답변 생성 REFS가 최대 5개로 제한된다.
_build_issue_cases는 max_cases=10까지 사례를 추출하지만 실제 LLM에 전달되는 REFS는 5개뿐이다.
[STAT:n] MAX_ANSWER_REFS=5, issue_top10_cases max_cases=10 -> 불일치
[FINDING 7-B] issue_ans 프롬프트는 사례 목록 구조화에 특화되어 있으나,
각 사례의 필수 항목(발생 조건/원인/해결책)을 강제하지 않는다.

---

## 8. 2단계 상세 답변 플로우

코드 근거: langgraph_agent.py:2943-3056
  issue_confirm_node -> (interrupt) 사용자 사례 선택
  issue_case_selection_node -> selected_doc_id 설정
  issue_detail_answer_node ->
    selected_refs = [r for r in all_refs if r['doc_id'] == selected_doc_id]
    # 1단계 검색 결과 재사용, 추가 검색 없음
    tmpl = issue_detail_ans_v2  # ## 이슈 내용, ## 해결 방안 강제

[FINDING 8-A] issue_detail_answer_node는 추가 검색 없이 1단계 검색 결과를 재사용한다.
1단계에서 해당 문서의 청크가 충분히 포함되지 않았다면 상세 답변 품질이 저하된다.
[STAT:n] 2단계에서 추가 ES 검색 호출: 0회

---

## 9. 전체 파이프라인 흐름 요약 (task_mode=issue)

  사용자 쿼리
    |
    v route_node: route='general' (LLM 호출 없이 하드코딩)
    |
    v mq_node: general_mq 프롬프트 (EN+KO 각 3개 = 최대 6개 쿼리)
    |          [이슈 특화 키워드 확장 없음]
    |
    v st_gate_node -> st_mq_node: 최종 EN 3개 + KO 3개
    |
    v retrieve_node:
    |   - doc_type filter: myservice+gcb+ts 14개 variant (hard filter, strict)
    |   - candidate_k=50 per query, max 6q x 50 = 300 -> dedup
    |   - sop_only_predicate=False -> procedure_boost/scope_penalty 비활성
    |   - reranker: rag_settings.rerank_enabled 의존, query_en 기반
    |   - final_top_k=20
    |
    v expand_related_docs_node:
    |   - gcb/myservice: 전체 문서 fetch (doc_fetcher)
    |   - ts: 챕터 단위 섹션 fetch (chapter_ok 의존)
    |   - expand_top_k=10
    |
    v answer_node (issue 분기):
    |   - MAX_ANSWER_REFS=5 절삭
    |   - issue_ans_v2 프롬프트 (사례 목록)
    |   - issue_top10_cases 생성 (max 10)
    |
    v issue_confirm_node -> (interrupt) 사용자 선택
    |
    v issue_case_selection_node -> selected_doc_id
    |
    v issue_detail_answer_node:
        - 1단계 결과 재사용 (추가 검색 없음)
        - issue_detail_ans_v2: ## 이슈 내용, ## 해결 방안 강제

---

## FINDING 우선순위 요약 (병목/품질 저하 포인트)

[FINDING P1] MQ 프롬프트 비최적화 (높은 영향)
  - task_mode=issue임에도 general_mq 사용 -> ts_mq 대비 이슈 도메인 커버리지 낮음
  - 개선: issue 전용 mq 프롬프트 추가, 또는 issue 모드에서 ts_mq 재사용

[FINDING P2] MAX_ANSWER_REFS=5 vs issue_top10_cases=10 불일치 (중간 영향)
  - LLM에 전달 REFS 5개, 사례 목록은 10개로 생성 -> 상위 5 doc만 답변 반영
  - 개선: issue 모드 MAX_ANSWER_REFS를 별도 설정값으로 분리 (예: 8~10)

[FINDING P3] 2단계 상세 답변의 추가 검색 부재 (중간 영향)
  - issue_detail_answer_node가 1단계 결과를 재사용
  - 선택된 문서의 청크가 1단계에서 누락됐을 때 답변 품질 저하
  - 개선: detail 단계에서 selected_doc_id로 추가 doc-level fetch 수행

[FINDING P4] 영어 쿼리 기반 리랭킹 (낮음~중간 영향, 모델 의존)
  - rerank_query = query_en 사용 -> 한국어 이슈 문서와 매칭 정확도 저하 가능
  - 개선: 한국어 이슈 모드에서 query_ko 우선 사용

[FINDING P5] issue 문서 전용 스코어 조정 부재 (낮은 영향)
  - SOP에는 4가지 스코어 조정이 있으나 issue 문서는 0가지
  - 개선: ts/myservice '해결 방안' 섹션 부스트 등 issue 전용 조정 추가

---

[LIMITATION]
1. 실제 이슈 문서의 chapter_ok 비율, metadata 충실도 미확인
2. rag_settings.rerank_enabled 운영 설정값 미확인 (환경변수 의존)
3. MQ 프롬프트 품질 차이의 recall 영향은 A/B 실험 없이 정량화 불가
4. query_en vs query_ko 리랭킹 품질은 cross-encoder 모델 다국어 지원 여부에 의존
5. MAX_ANSWER_REFS=5 제한의 답변 품질 영향은 평가 데이터셋 없이 측정 불가
6. 이 분석은 정적 코드 분석으로, 런타임 실제 동작과 상이할 수 있음