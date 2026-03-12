# SOP Filter Eval Analysis — sop_only_results.jsonl (79건)

Generated: 2026-03-12

---

## [OBJECTIVE]
`.sisyphus/evidence/2026-03-11_sop_filter_eval/sop_only_results.jsonl` (79건) 검색 품질 분석.
Doc/page hit rate, 문서 다양성, 질문 유형별 성능, hit_rank 분포, elapsed_ms 분포를 측정한다.

---

## [DATA]
- 총 질문: 79건 / 에러: 0건 / 검색 결과 상위: n_docs=20 (고정)
- 필터: SOP 문서 타입 한정 (sop_only)
- 장비: ZEDIUS XP 전용

---

## [FINDING 1] Doc hit rate = 94.9%, Page hit rate = 94.9%
[STAT:n] n = 79
[STAT:count] doc_hit=75/79, page_hit=75/79
[STAT:note] doc_hit과 page_hit이 완전히 동일 — hit_page는 hit_doc에 종속적임

## [FINDING 2] Install/Setting 유형이 유일한 실패 패턴 (hit rate 20%)
[STAT:count] Install/Setting: 1/5 hit (20.0%)
[STAT:count] Replacement: 53/53 (100%), Adjustment: 7/7 (100%), Other: 13/13 (100%)
[STAT:root_cause] gold_doc_id에 `&` 문자(`_&_`) 포함 vs 실제 검색 결과는 `_` 로 정규화된 ID 반환
  → gold_doc_id: `global_sop_supra_xp_sw_all_sw_installation_&_setting`
  → retrieved: `global_sop_supra_xp_sw_all_sw_installation_setting` (올바른 문서가 실제로 검색됨)
  → miss는 검색 실패가 아닌 **평가 스크립트의 ID 정규화 불일치** 문제
[STAT:note] idx=74 (O-ring 교체)도 마찬가지: gold_id의 `-`가 retrieved에서 `_`로 정규화

## [FINDING 3] 문서 다양성 낮음 — 쿼리당 평균 3.81개 unique doc (20개 슬롯 중)
[STAT:mean] avg unique docs per query = 3.81 (out of 20)
[STAT:median] median = 4.0
[STAT:range] min=1, max=9
[STAT:count] 19개 쿼리(24.1%)가 unique docs ≤ 2 → 특정 문서 편중 현상
[STAT:count] gold doc이 20슬롯 중 평균 4.71슬롯 점유 (중앙값 4.0)
[STAT:count] 8개 쿼리는 gold doc이 10슬롯 이상 점유 → 단일 문서 반복 검색

## [FINDING 4] hit_rank 분포 극단적으로 상위 집중
[STAT:count] rank 1 (top-1): 66/74 = 89.2% (found 쿼리 기준)
[STAT:count] rank 1-3: 71/74 = 95.9%
[STAT:count] rank 1-5: 73/74 = 98.6%
[STAT:count] rank 6 이상: 1건 (rank 6), rank 11-20: 0건
[STAT:mean] avg rank = 1.26, median rank = 1.0
[STAT:mrr] MRR (found 74건) = 0.9295 / MRR (전체 79건) = 0.8707

## [FINDING 5] 응답 시간 전체적으로 느림 — 평균 58.9초
[STAT:mean] mean = 58.9s (58,909ms)
[STAT:median] median = 56.5s
[STAT:std] stdev = 10.1s
[STAT:range] min=43.5s, max=93.8s
[STAT:percentile] P90=73.8s, P95=80.0s
[STAT:distribution] 45-60s 구간에 54/79건(68.4%) 집중
[STAT:correlation] elapsed vs unique_docs_retrieved: Pearson r = -0.210 (약한 음상관 — 다양한 문서 검색 시 약간 빠름)
[STAT:note] 빠른 절반(<56.5s) doc_hit=92.3% vs 느린 절반(≥56.5s) doc_hit=97.5% → 느린 쿼리가 오히려 더 잘 검색됨
[STAT:slow_pattern] 가장 느린 10건은 "Other"(PM ADJ, Calibration) 2건과 "Replacement" 8건 — 특정 유형 편향 없음

---

## [LIMITATION]
- `retrieved_doc_ids`는 최대 10개만 기록됨 (n_docs=20이나 실제 리스트 길이 10) — rank 11-20 분석 불가
- miss 4건 중 3건은 실제 올바른 문서가 검색되었으나 gold_id 정규화 불일치로 miss 판정 → 실제 검색 성능은 94.9%보다 높을 가능성
- elapsed_ms에는 LLM 답변 생성 시간이 포함되어 있어 순수 검색 지연 측정 불가
- 79건 모두 ZEDIUS XP 단일 장비 — 타 장비 일반화 불가
- 한국어 쿼리 70건 vs 영어 9건으로 언어별 비교 표본 불균형
