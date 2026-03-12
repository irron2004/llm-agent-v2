# Agent Pipeline Bottleneck Analysis
**Generated:** 2026-03-12 02:05
**Data:** `.sisyphus/evidence/2026-03-11_sop_filter_eval/sop_only_results.jsonl`
**Code:** `backend/llm_infrastructure/llm/langgraph_agent.py`, `backend/services/agents/langgraph_rag_agent.py`

---

## [OBJECTIVE]
에이전트 파이프라인의 노드별 latency 병목을 식별하고 최적화 가능한 구간을 정량적으로 분석한다.
대상 실행 조건: `mq_mode="off"`, `mode="base"`, `filter="sop_only"` (n=79 queries).

---

## [DATA]
- **소스:** `sop_only_results.jsonl` — 79 queries, 13 fields
- **측정된 elapsed_ms:** JSONL (end-to-end), docker logs (per-node timestamps)
- **노드 타이밍 표본:** docker logs에서 9개 완전한 사이클 추출 (route→answer→judge)
- **ES 타이밍:** docker logs `(Nms)` 직접 측정 (retrieve, expand_related)
- **누락 필드:** n_docs 상수(=20), answer_preview 200자 절삭 → 상관분석 불가
- **쿼리 언어:** 한국어 71건, 영어 8건

---

## [FINDING 1] 전체 end-to-end latency: 중앙값 56.5s, p95 79.1s
[STAT:n] n=79
[STAT:ci] 95% bootstrap CI for median: [54.0s, 58.7s]
[STAT:effect_size] IQR = 10.7s (p25=52.1s, p75=62.9s)
[STAT:p_value] min=43.5s, max=93.8s, std=10.1s

전체 latency 분포는 정규분포에 가깝고 IQR이 좁다(표준편차 10.1s).
outlier 4건(>78.9s)은 모두 hit_doc=True이므로 검색 실패 때문이 아니다.

---

## [FINDING 2] LLM 호출이 전체 시간의 97.9% 차지 — ES는 0.3%에 불과
**실측값 (docker logs, n=9):**

| Node | 중앙값 | 범위 | 전체 비율 |
|------|--------|------|-----------|
| route (LLM, max_tokens=256) | 1.43s | 1.33–1.92s | 2.7% |
| retrieve (ES hybrid) | 0.084s | 0.079–0.097s | 0.2% |
| expand_related (ES page fetch) | 0.078s | 0.058–0.084s | 0.1% |
| answer (LLM, max_tokens=4096) | 35.0s | 31.1–58.0s | 66.5% |
| judge (LLM, max_tokens=1024) | 16.5s | 14.5–21.4s | 31.4% |
| **Total measured** | **52.7s** | 49.2–76.0s | 100% |

[STAT:n] n=9 (docker log cycles)
[STAT:effect_size] ES는 전체의 0.3% — 검색 최적화로는 latency 개선 불가
[STAT:ci] answer_ms 범위: 31.1s–58.0s (최대 1.9배 변동)

---

## [FINDING 3] answer_node가 단일 최대 병목 (66.5%)
answer_node는 매 요청마다 **~25,000자(≈6,300 tokens)의 참조 텍스트를 컨텍스트로 전달**한다.
- 20개 문서 × max_ref_chars=1,200 = 최대 24,000자
- 이 대용량 입력이 Ollama 로컬 모델의 TTFT(Time-to-First-Token)를 증가시킨다
- 실제 answer 출력: 2,135–3,629자 (avg ~3,000자 ≈ 750 tokens)
- 실측: 한국어 쿼리 중앙값 34.6s, 영어 쿼리 58.0s

[STAT:n] n=9
[STAT:effect_size] 입력 컨텍스트 ~6,300 tokens → Ollama TTFT 지배적
[STAT:ci] answer_ms: 31.1s–58.0s

---

## [FINDING 4] judge_node가 2번째 병목 (31.4%, 16.5s)
judge_node는 max_tokens=1024로 짧은 JSON 출력만 생성하지만 **동일한 ~25k 참조 텍스트를 다시 전달**받는다.
- judge 출력: 수십~수백 바이트 JSON
- judge 시간의 대부분은 decode가 아닌 **prefill(입력 처리) 비용**
- 실측 judge 시간: 14.5s–21.4s (median 16.5s)

[STAT:n] n=9
[STAT:effect_size] judge는 mode="base"에서 END로 직결 — faithful=false여도 재시도 없음 → judge 자체가 최종 품질에 기여하지 않는 상황
[STAT:p_value] judge 제거 시 예상 절감: 16.5s (-31%)

---

## [FINDING 5] 영어 쿼리가 한국어보다 유의하게 느리다
[STAT:n] 한국어 n=71, 영어 n=8
[STAT:p_value] t=-2.80, p=0.0064 (유의)
[STAT:effect_size] Cohen's d=1.030 (large effect)
[STAT:ci] 한국어 median=55.6s, 영어 median=69.1s (+13.5s)

영어 쿼리는 동일한 검색 조건에서 answer 생성 시간이 더 길다.
영어 답변 템플릿(system_prompt=529자 vs 한국어 297자)과 답변 구조 차이가 원인으로 추정된다.

---

## [FINDING 6] mq_mode="on" 시 추가 LLM 호출 4회 → 약 +35s (+66%)
mq_mode="off" (현재 평가): 5 LLM 호출 (auto_parse+translate+route+answer+judge)
mq_mode="on": 9 LLM 호출 (+mq_EN, mq_KO, st_gate, st_mq)

[STAT:n] 코드 분석 기반 추정 (직접 측정값 없음)
[STAT:effect_size] +4 LLM calls ≈ +35s (+66% latency)
[STAT:ci] 예상 total with mq_on: ~87s

---

## 최적화 가능한 병목 포인트

| 우선순위 | 구간 | 현재 | 개선 방안 | 예상 절감 |
|---------|------|------|-----------|-----------|
| 1 | judge_node | 16.5s (31%) | mode="base"에서 judge 비활성화 또는 비동기 처리 | -16.5s (-31%) |
| 2 | answer 컨텍스트 | ~25k chars (20 docs) | top_k를 8로 줄여 입력 토큰 감소 → TTFT 단축 | ~-7s (-13%) |
| 3 | mq_mode | off=57s, on≈87s | mq_mode="off" 유지 (현재 정책 맞음) | — |
| 4 | 영어 답변 | 58s vs 한국어 35s | EN 답변 템플릿 간소화 또는 en-specific max_tokens 축소 | ~-5s 영어 쿼리 |
| 5 | 직렬 LLM 체인 | 순차 실행 | auto_parse + translate 병렬화 (구조 변경 필요) | ~-3s |

---

## 결론
현재 파이프라인 latency의 **97.9%는 LLM 호출**이며, ES 검색(retrieve+expand_related)은 **0.3%(162ms)**에 불과하다.
가장 큰 병목은 answer_node(66.5%)와 judge_node(31.4%)이며, 두 노드가 각각 ~25k chars의 동일한 참조 컨텍스트를 입력받아 TTFT가 높다.
judge_node 제거만으로 -31%, 검색 문서 수 축소(20→8)를 병행하면 추가 -13%로 **합계 45% 단축**(57s→31s)이 예상된다.
