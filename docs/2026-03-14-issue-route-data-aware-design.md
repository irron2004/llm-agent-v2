# 2026-03-14 Issue Route Data-Aware 개선 설계안

## 1) 목적

`task_mode=issue` 경로를 데이터 상태에 맞게 개선한다.
핵심은 라우팅 강제 자체를 바꾸는 것이 아니라, 검색 이후 선택 정책을 데이터 기반으로 조정해
recall 저하 없이 문서 다양성과 답변 품질을 높이는 것이다.

## 2) 범위와 제약

- 범위: issue 경로의 선행 개선(MQ/메타/섹션 병합), 신호 산출, 정책 tier, refs 선택 규칙, 롤아웃/검증 계획
- 비범위: 이번 문서 단계에서 코드 변경 없음
- 보호 계약:
  - `C-API-001` metadata contract 유지
  - `C-API-002` interrupt/resume continuity 유지
  - `C-API-003` retrieval_only semantics 유지 (shared retrieve/expand node 회귀 금지)

## 3) 데이터 상태 요약 (문서 + live ES)

### 3.1 문서 근거

1. `issue` 모드는 현재 `route='general'`로 고정되어 동작한다.
   - 근거: `.omc/scientist/reports/20260314_064351_issue_pipeline_analysis.md`
2. `results_to_ref_json()` 메타데이터는 `device_name`, `equip_id` 중심으로 매우 얇다.
   - 근거: `.omc/scientist/reports/20260314_064749_doc_type_content_quality_report.md`
3. `MAX_ANSWER_REFS=5`와 `issue_top10_cases` 의도 사이에 불일치가 존재한다.
   - 근거: `.omc/scientist/reports/20260314_064351_issue_pipeline_analysis.md`
4. GCB는 섹션별 정보가 분리 청크로 나뉘어 단일 청크 검색 시 정보 손실 위험이 크다.
   - 근거: `docs/2026-03-14-이슈조회-성능-개선-연구.md`
5. `general_mq` 프롬프트가 이슈 검색에 사용되어 이슈 특화 키워드 확장이 없다.
   - 근거: `docs/2026-03-14-이슈조회-성능-개선-연구.md` §2.2 B1

### 3.2 live 데이터 스냅샷 (2026-03-14)

**출처 1: all_chunks.jsonl 전수 집계**

| doc_type | 청크 수 | 비율 | 고유 doc_id |
|----------|---------|------|-------------|
| myservice | 329,206 | 83.2% | ~82,000 |
| gcb | 49,021 | 12.4% | 13,848 |
| sop | 13,116 | 3.3% | - |
| setup | 3,583 | 0.9% | - |
| ts | 760 | 0.2% | 79 |
| pems | 87 | 0.0% | - |
| **합계** | **395,773** | | |

**출처 2: live ES (`chunk_v3_content`) 집계**

전체 390,717건 (all_chunks.jsonl 대비 약 5,000건 차이 — 인제스트 시점 차이)

> **주의**: scientist 보고서에서 "ts 0건 적재", "myservice 62.2% 빈 콘텐츠"로 보고했으나,
> 실제 데이터 검증 결과 **ts 760건 정상 적재**, **myservice 빈 콘텐츠 0%**로 확인되었다.
> 아래 수치는 all_chunks.jsonl 전수 집계 기준이다.

### 3.3 doc_type별 실제 데이터 구조

#### myservice (329,206 청크)

각 유지보수 이력이 4개 섹션으로 **별도 청크** 저장:

| 섹션 | 청크 수 | 중앙값(자) | 역할 |
|------|---------|-----------|------|
| status | 78,953 | 46 | 증상/현상 |
| cause | 81,524 | **34** | 원인 (매우 짧음) |
| action | 81,336 | 221 | 수행 조치 |
| result | 87,393 | 82 | 결과 |

- 빈 콘텐츠 0% (인제스트 시 필터링됨)
- **cause 중앙값 34자**로 원인 정보 구조적 부족
- 한 이슈 = 4개 청크 → 검색 시 1개만 hit되면 나머지 3개 섹션 정보 유실

#### gcb (49,021 청크 / 13,848 문서)

섹션 조합 패턴:

| 패턴 | 문서 수 | 비율 |
|------|---------|------|
| summary + detail | 7,444 | 53.8% |
| summary + detail + timeline | 5,749 | 41.5% |
| 기타 (+ background/request/cause 등) | 655 | 4.7% |

주요 섹션 특성:

| 섹션 | 청크 수 | 중앙값(자) | 역할 |
|------|---------|-----------|------|
| summary | 16,228 | 103 | 제목 + 요청유형 + 상태 + 장비명 |
| detail | 16,485 | 1,256 | 종결 요약 또는 상세 이력 (일관성 낮음) |
| timeline | 15,259 | 1,375 | 담당자 간 커뮤니케이션 이력 |
| cause | 111 | - | 극소수 문서에만 존재 (0.8%) |

> **참고**: gcb_parser.py가 정의하는 question/resolution 구조와 **실제 적재 데이터의
> summary/detail/timeline 구조가 다르다.** 파서 코드와 실제 인제스트 파이프라인 간
> 불일치가 존재한다.

- 영문 기반이나 nori analyzer 적용 → 영문 BM25 정밀도 제한

#### ts (760 청크 / 79 문서)

- VLM 파싱된 PDF → markdown (구조화된 alarm/symptom/cause/action)
- 이슈 검색에 **가장 적합한 구조**이지만 전체의 0.2%로 소량
- device 대부분 "ALL"

### 3.4 샘플 쿼리 top-50 doc_type 분포

| 쿼리 | myservice | gcb | ts |
|------|-----------|-----|-----|
| `PM2 vacuum error` | 49 | 1 | 0 |
| `pressure instability after maintenance` | 50 | 0 | 0 |
| `alarm 발생 원인 조치` | 49 | 1 | 0 |
| `chamber leak troubleshooting` | 44 | 3 | 3 |

**결론: 대부분 쿼리에서 myservice_share >= 0.85 → 대부분 Tier 3에 빠질 가능성 높음**

### 3.5 doc_type별 데이터 형태와 프롬프트 적합성

현재 `task_mode=issue`로 진입하면 myservice/gcb/ts 모두 동일한 `issue_ans` 프롬프트를 사용한다.
각 doc_type의 실제 데이터 형태를 확인하고 프롬프트 적합성을 평가한다.

#### 현재 라우팅 구조

```
사용자 doc_type 선택
  ├─ myservice/gcb/ts 조합 → task_mode="issue", route="general" (강제)
  │   → issue_ans 프롬프트 (사례 나열) → confirm → case select → issue_detail_ans
  ├─ setup/sop 조합 → task_mode="sop"
  │   → LLM 라우터가 setup/ts/general 중 판단 → 각 route별 프롬프트
  └─ 혼합 → LLM 라우터 판단
```

#### myservice — 적합

- 데이터: 짧은 유지보수 이력 (status 46자, cause 34자, action 221자, result 82자)
- 프롬프트: `issue_ans` (사례 나열 + REFS 인용) → **적합**
- 다수의 유사 사례를 비교·나열하는 것이 myservice의 활용 목적에 맞음

#### gcb — 부분 부적합

- 데이터: **영문** 커뮤니케이션 체인. 담당자 간 코멘트, 로그 분석, 원인/결과 상세
  - detail 중앙값 1,256자, timeline 1,375자로 정보 밀도 높음
  - 예: `"2025.12.22 Evan Dear Anthony, 'ERROR 30' message is not related with host communication..."`
- 프롬프트 문제:
  1. `issue_ans`가 "반드시 한국어로" 강제 → 영문 원본의 번역 품질 이슈
  2. 한 건 자체가 긴 분석 보고서인데, 짧은 사례 나열로 축소
  3. 담당자 코멘트 체인에서 핵심 원인/조치를 추출하는 별도 지시 없음

#### ts — 부적합

- 데이터: VLM 파싱 PDF → markdown. **구조화된 진단 절차서**
  - Failure symptoms → Check point → Key point 테이블 형식
  - 예: `"FFU Pressure range error → A-1. Side door → Normal: Close"`
- 프롬프트 문제:
  1. ts는 "과거 사례"가 아니라 **표준 진단 절차서** → 사례 나열보다 절차 안내가 적합
  2. 기존 `ts_ans_v2.yaml`이 별도 존재하며 "## 준비/안전, ## 작업 절차, ## 복구/확인" 구조로 절차 중심 답변 생성
  3. `task_mode=issue`로 강제되면 `ts_ans`가 아닌 `issue_ans`가 사용됨

#### setup / sop — 적합 (참고)

- 데이터: 설치 매뉴얼(목차+절차), SOP(안전+절차+체크시트)
- 프롬프트: `setup_ans_v3` (극도로 엄격한 절차 중심) → **적합**
- issue 경로와 무관

#### 프롬프트 개선 방향

| doc_type 조합 | 현재 | 개선안 |
|---------------|------|--------|
| myservice 단독/혼합 | issue_ans | 유지 (섹션 병합 Q2만 개선) |
| gcb 단독/혼합 | issue_ans | issue_ans에 gcb 영문 원본 처리 지시 추가 (Q5) |
| ts 단독 | issue_ans | `ts_ans` 경로 사용 검토 (Q6) |
| myservice+gcb+ts 혼합 | issue_ans | REFS에 doc_type별 역할 설명 추가 (Q3) |

> **설계 판단**: 별도 route 분리까지는 불필요하다. `issue_ans` 프롬프트 내에서
> doc_type별 처리 지시를 추가하는 것이 더 효율적이다. 단, **ts 단독 선택**은 예외로
> `task_mode=issue` 강제를 해제하고 기존 `route=ts` → `ts_ans` 경로를 태우는 것이 적합하다.

## 4) 문제 재정의

1. **myservice 편중으로 다양성 부족** (데이터 구조 문제)
   - top-50 검색 결과 중 myservice가 85~100%를 차지 → gcb/ts 증거가 거의 노출되지 않음
   - §3.4 샘플 쿼리 4건 모두 myservice_share >= 0.88
2. **`general_mq` 프롬프트의 이슈 특화 부재** (검색 품질 문제)
   - `task_mode=issue`에서 `route='general'`로 하드코딩 → `general_mq` 사용
   - `ts_mq`에는 "root cause, check, reset, mitigation" 등 트러블슈팅 키워드 확장이 있으나 `general_mq`는 범용 변형만 생성
3. **섹션 분리 저장으로 인한 정보 유실** (데이터 구조 문제)
   - myservice: 1이슈 = 4청크(status/cause/action/result), 검색 시 1개만 hit되면 나머지 3섹션 누락
   - gcb: 1이슈 = 2~3청크(summary/detail/timeline), summary만 hit되면 상세 정보 없음
   - 상세 답변(`issue_detail_answer_node`)에서 추가 ES 검색 없이 1단계 결과만 재사용
4. **요약/사례 선택의 증거 다양성 부족** (정책 문제)
   - `MAX_ANSWER_REFS=5`와 `issue_top10_cases` 의도 사이에 불일치
   - 6~10번째 사례는 LLM이 요약에 인용하지 못한 채 선택지로만 노출
5. **REFS 메타데이터 부족** (프롬프트 문제)
   - `doc_type/chapter` 값은 포함되지만 각 값의 의미를 LLM에 설명하지 않음
   - gcb의 summary/detail, myservice의 cause/action 등을 LLM이 구분 불가
6. **하드 필터는 recall 손실 위험** (설계 제약)
   - hard filter는 contamination은 줄이지만 recall을 크게 손상시키는 경향이 확인됨
   - 소프트 제어(quota/다양성)로 접근해야 함
7. **doc_type별 프롬프트 부적합** (프롬프트 문제)
   - `task_mode=issue`로 강제되면 myservice/gcb/ts 모두 동일한 `issue_ans` 프롬프트 사용
   - **ts**: 진단 절차서(Failure symptoms → Check point → Key point 테이블)인데 사례 나열 형식으로 처리됨. 기존 `ts_ans`(절차 중심)가 더 적합
   - **gcb**: 영문 커뮤니케이션 체인(담당자 간 코멘트, 로그 분석)인데 "반드시 한국어" 강제. 긴 분석 보고서를 짧은 사례로 축소
   - **myservice**: 짧은 유지보수 이력 → 사례 나열 형식은 적합

## 5) 선행 Quick-win 개선 (tier/signal 이전)

tier/signal 아키텍처는 검색 결과의 후처리 정책이다. 그 효과를 최대화하려면 **검색 입력 품질 자체**를 먼저 개선해야 한다. 아래 6가지는 tier 도입과 독립적으로 즉시 적용 가능하다.

| # | 개선안 | 영향 | 난이도 | 변경 위치 |
|---|--------|------|--------|-----------|
| Q1 | **issue 전용 MQ 프롬프트** (`issue_mq_v2.yaml`) | 높음 | 낮음 | `prompts/`, `mq_node` 분기 |
| Q2 | **동일 doc_id 섹션 자동 병합** (expand 단계) | 높음 | 중간 | `expand_related_docs_node` |
| Q3 | **REFS에 섹션 유형 설명 추가** (프롬프트 수정) | 중간 | 낮음 | `issue_ans_v2.yaml` |
| Q4 | **MAX_ISSUE_REFS=10 상수 분리** | 중간 | 낮음 | `langgraph_agent.py` 상수 |
| Q5 | **gcb 영문 원본 처리 지시** (프롬프트 수정) | 중간 | 낮음 | `issue_ans_v2.yaml` |
| Q6 | **ts 단독 선택 시 ts_ans 경로 사용** | 중간 | 낮음 | `_infer_task_mode_from_doc_types()` |

### Q1. issue 전용 MQ 프롬프트

현재 `general_mq`가 사용되어 트러블슈팅 키워드 확장이 없다. `ts_mq`처럼 "root cause, alarm, symptom, mitigation, sensor, valve" 등의 도메인 키워드를 확장하는 `issue_mq_v2.yaml`을 추가한다.

- `mq_node`에서 `task_mode == "issue"` 분기 추가
- `route='general'` 하드코딩은 유지하되, MQ 프롬프트만 분리

### Q2. 동일 doc_id 섹션 자동 병합

myservice의 action 청크만 hit되면 status/cause/result 정보가 유실된다. `expand_related_docs_node`에서 hit된 청크의 `doc_id`로 같은 문서의 다른 섹션 청크를 자동 fetch하여 병합한다.

- `search_service.fetch_doc_chunks(doc_id)` 활용 (이미 구현됨)
- gcb도 동일하게 summary+detail+timeline을 병합

### Q3. REFS 섹션 유형 설명

`issue_ans` 프롬프트에 doc_type별 섹션 의미를 명시한다:
- myservice: status(증상), cause(원인), action(조치), result(결과)
- gcb: summary(요약), detail(상세), timeline(이력)
- ts: 구조화된 트러블슈팅 가이드

### Q4. MAX_ISSUE_REFS=10

issue 모드에서 case list(10개)와 answer refs(5개)를 분리 운영한다. §7.2에서 상세 설명.

### Q5. gcb 영문 원본 처리 지시

gcb는 영문 커뮤니케이션 체인으로, 담당자 간 코멘트·로그 분석이 원본 그대로 저장되어 있다. `issue_ans` 프롬프트에 gcb 처리 지시를 추가한다:

- gcb REFS는 영문 원본 → 핵심 원인/조치를 추출하여 한국어로 요약
- 담당자 코멘트 체인에서 최종 확정 조치(Close 시점 코멘트)를 우선 인용
- detail/timeline이 긴 경우 핵심만 발췌

### Q6. ts 단독 선택 시 ts_ans 경로 사용

ts 데이터는 "과거 사례"가 아닌 **표준 진단 절차서**(Failure symptoms → Check point → Key point)이다. ts만 선택한 경우 사례 나열(`issue_ans`)보다 절차 안내(`ts_ans`)가 더 적합하다.

- `_infer_task_mode_from_doc_types()`에서 ts 단독 선택 시 `task_mode="issue"` 강제 해제
- 기존 `route=ts` → `ts_ans` 프롬프트 경로를 태움
- myservice+ts, gcb+ts 등 혼합 선택 시에는 기존 `task_mode=issue` 유지

## 6) 설계 원칙

1. **Route override는 유지**: `task_mode=issue`의 안정 경로는 유지한다.
2. **Quick-win 선행**: tier/signal 도입 전에 §5의 MQ/섹션병합/메타 개선을 먼저 적용한다.
3. **하드 필터보다 소프트 제어 우선**: 쿼터/다양성 제어로 contamination을 관리한다.
4. **온라인 저비용 신호 사용**: 현재 retrieve 결과만으로 계산 가능한 신호를 우선 사용한다.
5. **Contract-safe 변경**: API 필수 키/interrupt semantics는 유지하고, 필요한 정보는 확장 필드로 추가한다.

## 7) 제안 아키텍처

### 7.1 신호 스키마 (retrieve 단계)

신호 계산 위치(구현 예정): `backend/llm_infrastructure/llm/langgraph_agent.py`의 `retrieve_node`

| 신호 | 정의 | 초기 threshold | 목적 |
| --- | --- | --- | --- |
| `score_gap_12` | `(s1-s2)/max(abs(s1),1e-9)` | high `>=0.12`, low `<0.06` | 상위 1위 확신도 |
| `myservice_share_50` | top-50 내 myservice 비율 | dominant `>=0.85` | 편중 감지 |
| `gcb_count_50` | top-50 내 gcb 개수 | available `>=1` | gcb 증거 가용성 |
| `ts_count_50` | top-50 내 ts 개수 | available `>=1` | ts 증거 가용성 |
| `non_myservice_presence_50` | top-50 내 gcb/ts 존재량 | available `>=1` | 대체 증거 가용성 |
| `doc_type_entropy_20` | `-sum(p_i*log p_i)/log(N_types_present)` | low `<0.35` | 문서 다양성 |
| `gcb_chapter_coverage_10` | top-10 gcb chapter distinct ratio | low `<0.20`, good `>=0.30` | gcb 구조 다양성 |
| `recentness_ratio_180d` | 최근 180일 문서 비율 | stale `<0.10` | tie-break/fallback (초기에는 optional, non-gating) |

state 저장 키(구현 예정):

- `issue_routing_signals`
- `issue_policy_tier`
- `issue_case_refs`
- `issue_case_ref_map`

설계 규칙:

- `K_signal = min(50, len(retrieval_results))`로 계산한다.
- `K_signal < 20`이면 Tier1 진입을 금지하고 Tier2/Tier3만 허용한다.
- telemetry에 `issue_signal_k_effective`를 기록한다.
- `recentness_ratio_180d`는 현재 metadata coverage가 확인되기 전까지 **보조 신호**로만 사용한다.
- Phase 1/2에서는 missing/empty recentness를 이유로 tier를 바꾸지 않는다.
- `issue_case_ref_map`은 `doc_id -> ref_json[]` 매핑을 저장해, case list에 노출된 문서를
  사용자가 선택했을 때 detail 단계가 `answer_ref_json` 축소 결과와 무관하게 같은 문서를
  다시 참조할 수 있도록 한다.
- `issue_case_ref_map`에는 JSON-serializable dict만 저장한다.
  (`RetrievalResult` 같은 객체 저장 금지)
- checkpoint 안정성을 위해 `issue_case_ref_map`은 아래 상한을 둔다:
  - doc_id 최대 10개
  - doc_id당 ref 최대 12개
  - ref 텍스트 필드 최대 800자 (초과 시 절단 + `*_truncated=true` 표기)

metadata 키 계약:

- canonical key: `doc_type`, `section`
- `chapter`만 있는 소스는 ingest/retrieve 단계에서 `section`으로 정규화
- `section` 누락 시 `section="unknown"`으로 처리하고,
  unknown 비율이 30%를 넘으면 `gcb_chapter_coverage_10`은 gating에서 제외한다.

### 7.2 정책 tier

핵심: tier는 ES hard filter를 추가하는 용도가 아니라, **retrieve 이후 refs 선택 정책**을 조정한다.

| Tier | 진입 조건 (초기안) | `issue_case_refs` (max 10) | `answer_ref_json` (max 5) |
| --- | --- | --- | --- |
| Tier 1 (high confidence) | `score_gap_12>=0.12` and `myservice_share_50<=0.70` and `(gcb_count_50>=3 or ts_count_50>=2)` | gcb/ts 우선, myservice `<=2` | myservice `<=1`, gcb/ts 최소 1개(가용 시) |
| Tier 2 (default mixed) | 기본값 (대부분 쿼리) | myservice `<=5`, gcb/ts 가용 시 최소 1개 | myservice `<=2`, 2개 이상 doc_type 확보 시도 |
| Tier 3 (sparse/low confidence) | docs 희소 또는 `myservice_share_50>=0.85` and non-myservice 없음 | 범위 완화, doc_id 다양성 우선 | 상위 점수 유지 + fallback 메시지 |

> **현실 데이터 주의**: §3.4 샘플 쿼리 분석에서 4건 모두 myservice_share >= 0.88이었다.
> §5 Quick-win(특히 Q1 issue_mq, Q2 섹션 병합) 적용 전에는 **대부분 쿼리가 Tier 3으로 진입**하며,
> Tier 1은 거의 발동하지 않을 것으로 예상된다. Quick-win 적용 후 doc_type 분포 변화를
> 재측정하여 threshold를 조정해야 한다.

추가 규칙:

- `issue_case_refs`는 "사용자에게 보여줄 10개 사례"를 위한 노출 세트다.
- `answer_ref_json`은 "요약 answer 생성용 5개 증거" 세트다.
- detail 단계는 `answer_ref_json`을 재사용하지 않고, 반드시 `issue_case_ref_map[selected_doc_id]`
  를 우선 사용한다.
- 즉, `issue_case_refs`에만 있고 `answer_ref_json`에는 없는 문서도 사용자가 선택하면
  동일 `doc_id` 기준 detail answer가 가능해야 한다.

## 8) 코드 훅 설계

### 8.1 유지할 훅

- `route_node` issue override 유지:
  - `backend/llm_infrastructure/llm/langgraph_agent.py:1341`
- post-answer 분기 유지:
  - `backend/services/agents/langgraph_rag_agent.py:658`

### 8.2 변경할 훅

1. 신호 계산 및 tier 결정
   - 위치: `backend/llm_infrastructure/llm/langgraph_agent.py` (`retrieve_node`)
2. diversity-aware ref selection
   - 위치: `backend/llm_infrastructure/llm/langgraph_agent.py` (`expand_related_docs_node` 경로)
3. refs metadata 보강 (issue 모드 한정)
   - 위치: `backend/llm_infrastructure/llm/langgraph_agent.py:961`
   - 추가 후보: `doc_type`, `chapter`, `updated_at`, `policy_tier`
4. case list와 answer refs 분리
   - 위치: `backend/llm_infrastructure/llm/langgraph_agent.py:2743` 부근 issue answer 흐름
   - 목표: `issue_case_refs`(10)와 `answer_ref_json`(5)를 분리 운영
5. detail refs persistence
   - 위치: issue summary 생성 시점 + `issue_detail_answer_node`
   - 목표: `issue_case_ref_map[doc_id]`를 interrupt/resume 동안 유지하고,
     detail 단계가 축소된 `answer_ref_json`이 아니라 선택 문서의 원래 refs를 읽도록 한다

### 8.3 Verification anchor map (code + tests)

- code anchors
  - `backend/llm_infrastructure/llm/langgraph_agent.py`
    - `route_node`, `mq_node`, `retrieve_node`, `expand_related_docs_node`, `answer_node`
    - `issue_case_selection_node`, `issue_detail_answer_node`, `issue_confirm_node`
  - `backend/services/agents/langgraph_rag_agent.py`
    - `_build_graph`, `_after_answer`
- test anchors
  - `backend/tests/test_issue_flow_interrupts.py`
  - `backend/tests/test_langgraph_rag_agent_canonical.py`
  - `tests/api/test_agent_response_metadata_contract.py`
  - `tests/api/test_agent_interrupt_resume_regression.py`
  - `tests/api/test_agent_retrieval_only.py`

## 9) 단계별 롤아웃

Quick-win(Q1~Q6)은 **Phase 0 (behavior change)** 로 취급한다.
즉, Phase 1 Observability baseline은 Quick-win 반영 후 스냅샷을 기준으로 삼는다.

### Phase 1: Observability only

- 신호/티어를 계산하고 로그/metadata에만 기록
- 실제 답변 선택 동작은 기존과 동일하게 유지
- `answer_ref_json`, `issue_case_refs`, detail refs 선택 로직은 절대 변경하지 않는다
- shadow 필드 예:
  - `issue_policy_tier_shadow`
  - `issue_case_refs_shadow`
  - `issue_answer_refs_shadow`

### Phase 2: Shadow policy

- 기존 선택 결과와 새 정책 선택 결과를 side-by-side로 기록
- 오프라인 비교 지표:
  - doc_type diversity
  - gcb/ts 포함 비율
  - answer fallback 발생률
- Phase 2에서도 live path는 기존 `answer_ref_json`을 유지한다
- shadow policy 결과는 metadata/logging으로만 남기고, 사용자 응답/interrupt 결과는 바꾸지 않는다

### Phase 3: Live gating

- Tier2를 기본으로 활성화
- Tier1은 가용성 조건 충족 시에만 활성화
- Tier3 fallback 메시지 활성화

## 10) 검증 계획 (구현 단계에서 실행)

1. unit
   - 신호 계산 경계값 테스트 (`0-hit`, `1-hit`, missing metadata)
2. integration
   - issue 모드에서 tier별 quota 선택 검증
   - case refs(10) / answer refs(5) 분리 검증
   - `issue_case_refs`에는 있으나 `answer_ref_json`에는 없는 문서를 선택해도,
     detail answer가 동일 `doc_id`의 refs를 사용하는지 검증
3. contract regression
   - `uv run pytest tests/api/test_agent_response_metadata_contract.py -v`
   - `uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v`
   - `uv run pytest tests/api/test_agent_retrieval_only.py -v`

## 11) 리스크와 대응

1. **threshold 과민반응으로 recall 하락**
   - 대응: Tier2 default + Tier1 보수적 진입 조건
2. **score scale 불안정성** (rerank on/off)
   - 대응: score 신호를 단독 판단에 쓰지 않고 entropy/share 신호와 결합
3. **metadata payload 비대화**
   - 대응: issue 모드 한정 + allowlist 키만 추가
4. **Quick-win 없이 tier만 도입 시 Tier 3 일변도**
   - §3.4에서 확인된 myservice 편중(85~100%)으로, §5 Quick-win 없이 tier만 도입하면 대부분 Tier 3으로 빠짐
   - 대응: Quick-win(Q1~Q6)을 Phase 1 이전에 적용하고, 적용 후 doc_type 분포를 재측정하여 threshold 조정
5. **gcb parser 코드와 실제 데이터 구조 불일치**
   - `gcb_parser.py`는 question/resolution 구조를 정의하나, 실제 chunk_v3에는 summary/detail/timeline으로 적재됨
   - 대응: 섹션 병합(Q2) 구현 시 실제 데이터 구조(summary/detail/timeline) 기준으로 구현
6. **recentness_ratio_180d 신호의 metadata coverage 불확실**
   - `updated_at` 필드가 모든 doc_type에 존재하는지 미확인
   - 대응: Phase 1에서 coverage를 먼저 측정하고, 불충분하면 gating 신호에서 제외

## 12) 실행 준비도 결론

- 설계 상태: **구현 가능(ready)**

### 실행 우선순위

```
Week 1 — Quick-win (§5, tier와 독립):
  Q1. issue_mq_v2.yaml 프롬프트 추가 + mq_node 분기
  Q3. issue_ans 프롬프트에 섹션 유형 설명 추가
  Q4. MAX_ISSUE_REFS=10 상수 분리
  Q5. issue_ans에 gcb 영문 원본 처리 지시 추가
  Q6. ts 단독 선택 시 ts_ans 경로 사용

Week 2 — Quick-win 계속:
  Q2. expand_related_docs_node에서 동일 doc_id 섹션 자동 병합

Week 3 — Phase 1 (Observability):
  - Quick-win 적용 후 doc_type 분포 재측정
  - 신호/tier 계산 로직 추가 (로그/metadata만, 답변 미변경)
  - recentness_ratio_180d metadata coverage 확인

Week 4 — Phase 2 (Shadow policy):
  - 기존 선택 vs 새 정책 선택 side-by-side 기록
  - threshold 조정 (Quick-win 이후 변화된 분포 반영)

Week 5+ — Phase 3 (Live gating):
  - Tier2 기본 활성화
  - 상세 답변 doc-level fetch 추가 (§4 문제 3 대응)
```

## 13) Implementation checklist

아래 체크리스트는 "문서 구현"이 아니라 실제 코드 착수 시 바로 따라갈 수 있는 순서다.

### 13.1 Step-by-step 파일 단위 작업

1. state schema 확장
   - 파일: `backend/llm_infrastructure/llm/langgraph_agent.py`
   - 작업: `AgentState`에 아래 키 추가
     - `issue_routing_signals: Dict[str, Any]`
     - `issue_policy_tier: str`
     - `issue_case_refs: List[Dict[str, Any]]`
     - `issue_case_ref_map: Dict[str, List[Dict[str, Any]]]`
2. issue 전용 MQ 분기 추가 (Q1)
   - 파일: `backend/llm_infrastructure/llm/langgraph_agent.py` (`mq_node`)
   - 작업: `task_mode == "issue"`일 때 `issue_mq_v2` 프롬프트 사용
3. 섹션 병합 로직 추가 (Q2)
   - 파일: `backend/llm_infrastructure/llm/langgraph_agent.py` (`expand_related_docs_node`)
   - 작업: hit 문서의 `doc_id` 기준으로 동일 문서 섹션을 합쳐 `issue_case_ref_map` 구성
4. answer/case refs 분리 (Q4)
   - 파일: `backend/llm_infrastructure/llm/langgraph_agent.py` (`answer_node`)
   - 작업: `issue_case_refs(max=10)`와 `answer_ref_json(max=5)`를 분리해 생성
5. detail 단계 refs 소스 고정
   - 파일: `backend/llm_infrastructure/llm/langgraph_agent.py` (`issue_detail_answer_node`)
   - 작업: `selected_doc_id`가 있으면 `issue_case_ref_map[selected_doc_id]`를 우선 사용
6. interrupt/resume 보존 확인
   - 파일: `backend/services/agents/langgraph_rag_agent.py`
   - 작업: `issue_case_ref_map` 포함 상태가 checkpointer 경로에서 유지되는지 확인

### 13.2 Done 정의 (구현 완료 기준)

- D1. issue summary 응답이 `issue_top10_cases` 10개를 안정적으로 노출
- D2. summary용 refs 5개 제한이 detail 선택 동작을 깨지 않음
- D3. `issue_case_refs`에만 있던 문서를 선택해도 detail 응답 생성
- D4. `task_mode=issue`에서 setup prompt로 역전되는 경로가 없음
- D5. `C-API-001`, `C-API-002`, `C-API-003` 회귀 없음

## 14) Telemetry schema

Phase 1/2에서 최소 아래 필드를 metadata/log에 남겨야 threshold 조정이 가능하다.

| 필드 | 타입 | 설명 | 사용 단계 |
| --- | --- | --- | --- |
| `issue_policy_tier_shadow` | string | shadow tier (`tier1|tier2|tier3`) | Phase 1/2 |
| `issue_signal_score_gap_12` | float | 상위 1-2위 score gap | Phase 1/2/3 |
| `issue_signal_myservice_share_50` | float | top-50 myservice 비율 | Phase 1/2/3 |
| `issue_signal_gcb_count_50` | int | top-50 내 gcb 개수 | Phase 1/2/3 |
| `issue_signal_ts_count_50` | int | top-50 내 ts 개수 | Phase 1/2/3 |
| `issue_signal_doc_type_entropy_20` | float | top-20 doc_type entropy | Phase 1/2/3 |
| `issue_signal_non_myservice_presence_50` | int | top-50 내 (gcb+ts) 합계 | Phase 1/2/3 |
| `issue_case_count` | int | 사용자 노출 사례 수 | Phase 2/3 |
| `issue_answer_ref_count` | int | summary answer refs 수 | Phase 2/3 |
| `issue_detail_ref_source` | string | `case_ref_map|answer_ref_json|fallback` | Phase 2/3 |
| `issue_fallback_reason` | string | Tier3 fallback/empty 사유 | Phase 2/3 |

로그 수집 규칙:

1. PII/원문 긴 텍스트는 로그에 남기지 않고 count/ratio만 저장
2. 신호값은 query당 1회 기록 (중복 기록 금지)
3. Phase 1/2에서는 사용자 응답 변경 없이 telemetry만 수집

## 15) Go/No-Go gates

측정 절차 고정:

- dataset: `data/paper_a/eval/query_gold_master_v0_6_generated_full.jsonl`
- 비교 기준:
  - contamination/gold hit: issue evaluation harness 출력
  - fallback: runtime telemetry (`issue_detail_ref_source`) 집계
- 모든 gate 판정은 동일 dataset + 동일 top-k 설정에서만 수행

### 15.1 Phase 전환 게이트

- Gate A (Phase 1 -> 2)
  - telemetry 누락률 `< 1%`
  - `issue_policy_tier_shadow`가 전체 issue 쿼리의 `>= 95%`에서 생성
- Gate B (Phase 2 -> 3)
  - shadow 대비 live 예상 contamination 악화 없음 (`delta <= +0.02`)
  - shadow 대비 gold hit 감소 허용치 이내 (`delta >= -0.03`)
  - `issue_detail_ref_source=fallback` 비율 `<= 5%`

### 15.2 운영 중 롤백 트리거

아래 중 하나라도 충족하면 live gating을 즉시 끄고 Phase 2(shadow)로 회귀:

1. `C-API-001` 회귀 실패
2. `C-API-002` 회귀 실패
3. `C-API-003` 회귀 실패
4. issue empty 응답 비율이 기준 대비 `+5%p` 이상 증가
5. 특정 doc_type만 반복 노출되는 편향이 급증 (`myservice_share_50` 평균 `>= 0.95`로 고착)

## 16) Risk register

| Risk ID | 리스크 | 영향 | 탐지 신호 | 완화 |
| --- | --- | --- | --- | --- |
| R-01 | Tier1 발동 희소 | gcb/ts 활용 개선 미미 | Tier1 비율 `< 2%` | Q1/Q2 선적용 후 threshold 재보정 |
| R-02 | summary/detail refs 분리 부작용 | 선택 후 detail 실패 | `issue_detail_ref_source=fallback` 급증 | `issue_case_ref_map` 강제 우선 |
| R-03 | metadata 확장으로 payload 증가 | 응답 지연/크기 증가 | 평균 metadata size 증가 | issue 모드 한정 allowlist 유지 |
| R-04 | parser 불안정으로 Tier 오판 | 잘못된 문서군 집중 | parser-no-detection 비율 증가 | parser 지표를 telemetry에 포함 |
| R-05 | 과도한 hard-like 동작 회귀 | recall 저하 | gold hit 하락 | Tier2 default + quota 방식 고정 |

## 17) Threshold calibration workflow

myservice 편중 코퍼스에서 고정 임계값 하나로 운영하면 drift에 취약하다. 아래 순서로 주기적으로 재보정한다.

1. 데이터 수집 (주 단위)
   - `issue_signal_*` telemetry에서 아래 분포를 저장:
     - `score_gap_12` p25/p50/p75
     - `myservice_share_50` p25/p50/p75
     - `doc_type_entropy_20` p25/p50/p75
2. shadow 재평가 (격주)
   - 동일 query set에서 baseline vs shadow를 비교:
     - contamination delta (`raw/adjusted`)
     - gold hit delta
     - detail fallback 비율
3. 임계값 조정 규칙
   - `myservice_share_50` p50이 0.90 이상으로 고착되면 Tier1 진입 조건 완화 대신
     Q1/Q2(검색입력/섹션병합) 품질을 먼저 점검한다.
   - `score_gap_12` 분산이 작아지고 평탄해지면(`p75-p25 < 0.04`) gap 단독 신뢰도를 낮추고
     entropy/share 복합 조건의 가중치를 높인다.
4. stop/go 판정
   - Go: contamination 개선(`<= -0.02`) + hit 하락 제한(`>= -0.03`) 동시 만족
   - Stop: hit 하락이 연속 2회 임계 초과 또는 detail fallback 증가(`> +0.05`)

## 18) Reporting governance (citation-backed)

아래 원칙은 metric revision 시 보고 품질을 유지하기 위한 최소 규칙이다.

1. p-value 단독 보고 금지, 효과크기(effect size)와 함께 보고
2. 주요 지표는 95% CI와 함께 보고
3. oracle vs realistic(parser) 결과를 분리 보고
4. old vs new baseline 수치를 같은 표에서 동시 제시
5. reproducibility metadata(데이터 버전, 실행 명령, seed, 환경) 기록
6. contamination audit 지표를 quality 지표와 분리해 병렬 보고

참고 문헌/가이드:

- Fuhr, "Some Common Mistakes in IR Evaluation" (SIGIR Forum, 2018)
- Sakai, "Statistical Reform in Information Retrieval?" (SIGIR Forum, 2014)
- Thakur et al., "BEIR" (NeurIPS, 2021)
- ACM SIGIR Artifact Badging guidance

---

## Appendix A) live ES 조회 재현 커맨드

아래 형식으로 `uv run python`에서 ES `_count`/`_search` 집계를 호출해
doc_type 분포, chapter 분포, 샘플 쿼리 top-50 mix를 재현할 수 있다.

```python
from backend.config.settings import search_settings
# host = search_settings.es_host
# index = search_settings.v3_content_index
# requests로 _count / _search(terms agg, query sample) 실행
```

실제 수치는 운영 데이터 증감으로 변할 수 있으므로,
구현 전/배포 전 동일 스크립트로 최신 스냅샷을 다시 확보한다.
