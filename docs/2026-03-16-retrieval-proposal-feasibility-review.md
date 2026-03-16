# Retrieval 개선 제안서 타당성 검토

> **작성일**: 2026-03-16
> **목적**: 12개 섹션으로 구성된 retrieval 재설계 제안서를 현재 시스템 상태 기준으로 검증
> **결론**: 즉시 적용 가능한 항목 4개, 선행 작업 필요 3개, 대규모 변경 필요 3개

---

## 1. 현재 시스템 요약 (검토 기준)

### 1.1 ES 인덱스 구조

| 항목 | 현재 상태 |
|------|-----------|
| 인덱스 분리 | `chunk_v3_content` (텍스트+메타) + `chunk_v3_embed_{model}_v1` (벡터) |
| 텍스트 필드 | 단일 `content` + `search_text` (Nori 분석기) |
| KO/EN 분리 | **없음** — `lang` 필드는 keyword 타입 (필터용) |
| 분석기 | `nori_tokenizer` + `nori_readingform` + `lowercase` |
| 메타데이터 필드 | `doc_type`, `device_name`, `equip_id`, `chapter`, `section_chapter`, `section_number`, `chapter_source`, `chapter_ok` (모두 keyword) |
| 벡터 | HNSW, cosine similarity, dims 768 (BGE-M3) |
| dynamic | `false` — 명시적 mapping 변경 없이 필드 추가 불가 |

### 1.2 청크 통계

| doc_type | 청크 수 | 비율 | 평균 길이 | 특징 |
|----------|---------|------|-----------|------|
| myservice | 329,206 | 83% | median 58자 | 4섹션(status/cause/action/result) 분리 저장 |
| gcb | 49,021 | 12.4% | avg 919자 | summary + detail 청크, `chunk_tier` extra_meta |
| sop | 13,116 | 3.3% | avg 879자 | VLM 파싱, 페이지 단위 |
| setup | 3,583 | 0.9% | - | VLM 파싱, 페이지 단위 |
| ts | 760 | 0.2% | avg 748자 | VLM 파싱, heading 기반 |

### 1.3 Retrieval 파이프라인

```
Multi-Query 확장 (EN 3 + KO 3, 최대 6개)
  ↓
Retriever.retrieve() × 각 쿼리 (BM25 + Dense → RRF)
  ↓
결과 합산 (RRF k=60)
  ↓
Noisy chunk 필터링
  ↓
Cross-encoder reranking (선택)
  ↓
doc_type / equip_id 사후 필터링
  ↓
Section expansion (상위 2 그룹)
  ↓
SOP soft boost / early page penalty
  ↓
최종 top_k 제한
```

### 1.4 LangGraph Agent 플로우

```
route_node → device_selection → mq_node → st_gate → st_mq →
retrieve_node → expand_related → answer_node → judge
```

- backend 내부 `task_mode` 종류: `sop`, `ts`, `issue`, (미지정)
- `route` 종류: `setup`, `ts`, `general`
- `task_mode="issue"` → `route="general"` 고정, issue 전용 answer/confirm 플로우
- `task_mode="ts"` → `route="ts"` 고정, TS 전용 MQ/answer 경로
- API/프론트 public contract는 아직 `sop`, `issue`, `all`을 기준으로 동작함

### 1.5 이번 검토에서 보정해야 할 전제

제안서의 방향성은 강하지만, 아래 5가지는 "현재 상태 설명"과 "목표 구조 제안"이 혼재되어 있어 보정이 필요하다.

1. **"2개 UX route 유지"는 현재 상태가 아니라 목표 상태에 가깝다**
   - 현재 public contract는 `issue / sop / all`이고, backend 내부에는 `ts` 특례도 존재한다.
2. **weighted fusion은 완전히 없는 것이 아니다**
   - dense/sparse hybrid에는 이미 list-wise weighted RRF가 구현돼 있다.
   - 아직 없는 것은 `doc_type branch`별 query-conditioned weighting이다.
3. **issue 쪽은 완전 flat retrieval이 아니다**
   - issue 전용 MQ, same-doc expansion, myservice cap, non-myservice 노출 보정이 일부 live 코드에 들어가 있다.
4. **SOP/setup은 이미 section-first에 일부 가깝다**
   - page-window는 기본 전략이라기보다 section fetch 실패 시 fallback에 가깝다.
5. **baseline 표기는 production과 experiment가 섞여 있다**
   - 일부 문서와 실험은 BGE-M3 1024-dim을 기준으로 하지만, runtime 기본 설정은 `bge_base`다.

---

## 2. 섹션별 타당성 평가

### §1. Branch Retrieval (canonical group별 분기 검색)

**타당성: ★★★★☆ (높음)**

| 항목 | 현재 상태 |
|------|-----------|
| doc_type 필터 | `build_filter(doc_types=["myservice"])` — ES 쿼리 레벨 지원 ✅ |
| 분기 검색 인프라 | `retrieve_node`는 단일 호출 — 분기 로직 부재 ❌ |
| 결과 합산 | RRF 함수 `merge_retrieval_result_lists_rrf()` 재사용 가능 ✅ |
| weighted fusion | dense/sparse hybrid에 list-wise weighted RRF 이미 존재 ✅ |

**즉시 가능:**
- `retrieve_node` 내에서 `task_mode`별로 `doc_types` 필터를 달리한 복수 검색 호출 후 RRF 합산
- preset YAML에 doc_type별 `top_k`, `weight` 파라미터 추가

**보정 포인트:**
- 제안서의 "Elastic 기본 RRF가 equal-weight라 application-layer weighted fusion이 필요"라는 문제의식은 맞지만,
  이 repo에는 이미 dense/sparse용 weighted RRF가 구현돼 있다.
- 따라서 실제 gap은 "weighted fusion 일반"이 아니라,
  **`myservice/gcb/ts` branch별 query-conditioned weighted fusion** 쪽이다.

**필요한 변경:**
- `retrieve_node` 리팩토링 (현재 ~500줄 단일 함수 → 분기별 서브함수 분리)
- doc_type별 quota 보장 로직 (현재 RRF는 전체 통합 순위만 산출)

---

### §2. Query Profiler

**타당성: ★★★☆☆ (중간)**

| 항목 | 현재 상태 |
|------|-----------|
| device 추출 | `auto_parse_node` → `_extract_devices_from_query()` ✅ |
| doc_type 추출 | `auto_parse_node` → `_extract_doc_types_from_query()` ✅ |
| 언어 감지 | rule-based CJK 문자 감지 ✅ |
| route 확률 | LLM 분류 (temperature=0.0), logprob 미활용 ❌ |
| alarm 코드 추출 | TS chunking에 `alarm_code` 필드 존재, 쿼리에서는 미추출 ❌ |
| doc_type_prior | 없음 ❌ |

**즉시 가능:**
- vLLM의 logprob 응답에서 route confidence 추출 (vLLM API 지원)
- alarm 코드 regex 추출 → `auto_parse_node` 확장

**주의점:**
- Query profiler를 **별도 LLM 호출 노드**로 추가하면 latency 증가
- `auto_parse_node` 확장이 더 현실적 (이미 device/doc_type/lang 추출 수행 중)

---

### §3. myservice Bundle Retrieval + Collapse

**타당성: ★★★★☆ (높음) — 단, 핵심 가정에 괴리**

| 항목 | 현재 상태 |
|------|-----------|
| 4섹션 분리 저장 | `chapter` 필드에 section명 (status/cause/action/result) ✅ |
| same-doc fetch | `expand_related` → `DOC_TYPES_SAME_DOC`에 myservice 포함 ✅ |
| doc_id 기준 dedup | `retrieve_node`에 base doc_id 중복 제거 로직 존재 ✅ |
| **case_cluster_id** | **존재하지 않음** ❌ — 원시 데이터에 해당 필드 없음 |

**현실과 제안서의 괴리:**

제안서의 핵심 개념인 `case_cluster_id` collapse는 **데이터에 해당 필드가 없어 즉시 적용 불가**.
myservice 원시 데이터의 메타데이터 필드:
```
Order No., Title, Model Name, Equip_ID, Activity Type,
Country, Reception Date, Customer Name, Warranty, completeness
```

→ 유사 케이스 클러스터링을 위해서는 **Title 유사도 기반 클러스터링 파이프라인을 새로 구축**해야 함.

**이미 구현된 부분:**
- Bundle retrieval: `expand_related`의 same-doc fetch가 사실상 bundle 역할 (한 섹션 hit → 전체 4섹션 가져옴)
- doc_id 기준 collapse: `retrieve_node`에서 base doc_id별 최고 점수 청크만 유지

**보정 포인트:**
- 제안서가 말하는 bundle retrieval은 여전히 타당하지만,
  "현재가 완전히 chunk-only"인 것은 아니다.
- 이미 post-retrieve 단계에서 same-doc bundle 성격의 확장이 일부 동작하므로,
  여기서의 핵심 차이는 **후처리 bundle**을 **1차 검색 unit bundle**로 끌어올릴지 여부다.

**추가로 가능한 개선:**
- `search_text` 개선: 현재 `title + cause + status + content`
  → 4섹션 통합 search_text를 인덱스 필드로 별도 저장하면 BM25 매칭 개선 가능
- myservice 전용 검색 시 `chapter` 필터 없이 doc_id 그룹핑 → 1 doc = 1 결과 단위

---

### §4. GCB KO/EN Dual-Field + Doc Bundle 2-Tier

**타당성: ★★★☆☆ (중간) — ES 인덱스 재구축 필요**

| 항목 | 현재 상태 |
|------|-----------|
| KO/EN 분리 필드 | **없음** — 단일 `content` (Nori 분석기) ❌ |
| chunk_tier | `extra_meta.chunk_tier = "summary"/"detail"` 존재, **ES 미인덱싱** ⚠️ |
| chapter 필드 | summary 청크는 `chapter="summary"` (keyword, 인덱싱됨) ✅ |
| summary→detail 확장 | `expand_related`에서 gcb는 `DOC_TYPES_SAME_DOC` → same-doc fetch ✅ |

**즉시 가능 (인덱스 변경 없이):**
- Summary→Detail 2-tier 검색:
  1. `chapter=="summary"` + `doc_type=="gcb"` 필터로 1차 검색
  2. hit된 doc_id로 detail 청크 가져오기 (expand_related 이미 수행)
- 이미 사실상 동작하는 구조이나, **1차 검색을 summary로 한정하는 분기**만 추가하면 됨

**보정 포인트:**
- 제안서의 bilingual dual-field는 분명 유효한 방향이지만, 이것은 retrieval 설정 변경이 아니라
  **mapping 변경 + 재인덱싱 + 언어 분리 전처리**가 포함된 중기 과제다.
- 반면 `summary → detail` 2-tier 자체는 현재 구조 위에서도 빠르게 검증 가능하다.

**인덱스 재구축 필요:**
- `content_ko`, `content_en` 듀얼 필드:
  - mapping 변경 (`dynamic: false`이므로 명시적 추가 필요)
  - 전체 재인덱싱 (gcb 49K 청크 → 비용 낮음)
  - 영어 전용 분석기 추가 (standard 또는 english analyzer)
- `chunk_tier`를 keyword 필드로 승격:
  - mapping에 `chunk_tier: keyword` 추가 → 재인덱싱

**주의점:**
- mapping 변경은 `chunk_v3_content` **전체 인덱스에 영향** (doc_type별 분리 인덱스 아님)
- GCB 원문이 영어/한국어 혼재 → 언어 감지 후 해당 필드에 저장하는 전처리 필요

---

### §5. TS as Anchor Corpus (Reserved Quota)

**타당성: ★★★★★ (매우 높음) — 가장 구현 용이**

| 항목 | 현재 상태 |
|------|-----------|
| TS 청크 수 | 760개 (전체의 0.2%) ✅ |
| doc_type 필터 | `doc_types=["ts"]` 검색 가능 ✅ |
| TS 전용 route | `route=="ts"` → TS 전용 MQ 템플릿 사용 ✅ |
| 별도 검색 호출 | 현재는 전체 통합 검색만 ❌ |

**즉시 구현 가능:**
```
retrieve_node 수정:
  if task_mode in ("issue", "ts") or route == "general":
      ts_results = retriever.retrieve(query, doc_types=["ts"], top_k=5)
      main_results = retriever.retrieve(query, doc_types=제외ts, top_k=N)
      merged = merge_with_quota(main_results, ts_results, ts_slots=3)
```

- 760 청크이므로 검색 비용 무시 가능
- TS 결과에서 최소 N개 보장 슬롯 배정

---

### §6. SOP/Setup Section/Step/Table Retrieval Units

**타당성: ★★★☆☆ (중간) — 데이터 구조 일부 존재, retrieval unit 재정의 필요**

| 항목 | 현재 상태 |
|------|-----------|
| section_chapter | keyword 필드, 인덱싱됨 ✅ |
| section_number | integer 필드 ✅ |
| section_expander | `fetch_section_chunks(doc_id, section_chapter, max_pages=8)` ✅ |
| heading 감지 | STEP\d+, 목적/범위/절차 등 키워드 감지 ✅ |
| logical_table unit | **없음** — 페이지 단위 청킹, 테이블 미분리 ❌ |
| manual_step_group | 개념 없음 — heading 감지만 존재 ❌ |

**즉시 가능:**
- Section 단위 그룹 검색: `section_chapter` 필드 기반 (이미 expand_related에서 활용)
- Step 연속성: `section_number` 필드로 연속 step 묶기

**보정 포인트:**
- 제안서의 "page retrieval을 버리고"는 현재 코드 기준으로는 표현이 너무 강하다.
- 실제로는 이미 section-first expansion이 우선이고, page-window는 fallback에 가깝다.
- 따라서 더 정확한 표현은 **"page fallback을 낮추고 section/step/table unit을 1급 retrieval unit으로 승격"** 이다.

**청킹 파이프라인 변경 필요:**
- `logical_table`: VLM이 마크다운으로 변환한 테이블을 별도 청크로 분리
  - 현재 테이블은 페이지 내 텍스트에 포함 → 추출 로직 추가 필요
- `manual_step_group`: heading 기반 step 그룹핑 단위 정의 + 청킹 로직 변경

---

### §7. Summary Rerank for Manuals

**타당성: ★★★★☆ (높음)**

| 항목 | 현재 상태 |
|------|-----------|
| Cross-encoder reranker | 통합됨 (ms-marco-MiniLM-L-6-v2) ✅ |
| chunk_summary 필드 | text 타입, standard 분석기 ✅ |
| reranker 입력 설정 | 현재 `content` 필드 사용 ✅ |

**즉시 가능:**
- Reranking 시 `chunk_summary`를 입력으로 사용하는 옵션 추가
- Two-stage rerank:
  1. `chunk_summary`로 coarse rerank (빠른 필터링)
  2. `content`로 fine rerank (정밀 순위)

**변경 범위:** reranker adapter에 `input_field` 파라미터 추가 (소규모)

---

### §8. Multimodal Fallback (Flowchart/Diagram)

**타당성: ★☆☆☆☆ (낮음) — 인프라 미비**

| 항목 | 현재 상태 |
|------|-----------|
| 이미지 저장 | MinIO에 저장 가능 ✅ |
| page_image_path | 필드 존재, stored only (미인덱싱) ⚠️ |
| 이미지 임베딩 | **없음** ❌ |
| 멀티모달 모델 | **미설정** ❌ |
| flowchart 감지 | **없음** ❌ |

**필요한 대규모 변경:**
- 이미지 임베딩 파이프라인 (CLIP 등) 신규 구축
- 별도 벡터 인덱스 (이미지용)
- Flowchart/diagram 감지 분류기
- 검색 시스템 ↔ 이미지 저장소 연동

→ **3순위 분류가 적절**. 현재 텍스트 기반 retrieval 개선이 우선.

---

### §9. Concrete Retrieval Flows (Issue/Procedure)

**타당성: ★★★★☆ (높음) — 현재 아키텍처의 자연스러운 확장**

| 항목 | 현재 상태 |
|------|-----------|
| Issue flow | `task_mode="issue"` → general route → issue-specific answer ✅ |
| Procedure flow | `route="setup"` → setup MQ → general retrieval ✅ |
| doc_type별 분기 검색 | 미구현 — 단일 통합 검색 ❌ |

**제안서 Issue flow vs 현재:**

```
[제안서]                              [현재]
myservice_bundle 검색                  단일 통합 검색
  + gcb_doc_search                     (모든 doc_type 혼합)
  + ts_anchor                          → 사후 doc_type 필터링
  → weighted merge                     → RRF 합산
```

**구현 방향:**
- `retrieve_node`에서 `task_mode`별 분기 추가
- 각 분기에서 `doc_types` 필터 적용하여 별도 검색
- §1(branch)과 §5(ts quota)를 결합하면 자연스럽게 §9의 구조가 됨

---

### §10. Lightweight LTR (Learning to Rank)

**타당성: ★★☆☆☆ (낮음-중간) — 학습 데이터 및 인프라 부족**

| 항목 | 현재 상태 |
|------|-----------|
| 학습 데이터 | `reranker_pairs.jsonl` 494쌍 (LTR에 부족) ⚠️ |
| 스코프 라벨 | `document_scope_table.csv` 428개 ✅ |
| LTR 인프라 | ES LTR 플러그인 미설치 ❌ |
| Cross-encoder | ms-marco 기본 모델 사용 중 ✅ |

**현실적 대안:**
- LTR보다 **cross-encoder fine-tuning**이 더 현실적
  - 기존 reranker 인프라 재사용
  - 494쌍 + gold_master 1,206개로 학습 데이터 확장 가능
- Weighted RRF 파라미터 튜닝 (grid search)이 즉시 가능한 대안

**주의점:**
- 기존 평가 자산은 gold set collapse 문제가 있었던 이력이 있어,
  작은 LTR를 바로 얹는 경우 학습 신호 오염 가능성을 먼저 점검해야 한다.
- 따라서 LTR은 `v0.6+`, strict subset, 또는 human relabel slice 기준으로 시작하는 편이 안전하다.

---

### §11. Priority Ordering 검증

**제안서 우선순위 vs 현재 시스템 기준 재평가:**

| 순위 | 제안 항목 | 현실 타당성 | 비고 |
|------|-----------|-------------|------|
| **1순위** | branch pool | ★★★★ | doc_types 필터 존재, retrieve_node 분기만 추가 |
| **1순위** | bundle (myservice) | ★★★ | same-doc fetch 이미 있으나 **case_cluster_id 없음** |
| **1순위** | bilingual field | ★★ | **재인덱싱 필수** — mapping 변경 필요 |
| **1순위** | ts quota | ★★★★★ | 760 청크, 변경 최소 |
| **2순위** | section_type | ★★★ | section_chapter 필드 있으나 table 미분리 |
| **2순위** | table stitching | ★★ | 청킹 파이프라인 변경 필요 |
| **3순위** | dual dense | ★★ | 별도 임베딩 모델 + 인덱스 필요 |
| **3순위** | multimodal | ★ | 이미지 파이프라인 전무 |
| **3순위** | reranker fine-tune | ★★★ | 494쌍 부족하나 gold_master로 확장 가능 |

**재조정 권장 순위:**

```
[즉시 시작] (1-2주)
  1. TS reserved quota (§5)         — 변경 최소, 검증 용이
  2. Branch retrieval (§1)          — retrieve_node 분기 추가
  3. GCB summary→detail 2-tier (§4) — chapter=="summary" 필터 활용

[단기] (2-4주)
  4. Summary rerank (§7)            — reranker input_field 확장
  5. Concrete flows (§9)            — §1+§5 결합
  6. Section 그룹 검색 (§6 일부)    — section_chapter 기반

[중기 — 선행 작업 필요] (1-2개월)
  7. myservice case_cluster_id      — 클러스터링 파이프라인 구축
  8. KO/EN dual-field (§4)          — ES mapping 변경 + 재인덱싱
  9. Logical table unit (§6)        — 청킹 파이프라인 변경

[장기] (2개월+)
  10. Cross-encoder fine-tuning     — 학습 데이터 확장 후
  11. Multimodal fallback (§8)      — 이미지 인프라 구축
```

---

### §12. New Evaluation Metrics

**타당성: ★★★★☆ (높음)**

| 제안 메트릭 | 계산 가능 여부 | 비고 |
|-------------|---------------|------|
| group_exposure@20 | ✅ | 검색 결과의 `doc_type` 메타데이터 포함 |
| unique_doc_id@20 | ✅ | doc_id 고유 수 즉시 계산 |
| myservice_case_cluster_dup_ratio | ⚠️ | **case_cluster_id 없음** → doc_id 기준 대체 가능 |
| task_mode별 분리 평가 | ✅ | `eval_chatflow_unified.jsonl`에 expected_task_mode 존재 |

**현재 eval 데이터로 즉시 추가 가능한 메트릭:**
- `doc_type_distribution@K`: 상위 K개 결과의 doc_type 분포
- `unique_doc_count@K`: 상위 K개 결과의 고유 문서 수
- `task_mode_accuracy`: 라우팅 정확도 (expected vs actual)
- Per-scope hit rate: scope_observability별 recall

---

## 3. 핵심 괴리 요약

### 3.1 case_cluster_id 부재

제안서의 myservice bundle collapse 핵심 개념인 `case_cluster_id`가 **원시 데이터에 존재하지 않음**.

- myservice 메타데이터: `Order No., Title, Model Name, Equip_ID, Activity Type` 등
- 유사 케이스 그룹핑을 위해서는 Title 기반 클러스터링 파이프라인 신규 구축 필요
- 대안: `doc_id` 기준 collapse는 이미 구현됨 (단, 유사 케이스 간 중복 제거는 불가)

### 3.2 단일 content 필드

- 현재: 단일 `content` 필드 (Nori 분석기)
- 제안: `content_ko` + `content_en` 듀얼 필드
- GCB 원문이 영어/한국어 혼재 → 언어별 분리 저장 시 전처리 파이프라인 추가 필요
- `dynamic: false` 설정 → mapping 변경 시 인덱스 재생성 필수

### 3.3 myservice bundle이 이미 부분 구현

- `expand_related`의 `DOC_TYPES_SAME_DOC`에 myservice 포함
- 한 섹션 hit → 같은 doc_id의 전체 4섹션 자동 가져옴
- 제안서의 bundle 개념은 **검색 단계에서의 통합**이 핵심 (사후 확장이 아닌 검색 시점)

### 3.4 "현재는 2-route"라는 설명의 부정확성

- 목표 방향으로는 `issue / procedure` 2-route가 이해하기 쉽다.
- 하지만 현재 시스템은 public contract에서 `all`을 유지하고,
  backend 내부에서는 `ts` 특례 route도 존재한다.
- 따라서 제안서에서 2-route를 말할 때는 **"현재 설명"이 아니라 "목표 구조"** 로 명시해야 한다.

### 3.5 weighted fusion은 이미 부분 구현

- 현재 hybrid retriever는 dense/sparse 결과 병합에서 list-wise weighted RRF를 이미 사용한다.
- 따라서 branch retrieval 설계의 핵심 차별점은
  `myservice/gcb/ts/manual` branch별 score prior와 quota를 query-conditioned하게 줄 수 있느냐이다.

### 3.6 production baseline과 experiment baseline 혼재

- 일부 실험 문서는 BGE-M3 1024-dim을 baseline처럼 설명한다.
- 반면 runtime 기본 설정은 `bge_base`다.
- 제안서가 "임베딩보다 retrieval unit이 더 중요하다"는 결론을 내리는 것은 타당할 수 있지만,
  이 결론을 쓸 때는 **production baseline / experiment baseline을 분리 표기** 해야 한다.

---

## 4. 결론

### 제안서의 강점
- doc_type별 특성을 반영한 분기 검색 설계가 현재 아키텍처(Engine-Adapter-Registry + preset)에 잘 맞음
- TS quota, branch retrieval 등 즉시 적용 가능한 실용적 제안 포함
- 평가 메트릭 제안이 현재 eval 데이터셋과 정합

### 제안서의 한계
- `case_cluster_id` 등 **존재하지 않는 필드를 가정**한 설계가 일부 포함
- KO/EN 듀얼 필드는 **인덱스 재구축**이 전제 — 1순위로 분류하기에 선행 작업 부담 큼
- Multimodal, LTR 등은 현재 인프라 대비 과도한 도약
- 현재를 "완전 flat retrieval"이나 "이미 2-route"로 설명하는 부분은 정확도가 떨어짐
- weighted fusion이 전혀 없다는 서술은 현재 코드와 다름

### 권장 접근
1. **현재 이미 들어간 quick-win 위에 다음 단계를 얹는다**
   - issue MQ, same-doc expansion, myservice cap 등 기존 대응을 baseline으로 명시
2. **TS quota + Branch retrieval**로 시작하여 doc_type별 분기 검색 프레임워크 확립
3. **GCB summary 2-tier**로 doc→section 확장 패턴 검증
4. 이후에만 인덱스 재구축 과제(KO/EN dual-field, logical table, bundle field)를 순차 진행

### 최종 판단

이 제안서는 방향성 자체는 상당히 타당하다.
특히 "더 좋은 임베딩"보다 **doc_type-aware candidate exposure, retrieval unit 재설계, query-conditioned fusion**이 우선이라는 진단은 현재 시스템과 잘 맞는다.

다만 이 문서를 바로 실행계획으로 쓰기 위해서는 아래 3가지를 먼저 명확히 해야 한다.

1. 현재 상태와 목표 상태를 분리해서 서술할 것
2. 이미 구현된 대응(issue quick-win, weighted fusion, section-first expansion)을 baseline으로 포함할 것
3. 재색인/전처리/신규 스키마가 필요한 과제를 별도 phase로 분리할 것
