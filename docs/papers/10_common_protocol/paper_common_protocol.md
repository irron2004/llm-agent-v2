# [Role: 공통 규격] Paper Common Protocol — Golden Set · Baseline · 평가 하네스

> 작성일: 2026-02-28
> 적용 범위: Paper A / Paper B / Paper C / Journal J 공통
> [Role: 실험 공통 규격 / Golden Set 스키마 / Baseline 정의 / 평가 프로토콜]

---

## 1. Golden Set 설계

### 1.1 스키마 (JSONL)

```jsonl
{
  "qid": "GS-001",
  "category": "troubleshooting",
  "canonical_query": "XXX 장비 YYY 에러 원인과 조치 방법",
  "paraphrases": [
    "XXX에서 YYY 에러 뜨면 어떻게 해?",
    "YYY error on XXX equipment, what to do?",
    "XXX 장비 YYY 코드 해결 방법 알려줘"
  ],
  "allowed_scope": {
    "equipment_group": "XXX",
    "module": ["ZZZ"],
    "doc_type": ["SOP", "Troubleshooting Guide"]
  },
  "gold_evidence": [
    {"doc_id": "DOC-123", "version": "v7", "passage_id": "DOC-123#p034", "relevance": "primary"},
    {"doc_id": "DOC-456", "version": "v7", "passage_id": "DOC-456#p012", "relevance": "supporting"}
  ],
  "gold_answer": {
    "summary": "기대 정답 요약",
    "numeric_constraints": [
      {"field": "temperature", "value": 150, "unit": "°C", "tolerance": 5}
    ]
  },
  "tags": ["numeric", "multi-doc", "scope-critical"],
  "paper_subset": ["B:paraphrase", "A:contamination"],
  "difficulty": "medium",
  "created_by": "annotator_id",
  "reviewed_by": "reviewer_id",
  "version": "v0"
}
```

### 1.2 필수 필드 설명

| 필드 | 타입 | 설명 |
|---|---|---|
| `qid` | string | 고유 질문 ID (GS-NNN) |
| `category` | string | troubleshooting / specification / procedure / safety |
| `canonical_query` | string | 대표 질의문 |
| `paraphrases` | string[] | 동의 질의문 3-5개 (한/영 혼합 권장) |
| `allowed_scope` | object | 허용 장비군/모듈/문서유형 |
| `gold_evidence` | object[] | 정답 근거 문서 (doc_id + version + passage_id) |
| `gold_answer` | object | 기대 정답 요약 + 수치 제약 |
| `tags` | string[] | 난이도/유형 태그 |
| `paper_subset` | string[] | Paper별 서브셋 소속 |
| `difficulty` | string | easy / medium / hard |

> **근거 라벨 단위**: 문서 단위보다 `passage_id` 우선 (예: `DOC-123#p034`).
> `version` 필드는 Paper C(라이프사이클 통제) 실험에서 필수.

### 1.3 크기 계획

| 버전 | 건수 | 용도 | 시기 |
|---|---|---|---|
| v0 (pilot) | 50건 | 스키마 검증 + 라벨링 워크플로 시험 | Stage 0 |
| v0.5 | 150-250건 | Paper B 안정성 실험 + Baseline 측정 | Stage 1 |
| v1 | 400-800건 | 전체 Paper 실험 + Journal J 종합 | Stage 2-3 |
| v2/v3 | v1 + 업데이트 | Paper C 운영통제/Q1 저널급 | Stage 4+ |

---

## 2. 데이터 소스 (S1-S5)

| 소스 | 설명 | 예상 건수 | 활용 |
|---|---|---|---|
| **S1: 매뉴얼/SOP** | 장비별 운영/정비 매뉴얼, PM 체크리스트 | 문서 수백 건 | gold_evidence 후보 |
| **S2: 정비이력** | 실제 정비 기록 (trouble ticket) | 수천 건 | canonical_query 추출 |
| **S3: Q&A 스레드** | 연구소-Fab Q&A, 내부 게시판/채팅 | 수백 건 | paraphrase 자연 발생원 |
| **S4: 엔지니어 인터뷰** | 현장 엔지니어 대상 질의 수집 (고통점/고위험) | 30-50건 | 고난도 질의, scope 경계 케이스 |
| **S5: 시스템 로그** | 기존 PE Agent 대화 이력 (재질의/실패 패턴) | 100+ 세션 | 실제 사용 패턴 반영 |

### 후보 질의 풀 구축 파이프라인

```
S2(정비이력) + S3(Q&A) + S5(시스템로그)
  → 질의 추출 (키워드/패턴 매칭 + 수작업 선별)
    → 정규화 (중복 제거, 표현 통일)
      → 스코프 메타 부착 (allowed_scope)
        → gold_evidence 매핑 (S1 매뉴얼 기반, passage_id 수준)
          → paraphrase 생성 (수작업 + LLM 보조)
            → 검토/승인 → Golden Set 등록
```

---

## 3. Paper별 전용 서브셋

### 3.1 Paper A: Contamination Challenge Set

- **목적**: 장비 계층 제약 검증
- **구성**: 스코프 경계에 있는 질의 (동종 타 장비 문서가 혼입되기 쉬운 케이스)
- **크기**: 60건 이상
- **필수 태그**: `scope-critical`, `cross-equipment`
- **특수 필드**: `confusing_docs[]` — 혼동 가능한 타 장비 문서 ID 목록
- **평가**: Contamination@k = |{d ∈ top-k : scope(d) ∉ allowed_scope}| / k

### 3.2 Paper B: Paraphrase Stability Set

- **목적**: 동의 질의 안정성 검증
- **구성**: paraphrase group 단위 (그룹당 4-5개 표현)
  - 한국어 표현 변형 (존댓말/반말, 어순 변경)
  - 한→영 번역 변형
  - 약어/풀네임 변형
  - 표현 강도별 태그: low/medium/high
- **크기**: 60 group × 4 queries = 240건 (합성 벤치마크 기 구축)
- **필수 태그**: `paraphrase-stability`
- **평가**: ParaphraseJaccard@k, RepeatJaccard@k

### 3.3 Paper C: Version-Stability Set

- **목적**: 문서 업데이트 후 성능 드리프트 검증
- **구성**: 동일 질의셋 × 인덱스 버전 (v1, v2, v3, ...)
- **크기**: 100건 × 3-5 버전
- **특수 필드**: `version_sensitive: true`, `expected_version: "v2"`
- **필수 태그**: `version-stability`
- **평가**: 버전 간 Recall@k 변동, 드리프트 감지 지연 (ARL₁)

---

## 4. Baseline 4종 정의

| # | Baseline | 설명 | 구현 경로 |
|---|---|---|---|
| **B0** | BM25 Only | 키워드 기반, ES default BM25 | `es_search.py` with `retrieval_type="sparse"` |
| **B1** | Dense Only | 임베딩 벡터 검색 (cosine sim) | `es_search.py` with `retrieval_type="dense"` |
| **B2** | Hybrid (BM25+Dense) | 가중 합산 (`α·BM25 + (1-α)·Dense`) | `hybrid.py` / `es_hybrid.py` |
| **B3** | Hybrid + Rerank | B2 + cross-encoder 리랭킹 | `retrieval_pipeline.py` with rerank enabled |

### Baseline 실행 조건

- **k**: {5, 10}
- **반복 횟수**: N=10 (안정성 측정용)
- **MQ 모드**: `off` (단일 쿼리, deterministic)
- **rerank**: B0-B2 = off, B3 = on
- **seed**: 고정 (재현성 보장)

---

## 5. 공통 평가 지표 6종

| # | 지표 | 정의 | 적용 |
|---|---|---|---|
| **M1** | `Recall@k` | gold_evidence 중 top-k에 포함된 비율 | 전체 |
| **M2** | `hit@k` | gold_evidence 중 1개 이상 top-k에 있으면 1 | 전체 |
| **M3** | `MRR` | 첫 번째 관련 문서의 역순위 | 전체 |
| **M4** | `Stability@k (Jaccard)` | 반복/동의 질의 간 top-k 집합 Jaccard | Paper B 중심 |
| **M5** | `Contamination@k` | top-k 중 scope 외 문서 비율 | Paper A 중심 |
| **M6** | `p95_latency_ms` | 95퍼센타일 응답 시간 | 전체 |

### 보조 지표 (Paper별)

- Paper B: `RepeatExactMatch@k`, `ParaphraseExactMatch@k`
- Paper C: `Drift_Recall@k`, `ARL₀`, `ARL₁`
- Paper D: `Violation_rate`, `Numeric_accuracy`
- Journal J: `Risk(q, θ)` = 복합 리스크

---

## 6. 평가 하네스 요구사항

### 6.1 실행 로그 스키마

```jsonl
{
  "run_id": "eval-2026-03-01-001",
  "timestamp": "2026-03-01T10:00:00Z",
  "config": {
    "retrieval_type": "hybrid",
    "mq_mode": "off",
    "deterministic": true,
    "top_k": 10,
    "rerank": false
  },
  "golden_set_version": "v0.5",
  "golden_set_hash": "sha256:abc123...",
  "index_name": "pe_docs_v3",
  "index_alias": "pe_docs",
  "results": [
    {
      "qid": "GS-001",
      "query": "XXX 장비 YYY 에러 원인",
      "top_k_doc_ids": ["DOC-123", "DOC-456"],
      "top_k_scores": [0.85, 0.72],
      "recall_at_5": 1.0,
      "recall_at_10": 1.0,
      "hit_at_5": 1,
      "hit_at_10": 1,
      "mrr": 1.0,
      "latency_ms": 142
    }
  ],
  "aggregate": {
    "mean_recall_at_5": 0.82,
    "mean_recall_at_10": 0.91,
    "mean_hit_at_5": 0.88,
    "mean_hit_at_10": 0.94,
    "mean_mrr": 0.76,
    "p95_latency_ms": 310,
    "repeat_jaccard_at_10": 1.0,
    "paraphrase_jaccard_at_10": 0.73
  }
}
```

### 6.2 재현성 요구사항

| 항목 | 요구 |
|---|---|
| Golden set 버전 | hash 기록 필수 |
| ES 인덱스 | alias + 실제 인덱스명 + 문서 수 기록 |
| 코드 버전 | git commit hash 기록 |
| 설정 | config dict 전체 기록 (MQ mode, top_k, rerank, seed 등) |
| 모델/프롬프트 | `llm_model_id`, `embedding_model_id`, `prompt_spec_version` |
| 환경 | Python/ES 버전, hardware spec |

### 6.3 통계 검증 프로토콜

- 기본 보고 형식: 모든 핵심 지표는 `95% CI`와 함께 보고
- 권장 검정: `paired bootstrap`(10,000 resamples) 또는 `Wilcoxon signed-rank`
- 비율 지표: `McNemar` 또는 bootstrap CI
- 판정 기준: `p < 0.05` 또는 95% CI가 0을 교차하지 않을 때 유의
- 다중 비교: Holm-Bonferroni 보정 적용
- 최소 표본: Paper A/B `canonical query >= 200`, Paper C `regression queries >= 50` + 업데이트 시나리오 3종+

### 6.4 구현 참조

- 기존 eval runner: `scripts/evaluation/retrieval_stability_audit.py`
- 합성 벤치마크: `data/synth_benchmarks/stability_bench_v1/`
- 안정성 테스트: `tests/api/test_retrieval_run_stability_and_golden.py`

---

## 7. 라벨링 워크플로

### 7.1 라벨링 가이드 요약

1. **질의 분류**: category 태그 부여 (troubleshooting / specification / procedure / safety)
2. **정답 근거 매핑**: 매뉴얼(S1)에서 gold_evidence 찾기 (doc_id + version + passage_id)
3. **scope 설정**: allowed_scope 기입 (장비군, 모듈, 문서유형)
4. **paraphrase 작성**: canonical_query 기준 3-5개 변형 생성
5. **수치 제약**: 해당 시 numeric_constraints 기입
6. **난이도 태그**: easy (단순 조회) / medium (다중 문서) / hard (추론 필요)

### 7.2 Split/Freeze 거버넌스

- **분할 원칙**: `train/dev/test`는 `paraphrase_group_id` 단위로 분할 (그룹 분산 금지)
- **동결 원칙**: 논문별 실험 시작 전 `golden_set_vX`를 동결, 결과에 버전 해시 명시
- **수정 규칙**: 동결 후 라벨 수정 필요 시 `vX+1`로만 반영, 기존 버전 재기록 금지
- **C용 특칙**: `test` 질의군 고정, 비교 대상은 코퍼스/인덱스 버전만 변경

### 7.3 품질 보증 (QA)

| QA 항목 | 방법 | 기준 |
|---|---|---|
| **Dual labeling** | 전체의 15-20% 이중 라벨링 | Cohen's κ ≥ 0.7 |
| **Adjudication** | κ < 0.7인 항목 3자 판정 | 합의 도달 필수 |
| **Baseline self-check** | B2(Hybrid) 기준 hit@10 기대치 확인 | hit@10 > 0 for easy/medium |
| **Leakage prevention** | 라벨러에게 모델 출력 비공개 | 프로세스 준수 확인 |
| **Paraphrase 품질** | 의미 동일성 + 표현 다양성 검토 | 그룹 내 어휘 겹침 < 50% |

---

## 8. Gate 기준 (Paper별 Done Criteria)

### Paper B Gate

- [x] Paraphrase Stability Set 240 queries (60 groups × 4) 합성 벤치마크 구축
- [x] RepeatJaccard@10 = 1.0 (deterministic mode) 달성
- [ ] ParaphraseJaccard@10 ≥ 0.7 (proposed method)
- [ ] hit@10 ≥ Baseline B2 수준
- [ ] Driver analysis table 완성
- [ ] 논문 6-8p 완성 + 투고

### Paper A Gate

- [ ] Contamination Challenge Set 60+ queries
- [ ] Contamination@10 ≤ 0.05 (proposed method)
- [ ] Recall@10 ≥ Baseline B2 × 0.95
- [ ] 계층 수준별 ablation 완성
- [ ] 논문 6-8p 완성 + 투고

### Paper C Gate

- [ ] Version-Stability Set 100+ queries × 3+ 버전
- [ ] 드리프트 감지 ARL₁ < 10
- [ ] 업데이트 정책 비교 (3종 이상)
- [ ] SPC 관리도 시각화 완성
- [ ] 논문 6-8p 완성 + 투고

### Journal J Gate

- [ ] Paper A/B/C 결과 수집 완료
- [ ] 다목적 최적화 파레토 분석 완성
- [ ] Golden set v1 (400+건) 종합 평가
- [ ] 현장 KPI 연결 데이터 확보
- [ ] 12-18p 원고 + Q1 저널 투고
