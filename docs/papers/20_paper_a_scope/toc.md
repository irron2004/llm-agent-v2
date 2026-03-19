# Paper A — Table of Contents

**논문 제목**: Device-Aware Scope Filtering for Cross-Equipment Contamination Control in Industrial RAG
**최종 업데이트**: 2026-03-19

---

## 1. 논문 본문 및 기획

| 파일 | 설명 |
|------|------|
| `README.md` | 프로젝트 개요 및 디렉토리 구조 |
| `paper_a_draft_v2.md` | 논문 초안 v2 (영문) |
| `paper_a_draft_v2_ko.md` | 논문 초안 v2 (한글) |
| `paper_a_scope.md` | 초기 논문 범위 정의 (Hierarchy-aware Scope Routing) |
| `paper_a_scope_spec.md` | **실험 정의서 v0.6** — 조건 정의, 메트릭, 평가셋 스펙 |
| `related_work.md` | 관련 연구 정리 |
| `references.bib` | 참고문헌 BibTeX |

---

## 2. 연구 로드맵 및 타임라인

| 파일 | 설명 |
|------|------|
| `paper_a_series_map.md` | Paper A 시리즈 전체 맵 (A / A-1 / A-2) |
| `paper_a_series_blueprint.md` | 시리즈 블루프린트 — 장기 연구 방향 |
| `2026-03-14_paper_a_timeline_consolidated.md` | 통합 타임라인 (상세) |
| `2026-03-18_paper_a_timeline_easy_ko.md` | 타임라인 쉬운 요약 (한글) |
| `2026-03-14_execution_tasks.md` | 실행 우선순위 작업 목록 |
| `evidence_mapping.md` | Evidence ↔ 논문 섹션 매핑 |
| `HVSR_idea.md` | HVSR(Hierarchical Verification + Scope Routing) 아이디어 노트 |

---

## 3. 실험 결과 문서 (evidence/)

### 3.1 코퍼스 및 평가 데이터

| 파일 | 설명 |
|------|------|
| `evidence/2026-03-04_corpus_statistics.md` | 코퍼스 통계 (장비, 문서, 청크 분포) |
| `evidence/2026-02-12_gcb_equip_id_matching_report.md` | GCB equip_id 매칭 분석 — equip_id 기반 장비 식별 |
| `evidence/2026-03-09_gold_rejudging_analysis.md` | Gold label 재판정 분석 — LLM 기반 gold 품질 검증 |
| `evidence/2026-03-12_dataset_protocol_redesign.md` | 데이터셋 프로토콜 재설계 (Bias-Resilient) |
| `evidence/2026-03-12_cross_device_topic_feasibility.md` | 교차 장비 토픽 분석 — 반사실 scope trap 가능성 |
| `evidence/2026-03-14_v06_gold_audit.md` | v0.6 gold 품질 감사 |
| `evidence/2026-03-14_v07_implicit_eval.md` | v0.7 implicit 평가 시도 |
| `evidence/2026-03-14_v07_mixed_eval_restoration.md` | v0.7 mixed eval 복원 |

### 3.2 초기 실험 (B3, B4, B4.5)

| 파일 | 설명 | 핵심 수치 |
|------|------|----------|
| `evidence/2026-03-05_paper_a_main_results.md` | 메인 실험 결과 (B3/B4 비교) | B3: 60.7%, B4: 91.2% |
| `evidence/2026-03-05_paper_a_error_analysis.md` | 오류 분석 및 실패 분류 | |
| `evidence/2026-03-05_paper_a_run_index.md` | 실험 실행 인덱스 | |
| `evidence/2026-03-14_full_experiment_results.md` | **전체 실험 결과 종합** — B3~B4.5 + P6/P7 통합 | |
| `evidence/2026-03-14_b45_failure_decomposition.md` | B4.5 실패 분해 — parser accuracy vs retrieval quality | |
| `evidence/2026-03-14_hybrid_rerank_recovery.md` | Hybrid/Rerank 복구 효과 분석 | |
| `evidence/2026-03-14_oracle_vs_parser_gap.md` | Oracle vs Parser gap 분석 | |
| `evidence/2026-03-12_slot_valve_hard_filter_recall_loss.md` | Hard filter recall loss 사례 (slot valve) | |

### 3.3 Soft Scoring (P6, P7, P7+)

| 파일 | 설명 | 핵심 수치 |
|------|------|----------|
| `evidence/2026-03-14_masked_p6p7_reexperiment.md` | Masked P6/P7 재실험 | |
| `evidence/2026-03-18_p6p7_deep_analysis.md` | **P6/P7 심층 분석** — soft scoring의 한계와 원인 | |
| `evidence/2026-03-18_p7plus_algorithm_proposal.md` | **P7+ 알고리즘 제안서** — confidence proxy, adaptive λ/μ/η | |
| `evidence/2026-03-18_p7plus_experiment.md` | P7+ 실험 결과 | P7+: 87.5% |

### 3.4 P8: Evidence-Based Scope Selection

| 파일 | 설명 | 핵심 수치 |
|------|------|----------|
| `evidence/2026-03-18_p8_algorithm_spec.md` | P8 알고리즘 스펙 (HVSR-lite: hypothesis → retrieve → verify) | |
| `evidence/2026-03-18_p8_implementation_issue_report.md` | P8 구현 이슈 리포트 (device_name 대소문자 불일치 등) | |
| `evidence/2026-03-19_p8_failure_analysis_and_p9_proposal.md` | **P8 실패 분석 + P9 제안** — Stage 3 역설 발견 | scope_acc=39.6% |
| `evidence/2026-03-19_p8_rerun_after_fix.md` | **P8 재실행 (버그 수정 후)** — 구조적 실패 확인 | 39.8% (B3 이하) |

### 3.5 P9a: Retrieval-Informed Device Identification (최종 제안)

| 파일 | 설명 | 핵심 수치 |
|------|------|----------|
| `evidence/2026-03-19_p9a_results_summary.md` | **P9a 실험 결과 요약** — P7+ top-1 hard scope | gold_strict=85.1%, MRR=0.618, cont=0.048, scope_acc=93.4% |
| `evidence/2026-03-19_p9a_retrieval_bug_report.md` | P9a 구현 중 발견한 retrieval 버그 2건 (device_name variant, doc_id dedup) | |
| `evidence/2026-03-19_p9a_case_examples.md` | **P9a 사례 기반 분석** — 5개 대표 케이스 (실제 쿼리 텍스트 포함) | |
| `evidence/2026-03-19_p9a_scope_miss_analysis.md` | **P9a Scope Miss 종합 분석** — B4 vs P9a 교차 분석 4 quadrant, 부품별 빈도/정확도 상관 | |

### 3.6 P9b: Margin-Gated Verifier (Negative Result)

| 파일 | 설명 | 핵심 수치 |
|------|------|----------|
| `evidence/2026-03-19_p9b_results_summary.md` | **P9b 실험 결과** — margin-gated verification은 P9a보다 나쁨 | 최선 -4건, 최악 -26건 |

### 3.7 P9c: TF-IDF Device Profile (Negative Result)

| 파일 | 설명 | 핵심 수치 |
|------|------|----------|
| `evidence/2026-03-19_p9c_results_summary.md` | **P9c 실험 결과** — TF-IDF 단독/hybrid 모두 실패, 부품명 빈도 분포 분석 포함 | scope_acc=29.3%, miss 32건 중 0건 복구 |

### 3.8 기타

| 파일 | 설명 |
|------|------|
| `evidence/2026-01-08_meta_guided_hierarchical_rag.md` | 초기 아이디어 — Meta-guided Hierarchical RAG |
| `evidence/2026-03-13_paper_a_progress_summary.md` | 진행 경과 및 결론 정리 (중간 checkpoint) |
| `evidence/2026-03-14_remaining_tasks.md` | 남은 작업 목록 |
| `evidence/2026-03-18_HVSR_idea.md` | HVSR 제안 검토 의견 |

---

## 4. 리뷰 문서 (review/)

| 파일 | 설명 |
|------|------|
| `review/preregistration.md` | 사전 등록된 비교 조건, 분할, 튜닝 규칙 |
| `review/hypotheses_experiments.md` | 검증 가능한 가설 및 실험 매트릭스 |
| `review/consistency_audit.md` | 일관성 감사 — 수치/주장 교차 검증 |
| `review/reviewer_report.md` | 리뷰어 리포트 (내부 검토) |

---

## 5. 실험 스크립트 (`scripts/paper_a/`)

### 5.1 데이터 구축

| 스크립트 | 설명 |
|----------|------|
| `build_shared_and_scope.py` | shared/device-specific 문서 분류 → doc_scope.jsonl, shared_doc_ids.txt |
| `build_corpus_meta.py` | 코퍼스 메타데이터 수집 |
| `build_family_map.py` | 장비 family 매핑 구축 |
| `build_eval_sets.py` | 평가셋 생성 |
| `build_v07_mixed_eval_set.py` | v0.7 mixed eval set 생성 |
| `build_gold_expansion_candidates.py` | Gold 확장 후보 생성 |
| `generate_question_gold_from_corpus.py` | 코퍼스로부터 질문-정답 쌍 자동 생성 |
| `generate_v06_gold_audit.py` | v0.6 gold 감사 생성 |
| `canonicalize.py` | 장비명 정규화 |
| `normalize_zedius_xp.py` | ZEDIUS XP 명칭 통일 |

### 5.2 검증 및 전처리

| 스크립트 | 설명 |
|----------|------|
| `preflight_es.py` | ES 인덱스 사전 검증 |
| `validate_policy_artifacts.py` | policy 아티팩트 검증 |
| `validate_eval_jsonl.py` | eval JSONL 형식 검증 |
| `validate_master_eval_jsonl.py` | master eval JSONL 검증 |
| `rebuild_query_gold_master_splits.py` | query gold master 분할 재구축 |
| `measure_parser_accuracy.py` | parser 정확도 측정 |
| `report_split_quality.py` | 분할 품질 리포트 |

### 5.3 실험 실행

| 스크립트 | 설명 | 출력 |
|----------|------|------|
| `run_masked_hybrid_experiment.py` | B3/B4 마스킹 실험 (baseline) | `data/paper_a/masked_hybrid_results.json` |
| `run_masked_p6p7_experiment.py` | **P6/P7/P7+ soft scoring 실험** | `data/paper_a/masked_p6p7_results.json` |
| `run_p8_evidence_scope_experiment.py` | **P8 evidence-based scope selection** | `data/paper_a/p8_results.json` |
| `run_p9a_top1_hard_scope.py` | **P9a: P7+ top-1 hard scope (최종 제안)** | `data/paper_a/p9a_results.json` |
| `run_p9_stage1_diagnostic.py` | P9 Stage 1 진단 (device mass 분석) | |
| `run_p9b_margin_gated_verifier.py` | **P9b: margin-gated verifier (negative)** | `data/paper_a/p9b_results.json` |
| `run_p9c_tfidf_proposal.py` | **P9c: TF-IDF device profile (negative)** | `data/paper_a/p9c_results.json` |

### 5.4 평가

| 스크립트 | 설명 |
|----------|------|
| `evaluate_paper_a.py` | 단일 실험 평가 |
| `evaluate_paper_a_master.py` | 마스터 평가 (전체 조건 비교) |
| `phase3_retrieve_and_pool.py` | Phase 3 retrieval + pooling |
| `retrieval_runner.py` | 범용 retrieval 실행기 |
| `analyze_b45_failure_decomposition.py` | B4.5 실패 분해 분석 |
| `stats_addons.py` | 통계 부가 기능 |
| `_io.py` | I/O 유틸 (JSONL 읽기/쓰기) |
| `_paths.py` | 경로 상수 |

---

## 6. 데이터 파일 (`data/paper_a/`)

### 6.1 평가 쿼리셋

| 파일 | 설명 |
|------|------|
| `eval/query_gold_master.jsonl` | 원본 master 평가셋 |
| `eval/query_gold_master_v0_6_generated_full.jsonl` | **v0.6 생성 평가셋 (578건, 현재 사용)** |
| `eval/query_gold_master_v0_6_generated_full_strict.jsonl` | v0.6 strict gold 버전 |
| `eval/query_gold_master_v0_4_frozen.jsonl` | v0.4 동결 버전 |
| `eval/query_gold_master_v0_5.jsonl` | v0.5 버전 |

### 6.2 실험 결과

| 파일 | 설명 | 생성일 |
|------|------|--------|
| `masked_hybrid_results.json` | B3/B4 마스킹 실험 결과 | 03-14 |
| `masked_p6p7_results.json` | P6/P7/P7+ 실험 결과 | 03-18 |
| `p8_results.json` | P8 evidence scope 결과 (버그 수정 후) | 03-19 |
| `p9a_results.json` | **P9a 결과 (최종 제안)** | 03-19 |
| `p9b_results.json` | P9b margin-gated verifier 결과 (negative) | 03-19 |
| `p9c_results.json` | P9c TF-IDF profile 결과 (negative) | 03-19 |

### 6.3 코퍼스 레이블

| 디렉토리/파일 | 설명 |
|---------------|------|
| `corpus_labels/document_scope_table_es_generated.csv` | 문서-장비 매핑 테이블 |
| `corpus_labels/document_scope_shared_es_generated.csv` | 공유 문서 목록 |
| `corpus_labels/document_scope_device_es_generated.csv` | 장비별 문서 |
| `corpus_labels/document_scope_equip_es_generated.csv` | equip별 문서 |
| `metadata/device_catalog.csv` | 장비 카탈로그 |
| `metadata/equip_catalog.csv` | 설비 카탈로그 |
| `metadata/doc_type_map.csv` | 문서 타입 매핑 |

### 6.4 기타

| 디렉토리 | 설명 |
|-----------|------|
| `rejudge/` | Gold label 재판정 데이터 (LLM judge 결과) |
| `reports/` | Gold 확장 보고서 (T17 시리즈) |
| `runs/` | 실험 실행 결과 (per_query, summary, bootstrap CI 등) |
| `train_optional/` | 선택적 학습 데이터 (reranker pairs, router train, intent labels) |

---

## 7. 실험 흐름도

```
[평가 데이터]                    [실험 조건]                     [결과]
eval/v0_6_generated_full.jsonl
        │
        ├──→ B3_masked (unscoped)  ──→ gold_strict=60.7%, cont=0.584
        ├──→ B4_masked (oracle)    ──→ gold_strict=91.2%, cont=0.001
        │
        ├──→ P6/P7 (soft scoring)  ──→ P7: cont 개선, gold_strict 미미
        ├──→ P7+ (adaptive scoring)──→ gold_strict=87.5%
        │
        ├──→ P8 (hypothesis→retrieve→verify)
        │       └──→ 실패: scope_acc=40.8%, Stage 3 역설
        │
        ├──→ P9a (P7+ top-1 hard scope) ★ 최종 제안
        │       └──→ gold_strict=85.1%, MRR=0.618, cont=0.048, scope_acc=93.4%
        │
        ├──→ P9b (P9a + margin-gated verifier)
        │       └──→ NEGATIVE: 최선 -4건, 최악 -26건
        │
        └──→ P9c (TF-IDF device profile + P7+ hybrid)
                └──→ NEGATIVE: miss 32건 중 0건 복구, scope_acc=29.3%
```

---

## 8. 핵심 결론 요약

| # | 결론 | 근거 문서 |
|---|------|----------|
| 1 | B3(unscoped)는 교차 장비 오염이 심각 (cont=58.4%) | `evidence/2026-03-05_paper_a_main_results.md` |
| 2 | B4(oracle filter)는 상한선 (91.2%) | 상동 |
| 3 | Soft scoring(P6/P7)은 contamination 감소에 한계 | `evidence/2026-03-18_p6p7_deep_analysis.md` |
| 4 | P7+ adaptive scoring으로 87.5% 도달 | `evidence/2026-03-18_p7plus_experiment.md` |
| 5 | P8 evidence-based verification은 구조적 실패 (Stage 3 역설) | `evidence/2026-03-19_p8_failure_analysis_and_p9_proposal.md`, `evidence/2026-03-19_p8_rerun_after_fix.md` |
| 6 | **P9a(P7+ top-1 hard scope)가 최적**: 85.1% strict, 93.4% scope_acc | `evidence/2026-03-19_p9a_results_summary.md` |
| 7 | P9b(margin-gated verifier)는 negative result — 정확도 > 93%이면 검증이 해로움 | `evidence/2026-03-19_p9b_results_summary.md` |
| 8 | P9c(TF-IDF profile)도 negative — 마스킹 쿼리의 정보량 한계 | `evidence/2026-03-19_p9c_results_summary.md` |
| 9 | Scope miss ~5.7%는 마스킹 평가의 구조적 상한 | `evidence/2026-03-19_p9a_scope_miss_analysis.md` |
| 10 | Both miss 40건(7.1%)은 코퍼스 커버리지 문제 (주로 SUPRA Vplus) | 상동 |
