# Paper A 남은 작업 목록

Date: 2026-03-14
기준: masked query BM25 실험 완료 후 (2026-03-13)

---

## 현재 상태

- **Thesis 입증 가능**: masked query에서 device filter가 contamination 52%→0%, recall 50%→92%
- **BM25-only**: Hybrid/Rerank/Dense 실험 미실시
- **Eval set**: v0.6 (578q), explicit_device + explicit_equip만 (implicit/ambiguous 없음)

---

## Task 1: Hybrid+Rerank 실험 (핵심, 최우선)

**목적**: BM25에서 확인된 결론이 Hybrid+Rerank에서도 성립하는지 검증

**현재 문제**:
- `chunk_v3_content`: text 있음, embedding 없음
- `chunk_v3_embed_bge_m3_v1`: 1024-dim embedding 있음, text/doc_id 없음
- 현재 EmbeddingService: 1024-dim 출력

**해결 방법**: chunk_id 기반 cross-index 검색
1. BM25: `chunk_v3_content`에서 `search_text`로 검색
2. Dense: `chunk_v3_embed_bge_m3_v1`에서 kNN 검색
3. Hybrid: 두 결과를 RRF로 합산
4. Rerank: hybrid 결과를 CrossEncoderReranker로 재정렬

**구현 방안**:
- `EsSearchEngine`에 cross-index hybrid 메서드 추가 또는
- 스크립트에서 직접 두 인덱스를 조합하는 로직 작성

**실험 매트릭스** (question × question_masked × 6 systems):
| System | Question | Filter |
|--------|----------|--------|
| B0 | original + masked | none |
| B1 | original + masked | none |
| B2 | original + masked | none |
| B3 | original + masked | none |
| B4 | masked only | hard device |
| B4.5 | masked only | device + shared |

**산출물**: `data/paper_a/masked_hybrid_results.json`
**예상 시간**: 코드 작성 1-2h, 실행 1-2h

---

## Task 2: implicit/ambiguous Query 추가

**목적**: 실제 운영에서 가장 중요한 시나리오 커버

**현재 상태**: v0.6에 explicit_device(429) + explicit_equip(149)만 있음

**방법**:
- 옵션 A: `question_masked`를 그대로 scope_observability="ambiguous"로 사용
  - 장점: 즉시 가능, 추가 생성 불필요
  - 단점: 마스킹 토큰 `[DEVICE]`가 부자연스러움
- 옵션 B: device명 없는 자연스러운 질문 재생성 (LLM 활용)
  - "SUPRA N series EFEM Controller 교체 절차" → "EFEM Controller 교체 절차"
  - 장점: 자연스러운 질문
  - 단점: LLM 생성 비용
- 옵션 C: cross_device_trap_candidates.json에서 추가 trap query 생성
  - 68개 trap-ready topic 활용
  - 장점: counterfactual 시나리오 직접 테스트
  - 단점: 수작업 필요

**추천**: 옵션 A (즉시) + 옵션 B (50-100개, LLM 생성)

**산출물**: `data/paper_a/eval/query_gold_master_v0_7_with_implicit.jsonl`
**예상 시간**: 옵션 A는 즉시, 옵션 B는 1-2h

---

## Task 3: Parser Accuracy 측정

**목적**: oracle filter vs real parser의 갭 정량화 → practical contribution 입증

**방법**:
1. 578개 질문(원본)에 실제 device parser 돌리기
2. Parser 결과 vs gold device 비교 → accuracy, precision, recall
3. Parser 결과로 B4 filter 돌리기 → oracle B4 vs real B4 비교

**코드 위치**: device parsing 로직은 `backend/llm_infrastructure/llm/langgraph_agent.py` 또는 관련 모듈에 있음

**산출물**: `data/paper_a/parser_accuracy_report.json`
**예상 시간**: 1-2h

---

## Task 4: P6/P7 Soft Scoring 재실험

**목적**: masked query + v0.6 gold로 soft vs hard scoring 비교

**방법**:
1. Phase 4 코드 재사용 (`data/paper_a/phase4_p6p7_results.json` 생성 코드)
2. v0.6 masked query + doc_scope 기반 v_scope 계산
3. `Score(d,q) = Base(d,q) - λ·v_scope(d,q)` 적용
4. B3 vs B4 vs P6 vs P7 비교

**의존성**: Task 1 완료 필요 (Hybrid 점수가 Base score)

**산출물**: `data/paper_a/masked_p6p7_results.json`
**예상 시간**: 1h (Task 1 코드 재사용)

---

## Task 5: Gold Label LLM 검증

**목적**: v0.6 자동생성 gold의 정확도 확인

**방법**:
1. v0.6 strict gold (query, doc) 쌍 추출
2. Phase 1과 동일한 LLM judge로 relevance 판정
3. 자동 gold vs LLM judge 일치율 계산
4. 불일치 사례 분석

**샘플링**: 전수 검증 대신 stratified sample (100-150 쌍) 권장

**산출물**: `data/paper_a/gold_verification_report.json`
**예상 시간**: 2-3h (LLM 호출 시간 포함)

---

## Task 6: 논문 본문 작성

**의존성**: Task 1-5 완료 후

**구조**:
1. Introduction — contamination 문제 정의
2. Related Work — scope-aware retrieval, industrial RAG evaluation
3. Method — device filter, shared policy, soft scoring, masked evaluation
4. Experimental Setup — eval set, systems, metrics
5. Results — contamination@k, gold hit, device별 분석
6. Analysis — evaluation bias 발견, masking의 의미
7. Conclusion

---

## 우선순위 및 의존관계

```
Task 1 (Hybrid) ──────────┐
                          ├──→ Task 4 (P6/P7)
Task 2 (implicit query) ──┘         │
                                    ├──→ Task 6 (논문)
Task 3 (Parser) ────────────────────┘
Task 5 (Gold 검증) ─────────────────┘
```

**추천 실행 순서**:
1. **Task 1 + Task 2 + Task 3** (병렬 가능)
2. **Task 4** (Task 1 완료 후)
3. **Task 5** (독립, 언제든 가능)
4. **Task 6** (전체 완료 후)

**총 예상 시간**: 코드 작업 4-6h, 실험 실행 2-4h (병렬 시)
