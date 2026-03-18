# P8: Evidence-Based Scope Selection (HVSR-lite)

**작성일**: 2026-03-18
**상태**: 실험 준비 완료
**목적**: HVSR 아이디어를 최소 복잡도로 end-to-end 구현하여 B3/B4 사이 실제 갭 측정

---

## 1. 알고리즘 개요

P8은 "어떤 device scope에서 검색해야 하는가?"를 retrieval evidence로 결정한다.

핵심 차별점:
- **P6/P7**: B3 top-10을 고정하고 점수만 조정 (post-retrieval re-ranking)
- **P7+**: B3/B4/B4.5 cached candidates를 섞는 offline policy simulation
- **P8**: 각 device 후보별로 **독립 retrieval**을 수행하고, evidence로 scope를 선택 (end-to-end)

---

## 2. 알고리즘 정의

### Stage 1: Device Hypothesis Generation

쿼리 q의 unfiltered retrieval 결과에서 device 분포를 추출한다.

```
Input:  B3 unfiltered top-N results (N ≥ 10)
Output: H(q) = top-M devices by frequency

procedure GENERATE_HYPOTHESES(results, doc_scope, M):
    device_counts = Counter()
    for doc in results:
        device = doc_scope[doc.doc_id]
        if device != "" and device != "shared":
            device_counts[device] += 1
    return device_counts.most_common(M)  # [(device, count), ...]
```

파라미터: M = 3 (default)

### Stage 2: Per-Hypothesis Hard Retrieval

각 후보 device에 대해 독립적으로 hard-filtered hybrid+rerank를 수행한다.

```
procedure PER_HYPOTHESIS_RETRIEVAL(query, hypotheses, es):
    results = {}
    for device in hypotheses:
        filter = build_device_filter(device, doc_scope)
        hits = hybrid_rerank_search(query, filter, top_k=L)
        results[device] = hits
    return results
```

파라미터: L = 10 (per-hypothesis top-k)

**핵심**: P6/P7과 달리, 각 hypothesis에서 **새로운 candidate set**이 생성된다.
→ candidate ceiling 문제가 없다.

### Stage 3: Evidence-Based Selection

각 hypothesis의 retrieval 결과 품질을 비교하여 최종 device를 선택한다.

```
procedure SELECT_SCOPE(hypothesis_results):
    scores = {}
    for device, hits in hypothesis_results.items():
        # Evidence score = 총 retrieval score mass
        scores[device] = sum(hit.score for hit in hits[:L])
    return argmax(scores)
```

Evidence score로 reranker score의 합을 사용한다.
이는 "해당 device scope 안에서 얼마나 강한 근거가 있는지"를 측정한다.

### Stage 4: Final Output

선택된 device의 결과 + optional shared docs

```
procedure BUILD_FINAL(selected_device, hypothesis_results, shared_cap):
    device_results = hypothesis_results[selected_device][:top_k]

    if shared_cap > 0:
        shared_hits = retrieve_shared_only(query, top_k=shared_cap)
        device_results = merge_and_dedupe(device_results, shared_hits, max=top_k)

    return device_results[:top_k]
```

파라미터: shared_cap = 0, 1, 2 (ablation)

---

## 3. 파라미터 요약

| 파라미터 | 의미 | 기본값 | 탐색 범위 |
|---------|------|--------|----------|
| M | device 후보 수 | 3 | {2, 3, 5} |
| L | per-hypothesis top-k | 10 | 고정 |
| shared_cap | shared doc 최대 포함 수 | 0 | {0, 1, 2} |

**총 파라미터: 3개** (HVSR의 14+개 대비 대폭 간소화)

---

## 4. 실험 설계

### 4.1 비교군

| 시스템 | 설명 |
|--------|------|
| B3_masked | Hybrid+Rerank, no filter (baseline) |
| B4_masked | Hybrid+Rerank + oracle hard device filter (upper bound) |
| B4.5_masked | B4 + shared docs allowed |
| P6_masked | B3 + soft scoring λ=0.05 |
| P7_masked | B3 + adaptive soft scoring |
| P7+_masked | Offline cached-candidate policy (reference) |
| **P8_masked** | Evidence-based scope selection (proposed) |
| P8_sc1_masked | P8 + shared_cap=1 |
| P8_sc2_masked | P8 + shared_cap=2 |

### 4.2 메트릭

- `cont@10`: contamination at 10
- `gold_hit_strict@10`: gold doc in top-10 (strict)
- `gold_hit_loose@10`: gold doc in top-10 (loose)
- `MRR`: Mean Reciprocal Rank
- `scope_accuracy`: 선택된 device == target device 비율

### 4.3 분석 축

- 전체 (n=578)
- scope_observability별: explicit_device (429), explicit_equip (149)
- hypothesis_rank: target device가 hypothesis 몇 번째에 있었는지

### 4.4 기대 성공 조건

1. `P8_masked gold_strict > B3_masked gold_strict` (후보집합 개선 효과)
2. `P8_masked cont@10 < B3_masked cont@10` (오염 감소)
3. `P8_masked gold_strict`가 `B4_masked`에 접근 (scope selection 정확도)
4. `scope_accuracy`가 60% 이상 (무작위 1/27 = 3.7% 대비)

---

## 5. 구현 노트

### 5.1 Hypothesis generation source

**옵션 A (cached)**: B3_masked cached top-10에서 device 분포 추출
- 장점: 추가 ES 쿼리 불필요, 빠름
- 단점: top-10만으로 분포가 noisy할 수 있음

**옵션 B (live probe)**: unfiltered hybrid top-40 fresh 쿼리
- 장점: 더 정확한 device 분포
- 단점: 추가 ES 쿼리 필요

→ 초기 실험은 옵션 A로 시작. 결과 부족 시 옵션 B로 전환.

### 5.2 Evidence score variants (ablation)

- `score_sum`: reranker score 합 (default)
- `score_max`: reranker score 최대값
- `score_top3`: top-3 score 합
- `hit_count`: 결과 수 (filter 후 문서가 적으면 낮은 score)

### 5.3 레이턴시

```
Per query: M × hybrid+rerank = M × B4 latency
M=3: ~3× B4 latency
```

B4 단일 쿼리가 ~200ms라면, P8은 ~600ms.
생산 환경에서 허용 가능한 수준 (HVSR의 4-6×보다 적음).

---

## 6. P6/P7 실패와의 이론적 대비

| 문제 | P6/P7 | P8 |
|------|-------|-----|
| Candidate ceiling | B3 top-10 고정 → recall@10=60.7% 천장 | 각 device별 독립 retrieval → 천장 없음 |
| Scale mismatch | λ=0.05 vs rank gap 0.5 → top rank 이동 불가 | Hard filter → 완전 분리 |
| λ convergence | λ↑ → hard filter 수렴 | 처음부터 hard filter 사용 |
| Scope selection | 없음 (B3 결과 그대로) | Evidence score로 최적 scope 선택 |

---

## 7. 논문 포지셔닝

P8은 다음 논문 메시지를 실증한다:

> "Post-hoc soft penalties cannot fix cross-equipment contamination because
> the error arises before ranking, at scope selection time.
> We therefore replace post-hoc demotion with evidence-based hard-scope
> selection: generate device hypotheses, retrieve independently per scope,
> and select the scope with strongest retrieval evidence."

기여 구조:
1. 문제 정의: cross-equipment contamination 정량화
2. 음성 결과: P6/P7 soft scoring 구조적 실패 분석
3. **제안 알고리즘**: P8 evidence-based scope selection
4. 실증: P8이 B3과 B4 사이 격차를 줄임
