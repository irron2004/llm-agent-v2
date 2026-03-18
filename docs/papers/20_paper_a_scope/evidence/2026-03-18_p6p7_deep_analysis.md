# P6/P7 Soft Scoring 심층 분석

**작성일**: 2026-03-18
**목적**: P6/P7 soft scoring의 실패 원인을 구조적으로 분석하고, 논문에 반영할 수정사항 정리

---

## 1. P6/P7 정확한 정의

### 1.1 수식

```
P6: score(d) = 1/(rank+1) - 0.05 × v_scope(d, q)
P7: score(d) = 1/(rank+1) - λ_q × v_scope(d, q)

v_scope(d, q) = 0  if dev(d) = dev(q)     (같은 장비)
              = 0  if d ∈ D_shared         (공용 문서)
              = 1  otherwise               (다른 장비)

λ_q = 0.05 × |{d ∈ top-k : v_scope(d,q)=1}| / |top-k|
```

### 1.2 동작 방식

1. B3 (Hybrid+Rerank)의 top-10 결과를 가져옴
2. 각 문서에 rank-based base score 부여: rank 0 → 1.0, rank 1 → 0.5, ..., rank 9 → 0.1
3. out-of-scope 문서에 λ 만큼 감점
4. 감점된 점수로 재정렬
5. **새 문서를 추가하지 않음** — B3 top-10 안에서만 순서 변경

### 1.3 실제 구현

- 스크립트: `scripts/paper_a/run_masked_p6p7_experiment.py`
- 입력: `data/paper_a/masked_hybrid_results.json` (B3_masked top_doc_ids)
- 출력: `data/paper_a/masked_p6p7_results.json`

---

## 2. 실패 원인 분석

### 2.1 원인 1: Scale Mismatch (λ가 구조적으로 부족)

rank-based score에서 인접 rank 간 gap:

```
gap(r, r+1) = 1/[(r+1)(r+2)]
```

| Rank pair | Gap    | λ=0.05 대비 |
|:---------:|:------:|:----------:|
| 0 → 1    | 0.500  | 10배 부족  |
| 1 → 2    | 0.167  | 3배 부족   |
| 2 → 3    | 0.083  | 1.7배 부족 |
| 3 → 4    | 0.050  | 동일 (역전 불가) |
| 4 → 5    | 0.033  | ✅ 가능    |
| 5 → 6    | 0.024  | ✅ 가능    |

**결론**: λ=0.05는 rank 5 이하의 문서만 이동시킬 수 있음.
오염 문서가 rank 0~3에 있으면 (대부분의 경우) 전혀 건드리지 못함.

### 2.2 원인 2: 후보 집합의 천장 (Candidate Set Ceiling)

B3 recall@10 = 60.7%. 즉 39.3%의 쿼리에서 B3 top-10에 gold doc이 없음.
P6/P7는 B3 top-10을 재정렬만 하므로, 이 39.3%는 **어떤 λ를 써도 개선 불가**.

B4가 91.2%인 이유: ES에서 device filter를 걸어 아예 다른 후보군을 가져옴.
→ 30.5pp 차이는 re-ranking 문제가 아니라 candidate generation 문제.

Per-query 확인:
- B4 성공 + P6 실패: **157건 (27.2%)**
- P6 성공 + B4 실패: **0건**
- 157건 모두 B3 top-10에 gold가 없는 케이스

### 2.3 원인 3: λ 수렴 역설 (Convergence Paradox)

λ를 키우면 어떻게 되는가:

| λ     | 효과 |
|:-----:|------|
| 0.05  | 무효과 (rank 0~3 불변) |
| 0.50  | rank-0 out-of-scope = rank-1 in-scope (동점) |
| 0.67  | rank-0 out-of-scope < rank-2 in-scope |
| 0.80  | top-3 오염 문서 아래로 rank-4 in-scope 상승 |
| **0.90** | **모든 in-scope > 모든 out-of-scope → hard filter와 동일** |

일반 공식: k개의 out-of-scope 문서를 넘으려면 λ > k/(k+1)

```
k=1: λ > 0.500
k=2: λ > 0.667
k=3: λ > 0.750
k=5: λ > 0.833
k=9: λ > 0.900
```

**"효과적인 중간 지대"가 존재하지 않음:**
- λ < 0.5: 무효과
- λ ≥ 0.5: 이미 hard filter에 수렴 시작
- λ ≥ 0.9: hard filter와 완전 동일

### 2.4 P7은 P6과 사실상 동일

- P7의 λ_q 범위: 0.000 ~ 0.050 (P6의 λ=0.05 이하)
- 분포: 중앙값 0.030, 평균 0.0325
- 161/578 쿼리 (27.9%)에서 λ_q = 0.05 = P6과 동일
- 177/578 쿼리에서 P6과 P7의 순서가 다르지만, gold_hit이 바뀐 쿼리: **0건**

---

## 3. 측정 오류 발견 (Critical)

### 3.1 문제

논문 이전 버전에서 "P6이 contamination을 +6.5pp 증가시킨다"고 보고함.
이것은 **측정 artifact**임.

### 3.2 원인

두 실험 스크립트의 device fallback 로직이 다름:

```python
# B3 실험 (run_masked_hybrid_experiment.py):
device = doc_device_map.get(doc_id, doc.get("device_name", ""))
# → ES에서 가져온 device_name 사용 → gcb_ docs = 정상 처리

# P6 실험 (run_masked_p6p7_experiment.py):
device = doc_device_map.get(doc_id, "")
# → fallback 없음 → gcb_ docs = "" ≠ target → 오염으로 집계
```

### 3.3 영향

- 188건에서 P6 contamination > B3 contamination으로 보고됨
- 151건: `gcb_` prefix 청크 ID (cross-index join artifact)
- 37건: 숫자형 doc ID
- **진짜 P6 재정렬로 인한 contamination 변화: 0건**

### 3.4 수정

논문에서 P6/P7 contamination 수치를 제거하고, "P6/P7는 모든 metric에서 B3과 동일"로 수정함.

---

## 4. 학술적 근거

### 4.1 Hard constraint vs Soft preference (IR 문헌)

| 출처 | 결론 |
|------|------|
| Hearst (2006) | 장비 scope는 "hard constraint"로 분류 — 잘못된 값 = 잘못된 답 |
| Tunkelang (2009) | Safety-critical facet는 hard filter 필수 |
| Stoica et al. (2007) | 계층적 장비 분류에서 hard filter가 최적 |

### 4.2 Collection selection (연합 검색)

| 출처 | 결론 |
|------|------|
| Shokouhi & Si (2011) | Pre-retrieval collection selection > post-retrieval score merging |
| B4의 위치 | Oracle collection selection과 동일 구조 |

### 4.3 Score dominance (BM25)

| 출처 | 결론 |
|------|------|
| Robertson et al. (2004) | Additive penalty는 strong base score dominance를 극복 불가 |

---

## 5. 개선 대안

### 5.1 ES function_score (Pre-retrieval boost)

```json
{
  "function_score": {
    "query": {"match": {"content": "PM 절차"}},
    "functions": [{
      "filter": {"term": {"device_name": "SUPRA XP"}},
      "weight": 10
    }],
    "boost_mode": "multiply"
  }
}
```

**장점**: 후보군 자체를 바꿈 (P6/P7의 천장 문제 해결)
**현황**: 코드베이스에 `device_boost` 구현 존재 (`es_search.py` line 514-545), 미평가

### 5.2 Confidence-Gated Hybrid

```
if parser_confidence > θ:
    hard filter (device_name = inferred_device)
else:
    soft penalty (λ × v_scope)  with λ ≥ 0.5
```

**장점**: Parser 오류에 대한 graceful degradation
**근거**: Asadi & Lin (2013) selective search

### 5.3 Family-Level Filter

```
if device_name detected:
    filter by device_name          (B4)
elif equip_type detected:
    filter by equipment_family     (cross-type 오염 제거)
else:
    no filter (fallback)
```

**장점**: explicit_equip 쿼리 (parser 0%) 해결
**근거**: Stoica et al. (2007) hierarchical faceted metadata
**효과 예상**: 최대 오염 쌍 PRECIA→INTEGER plus (173건, cross-family) 제거

---

## 6. 논문 수정 내역

| 위치 | 변경 전 | 변경 후 |
|------|---------|---------|
| Table 3 | Cont@10 컬럼 포함 (P6=0.649) | Cont@10 제거, Gold Loose/NDCG 추가 |
| Section 5.2 본문 | "P6이 contamination +6.5pp 증가" | 3가지 구조적 실패 원인 상세 분석 |
| Scale mismatch 설명 | 1문장 | Rank gap vs λ 테이블 + 수렴 역설 공식 |
| Candidate ceiling | 미언급 | B3 recall@10=60.7% 천장 + 157건 분석 |

---

## 7. 결론

P6/P7의 실패는 parameter tuning (λ값)의 문제가 아니라 **아키텍처의 근본적 한계**:

1. **Post-retrieval re-ranking은 candidate set을 바꿀 수 없다** — gold가 top-10에 없으면 해결 불가
2. **Additive penalty는 rank-based score의 harmonic decay를 극복할 수 없다** — 효과적인 λ는 hard filter와 수렴
3. **"효과적인 중간 지대"가 존재하지 않는다** — soft ↔ hard 사이에 유용한 operating point 없음

따라서 scope filtering은 반드시 **pre-retrieval** 단계에서 수행해야 하며, 이것이 Context-Aware Scope Routing (B4)의 설계 근거이다.
