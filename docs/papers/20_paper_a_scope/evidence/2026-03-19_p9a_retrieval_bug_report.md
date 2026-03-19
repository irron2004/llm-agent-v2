# P9a Retrieval 버그 리포트

**작성일**: 2026-03-19
**대상**: `scripts/paper_a/run_p9a_top1_hard_scope.py` 및 `run_p8_evidence_scope_experiment.py`
**상태**: 수정 완료 (P9a), P8은 미수정

---

## 1. 증상

P9a (P7+ top-1 device → hard filter retrieval) 결과:
- **scope_acc = 93.4%** — device 선택은 정확
- **gold_strict = 150/578 (26.0%)** — 올바른 device에서도 gold 문서를 못 찾음
- **평균 반환 결과: 3.0개** (10개여야 정상)

비교: B4 (같은 device로 필터링)는 gold_strict = 527/578 (91.2%), 10개 반환.

---

## 2. 근본 원인: doc_id 레벨 RRF vs chunk_id 레벨 RRF

### B4의 retrieve() (run_masked_hybrid_experiment.py)

```python
# chunk_id 레벨로 RRF 수행
bm25_ranked = [h["chunk_id"] for h in bm25_hits]
dense_ranked = [h["chunk_id"] for h in dense_hits]
fused = rrf_fuse(bm25_ranked, dense_ranked)

# chunk_id로 content 조회 → chunk별 reranking
fused_ids = [cid for cid, _ in fused[:top_k * 2]]
content_map = _fetch_content_by_chunk_ids(es, fused_ids)
```

- 문서당 여러 chunk가 RRF 후보에 포함
- 최적 chunk이 reranking에 사용됨
- 결과: 10개 doc 반환 정상

### P9a/P8의 retrieve_hybrid_rerank() (버그 버전)

```python
# doc_id 레벨로 RRF 수행 ← 문제
scores: dict[str, float] = defaultdict(float)
for rank, h in enumerate(bm25_hits):
    scores[h["doc_id"]] += 1.0 / (k + rank + 1)

# doc_id로 content 조회 → 문서당 1개 chunk만 반환
contents = fetch_contents(es, fused_ids)  # {doc_id: content}
```

- 문서당 1개 chunk만 RRF에 기여
- `fetch_contents`가 doc_id 기준으로 1개 chunk만 반환 (ES에서 먼저 매칭되는 것)
- 해당 chunk이 쿼리와 무관한 내용일 수 있음 → reranking 품질 급락
- 결과: 3개 doc만 반환 (나머지는 content가 없거나 중복 제거)

---

## 3. 추가 발견: dense search device_name 대소문자 불일치

### 문제

```
doc_scope device_name: "GENEVA XP"  (대문자)
ES embed index device_name: "GENEVA xp"  (소문자)
```

`{"terms": {"device_name": ["GENEVA XP"]}}` → ES에서 0건 매치.

### 영향

- dense search가 전혀 기여하지 않음 → RRF가 BM25 single-source
- BM25만으로는 후보 다양성이 부족

### 수정

`build_allowed_devices_map()`에서 ES aggregation으로 실제 device_name 값을 수집하여 매핑에 포함.

```python
# ES embed index에서 실제 device_name 값 수집
resp = es.search(index=EMBED_INDEX, body={
    "size": 0,
    "aggs": {"devices": {"terms": {"field": "device_name", "size": 200}}},
})
for bucket in resp["aggregations"]["devices"]["buckets"]:
    raw = bucket["key"]
    norm = normalize_device_name(raw)
    upper_to_raw[norm].add(raw)
```

**참고**: 이 수정만으로는 gold_strict 개선이 없었음. 근본 원인은 doc_id 레벨 RRF.

---

## 4. 수정 내용

### P9a 수정 (완료)

1. `dense_search()`: `chunk_id`, `device_name` 반환하도록 수정
2. `rrf_fuse()`: chunk_id 레벨로 RRF 수행
3. `_fetch_content_by_chunk_ids()`: chunk_id로 content 조회 (B4와 동일)
4. `retrieve_hybrid_rerank()`: chunk_id 레벨 RRF → reranking → doc_id dedup
5. `build_allowed_devices_map()`: ES aggregation으로 device_name variant 수집

### P8 수정 (미수정)

P8도 동일한 doc_id 레벨 RRF 버그를 가지고 있음.
P8 결과 재해석 시 주의: P8의 낮은 성능 일부는 이 버그에 기인할 수 있음.

---

## 5. 검증 체크리스트

- [ ] P9a 재실행 후 평균 반환 결과 수 ≥ 8
- [ ] P9a gold_strict이 scope_correct 비율에 근접 (93.4% × B4_within_device_rate)
- [ ] P8 결과도 동일 버그 수정 후 재평가 필요 여부 판단
- [ ] dense search device_name 매칭 확인 (0건 아닌지)

---

## 6. 교훈

1. **chunk-level vs doc-level 구분이 중요**: ES 인덱스는 chunk 단위. retrieval pipeline은 chunk 레벨로 동작해야 하며, doc 레벨 dedup은 최종 단계에서만 수행
2. **기존 코드 재사용 우선**: `run_masked_hybrid_experiment.py`의 `retrieve()`가 이미 올바른 구현. 새로 작성 시 반드시 참조
3. **device_name 필드의 대소문자/포맷 불일치는 반복적으로 발생**: normalize → ES actual value 매핑을 공통 유틸로 분리 필요
