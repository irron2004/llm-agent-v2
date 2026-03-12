# SOP 검색 실패/미스 원인 분석 보고서

날짜: 2026-03-12
분석 대상:
- `.sisyphus/evidence/2026-03-11_sop_filter_eval/sop_only_results.jsonl` (79건)
- `.sisyphus/evidence/2026-03-11_sop_retrieval_eval/sop_only_results.jsonl` (79건)

---

[OBJECTIVE] SOP filter eval의 doc_hit=False 4건과 retrieval eval의 page_hit=False 15건의 원인 규명 및 개선 방안 제시

[DATA] 79건 SOP 평가 세트, SUPRA XP(ZEDIUS XP) 장비 문서, chunk_v3_content/rag_chunks_dev_v2 인덱스

---

## Part 1: Filter eval doc_hit=False 4건 (idx 75-78)

[FINDING] 4건의 doc_hit=False는 실제 검색 실패가 아닌 평가 스크립트의 정규화 버그에 의한 오판정(false negative)이다.
[STAT:n] n=4, 전체 평가 건수 79건
[STAT:effect_size] 보정 시 doc_hit 94.9% → 100.0% (4.9%p 상승)

### 근본 원인: `_normalize()` 함수가 `&` 문자를 처리하지 않음

4건 모두 동일 문서 `global sop_supra xp_sw_all_sw installation & setting.pdf`를 gold_doc으로 갖는다.

```
# 평가 스크립트의 현재 _normalize (run_sop_filter_eval.py:76, run_sop_retrieval_eval.py:47)
def _normalize(name):
    return name.lower().strip().replace(" ", "_").replace("-", "_").replace(".pdf", "")

# gold_doc 정규화 결과
gold_doc  = 'global sop_supra xp_sw_all_sw installation & setting.pdf'
gold_norm = 'global_sop_supra_xp_sw_all_sw_installation_&_setting'   # & 잔존!

# ES에 색인된 doc_id (& 제거됨)
es_doc_id = 'global_sop_supra_xp_sw_all_sw_installation_setting'

# substring check: gold_norm IN es_doc_id -> False  (불일치)
```

검색 시스템은 4건 모두 올바르게 해당 문서를 반환하고 있으며, ES 인덱스에도 정상 색인됨을 확인.

### 검증: ES 인덱스 확인
- `chunk_v3_content`: 82 chunks, `doc_type='sop'`
- `rag_chunks_dev_v2`: 74 chunks, `doc_type='SOP'`
- 문서 타입 불일치 없음, 인덱싱 누락 없음

[LIMITATION] 다른 gold_doc에도 `&` 포함 파일명이 있을 경우 동일 오판정 발생 가능. 현재 79건 평가셋에서는 이 파일 1종만 해당.

---

## Part 2: Retrieval eval page_hit=False 15건

[FINDING] page_hit=False 15건은 두 가지 별개 원인으로 발생하며, 실제로는 15건 모두 top-20 내에 정답 페이지 청크가 존재한다.
[STAT:n] n=15 (79건 중)
[STAT:effect_size] 보정된 page_hit: 60/79(75.9%) → 79/79(100.0%)

### 원인 A: `_normalize()` 버그 (Part 1과 동일)

retrieval eval의 4건 doc_hit=False (idx 75-78)도 동일한 `&` 정규화 버그로 발생.

### 원인 B: `_check_hit()` 함수의 first-match-wins 로직

```python
# run_sop_retrieval_eval.py:67-93
def _check_hit(docs, gold_doc, gold_pages):
    for rank, doc in enumerate(docs):
        if matched:                          # 첫 번째 doc 매치에서
            if page_int in gold_page_set:
                return True, True, rank+1
            return True, False, rank+1      # 페이지 불일치여도 여기서 종료!
    return False, False, None
```

15건에서 top-1 반환 청크가 커버 페이지(page 1)인 경우:
- 12/15건: top-1 청크 = page 1 (커버)
- 14/15건: top-1 청크 < gold_start (정답 범위 이전 페이지)
- 15/15건: top-20 내에 정답 페이지 청크 존재

[STAT:effect_size] 정규화 Fix만 적용: 64/79=81.0% / 평가 기준도 수정(any-page): 79/79=100.0%

### 커버 페이지가 상위 랭크되는 이유

청크 구조 분석 결과, 모든 SOP 페이지 청크가 동일한 헤더를 반복 포함:

```
# 모든 페이지의 search_text 시작 부분 (동일)
# Global SOP_ZEDIUS XP_ ALL_EFEM_PIO SENSOR BOARD
# Revision No: 0  Page: N / 14
## Scope
이 Global SOP는 ZEDIUS XP 설비의 EFEM PIO SENSOR BOARD 관련 작업 시에...
```

- **Page 1 (커버)**: 헤더+범위 설명만 존재 → 쿼리 키워드 밀도 100%
- **Page 6+ (절차)**: 헤더+범위 설명+작업 절차 → 쿼리 키워드 밀도 희석
- 쿼리 `"EFEM PIO SENSOR BOARD 교체 방법"`에 대해 dense embedding과 BM25 모두 page 1이 상위 랭크됨

[LIMITATION] 커버 페이지 문제는 청크 파이프라인 수준의 구조적 이슈로, 평가 기준 변경으로 측정 상 해결되지만 실제 시스템도 사용자에게 커버 페이지를 먼저 제공할 수 있음.

---

## Part 3: ES 인덱스 상태 확인

[FINDING] miss 문서 4건의 gold_doc은 ES에 정상 색인되어 있으며, doc_type도 SOP로 올바르게 설정됨.
[STAT:n] n=1 (유일한 miss 대상 문서)

- `chunk_v3_content`: 82 chunks, `doc_type='sop'`, `device_name='SUPRA_XP'`
- `rag_chunks_dev_v2`: 74 chunks, `doc_type='SOP'`
- set_up_manual로 잘못 분류된 것 아님

---

## 개선 방안

### 1. 평가 스크립트 `_normalize()` 수정 [즉시 적용 가능]

```python
# 수정안: & 문자 처리 추가
import re
def _normalize(name):
    n = name.lower().strip()
    n = re.sub(r'\s*&\s*', '_', n)   # ' & ' -> '_'
    n = n.replace(' ', '_').replace('-', '_').replace('.pdf', '')
    n = re.sub(r'_+', '_', n).strip('_')
    return n
```

적용 파일:
- `scripts/evaluation/run_sop_filter_eval.py:76-78`
- `scripts/evaluation/run_sop_retrieval_eval.py:47-48`

### 2. `_check_hit()` 평가 기준 변경 [선택적]

현재: 첫 번째 매칭 청크의 페이지만 확인 (first-match-wins)
권고: 매칭 문서의 모든 청크 중 정답 페이지가 있으면 hit (best-page-for-doc)

### 3. 커버 페이지 랭킹 우선순위 저하 [구조적 개선]

- 옵션 A: `search_text` 생성 시 커버/ToC 페이지 헤더 중복 제거
- 옵션 B: 페이지 타입 메타데이터(cover, toc, procedure) 추가 후 검색 시 procedure 페이지 가중치 부여
- 옵션 C: 재랭킹(cross-encoder) 단계에서 커버 페이지 페널티 적용

---

## 수치 요약

| 지표 | 현재 (보고값) | 수정 후 |
|------|-------------|---------|
| filter eval doc_hit | 75/79 = 94.9% | 79/79 = 100.0% |
| retrieval eval doc_hit | 75/79 = 94.9% | 79/79 = 100.0% |
| retrieval eval page_hit (norm fix만) | 60/79 = 75.9% | 64/79 = 81.0% |
| retrieval eval page_hit (norm+기준변경) | 60/79 = 75.9% | 79/79 = 100.0% |
