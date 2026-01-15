# Retrieval Golden Set 구축 전략

**작성일**: 2026-01-07
**목적**: 검색 품질 평가를 위한 Golden Set(qrels) 구축 방법론 정의

---

## 목차

1. [개요](#1-개요)
2. [전략의 핵심 원칙](#2-전략의-핵심-원칙)
3. [Pooling 방법론](#3-pooling-방법론)
4. [Graded Relevance](#4-graded-relevance)
5. [Annotation 프로세스](#5-annotation-프로세스)
6. [품질 관리](#6-품질-관리)
7. [프로젝트 특화 사항](#7-프로젝트-특화-사항)
8. [실행 계획](#8-실행-계획)
9. [비용 및 시간 추정](#9-비용-및-시간-추정)
10. [실무 함정 및 보완](#10-실무-함정-및-보완)
11. [Answer Quality 평가 확장](#11-answer-quality-평가-확장)
12. [부록](#12-부록)

---

## 1. 개요

### 1.1 배경

현재 RAG 시스템은 하이브리드 검색(BM25 + Dense Vector)을 사용하고 있으나, 검색 품질을 정량적으로 평가할 수단이 부족함. 다음 개선 사항들을 평가하기 위해 Golden Set이 필요:

- Nori analyzer 효과 측정
- 하이브리드 가중치 튜닝 (현재 dense:sparse = 0.7:0.3)
- Multi-query expansion 효과
- Cross-encoder re-ranking 효과
- Multi-hop retrieval 개선

### 1.2 Golden Set이란?

**정의**: 질문-문서 관련성(relevance)을 사람이 라벨링한 정답 데이터셋

```
Golden Set = {
  "질문": "RFID 센서 교체 절차는?",
  "정답 문서": [
    {"doc_id": "sop_rfid#12", "relevance": 3},  # 필수
    {"doc_id": "ts_guide_rfid#5", "relevance": 2},  # 필요
    {"doc_id": "log_40001#3", "relevance": 1},  # 참고
  ]
}
```

**용도**:
- Recall@k, nDCG@k, MRR 같은 검색 품질 지표 계산
- 검색 시스템 변경 전/후 성능 비교
- A/B 테스트 기준

### 1.3 왜 Golden Set 구축이 어려운가?

**문제점**:
1. **전수 조사 불가능**: 338K+ chunks 전체를 사람이 볼 수 없음
2. **주관성**: 사람마다 "관련 있다"의 기준이 다름
3. **비용**: 질문 40개 × 문서 100개/질문 × 30초 = 33시간 소요
4. **편향**: 한 검색 방식만 사용하면 그 방식에 유리한 Golden Set이 됨

**해결책**: 정보검색 커뮤니티의 표준 방법론 적용
- **Pooling**: 여러 검색 방식의 상위 결과만 평가
- **Graded Relevance**: 0/1이 아닌 3~4단계로 구분
- **Dev/Test Split**: 평가용 데이터 고정
- **품질 관리**: 2인 라벨링으로 일치도 확인

---

## 2. 전략의 핵심 원칙

### 2.1 Multi-Relevant 인정

**원칙**: 하나의 질문에 여러 문서가 관련될 수 있음

```
질문: "RFID 센서 에러 해결 방법"

관련 문서:
- maintenance_log: 과거 해결 사례 (참고)
- ts_guide: 진단 절차 (필요)
- sop: 교체 절차 (필수)
- setup_manual: 센서 사양 (배경)

→ 모두 relevant하지만 "중요도"가 다름
```

### 2.2 Pooling으로 후보 축소

**문제**: 338K chunks를 모두 볼 수 없음

**해결**: 여러 검색 방식의 Top-N 합집합만 평가

```python
# 질문당 후보 pool 생성
pool = []
pool += bm25_search(query, top_k=50)           # 희소 검색
pool += dense_search(query, top_k=50)          # 밀집 검색
pool += hybrid_search(query, top_k=50)         # 하이브리드
pool += stratified_search(query, top_k=80)     # doc_type별
pool += tag_based_search(query, top_k=30)      # tag 확장

pool = deduplicate(pool)[:150]  # 최종 150개
```

**효과**:
- 338K → 150개로 축소 (99.96% 감소)
- 다양한 방식 포함 → 편향 감소
- 놓친 relevant 최소화

### 2.3 Dev/Test 분리

**원칙**: 평가용 데이터는 고정, 개발용 데이터는 확장 가능

```
전체 40개 질문
├─ Dev Set (30개): 개발/튜닝용, 계속 확장 가능
└─ Test Set (10개): 평가 전용, v1로 freeze

모델 변경 시:
- Dev로 튜닝 (가중치 조정, 파라미터 실험)
- Test로 최종 평가 (공정 비교)
```

**버전 관리**:
- Test v1 (2026-01-15): 초기 버전, freeze
- Test v2 (2026-04-01): 분기별 리프레시 (문서 대규모 변경 시)
- Dev: 지속 확장 (새 query/document 추가)

### 2.4 Chunk vs Document 단위 분리 ⚠️ 중요

**문제**: 현재 시스템은 "chunk 단위" 인덱싱이지만, 정답은 "문서/섹션 단위"인 경우가 많음

```
예: "RFID 센서 교체 SOP"
- 문서: global_sop_supra_xp_all_pm_rfid
- Chunk: #0012, #0013, #0014 (3개로 분할됨)

라벨링 시:
- "이 문서가 relevant" (문서 전체)
- 하지만 qrels는 chunk 단위로 저장해야 함
```

**해결책: 안정적인 메타데이터 추가**

qrels에 다음 필드를 **반드시** 포함:

```json
{
  "chunk_id": "global_sop_supra_xp_all_pm_rfid#0012",  // 평가 단위
  "parent_doc_id": "global_sop_supra_xp_all_pm_rfid",  // 문서 단위
  "section_id": "3. RFID 센서 교체 절차",               // 섹션 단위
  "doc_type": "sop",
  "device_name": "SUPRA XP",
  "module": "PM",
  "chapter": "RFID Sensor",
  "version": "Rev. 1.2",
  "last_updated": "2024-08-15"
}
```

**효과**:
- Chunk 재분할 시 parent_doc_id로 매핑 가능
- 문서 버전 변경 추적
- Doc-level aggregation 가능 (chunk 평가 → doc 평가)

**Stable ID 전략** (Optional, 고급):

```
현재: chunk_id = f"{doc_id}#{sequential_number}"
      → 재인덱싱 시 번호 바뀜 (불안정)

권장: chunk_id = f"{doc_id}#{section_anchor}#{content_hash[:8]}"
      → 섹션 기준 + 내용 해시 (재인덱싱에 강건)

예: global_sop_rfid#section3_replacement#a3f8b2d1
```

---

## 3. Pooling 방법론

### 3.1 기본 Pooling (5가지 방식)

```python
def create_pool(query: str, retriever, top_k=50) -> list:
    """질문당 후보 pool 생성"""
    pools = []

    # 1. BM25 (희소 검색)
    pools.append(retriever.search(
        query,
        method="bm25",
        top_k=top_k
    ))

    # 2. Dense (밀집 검색)
    pools.append(retriever.search(
        query,
        method="dense",
        top_k=top_k
    ))

    # 3. Hybrid (현재 운영 방식)
    pools.append(retriever.search(
        query,
        method="hybrid",
        dense_weight=0.7,
        sparse_weight=0.3,
        top_k=top_k
    ))

    # 4. Query 확장 (동의어/약어)
    expanded_query = expand_query(query)  # 예: "점검" → "점검 정기점검 check"
    pools.append(retriever.search(
        expanded_query,
        method="bm25",
        top_k=top_k
    ))

    # 5. Multi-query (LLM 생성)
    if enable_multi_query:
        variants = llm_generate_query_variants(query, n=2)
        for variant in variants:
            pools.append(retriever.search(variant, top_k=top_k//2))

    # Deduplicate
    return deduplicate(flatten(pools))
```

### 3.2 Stratified Pooling (Doc-type 다양성 보장)

**문제**: 단순 top-N은 dominant type에 편향됨

```
예: "RFID 센서 교체"
- Setup manual: 200개 (문서 많음)
- SOP: 50개
- TS-guide: 30개
- Log: 10개

→ Pool에 setup manual만 가득 참
```

**해결 A: 고정 Quota 방식 (기본)**

```python
def stratified_pool(query: str, retriever, type_quotas: dict) -> list:
    """Doc-type별 다양성 보장"""
    pool = []

    # Type별로 검색
    for doc_type, quota in type_quotas.items():
        results = retriever.search(
            query,
            filters={"doc_type": doc_type},
            top_k=quota
        )
        pool.extend(results)

    return pool

# 사용 예시
type_quotas = {
    "sop": 30,           # SOP 30개
    "ts_guide": 25,      # TS-guide 25개
    "maintenance_log": 20,     # 정비로그 20개
    "setup": 20,         # Setup manual 20개
}
pool = stratified_pool(query, retriever, type_quotas)  # 총 95개
```

**해결 B: Min Quota 방식 (권장, 더 유연)** ⭐

문제: 어떤 질문은 특정 doc_type이 거의 무관할 수 있음 (고정 quota는 비효율)

```python
def stratified_pool_min_quota(
    query: str,
    retriever,
    total_pool_size: int = 150,
    min_per_type: int = 10
) -> list:
    """전체 pool에서 type별 최소 보장 + 나머지는 전체 랭킹"""

    # Step 1: 전체 검색 (hybrid)
    all_results = retriever.search(query, top_k=total_pool_size)

    # Step 2: Type별 카운트
    type_counts = defaultdict(int)
    for doc in all_results:
        type_counts[doc.metadata["doc_type"]] += 1

    # Step 3: 부족한 type만 추가 검색
    additional = []
    for doc_type in ["sop", "ts_guide", "maintenance_log", "setup"]:
        if type_counts[doc_type] < min_per_type:
            needed = min_per_type - type_counts[doc_type]
            extra = retriever.search(
                query,
                filters={"doc_type": doc_type},
                top_k=needed + 5  # 약간 여유
            )
            # 중복 제외하고 추가
            for doc in extra:
                if doc.chunk_id not in {d.chunk_id for d in all_results}:
                    additional.append(doc)
                    if len(additional) >= needed:
                        break

    # Step 4: 병합 (전체 랭킹 우선 + 추가)
    final_pool = all_results + additional
    return final_pool[:total_pool_size]

# 효과:
# - 자연스러운 랭킹 보존 (전체 hybrid 우선)
# - Type별 최소 보장 (편향 방지)
# - 비효율 감소 (무관한 type 강제 안 함)
```

### 3.3 Tag 기반 확장

**목적**: Seed relevant 문서의 tag로 추가 후보 발굴

```python
def expand_by_tags(seed_docs: list, retriever, max_per_tag=20) -> list:
    """Tag 기반 확장 (연쇄 확장 방지)"""

    # Step 1: Seed 문서에서 tag 추출
    all_tags = []
    for doc in seed_docs:
        all_tags.extend(doc.metadata.get("chunk_keywords", []))
        all_tags.extend(doc.metadata.get("tags", []))

    # Step 2: Tag 필터링 (도메인 특화)
    filtered_tags = [
        tag for tag in all_tags
        if len(tag) <= 20                          # 단어 수준만
        and tag not in STOPWORD_TAGS               # "batch", "manual" 제외
        and (is_equipment(tag) or is_part(tag))    # 장비/부품명 우선
    ]

    # Step 3: 빈도 기반 상위 선택
    tag_counts = Counter(filtered_tags)
    top_tags = [tag for tag, _ in tag_counts.most_common(5)]  # 최대 5개

    # Step 4: Tag당 검색 (연쇄 확장 안 함!)
    expanded = []
    for tag in top_tags:
        results = retriever.search(
            query=tag,
            filters={"chunk_keywords": tag},
            top_k=max_per_tag
        )
        expanded.extend(results)

    return expanded

# 사용 예시
seed = hybrid_search(query, top_k=5)  # 상위 5개
expanded = expand_by_tags(seed, retriever, max_per_tag=20)
```

**주의사항**:
- ❌ 연쇄 확장 금지: Tag A → Doc B → Tag C → Doc D... (범위 폭발)
- ✅ 1단계만: Seed docs의 tag → 추가 docs (여기서 멈춤)
- ✅ Tag 품질 관리: 너무 일반적/너무 긴 tag 제외

### 3.4 Hard Negative Mining

**목적**: "비슷해 보이지만 답은 아닌" 문서 포함 (평가 엄격화)

```python
def mine_hard_negatives(query: str, relevant_docs: list, retriever, n=10) -> list:
    """같은 tag/장비인데 관련 없는 문서"""

    # Relevant 문서의 tag 추출
    tags = extract_tags(relevant_docs)
    device = relevant_docs[0].metadata.get("device_name")

    # Tag는 같은데 relevant 아닌 것
    candidates = retriever.search_by_tags(tags, top_k=100)

    hard_negatives = [
        doc for doc in candidates
        if doc.chunk_id not in {d.chunk_id for d in relevant_docs}
        and doc.metadata.get("device_name") == device  # 같은 장비
    ][:n]

    return hard_negatives

# 예시
query = "RFID 센서 교체 절차"
relevant = [sop_rfid_replacement]  # 교체 SOP
hard_neg = mine_hard_negatives(query, relevant, retriever)
# → sop_rfid_cleaning (청소 SOP, 교체 아님)
# → sop_rfid_calibration (교정 SOP, 교체 아님)
```

### 3.5 Near-Duplicate 제거 ⚠️ 중요 (특히 Log 문서)

**문제**: 정비 로그(maintenance_log)는 중복/유사 chunk가 많아 pool이 도배됨

```
예: "RFID 센서 에러" 검색
→ Pool 150개 중 100개가 거의 동일한 로그
→ 라벨링 시간 낭비 + 다양성 저하
```

**해결: Pool 생성 단계에서 near-duplicate clustering**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def deduplicate_pool_with_clustering(
    pool: list,
    similarity_threshold: float = 0.85,
    doc_type_specific: dict = None
) -> list:
    """Near-duplicate를 clustering해서 대표 1개만 남김"""

    if doc_type_specific is None:
        doc_type_specific = {
            "maintenance_log": 0.90,  # 로그는 더 엄격
            "sop": 0.75,              # SOP는 덜 엄격
            "ts_guide": 0.75,
            "setup": 0.70
        }

    # Doc-type별로 분리
    by_type = defaultdict(list)
    for doc in pool:
        doc_type = doc.metadata.get("doc_type", "other")
        by_type[doc_type].append(doc)

    deduplicated = []

    for doc_type, docs in by_type.items():
        if len(docs) <= 1:
            deduplicated.extend(docs)
            continue

        # TF-IDF로 유사도 계산
        texts = [d.content for d in docs]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)

        # Clustering (greedy)
        threshold = doc_type_specific.get(doc_type, similarity_threshold)
        used = set()
        clusters = []

        for i in range(len(docs)):
            if i in used:
                continue

            cluster = [i]
            for j in range(i+1, len(docs)):
                if j not in used and sim_matrix[i, j] >= threshold:
                    cluster.append(j)
                    used.add(j)

            clusters.append(cluster)

        # 각 cluster에서 대표 1개 선택 (랭킹 우선)
        for cluster in clusters:
            representative = docs[cluster[0]]  # 첫 번째 = 랭킹 높은 것
            deduplicated.append(representative)

    return deduplicated

# 사용 예시
pool_raw = create_pool_for_query(query, retriever, top_k=200)
pool_final = deduplicate_pool_with_clustering(pool_raw)[:150]
# → 로그 중복 제거 후 150개 유지
```

**Alternative: Simhash/MinHash (더 빠름, 대용량용)**

```python
from simhash import Simhash

def quick_deduplicate(pool: list, hamming_threshold: int = 3) -> list:
    """Simhash로 빠르게 중복 제거"""

    hashes = {}
    deduplicated = []

    for doc in pool:
        h = Simhash(doc.content)

        # 기존 hash와 비교
        is_duplicate = False
        for existing_hash in hashes.keys():
            if h.distance(existing_hash) <= hamming_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            hashes[h] = doc
            deduplicated.append(doc)

    return deduplicated
```

**효과**:
- 라벨링 시간 30~50% 절약
- Pool 다양성 증가
- Annotator 피로도 감소

### 3.6 최종 Pool 크기 권장

```
질문 난이도별 pool 크기:

Easy (키워드 명확):    80~100개
Medium (일반):         120~150개
Hard (Multi-hop):      150~200개

평균: 150개 권장
```

**이유**:
- 너무 작으면 (<50): Relevant 놓칠 위험
- 너무 크면 (>200): 라벨링 비용 증가
- 150개 × 30초 = 75분/질문 (적당)

---

## 4. Graded Relevance

### 4.1 왜 0/1이 아닌 다단계인가?

**문제: Binary relevance (0/1)**

```
질문: "RFID 센서 교체 절차"

문서 A: SOP - 교체 절차 직접 기술
문서 B: TS-guide - 교체 전 진단 (사전 작업)

Binary로는:
- A = 1 (relevant)
- B = 1 (relevant)

→ A와 B가 동등? 아니야!
```

**해결: Graded relevance (0/1/2/3)**

```
문서 A: Grade 3 (Must-have)
문서 B: Grade 2 (Should-have)

→ nDCG@k 계산 시 A가 B보다 높은 점수
→ "필수 문서를 상위에 올리는 검색"을 정확히 평가
```

### 4.2 4-Level Grading 기준 (프로젝트 특화)

#### Grade 3: Must-have (필수)

**정의**: 이 문서만 있으면 질문 해결 가능

**기준**:
- 질문의 직접 답(절차, 해결책, 정확한 진단) 포함
- 핵심 정보가 명확히 기술됨
- 추가 문서 없이도 이해 가능

**예시**:

```yaml
질문: "RFID 센서 교체 절차는?"

Grade 3 문서:
- global_sop_supra_xp_all_pm_rfid#0012
  내용: "3. RFID 센서 교체 절차
         3-1. 센서 탈거 (나사 4개 제거)
         3-2. 새 센서 장착
         3-3. 통신 테스트"
  이유: 교체 절차가 단계별로 명시됨
```

#### Grade 2: Should-have (필요)

**정의**: 문제 해결에 필요한 배경지식/사전 작업

**기준**:
- 질문 답을 이해하는 데 필요한 정보
- 사전 확인/준비 사항
- 관련 개념/원리 설명

**예시**:

```yaml
질문: "RFID 센서 교체 절차는?"

Grade 2 문서:
- supra_xp_ts_guide_rfid_abnormal#0008
  내용: "RFID 센서 고장 진단
         - 통신 확인 방법
         - 전원 체크
         - 교체 필요 여부 판단"
  이유: 교체 전 고장 확인 필요 (사전 작업)

- global_sop_safety_esd#0003
  내용: "정전기 방지 절차
         - 접지 확인
         - ESD 밴드 착용"
  이유: 센서 교체 시 안전 수칙 (필수 배경)
```

#### Grade 1: Nice-to-have (참고)

**정의**: 관련 사례/추가 정보

**기준**:
- 유사 작업 사례
- 관련 부품/장비 정보
- 있으면 도움되지만 필수 아님

**예시**:

```yaml
질문: "RFID 센서 교체 절차는?"

Grade 1 문서:
- maintenance_log_40001648#0002
  내용: "EPAH54 RFID 센서 교체 작업
         - 작업자: 김OO
         - 소요시간: 30분
         - 특이사항: 케이블 단자 부식 발견"
  이유: 과거 교체 사례 (참고용)

- global_sop_integer_rfid#0005
  내용: "Integer Plus RFID 교체 절차"
  이유: 다른 장비지만 절차 유사 (참고용)
```

#### Grade 0: Not relevant (무관)

**정의**: 키워드만 겹치고 실질적 도움 안 됨

**기준**:
- 다른 작업/다른 문제
- 키워드 우연히 포함
- 질문 해결에 기여 안 함

**예시**:

```yaml
질문: "RFID 센서 교체 절차는?"

Grade 0 문서:
- global_sop_supra_xp_all_pm_rfid_cleaning#0007
  내용: "RFID 센서 청소 절차"
  이유: 청소는 교체가 아님 (다른 작업)

- setup_manual_geneva_xp#0234
  내용: "Geneva XP RFID 센서 사양
         - 통신: RS-485
         - 전압: 24V"
  이유: 사양 정보만, 교체 절차 없음
```

### 4.3 Edge Cases

#### Case 1: 버전/장비 차이

```yaml
질문: "SUPRA XP RFID 센서 교체"
문서: "SUPRA N RFID 센서 교체" (다른 시리즈)

판단:
- 절차가 거의 동일 → Grade 1 (참고용)
- 절차가 크게 다름 → Grade 0 (무관)
```

#### Case 2: 부분 Overlap

```yaml
질문: "챔버 온도가 안 올라가는 이유"
문서: "챔버 히터 교정 절차"

판단:
- "원인" 포함 → Grade 2~3
- "해결책만" → Grade 1
- "교정만" (진단 없음) → Grade 0~1
```

#### Case 3: Multi-document 조합

```yaml
질문: "에러 E1234 해결 방법"

필요 문서:
- Log: E1234 발생 사례 → Grade 3 (현상 확인)
- TS-guide: E1234 진단 → Grade 3 (원인 파악)
- SOP: E1234 조치 → Grade 3 (해결책)

→ 각 단계에서 필수이므로 모두 Grade 3
```

---

## 5. Annotation 프로세스

### 5.1 Annotator (라벨 작업자)

**누가 하나?**

1순위: **정비 엔지니어** (도메인 전문가)
2순위: 문서 관리자 (어떤 문서가 관련있는지 아는 사람)
3순위: 데이터 과학자 (도메인 지식 학습 후)

**필요 인원**:
- 주 annotator: 1명
- 검증 annotator: 1명 (20% 중복 라벨링)

### 5.2 Annotation 도구

#### Option A: CLI 도구 (간단, 빠름)

```bash
# 실행
python scripts/golden_set/annotate.py \
  --queries data/golden_set/queries.jsonl \
  --pools data/golden_set/pools/ \
  --output data/golden_set/qrels_dev.jsonl \
  --mode interactive
```

**화면 예시**:

```
Query [1/40]: RFID 센서 교체 절차는?

Document [1/147]
─────────────────────────────────────────────────
ID: global_sop_supra_xp_all_pm_rfid#0012
Type: sop
Device: SUPRA XP

Content:
  3. RFID 센서 교체 절차
  3-1. 센서 탈거
    - 전원 차단 확인
    - 센서 고정 나사 4개 제거
  ...

─────────────────────────────────────────────────
Grade (0/1/2/3/skip): 3
Rationale (optional): 교체 절차 직접 기술
Snippet (optional): 3-1. 센서 탈거...

✓ Saved! [1/147 done, 75 min remaining]
```

#### Option B: Web UI (고급, 편리)

```bash
# 서버 실행
python scripts/golden_set/annotation_server.py

# 브라우저에서 http://localhost:5000 접속
```

**기능**:
- 문서 full-text 검색
- 이전/다음 탐색
- 진행률 tracking
- 여러 annotator 동시 작업

### 5.3 LLM 보조 (Optional)

**목적**: 라벨링 시간 단축 (33시간 → 15시간)

```python
def llm_assisted_annotation(query, document, llm):
    """LLM이 초안 생성 → 사람이 검토"""

    # LLM에게 draft grade 요청
    prompt = f"""
질문: {query}

문서:
{document.content[:500]}

이 문서가 질문에 답하는 데 도움이 되나요?
0: 도움 안 됨
1: 참고 정도
2: 필요한 배경지식
3: 직접 답 포함

JSON 형식으로 답변:
{{"grade": 0~3, "reason": "간단한 이유"}}
"""

    response = llm.generate(prompt, max_tokens=100)
    draft = json.loads(response)

    # 사람에게 보여주기
    print(f"🤖 LLM 제안: Grade {draft['grade']}")
    print(f"   이유: {draft['reason']}")
    print("\n👤 LLM 판단이 맞나요?")

    user_input = input("  [y] 맞음  [n] 틀림  [v] 문서 전체 보기: ")

    if user_input == 'y':
        final_grade = draft['grade']  # LLM 제안 승인
    elif user_input == 'v':
        print(document.content)  # 전체 보기
        final_grade = input("  Grade (0/1/2/3): ")
    else:
        final_grade = input("  정확한 grade: ")

    return {
        "grade": int(final_grade),
        "llm_draft": draft['grade'],
        "llm_agreed": (final_grade == draft['grade'])
    }
```

**효과**:
- LLM이 80% 맞추면 → 시간 55% 절약 (33h → 15h)
- 비용: GPT-4o-mini 기준 $0.30 (400원)

**주의**:
- ⚠️ LLM 판단을 그대로 쓰면 안 됨 (사람이 최종 결정)
- ✅ "초안" 또는 "두 번째 의견"으로만 사용

### 5.4 저장 포맷

```jsonl
// data/golden_set/qrels_dev.jsonl (JSONL format, 1 query per line)
{
  "query_id": "q001",
  "query_text": "RFID 센서 교체 절차는?",
  "query_type": "procedure",
  "device_name": "SUPRA XP",
  "difficulty": "easy",

  "relevance_judgments": [
    {
      "chunk_id": "global_sop_supra_xp_all_pm_rfid#0012",
      "doc_id": "global_sop_supra_xp_all_pm_rfid",
      "grade": 3,
      "rationale": "교체 절차 직접 기술",
      "supporting_snippet": "3-1. 센서 탈거 - 고정 나사 4개 제거...",
      "annotator": "engineer_kim",
      "llm_draft": 3,
      "llm_agreed": true,
      "annotated_at": "2026-01-07T10:30:00"
    },
    {
      "chunk_id": "supra_xp_ts_guide_rfid#0008",
      "doc_id": "supra_xp_ts_guide_rfid",
      "grade": 2,
      "rationale": "교체 전 진단 절차 (사전 작업)",
      "supporting_snippet": "RFID 통신 확인...",
      "annotator": "engineer_kim",
      "llm_draft": 2,
      "llm_agreed": true,
      "annotated_at": "2026-01-07T10:32:00"
    }
  ],

  "pool_info": {
    "pool_size": 147,
    "methods": ["bm25", "dense", "hybrid", "stratified", "tag-based"],
    "annotated_count": 8
  }
}
```

**TREC format 변환**:

```bash
# Standard TREC qrels format으로 export
python scripts/golden_set/export_trec.py \
  --input data/golden_set/qrels_dev.jsonl \
  --output data/golden_set/qrels_dev.trec

# Output: qrels_dev.trec
# query_id 0 doc_id relevance
q001 0 global_sop_supra_xp_all_pm_rfid#0012 3
q001 0 supra_xp_ts_guide_rfid#0008 2
q001 0 maintenance_log_40001#3 1
...
```

---

## 6. 품질 관리

### 6.1 Annotation Guidelines 문서화

**목적**: 사람마다 판단 기준이 달라지는 것 방지

**필수 내역**:

```markdown
# Annotation Guidelines

## 1. Relevance 정의

"문서가 질문 해결에 실질적으로 기여하는가?"

- 키워드만 겹치는 것 ≠ relevant
- 실제로 엔지니어가 읽고 도움받는가?

## 2. Grade별 기준

[위의 4.2절 내용 복사]

## 3. 판단 원칙

- 문서 전체를 읽고 판단 (제목/첫 문장만 보고 결정 금지)
- 애매하면 낮은 등급 선택 (precision 우선)
- 장비/버전이 다르면 1등급 낮춤

## 4. 자주 묻는 질문

Q: 절차는 없고 사양만 있으면?
A: Grade 0~1 (질문이 "사양"을 물으면 3, "절차"를 물으면 0)

Q: 다른 장비지만 절차가 유사하면?
A: Grade 1 (참고용)

...
```

### 6.2 Inter-Annotator Agreement (2인 라벨링)

**목적**: 가이드라인이 명확한지 검증

**방법**: 전체 40개 중 8개(20%)를 2명이 독립적으로 라벨링

```python
# 같은 질문/문서를 2명이 각각 라벨링
annotator_A = {
    "doc1": 3,
    "doc2": 2,
    "doc3": 1,
    "doc4": 0,
    "doc5": 0,
}

annotator_B = {
    "doc1": 3,  # ✓ 일치
    "doc2": 2,  # ✓ 일치
    "doc3": 0,  # ✗ 불일치 (A=1, B=0)
    "doc4": 0,  # ✓ 일치
    "doc5": 1,  # ✗ 불일치 (A=0, B=1)
}

# 일치도 계산
from sklearn.metrics import cohen_kappa_score

grades_A = [3, 2, 1, 0, 0]
grades_B = [3, 2, 0, 0, 1]

kappa = cohen_kappa_score(grades_A, grades_B, weights="linear")
# kappa = 0.75
```

**해석**:

```
κ = 1.0      : 완벽한 일치
κ = 0.8~1.0  : 매우 좋음 (가이드라인 우수)
κ = 0.6~0.8  : 좋음 (일부 조정 필요)
κ = 0.4~0.6  : 보통 (가이드라인 개선 필요)
κ < 0.4      : 나쁨 (가이드라인 재작성)
```

**조치**:

```python
if kappa < 0.6:
    # 불일치 케이스 분석
    disagreements = find_large_disagreements(A, B, threshold=2)

    for case in disagreements:
        print(f"Doc: {case['doc_id']}")
        print(f"  A: {case['grade_A']} (이유: {case['reason_A']})")
        print(f"  B: {case['grade_B']} (이유: {case['reason_B']})")

        # 토론 후 가이드라인 업데이트
        guideline_update = discuss(case)

    # 재라벨링
    re_annotate_with_updated_guidelines()
```

### 6.3 Quality Check Metrics

```python
# 1. Coverage check
python scripts/golden_set/check_coverage.py \
  --qrels data/golden_set/qrels_dev.jsonl

# Output:
# Doc type distribution:
#   sop: 42% (target: 30-40%)
#   ts_guide: 28% (target: 25-30%)
#   maintenance_log: 18% (target: 15-20%)
#   setup: 12% (target: 10-15%)
#   ✓ Balanced!
#
# Device distribution:
#   SUPRA: 52% (corpus: 50%)
#   Integer: 22% (corpus: 20%)
#   ...
#   ✓ Representative!
#
# Grade distribution:
#   0: 65% (most are not relevant)
#   1: 12%
#   2: 13%
#   3: 10%
#   ✓ Reasonable!
```

```python
# 2. Pool utilization
# Pool에서 실제로 몇 개가 relevant로 라벨링되었나?

total_docs_in_pool = 40 queries × 150 docs = 6,000
relevant_docs = sum(grade >= 1) = 600

recall_estimate = 600 / 6000 = 10%

# 10%가 적당 (5~15% 권장)
# - 너무 높으면 (>20%): Pool 너무 작음, relevant 놓칠 위험
# - 너무 낮으면 (<5%): Pool 너무 큼, 비효율
```

---

## 7. 프로젝트 특화 사항

### 7.1 질문 40개 구성 전략

**유형별 분포**:

```yaml
1. 절차 질문 (How-to): 15개
   예: "RFID 센서 교체 절차는?"
   Relevant: SOP (grade 3) + TS-guide (grade 2)

2. 진단 질문 (Why/What): 10개
   예: "챔버 온도가 안 올라가는 이유는?"
   Relevant: TS-guide (grade 3) + Log (grade 2)

3. 사례 질문 (Past issues): 5개
   예: "EFEM 로봇 에러 E1234 해결 사례"
   Relevant: Log (grade 3) + SOP (grade 2)

4. 부품 정보 (Spec/Location): 5개
   예: "Pirani Gauge 교체 주기는?"
   Relevant: Setup manual (grade 3)

5. Multi-hop 질문: 5개
   예: "로그 #40001648의 에러를 해결하려면?"
   Relevant: Log (step 1) + SOP (step 2)
```

**장비별 분포** (문서 corpus 비율 반영):

```yaml
SUPRA 시리즈: 50% (20개)
Integer Plus: 20% (8개)
Precia: 15% (6개)
Geneva: 10% (4개)
기타: 5% (2개)
```

**난이도 분포**:

```yaml
Easy (키워드 매칭으로 해결): 10개
  예: "Pirani Gauge SOP"

Medium (의미 이해 필요): 20개
  예: "진공이 떨어지는 이유"

Hard (멀티홉/문맥 필요): 10개
  예: "로그의 에러코드로 SOP 찾기"
```

### 7.2 Multi-hop 질문 라벨링

**특수 형식**: 단계별 relevant 라벨링

```json
{
  "query_id": "q036",
  "query_text": "로그 #40001648의 에러를 해결하려면 어떤 SOP를 참고해야 하나?",
  "query_type": "multihop",

  "hops": [
    {
      "hop": 1,
      "sub_task": "로그에서 에러코드/증상 추출",
      "relevant": [
        {
          "chunk_id": "40001648#0001",
          "grade": 3,
          "rationale": "에러 발생 로그"
        }
      ]
    },
    {
      "hop": 2,
      "sub_task": "에러코드로 SOP 검색",
      "relevant": [
        {
          "chunk_id": "global_sop_supra_vplus_pm_controller#0023",
          "grade": 3,
          "rationale": "Controller PM SOP"
        }
      ]
    }
  ],

  "relevance_judgments": [
    // Hop 1+2 통합 (end-to-end 평가용)
    {"chunk_id": "40001648#0001", "grade": 3},
    {"chunk_id": "global_sop_supra_vplus_pm_controller#0023", "grade": 3}
  ]
}
```

**평가 방식**:

```python
# Option 1: End-to-end 평가 (전체 relevant로 평가)
recall_e2e = len(retrieved ∩ all_hops_relevant) / len(all_hops_relevant)

# Option 2: Hop-by-hop 평가 (단계별로 평가)
for hop in hops:
    recall_hop = len(retrieved ∩ hop.relevant) / len(hop.relevant)
```

### 7.3 한국어 형태소 변형 테스트

**목적**: Nori analyzer 효과 측정

```json
{
  "query_id": "q001",
  "query_text": "점검 절차",
  "query_variants": [
    "점검 절차",        // 원형
    "점검하는 절차",    // 동사형
    "점검중인 절차",    // 진행형
    "정기점검 절차",    // 복합어
    "점검하다"          // 동사 원형
  ],

  "relevance_judgments": [
    // 동일한 qrels 사용
  ]
}
```

**평가**:

```python
# 각 variant의 recall 비교
for variant in query_variants:
    results = retriever.search(variant, top_k=10)
    recall = calculate_recall(results, qrels)
    print(f"{variant}: Recall@10 = {recall:.2f}")

# Nori 있을 때:
# "점검 절차": 0.90
# "점검하는 절차": 0.88 (slightly lower, but still good)
# "점검중인 절차": 0.85
# → Nori가 형태소 정규화 잘 함

# Nori 없을 때 (standard analyzer):
# "점검 절차": 0.90
# "점검하는 절차": 0.45 (much lower!)
# "점검중인 절차": 0.30
# → 형태소 변형에 취약
```

### 7.4 Hybrid Weight 분석

**목적**: Query별로 BM25/Dense 중 어느 쪽이 더 좋은지 파악

```python
def analyze_method_preference(qrels, retriever):
    """Query별로 어떤 method가 더 잘 찾았는지"""

    for query_id, judgments in qrels.items():
        query = get_query_text(query_id)
        relevant_ids = {j["chunk_id"] for j in judgments if j["grade"] >= 2}

        # BM25 vs Dense 비교
        bm25_results = retriever.search(query, method="bm25", top_k=10)
        dense_results = retriever.search(query, method="dense", top_k=10)

        bm25_recall = len(relevant_ids & set(bm25_results)) / len(relevant_ids)
        dense_recall = len(relevant_ids & set(dense_results)) / len(relevant_ids)

        if bm25_recall > dense_recall + 0.1:
            print(f"{query_id}: BM25-favored (keyword-heavy)")
            # → sparse_weight 올려야 함
        elif dense_recall > bm25_recall + 0.1:
            print(f"{query_id}: Dense-favored (semantic)")
            # → dense_weight 올려야 함
        else:
            print(f"{query_id}: Balanced")

# 결과 예시:
# q001 (RFID 센서 교체): BM25-favored (specific term)
# q005 (온도가 안 올라가는 이유): Dense-favored (semantic)
# q010 (에러 E1234): BM25-favored (error code)
#
# → Dynamic weighting 전략 수립 가능
#   - Error code 포함 → sparse_weight 0.5
#   - Why/How 질문 → dense_weight 0.8
```

---

## 8. 실행 계획

### Phase 0: 준비 (1일)

**작업 내용**:

```bash
# 1. 질문 40개 준비
python scripts/golden_set/prepare_queries.py \
  --source user_logs \
  --output data/golden_set/queries.jsonl \
  --distribution "procedure:15,diagnosis:10,case:5,spec:5,multihop:5"

# 2. Annotation 가이드라인 작성
# → docs/golden_set_annotation_guidelines.md

# 3. 도구 설치
pip install sentence-transformers scikit-learn
```

**산출물**:
- `data/golden_set/queries.jsonl`: 40개 질문
- `docs/golden_set_annotation_guidelines.md`: 라벨링 가이드
- Annotation 스크립트 준비

---

### Phase 1: Pooling (2일)

**작업 내용**:

```bash
# 각 query에 대해 pool 생성
python scripts/golden_set/create_pools.py \
  --queries data/golden_set/queries.jsonl \
  --output data/golden_set/pools/ \
  --methods "bm25,dense,hybrid,stratified,tag-based" \
  --pool-size 150

# 실행 예시
# Query q001: "RFID 센서 교체 절차"
#   BM25: 50 docs
#   Dense: 50 docs
#   Hybrid: 50 docs
#   Stratified: 95 docs (sop:30, ts:25, log:20, setup:20)
#   Tag-based: 40 docs (tags: RFID, sensor, PM)
#   → Dedup: 147 docs
#
# Progress: [40/40] 100% | ETA: 0s
# Saved pools to data/golden_set/pools/*.jsonl
```

**산출물**:
- `data/golden_set/pools/q001.jsonl` ~ `q040.jsonl`: 각 질문당 pool
- `data/golden_set/pool_stats.json`: Pool 통계

---

### Phase 2: Annotation (1주)

#### 2.1 주 Annotator 작업 (5일)

**방법 A: CLI 도구**

```bash
python scripts/golden_set/annotate.py \
  --queries data/golden_set/queries.jsonl \
  --pools data/golden_set/pools/ \
  --output data/golden_set/qrels_annotator_A.jsonl \
  --annotator engineer_kim \
  --mode interactive

# 진행 상황
# [Query 1/40] RFID 센서 교체 절차
#   [Doc 1/147] ... Grade: 3 ✓
#   [Doc 2/147] ... Grade: 0 ✓
#   ...
#   Progress: 8/147 (5.4%) | Time: 3m | ETA: 52m
```

**방법 B: LLM 보조 (권장)**

```bash
python scripts/golden_set/annotate.py \
  --queries data/golden_set/queries.jsonl \
  --pools data/golden_set/pools/ \
  --output data/golden_set/qrels_annotator_A.jsonl \
  --annotator engineer_kim \
  --mode llm-assisted \
  --llm-endpoint http://localhost:8003/v1

# LLM이 먼저 draft 생성 (1시간)
# → Annotator가 검토 (15시간)
# → 총 16시간 (vs. 순수 사람: 33시간)
```

#### 2.2 검증 Annotator 작업 (2일)

```bash
# 8개 질문(20%)만 중복 라벨링
python scripts/golden_set/annotate.py \
  --queries data/golden_set/queries_overlap.jsonl \
  --pools data/golden_set/pools/ \
  --output data/golden_set/qrels_annotator_B.jsonl \
  --annotator engineer_park \
  --mode interactive
```

**산출물**:
- `data/golden_set/qrels_annotator_A.jsonl`: 40개 전체
- `data/golden_set/qrels_annotator_B.jsonl`: 8개 중복

---

### Phase 3: Quality Check (2일)

```bash
# 1. Inter-annotator agreement
python scripts/golden_set/check_agreement.py \
  --annotator-a data/golden_set/qrels_annotator_A.jsonl \
  --annotator-b data/golden_set/qrels_annotator_B.jsonl \
  --output data/golden_set/agreement_report.json

# Output:
# Cohen's Kappa: 0.78 (Good!)
# Disagreements: 12 cases
# - q003, doc#45: A=2, B=1 (minor)
# - q007, doc#89: A=3, B=0 (major!) ← Review needed
```

```bash
# 2. Coverage check
python scripts/golden_set/check_coverage.py \
  --qrels data/golden_set/qrels_annotator_A.jsonl \
  --output data/golden_set/coverage_report.json

# Output:
# Doc type distribution:
#   sop: 40% ✓
#   ts_guide: 28% ✓
#   maintenance_log: 18% ✓
#   setup: 14% ✓
#
# Grade distribution:
#   Grade 3: 10%
#   Grade 2: 13%
#   Grade 1: 12%
#   Grade 0: 65%
#   ✓ Reasonable!
```

```bash
# 3. 불일치 해결
python scripts/golden_set/resolve_disagreements.py \
  --disagreements data/golden_set/agreement_report.json \
  --output data/golden_set/qrels_resolved.jsonl

# Interactive mode:
# Case 1: q007, doc#89
#   Annotator A: Grade 3 (이유: "절차 포함")
#   Annotator B: Grade 0 (이유: "다른 장비")
#
# 👤 Expert decision:
#   [a] A가 맞음
#   [b] B가 맞음
#   [c] 둘 다 아님 (새로 판단)
#
# 입력> b  ✓ Saved!
```

**산출물**:
- `data/golden_set/qrels_final.jsonl`: 불일치 해결된 최종 qrels
- `data/golden_set/agreement_report.json`: 품질 보고서

---

### Phase 4: Dev/Test Split (0.5일)

```bash
# Stratified split (query_type 비율 유지)
python scripts/golden_set/split_dev_test.py \
  --qrels data/golden_set/qrels_final.jsonl \
  --dev data/golden_set/qrels_dev.jsonl \
  --test data/golden_set/qrels_test_v1.jsonl \
  --split 0.75 \
  --stratify query_type \
  --seed 42

# Output:
# Dev set: 30 queries
#   - procedure: 12
#   - diagnosis: 8
#   - case: 4
#   - spec: 4
#   - multihop: 2
#
# Test set: 10 queries (FROZEN!)
#   - procedure: 3
#   - diagnosis: 2
#   - case: 1
#   - spec: 1
#   - multihop: 3
```

**버전 관리**:

```bash
# Test는 git으로 버전 관리
git add data/golden_set/qrels_test_v1.jsonl
git commit -m "feat: Add test golden set v1 (frozen)"
git tag golden-set-test-v1

# Dev는 계속 확장 가능
# (새 query/document 추가)
```

**산출물**:
- `data/golden_set/qrels_dev.jsonl`: 30 queries (확장 가능)
- `data/golden_set/qrels_test_v1.jsonl`: 10 queries (frozen)

---

### Phase 5: 평가 실행 (반복)

```bash
# Baseline 평가
python scripts/golden_set/evaluate.py \
  --qrels data/golden_set/qrels_test_v1.jsonl \
  --retriever-config configs/retrieval/baseline.yaml \
  --output results/baseline_test_v1.json

# Output: results/baseline_test_v1.json
{
  "config": {
    "method": "hybrid",
    "dense_weight": 0.7,
    "sparse_weight": 0.3,
    "re_ranker": null
  },

  "metrics": {
    "recall@5": 0.72,
    "recall@10": 0.85,
    "recall@20": 0.92,
    "ndcg@5": 0.68,
    "ndcg@10": 0.71,
    "mrr": 0.65
  },

  "by_query_type": {
    "procedure": {
      "recall@5": 0.78,
      "ndcg@5": 0.72
    },
    "diagnosis": {
      "recall@5": 0.65,
      "ndcg@5": 0.60
    },
    "multihop": {
      "recall@5": 0.55,
      "ndcg@5": 0.52
    }
  },

  "by_doc_type": {
    "sop": {"precision@5": 0.82},
    "ts_guide": {"precision@5": 0.71},
    "maintenance_log": {"precision@5": 0.58}
  }
}
```

**개선 실험 예시**:

```bash
# 실험 1: Dense weight 올리기
python scripts/golden_set/evaluate.py \
  --qrels data/golden_set/qrels_test_v1.jsonl \
  --retriever-config configs/retrieval/dense_heavy.yaml \
  --output results/dense_heavy_test_v1.json

# 실험 2: Re-ranker 추가
python scripts/golden_set/evaluate.py \
  --qrels data/golden_set/qrels_test_v1.jsonl \
  --retriever-config configs/retrieval/with_reranker.yaml \
  --output results/with_reranker_test_v1.json

# 비교
python scripts/golden_set/compare_results.py \
  --baseline results/baseline_test_v1.json \
  --experiments results/dense_heavy_test_v1.json \
                results/with_reranker_test_v1.json \
  --output results/comparison.md
```

**비교 결과**:

| Config | Recall@10 | nDCG@10 | 변화 |
|--------|-----------|---------|------|
| Baseline (0.7/0.3) | 0.85 | 0.71 | - |
| Dense heavy (0.8/0.2) | 0.87 | 0.73 | +2.4% ✓ |
| With re-ranker | 0.88 | 0.78 | +9.9% ✓✓ |

---

## 9. 비용 및 시간 추정

### 9.1 인력 투입

| Phase | 작업 | 인원 | 기간 | Person-days |
|-------|------|------|------|-------------|
| 0. 준비 | 질문/가이드라인 | 1 | 1일 | 1 |
| 1. Pooling | Pool 생성 | 1 | 2일 | 2 |
| 2. Annotation | 주 annotator | 1 | 5일 | 5 |
| | 검증 annotator | 1 | 2일 | 2 |
| 3. Quality | Agreement/Coverage | 1 | 2일 | 2 |
| 4. Split | Dev/Test 분리 | 1 | 0.5일 | 0.5 |
| **합계** | | | | **12.5** |

**실제 소요 기간**: 2주 (2인 병렬 작업)

### 9.2 비용

#### Option A: 순수 사람

```
주 annotator: 33시간 × 5만원/시간 = 165만원
검증 annotator: 7시간 × 5만원/시간 = 35만원

합계: 200만원
```

#### Option B: LLM 보조 (권장)

```
LLM 비용:
- 4,000 docs × 500 tokens = 2M input tokens
- 4,000 grades = 4K output tokens
- GPT-4o-mini: $0.15/1M in + $0.60/1M out
- 총: $0.30 + $0.003 = $0.30 (400원)

사람 비용:
- 주 annotator: 16시간 × 5만원 = 80만원 (55% 절약!)
- 검증 annotator: 7시간 × 5만원 = 35만원

합계: 115만원 + 400원 = 약 115만원

절약: 85만원 (42%)
```

### 9.3 컴퓨팅 비용

```
Pooling (ES query): 무시 가능 (<$1)
Dense embedding: 40 queries × 768 dim = 무시 가능
평가 (ranx/pytrec_eval): Local, 무료

총: <$1
```

### 9.4 총 비용

**추천 방식 (LLM 보조)**: 115만원 + 인프라 <$1 = **약 115만원**

---

## 10. 실무 함정 및 보완 ⚠️

이 섹션은 IR 평가 실무에서 흔히 겪는 함정들과 대응 방법을 정리함.

### 10.1 Incomplete Qrels (불완전한 정답 데이터) 문제

**문제**: Pooling 방식은 구조적으로 "unjudged documents"를 만듦

```
Pool에 포함 안 된 문서 = 라벨링 안 됨
→ 평가 시 "0점(not relevant)"으로 간주됨
→ 실제로는 relevant일 수 있음 (false negative)
```

**대응 A: Judged@k 지표를 항상 함께 리포트 (필수)**

```python
def calculate_judged_at_k(results: list, qrels: dict, k: int) -> float:
    """상위 k개 중 라벨링된 비율"""
    top_k_docs = results[:k]
    judged_count = sum(1 for doc in top_k_docs if doc.id in qrels)
    return judged_count / k

# 평가 리포트 예시
{
  "recall@10": 0.85,
  "judged@10": 0.90,  # ← 상위 10개 중 90%만 라벨링됨
  "unjudged@10": 0.10  # 10%는 판단 불가
}
```

**해석**:
- `judged@10` < 0.7: Pool 너무 작음, recall 신뢰 불가
- `judged@10` > 0.9: 적절함

**대응 B: Unjudged-robust 지표 병행 (권장)**

```python
# Bpref: unjudged에 덜 민감한 지표
from ranx import Qrels, Run, evaluate

qrels = Qrels(qrels_dict)
run = Run(results_dict)

metrics = evaluate(qrels, run, ["bpref", "ndcg@10", "recall@10"])
# bpref: unjudged를 무시하고 relevant/non-relevant만 비교
```

**도구**:
- `pytrec_eval`: bpref 지원
- `ranx`: 빠른 IR 평가 라이브러리
- `ir_measures`: 50+ 지표 지원

### 10.2 Graded Relevance 정의 충돌 방지

**충돌 A: Grade 3(필수) 남발**

```
Multi-hop에서:
- Log: 에러 확인 (hop 1 필수)
- TS-guide: 진단 (hop 2 필수)
- SOP: 조치 (hop 3 필수)

→ 모두 Grade 3? nDCG 변별력 저하!
```

**해결책**:

```yaml
원칙: Grade 3은 "최종 조치/정답"만

Multi-hop:
  - Hop 1,2 (중간 단계): Grade 2
  - Hop 3 (최종 해결): Grade 3

단일 질문:
  - 직접 답: Grade 3
  - 사전 작업/배경: Grade 2
  - 참고 사례: Grade 1
```

**충돌 B: 장비/버전 차이 처리**

```
질문: "SUPRA XP RFID 교체"
문서: "SUPRA N RFID 교체"

절차 동일 → Grade?
절차 유사 → Grade?
절차 다름 → Grade?
```

**해결책 (안전 리스크 고려)**:

```yaml
저위험 작업 (청소/점검):
  - 동일 시리즈: Grade 3
  - 다른 시리즈 (절차 동일): Grade 2
  - 다른 시리즈 (절차 유사): Grade 1

고위험 작업 (전원/가스/진공):
  - 동일 시리즈만: Grade 3
  - 다른 시리즈: Grade 0~1 (보수적)
  → 안전 사고 예방 우선
```

### 10.3 평가 도구 선택

**문제**: `sentence-transformers.InformationRetrievalEvaluator`는 하이브리드 평가에 불편

```python
# sentence-transformers는 embedding 중심
evaluator = InformationRetrievalEvaluator(...)
# → BM25, hybrid, re-ranking 평가가 어색함
```

**권장: Run-based 평가 (TREC 방식)**

```python
# Step 1: Retriever 결과를 "run" 파일로 저장
run = {}
for query_id, query_text in queries.items():
    results = retriever.search(query_text, top_k=100)
    run[query_id] = {doc.id: score for doc, score in results}

# Step 2: Qrels + Run으로 평가
from ranx import Qrels, Run, evaluate

qrels = Qrels.from_file("qrels_test_v1.trec")
run = Run(run)

metrics = evaluate(
    qrels, run,
    ["ndcg@10", "recall@10", "mrr", "bpref", "judged@10"]
)
```

**효과**:
- BM25/Dense/Hybrid/Re-rank 모두 동일 프레임워크
- 실험 자동화 쉬움
- TREC 표준 준수

**도구 비교**:

| 도구 | 장점 | 단점 |
|------|------|------|
| sentence-transformers | 간단, embedding 통합 | Hybrid 평가 불편 |
| pytrec_eval | 표준, bpref 지원 | Python 바인딩 느림 |
| ranx | 빠름, 현대적 API | 신규 라이브러리 |
| ir_measures | 50+ 지표 | 의존성 많음 |

**권장**: `ranx` (빠르고 사용 쉬움)

### 10.4 파일럿 우선 접근 (실패 확률 최소화)

**문제**: 40개 전체 라벨링 후 "가이드라인 잘못됐음" 발견 → 재작업

**해결: 6~8개 파일럿 먼저**

```
Phase 0: 파일럿 (3일)
  - 6~8개 질문 선택 (유형별 골고루)
  - Pool 생성 → 라벨링 → Agreement 확인
  - Disagreement 분석 → 가이드라인 수정

Phase 1: 전체 확장 (1주)
  - 수정된 가이드라인으로 32개 추가
  - Dev/Test split

효과:
  - 재작업 리스크 80% 감소
  - 가이드라인 품질 확보
  - 도구/프로세스 검증
```

**파일럿 질문 선택 기준**:

```yaml
필수 포함:
  - Procedure: 2개 (쉬움/어려움 각 1)
  - Diagnosis: 2개
  - Multi-hop: 1개
  - Log search: 1개

장비 분포:
  - SUPRA: 3개
  - Integer: 2개
  - 기타: 1개

난이도:
  - Easy: 2개
  - Medium: 3개
  - Hard: 1개
```

### 10.5 Doc-type 네이밍 통일 ⚠️

**문제 발견**: 문서에서 `myservice` / `maintenance_log` 혼용

```python
# ES 실제 필드값
doc_type = "maintenance_log"  # ← 실제 값

# 문서 예시에서
"myservice": 20,  # ← 다른 이름

→ 코드 작성 시 혼란!
```

**해결**:

```python
# 현재 프로젝트 실제 doc_type 확인
es.search(
    index="rag_chunks_dev_current",
    body={
        "size": 0,
        "aggs": {"types": {"terms": {"field": "doc_type.keyword"}}}
    }
)

# 결과에 따라 문서 통일
# 예: "maintenance_log"로 통일
```

**권장 네이밍 (확인 후 결정)**:

```python
DOC_TYPES = {
    "sop": "SOP (Standard Operating Procedure)",
    "ts_guide": "Troubleshooting Guide",
    "maintenance_log": "Maintenance Log",
    "setup": "Setup Manual"
}
```

---

## 11. Answer Quality 평가 확장

Retrieval 평가만으로는 불충분. 최종 답변 품질도 평가 필요.

### 11.1 Answer Rubric 추가

**방법**: qrels에 "답변이 포함해야 할 내용" 체크리스트 추가

```json
{
  "query_id": "q001",
  "query_text": "RFID 센서 교체 절차는?",

  "relevance_judgments": [...],  // 기존 retrieval qrels

  "answer_rubric": {
    "must_have": [
      "전원 차단 및 LOTO 확인",
      "센서 탈거 절차 (나사 제거, 케이블 분리)",
      "새 센서 장착 절차",
      "동작 확인 및 캘리브레이션"
    ],
    "must_not_have": [
      "전원 ON 상태에서 작업 지시",
      "다른 장비 시리즈 절차 혼용"
    ],
    "quality_criteria": {
      "safety_check": "필수",
      "step_by_step": "필수",
      "verification": "필수"
    }
  }
}
```

### 11.2 Oracle vs Retrieved 2트랙 평가

```python
# Track A: Oracle (검색 품질 영향 제거)
oracle_context = get_docs_by_ids(qrels[query_id].relevant_docs)
oracle_answer = llm.generate(query, context=oracle_context)

# Track B: Retrieved (실제 E2E)
retrieved_context = retriever.search(query, top_k=5)
retrieved_answer = llm.generate(query, context=retrieved_context)

# 평가
oracle_score = evaluate_answer(oracle_answer, rubric)
retrieved_score = evaluate_answer(retrieved_answer, rubric)

# 분석
if retrieved_score < oracle_score:
    print("검색이 문제")
else:
    print("생성 모델/프롬프트 개선 필요")
```

### 11.3 평가 지표

```python
# Rubric-based 평가
def evaluate_answer(answer: str, rubric: dict) -> dict:
    scores = {}

    # Must-have check
    must_have_count = sum(
        1 for item in rubric["must_have"]
        if item.lower() in answer.lower()
    )
    scores["must_have_coverage"] = must_have_count / len(rubric["must_have"])

    # Must-not-have check
    must_not_violations = sum(
        1 for item in rubric["must_not_have"]
        if item.lower() in answer.lower()
    )
    scores["safety_violations"] = must_not_violations

    # LLM-as-judge (optional)
    scores["llm_judge"] = llm_judge_quality(answer, rubric)

    return scores
```

**Test Rubric도 freeze**:

```
Dev rubric: 계속 확장 가능
Test rubric: v1 freeze (retrieval test와 동일)
```

---

## 12. 부록

### 12.1 디렉토리 구조

```
llm-agent-v2/
├─ data/
│  └─ golden_set/
│     ├─ queries.jsonl                  # 40개 질문
│     ├─ queries_overlap.jsonl          # 중복 라벨링용 8개
│     ├─ pools/
│     │  ├─ q001.jsonl                  # Query별 pool
│     │  ├─ q002.jsonl
│     │  └─ ...
│     ├─ qrels_dev.jsonl                # Dev set (30개)
│     ├─ qrels_test_v1.jsonl            # Test set v1 (10개, frozen)
│     ├─ qrels_test_v1.trec             # TREC format
│     └─ reports/
│        ├─ agreement_report.json
│        └─ coverage_report.json
│
├─ scripts/
│  └─ golden_set/
│     ├─ prepare_queries.py             # Phase 0
│     ├─ create_pools.py                # Phase 1
│     ├─ annotate.py                    # Phase 2
│     ├─ check_agreement.py             # Phase 3
│     ├─ check_coverage.py              # Phase 3
│     ├─ resolve_disagreements.py       # Phase 3
│     ├─ split_dev_test.py              # Phase 4
│     ├─ evaluate.py                    # Phase 5
│     ├─ compare_results.py             # Phase 5
│     └─ export_trec.py                 # 변환 도구
│
├─ docs/
│  ├─ golden_set_annotation_guidelines.md   # Annotation 가이드
│  └─ 2026-01-07_retrieval_golden_set_strategy.md  # 이 문서
│
└─ results/
   ├─ baseline_test_v1.json
   ├─ dense_heavy_test_v1.json
   └─ comparison.md
```

### 12.2 주요 라이브러리

```bash
# 필수 (평가)
pip install ranx                   # Run-based IR 평가 (권장, 10.3절 참고)
pip install scikit-learn           # cohen_kappa_score (Inter-annotator agreement)

# 필수 (검색)
pip install sentence-transformers  # Dense embedding

# Optional (LLM 보조)
pip install openai                 # LLM draft annotation

# Optional (추가 평가 지표)
pip install pytrec_eval            # bpref 등 TREC 표준 지표
pip install ir_measures            # 50+ IR 지표

# 평가 예시 (권장)
from ranx import Qrels, Run, evaluate
metrics = evaluate(qrels, run, ["ndcg@10", "recall@10", "bpref"])
```

### 12.3 평가 지표 정의

#### Recall@k

```
Recall@k = (상위 k개에 포함된 relevant 수) / (전체 relevant 수)

예:
- 전체 relevant: 5개
- 상위 10개에 포함: 4개
- Recall@10 = 4/5 = 0.8
```

#### nDCG@k (Normalized Discounted Cumulative Gain)

```
DCG@k = Σ(relevance_i / log2(position_i + 1))

예:
위치 1: grade 3 → 3 / log2(2) = 3.0
위치 2: grade 2 → 2 / log2(3) = 1.26
위치 3: grade 0 → 0 / log2(4) = 0
...

nDCG@k = DCG@k / Ideal_DCG@k
```

**특징**: Graded relevance 반영, 상위 랭킹 중요

#### MRR (Mean Reciprocal Rank)

```
RR = 1 / (첫 relevant 문서 위치)

예:
- 첫 relevant가 3위 → RR = 1/3 = 0.33
- 40개 질문 평균 → MRR
```

### 12.4 참고 자료

**학술 자료**:
- TREC (Text REtrieval Conference) methodology
- MS MARCO dataset 구축 방법론
- BEIR benchmark annotation guidelines

**도구**:
- ranx (권장): https://github.com/AmenRa/ranx
- pytrec_eval: https://github.com/cvangysel/pytrec_eval
- sentence-transformers: https://www.sbert.net/
- TREC format: https://trec.nist.gov/data/qrels_eng/

**프로젝트 문서**:
- `docs/2026-01-02_retrieval review.md`: 현재 retrieval 아키텍처
- `docs/2026-01-07_rlm_paper_review.md`: Multi-hop retrieval 개선

---

## 다음 단계

1. **질문 40개 초안 작성** (user logs 분석)
2. **Annotation guidelines 문서 작성**
3. **Pooling 스크립트 구현** (기존 es_search_service 확장)
4. **Annotation 도구 선택** (CLI vs Web UI)
5. **Phase 0~5 실행**

**문의**: 각 단계 구현 시 추가 가이드 필요 시 요청 바람.
