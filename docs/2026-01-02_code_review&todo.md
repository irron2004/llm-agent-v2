# 2026-01-02 Code Review & TODO

> 기준 문서: `docs/2026-01-02_retrieval review.md`
>
> 목적: 리트리벌 품질(Recall/Precision) 개선 + 운영 안정성 + “설정 주도(pluggable)” 구조로 컴포넌트 교체(LLM/토크나이저/MQ/reranker/retriever/agent)를 쉽게 만들기.

---

## 0) 현재 상태 요약(코드 기준)

- 설정 로딩: `.env` + Pydantic Settings (`RAG_*`, `SEARCH_*`, `VLLM_*`, `TEI_*` 등) 기반 (`backend/config/settings.py`).
- Search backend 분기: `SEARCH_BACKEND=local|es`에 따라 `SearchService(local index)` 또는 `EsSearchService(ES)`가 startup에서 주입됨 (`backend/api/main.py`).
- LLM/embedding/reranker/query-expander/retriever는 모두 레지스트리 패턴이 이미 존재(= 플러그인 확장 가능)하나, **일부는 런타임 wiring이 고정/불완전**:
  - 기본 LLM DI는 현재 `vllm`으로 고정 (`backend/api/dependencies.py`).
  - `RAG_RETRIEVAL_PRESET`은 존재하지만 실제로 preset 적용 로직이 없음(코드 preset/YAML preset 둘 다 있으나 런타임 사용이 제한적).
  - Multi-query/rerank는 local(SearchService)에서만 통합되어 있고, ES(EsSearchService) 경로에는 아직 통합되지 않음.
  - LangGraph agent는 그래프 레벨에서 MQ/재시도를 수행하며, 내부 search의 MQ/rerank는 강제로 끔(중복 방지 목적).

### ✅ 잘 되어 있는 부분

- **레지스트리(플러그인) 패턴 기반이 탄탄함:** LLM/Embedding/Retriever/Reranker/QueryExpander/Preprocessor 등이 “이름+버전 → 인스턴스”로 교체 가능.
- **Pydantic Settings로 설정 구조화:** 기본값이 명확하고, `.env`/환경변수로 재정의 가능.
- **FastAPI DI + 캐싱 패턴:** 무거운 객체를 `Depends()`로 주입하고 `@lru_cache`로 재사용(성능/일관성/테스트 용이성 측면 장점).

### ⚠️ 개선이 필요한 부분

- **설정 파일(YAML/JSON) 기반 프리셋 적용이 미완성:** `retrieval_preset`/`preset_loader.py`/`backend/config/presets/*.yaml`이 있으나 실제 런타임 wiring에 연결되지 않음.
- **런타임 교체(동적 재구성) 제약:** DI가 `@lru_cache` 기반이라 프로세스 실행 중 설정 변경만으로 교체가 되지 않음(실험/멀티파이프라인 운영에 불리).
- **검색 파이프라인 조립이 하드코딩:** 단계(전처리→확장→검색→머지→재랭킹)가 코드 내부 `if/else`로 고정되어 확장/삽입(예: 커스텀 토크나이저/게이팅/다단계 검색)이 어려움. 특히 ES 경로는 MQ/rerank 미통합.
- **프리셋 UX 부족:** “preset 하나로 구성 전환”을 CLI/API/UI에서 쉽게 하지 못함.
- **Agent 구성 설정화 부족:** LangGraph agent의 toolchain/flow/정책(시도 횟수, MQ/rerank 활용 방식 등)이 코드에 고정되어 운영/실험 전환 비용이 큼.
- **실행 방식이 다면적(FastAPI/CLI/노트북):** 설정 공유/재현성(“같은 preset으로 FastAPI/CLI가 동일 동작”)을 강화할 필요가 있음.

---

## 1) TODO (우선순위)

### P0 — 사실 확인/정합성(최우선)

- [x] **현재 운영(또는 dev) ES mapping/settings 스냅샷 확보**
  - 산출물:
    - `docs/es_mapping_snapshot_2026-01-02.json`
    - `docs/es_settings_snapshot_2026-01-02.json`
  - 스냅샷 기반 핵심 확인사항:
    - `embedding.dims = 768` + `index_options.type = int8_hnsw`
    - `chunk_summary`/`doc_description`/`chunk_keywords`는 `text`로 검색 가능
    - 다수 문자열 필드가 `text` + `.keyword` 멀티필드(동적 매핑) 형태 → 필터는 `.keyword` 사용 권장
    - settings에 analysis가 없어서 Nori analyzer 미사용(한국어 형태소 분석 불가)
    - `chunk_keywords.text` 같은 서브필드는 없음(쿼리/코드에서 필드명 정합성 필요)
- [x] **임베딩 차원 불일치 방지 가드레일 정리** ✅ 완료 (2026-01-02)
  - 목표: "인덱스 생성/인제스천/서빙" 전 구간에서 dims 일치가 자동 검증되도록 체크리스트/검증 로직 추가.
  - 체크포인트: `SEARCH_ES_EMBEDDING_DIMS` ↔ 실제 embedder dimension ↔ ES mapping dims.
  - 완료 내용:
    - 설정/매핑 기본값 통일 (1024 → 768)
    - `EsIndexManager.create_index()`에 `validate_dims` 검증 로직 추가
    - 검증 스크립트 작성: `scripts/validate_embedding_dimensions.py`
    - 현재 상태 검증: 모든 차원 768로 일치 확인
  - 산출물: `docs/2026-01-02_es_guardrails_snapshot.md`, `docs/2026-01-02_es_guardrails_improvements.md`
- [X] **alias/인덱스 네이밍 실태 점검** ✅ 완료 (2026-01-02)
  - 목표: "`rag_chunks_{env}_current`가 alias인지 실제 index인지"를 명확히 하고, 롤링 업데이트 전략이 실제로 작동하는지 확인.
  - rag_chunks_dev_current를 조회하게 코드를 작성해놓았음. 나중에 data의 버전이 바뀌더라도 코드는 그대로 유지하고, 바뀐 데이터를 조회할 수 있도록 하는 것이 목표.
  - 체크포인트: `_cat/aliases`, `_cat/indices`, EsIndexManager/ingest가 동일 규칙을 쓰는지.
  - 완료 내용:
    - 실태 확인: `rag_chunks_dev_current`가 **실제 인덱스**로 존재 (alias 아님)
    - 마이그레이션 스크립트 작성: `scripts/migrate_to_alias_strategy.py`
    - Dry-run 테스트 성공 (340K 문서, 5.5GB, dims=768)
    - 롤링 업데이트 전략 문서화
  - **다음 작업(실행 보류)**: alias 마이그레이션 실행
    - 순서: `--dry-run` 확인 ✅ → (필요 시 백업) → 실제 실행 → `_cat/aliases`로 검증
- [x] **ES hybrid 검색의 후보군(candidates) 제한 이슈 점검** ✅ 완료 (2026-01-02)
  - 현 구조는 `script_score`가 텍스트 쿼리를 기반으로 후보군을 제한할 수 있음(semantic-only recall 저하 가능).
  - 목표: 기본 hybrid 전략을 "RRF 또는 2-stage(벡터 후보 + BM25 후보 union → merge)"로 전환하는 방안 검토/PoC.
  - 완료 내용:
    - **RRF를 기본값으로 변경**: `EsHybridRetriever.use_rrf = True` (기존 False)
    - script_score vs RRF 비교 분석 문서화
    - 후보군 제한 이슈 해결: 벡터 검색과 BM25 검색이 독립 실행
  - 예상 효과: Semantic recall 향상 (특히 한국어 형태소 변형, 다국어 질의)
  - 롤백 가능: 환경변수 또는 코드에서 `use_rrf=False` 설정

### P1 — 검색 품질 개선(리콜/정밀도)

- [ ] **한국어 analyzer(Nori) 활성화 계획 수립 + 리인덱싱 절차 마련**
  - 작업:
    - ES nori plugin 준비(도커 이미지/배포 환경 포함).
    - `content/search_text/chunk_summary/chunk_keywords`에 대해 nori 적용(또는 `standard` + `nori` 멀티필드 병행) 설계.
    - alias 롤링 업데이트로 신규 인덱스 생성(v2) → 재색인 → alias 스위치.
- [ ] **필드 boost/멀티필드 전략 재정의**
  - 목표: `search_text` 원툴을 넘어, `content`, `chunk_summary`, `chunk_keywords`, (필요시) `title`/`doc_description`의 역할과 weight를 명확히.
  - 체크포인트: “쿼리에 포함하는 필드는 반드시 index=true” 보장(불필요 필드는 쿼리에서 제거).
- [ ] **Reranking(cross-encoder) 적용 범위/정책 정의**
  - 목표: 상위 N개 후보(예: 20~50)를 rerank 후 top_k 반환하는 정책을 설정으로 제어.
  - 고려사항: latency/비용/캐싱 전략, GPU/CPU 배치 환경, 장애 시 graceful fallback.
- [ ] **Multi-Query Expansion(MQE) 적용 범위/정책 정의**
  - 목표: “언제 MQE를 켤지”를 명확히(항상 on이 아니라 트리거/게이팅 기반 권장).
  - 예: 짧은 질의/모호 질의/라우팅 결과(ts/setup/general) 기반으로 MQE on.
- [ ] **ES 백엔드에도 MQE/Rerank 통합(또는 공통 파이프라인으로 일원화)**
  - 목표: `SEARCH_BACKEND=es`에서도 “확장→(복수 검색)→머지→(재랭킹)”을 설정으로 온/오프 가능하게.
  - 체크포인트: LangGraph agent가 MQ/rerank를 자체 수행하는 경우에는 중복 실행이 발생하지 않도록 정책/플래그 정리.

### P1 — 메타데이터 기반 정밀도 향상(필터링/타게팅)

- [ ] **`/search` API에 메타데이터 필터 파라미터 추가**
  - 대상: `doc_type`, `device_name`, `tenant_id`, `project_id`, `lang` 등.
  - 목표: UI/클라이언트에서 손쉽게 범위를 좁혀 precision 개선.
- [ ] **질의에서 엔티티(장비명/문서타입/알람코드 등) 추출 → 자동 필터 옵션화**
  - 목표: 사용자가 필터를 직접 고르지 않아도, 명시된 엔티티가 있으면 자동 적용(옵트아웃 가능).

### P2 — “설정 주도(pluggable)” 구조로 리팩터링

- [ ] **YAML/JSON 설정 파일 기반 구성 도입(프리셋 파일 선택)**
  - 목표: `.env`만으로는 관리가 어려운 “조합(embedding+retrieval+MQE+rerank+agent)”을 단일 파일로 정의/전환.
  - 제안: `RAG_PRESET_FILE=backend/config/presets/retrieval_full_pipeline.yaml` 또는 `RAG_PRESET_NAME=full_pipeline` 같은 형태로 런타임 선택.
  - 체크포인트: Pydantic Settings → preset overlay(덮어쓰기) 우선순위 규칙(ENV > preset > defaults) 명문화.
- [ ] **Preset 단일화(코드 preset vs YAML preset) 및 실제 런타임 적용**
  - 목표: `RAG_RETRIEVAL_PRESET` 하나로 아래가 일관되게 결정되도록:
    - retriever 종류(local/es), hybrid 방식(script_score vs rrf), top_k, weights, MQE, rerank.
  - 산출물: preset 목록/설명 조회 엔드포인트(또는 CLI) + “현재 활성 preset” introspection.
- [ ] **LLM/Embedder/Reranker/QueryExpander 선택을 설정으로 완전 외부화**
  - 목표: `RAG_LLM_METHOD/VERSION`, `RAG_RERANK_METHOD`, `RAG_QUERY_EXPAND_METHOD` 등으로 런타임 선택.
  - 체크포인트: FastAPI DI(`backend/api/dependencies.py`)가 “고정 구현”이 아니라 “설정 기반 팩토리”가 되도록 정리.
- [ ] **동적 재구성(실험용) 지원 여부 결정 + 구현**
  - 목표: “재시작 없이” 또는 “요청 단위로” 여러 파이프라인을 비교 실험할 수 있게 할지 결정.
  - 옵션:
    - 운영: 재시작 전제(현 구조 유지, 문서화 강화)
    - 실험: provider 패턴(키 기반 캐시 + reload) + 관리용 엔드포인트로 캐시 무효화
- [ ] **토크나이저/분석기 설정 경로 정리**
  - local BM25 tokenizer 주입 경로를 “설정 → 토크나이저 팩토리/레지스트리”로 통일.
  - chunking의 `split_by=token`이 실제로 동작하도록 tokenizer 전달(필요 시 embedder의 tokenizer 재사용).
  - ES는 analyzer가 토크나이저 역할을 하므로, “언어/분석기 프로파일”로 관리(standard/nori/synonyms).
- [ ] **검색 파이프라인 DSL/스텝 기반 조립(선택사항)**
  - 목표: “단계 추가/순서 변경/조건부 실행”을 코드 수정 없이 preset으로 실험 가능하게.
  - 산출물: `pipeline.steps`(preprocess/expand/embed/retrieve/merge/rerank 등) 스키마 + 실행기 + 로깅/프로파일링.
- [ ] **Agent( LangGraph )와 Search pipeline의 역할 분리/통합 정책 명문화**
  - 목표: “MQE는 그래프에서만 한다/서치에서만 한다/둘 다 한다(비권장)” 중 운영 표준을 결정하고 설정으로 제어.
  - 추가: agent용 top_k/모드/라우팅 프롬프트 버전도 설정화.
- [ ] **Agent preset/config 도입(툴/워크플로우/프롬프트 버전 외부화)**
  - 목표: “에이전트 플로우/툴체인”을 프리셋으로 선택 가능하게 하여 운영/실험 전환 비용을 낮춤.

### P2 — 운영/유지보수/평가

- [ ] **인덱스 유지보수 런북 정리**
  - alias 롤링 업데이트 절차(생성→재색인→alias switch→검증→롤백).
  - `_meta(pipeline)`를 활용한 “어떤 설정으로 만든 인덱스인지” 추적 표준화.
- [ ] **회귀 평가 세트(queries + expected evidence) 구축**
  - 목표: analyzer 변경, weights 변경, MQE/rerank 도입 시 품질 회귀를 자동 감지.
  - 산출물: 최소 30~100개 질의(설치/TS/일반) + 기대 top docs 또는 최소 포함 조건.
- [ ] **성능/비용 벤치마크**
  - MQE on/off, rerank on/off, RRF vs script_score 등 조합별 latency/throughput 측정.

---

## 2) 빠른 체크리스트(실행 전 확인)

- [ ] ES 버전/플러그인(nori) 가능 여부 확인
- [ ] 현재 alias가 가리키는 인덱스의 mapping/settings 스냅샷 확보
- [ ] 임베딩 모델 변경 시: dimension 변경 → 신규 인덱스(vNext) 생성 후 reindex가 필요한지 확인
- [ ] MQE/rerank 도입 시: p95 latency 목표/리소스 예산(GPU/CPU) 정의

---

## 3) 상세 실행 계획 (리트리벌 아키텍처 리뷰 반영)

> **기반 문서**: 리트리벌 아키텍처 진단 및 개선 (8가지 이슈)
> **작성일**: 2026-01-02

### 📊 현재 시스템 진단 요약

**검색 구성**:
- 백엔드: Elasticsearch 8.x
- 방식: Hybrid Search (Dense kNN + BM25)
- 임베딩: 768-dim (추정: BGE-base 또는 KoE5)
- 인덱스/alias: `rag_chunks_dev_current` (alias 여부 확인 필요; `_cat/aliases`로 검증)

**주요 필드**:
```
✅ embedding (dense_vector, cosine, 768 dims, int8_hnsw)
✅ search_text (text, standard analyzer) - 복합 필드 (content+title+tags)
✅ content (text, standard analyzer)
✅ chunk_summary (text) - 검색 가능 (스냅샷 기준)
✅ doc_description (text) - 검색 가능 (스냅샷 기준)
✅ chunk_keywords (text + keyword 멀티필드) - 검색 가능 (스냅샷 기준)
```

**문제점**:
1. 한국어 형태소 분석 미적용 (Nori analyzer 주석 처리)
2. 요약/키워드/설명 필드 활용(부스트/노이즈/필드명 정합성) 재점검 필요
3. Multi-Query Expansion 비활성화
4. Reranking 비활성화
5. 동의어 사전 미구축
6. 하이브리드 가중치 고정 (dynamic weighting 미적용)

---

### 🔴 Phase 1: 한국어 처리 및 인덱싱 최적화 (HIGH PRIORITY)

#### [TODO-1.1] Nori Analyzer 활성화 및 리인덱싱

**목표**: Recall +15% (한국어 형태소 분석)

**배경**:
```
현재: "장비를 가동했다" ≠ "장비 가동" (매칭 실패)
개선 후: "장비", "가동" 형태소 추출 → 매칭 성공
```

**Step 1: Nori Plugin 설치 확인**
```bash
# ES 컨테이너에서 플러그인 확인
docker exec -it <es-container> bin/elasticsearch-plugin list

# nori 미설치 시
docker exec -it <es-container> bin/elasticsearch-plugin install analysis-nori
```

- [ ] Docker Compose에 플러그인 자동 설치 추가
  ```yaml
  # docker-compose.yml
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms4g -Xmx4g"
    command: >
      sh -c "bin/elasticsearch-plugin install analysis-nori &&
             bin/elasticsearch"
  ```

**Step 2: 인덱스 매핑 수정**

- [ ] `backend/llm_infrastructure/elasticsearch/mappings.py:187` 수정
  ```python
  def get_index_settings(...):
      return {
          "analysis": {
              "analyzer": {
                  "nori_analyzer": {
                      "type": "custom",
                      "tokenizer": "nori_tokenizer",
                      "filter": [
                          "nori_readingform",  # 한자 → 한글 변환
                          "lowercase",
                          "nori_part_of_speech",  # 품사 필터
                      ],
                  }
              },
              "filter": {
                  "nori_part_of_speech": {
                      "type": "nori_part_of_speech",
                      "stoptags": ["E", "IC", "J", "MAG", "MM", "SP", "SSC", "SSO", "SC", "SE", "XPN", "XSA", "XSN", "XSV", "UNA", "NA", "VSV"]
                  }
              }
          }
      }
  ```

- [ ] 텍스트 필드에 analyzer 적용 (mappings.py:44, 50)
  ```python
  "content": {
      "type": "text",
      "analyzer": "nori_analyzer",
  },
  "search_text": {
      "type": "text",
      "analyzer": "nori_analyzer",
  }
  ```

**Step 3: 리인덱싱 절차**

- [ ] 리인덱싱 스크립트 작성 (`scripts/es_reindex_with_nori.py`)
  ```python
  # 1. 새 인덱스 생성 (v2)
  index_name = f"{prefix}_{env}_v2"
  es.indices.create(index=index_name, body={
      "settings": get_index_settings(...),  # with nori
      "mappings": get_rag_chunks_mapping(dims=768),
  })

  # 2. 데이터 복사
  helpers.reindex(
      es,
      source_index=f"{prefix}_{env}_v1",
      target_index=index_name,
      chunk_size=500,
  )

  # 3. Alias 전환
  es.indices.update_aliases(body={
      "actions": [
          {"remove": {"index": f"{prefix}_{env}_v1", "alias": f"{prefix}_{env}_current"}},
          {"add": {"index": index_name, "alias": f"{prefix}_{env}_current"}},
      ]
  })
  ```

- [ ] 롤백 계획 수립
  ```bash
  # Alias를 v1로 되돌리기
  python scripts/es_index_manager.py rollback --from v2 --to v1
  ```

**Step 4: A/B 테스트**

- [ ] 테스트 쿼리 세트 준비
  ```python
  test_cases = [
      # (쿼리, 예상 매칭 문서 키워드)
      ("장비를 가동했다", ["장비 가동", "장비 시동"]),
      ("센서를 교체했습니다", ["센서 교체", "센서 장착"]),
      ("챔버 청소 절차", ["챔버 클리닝", "챔버를 청소"]),
      ("펌프가 작동하지 않아요", ["펌프 작동 불량", "펌프 고장"]),
  ]
  ```

- [ ] Recall@10 비교
  ```python
  for query, keywords in test_cases:
      # v1 (standard) 검색
      results_v1 = search_index_v1(query, top_k=10)
      recall_v1 = calculate_recall(results_v1, keywords)

      # v2 (nori) 검색
      results_v2 = search_index_v2(query, top_k=10)
      recall_v2 = calculate_recall(results_v2, keywords)

      print(f"{query}: v1={recall_v1:.2f}, v2={recall_v2:.2f}")
  ```

**예상 효과**: Recall +15%, 한국어 동사/형용사 활용형 쿼리 대응

**소요 시간**: 2-3일

**파일**:
- `docker-compose.yml`
- `backend/llm_infrastructure/elasticsearch/mappings.py:187`
- `scripts/es_reindex_with_nori.py` (신규)

---

#### [TODO-1.2] chunk_summary 필드 검색 활성화

**목표**: LLM 생성 요약을 BM25 검색에 활용

**배경**:
- 스냅샷 기준 `chunk_summary`는 `text`로 검색 가능(“검색 불가” 상태는 아님)
- 다만 (1) 한국어 분석기 적용 여부, (2) 쿼리 부스트/노이즈, (3) `search_text` 포함 여부를 정리할 필요가 있음

**작업**:

1. **매핑 수정** (mappings.py:117)
   ```python
   "chunk_summary": {
       "type": "text",
       "index": True,
       "analyzer": "nori",  # (또는 multi-field로 nori 병행)
   },
   ```
   - [ ] 매핑 수정 (TODO-1.1과 함께 v2 인덱스에 적용)

2. **BM25 쿼리에 필드 추가** (es_search.py:121)
   ```python
   text_fields=[
       "search_text^1.0",
       "chunk_summary^0.7",  # 추가
       "chunk_keywords^0.8",
   ]
   ```
   - [ ] EsSearchEngine 기본 text_fields 수정

3. **Boost 튜닝**
   - [ ] 초기값: `chunk_summary^0.5` (보수적 시작)
   - [ ] A/B 테스트: 0.5 → 0.7 → 1.0
   - [ ] 노이즈 발생 시 가중치 하향 또는 비활성화

**주의사항**:
- 요약 품질 검증 필요 (LLM hallucination 가능성)
- 너무 높은 boost는 원본 content 압도할 수 있음

**소요 시간**: 1일 (TODO-1.1과 병행)

**파일**:
- `backend/llm_infrastructure/elasticsearch/mappings.py:117`
- `backend/llm_infrastructure/retrieval/engines/es_search.py:121`

---

#### [TODO-1.3] 임베딩 차원 불일치 해결

**목표**: 인덱스 매핑 ↔ 임베딩 모델 차원 100% 동기화

**현재 문제**:
- `.env`: `SEARCH_ES_EMBEDDING_DIMS=768`
- `mappings.py` 기본값: `dims=1024`
- 불일치 시 인제스션 실패 가능

**검증 절차**:

1. **실제 차원 확인**
   ```python
   from backend.services.embedding_service import EmbeddingService
   from backend.config.settings import rag_settings, search_settings

   print(f"설정 method: {rag_settings.embedding_method}")
   print(f"환경변수 dims: {search_settings.es_embedding_dims}")

   svc = EmbeddingService()
   actual_dim = svc.dimension()
   print(f"실제 출력 dims: {actual_dim}")
   ```
   - [ ] 차원 확인 및 문서화

2. **ES 매핑 확인**
   ```bash
   curl -X GET "http://localhost:8002/rag_chunks_dev_current/_mapping" | \
     jq '.[] | .mappings.properties.embedding.dims'
   ```
   - [ ] 결과 저장: `docs/es_current_mapping_snapshot.json`

3. **불일치 시 해결**
   - [ ] **Option A**: 매핑 수정 (v2 인덱스에서 768 적용)
   - [ ] **Option B**: 임베딩 모델 교체 (1024-dim 모델 사용)
     - KoE5 large 또는 multilingual-e5-large

4. **일관성 체크 강화**
   ```python
   # backend/services/es_ingest_service.py:243
   def _validate_embedding_dimension(self, embeddings: np.ndarray):
       actual_dim = embeddings.shape[1]
       expected_dim = search_settings.es_embedding_dims

       if actual_dim != expected_dim:
           raise ValueError(
               f"❌ Embedding dimension mismatch!\n"
               f"   Actual:   {actual_dim}\n"
               f"   Expected: {expected_dim}\n"
               f"   Fix: Check RAG_EMBEDDING_METHOD and SEARCH_ES_EMBEDDING_DIMS"
           )
   ```
   - [ ] 인제스션 시작 시 검증 추가
   - [ ] Health check에 차원 검증 추가

**소요 시간**: 0.5일

**파일**:
- `.env:52`
- `backend/llm_infrastructure/elasticsearch/mappings.py:61`
- `backend/services/es_ingest_service.py:243`

---

### 🟡 Phase 2: 검색 품질 향상 (MEDIUM PRIORITY)

#### [TODO-2.1] Cross-Encoder Reranking 활성화

**목표**: Precision@5 +20%

**작업**:

1. **환경변수 설정**
   ```bash
   # .env 추가
   RAG_RERANK_ENABLED=true
   RAG_RERANK_METHOD=cross_encoder
   RAG_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
   RAG_RERANK_TOP_K=5
   ```
   - [ ] `.env` 수정

2. **모델 로딩 테스트**
   ```python
   from backend.api.dependencies import get_reranker
   reranker = get_reranker()

   # 샘플 리랭킹
   query = "EFEM 센서 오류 해결"
   results = [...] # 검색 결과 (top 20)
   reranked = reranker.rerank(query, results, top_k=5)

   # 추론 속도 측정
   import time
   start = time.time()
   reranker.rerank(query, results, top_k=5)
   print(f"Rerank time: {(time.time() - start)*1000:.1f}ms")
   ```
   - [ ] GPU/CPU 환경별 속도 측정

3. **ES 경로에 reranking 통합**
   - [ ] `backend/services/es_search_service.py` 수정 필요 확인
   ```python
   def search(self, query: str, **kwargs):
       results = self.retriever.retrieve(...)

       # Reranking 추가
       if rag_settings.rerank_enabled:
           reranker = get_reranker()
           results = reranker.rerank(query, results, top_k=rag_settings.rerank_top_k)

       return results
   ```

4. **성능 평가**
   - [ ] Precision@5 측정
   - [ ] MRR (Mean Reciprocal Rank) 측정
   - [ ] NDCG@5 측정
   - [ ] P95 latency 측정 (목표: <500ms)

5. **장애 대응**
   ```python
   try:
       results = reranker.rerank(query, results, top_k=5)
   except Exception as e:
       logger.warning(f"Reranking failed: {e}")
       results = results[:5]  # Fallback to top-5
   ```
   - [ ] Graceful degradation 구현

**소요 시간**: 1-2일

**파일**:
- `.env`
- `backend/services/es_search_service.py`

---

#### [TODO-2.2] Multi-Query Expansion 조건부 활성화

**목표**: 모호한 질의의 recall 향상 (비용 최소화)

**전략**: 모든 질의에 MQE 적용 시 지연/비용 증가 → 선택적 트리거

**작업**:

1. **질의 분류기 구현**
   ```python
   # backend/services/query_analyzer.py (신규)
   import re

   def should_expand_query(query: str) -> bool:
       """MQE 트리거 여부 판단"""
       tokens = query.split()

       # 짧은 질의 (< 3 단어)
       if len(tokens) < 3:
           return True

       # 질문형
       if any(q in query for q in ['?', '왜', '어떻게', '무엇', '언제']):
           return True

       # 에러 코드만 (예: "EFEM-1234")
       if re.match(r'^[A-Z0-9\-]+$', query.strip()):
           return True

       return False
   ```
   - [ ] 분류기 구현 및 단위 테스트

2. **SearchService에 조건부 로직 추가**
   ```python
   # backend/services/search_service.py
   from backend.services.query_analyzer import should_expand_query

   def search(self, query: str, **kwargs):
       # 조건부 MQE
       if self.multi_query_enabled and should_expand_query(query):
           queries = self.query_expander.expand(query, n=2)
           logger.info(f"✓ MQE triggered: {len(queries)} queries")
       else:
           queries = [query]

       # 검색 및 병합
       all_results = []
       for q in queries:
           results = self.retriever.retrieve(q, **kwargs)
           all_results.append((q, results))

       # RRF로 병합
       final_results = self._merge_with_rrf(all_results)
       return final_results
   ```
   - [ ] 조건부 로직 구현

3. **모니터링**
   - [ ] MQE 트리거 비율 로깅
   - [ ] 확장 쿼리당 지연 시간 측정
   - [ ] 월별 LLM API 비용 추적

**Alternative**: UI에 "확장 검색" 토글 제공

**소요 시간**: 2일

**파일**:
- `backend/services/query_analyzer.py` (신규)
- `backend/services/search_service.py`

---

#### [TODO-2.3] 동적 하이브리드 가중치 조정

**목표**: 질의 타입별 최적 dense/sparse 비율 적용

**현재 문제**: 모든 질의에 `dense=0.7, sparse=0.3` 고정

**작업**:

1. **가중치 전략 구현**
   ```python
   # backend/services/query_analyzer.py
   def get_hybrid_weights(query: str) -> tuple[float, float]:
       """질의 특성에 따른 가중치 반환

       Returns:
           (dense_weight, sparse_weight)
       """
       tokens = query.split()

       # 에러 코드: BM25 우선
       if re.search(r'\b[A-Z]{2,}-?\d{3,}\b', query):
           return (0.3, 0.7)  # BM25 우선

       # 자연어 질문: Dense 우선
       elif len(tokens) > 5 and any(q in query for q in ['?', '왜', '어떻게']):
           return (0.8, 0.2)  # 의미 검색 우선

       # 기본값
       else:
           return (0.7, 0.3)
   ```
   - [ ] 전략 구현

2. **EsHybridRetriever에 통합**
   ```python
   # backend/llm_infrastructure/retrieval/adapters/es_hybrid.py:143
   from backend.services.query_analyzer import get_hybrid_weights

   def retrieve(self, query: str, **kwargs):
       # 동적 가중치
       dense_w, sparse_w = get_hybrid_weights(query)
       logger.debug(f"Dynamic weights: dense={dense_w}, sparse={sparse_w}")

       # ... (기존 로직에서 가중치만 교체)
       hits = self.es_engine.hybrid_search(
           ...,
           dense_weight=dense_w,
           sparse_weight=sparse_w,
       )
       return [hit.to_retrieval_result() for hit in hits]
   ```
   - [ ] 가중치 주입

3. **A/B 테스트**
   ```python
   test_cases = [
       ("EFEM-1234 알람 해결", "BM25 우선"),  # → 0.3, 0.7
       ("왜 온도가 안 올라가나요?", "Dense 우선"),  # → 0.8, 0.2
       ("센서 교체 방법", "기본값"),  # → 0.7, 0.3
   ]
   ```
   - [ ] NDCG@10 비교

**소요 시간**: 2일

**파일**:
- `backend/services/query_analyzer.py`
- `backend/llm_infrastructure/retrieval/adapters/es_hybrid.py`

---

### 🟢 Phase 3: 고급 기능 (LOW PRIORITY)

#### [TODO-3.1] 메타데이터 필터 UI/API 통합

**목표**: Precision 향상

**작업**:

1. **API 파라미터 추가**
   ```python
   # backend/api/routers/search.py
   @router.get("")
   async def search(
       q: str,
       doc_type: Optional[str] = Query(None, description="sop|maintenance|setup"),
       device_name: Optional[str] = Query(None, description="SUPRA|EFEM|..."),
       ...
   ):
       results = search_service.search(
           q,
           doc_type=doc_type,
           device_name=device_name,
       )
   ```
   - [ ] API 수정

2. **필터 옵션 집계 API**
   ```python
   @router.get("/filters/device_names")
   async def get_device_names():
       # ES aggregation
       agg_result = es.search(
           index=index,
           body={"size": 0, "aggs": {"devices": {"terms": {"field": "device_name.keyword"}}}}
       )
       return [b["key"] for b in agg_result["aggregations"]["devices"]["buckets"]]
   ```
   - [ ] 집계 API 추가

3. **자동 필터 추출**
   ```python
   DEVICE_NAMES = ['SUPRA', 'EFEM', 'PRECIA']

   def extract_device_filter(query: str) -> str | None:
       for device in DEVICE_NAMES:
           if device.upper() in query.upper():
               return device
       return None
   ```
   - [ ] 자동 추출 로직 (선택적)

**소요 시간**: 2-3일

---

#### [TODO-3.2] RRF vs Script Score 실험

**목표**: 가중치 튜닝 없이 하이브리드 결합 개선

**작업**:

1. **RRF 활성화**
   ```bash
   RAG_HYBRID_USE_RRF=true
   RAG_HYBRID_RRF_K=60
   ```

2. **성능 비교**
   ```python
   configs = [
       {"method": "script_score", "dense": 0.7, "sparse": 0.3},
       {"method": "script_score", "dense": 0.5, "sparse": 0.5},
       {"method": "rrf", "rrf_k": 60},
   ]

   for config in configs:
       # NDCG, Precision, Recall 측정
       ...
   ```
   - [ ] 평가 실행

3. **결과 기반 선택**
   - [ ] RRF가 우수하면 기본값 변경
   - [ ] 아니면 현재 유지

**소요 시간**: 1일

---

#### [TODO-3.3] 동의어 사전 구축 (도메인 특화)

**목표**: 반도체 장비 용어 정규화

**작업**:

1. **용어 수집**
   ```
   # config/synonyms/semiconductor.txt
   EFEM, efem, Equipment Front End Module
   PM, pm, Preventive Maintenance, 예방 정비
   RF, rf, Radio Frequency
   ```
   - [ ] 최소 50개 용어 수집

2. **Nori analyzer에 적용**
   ```python
   "filter": [
       "nori_readingform",
       "lowercase",
       {
           "type": "synonym",
           "synonyms_path": "config/synonyms/semiconductor.txt",
       }
   ]
   ```
   - [ ] 매핑 수정 및 리인덱싱

**소요 시간**: 2-3일 (수집 시간 포함)

---

### 📊 Timeline 및 우선순위

#### Week 1-2: Critical Path
```
Day 1-2:  [TODO-1.3] 임베딩 차원 검증
Day 3-6:  [TODO-1.1] Nori analyzer + [TODO-1.2] chunk_summary
Day 7-10: 리인덱싱 및 A/B 테스트
```

#### Week 3: Search Quality
```
Day 11-12: [TODO-2.1] Reranking
Day 13-15: [TODO-2.2] 조건부 MQE
```

#### Week 4: Advanced
```
Day 16-17: [TODO-2.3] 동적 가중치
Day 18-20: [TODO-3.1] 메타데이터 필터
Day 21:    [TODO-3.2] RRF 실험
```

---

### 📈 예상 성능 개선

| 지표 | 현재 | 목표 | 주요 개선 사항 |
|------|------|------|----------------|
| Recall@10 | 기준 | +15% | Nori analyzer |
| Precision@5 | 기준 | +20% | Cross-encoder reranking |
| NDCG@10 | 기준 | +10% | 전체 개선 |
| 응답 시간 (P95) | ? | <500ms | Reranking 포함 |
| 제로 결과 비율 | ? | -30% | MQE + 동의어 |

---

---

## 4) Nori 활성화 상세 구현 계획 (2026-01-02 추가)

> **목표**: 한국어 형태소 분석으로 Recall +15% 향상
> **작업일**: 2026-01-02
> **우선순위**: P1 (High Priority)

### 📋 현재 상태 분석

**발견된 사실**:
1. **Docker (docker-compose.yml:55)**:
   - Nori plugin 미설치 (주석만 존재: `# For Korean analysis, consider building custom image with nori plugin`)
   - ES 8.14.0 이미지 사용 중

2. **Mappings (backend/llm_infrastructure/elasticsearch/mappings.py)**:
   - Nori analyzer 설정 주석 처리 (line 191-199)
   - 텍스트 필드들이 모두 `standard` analyzer 사용:
     - `content` (line 44-48)
     - `search_text` (line 50-55)
   - `chunk_summary`: index=True이지만 analyzer 미지정 (기본 standard 적용)
   - `chunk_keywords.text`: standard analyzer

3. **인덱스 관리**:
   - ✅ EsIndexManager가 alias 전략 완벽 지원
   - ✅ validate_dims 기능 구현됨
   - ✅ 마이그레이션 스크립트 존재: `scripts/migrate_to_alias_strategy.py`

4. **하이브리드 검색**:
   - ✅ RRF 기본 활성화 (`use_rrf=True`, es_hybrid.py:65)
   - ✅ script_score 후보군 제한 이슈 해결됨

### 🎯 구현 계획 (4단계)

---

#### **Phase 1: Nori Plugin 설치 (Docker)**

**파일**: `docker-compose.yml`

**현재 (line 47-76)**:
```yaml
elasticsearch:
  container_name: rag-elasticsearch
  image: docker.elastic.co/elasticsearch/elasticsearch:8.14.0
  environment:
    - discovery.type=single-node
    - ES_JAVA_OPTS=-Xms2g -Xmx2g
    - xpack.security.enabled=false
    - xpack.security.enrollment.enabled=false
    # For Korean analysis, consider building custom image with nori plugin
```

**변경 후**:
```yaml
elasticsearch:
  container_name: rag-elasticsearch
  image: docker.elastic.co/elasticsearch/elasticsearch:8.14.0
  # Nori plugin 자동 설치
  entrypoint: >
    sh -c "
    if ! bin/elasticsearch-plugin list | grep -q analysis-nori; then
      echo 'Installing analysis-nori plugin...';
      bin/elasticsearch-plugin install --batch analysis-nori;
    fi &&
    /usr/local/bin/docker-entrypoint.sh
    "
  environment:
    - discovery.type=single-node
    - ES_JAVA_OPTS=-Xms2g -Xmx2g
    - xpack.security.enabled=false
    - xpack.security.enrollment.enabled=false
```

**검증**:
```bash
docker exec -it rag-elasticsearch bin/elasticsearch-plugin list
# 출력: analysis-nori
```

**체크리스트**:
- [ ] docker-compose.yml 수정
- [ ] ES 컨테이너 재시작: `docker compose down elasticsearch && docker compose up -d elasticsearch`
- [ ] Nori plugin 설치 확인
- [ ] ES health check 통과 확인

---

#### **Phase 2: Nori Analyzer 설정 (Mappings)**

**파일**: `backend/llm_infrastructure/elasticsearch/mappings.py`

**Step 2-1: get_index_settings() 수정 (line 173-200)**

**현재**:
```python
def get_index_settings(
    number_of_shards: int = 1,
    number_of_replicas: int = 0,
) -> dict[str, Any]:
    return {
        "number_of_shards": number_of_shards,
        "number_of_replicas": number_of_replicas,
        "refresh_interval": "1s",
        # Nori analyzer 주석 처리됨
    }
```

**변경 후**:
```python
def get_index_settings(
    number_of_shards: int = 1,
    number_of_replicas: int = 0,
    enable_nori: bool = True,
) -> dict[str, Any]:
    """Get index settings.

    Args:
        number_of_shards: Number of primary shards (default: 1 for dev)
        number_of_replicas: Number of replica shards (default: 0 for dev)
        enable_nori: Enable Korean (Nori) analyzer (default: True)

    Returns:
        Elasticsearch index settings
    """
    settings = {
        "number_of_shards": number_of_shards,
        "number_of_replicas": number_of_replicas,
        "refresh_interval": "1s",
    }

    if enable_nori:
        settings["analysis"] = {
            "analyzer": {
                "nori_analyzer": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "filter": [
                        "nori_readingform",  # 한자 → 한글 변환
                        "lowercase",
                        "nori_part_of_speech",  # 품사 필터 (조사/어미 제거)
                    ],
                }
            },
            "filter": {
                "nori_part_of_speech": {
                    "type": "nori_part_of_speech",
                    # 제거할 품사 태그 (조사, 어미, 접미사 등)
                    "stoptags": [
                        "E",    # 어미
                        "IC",   # 감탄사
                        "J",    # 조사
                        "MAG",  # 일반 부사
                        "MM",   # 관형사
                        "SP",   # 쉼표, 마침표
                        "SSC",  # 닫는 괄호
                        "SSO",  # 여는 괄호
                        "SC",   # 구분자
                        "SE",   # 줄임표
                        "XPN",  # 접두사
                        "XSA",  # 형용사 파생 접미사
                        "XSN",  # 명사 파생 접미사
                        "XSV",  # 동사 파생 접미사
                        "UNA",  # 알 수 없음
                        "NA",   # 분석 불능
                        "VSV",  # 동사
                    ],
                }
            },
        }

    return settings
```

**Step 2-2: 텍스트 필드에 nori analyzer 적용**

**변경할 필드들**:

```python
# Line 44-48: content 필드
"content": {
    "type": "text",
    "analyzer": "nori_analyzer",  # standard → nori_analyzer
},

# Line 50-55: search_text 필드
"search_text": {
    "type": "text",
    "analyzer": "nori_analyzer",  # standard → nori_analyzer
},

# Line 117-121: chunk_summary 필드
"chunk_summary": {
    "type": "text",
    "index": True,
    "analyzer": "nori_analyzer",  # 추가
},

# Line 122-132: chunk_keywords 필드
"chunk_keywords": {
    "type": "keyword",
    "doc_values": True,
    "fields": {
        "text": {
            "type": "text",
            "analyzer": "nori_analyzer",  # standard → nori_analyzer
        },
    },
},
```

**Step 2-3: get_rag_chunks_mapping() 파라미터 추가**

```python
def get_rag_chunks_mapping(dims: int = 768, use_nori: bool = True) -> dict[str, Any]:
    """Get RAG chunks index mapping with specified embedding dimensions.

    Args:
        dims: Embedding vector dimensions (default: 768)
        use_nori: Use Nori analyzer for Korean text (default: True)

    Returns:
        Elasticsearch mapping definition
    """
    analyzer = "nori_analyzer" if use_nori else "standard"

    return {
        "properties": {
            # ... (위에서 수정한 필드들에 analyzer 변수 사용)
            "content": {
                "type": "text",
                "analyzer": analyzer,
            },
            # ...
        }
    }
```

**체크리스트**:
- [ ] `get_index_settings()` 함수 수정 (enable_nori 파라미터 추가)
- [ ] Nori analyzer/filter 설정 추가
- [ ] `get_rag_chunks_mapping()` 함수 수정 (use_nori 파라미터 추가)
- [ ] content, search_text, chunk_summary, chunk_keywords.text 필드 analyzer 변경
- [ ] 단위 테스트 실행 (있다면)

---

#### **Phase 3: 리인덱싱 스크립트 작성**

**파일**: `scripts/reindex_with_nori.py` (신규)

**주요 기능**:
1. 현재 인덱스 상태 확인 (버전, 문서 수, dims)
2. Nori 포함 신규 인덱스 생성 (v2)
3. 데이터 reindex (백그라운드 또는 동기)
4. Alias 전환 (atomic operation)
5. 검증 (문서 수, 샘플 쿼리)
6. 롤백 기능

**스크립트 구조**:
```python
#!/usr/bin/env python3
"""Reindex existing data with Nori analyzer.

This script creates a new index version with Nori analyzer enabled,
reindexes all data, and switches the alias atomically.

Usage:
    # Dry run (preview changes)
    python scripts/reindex_with_nori.py --dry-run

    # Execute reindexing
    python scripts/reindex_with_nori.py

    # Rollback to previous version
    python scripts/reindex_with_nori.py --rollback

    # Custom version numbers
    python scripts/reindex_with_nori.py --from-version 1 --to-version 2
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from elasticsearch import Elasticsearch

from backend.config.settings import search_settings
from backend.llm_infrastructure.elasticsearch import EsIndexManager
from backend.llm_infrastructure.elasticsearch.mappings import (
    get_rag_chunks_mapping,
    get_index_settings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def reindex_with_nori(
    es_host: str,
    env: str,
    from_version: int,
    to_version: int,
    index_prefix: str = "rag_chunks",
    dry_run: bool = False,
) -> bool:
    """Reindex data with Nori analyzer.

    Args:
        es_host: Elasticsearch host URL
        env: Environment name (dev, staging, prod)
        from_version: Source index version
        to_version: Target index version (with Nori)
        index_prefix: Index prefix (default: rag_chunks)
        dry_run: If True, only preview changes

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("Nori Reindexing Strategy")
    logger.info("=" * 80)
    logger.info(f"ES Host: {es_host}")
    logger.info(f"Environment: {env}")
    logger.info(f"From version: v{from_version}")
    logger.info(f"To version: v{to_version} (with Nori)")
    logger.info(f"Dry run: {dry_run}")
    logger.info("=" * 80)

    # Initialize ES client and manager
    es_client = Elasticsearch([es_host], verify_certs=False)
    manager = EsIndexManager(
        es_client=es_client,
        env=env,
        index_prefix=index_prefix,
    )

    # Step 1: Check source index
    logger.info(f"\n[Step 1] Checking source index v{from_version}...")
    source_index = manager.get_index_name(from_version)

    if not manager.index_exists(from_version):
        logger.error(f"Source index {source_index} does not exist!")
        return False

    # Get source index stats
    stats = es_client.indices.stats(index=source_index)
    doc_count = stats["indices"][source_index]["total"]["docs"]["count"]
    size_bytes = stats["indices"][source_index]["total"]["store"]["size_in_bytes"]
    size_gb = size_bytes / (1024**3)

    logger.info(f"  ✓ Source index: {source_index}")
    logger.info(f"  ✓ Documents: {doc_count:,}")
    logger.info(f"  ✓ Size: {size_gb:.2f} GB")

    # Get current dims
    current_dims = manager.get_index_dims(version=from_version)
    logger.info(f"  ✓ Embedding dims: {current_dims}")

    # Step 2: Create target index with Nori
    logger.info(f"\n[Step 2] Creating target index v{to_version} with Nori...")
    target_index = manager.get_index_name(to_version)

    if manager.index_exists(to_version):
        logger.warning(f"  ⚠ Target index {target_index} already exists!")
        if not dry_run:
            response = input(f"Delete and recreate {target_index}? [y/N]: ")
            if response.lower() != "y":
                logger.info("Aborted by user")
                return False
            manager.delete_index(to_version)

    if dry_run:
        logger.info(f"  [DRY RUN] Would create index: {target_index}")
        logger.info(f"  [DRY RUN] With Nori analyzer enabled")
    else:
        try:
            # Create index with Nori
            body = {
                "settings": get_index_settings(
                    number_of_shards=1,
                    number_of_replicas=0,
                    enable_nori=True,  # ← Nori 활성화
                ),
                "mappings": get_rag_chunks_mapping(
                    dims=current_dims or 768,
                    use_nori=True,  # ← Nori 활성화
                ),
            }
            es_client.indices.create(index=target_index, body=body)
            logger.info(f"  ✓ Created index: {target_index}")
        except Exception as e:
            logger.error(f"  ✗ Failed to create index: {e}")
            return False

    # Step 3: Reindex data
    logger.info(f"\n[Step 3] Reindexing {doc_count:,} documents...")

    if dry_run:
        logger.info(f"  [DRY RUN] Would reindex from {source_index} to {target_index}")
        estimated_time = doc_count / 10000  # ~10k docs/sec
        logger.info(f"  [DRY RUN] Estimated time: ~{estimated_time:.1f} seconds")
    else:
        try:
            logger.info("  Starting reindex (this may take several minutes)...")
            result = es_client.reindex(
                body={
                    "source": {"index": source_index},
                    "dest": {"index": target_index},
                },
                wait_for_completion=True,
                refresh=True,
            )

            created = result.get("created", 0)
            logger.info(f"  ✓ Reindexed {created:,} documents")

            if created != doc_count:
                logger.warning(
                    f"  ⚠ Document count mismatch: expected {doc_count:,}, got {created:,}"
                )
        except Exception as e:
            logger.error(f"  ✗ Reindex failed: {e}")
            return False

    # Step 4: Test Nori analyzer
    logger.info("\n[Step 4] Testing Nori analyzer...")

    test_cases = [
        ("장비를 가동했다", ["장비", "가동"]),
        ("센서를 교체했습니다", ["센서", "교체"]),
        ("챔버 청소 절차", ["챔버", "청소", "절차"]),
    ]

    if dry_run:
        logger.info("  [DRY RUN] Would test analyzer with sample queries")
    else:
        try:
            for text, expected_tokens in test_cases:
                result = es_client.indices.analyze(
                    index=target_index,
                    body={"analyzer": "nori_analyzer", "text": text},
                )
                tokens = [t["token"] for t in result["tokens"]]
                logger.info(f"  '{text}' → {tokens}")

                # Check if expected tokens are present
                missing = set(expected_tokens) - set(tokens)
                if missing:
                    logger.warning(f"    ⚠ Missing tokens: {missing}")
                else:
                    logger.info(f"    ✓ All expected tokens found")
        except Exception as e:
            logger.warning(f"  ⚠ Analyzer test failed: {e}")

    # Step 5: Switch alias
    alias_name = manager.get_alias_name()
    logger.info(f"\n[Step 5] Switching alias {alias_name} → v{to_version}...")

    if dry_run:
        logger.info(f"  [DRY RUN] Would switch alias to {target_index}")
    else:
        try:
            manager.switch_alias(version=to_version)
            logger.info(f"  ✓ Alias switched: {alias_name} → {target_index}")
        except Exception as e:
            logger.error(f"  ✗ Failed to switch alias: {e}")
            return False

    # Step 6: Verification
    logger.info("\n[Step 6] Verification...")

    if dry_run:
        logger.info("  [DRY RUN] Verification skipped")
    else:
        try:
            # Verify alias
            alias_resp = es_client.indices.get_alias(name=alias_name)
            if target_index in alias_resp:
                logger.info(f"  ✓ Alias verified: {alias_name} → {target_index}")
            else:
                logger.error("  ✗ Alias verification failed!")
                return False

            # Verify document count
            new_stats = es_client.indices.stats(index=alias_name)
            new_doc_count = new_stats["indices"][target_index]["total"]["docs"]["count"]

            if new_doc_count == doc_count:
                logger.info(f"  ✓ Document count verified: {new_doc_count:,}")
            else:
                logger.warning(
                    f"  ⚠ Count mismatch: expected {doc_count:,}, got {new_doc_count:,}"
                )
        except Exception as e:
            logger.error(f"  ✗ Verification failed: {e}")
            return False

    # Success!
    logger.info("\n" + "=" * 80)
    if dry_run:
        logger.info("✓ DRY RUN COMPLETE")
        logger.info("  Run without --dry-run to execute reindexing")
    else:
        logger.info("✓ REINDEXING COMPLETE")
        logger.info(f"  Source: {source_index} (v{from_version})")
        logger.info(f"  Target: {target_index} (v{to_version}, Nori enabled)")
        logger.info(f"  Alias: {alias_name} → {target_index}")
        logger.info("\n  Next steps:")
        logger.info("  1. Test search with Korean queries")
        logger.info("  2. Compare Recall@10 vs previous version")
        logger.info("  3. Monitor for issues")
        logger.info(f"  4. Rollback if needed: python scripts/reindex_with_nori.py --rollback")
        logger.info(f"  5. Delete old index after validation: es_client.indices.delete('{source_index}')")
    logger.info("=" * 80)

    return True


def rollback(es_host: str, env: str, to_version: int, index_prefix: str = "rag_chunks") -> bool:
    """Rollback alias to previous version.

    Args:
        es_host: Elasticsearch host URL
        env: Environment name
        to_version: Version to rollback to
        index_prefix: Index prefix

    Returns:
        True if successful
    """
    logger.info("=" * 80)
    logger.info("ROLLBACK: Switching alias back to previous version")
    logger.info("=" * 80)

    es_client = Elasticsearch([es_host], verify_certs=False)
    manager = EsIndexManager(es_client=es_client, env=env, index_prefix=index_prefix)

    if not manager.index_exists(to_version):
        logger.error(f"Target version v{to_version} does not exist!")
        return False

    try:
        manager.switch_alias(version=to_version)
        alias_name = manager.get_alias_name()
        target_index = manager.get_index_name(to_version)
        logger.info(f"✓ Rolled back: {alias_name} → {target_index}")
        return True
    except Exception as e:
        logger.error(f"✗ Rollback failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Reindex with Nori analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--es-host",
        default=search_settings.es_host,
        help=f"Elasticsearch host (default: {search_settings.es_host})",
    )
    parser.add_argument(
        "--env",
        default=search_settings.es_env,
        help=f"Environment name (default: {search_settings.es_env})",
    )
    parser.add_argument(
        "--index-prefix",
        default=search_settings.es_index_prefix,
        help=f"Index prefix (default: {search_settings.es_index_prefix})",
    )
    parser.add_argument(
        "--from-version",
        type=int,
        default=1,
        help="Source index version (default: 1)",
    )
    parser.add_argument(
        "--to-version",
        type=int,
        default=2,
        help="Target index version (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback to previous version",
    )

    args = parser.parse_args()

    if args.rollback:
        success = rollback(
            es_host=args.es_host,
            env=args.env,
            to_version=args.from_version,
            index_prefix=args.index_prefix,
        )
    else:
        success = reindex_with_nori(
            es_host=args.es_host,
            env=args.env,
            from_version=args.from_version,
            to_version=args.to_version,
            index_prefix=args.index_prefix,
            dry_run=args.dry_run,
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
```

**체크리스트**:
- [ ] `scripts/reindex_with_nori.py` 작성
- [ ] Dry-run 테스트: `python scripts/reindex_with_nori.py --dry-run`
- [ ] 실행 권한 부여: `chmod +x scripts/reindex_with_nori.py`

---

#### **Phase 4: 검증 및 롤백 전략**

**검증 체크리스트**:

1. **기술적 검증**:
   - [ ] 문서 수 일치 확인
   - [ ] Embedding 차원 일치 확인
   - [ ] Nori analyzer 동작 확인 (analyze API)
   - [ ] Alias 전환 확인

2. **기능 검증 (샘플 쿼리)**:
   ```python
   test_queries = [
       # (쿼리, 예상 매칭 키워드)
       ("장비를 가동했다", ["장비 가동", "장비 시동"]),
       ("센서를 교체했습니다", ["센서 교체", "센서 장착"]),
       ("챔버 청소 절차", ["챔버 클리닝", "챔버를 청소"]),
       ("펌프가 작동하지 않아요", ["펌프 작동 불량", "펌프 고장"]),
       ("EFEM 설치 방법", ["EFEM 장착", "EFEM 설치"]),
   ]
   ```
   - [ ] 각 쿼리의 Recall@10 측정
   - [ ] v1 vs v2 비교 (예상: +15% Recall)

3. **성능 검증**:
   - [ ] 검색 응답 시간 (<500ms 목표)
   - [ ] 인덱스 크기 비교
   - [ ] 메모리 사용량 모니터링

**롤백 절차**:

```bash
# 1. 문제 발견 시 즉시 롤백
python scripts/reindex_with_nori.py --rollback

# 또는 직접 alias 전환
python -c "
from backend.llm_infrastructure.elasticsearch import EsIndexManager
manager = EsIndexManager(es_host='http://localhost:8002', env='dev')
manager.switch_alias(version=1)  # v1로 롤백
print('Rolled back to v1')
"

# 2. 검증
curl http://localhost:8002/rag_chunks_dev_current/_cat/aliases

# 3. v2 인덱스는 즉시 삭제하지 말고 보관 (재시도 가능)
```

**롤백 체크리스트**:
- [ ] 롤백 스크립트 테스트
- [ ] 롤백 후 기능 검증
- [ ] v2 인덱스 보관 기간 결정 (예: 1주일)

---

### 📊 예상 효과

| 지표 | Before (v1) | After (v2) | 개선율 |
|------|-------------|------------|--------|
| **Recall@10 (한국어)** | 기준 | 예상 +15% | +15% |
| **형태소 매칭** | ❌ 실패 | ✅ 성공 | - |
| **검색 속도** | 기준 | 유사 (±5%) | 0% |
| **인덱스 크기** | 기준 | 예상 +10% | +10% |

**예시**:
- **Before**: "장비를 가동했다" → 매칭 실패 (exact match만 가능)
- **After**: "장비를 가동했다" → "장비", "가동" 토큰으로 분리 → "장비 가동" 문서 매칭 ✓

---

### 🚀 실행 순서

```bash
# Phase 1: Docker 설정
vi docker-compose.yml  # entrypoint 추가
docker compose down elasticsearch
docker compose up -d elasticsearch
docker exec -it rag-elasticsearch bin/elasticsearch-plugin list

# Phase 2: 코드 수정
vi backend/llm_infrastructure/elasticsearch/mappings.py
# - get_index_settings() 수정
# - get_rag_chunks_mapping() 수정

# Phase 3: 스크립트 작성
vi scripts/reindex_with_nori.py
chmod +x scripts/reindex_with_nori.py

# Dry-run 테스트
python scripts/reindex_with_nori.py --dry-run

# 실제 실행
python scripts/reindex_with_nori.py

# Phase 4: 검증
python scripts/test_nori_search.py  # 별도 작성 필요
```

---

### ⚠️ 주의사항

1. **다운타임 최소화**:
   - Reindex는 백그라운드로 실행 가능 (wait_for_completion=False)
   - Alias 전환은 atomic operation (무중단)

2. **디스크 용량**:
   - v1과 v2가 동시 존재 (일시적으로 2배 용량 필요)
   - 현재 5.5GB → 검증 후 v1 삭제

3. **임베딩은 재생성 안 함**:
   - Reindex는 문서만 복사 (embedding 그대로 유지)
   - Nori는 텍스트 인덱싱(BM25)에만 영향

4. **설정 백업**:
   - 현재 v1 mapping/settings는 이미 백업됨 (`docs/es_mapping_snapshot_2026-01-02.json`)

---

### 📝 관련 문서

- **TODO 항목**: P1 — 한국어 analyzer(Nori) 활성화 계획 수립 (line 82-86)
- **기존 스냅샷**: `docs/es_mapping_snapshot_2026-01-02.json`
- **마이그레이션 참고**: `scripts/migrate_to_alias_strategy.py`
- **ES Nori 공식 문서**: https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-nori.html

---

**마지막 업데이트**: 2026-01-02
**다음 리뷰**: Phase 1 완료 후 (2주 후)
