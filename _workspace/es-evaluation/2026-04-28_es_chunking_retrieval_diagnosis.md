# PE Agent ES/Chunking/Retrieval 진단 보고서

작성일: 2026-04-28
범위: chunk_v3 인덱스 토폴로지·청킹 파이프라인·검색 파이프라인·과거 진단 종합
대상 코드베이스: `/home/hskim/work/llm-agent-v2`
ES 엔드포인트: `localhost:8002` (Elasticsearch 8.14.0, Lucene 9.10.0, single-shard, 0 replica)

---

## 0. Executive Summary (정정판)

> **2026-04-28 14:00 정정**: 초판은 코드 field default만 보고 "agent default 경로가 in-memory BM25Okapi + bge-base-en 768d로 돈다"고 단정했으나, **`.env`에서 `SEARCH_BACKEND=es`, `SEARCH_CHUNK_VERSION=v3`, `SEARCH_V3_EMBED_INDEX=chunk_v3_embed_bge_m3_v1`, `SEARCH_ES_EMBEDDING_DIMS=1024`, `RAG_EMBEDDING_METHOD=bge_m3`, `RAG_RAPTOR_ENABLED=true`, `RAG_RERANK_ENABLED=true`, `RAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3`이 모두 활성화되어 있어** Pydantic Settings가 default를 override한다. 따라서 chat·agent·search 라우터 모두 ES 경로(EsChunkV3SearchService)를 사용 중이며, BM25Okapi/numpy VectorStore는 어댑터가 등록만 되어 있을 뿐 실제 호출 경로가 없다(dead path). reranker default mismatch도 `.env`로 정합되어 실제 사용 모델은 `bge-reranker-v2-m3`이다.

1. **chat의 retrieval은 이미 ES 경로(BM25 nori + bge_m3 1024d kNN + RAPTOR 보조검색 + bge-reranker-v2-m3)로 돌고 있다.**
   - `api/main.py:182-228 _configure_search_service()` startup → `EsChunkV3SearchService.from_settings(content_index, embed_index)` → `set_search_service()` 슬롯 주입.
   - `api/dependencies.py:142-149 get_search_service()`가 이 ES 서비스를 반환, chat/agent/search/retrieval/devices 라우터 모두 `Depends(get_search_service)` 주입.
   - `services/es_chunk_v3_search_service.py:181, 403, 584-615` ES `multi_match`(search_text^1.0 등) + dense kNN + RAPTOR summary→leaf expansion까지 ES 안에서 일괄 처리.
   - 따라서 "스위치가 옛날 모드에 묶여 있다"는 결론은 **사실이 아니다**. 운영에서 도는 path는 ES 풀스택이다.

2. **인덱스 매핑·토폴로지는 정교하지만, 실제 적재된 corpus 품질이 미흡하다.**
   chunk_v3_content는 nori·풍부 메타·RAPTOR 계층 필드까지 갖춘 정교한 인덱스이지만, 실제 적재 corpus는 doc_type imbalance(MyService 329k vs TS 760), content_hash 중복(21,159 hash가 117,774 chunk 점유), MyService 80자 미만 191,163 chunk, setup chapter 전수 누락, TS section_empty 716/760, VLM 표/기호 noise 등 evidence 품질 문제가 크다. **"스토리지 구조는 양호하나 evidence 품질은 미흡"**으로 평가해야 하며, **이것이 정정 후 가장 큰 ROI 항목**이다.

3. **Chunking은 도메인 특화 멀티 전략이라는 면에서 잘 짜였으나, 토큰 정의 불일치·헤더 손실·페이지1(TOC) 편중·표/이미지 미처리가 누적된다.** 특히 SOP의 page1 BM25 편중은 `page-hit@1`이 67% → 100%로 회복 가능함이 ad-hoc 보고서에서 입증됐는데, `second_stage_doc_retrieve_enabled` default `False`(`settings.py:658`)가 `.env`에 override되어 있지 않아 정식 default로 승격이 미뤄져 있다.

4. **가장 ROI 높은 4건 (정정판)**:
   ① **corpus cleanup + content_hash dedup + 노이즈 chunk 제거** — 운영 path가 이미 ES이므로 retrieval 코드 변경보다 corpus 품질 정리가 가장 큰 효과.
   ② **page-aware retrieval 정식화** (`second_stage_doc_retrieve_enabled=true` 승격 + early-page penalty 평가 게이트화).
   ③ **bilingual search_queries 강제** (agent layer; .env로는 못 고침).
   ④ **chunking 약점 보강**(B: 헤더 prefix 본문 삽입, C: 토크나이저-aware sliding window). 이 넷이 정합되면 +15~25% recall, +20% precision, page-hit@1 90%+ 회복이 합리적으로 기대된다.

---

## 1. ES 인덱스 토폴로지·헬스

### 1.1 활성 인덱스 (live, 2026-04-28)

| 역할 | 인덱스 | docs | size | 비고 |
|---|---|---|---|---|
| 텍스트+메타 (BM25/필터) | `chunk_v3_content` | 395,069 (deleted 14,873) | 288 MB | nori, 풍부 메타 |
| 벡터 — BGE-M3 1024d | `chunk_v3_embed_bge_m3_v1` | 394,824 | 7.6 GB | HNSW m=16, ef_construction=100, cosine |
| 벡터 — Jina v5 1024d | `chunk_v3_embed_jina_v5_v1` | 390,472 | 3.8 GB | (양자화 추정) |
| 벡터 — Qwen3-Emb-4B 2560d | `chunk_v3_embed_qwen3_emb_4b_v1` | 390,385 | 18.5 GB | dim 폭 큼 |
| 한국어 슬랭 사전 | `slang_dict` | 75,952 | 5 MB | 별도, 분석기 미연동 |
| 평가/감사 인프라 | `chat_turns_dev_v1`, `feedback_dev_v1`, `eval_runs/jobs`, `batch_answer_*` | — | — | 운영 감사 OK |
| **legacy 잔존** | `rag_chunks_dev_v1`, `rag_chunks_dev_v2` | 340k / 345k | 5.4 / 5.9 GB | **약 11 GB 사용 중** — 정리 필요 |
| 합성/실험 | `rag_synth_*`, `rag_contam_paper_a_v1` | 80~120 | — | paper 실험용 |

> **운영 위생**: legacy `rag_chunks_dev_v*` 두 인덱스가 **11 GB**를 점유하고 있다. v3 마이그레이션이 완료됐다면 alias 정리·삭제 후보. `eval_jobs`/`eval_runs`/`eval_scores`/`eval_datasets` 4개는 yellow 상태 (replica=1인데 단일 노드라서) — 이건 운영상 무해.

### 1.2 매핑·analyzer 핵심 (`backend/llm_infrastructure/elasticsearch/mappings.py`)

- `chunk_v3_content`:
  - `content`, `search_text`: `text` + `analyzer: nori`
  - `chunk_keywords`: **`keyword` + multi-field `.text` (analyzer=standard)** — 한국어인데 standard라 nori 미적용
  - 풍부한 keyword 필드: `doc_id`, `chunk_id`, `device_name`, `equip_id`, `chapter`, `pipeline_version`, `content_hash`, `lang`, `doc_type`, `tenant_id`, `project_id`
  - RAPTOR 계층 필드: `partition_key`, `raptor_level`, `raptor_parent_id`, `raptor_children_ids`, `is_summary_node`, `cluster_id`, `group_edges` (nested)
  - `extra_meta`, `evidence_links`: `enabled: false` (저장만, 인덱싱 안 함)
  - `dynamic: false`, `total_fields.limit=2000`
- `chunk_v3_embed_*`:
  - `embedding`: `dense_vector`, `dims∈{1024, 2560}`, `cosine`, HNSW(m=16, ef_construction=100), `_meta.normalize="l2"`
  - **filterable 메타 부족**: `chunk_id`, `doc_id`, `content_hash`, `device_name`, `doc_type`, `lang`, `tenant_id`, `project_id`, `chapter`만 — `equip_id`, `section_chapter`, `section_number`, `is_summary_node`, `partition_key` 등 누락 → 메타 필터+벡터 동시 검색 시 content 인덱스에 별도 join 필요

### 1.3 nori 분석기 정의 (`mappings.py:246-251`)

```json
"nori": { "type": "custom",
          "tokenizer": "nori_tokenizer",
          "filter": ["nori_readingform", "lowercase"] }
```

- nori_tokenizer 기본 모드(mixed). decompound_mode·user_dict·discard_punctuation 등 옵션 미설정.
- **synonym/stopword/edge_ngram 필터 없음** → 동의어·prefix 검색·도메인 약어 처리 불가.
- `slang_dict`(75k)는 별도 인덱스로 떠 있을 뿐, analyzer 사슬에 연결되어 있지 않다.

---

## 2. Chunking 진단

### 2.1 현 전략 (`scripts/chunk_v3/chunkers.py`, `backend/llm_infrastructure/preprocessing/chunking/`)

| 문서 타입 | 알고리즘 | 토큰/크기 | overlap | 비고 |
|---|---|---|---|---|
| SOP | `chunk_vlm_parsed()` (구조 인식, 페이지 단위) | 700 word-token | 80 | 정규식 헤더 탐지 |
| TS  | 동일 | 650 | 60 | |
| Setup | 동일 | 750 | 80 | |
| MyService | `chunk_myservice()` (4섹션 통합) | 섹션별 250~700 | 30~80 | 합성 청크 2000+ 가능 |
| GCB | `chunk_gcb()` (요약/섹션 분리) | 512자 고정 | 50 | 단순 fixed |
| 일반 | `FixedSizeChunker` | 512자 | 50 | registry default |

총 **115k 입력 문서 → 396k 청크 (3.4배 확대)** — 평균 적정.

### 2.2 잘 짜인 점

- 문서 타입별 차등 전략(특히 MyService의 `[meta]/[status]/[action]/[cause]/[result]` 섹션별 차등 토큰 한도)은 일반적인 fixed-size보다 진보적.
- `device_name`, `chapter`, `module`, `topic`, `work_type`, `alarm_code` 메타 보존은 **국내 RAG에선 상위 수준** — 메타 가이드 RAG로 확장 가능한 토대(`docs/2026-01-08_meta_guided_hierarchical_rag.md`).
- RAPTOR 계층 필드(`raptor_level`, `cluster_id`, `is_summary_node`)가 mapping에 이미 존재 → hierarchical retrieval로 확장 가능한 hook.
- VLM(Qwen3-VL)으로 페이지 단위 텍스트화 후 청킹 → OCR 단계 결합을 분리한 정상 설계.

### 2.3 약점·관찰된 문제 패턴 (코드 근거)

1. **word-token 정의가 embedding 토크나이저와 불일치** — `_word_window_split()` (`scripts/chunk_v3/chunkers.py:149-165`)는 공백 분리. 그런데 BGE-M3·Jina·Qwen3 임베더는 SentencePiece/BPE 서브워드. **선언된 700이 실제 임베딩 토큰으론 1.4~2.2배가 된다.** MyService의 `search_text = title+cause+status+action+result` 합성(라인 489)은 실제 임베딩에서 **512~1024 컨텍스트 한도 초과** 위험. 한도를 넘으면 truncation에 따라 마지막 섹션(action/result)이 잘림.

2. **헤더 텍스트가 본문에서 분리되어 손실** (`chunk_vlm_parsed`, 라인 314-317). `section_title`은 메타에만 저장되고 `content`에는 다시 포함되지 않는다. → "STEP 1" 같은 키워드로 검색해도 본문에 없으면 매칭 실패.

3. **VLM 페이지가 짧을 때 overlap이 무효화**. 페이지 자체가 청크 최소 단위라 page=1짜리 짧은 표지는 그대로 청크가 됨. SOP page1(TOC) 편중 현상의 한 원인.

4. **표·이미지 미처리**. VLM은 `page.get("text")`만 사용 → 표 셀이 `|`로 펼쳐진 평문으로 남거나 이미지 캡션이 누락. 실제 sample 문서에서 OCR 잡음 관찰 (`chunk_v3_content` 샘플 — 테이블 셀 파이프 반복).

5. **정규화 부작용**. `normalize_engine/domain.py`의 PM 주소 마스킹(`PM-1_PM-2` → `PM`)은 위치 모호성을 만든다. SOP 챕터 매칭 시 "PM-1 교체 절차" vs "PM-2 교체 절차" 구분 실패 가능.

6. **MyService completeness 필터의 비대칭**. `empty`만 스킵, `incomplete`도 그대로 인덱싱(scripts/chunk_v3/chunkers.py:418) → 품질 편차 누적.

7. **alarm_code 정규식 단순화** (`chunkers.py:311`). `alarm (123456)` 형식만 잡고, `ERROR_CODE: ABC-123` 같은 변형은 놓침. PE 도메인에서 alarm code는 핵심 검색어인데 메타 누락이 잦다.

8. **chunk_keywords의 `.text` 서브필드가 standard analyzer**. 한국어 키워드가 들어있지만 nori를 안 타서 부분 매칭이 약하다.

### 2.4 실제 적재 corpus 품질 지표 (`data/chunks_v3/all_chunks.jsonl` 기준)

총 chunk 수 **398,497**. doc_type 분포는 다음과 같다:

| doc_type | chunks | 비고 |
|---|---|---|
| myservice | 329,206 | 전체의 **82.6%** — global top-K 후보 공간 지배 |
| gcb | 49,021 | 12.3% |
| sop | 15,840 | 4.0% |
| setup | 3,583 | 0.9% — chapter 메타 **전수 누락** |
| ts | 760 | 0.2% — **716/760이 section_empty** |
| pems | 87 | 0.02% |

**중복(content_hash) 문제**: **21,159개 hash가 117,774개 chunk를 점유**(전체의 약 30%). 동일 내용 chunk가 dense/sparse 양쪽에서 같은 점수로 top-K 후보를 잠식할 가능성이 매우 크다. RRF dedup는 chunk_id 키이므로 content_hash 중복은 거르지 못한다 (`backend/llm_infrastructure/retrieval/rrf.py:28-37`).

**길이 분포 이상**:
- MyService 80자 미만 chunk **191,163개** (MyService 전체의 약 58%) — 대부분 alarm 한 줄, 짧은 cause 라인. 자체로는 검색 신호 약하면서 후보를 가득 채움.
- GCB·SOP·setup에 2,000자 초과 chunk 존재 — 임베딩 컨텍스트 한도 초과 위험.

**노이즈 패턴**: SOP/setup/TS에 `| | |`, `& \\ &` 같은 VLM table artifact가 content에 직접 잔류. BM25 토큰화 시 신호도 잡음도 아닌 채로 점수에 기여.

**메타 누락**: setup의 `chapter` 전수 누락, TS의 `section_*` 거의 비어 있음. 메타 가이드 RAG·필터·page-aware 로직이 전부 무력화되는 영역이 발생.

→ ES BM25/임베더 튜닝 이전에 **corpus cleanup·deduplication·메타 보강**이 선결되어야 한다. 이것이 P0의 가장 큰 누락이었다.

### 2.5 더 좋은 chunking 방법 (도메인 적합 우선순위)

| # | 방법 | 채택 난이도 | 기대 효과 | 우리 코드 적용 포인트 |
|---|---|---|---|---|
| **A** | **Contextual Retrieval (Anthropic 2024)** — 각 청크 앞에 LLM이 50~100토큰 "이 청크는 X 섹션의 Y 절차에 속한다" 컨텍스트를 prepend 후 임베딩·인덱싱 | 중 | recall 상승(논문 기준 hybrid+rerank와 결합 시 -67% retrieval failure) | `chunkers.py`의 `_chunk_page_by_heading` 직후 Ollama 호출로 컨텍스트 prefix 생성 후 `content`/`search_text`에 포함 |
| **B** | **Section-title prefix injection (B 즉시 가능판)** — `section_title`을 `content` 첫 줄에 다시 삽입 | 하 | 헤더 키워드 직접 매칭 회복, BM25/dense 모두 즉시 효과 | `chunkers.py:314-317` 한 줄 변경 (`segment_text = f"[{title}]\n{body}"`) |
| **C** | **Token-aware sliding window** — 임베딩 토크나이저(BGE-M3 SentencePiece)로 실제 토큰 카운트 후 분할. fixed word-token 폐기 | 중 | 컨텍스트 초과 truncation 제거, 청크 길이 분포 정상화 | `_word_window_split` 대신 `transformers.AutoTokenizer.from_pretrained("BAAI/bge-m3")` 기반 분할 |
| **D** | **Multi-granular indexing** — chunk(현행) + section + document 3계층 동시 인덱싱, 검색 시 RRF로 병합 | 중상 | 짧은 키워드 질의는 doc-level, 긴 자연어는 chunk-level이 살아남음 | content 인덱스에 `granularity` 필드 추가, partition_key 활용 |
| **E** | **Late chunking (BGE-M3·Jina v3 지원)** — 전체 문서(<8k 토큰)를 한 번 임베딩 후 청크 위치별 풀링 → 청크가 문서 컨텍스트를 보존 | 중상 | 청크 간 단절(문맥 절단)로 인한 dense miss 완화 | embedding 어댑터 측에서 long-context 모드 추가 |
| **F** | **Proposition chunking (DenseX)** — 문장을 자체 완결 명제로 LLM이 변환 후 임베딩 | 상 | precision↑ (특히 SOP 단계별 절차) | Ollama로 batch 변환 비용 큼 — Phase 2 |
| **G** | **TOC/표지 분리 인덱싱** — page≤2의 TOC/표지는 별도 인덱스 또는 `is_toc=true` 플래그 → 검색 시 down-weight (이미 ad-hoc 적용된 0.3x penalty의 정식화) | 하 | page-hit@1 회복, page1 편중 제거 | 인제스트 단계에서 TOC 페이지 자동 검출 + 메타 추가 |
| **H** | **표 추출 분리** — VLM이 표를 별도 markdown table로 추출, row 단위로 별도 청크화하고 metadata에 `is_table=true` | 중 | 표 안의 수치/조건 검색 정확도↑ | VLM 파싱 단계 출력 스키마 확장 |

권장: **B(즉시) → G → A → C** 순으로 단계적 채택. B·G는 한 PR로 가능하다.

---

## 3. Retrieval 파이프라인 진단

### 3.1 현 운영 경로 (정정판, .env override 반영)

```
Query
  └── SEARCH_BACKEND (.env: "es") + SEARCH_CHUNK_VERSION (.env: "v3")
        → EsChunkV3SearchService.from_settings(
             content_index = chunk_v3_content,
             embed_index   = chunk_v3_embed_bge_m3_v1)
   ↓
(LLM Multi-Query 확장 optional)
   ↓
ES 동시 호출
  ├── Dense kNN: chunk_v3_embed_bge_m3_v1 (1024d, cosine HNSW m=16)
  │     embedder = bge_m3 (RAG_EMBEDDING_METHOD)
  ├── Sparse:    chunk_v3_content multi_match (search_text^1.0, content, ...; analyzer=nori)
  └── RAPTOR 보조: is_summary_node=true 노드를 BM25로 찾고 leaf children 확장
                  (RAG_RAPTOR_ENABLED=true)
   ↓
Hybrid 결합 (weighted dense+sparse 또는 RRF; rrf_k/dense_weight/sparse_weight from rag_settings)
   ↓
Cross-encoder rerank (RAG_RERANK_ENABLED=true; model = BAAI/bge-reranker-v2-m3)
   ↓
Scope filter (device_name/equip_id/doc_type) → SectionExpander(top_2) → top_k=10
```

운영에서 실제로 사용되는 코드:
- `api/main.py:182-228` startup이 `set_search_service(EsChunkV3SearchService.from_settings(...))` 주입
- `api/dependencies.py:142-149` `get_search_service()`가 그 ES 서비스를 반환
- `services/es_chunk_v3_search_service.py:181, 403, 584-615` ES BM25 + dense kNN + RAPTOR
- chat/agent/search/retrieval/devices 라우터 모두 `Depends(get_search_service)`로 동일 ES 서비스 주입

dead/unused path (확인 필요한 fallback):
- `backend/llm_infrastructure/retrieval/engines/bm25.py` `BM25Okapi` — backend 런타임에서 import되지 않음(테스트 픽스처/legacy 가능성)
- `backend/llm_infrastructure/embedding/adapters/sentence.py:18` `bge_base = "BAAI/bge-base-en-v1.5"` — `RAG_EMBEDDING_METHOD=bge_m3` 환경에서 사용되지 않음

여전히 default(.env 미override) 상태로 남은 항목:
- `backend/config/settings.py:658` `second_stage_doc_retrieve_enabled = False` — page-aware 2단계 검색이 default off로 유지
- `backend/config/settings.py:659` `early_page_penalty_enabled = True` — 이미 default on
- preset 라우팅(`retrieval_preset = "hybrid_rrf_v1"`)과 ES 서비스 내부 hybrid 로직의 정합 여부는 별도 점검 필요

reranker default mismatch는 **`.env`의 `RAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3`로 정합**되어 운영에서는 문제 없음. 단 `.env` override가 사라진 상태에서는 `settings.py:161` "ms-marco-MiniLM-L-6-v2" vs `cross_encoder.py:33` `bge-reranker-v2-m3` mismatch가 다시 활성 — settings.py 측을 정합으로 수정해 fallback 안전성 확보 권장.

### 3.2 정정된 진단 — Storage/Runtime은 이미 정합

> **chat/agent/search 모두 ES + nori BM25 + bge_m3(1024d) kNN + bge-reranker-v2-m3 + RAPTOR 보조검색으로 작동 중.**

- `.env`의 7개 키(SEARCH_BACKEND·SEARCH_CHUNK_VERSION·SEARCH_V3_EMBED_INDEX·SEARCH_ES_EMBEDDING_DIMS·RAG_EMBEDDING_METHOD·RAG_RERANK_ENABLED·RAG_RAPTOR_ENABLED)가 Pydantic Settings를 통해 코드 default를 override하여 ES 풀스택을 활성화한다.
- `BM25Okapi`/numpy VectorStore/bge-base-en-v1.5 768d는 backend 런타임에서 import되지 않는 dead path다. (Adapter는 등록되어 있으나 startup에서 `set_search_service(EsChunkV3SearchService(...))`로 슬롯이 채워지므로 LRU 캐시된 `get_search_service()`가 무조건 ES 서비스를 반환한다.)
- 따라서 어제의 결론 "이걸 안 쓰고 in-memory로 돈다"는 사실이 아니다. **운영 path 자체는 이미 만들어진 인덱스를 잘 쓰고 있다.**
- 한편 settings.py field default(`backend="local"`, `embedding_method="bge_base"`, `rerank_model="ms-marco-MiniLM-L-6-v2"`)는 **`.env` 누락 시의 fallback**이며, dev/prod 환경에서 `.env`가 비거나 잘못 마운트되면 silent regression 위험이 있다 — 정합 안전성을 위해 settings.py field default도 ES/bge_m3/bge-reranker-v2-m3로 통일하는 것이 권장된다.
- search/retrieval-test 라우터(`api/routers/search.py:105-187`, `api/routers/retrieval.py:304-456`)도 동일 의존성을 받으므로 UI 테스트와 agent 답변은 동일 backend(ES)를 사용한다 — 결과 차이가 발생한다면 preset/parameter override 또는 query 변형(MQ on/off, bilingual 등)에서 비롯된다.

#### 3.2.1 그렇다면 진짜 병목은 무엇인가

운영 path가 이미 ES 풀스택이라면, 검색 품질이 mybot 대비 떨어지는 원인은 인프라가 아니라 (a) **corpus 품질**과 (b) **검색·답변 워크플로우**다:

1. **corpus**: §2.4의 중복(117k)·MyService imbalance(82.6%)·노이즈·메타 누락. ES BM25는 빠르고 정확하지만 잡음을 거르지 못한다.
2. **page-aware 흐름 미정착**: `second_stage_doc_retrieve_enabled` default false. early-page penalty는 켜져있지만 평가 게이트가 없어 회귀 시 즉시 잡히지 않는다.
3. **bilingual search_queries**: agent layer 책임. 영문 강제 시 한국어 corpus와 미스매치(과거 보고서 §11에서 DOC_MISS 23/79=29% 사례).
4. **RRF/dense·sparse 가중치 튜닝** 및 RAPTOR 활용 효과 측정 미실행.
5. **chunking 자체 약점**(헤더 손실·word-token 불일치·page1 편중·표/이미지 미처리) — §2.3 그대로 유효. ES가 좋아도 청크 본문에 헤더가 없으면 못 찾는다.

### 3.3 그 외 약점

| 항목 | 증거 | 영향 |
|---|---|---|
| 한국어 BM25 형태소 미적용 | `BM25Okapi` + `\b\w+\b` | 조사/어미로 토큰 분할 오류, recall 큰 손실 |
| dense 모델 EN-centric | `bge-base-en-v1.5` | 한국어 PE 용어 의미 표현 약화 |
| RRF 가중치 고정 | `[0.7, 0.3]` | dense 편향, sparse 신호 약함 |
| similarity threshold 미사용 | default 0.0 | 저품질 결과 컷오프 없음 |
| reranker default off | preset full_pipeline에서만 활성 | precision@5 손실 |
| query expansion default off | 문서 §8.3 | 짧은/모호한 질의 recall 손실 |
| 영문 search_queries 강제 | agent 코드에서 영문 번역만 사용 | DOC_MISS 23/79 (29%) 사례 보고됨 (`docs/2026-03-01_page_accuracy_improvement_report.md`) |
| MQ 비결정성 | Stability@k 0.039~0.411 | 동일 질의에서 결과 흔들림 (`docs/2026-02-28_*`) |
| page≤2 편중 | TOC 가중 BM25 hit | page-hit@1 67% (개선 후 100%) |
| embed 인덱스 메타 부족 | `equip_id`/`section_chapter`/`partition_key` 없음 | 메타 필터+벡터 동시 검색 시 추가 join |
| section_expander top_2만 | `postprocessors/section_expander.py` | 관련 섹션 누락 |

### 3.4 retrieval 개선안 (우선순위)

#### P0 — Corpus 품질 + 워크플로우 (정정판: 이미 ES 풀스택 운영 중이므로 인프라 항목 제거)

> 정정 전 P0에 들어있던 "ES BM25 전환 / BGE-M3 default 전환 / ES kNN 전환"은 `.env`로 **이미 운영 중**임이 확인되어 P0에서 제외한다. 정합 검증은 §5 Quick Wins로 이동.

1. **Corpus cleanup + content_hash deduplication (선결, 가장 큰 ROI)**
   - `data/chunks_v3/all_chunks.jsonl` 기준 21,159 hash가 117,774 chunk를 점유 → dedup 후 재인덱싱.
   - 반복 기호/표 잔재(`| | |`, `& \\ &` 등) 토큰 비율이 임계 이상인 chunk 제거 또는 정규화.
   - **너무 짧은 chunk(<80 char)는 alarm code, part number, SVID, recipe ID 등 high-signal token이 있을 때만 보존**, 그 외 drop.
   - dedup 시 원본 provenance(원래 chunk_id 리스트)를 metadata에 보관 — 인용 추적 보존.
   - 대상: MyService 80자 미만 191k건, content_hash 중복 117k건. setup chapter 누락분은 챕터 추출기 재실행.
   - 효과: ES 후보 공간이 ~70%로 축소, dense top-K가 진짜 evidence로 채워짐. ES가 이미 잘 작동 중이므로 noise만 걷어내면 정밀도가 즉시 회복된다.

2. **RRF/hybrid 이전·이후 noise/dedup filtering**
   - dense·sparse 후보를 hybrid merge **이전**에 각각 chunk 길이/노이즈 비율로 필터(EsChunkV3SearchService 내부 `_sparse_search_results`/dense 호출 후, hybrid 결합 직전).
   - hybrid merge **이후** final top-K 직전 `content_hash`로 collapse(현 dedup은 chunk_id만, `rrf.py:28-37`).
   - MyService는 chunk 단위가 아니라 `doc_id` 또는 issue record 단위 collapse — 동일 이슈의 status/cause/action/result chunk가 top-K를 점유하지 않게.

3. **Bilingual search_queries 기본 활성화 (agent layer)**
   - .env로는 못 고치는 영역. agent의 search_queries 생성 시 원본 한국어 + 번역 모두 포함.
   - `docs/2026-03-01_page_accuracy_improvement_report.md` §5.5 권고: DOC_MISS 23/79(29%) 사례의 직접 원인.
   - 영향 범위는 `react_agent.py`/`langgraph_agent.py`/관련 prompt YAML.

4. **Page-aware retrieval 정식화**
   - `second_stage_doc_retrieve_enabled` 현 default `False` (`settings.py:658`) → **`True`로 승격**.
   - `early_page_penalty_enabled`는 이미 default `True`(`settings.py:659`). paired A/B로 효과를 평가 게이트화하고 보고서 §11 결과(page-hit@1 67→100%)를 회귀 테스트로 고정.
   - `early_page_penalty_max_page=2`, `early_page_penalty_factor=0.3` 그대로.

5. **헤더 prefix 본문 삽입 (B) + 재인덱싱**
   - `scripts/chunk_v3/chunkers.py:314-317` 한 줄 변경: `segment_text = f"[{title}]\n{body}"`.
   - 재청킹 후 `chunk_v3_content`/`chunk_v3_embed_bge_m3_v1` 전체 또는 partition별 재인덱싱.
   - "STEP 1" 같은 헤더 키워드 직접 매칭 회복 — BM25/dense 모두 즉시 효과.

#### P1 — 정밀도 개선

6. **Cross-encoder reranker default on — 모델 default mismatch 정합 필수**.
   - 현재 `settings.py:161` default = `cross-encoder/ms-marco-MiniLM-L-6-v2`(영문 중심).
   - `cross_encoder.py:33` `DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"`(다국어).
   - 어댑터 인스턴스 생성 경로에 따라 실제로 어떤 모델이 로드되는지 분기되어 **재현성 위험**. settings 쪽을 `BAAI/bge-reranker-v2-m3`로 통일 권장.
   - 적용 후 top-100 → top-20 → top-10. 대안: `Alibaba-NLP/gte-multilingual-reranker-base`(다국어 안정). PE 도메인 한국어 fine-tune 별도 검토.

7. **RRF 가중치/k 튜닝**: dense=0.5, sparse=0.5, k=60 → 40 비교 A/B. preset YAML로 노출.

8. **Similarity threshold + 동적 cutoff**. dense top1과의 격차가 임계 이상 떨어지는 결과 drop.

9. **Section-title prefix injection (B)** 적용 후 재인덱싱. 매핑 변경 없음.

10. **doc_type 자동 라우팅**. 질의 패턴이 "교체/조정/작업"이면 `doc_type=SOP` 필터 자동, "원인/현상/이력"이면 `doc_type=MyService|GCB`로.

#### P2 — 인프라/운영

11. **synonym filter 활성화**. `slang_dict`(75k)를 ES synonym graph token filter로 export. PE 약어(예: `CTC ↔ 회로설정 보드`, `EPD ↔ End Point Detection`) 도메인 사전 별도 큐레이션.

12. **edge_ngram 필드 추가** — `device_name`, `equip_id`, `chapter`에 prefix 검색용 multi-field. "PM-1.."로 시작하는 자동완성/필터.

13. **embed 인덱스에 filterable 메타 보강**. `equip_id`, `section_chapter`, `partition_key`, `is_summary_node` 추가. 메타 필터+벡터 단일 쿼리화.

14. **legacy `rag_chunks_dev_v1/v2` 정리**. 11 GB 회수.

15. **Qwen3-Emb-4B int8 양자화 검토**. 18.5 GB → ~5 GB. 정확도 손실 1~3% 수준이면 채택 가능.

16. **MQ 결정성 정책**: `mq_mode ∈ {off, fallback, on}` × `deterministic=true` default. Stability@k 회복.

17. **RAPTOR 계층 활성화** (`is_summary_node` 필드는 이미 매핑에 존재). doc 요약 노드를 별도 부스트, 짧은 키워드 질의에 강함. (이미 `backend/llm_infrastructure/raptor/`가 존재 — 활성화 경로만 점검.)

#### P3 — 품질/신뢰성

18. **Citation/page accuracy guardrail**. 답변에 인용된 chunk_id가 실제 retrieval 결과 top-K에 포함되는지 검증. `backend/llm_infrastructure/llm/citation_verifier.py`(이미 존재)와 연결.

19. **Negative cache**. 자주 false-positive로 평가된 chunk를 down-weight (`feedback_dev_v1`에서 학습).

20. **Golden set CI 게이트**. `data/golden_set/`/`data/eval_mybot_march_2026-03.jsonl` 기반 회귀 가드 — agent-improvement-harness가 이미 도구를 갖췄으므로 retrieval 측에 동일 가드 연결.

---

## 4. 측정 방법 (개선 효과 정량화)

### 4.1 평가셋

평가셋 | 사이즈 | 용도
---|---|---
`data/eval_mybot_march_2026-03.jsonl` | 228 | mybot(GPT-5.4) 대비 paired A/B (최우선)
`data/eval_gold_master_base_v1.json` | — | 검색 회귀 가드
`data/eval_dev_bge_reranker_v2_m3.json` | — | reranker on/off 비교
`data/eval_sop_retrieval.py` 출력 | 79 | page-hit@1/@10 (SOP 도메인)
`data/golden_set/retrieval_golden_set_v2.md` | 20 | 시나리오 정성
**`_workspace/supra_xp_vplus_questions_analysis.md` (신규 승격)** | 4+ | mixed retrieval golden set (실제 데모형 질의)

### 4.2 SUPRA XP/Vplus mixed retrieval golden set 승격

추가로 `_workspace/supra_xp_vplus_questions_analysis.md`의 반복 질문을 mixed retrieval golden set으로 승격한다. 이 질문들은 SOP·MyService·GCB·TS가 함께 경쟁하는 실제 데모형 질의이므로 SOP-only eval보다 agent retrieval 품질을 더 잘 드러낸다.

우선 포함할 질의(반복 빈도 기준):

1. SUPRA XP의 FFU 교체 방법은?
2. SUPRA N FCIP R3에서 R5로 교체 시 유의사항은?
3. ZEDIUS XP 또는 SUPRA XP Particle Issue 양상은?
4. SUPRA Vplus에서 Ashing rate low issue 발생 시 조치 방법은?

각 질의에 대해 다음 메트릭을 함께 측정한다:

- `doc_type_hit@K` — 정답 doc_type(SOP/MyService/GCB/TS)이 top-K에 포함되는 비율
- `wrong_doc_type@K` — 정답이 아닌 doc_type이 top-K를 점유하는 비율(MyService imbalance 영향 측정)
- `unique_content_hash@K` — top-K의 고유 content_hash 수(중복 점유 측정)
- `noise_ratio@K` — VLM table artifact / 80자 미만 chunk가 top-K에서 차지하는 비율
- `citation_precision` — 답변 인용 chunk_id가 실제 retrieval top-K 안에 있는 비율(`backend/llm_infrastructure/llm/citation_verifier.py` 연동)

### 4.3 공통 메트릭·통계

핵심 메트릭: Recall@10, Recall@20, MRR@10, nDCG@10, **page-hit@1**, **doc-hit@10**, Stability@k(반복 5회 Jaccard 중위), latency p50/p95.
변경 적용 시 반드시 paired McNemar 또는 bootstrap CI로 통계 유의성 보고.

---

## 5. 즉시 실행 가능한 7건 (Quick Wins, ≤1주, 정정판)

1. **운영 path 정합 검증 + settings.py field default 통일 (silent regression 방지)**
   - 현재 `.env`로 ES 풀스택이 활성화되어 있으나, settings.py field default(`backend="local"`, `embedding_method="bge_base"`, `rerank_model="ms-marco-MiniLM-L-6-v2"`)는 fallback 가치가 거의 없고 .env 누락 시 silent regression 유발.
   - `settings.py:44/161/573-575` field default를 `"es"`/`"bge_m3"`/`"BAAI/bge-reranker-v2-m3"`로 통일. 1~3줄 diff.
   - 동시에 dev 환경에서 startup 로그(`api/main.py:226 "Search service configured with chunk_v3 split indices: content=... embed=..."`)를 확인하여 운영 path가 실제로 ES인지 검증.
2. **Corpus 품질 baseline 측정** — `data/chunks_v3/all_chunks.jsonl`에 대해 content_hash 분포·doc_type·길이·노이즈 비율·메타 결측 통계 스크립트 실행, `_workspace/es-evaluation/`에 baseline 저장. cleanup 효과 측정 기반.
3. **`second_stage_doc_retrieve_enabled` default true 승격** — `settings.py:658` 한 줄 변경. paired A/B로 page-hit@1 회복 검증.
4. **헤더 prefix 한 줄 삽입 (B)** — `scripts/chunk_v3/chunkers.py:314-317` `segment_text = f"[{title}]\n{body}"`. 재청킹 후 일부 partition만 재인덱싱하여 A/B.
5. **Bilingual search query 활성화** — agent search_queries 생성 시 원본 한국어 강제 포함(영문 번역 단독 사용 끄기).
6. **content_hash collapse를 RRF dedup에 추가** — `rrf.py:28-37`의 `_dedupe_key`에 content_hash를 chunk_id 다음 우선순위로 추가. dedup된 chunk는 metadata에 collapsed_chunk_ids 보존.
7. **legacy `rag_chunks_dev_v1` 삭제** — alias 확인 후. 약 5.4 GB 회수, v2도 v3 마이그레이션 검증 후 정리 예정.

---

## 6. 미해결·후속 과제

- **Reranker default mismatch 정합** — `settings.py:161` "ms-marco-MiniLM-L-6-v2" vs `cross_encoder.py:33` `DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"`. 어댑터 인스턴스 생성 경로에 따라 실제 로드 모델이 분기되어 재현성·평가 신뢰도 저해. settings 측을 `BAAI/bge-reranker-v2-m3`로 통일하고 어댑터의 `DEFAULT_MODEL`은 "fallback only" 주석으로 의도 명시.
- **search/retrieval-test/agent 라우터별 backend 경로 일관성 점검** — 사용자 화면(retrieval-test, search ES)에서는 ES 경로가 호출될 수 있고, agent default는 local일 수 있어 두 결과가 비교될 때 모델 차이가 아니라 backend 차이일 수 있음. 라우터별 명시 로깅 추가.
- VLM 표 추출/이미지 캡션 추출 — Qwen3-VL 출력 스키마 확장 필요.
- proposition chunking은 비용 검토 필요(Ollama 배치 처리량 측정).
- KoE5 학습/평가 비교 — H/W·시간 큰 비용.
- multi-tenant(`tenant_id`/`project_id`) 활용 시나리오 미정.
- `extra_meta` `enabled=false` 필드를 인덱싱으로 승격할지 결정 필요(매핑 폭발 vs 검색 가능성).
- corpus cleanup 진행 시 dedup 기준(80자 미만 보존 token 화이트리스트, table-noise 정규식)의 PE 도메인 검증 — semiconductor-process-expert 검토 필요.

---

## 부록 A. 핵심 파일 링크

- 매핑: `backend/llm_infrastructure/elasticsearch/mappings.py:246-251` (nori), `manager.py:371-426` (인덱스 생성)
- 청킹: `scripts/chunk_v3/chunkers.py:217-361` (SOP/TS/Setup), `:427-517` (MyService), `:592-702` (GCB), `:149-165` (word-window)
- 메타: `scripts/chunk_v3/section_extractor.py:29-48`, `scripts/chunk_v3/common.py:14-39`
- BM25: `backend/llm_infrastructure/retrieval/engines/bm25.py:18,35`
- Hybrid/RRF: `backend/llm_infrastructure/retrieval/rrf.py:49-114`, `adapters/hybrid.py:85-89`
- Embedding: `backend/llm_infrastructure/embedding/adapters/sentence.py:14-67`
- Reranker: `backend/llm_infrastructure/reranking/adapters/cross_encoder.py:33,78-137`
- Query Expansion: `backend/llm_infrastructure/query_expansion/adapters/llm.py:20-134`
- Section Expander: `backend/llm_infrastructure/retrieval/postprocessors/section_expander.py:101-238`
- Settings: `backend/config/settings.py:44,53`

## 부록 B. 과거 진단 보고서 (정독 권장)

- `docs/2026-01-02_retrieval review.md` — nori 미적용·Reranker off·MQ off 진단
- `docs/2026-01-02_es_guardrails_*.md` — RRF default 변경 권고
- `docs/2026-01-08_meta_guided_hierarchical_rag.md` — RAPTOR 활용 설계
- `docs/2026-02-26_retrieval_quality_analysis_report.md` — Stability@k
- `docs/2026-02-28_agent_retrieval_integrated_diagnosis_report.md` — MQ 비결정성·doc_id 혼합
- `docs/2026-03-01_page_accuracy_improvement_report.md` — page-hit@1 67→100% 실증
- `docs/2026-03-01_sop_questionlist_accuracy_and_tuning_report.md` — SOP 79건 평가
- `docs/2026-02-12_structural_diagnosis_report.md` — FE/BE 필터 키 정합
