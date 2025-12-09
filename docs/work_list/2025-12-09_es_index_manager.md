# ES Index Manager 구현 (SU-2509)
- 날짜: 2025-12-09
- 담당자: hskim
- 역할: BE
- 관련 이슈/티켓: SU-2509 (ClickUp)
- 관련 브랜치/PR: feature/SU-2509-es-index-manager
- 영역(Tags): [BE], [RAG], [Elasticsearch], [Index]

## 1. 작업 목표(What & Why)
- RAG 문서 청크를 저장할 ES 인덱스 매핑 설계
- Dev/Prod 환경 인덱스 이름/alias 규칙 정립 및 롤링 전략 구현
- 인덱스 생성/삭제/alias 관리 CLI 제공

## 2. 최종 결과 요약(Outcome)
- ES 인덱스 매핑 스키마 구현 (`backend/llm_infrastructure/elasticsearch/mappings.py`)
- `EsIndexManager` 클래스 구현 (`backend/llm_infrastructure/elasticsearch/manager.py`)
- CLI 스크립트 작성 (`backend/llm_infrastructure/elasticsearch/cli.py`)
- Settings 확장 (ES 인덱스 관련 설정 추가)
- elasticsearch 패키지 의존성 추가

## 3. 작업 과정(Timeline/Steps)
1. SU-2509 티켓 요구사항 분석
2. 브랜치 생성: `feature/SU-2509-es-index-manager`
3. ES 인덱스 매핑 JSON 스키마 설계
   - 기본 키: doc_id, chunk_id, page
   - 텍스트: content, search_text
   - 벡터: embedding (dense_vector, dims 설정 가능)
   - 메타/필터: lang, doc_type, tenant_id, project_id 등
   - _meta.pipeline: 파이프라인 정보 저장
4. EsIndexManager 클래스 구현
   - 인덱스 이름/alias 규칙: `rag_chunks_{env}_v{version}` / `rag_chunks_{env}_current`
   - create_index, delete_index, switch_alias 등 핵심 메서드
5. CLI 스크립트 작성 (create, delete, switch, list, info, health)
6. SearchSettings 확장 (es_env, es_index_prefix, es_index_version, es_embedding_dims)
7. .env.example 업데이트
8. elasticsearch 패키지 의존성 추가

## 4. 추가/수정한 테스트(Tests)
- (단위 테스트는 별도 작성 예정)
- Dev 환경에서 CLI 스모크 테스트 수행 예정:
  - `ES_HOST=http://localhost:9200 ES_ENV=dev ES_INDEX_PREFIX=rag_chunks python -m backend.llm_infrastructure.elasticsearch.cli create --version 1 --dims 1024 --switch-alias`
  - `python -m backend.llm_infrastructure.elasticsearch.cli list`

## 5. 설계 및 의사결정(Design & Decisions)

### 인덱스 네이밍 규칙
```
인덱스: rag_chunks_{env}_v{version}
  예: rag_chunks_dev_v1, rag_chunks_prod_v1

Alias: rag_chunks_{env}_current
  예: rag_chunks_dev_current, rag_chunks_prod_current
```

### 롤링 업데이트 전략
1. 새 버전 인덱스 생성 (v2)
2. 데이터 마이그레이션 (필요시)
3. alias를 새 인덱스로 스위칭 (atomic)
4. 구버전 인덱스는 롤백 대비 보관 또는 삭제

### 필드 설계 요약
| 구분 | 필드 | 타입 |
|------|------|------|
| 기본 키 | doc_id, chunk_id | keyword |
| | page | integer |
| 텍스트 | content | text |
| | search_text | text |
| 벡터 | embedding | dense_vector |
| 메타 | lang, doc_type, tenant_id, project_id | keyword |
| | pipeline_version, content_hash | keyword |
| 옵션 | page_image_path, bbox, quality_score | - |

### CLI 사용법
```bash
# 인덱스 생성
python -m backend.llm_infrastructure.elasticsearch.cli create --version 1 --dims 1024 --switch-alias

# 인덱스 목록
python -m backend.llm_infrastructure.elasticsearch.cli list

# alias 스위칭
python -m backend.llm_infrastructure.elasticsearch.cli switch --version 2

# 인덱스 삭제
python -m backend.llm_infrastructure.elasticsearch.cli delete --version 1 --force
```

## 6. 회고 및 다음 할 일(Retrospective / Next Steps)
- [ ] 테스트 코드 작성 (EsIndexManager 단위 테스트)
- [ ] SU-2511 (ES Retriever) 연동: 검색 시 alias 사용
- [ ] SU-2512 (Hybrid 점수 결합) 호환성 검증
- [ ] SU-2513 (성능/운영) 샤드/레플리카 설정 최적화
- [ ] Korean 분석기(nori) 지원 추가 검토
