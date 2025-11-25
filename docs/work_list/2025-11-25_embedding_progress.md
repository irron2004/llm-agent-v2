# Embedding 구조/테스트 정리 (TEI/OpenAI 추후)

- 날짜(Date): 2025-11-25
- 담당자(Author): hskim
- 역할(Role): BE
- 관련 이슈/티켓: -
- 관련 브랜치/PR: -
- 영역(Tags): [BE], [Embedding], [Tests], [Docs]

---

## 1. 작업 목표 (What & Why)
- 임베딩 모듈을 엔진-어댑터-레지스트리 패턴에 맞춰 재구성하고 테스트를 보강.
- alias(별칭)로 엔진 선택, 캐시/디바이스/모델 설정을 정리.
- TEI/OpenAI 엔진은 추후 구현 예정임을 명시.

## 2. 최종 결과 요약 (Outcome)
- 구조: `embedding/engines/sentence/`(엔진), `adapters/`(레지스트리 어댑터), `registry.py`/`base.py` 정비. TEI/OpenAI 엔진 폴더만 placeholder로 추가.
- 설정: `embedding_device`, `embedding_use_cache`, `embedding_cache_dir` 기본 `.cache/embeddings`로 추가.
- 테스트: 스모크 테스트로 alias 매핑, 캐시 hit/miss, L2 정규화, TEI 모킹, 라운드로빈 디바이스 등을 검증.
- 서비스: `EmbeddingService` 추가로 설정/정책을 한 곳에서 관리.
- 문서: ARCHITECTURE, work_list 업데이트.
- TEI/OpenAI 실제 엔진/어댑터 구현은 **추후 작업**으로 남김.

## 3. 작업 과정 (Timeline / Steps)
1) 임베딩 엔진/어댑터 분리: SentenceTransformer 엔진(`utils/cache/embedder/factory`) + 어댑터(`adapters/sentence.py`, alias 매핑).  
2) TEI 어댑터 이동, OpenAI/TEI 엔진 폴더 placeholder 추가.  
3) 설정 확장: Pydantic Settings에 디바이스/캐시 경로 추가.  
4) 레지스트리 alias 전달(`alias`)로 기본 모델 매핑 정리.  
5) 테스트 추가: 엔진/어댑터/서비스 스모크, TEI 모킹, alias 매핑, 라운드로빈 디바이스.  
6) 서비스 레이어: `EmbeddingService`로 공통 설정 주입/재사용.  
7) 문서/워크리스트 업데이트, TEI/OpenAI는 추후로 명시.

## 4. 추가/수정한 테스트 목록 (Tests)
- `backend/tests/test_embedding_engine.py`
  - alias 매핑/오버라이드, 캐시 hit/miss+L2, encode 편의, TEI 모킹/에러, GPU 라운드로빈 등.
- `backend/tests/test_embedding_service.py`
  - EmbeddingService가 임베더 재사용 및 단일/배치/차원 호출을 정상 처리하는지 확인.

## 5. 설계 및 의사결정 기록 (Design & Decisions)
- 엔진-어댑터-레지스트리 패턴 유지: 엔진은 순수 로직, 어댑터는 레지스트리 연결/설정 주입.
- alias로 엔진 선택: `get_embedder("bge_base" | "koe5" | "multilingual_e5" | "tei")` 등. OpenAI는 추후.
- lazy init 유지(EmbeddingService): 무거운 모델/네트워크 로드를 사용 시점으로 지연.
- 캐시 기본 경로 `.cache/embeddings`: 프로젝트 내부 경로, 필요 시 ENV로 변경.
- 외부 의존이 무거워 테스트는 fake/mock 기반 스모크 위주.

## 6. 회고 및 다음 할 일 (Retrospective / Next Steps)
- TEI/OpenAI 엔진/어댑터 구현 및 실 동작 테스트는 **추후** 진행.
- 엔진 실모델 테스트가 필요하면 slow 마커/옵션으로 분리 예정.
- 필요 시 서비스 레이어에 로깅/재시도/청킹 정책 등을 추가 고려.  
