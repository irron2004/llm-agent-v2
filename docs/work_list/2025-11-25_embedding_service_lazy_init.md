# EmbeddingService lazy init 문서화

- 날짜(Date): 2025-11-25
- 담당자(Author): hskim
- 역할(Role): BE
- 관련 이슈/티켓: -
- 관련 브랜치/PR: -
- 영역(Tags): [BE], [Embedding], [Docs]

---

## 1. 작업 목표 (What & Why)
- `EmbeddingService`에서 `_embedder`를 지연 초기화(lazy init)하는 이유를 기록.
- 왜 `__init__`에서 즉시 엔진을 생성하지 않는지, 사용 시점에 생성하는 설계 의도 공유.

## 2. 최종 결과 요약 (Outcome)
- lazy init rationale 정리: 무거운 모델 로드/네트워크 호출을 늦추고, 사용하지 않을 때 비용/부작용을 피함.
- 테스트/모킹 편의성, TEI 네트워크 의존 지연 등 이유를 문서화.
- 필요 시 `__init__`에서 `_get_embedder()`를 호출해 eager init 할 수 있다는 대안 명시.

## 3. 작업 과정 (Timeline / Steps)
1. `EmbeddingService` 코드 동작 방식 확인(`self._embedder`가 None → 첫 사용 시 생성).
2. lazy init 필요성(모델/네트워크 비용, 테스트 모킹) 이유 정리.
3. eager init 대안(항상 사용 시) 가능함을 메모.

## 4. 추가/수정한 테스트 목록 (Tests)
- 문서 작업만 수행; 추가 테스트 없음.

## 5. 설계 및 의사결정 기록 (Design & Decisions)
- 기본은 lazy init 유지: SentenceTransformer/TEI 로드 비용과 네트워크 부작용 최소화.
- 테스트 용이성: 외부 의존을 mock하기 쉬움.
- TEI 등 네트워크 클라이언트는 사용 시점에만 생성하여 불필요한 호출 방지.
- 만약 항상 임베딩을 즉시 준비해야 한다면 `__init__` 끝에 `_get_embedder()` 호출로 eager init 가능.

## 6. 회고 및 다음 할 일 (Retrospective / Next Steps)
- 현재 설계 유지, 별도 후속 작업 없음.
- 향후 OpenAI/기타 엔진 추가 시에도 lazy init 기본 정책을 유지하되, 설정으로 eager 모드를 선택 가능하게 하는 옵션을 검토할 수 있음.
