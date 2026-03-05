# 2026-03-05 운영 개발과 논문 구현 분리 운영 가이드

## 1) 목적
- 운영 기능 개발과 논문 실험 구현을 동시에 진행하되, 코드/데이터/배포 경계가 섞이지 않도록 표준 운영 방식을 정의한다.
- 실험 속도는 유지하면서 운영 안정성(회귀 최소화, 롤백 용이성)을 보장한다.

## 2) 핵심 원칙 (반드시 지킬 것)
1. 실행면 분리: `Prod API`와 `Research API`는 포트/설정/로그를 분리한다.
2. 데이터 분리: 운영 인덱스와 연구 인덱스를 분리하고, 연구에서 운영 인덱스 write를 금지한다.
3. 코드 분리: API/오케스트레이션은 분리하고, 순수 알고리즘(core)만 공유한다.
4. 배포 분리: 운영 배포 파이프라인과 실험 파이프라인을 분리한다.
5. 승격 규칙: 연구 성과는 검증 후 단계적으로 운영에 승격한다.

## 3) 권장 구조 (이 리포 기준)
### 3.1 코드 경계
- 공통(core): `backend/llm_infrastructure/*`, `backend/domain/*`, `backend/services/*`의 순수 로직
- 운영 API: `backend/api/*`
- 연구 전용 진입점(권장): `backend/research_api/*` 또는 `backend/api/main_research.py`
- 연구 실험 스크립트: `scripts/paper_a/*`, `scripts/evaluation/*`
- 연구 결과물: `.sisyphus/evidence/paper-a/*`

### 3.2 데이터 경계
- 운영 ES alias/index: `rag_chunks_{env}_*`
- 연구 ES alias/index: `paper_a_*` 또는 `rag_chunks_exp_*`
- 규칙:
  - 연구는 운영 alias 변경 금지
  - 연구 백필/매핑 변경은 연구 인덱스에서만 수행

## 4) 실행 분리 방식
### 4.1 단기(즉시 적용 가능)
- 동일 앱(`backend.api.main`)을 사용하되, 포트/환경변수/인덱스를 분리해서 2개 프로세스로 실행한다.
- 예시 원칙:
  - 운영 API: `127.0.0.1:18021`, `SEARCH_ES_ENV=dev`
  - 연구 API: `127.0.0.1:18031`, `SEARCH_ES_ENV=exp` (또는 연구 전용 prefix)
  - 로그 경로 분리: 운영/연구 별도 파일
- 실행 예시:
```bash
# 운영 API
API_LOG_FILE_PATH=./data/logs/api/prod.log SEARCH_ES_ENV=dev \
uv run uvicorn backend.api.main:app --host 127.0.0.1 --port 18021

# 연구 API
API_LOG_FILE_PATH=./data/logs/api/research.log SEARCH_ES_ENV=exp \
uv run uvicorn backend.api.main:app --host 127.0.0.1 --port 18031
```

### 4.2 중기(권장)
- 연구 전용 API 엔트리포인트를 추가해 라우터와 기본 설정을 분리한다.
- 운영 API에는 실험 라우트가 노출되지 않도록 유지한다.

## 5) Git/브랜치 운영 규칙
- 운영 개발: `main` 기준 기능 브랜치 (`feat/*`, `fix/*`)
- 논문 실험: `paper-a/*` 브랜치
- 원칙:
  - 실험 커밋을 운영 브랜치에 직접 합치지 않는다.
  - 검증된 공통 로직만 cherry-pick 또는 별도 PR로 승격한다.
  - 실험 산출물(`.sisyphus/evidence/*`)은 코드 PR에 포함하지 않는다.

## 6) CI 분리 규칙
### 6.1 Prod CI (필수 게이트)
- 단위/통합 테스트
- API smoke test
- 회귀 체크(핵심 라우트/검색 품질 최소 기준)

### 6.2 Research CI (비동기/수동 트리거 가능)
- 평가 스크립트 실행
- 지표 산출(`per_query.csv`, `summary_all.csv`, `bootstrap_ci.json`)
- 실험 리포트 생성

## 7) 설정/플래그 규칙
- 운영 기본값:
  - 실험 기능 플래그는 기본 `OFF`
  - 운영에서 실험 전용 인덱스/라우트 접근 금지
- 연구 기본값:
  - 실험 플래그 `ON` 가능
  - 단, 운영 자원(인덱스/alias) 변경 권한은 제거

## 8) 논문 구현 결과의 운영 승격 절차
1. 연구 브랜치에서 오프라인 성능 검증(재현 가능한 evidence 포함)
2. 공통 로직 추출(연구 API 의존 제거)
3. 운영 안정성 테스트(회귀/부하/실패 복구)
4. feature flag로 운영 반영(기본 OFF)
5. 단계적 활성화 후 모니터링

## 9) 일일 작업 체크리스트
- 오늘 작업이 운영/연구 중 어디인지 명시했다.
- 실행 중인 API 포트와 ES env가 분리되어 있다.
- 연구 코드가 운영 기본 동작을 바꾸지 않는다.
- 실험 결과는 `.sisyphus/evidence/paper-a/*`에만 기록했다.
- PR 대상이 운영인지 연구인지 라벨/브랜치가 일치한다.

## 10) 금지 패턴
- 연구 코드에서 운영 alias를 직접 update/switch
- 운영 API 기본 설정에 실험 파라미터를 섞어 커밋
- 실험 결과 파일을 운영 코드 PR에 포함
- 실험 실패를 숨기기 위해 기준/데이터를 PR마다 변경

## 11) 추천 실행 순서
1. 단기 분리부터 즉시 적용(포트/환경/인덱스/로그 분리)
2. 1~2주 내 연구 전용 API 엔트리포인트 분리
3. CI 분리 및 승격 체크리스트를 PR 템플릿에 반영
4. Paper A 등 실험은 연구 API + 스크립트 기반으로 고정 운영
