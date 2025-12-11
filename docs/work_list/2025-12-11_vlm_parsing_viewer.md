# VLM Parsing Viewer 화면 구현

**날짜**: 2025-12-11
**영역**: Frontend
**담당자**: hskim
**상태**: done

## 개요

VLM으로 파싱된 PDF 결과를 시각적으로 검증할 수 있는 React 기반 뷰어 화면을 구현했다.
원본 이미지(왼쪽)와 VLM 파싱 텍스트(오른쪽)를 나란히 비교하여 파싱 품질을 확인할 수 있다.

## 구현 내용

### 1. Frontend - Parsing Viewer (`/parsing`)

```
frontend/src/features/parsing/
├── components/
│   ├── page-viewer.tsx      # 좌우 분할 뷰 (이미지 | VLM 텍스트)
│   ├── page-navigator.tsx   # 페이지 네비게이션 + 썸네일
│   └── *.css
├── hooks/
│   ├── use-ingestion-data.ts  # 페이지 이미지/텍스트 로딩
│   └── use-document-list.ts   # 문서 목록 로딩 (index.html 파싱)
├── pages/
│   └── parsing-page.tsx     # 메인 페이지
└── types/
    └── index.ts
```

### 2. Vite 설정

- `vite.config.ts`에 data 폴더 서빙 플러그인 추가
- `/data` 요청을 프로젝트 루트의 `data/` 폴더로 라우팅

### 3. 환경 변수 (`frontend/.env`)

```env
VITE_INGESTIONS_BASE=/data/ingestions
VITE_DEFAULT_INGESTION_RUN=20251211_070055
```

### 4. VLM 시각화 배치 스크립트

`scripts/vlm_visualize.py` - PDF를 VLM으로 파싱하고 결과를 폴더 구조로 저장:

```
data/ingestions/<timestamp>/<doc_name>/
├── source.pdf
├── pages/page_001.png ...
├── vlm/page_001.txt ...
├── sections.json
└── preview.html
```

## 사용법

1. VLM 파싱 실행:
   ```bash
   python scripts/vlm_visualize.py --pdf data/sample.pdf
   ```

2. Frontend 실행:
   ```bash
   cd frontend && npm run dev
   ```

3. 브라우저에서 `http://localhost:4173/parsing` 접속

## URL 파라미터

- `run`: ingestion 타임스탬프 폴더명
- `doc`: 문서 폴더명 (생략 시 첫 번째 문서 자동 선택)

예: `/parsing?run=20251211_070055`

## 관련 파일

- `frontend/src/app/router.tsx` - `/parsing` 라우트 추가
- `frontend/src/config/env.ts` - parsing 관련 환경변수
- `scripts/vlm_visualize.py` - 배치 시각화 스크립트
