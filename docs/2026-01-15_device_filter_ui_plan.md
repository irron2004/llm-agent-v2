# 기기 필터 UI 화면 기획

## 1. 목표

- 조회된 문서에서 기기정보(device_name)를 별도로 표시
- 기기별 필터링 기능 제공
- 사용자가 특정 기기에 대한 문서만 검색할 수 있도록 함

---

## 2. 현재 데이터 현황

### 2.1 주요 기기 목록 (50개+)

| 기기명 | 문서 수 | 비고 |
|--------|---------|------|
| SUPRA N | 267,786 | 최다 |
| SUPRA Vplus | 101,826 | |
| SUPRA V | 49,862 | |
| INTEGER plus | 37,344 | |
| SUPRA XP | 36,132 | |
| PRECIA | 22,422 | |
| SUPRA Vm | 19,800 | |
| TIGMA Vplus | 17,254 | |
| SUPRA IV | 16,846 | |
| ZEDIUS XP | 14,634 | |
| ... | ... | 총 50개+ |

### 2.2 문서 타입

| 타입 | 문서 수 |
|------|---------|
| myservice | 648,964 |
| SOP | 22,984 |
| Manual | 2,082 |
| Troubleshooting Guide | 1,710 |
| 설치 매뉴얼 | 1,036 |

### 2.3 ES 필드 구조

```json
{
  "doc_id": "40088819",
  "device_name": "SUPRA N",
  "doc_type": "myservice",
  "chapter": "status|action|cause|result",
  "content": "...",
  "embedding": [...]
}
```

---

## 3. UI 설계 방안

### 방안 A: 사이드바 필터 패널

```
┌──────────────────────────────────────────────────────────┐
│ Left Sidebar          │ Main Content                     │
│ ──────────────────────│──────────────────────────────────│
│ [PE Agent]            │                                  │
│ [+ New Chat]          │  채팅 메시지 영역                │
│                       │                                  │
│ ┌──────────────────┐  │                                  │
│ │ 🔍 기기 필터     │  │                                  │
│ ├──────────────────┤  │                                  │
│ │ □ SUPRA N (267K) │  │                                  │
│ │ □ SUPRA Vplus    │  │                                  │
│ │ □ SUPRA V        │  │                                  │
│ │ □ INTEGER plus   │  │                                  │
│ │ [+ 더보기...]    │  │                                  │
│ └──────────────────┘  │                                  │
│                       │                                  │
│ ┌──────────────────┐  │                                  │
│ │ 📄 문서 타입     │  │                                  │
│ ├──────────────────┤  │                                  │
│ │ □ SOP            │  │                                  │
│ │ □ Manual         │  │                                  │
│ │ □ TS Guide       │  │                                  │
│ └──────────────────┘  │                                  │
│                       │──────────────────────────────────│
│ [Chat] [Search] ...   │  [입력창]                        │
└──────────────────────────────────────────────────────────┘
```

**장점:**
- 항상 보이는 필터
- 검색 전 기기 선택 가능

**단점:**
- 사이드바 공간 제한
- 기기가 많으면 스크롤 필요

---

### 방안 B: 입력창 위 필터 바

```
┌──────────────────────────────────────────────────────────┐
│                    Main Content                          │
│                                                          │
│   [User] SUPRA N의 PM 절차를 알려줘                      │
│                                                          │
│   [Assistant] 답변...                                    │
│     ┌─────────────────────────────────────┐              │
│     │ 📎 참조 문서                        │              │
│     │ ┌─────┐ ┌─────┐ ┌─────┐            │              │
│     │ │SUPRA│ │SUPRA│ │SUPRA│            │              │
│     │ │  N  │ │  N  │ │  N  │            │              │
│     │ │p.12 │ │p.34 │ │p.56 │            │              │
│     │ └─────┘ └─────┘ └─────┘            │              │
│     └─────────────────────────────────────┘              │
│                                                          │
│──────────────────────────────────────────────────────────│
│ ┌──────────────────────────────────────────────────────┐ │
│ │ 필터: [SUPRA N ▼] [SOP ▼] [전체 ✕]                   │ │
│ └──────────────────────────────────────────────────────┘ │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ 질문을 입력하세요...                          [전송] │ │
│ └──────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

**장점:**
- 검색 직전에 필터 선택
- 메인 콘텐츠 공간 확보

**단점:**
- 드롭다운 UI 복잡
- 모바일에서 공간 부족

---

### 방안 C: 참조 문서에 기기 태그 표시 + 클릭 필터

```
┌──────────────────────────────────────────────────────────┐
│ 📎 참조 문서 (3개)                              [필터 ▼]│
├──────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────┐  │
│ │ [🏷️ SUPRA N] [📄 SOP]                               │  │
│ │ PM_FCIP_R5 - Page 12                                │  │
│ │ "FCIP 모듈의 정기 점검 절차는..."                   │  │
│ │ [이미지 썸네일]                                     │  │
│ └─────────────────────────────────────────────────────┘  │
│ ┌─────────────────────────────────────────────────────┐  │
│ │ [🏷️ SUPRA N] [📄 SOP]                               │  │
│ │ PM_Temp_Controller - Page 8                         │  │
│ │ "온도 컨트롤러 교체 시..."                          │  │
│ └─────────────────────────────────────────────────────┘  │
│                                                          │
│ ┌─ 기기별 문서 수 ───────────────────────────────────┐  │
│ │ SUPRA N: 2개  │  SUPRA XP: 1개  │  [+ 필터 적용]   │  │
│ └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

**장점:**
- 검색 결과에서 기기 정보 직관적
- 클릭으로 추가 필터 가능

**단점:**
- 검색 후에만 필터 가능
- 구현 복잡도 증가

---

## 4. 권장안: 하이브리드 (B + C)

### 4.1 입력 영역 필터 바 + 결과 태그

1. **입력창 위**: 기기/문서타입 드롭다운 필터
2. **참조 문서**: 기기 태그 표시 + 클릭 필터

```
┌──────────────────────────────────────────────────────────┐
│                    채팅 영역                             │
│                                                          │
│   [Assistant] 답변...                                    │
│     📎 참조 문서                                         │
│     ┌─────────────────────────────────────┐              │
│     │ [🏷️ SUPRA N] [📄 SOP]               │ ← 클릭 가능 │
│     │ PM_FCIP_R5 - p.12                   │              │
│     │ [이미지]                            │              │
│     └─────────────────────────────────────┘              │
│                                                          │
│──────────────────────────────────────────────────────────│
│ 🔧 기기: [전체 ▼]  📄 타입: [전체 ▼]  [필터 초기화]     │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ 질문을 입력하세요...                          [전송] │ │
│ └──────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### 4.2 기기 선택 드롭다운 UI

```
┌─ 🔧 기기 선택 ────────────────────┐
│ 🔍 기기 검색...                   │
├───────────────────────────────────┤
│ ⭐ 자주 사용                      │
│   ☑️ SUPRA N (267K)               │
│   □ SUPRA Vplus (101K)            │
│   □ INTEGER plus (37K)            │
├───────────────────────────────────┤
│ 📋 전체 기기                      │
│   □ PRECIA (22K)                  │
│   □ SUPRA Vm (19K)                │
│   □ TIGMA Vplus (17K)             │
│   □ ... (더보기)                  │
├───────────────────────────────────┤
│ [선택 초기화]        [적용] (1개) │
└───────────────────────────────────┘
```

---

## 5. 구현 범위

### Phase 1: 기본 필터 (MVP)

- [ ] 입력창 위 기기 드롭다운 필터
- [ ] 기기 목록 API (`GET /api/devices`)
- [ ] 검색 시 기기 필터 적용
- [ ] 참조 문서에 기기 태그 표시

### Phase 2: 고급 필터

- [ ] 문서 타입 드롭다운 필터
- [ ] 다중 기기 선택
- [ ] 필터 상태 URL 파라미터 저장
- [ ] 기기 태그 클릭 시 필터 적용

### Phase 3: UX 개선

- [ ] 자주 사용 기기 저장 (로컬스토리지)
- [ ] 기기 검색 (자동완성)
- [ ] 기기별 문서 수 표시
- [ ] 필터 프리셋 저장

---

## 6. API 설계

### 6.1 기기 목록 API

```
GET /api/devices
Response:
{
  "devices": [
    { "name": "SUPRA N", "doc_count": 267786 },
    { "name": "SUPRA Vplus", "doc_count": 101826 },
    ...
  ],
  "total": 50
}
```

### 6.2 검색 API 수정

```
POST /api/agent/run
Body:
{
  "message": "PM 절차 알려줘",
  "filters": {
    "device_names": ["SUPRA N", "SUPRA XP"],
    "doc_types": ["SOP", "Manual"]
  }
}
```

### 6.3 검색 응답 수정

```json
{
  "answer": "...",
  "retrieved_docs": [
    {
      "id": "doc_001",
      "title": "PM_FCIP_R5",
      "snippet": "...",
      "device_name": "SUPRA N",  // 추가
      "doc_type": "SOP",          // 추가
      "page": 12,
      "page_image_url": "/api/assets/docs/..."
    }
  ]
}
```

---

## 7. 프론트엔드 컴포넌트

### 7.1 새로운 컴포넌트

```
src/
├── components/
│   └── device-filter/
│       ├── index.tsx              # 메인 필터 컴포넌트
│       ├── device-dropdown.tsx    # 기기 드롭다운
│       ├── doc-type-dropdown.tsx  # 문서타입 드롭다운
│       └── device-filter.css
└── features/
    └── chat/
        └── hooks/
            └── use-device-filter.ts  # 필터 상태 관리
```

### 7.2 상태 관리

```typescript
interface FilterState {
  selectedDevices: string[];
  selectedDocTypes: string[];
}

// Context 또는 URL 파라미터로 관리
```

---

## 8. 타임라인

| 단계 | 작업 | 예상 |
|------|------|------|
| 1 | API 설계 및 구현 | - |
| 2 | 기기 목록 컴포넌트 | - |
| 3 | 필터 드롭다운 UI | - |
| 4 | 검색 API 필터 연동 | - |
| 5 | 참조 문서 태그 표시 | - |
| 6 | 테스트 및 개선 | - |

---

## 9. 참고사항

- 기기명 정규화 필요 (SUPRA N, SUPRAN, SUPRA N series 등 통합)
- 문서타입도 정규화 고려 (SOP, Global SOP, Global_SOP 등)
- 모바일 대응 필요 (드롭다운 → 모달)

---

# 검색 문서 이미지 표시 화면 기획

## 1. 목표

- 검색된 문서를 텍스트 스니펫 대신 페이지 이미지로 표시하여 가독성 향상
- 오른쪽 사이드바에서 문서 이미지를 직관적으로 확인할 수 있도록 개선
- 이미지 확대/축소 및 상세 보기 기능 제공

### 1.1 실제 화면 확인 요약 (2026-01-15)

**확인 URL**: http://10.10.100.45:9097

**확인 내용**:
- 오른쪽 사이드바에 "검색된 문서 (10개)" 섹션 표시
- 첫 번째 문서에만 이미지가 표시되고, 나머지 9개는 텍스트 스니펫만 표시
- 텍스트 스니펫이 markdown 형식의 원시 텍스트로 표시되어 가독성 매우 낮음
- HTML 태그(`<br>`)가 그대로 노출되어 표시됨
- 이미지 크기가 작아서 가독성이 떨어짐
- 이미지 확대 기능 없음

**핵심 문제점**:
1. 텍스트 스니펫의 가독성이 매우 떨어짐 (markdown 원시 텍스트, HTML 태그 노출)
2. 첫 번째 문서만 이미지가 표시되고 나머지는 텍스트만 표시됨
3. 이미지 크기가 작아서 가독성이 떨어짐

---

## 2. 현재 상황 분석

### 2.1 실제 화면 확인 결과 (2026-01-15)

**위치**: 
- 오른쪽 사이드바: `frontend/src/components/layout/index.tsx` - `RetrievedDocsContent` 컴포넌트
- 메인 채팅 영역: `frontend/src/features/chat/components/message-item.tsx` - `MessageItem` 컴포넌트

**현재 동작**:
- 오른쪽 사이드바에 "검색된 문서" 섹션 표시
- 각 문서는 다음 정보 포함:
  - 문서 번호 (#1, #2, ...)
  - 제목/스니펫 (텍스트 형태)
  - 페이지 번호 (p.52, p.69 등)
  - 점수 (8.153, 7.934 등)
- 첫 번째 문서에만 이미지가 표시됨 (나머지는 텍스트만)
- 이미지가 있으면 작은 크기로 표시 (max-height: 300px 추정)
- 메인 채팅 영역에도 "검색 문서 (10)" 섹션이 있으나 접혀있음

**실제 확인된 문제점**:

1. **텍스트 스니펫 가독성 문제** (최우선 해결 필요 ⚡):
   - 텍스트 스니펫이 markdown 형식의 원시 텍스트로 표시됨
   - 예: ````markdown # Global SOP_SUPRA N series_SW_TM_CTC PATCH | Global SOP No : | | |-`
   - HTML 태그가 그대로 노출됨: `<br>` 태그가 텍스트로 표시됨
   - 긴 텍스트가 잘려서 표시되어 전체 내용 파악 어려움
   - 표 형식의 데이터가 텍스트로만 표시되어 구조 파악 불가능
   - **해결 필요**: Markdown 렌더링 적용하여 보기 좋은 형태로 표시

2. **이미지 표시 문제**:
   - 첫 번째 문서에만 이미지가 표시되고, 나머지 문서는 텍스트만 표시됨
   - 이미지 크기가 작아서 가독성이 떨어짐 (max-height: 300px)
   - 이미지 클릭 시 확대 기능 없음
   - 이미지가 없는 문서는 텍스트 스니펫만 표시되어 가독성 매우 낮음

3. **사용자 경험 문제**:
   - 여러 문서를 비교하기 어려움
   - 문서의 실제 레이아웃을 확인할 수 없음
   - 표, 그림 등이 포함된 문서의 내용 파악 불가능
   - 스크롤이 길어져서 전체 문서 목록 확인이 불편함

### 2.2 현재 구현 상태 (코드 기준)

**위치**: `frontend/src/components/layout/index.tsx` - `RetrievedDocsContent` 컴포넌트

**현재 동작**:
- `page_image_url`이 있으면 이미지 표시 (max-height: 300px)
- 이미지가 없으면 텍스트 스니펫 표시 (max-height: 200px)
- 이미지 로딩 실패 시 텍스트 스니펫으로 fallback

### 2.2 데이터 구조

```typescript
interface RetrievedDoc {
  id: string;
  title: string;
  snippet: string;
  score?: number | null;
  score_percent?: number | null;
  metadata?: Record<string, unknown> | null;
  page?: number | null;
  page_image_url?: string | null;  // 이미지 URL
}
```

### 2.3 API 응답 구조

```json
{
  "retrieved_docs": [
    {
      "id": "40088819",
      "title": "PM_FCIP_R5",
      "snippet": "...",
      "page": 12,
      "page_image_url": "/api/assets/docs/40088819/pages/12"
    }
  ]
}
```

---

## 3. UI 설계 방안

### 방안 A: 이미지 우선 표시 (권장) ⭐

**핵심 아이디어**: 
- **모든 문서에 대해** `page_image_url`이 있으면 이미지를 우선 표시
- 이미지 크기를 더 크게 (max-height: 500px ~ 600px)
- 이미지 클릭 시 모달/라이트박스로 확대 보기
- 텍스트 스니펫은 보조 정보로 표시 (토글 가능)

**현재 문제점 해결**:
- ✅ 텍스트 스니펫의 가독성 문제 → 이미지로 대체
- ✅ 첫 번째 문서만 이미지 표시 → 모든 문서에 이미지 표시
- ✅ 이미지 크기 작음 → 크기 증가 및 확대 기능 추가

```
┌─────────────────────────────────────────┐
│ 검색된 문서 (10개)                      │
├─────────────────────────────────────────┤
│                                         │
│ ┌─────────────────────────────────────┐ │
│ │ #1  Global SOP_SUPRA N...  p.52    │ │
│ │     8.153                          │ │
│ ├─────────────────────────────────────┤ │
│ │ ┌───────────────────────────────┐   │ │
│ │ │                               │   │ │
│ │ │     [페이지 이미지]            │   │ │
│ │ │     (max-height: 500px)       │   │ │
│ │ │     (클릭 시 확대)             │   │ │
│ │ │                               │   │ │
│ │ └───────────────────────────────┘   │ │
│ │ [📄 텍스트 보기] [🔍 확대]          │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ ┌─────────────────────────────────────┐ │
│ │ #2  set_up_manual_geneva...  p.69  │ │
│ │     7.934                          │ │
│ ├─────────────────────────────────────┤ │
│ │ [페이지 이미지 - 동일한 크기]       │ │
│ │ [📄 텍스트 보기] [🔍 확대]          │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ ... (나머지 8개 문서도 동일한 형식)     │
│                                         │
└─────────────────────────────────────────┘
```

**장점**:
- ✅ 가독성 크게 향상 (텍스트 스니펫 대신 실제 문서 이미지)
- ✅ 문서의 실제 레이아웃 확인 가능 (표, 그림 등 포함)
- ✅ 모든 문서에 일관된 표시 방식 적용
- ✅ 이미지와 텍스트 모두 제공 (토글 가능)

**단점**:
- ⚠️ 이미지 로딩 시간 증가 가능 (하지만 가독성 향상이 더 중요)
- ⚠️ 네트워크 대역폭 사용 증가 (캐싱으로 완화 가능)

---

### 방안 B: 썸네일 + 확대 보기

**핵심 아이디어**:
- 기본적으로 썸네일 이미지 표시 (작은 크기)
- 클릭 시 모달에서 전체 크기 이미지 표시
- 텍스트 스니펫은 항상 함께 표시

```
┌─────────────────────────────────────────┐
│ 검색된 문서 (5개)                       │
├─────────────────────────────────────────┤
│ ┌─────────────────────────────────────┐ │
│ │ #1  PM_FCIP_R5  p.12  [85.3%]      │ │
│ ├─────────────────────────────────────┤ │
│ │ ┌──────┐                            │ │
│ │ │ 썸네일│  "FCIP 모듈의 정기 점검..." │ │
│ │ │ 이미지│                            │ │
│ │ └──────┘                            │ │
│ │ [🔍 확대 보기]                      │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

**장점**:
- 초기 로딩 빠름
- 공간 효율적

**단점**:
- 썸네일 크기가 작아 가독성 여전히 낮음
- 추가 클릭 필요

---

### 방안 C: 그리드 레이아웃 (다중 문서 비교)

**핵심 아이디어**:
- 여러 문서를 그리드 형태로 나란히 표시
- 각 문서를 카드 형태로 표시
- 이미지 크기 조절 가능

```
┌─────────────────────────────────────────┐
│ 검색된 문서 (5개)  [📋 리스트] [⊞ 그리드]│
├─────────────────────────────────────────┤
│ ┌──────┐ ┌──────┐ ┌──────┐            │
│ │ #1   │ │ #2   │ │ #3   │            │
│ │ 이미지│ │ 이미지│ │ 이미지│            │
│ │ p.12 │ │ p.8  │ │ p.15 │            │
│ └──────┘ └──────┘ └──────┘            │
│ ┌──────┐ ┌──────┐                      │
│ │ #4   │ │ #5   │                      │
│ │ 이미지│ │ 이미지│                      │
│ └──────┘ └──────┘                      │
└─────────────────────────────────────────┘
```

**장점**:
- 여러 문서 한눈에 비교 가능
- 공간 활용 효율적

**단점**:
- 각 이미지 크기 작아짐
- 구현 복잡도 증가

---

## 4. 권장안: 방안 A (이미지 우선 표시) + 확대 기능

### 4.1 기본 레이아웃 (오른쪽 사이드바 기준)

**현재 구조**:
- 오른쪽 사이드바 너비: 약 320px (CSS 변수: `--right-sidebar-width`)
- 각 문서 카드: 배경색, 왼쪽 테두리 강조, 패딩 포함

**개선된 레이아웃**:

```
┌─────────────────────────────────────────┐
│ 검색된 문서 (10개)                      │
├─────────────────────────────────────────┤
│                                         │
│ ┌─────────────────────────────────────┐ │
│ │ #1  Global SOP_SUPRA N series...    │ │
│ │     p.52  8.153                     │ │
│ ├─────────────────────────────────────┤ │
│ │                                     │ │
│ │   ┌─────────────────────────────┐   │ │
│ │   │                             │   │ │
│ │   │    [페이지 이미지]          │   │ │
│ │   │    (max-height: 500px)      │   │ │
│ │   │    (width: 100%)            │   │ │
│ │   │                             │   │ │
│ │   └─────────────────────────────┘   │ │
│ │                                     │ │
│ │ [📄 텍스트] [🔍 확대]              │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ ┌─────────────────────────────────────┐ │
│ │ #2  set_up_manual_geneva_stp300...  │ │
│ │     p.69  7.934                     │ │
│ ├─────────────────────────────────────┤ │
│ │ [페이지 이미지 - 동일한 크기]       │ │
│ │ [📄 텍스트] [🔍 확대]              │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ ... (나머지 8개 문서)                   │
│                                         │
└─────────────────────────────────────────┘
```

**주요 변경사항**:
1. **모든 문서에 이미지 표시**: 현재는 첫 번째만 이미지가 표시되지만, 모든 문서에 이미지 표시
2. **이미지 크기 증가**: 300px → 500px로 증가하여 가독성 향상
3. **일관된 레이아웃**: 모든 문서가 동일한 형식으로 표시
4. **액션 버튼 추가**: 텍스트 보기 토글 및 확대 기능

### 4.2 이미지 확대 모달 (라이트박스)

```
┌─────────────────────────────────────────┐
│ [×]                                     │
│                                         │
│         ┌───────────────────────┐       │
│         │                       │       │
│         │   [전체 크기 이미지]  │       │
│         │                       │       │
│         └───────────────────────┘       │
│                                         │
│ PM_FCIP_R5 - Page 12  [85.3%]          │
│                                         │
│ [← 이전]  [다운로드]  [다음 →]          │
└─────────────────────────────────────────┘
```

### 4.3 텍스트 보기 토글

```
┌─────────────────────────────────────────┐
│ #1  PM_FCIP_R5  p.12  [85.3%]          │
├─────────────────────────────────────────┤
│ [이미지 표시 중...]                     │
│                                         │
│ [📄 텍스트 보기] ← 클릭 시              │
└─────────────────────────────────────────┘

↓ 클릭 후

┌─────────────────────────────────────────┐
│ #1  PM_FCIP_R5  p.12  [85.3%]          │
├─────────────────────────────────────────┤
│ "FCIP 모듈의 정기 점검 절차는 다음과    │
│  같습니다. 먼저 전원을 차단하고..."    │
│                                         │
│ [🖼️ 이미지 보기] ← 클릭 시             │
└─────────────────────────────────────────┘
```

---

## 5. 상세 UI 스펙

### 5.1 문서 카드 구조

```
┌─────────────────────────────────────────┐
│ 헤더 영역                               │
│ - 순위 (#1, #2, ...)                   │
│ - 문서 제목                             │
│ - 페이지 번호 (p.12)                    │
│ - 점수 (85.3%)                         │
├─────────────────────────────────────────┤
│ 이미지 영역                             │
│ - 페이지 이미지 (max-height: 500px)    │
│ - 로딩 상태 표시                        │
│ - 에러 시 텍스트 스니펫 표시            │
├─────────────────────────────────────────┤
│ 액션 버튼                               │
│ - [📄 텍스트 보기/이미지 보기] 토글     │
│ - [🔍 확대] (모달 열기)                 │
│ - [⬇ 다운로드] (선택사항)              │
└─────────────────────────────────────────┘
```

### 5.2 이미지 표시 규칙

1. **우선순위**:
   - `page_image_url`이 있으면 → 이미지 표시
   - 이미지 로딩 실패 → 텍스트 스니펫 표시
   - `page_image_url`이 없으면 → 텍스트 스니펫 표시

2. **이미지 크기**:
   - 기본: `max-height: 500px` (현재 300px에서 증가)
   - 너비: `max-width: 100%`
   - 비율 유지: `object-fit: contain`

3. **로딩 상태**:
   - 스켈레톤 UI 또는 스피너 표시
   - 로딩 실패 시 에러 메시지 + 텍스트 스니펫

### 5.3 확대 모달 (라이트박스)

**기능**:
- 현재 문서의 전체 크기 이미지 표시
- 이전/다음 문서 네비게이션
- 키보드 단축키 지원 (← → ESC)
- 다운로드 기능

**레이아웃**:
- 전체 화면 오버레이
- 중앙에 이미지 배치
- 하단에 문서 정보 및 네비게이션 버튼

---

## 6. 구현 범위

### Phase 1: 기본 이미지 표시 개선 (MVP) - 최우선 ⚡

**목표**: 텍스트 스니펫 가독성 문제 해결

#### 1-1. 텍스트 스니펫 Markdown 렌더링 (최우선) 🔥

- [ ] **Markdown 렌더링 적용**
  - 기존 `MarkdownContent` 컴포넌트 재사용 (`frontend/src/features/chat/components/markdown-content.tsx`)
  - `RetrievedDocsContent` 컴포넌트에서 텍스트 스니펫을 `<div>{doc.snippet}</div>` 대신 `<MarkdownContent content={doc.snippet} />` 사용
  - HTML 태그 제거 및 markdown 파싱 적용
  - 표 형식 데이터를 실제 표로 렌더링

- [ ] **텍스트 전처리**
  - markdown 코드 블록 제거 (예: ````markdown` 제거)
  - HTML 태그 정리 (`<br>` → 줄바꿈)
  - 불필요한 공백 정리

- [ ] **스타일 적용**
  - markdown 콘텐츠에 적절한 스타일 적용
  - 표 스크롤 처리 (가로 스크롤)
  - 코드 블록 스타일 적용

**예상 효과**:
- ✅ 텍스트 스니펫이 보기 좋은 형태로 표시됨
- ✅ 표 형식 데이터가 실제 표로 렌더링되어 구조 파악 가능
- ✅ HTML 태그가 제거되고 적절히 렌더링됨
- ✅ Markdown 형식이 제대로 적용됨 (제목, 리스트, 강조 등)

#### 1-2. 이미지 표시 개선

- [ ] **모든 문서에 이미지 표시** (현재는 첫 번째만 표시됨)
  - `RetrievedDocsContent` 컴포넌트 수정
  - 모든 문서에 대해 `page_image_url` 확인 및 이미지 표시
- [ ] **이미지 크기 증가** (max-height: 300px → 500px)
  - CSS 수정: `.retrieved-doc-image` 스타일 업데이트
- [ ] **이미지 로딩 상태 표시**
  - 스켈레톤 UI 또는 스피너 추가
- [ ] **이미지 에러 처리 개선**
  - 이미지 로딩 실패 시 텍스트 스니펫으로 fallback
- [ ] **텍스트/이미지 토글 기능** (선택사항)
  - 이미지와 텍스트 스니펫 간 전환 가능

**예상 효과**:
- ✅ 텍스트 스니펫의 가독성 문제 완전 해결
- ✅ 모든 문서에 일관된 표시 방식 적용
- ✅ 문서의 실제 레이아웃 확인 가능

### Phase 2: 확대 기능 추가

**목표**: 이미지 상세 보기 기능 제공

- [ ] 이미지 클릭 시 모달 열기
- [ ] 모달에서 전체 크기 이미지 표시
- [ ] 이전/다음 문서 네비게이션
- [ ] 키보드 단축키 지원 (← → ESC)

### Phase 3: 고급 기능

**목표**: 사용자 경험 추가 개선

- [ ] 이미지 다운로드 기능
- [ ] 이미지 줌 인/아웃 (핀치 제스처)
- [ ] 그리드/리스트 뷰 전환 (선택사항)
- [ ] 이미지 캐싱 최적화
- [ ] Lazy Loading 구현

---

## 7. 컴포넌트 설계

### 7.1 기존 컴포넌트 활용

**Markdown 렌더링**:
- 기존 `MarkdownContent` 컴포넌트 재사용 (`frontend/src/features/chat/components/markdown-content.tsx`)
- `react-markdown`, `remark-gfm`, `remark-math` 등 이미 설치되어 있음
- 표, 코드 블록, 수식 등 다양한 markdown 기능 지원

### 7.2 새로운 컴포넌트 (선택사항)

```
frontend/src/
├── components/
│   └── retrieved-docs/
│       ├── index.tsx                    # 메인 컨테이너
│       ├── doc-card.tsx                  # 개별 문서 카드
│       ├── doc-image.tsx                 # 이미지 컴포넌트
│       ├── doc-snippet.tsx               # 텍스트 스니펫 컴포넌트 (Markdown 렌더링)
│       ├── image-lightbox.tsx            # 확대 모달
│       └── retrieved-docs.css
```

### 7.3 텍스트 스니펫 컴포넌트 예시

```typescript
// doc-snippet.tsx
import { MarkdownContent } from "../../features/chat/components/markdown-content";

function preprocessSnippet(snippet: string): string {
  // markdown 코드 블록 제거
  let processed = snippet.replace(/^```markdown\s*/g, '').replace(/```\s*$/g, '');
  
  // HTML 태그 정리
  processed = processed.replace(/<br\s*\/?>/gi, '\n');
  processed = processed.replace(/<[^>]+>/g, ''); // 나머지 HTML 태그 제거
  
  // 불필요한 공백 정리
  processed = processed.trim();
  
  return processed;
}

export function DocSnippet({ snippet }: { snippet: string }) {
  const processedSnippet = preprocessSnippet(snippet);
  
  return (
    <div className="retrieved-doc-snippet">
      <MarkdownContent content={processedSnippet} />
    </div>
  );
}
```

### 7.2 컴포넌트 구조

```typescript
// RetrievedDocsContent (기존 컴포넌트 개선)
function RetrievedDocsContent({ docs }: Props) {
  const [expandedImageIndex, setExpandedImageIndex] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<'image' | 'text'>('image');
  
  return (
    <div className="retrieved-docs-container">
      {docs.map((doc, index) => (
        <DocCard
          key={doc.id || index}
          doc={doc}
          index={index}
          viewMode={viewMode}
          onImageClick={() => setExpandedImageIndex(index)}
        />
      ))}
      
      {expandedImageIndex !== null && (
        <ImageLightbox
          docs={docs}
          initialIndex={expandedImageIndex}
          onClose={() => setExpandedImageIndex(null)}
        />
      )}
    </div>
  );
}

// DocCard 컴포넌트
interface DocCardProps {
  doc: RetrievedDoc;
  index: number;
  viewMode: 'image' | 'text';
  onImageClick: () => void;
}

function DocCard({ doc, index, viewMode, onImageClick }: DocCardProps) {
  const [showText, setShowText] = useState(false);
  
  return (
    <div className="retrieved-doc-item">
      <DocHeader doc={doc} index={index} />
      
      {viewMode === 'image' && doc.page_image_url ? (
        <>
          <DocImage 
            src={doc.page_image_url}
            alt={doc.title}
            onClick={onImageClick}
          />
          <DocActions
            onToggleText={() => setShowText(!showText)}
            onExpand={onImageClick}
          />
          {showText && <DocSnippet snippet={doc.snippet} />}
        </>
      ) : (
        <DocSnippet snippet={doc.snippet} />
      )}
    </div>
  );
}
```

---

## 8. 스타일 가이드

### 8.1 이미지 스타일

```css
.retrieved-doc-image {
  max-width: 100%;
  max-height: 500px;  /* 300px → 500px 증가 */
  width: auto;
  height: auto;
  border-radius: var(--radius-md);
  border: 1px solid var(--color-border);
  object-fit: contain;
  background-color: var(--color-bg-base);
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.retrieved-doc-image:hover {
  transform: scale(1.01);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.retrieved-doc-image-loading {
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: var(--color-bg-secondary);
  border-radius: var(--radius-md);
}
```

### 8.2 모달 스타일

```css
.image-lightbox-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.9);
  z-index: 2000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--spacing-xl);
}

.image-lightbox-content {
  max-width: 90vw;
  max-height: 90vh;
  position: relative;
}

.image-lightbox-image {
  max-width: 100%;
  max-height: 90vh;
  object-fit: contain;
}
```

---

## 9. 사용자 시나리오

### 시나리오 1: 기본 문서 확인

1. 사용자가 채팅에서 질문 입력
2. 오른쪽 사이드바에 검색된 문서들이 이미지로 표시됨
3. 사용자가 이미지를 스크롤하며 확인
4. 특정 문서의 이미지가 궁금하면 클릭하여 확대

### 시나리오 2: 텍스트 확인

1. 이미지가 표시된 상태에서 "텍스트 보기" 클릭
2. 텍스트 스니펫이 표시됨
3. "이미지 보기" 클릭하여 다시 이미지로 전환

### 시나리오 3: 이미지 확대

1. 문서 이미지 클릭
2. 모달이 열리며 전체 크기 이미지 표시
3. ← → 키로 이전/다음 문서 탐색
4. ESC 키 또는 X 버튼으로 닫기

---

## 10. 성능 고려사항

### 10.1 이미지 로딩 최적화

- **Lazy Loading**: 뷰포트에 들어올 때만 이미지 로드
- **이미지 캐싱**: 브라우저 캐시 활용
- **프리로딩**: 다음 문서 이미지 미리 로드 (선택사항)

### 10.2 네트워크 최적화

- **이미지 압축**: 서버에서 적절한 크기로 제공
- **WebP 포맷**: 지원하는 브라우저에서 WebP 사용
- **프로그레시브 로딩**: 저해상도 → 고해상도 순차 로드

---

## 11. 접근성 (Accessibility)

- 이미지에 적절한 `alt` 텍스트 제공
- 키보드 네비게이션 지원 (Tab, Enter, ESC)
- 스크린 리더 지원
- 고대비 모드 지원

---

## 12. 타임라인

| 단계 | 작업 | 예상 시간 | 우선순위 |
|------|------|----------|---------|
| 1-1 | 텍스트 스니펫 Markdown 렌더링 적용 | 2-3시간 | 🔥 최우선 |
| 1-2 | 텍스트 전처리 함수 구현 | 1시간 | 🔥 최우선 |
| 1-3 | Markdown 스타일 적용 및 테스트 | 1시간 | 🔥 최우선 |
| 2 | 모든 문서에 이미지 표시 | 1-2시간 | ⚡ 높음 |
| 3 | 이미지 크기 증가 및 스타일 개선 | 1시간 | ⚡ 높음 |
| 4 | 이미지 로딩 상태 및 에러 처리 | 1시간 | 중간 |
| 5 | 텍스트/이미지 토글 기능 | 1-2시간 | 중간 |
| 6 | 확대 모달 구현 | 2-3시간 | 낮음 |
| 7 | 네비게이션 및 키보드 단축키 | 1-2시간 | 낮음 |
| 8 | 테스트 및 버그 수정 | 1-2시간 | - |
| **Phase 1 총계** | | **5-7시간** | |
| **전체 총계** | | **12-18시간** | |

---

## 13. 실제 구현 시 고려사항

### 13.1 현재 코드 분석

**문제점 발견**:
- `RetrievedDocsContent` 컴포넌트에서 첫 번째 문서에만 이미지가 표시되는 이유 확인 필요
- 이미지가 없는 문서에 대한 처리 로직 확인 필요

**확인된 코드 위치**:
- `frontend/src/components/layout/index.tsx` (382-436줄): `RetrievedDocsContent` 컴포넌트
- `frontend/src/components/layout/right-sidebar.css` (296-313줄): 이미지 스타일

### 13.2 해결 방법

1. **텍스트 스니펫 Markdown 렌더링 적용** (최우선):
   ```typescript
   // 현재: 원시 텍스트로 표시
   <div className="retrieved-doc-snippet">{doc.snippet}</div>
   
   // 해결: Markdown 렌더링 적용
   import { MarkdownContent } from "../../features/chat/components/markdown-content";
   
   // 텍스트 전처리 함수
   function preprocessSnippet(snippet: string): string {
     // markdown 코드 블록 제거 (예: ```markdown 제거)
     let processed = snippet.replace(/^```markdown\s*/g, '').replace(/```\s*$/g, '');
     
     // HTML 태그 정리 (<br> → 줄바꿈)
     processed = processed.replace(/<br\s*\/?>/gi, '\n');
     
     // 불필요한 공백 정리
     processed = processed.trim();
     
     return processed;
   }
   
   // 사용
   <div className="retrieved-doc-snippet">
     <MarkdownContent content={preprocessSnippet(doc.snippet)} />
   </div>
   ```

2. **모든 문서에 이미지 표시**:
   ```typescript
   // 현재: 첫 번째 문서에만 이미지 표시되는 문제
   // 해결: 모든 문서에 대해 page_image_url이 있으면 이미지 표시
   {docs.map((doc, index) => (
     <div key={doc.id || index} className="retrieved-doc-item">
       {doc.page_image_url ? (
         <DocImage src={doc.page_image_url} ... />
       ) : (
         <DocSnippet snippet={doc.snippet} />
       )}
     </div>
   ))}
   ```

3. **이미지 크기 증가**:
   ```css
   /* 현재: max-height: 300px */
   /* 변경: max-height: 500px */
   .retrieved-doc-image {
     max-height: 500px;  /* 300px → 500px */
     max-width: 100%;
     object-fit: contain;
   }
   ```

4. **텍스트 스니펫 스타일 개선**:
   ```css
   .retrieved-doc-snippet {
     font-size: var(--font-size-sm);
     color: var(--color-text-primary);
     line-height: 1.6;
     white-space: normal;  /* pre-wrap 제거 */
     word-break: break-word;
     max-height: 400px;  /* 200px → 400px 증가 */
     overflow-y: auto;
     padding: var(--spacing-xs);
     background-color: var(--color-bg-base);
     border-radius: var(--radius-sm);
   }
   
   /* Markdown 콘텐츠 스타일 */
   .retrieved-doc-snippet .markdown-content {
     font-size: var(--font-size-sm);
   }
   
   /* 표 스크롤 처리 */
   .retrieved-doc-snippet .markdown-content table {
     display: block;
     overflow-x: auto;
     white-space: nowrap;
   }
   ```

### 13.3 기존 API 활용

- 기존 `page_image_url` API 엔드포인트 활용 (`/api/assets/docs/{doc_id}/pages/{page}`)
- 이미지가 없는 문서는 기존처럼 텍스트 스니펫 표시
- 모바일 환경에서는 터치 제스처 지원 고려
- 이미지 로딩 실패 시 graceful degradation (텍스트로 fallback)

### 13.4 성능 최적화

- **Lazy Loading**: 뷰포트에 들어올 때만 이미지 로드
- **이미지 캐싱**: 브라우저 캐시 활용
- **프리로딩**: 다음 문서 이미지 미리 로드 (선택사항)
- **이미지 압축**: 서버에서 적절한 크기로 제공

---

## 14. 체크리스트

### 개발 전
- [ ] 디자인 리뷰 및 승인
- [ ] API 응답 구조 확인
- [ ] 이미지 URL 형식 확인

### 개발 중
- [ ] 이미지 크기 조정
- [ ] 로딩 상태 표시
- [ ] 에러 처리
- [ ] 토글 기능
- [ ] 모달 구현
- [ ] 키보드 단축키

### 개발 후
- [ ] 다양한 화면 크기 테스트
- [ ] 이미지 로딩 성능 테스트
- [ ] 접근성 테스트
- [ ] 사용자 피드백 수집
