# UI 개선 개발 계획

**작성일**: 2026-01-14
**대상 항목**: 글로벌 검색, 우측 사이드바 접기, 테마 전환 애니메이션, 빈 상태 개선

---

## 1. 글로벌 검색 (Cmd/Ctrl + K)

### 1.1 개요
- 어디서든 `Cmd+K` (Mac) / `Ctrl+K` (Windows)로 검색 모달 열기
- 채팅 히스토리, 페이지 네비게이션 검색 지원

### 1.2 구현 파일

| 파일 | 작업 내용 |
|------|----------|
| `frontend/src/components/global-search/` | 새 디렉토리 생성 |
| `global-search/index.tsx` | 검색 모달 컴포넌트 |
| `global-search/global-search.css` | 스타일 |
| `global-search/use-global-search.ts` | 단축키 훅, 검색 로직 |
| `frontend/src/app/providers.tsx` | GlobalSearchProvider 추가 |

### 1.3 상세 구현

#### 1.3.1 GlobalSearchProvider (Context)
```tsx
interface GlobalSearchContextValue {
  isOpen: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
}
```

#### 1.3.2 useGlobalSearch 훅
```tsx
// 단축키 감지
useEffect(() => {
  const handleKeyDown = (e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      toggle();
    }
    if (e.key === 'Escape' && isOpen) {
      close();
    }
  };
  window.addEventListener('keydown', handleKeyDown);
  return () => window.removeEventListener('keydown', handleKeyDown);
}, [isOpen, toggle, close]);
```

#### 1.3.3 검색 모달 UI
```
┌─────────────────────────────────────────────────────┐
│  🔍 검색...                                    [ESC] │
├─────────────────────────────────────────────────────┤
│  페이지                                              │
│  ├─ 💬 Chat                              Enter →    │
│  ├─ 🔍 Search                            Enter →    │
│  ├─ 🧪 Retrieval Test                    Enter →    │
│  └─ 📄 Parsing                           Enter →    │
├─────────────────────────────────────────────────────┤
│  최근 대화                                           │
│  ├─ "에이전트 설정 방법"                  2시간 전   │
│  ├─ "문서 인덱싱 오류"                    어제       │
│  └─ "검색 결과 개선"                      2일 전     │
└─────────────────────────────────────────────────────┘
```

#### 1.3.4 검색 데이터 소스
- **페이지 목록**: 하드코딩된 라우트 정보
- **채팅 히스토리**: `useChatHistoryContext`에서 가져옴

### 1.4 작업 순서
1. `GlobalSearchContext` 생성
2. `useGlobalSearch` 훅 구현 (단축키 감지)
3. 검색 모달 UI 컴포넌트 구현
4. 검색 로직 구현 (필터링)
5. `providers.tsx`에 Provider 추가
6. 키보드 네비게이션 (화살표 키로 항목 선택)

---

## 2. 우측 사이드바 접기/펼치기 버튼

### 2.1 개요
- 우측 사이드바에 접기 버튼 추가
- 사용자가 수동으로 사이드바를 숨길 수 있음
- 닫힌 상태에서는 열기 버튼 표시

### 2.2 구현 파일

| 파일 | 작업 내용 |
|------|----------|
| `frontend/src/components/layout/index.tsx` | 상태 관리, 토글 로직 |
| `frontend/src/components/layout/right-sidebar.tsx` | 접기 버튼 추가 |
| `frontend/src/components/layout/layout.css` | 열기 버튼 스타일 |
| `frontend/src/components/layout/right-sidebar.css` | 접힌 상태 스타일 |

### 2.3 상세 구현

#### 2.3.1 Layout 상태 추가
```tsx
const [isRightSidebarCollapsed, setIsRightSidebarCollapsed] = useState(false);

const handleToggleRightSidebar = useCallback(() => {
  setIsRightSidebarCollapsed(prev => !prev);
}, []);
```

#### 2.3.2 RightSidebar props 추가
```tsx
interface RightSidebarProps {
  isOpen: boolean;
  isCollapsed: boolean;  // 추가
  onClose: () => void;
  onToggleCollapse: () => void;  // 추가
  title?: string;
  subtitle?: string;
  children?: React.ReactNode;
}
```

#### 2.3.3 UI 변경
```
열린 상태:
┌──────────────────────────────────────┐
│ [<] 실행 로그              3개 항목  │  ← 접기 버튼 추가
├──────────────────────────────────────┤
│  로그 내용...                        │
└──────────────────────────────────────┘

닫힌 상태:
┌────┐
│ [>]│  ← 열기 버튼 (메인 콘텐츠 오른쪽에 표시)
└────┘
```

#### 2.3.4 CSS 추가
```css
/* 우측 사이드바 열기 버튼 (닫혔을 때) */
.right-sidebar-toggle {
  position: fixed;
  right: var(--spacing-md);
  top: 50%;
  transform: translateY(-50%);
  width: 32px;
  height: 48px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background-color: var(--color-bg-card);
  cursor: pointer;
  z-index: 100;
}
```

### 2.4 작업 순서
1. Layout에 `isRightSidebarCollapsed` 상태 추가
2. RightSidebar에 접기 버튼 추가
3. 닫힌 상태에서 열기 버튼 표시
4. CSS 스타일 추가
5. 트랜지션 애니메이션 적용

---

## 3. 테마 전환 애니메이션

### 3.1 개요
- 다크모드 ↔ 라이트모드 전환 시 부드러운 페이드 효과
- 색상, 배경, 테두리 등이 0.3초에 걸쳐 전환

### 3.2 구현 파일

| 파일 | 작업 내용 |
|------|----------|
| `frontend/src/styles.css` | 전역 트랜지션 추가 |
| `frontend/src/components/theme-provider.tsx` | 전환 중 클래스 추가 (선택) |

### 3.3 상세 구현

#### 3.3.1 CSS 트랜지션 추가
```css
/* styles.css - :root 또는 body에 추가 */
:root {
  /* 테마 전환 트랜지션 */
  --theme-transition-duration: 0.3s;
  --theme-transition-timing: ease;
}

/* 모든 요소에 색상 트랜지션 적용 */
*,
*::before,
*::after {
  transition:
    background-color var(--theme-transition-duration) var(--theme-transition-timing),
    border-color var(--theme-transition-duration) var(--theme-transition-timing),
    color var(--theme-transition-duration) var(--theme-transition-timing),
    box-shadow var(--theme-transition-duration) var(--theme-transition-timing);
}

/* 트랜지션 제외 요소 (성능 최적화) */
.no-theme-transition,
.no-theme-transition *,
input,
textarea,
select {
  transition: none !important;
}
```

#### 3.3.2 주의사항
- 모든 요소에 트랜지션을 적용하면 성능에 영향을 줄 수 있음
- 필요한 경우 주요 컨테이너에만 적용하는 방식으로 최적화

#### 3.3.3 대안: 주요 요소만 적용
```css
/* 주요 컨테이너에만 적용 */
body,
.gpt-layout,
.left-sidebar,
.right-sidebar,
.main-content,
.chat-container {
  transition:
    background-color 0.3s ease,
    border-color 0.3s ease;
}
```

### 3.4 작업 순서
1. CSS 변수 추가 (duration, timing)
2. 주요 컨테이너에 트랜지션 적용
3. 테스트 및 성능 확인
4. 필요시 범위 조정

---

## 4. 빈 상태 개선

### 4.1 개요
- 채팅 히스토리 없음, 로그 없음, 검색 결과 없음 등의 빈 상태 UI 개선
- 아이콘, 설명 텍스트, 액션 버튼 추가

### 4.2 구현 파일

| 파일 | 작업 내용 |
|------|----------|
| `frontend/src/components/empty-state/` | 새 디렉토리 생성 |
| `empty-state/index.tsx` | 재사용 가능한 EmptyState 컴포넌트 |
| `empty-state/empty-state.css` | 스타일 |
| `frontend/src/components/layout/left-sidebar.tsx` | EmptyState 적용 |
| `frontend/src/components/layout/index.tsx` | 로그 빈 상태 개선 |

### 4.3 상세 구현

#### 4.3.1 EmptyState 컴포넌트
```tsx
interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
  return (
    <div className="empty-state">
      {icon && <div className="empty-state-icon">{icon}</div>}
      <h3 className="empty-state-title">{title}</h3>
      {description && <p className="empty-state-description">{description}</p>}
      {action && (
        <button className="empty-state-action" onClick={action.onClick}>
          {action.label}
        </button>
      )}
    </div>
  );
}
```

#### 4.3.2 적용 위치 및 내용

| 위치 | 현재 | 개선 후 |
|------|------|---------|
| 채팅 히스토리 없음 | "No recent chats" | 💬 아이콘 + "아직 대화가 없습니다" + [새 대화 시작] 버튼 |
| 실행 로그 없음 | "로그가 없습니다" | 📋 아이콘 + "아직 로그가 없습니다" + "대화를 시작하면 로그가 표시됩니다" |
| 검색 결과 없음 | "검색 결과가 없습니다" | 🔍 아이콘 + "검색 결과가 없습니다" + "다른 검색어를 시도해 보세요" |

#### 4.3.3 CSS 스타일
```css
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: var(--spacing-xl);
  text-align: center;
  height: 100%;
  min-height: 200px;
}

.empty-state-icon {
  font-size: 48px;
  margin-bottom: var(--spacing-lg);
  opacity: 0.5;
  color: var(--color-text-secondary);
}

.empty-state-title {
  font-size: var(--font-size-base);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
  margin: 0 0 var(--spacing-sm) 0;
}

.empty-state-description {
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
  margin: 0 0 var(--spacing-lg) 0;
  max-width: 240px;
}

.empty-state-action {
  padding: var(--spacing-sm) var(--spacing-lg);
  border: 1px solid var(--color-border-light);
  border-radius: var(--radius-lg);
  background: transparent;
  color: var(--color-text-primary);
  font-size: var(--font-size-sm);
  cursor: pointer;
  transition: all 0.2s ease;
}

.empty-state-action:hover {
  background-color: var(--color-action-bg);
  border-color: var(--color-text-secondary);
}
```

### 4.4 작업 순서
1. EmptyState 컴포넌트 생성
2. CSS 스타일 작성
3. left-sidebar.tsx 채팅 히스토리 빈 상태 적용
4. layout/index.tsx 로그 빈 상태 적용
5. 기타 빈 상태 위치에 적용

---

## 5. 전체 작업 우선순위

| 순서 | 항목 | 예상 작업량 | 이유 |
|------|------|------------|------|
| 1 | 빈 상태 개선 | 작음 | 간단하고 즉시 사용자 경험 개선 |
| 2 | 테마 전환 애니메이션 | 작음 | CSS만 수정, 빠르게 적용 가능 |
| 3 | 우측 사이드바 접기 | 중간 | 상태 관리 및 UI 변경 필요 |
| 4 | 글로벌 검색 | 큼 | 새 컴포넌트, Context, 검색 로직 필요 |

---

## 6. 예상 파일 구조

```
frontend/src/
├── components/
│   ├── empty-state/           # 신규
│   │   ├── index.tsx
│   │   └── empty-state.css
│   ├── global-search/         # 신규
│   │   ├── index.tsx
│   │   ├── global-search.css
│   │   └── use-global-search.ts
│   └── layout/
│       ├── index.tsx          # 수정 (우측 사이드바 상태)
│       ├── right-sidebar.tsx  # 수정 (접기 버튼)
│       ├── left-sidebar.tsx   # 수정 (빈 상태)
│       └── layout.css         # 수정 (열기 버튼 스타일)
├── styles.css                 # 수정 (테마 트랜지션)
└── app/
    └── providers.tsx          # 수정 (GlobalSearchProvider)
```

---

## 7. 체크리스트

### 빈 상태 개선
- [ ] EmptyState 컴포넌트 생성
- [ ] CSS 스타일 작성
- [ ] 채팅 히스토리 빈 상태 적용
- [ ] 로그 빈 상태 적용
- [ ] 테스트

### 테마 전환 애니메이션
- [ ] CSS 변수 추가
- [ ] 주요 컨테이너에 트랜지션 적용
- [ ] 테스트 (다크 ↔ 라이트)
- [ ] 성능 확인

### 우측 사이드바 접기
- [ ] Layout 상태 추가
- [ ] RightSidebar 접기 버튼 추가
- [ ] 열기 버튼 구현
- [ ] CSS 스타일 추가
- [ ] 트랜지션 애니메이션

### 글로벌 검색
- [ ] GlobalSearchContext 생성
- [ ] useGlobalSearch 훅 구현
- [ ] 검색 모달 UI 구현
- [ ] 검색 로직 구현
- [ ] 키보드 네비게이션
- [ ] Provider 추가
- [ ] 테스트
