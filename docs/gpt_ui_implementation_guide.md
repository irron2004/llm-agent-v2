# GPT UI 구현 가이드

## 1. 컴포넌트 구조 설계

### 새로운 레이아웃 컴포넌트 구조

```
src/
├── components/
│   ├── layout/
│   │   ├── index.tsx              # 메인 레이아웃 (3단 컬럼)
│   │   ├── left-sidebar.tsx       # 왼쪽 사이드바
│   │   ├── right-sidebar.tsx      # 오른쪽 사이드바 (선택적)
│   │   └── main-content.tsx       # 중앙 콘텐츠 영역
│   └── ...
```

---

## 2. 구현 예시 코드

### 2.1 메인 레이아웃 컴포넌트

```typescript
// src/components/layout/index.tsx
import { useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import LeftSidebar from "./left-sidebar";
import RightSidebar from "./right-sidebar";
import MainContent from "./main-content";
import "./layout.css";

export default function Layout() {
  const location = useLocation();
  const [isRightSidebarOpen, setIsRightSidebarOpen] = useState(false);

  return (
    <div className="gpt-layout">
      <LeftSidebar />
      <MainContent>
        <Outlet />
      </MainContent>
      {isRightSidebarOpen && (
        <RightSidebar onClose={() => setIsRightSidebarOpen(false)} />
      )}
    </div>
  );
}
```

### 2.2 왼쪽 사이드바 컴포넌트

```typescript
// src/components/layout/left-sidebar.tsx
import { useNavigate, useLocation } from "react-router-dom";
import { 
  MessageOutlined, 
  SearchOutlined, 
  FileTextOutlined, 
  ExperimentOutlined,
  EditOutlined,
  SearchOutlined as SearchIcon
} from "@ant-design/icons";
import { useState } from "react";
import "./left-sidebar.css";

interface MenuItem {
  key: string;
  icon: React.ReactNode;
  label: string;
}

export default function LeftSidebar() {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchQuery, setSearchQuery] = useState("");

  const menuItems: MenuItem[] = [
    {
      key: "/",
      icon: <MessageOutlined />,
      label: "Chat",
    },
    {
      key: "/search",
      icon: <SearchOutlined />,
      label: "Search",
    },
    {
      key: "/retrieval-test",
      icon: <ExperimentOutlined />,
      label: "Retrieval Test",
    },
    {
      key: "/parsing",
      icon: <FileTextOutlined />,
      label: "Parsing",
    },
  ];

  const handleNewChat = () => {
    navigate("/");
    // TODO: 새 채팅 세션 시작
  };

  const handleMenuClick = (key: string) => {
    navigate(key);
  };

  return (
    <aside className="left-sidebar">
      {/* 상단: 로고 + 새 채팅 */}
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <div className="logo-icon">PE</div>
          <span className="logo-text">PE Agent</span>
        </div>
        <button className="new-chat-button" onClick={handleNewChat}>
          <EditOutlined />
          <span>새 채팅</span>
        </button>
      </div>

      {/* 채팅 검색 */}
      <div className="sidebar-search">
        <SearchIcon className="search-icon" />
        <input
          type="text"
          placeholder="채팅 검색"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="search-input"
        />
      </div>

      {/* 네비게이션 메뉴 */}
      <nav className="sidebar-nav">
        {menuItems.map((item) => {
          const isActive = location.pathname === item.key;
          return (
            <button
              key={item.key}
              className={`nav-item ${isActive ? "active" : ""}`}
              onClick={() => handleMenuClick(item.key)}
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* 채팅 히스토리 (선택적) */}
      <div className="sidebar-history">
        <div className="history-header">최근 대화</div>
        <div className="history-list">
          {/* TODO: 채팅 히스토리 목록 렌더링 */}
        </div>
      </div>
    </aside>
  );
}
```

### 2.3 중앙 콘텐츠 영역 컴포넌트

```typescript
// src/components/layout/main-content.tsx
import { PropsWithChildren } from "react";
import "./main-content.css";

export default function MainContent({ children }: PropsWithChildren) {
  return (
    <main className="main-content">
      <div className="main-content-inner">
        {children}
      </div>
    </main>
  );
}
```

### 2.4 오른쪽 사이드바 컴포넌트 (선택적)

```typescript
// src/components/layout/right-sidebar.tsx
import { CloseOutlined } from "@ant-design/icons";
import "./right-sidebar.css";

interface RightSidebarProps {
  title?: string;
  subtitle?: string;
  onClose: () => void;
  children?: React.ReactNode;
}

export default function RightSidebar({ 
  title, 
  subtitle, 
  onClose,
  children 
}: RightSidebarProps) {
  return (
    <aside className="right-sidebar">
      <div className="right-sidebar-header">
        <div className="right-sidebar-title-section">
          {title && <h3 className="right-sidebar-title">{title}</h3>}
          {subtitle && <span className="right-sidebar-subtitle">{subtitle}</span>}
        </div>
        <button className="right-sidebar-close" onClick={onClose}>
          <CloseOutlined />
        </button>
      </div>
      <div className="right-sidebar-content">
        {children}
      </div>
    </aside>
  );
}
```

---

## 3. CSS 스타일 가이드

### 3.1 메인 레이아웃 스타일

```css
/* src/components/layout/layout.css */
.gpt-layout {
  display: flex;
  height: 100vh;
  width: 100%;
  background-color: var(--color-bg-canvas);
  overflow: hidden;
}

/* 다크 테마 기본값 */
.gpt-layout {
  --sidebar-width: 288px;
  --right-sidebar-width: 320px;
  --main-content-max-width: 1200px;
}
```

### 3.2 왼쪽 사이드바 스타일

```css
/* src/components/layout/left-sidebar.css */
.left-sidebar {
  width: var(--sidebar-width);
  height: 100%;
  background-color: var(--color-bg-card);
  border-right: 1px solid var(--color-border);
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  flex-shrink: 0;
}

.sidebar-header {
  padding: var(--spacing-lg);
  border-bottom: 1px solid var(--color-border);
}

.sidebar-logo {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  margin-bottom: var(--spacing-md);
}

.logo-icon {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background-color: var(--color-accent-primary);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
}

.logo-text {
  font-size: var(--font-size-base);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
}

.new-chat-button {
  width: 100%;
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-sm) var(--spacing-md);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background-color: var(--color-bg-base);
  color: var(--color-text-primary);
  font-size: var(--font-size-sm);
  cursor: pointer;
  transition: all 0.2s ease;
}

.new-chat-button:hover {
  background-color: var(--color-action-bg-hover);
}

.sidebar-search {
  padding: var(--spacing-md) var(--spacing-lg);
  border-bottom: 1px solid var(--color-border);
  position: relative;
}

.search-input {
  width: 100%;
  padding: var(--spacing-sm) var(--spacing-md) var(--spacing-sm) 32px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background-color: var(--color-bg-base);
  color: var(--color-text-primary);
  font-size: var(--font-size-sm);
}

.search-input:focus {
  outline: none;
  border-color: var(--color-accent-primary);
}

.search-icon {
  position: absolute;
  left: calc(var(--spacing-lg) + var(--spacing-md));
  top: 50%;
  transform: translateY(-50%);
  color: var(--color-text-secondary);
  font-size: var(--font-size-sm);
}

.sidebar-nav {
  padding: var(--spacing-md) var(--spacing-lg);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.nav-item {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-sm) var(--spacing-md);
  border: none;
  border-radius: var(--radius-md);
  background-color: transparent;
  color: var(--color-text-secondary);
  font-size: var(--font-size-sm);
  text-align: left;
  cursor: pointer;
  transition: all 0.2s ease;
}

.nav-item:hover {
  background-color: var(--color-action-bg);
  color: var(--color-text-primary);
}

.nav-item.active {
  background-color: var(--color-action-bg-hover);
  color: var(--color-text-primary);
  border-left: 3px solid var(--color-accent-primary);
  padding-left: calc(var(--spacing-md) - 3px);
}

.sidebar-history {
  flex: 1;
  padding: var(--spacing-md) var(--spacing-lg);
  border-top: 1px solid var(--color-border);
  overflow-y: auto;
}

.history-header {
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: var(--spacing-md);
}

.history-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}
```

### 3.3 중앙 콘텐츠 영역 스타일

```css
/* src/components/layout/main-content.css */
.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  min-width: 0; /* flexbox에서 overflow 처리 */
}

.main-content-inner {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  padding: var(--spacing-xl);
  max-width: var(--main-content-max-width);
  width: 100%;
  margin: 0 auto;
}
```

### 3.4 오른쪽 사이드바 스타일

```css
/* src/components/layout/right-sidebar.css */
.right-sidebar {
  width: var(--right-sidebar-width);
  height: 100%;
  background-color: var(--color-bg-card);
  border-left: 1px solid var(--color-border);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}

.right-sidebar-header {
  padding: var(--spacing-lg);
  border-bottom: 1px solid var(--color-border);
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--spacing-md);
}

.right-sidebar-title-section {
  flex: 1;
  min-width: 0;
}

.right-sidebar-title {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  color: var(--color-text-primary);
  margin: 0 0 var(--spacing-xs) 0;
}

.right-sidebar-subtitle {
  font-size: var(--font-size-xs);
  color: var(--color-text-secondary);
}

.right-sidebar-close {
  padding: var(--spacing-xs);
  border: none;
  background: transparent;
  color: var(--color-text-secondary);
  cursor: pointer;
  border-radius: var(--radius-sm);
  transition: all 0.2s ease;
}

.right-sidebar-close:hover {
  background-color: var(--color-action-bg);
  color: var(--color-text-primary);
}

.right-sidebar-content {
  flex: 1;
  padding: var(--spacing-lg);
  overflow-y: auto;
}
```

---

## 4. 다크 테마 색상 팔레트 업데이트

### 4.1 styles.css에 GPT 스타일 색상 추가

```css
/* 기존 다크 테마를 GPT 스타일로 업데이트 */
.dark {
  --color-bg-base: #161618;        /* 메인 배경 */
  --color-bg-canvas: #000000;      /* 전체 배경 */
  --color-bg-card: #1f1f23;        /* 카드/사이드바 */
  --color-bg-elevated: #27272a;    /* 상위 레이어 */
  --color-bg-secondary: #2d2d30;   /* 보조 배경 */
  
  --color-text-primary: #f6f6f7;   /* 주요 텍스트 */
  --color-text-secondary: #b2b5b7; /* 보조 텍스트 */
  --color-text-disabled: #75787a;  /* 비활성 텍스트 */
  
  --color-border: #2d2d30;         /* 경계선 */
  --color-border-light: #3f3f46;   /* 밝은 경계선 */
  
  --color-action-bg: rgba(255, 255, 255, 0.1);
  --color-action-bg-hover: rgba(255, 255, 255, 0.16);
}
```

---

## 5. 채팅 히스토리 기능 구현

### 5.1 채팅 히스토리 훅

```typescript
// src/features/chat/hooks/use-chat-history.ts
import { useState, useEffect } from "react";

export interface ChatHistoryItem {
  id: string;
  title: string;
  preview: string;
  createdAt: string;
  messageCount: number;
}

const STORAGE_KEY = "pe-agent-chat-history";
const MAX_HISTORY_ITEMS = 50;

export function useChatHistory() {
  const [history, setHistory] = useState<ChatHistoryItem[]>([]);

  useEffect(() => {
    // 로컬 스토리지에서 히스토리 로드
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        setHistory(JSON.parse(stored));
      } catch {
        setHistory([]);
      }
    }
  }, []);

  const saveHistory = (newHistory: ChatHistoryItem[]) => {
    // 최대 개수 제한
    const limited = newHistory.slice(0, MAX_HISTORY_ITEMS);
    setHistory(limited);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(limited));
  };

  const addChat = (item: ChatHistoryItem) => {
    const newHistory = [item, ...history];
    saveHistory(newHistory);
  };

  const updateChat = (id: string, updates: Partial<ChatHistoryItem>) => {
    const newHistory = history.map((item) =>
      item.id === id ? { ...item, ...updates } : item
    );
    saveHistory(newHistory);
  };

  const deleteChat = (id: string) => {
    const newHistory = history.filter((item) => item.id !== id);
    saveHistory(newHistory);
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem(STORAGE_KEY);
  };

  return {
    history,
    addChat,
    updateChat,
    deleteChat,
    clearHistory,
  };
}
```

### 5.2 사이드바에 히스토리 통합

```typescript
// left-sidebar.tsx에 추가
import { useChatHistory } from "../../features/chat/hooks/use-chat-history";

export default function LeftSidebar() {
  const { history, deleteChat } = useChatHistory();
  // ... 기존 코드

  return (
    <aside className="left-sidebar">
      {/* ... 기존 헤더, 검색, 네비게이션 */}
      
      <div className="sidebar-history">
        <div className="history-header">최근 대화</div>
        <div className="history-list">
          {history.map((item) => (
            <div key={item.id} className="history-item">
              <div className="history-item-content">
                <div className="history-item-title">{item.title}</div>
                <div className="history-item-preview">{item.preview}</div>
              </div>
              <button
                className="history-item-delete"
                onClick={() => deleteChat(item.id)}
                title="삭제"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
```

---

## 6. 반응형 레이아웃

### 6.1 모바일 대응

```css
/* 모바일에서는 사이드바를 오버레이로 표시 */
@media (max-width: 768px) {
  .left-sidebar {
    position: fixed;
    left: 0;
    top: 0;
    z-index: 1000;
    transform: translateX(-100%);
    transition: transform 0.3s ease;
  }

  .left-sidebar.open {
    transform: translateX(0);
  }

  .main-content {
    width: 100%;
  }
}
```

---

## 7. 마이그레이션 전략

### 단계별 마이그레이션

1. **Phase 1**: 새로운 레이아웃 컴포넌트 생성 (기존 레이아웃 유지)
2. **Phase 2**: 라우터에서 새 레이아웃으로 전환 (기능 테스트)
3. **Phase 3**: 스타일 개선 및 다크 테마 조정
4. **Phase 4**: 채팅 히스토리 기능 추가
5. **Phase 5**: 오른쪽 사이드바 기능 추가 (선택적)

### 기존 코드와의 호환성

- 기존 `ChatPage`, `ParsingPage` 등은 그대로 사용 가능
- `Layout` 컴포넌트만 교체하면 됨
- CSS 변수 기반이므로 테마 시스템 유지

---

## 8. 체크리스트

### 필수 구현 항목
- [ ] 3단 컬럼 레이아웃 구조
- [ ] 왼쪽 사이드바 컴포넌트
- [ ] 네비게이션 메뉴 이동
- [ ] 다크 테마 색상 업데이트
- [ ] 반응형 레이아웃 (모바일)

### 선택 구현 항목
- [ ] 채팅 히스토리 기능
- [ ] 채팅 검색 기능
- [ ] 오른쪽 사이드바
- [ ] 사이드바 접기/펼치기
- [ ] 애니메이션 효과

---

## 9. 참고 사항

- 모든 컴포넌트는 TypeScript로 작성
- CSS 변수를 활용한 테마 시스템 유지
- 접근성 고려 (키보드 네비게이션, ARIA 레이블)
- 성능 최적화 (메모이제이션, 가상 스크롤 등)
