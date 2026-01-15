# Chat History 로드 기능 구현

## 개요

채팅 기록을 Elasticsearch에 저장하고, 사이드바에서 이전 채팅을 클릭하면 해당 대화를 불러와 이어갈 수 있는 기능 구현.

## 아키텍처

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │     │   Backend API   │     │  Elasticsearch  │
│   React App     │────▶│   FastAPI       │────▶│  chat_turns_*   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 데이터 흐름

```
[사용자 질문 입력]
        ↓
[Agent 응답 완료]
        ↓
[POST /api/conversations/{id}/turns] ───▶ [ES: chat_turns 인덱스에 턴 저장]
        ↓
[사이드바 목록 갱신] ◀─── [GET /api/conversations]
        ↓
[사이드바에서 채팅 클릭]
        ↓
[navigate('/?session={id}')]
        ↓
[GET /api/conversations/{id}] ───▶ [ES에서 세션 조회]
        ↓
[메시지 목록 렌더링]
        ↓
[대화 이어가기]
```

---

## Backend 구현

### 1. ES 인덱스 매핑

**파일:** `backend/llm_infrastructure/elasticsearch/mappings.py`

```python
def get_chat_turns_mapping() -> dict[str, Any]:
    return {
        "properties": {
            # Session / Turn Identity
            "session_id": {"type": "keyword", "doc_values": True},
            "turn_id": {"type": "integer"},
            "ts": {"type": "date"},

            # Conversation Content
            "user_text": {"type": "text", "analyzer": "nori"},
            "assistant_text": {"type": "text", "analyzer": "nori"},

            # Document References
            "doc_refs": {
                "type": "nested",
                "properties": {
                    "slot": {"type": "integer"},
                    "doc_id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "snippet": {"type": "text"},
                    "page": {"type": "integer"},
                    "score": {"type": "float"},
                }
            },

            # Session Metadata
            "title": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
            "summary": {"type": "text"},

            # Timestamps
            "schema_version": {"type": "keyword"},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
        }
    }
```

### 2. ChatHistoryService

**파일:** `backend/services/chat_history_service.py`

| 메서드 | 설명 |
|--------|------|
| `save_turn(turn: ChatTurn)` | 대화 턴 저장 |
| `get_session(session_id: str)` | 세션의 모든 턴 조회 |
| `list_sessions(limit, offset)` | 최근 세션 목록 조회 |
| `delete_session(session_id: str)` | 세션 삭제 |
| `get_next_turn_id(session_id: str)` | 다음 턴 ID 조회 |
| `ensure_index()` | 인덱스 생성 (없으면) |

**데이터 클래스:**

```python
@dataclass
class DocRef:
    slot: int          # 사용자에게 보이는 문서 번호 (1, 2, 3...)
    doc_id: str
    title: str
    snippet: str
    page: Optional[int] = None
    score: Optional[float] = None

@dataclass
class ChatTurn:
    session_id: str
    turn_id: int
    user_text: str
    assistant_text: str
    doc_refs: list[DocRef]
    title: Optional[str] = None    # 세션 제목 (첫 턴에만 설정)
    summary: Optional[str] = None  # 턴 요약 (향후 RLM용)
    ts: Optional[datetime] = None

@dataclass
class SessionSummary:
    session_id: str
    title: str
    preview: str       # 첫 번째 사용자 메시지
    turn_count: int
    created_at: datetime
    updated_at: datetime
```

### 3. REST API

**파일:** `backend/api/routers/conversations.py`

| Endpoint | Method | 설명 |
|----------|--------|------|
| `/api/conversations` | GET | 세션 목록 조회 |
| `/api/conversations/{id}` | GET | 세션 상세 조회 (모든 턴 포함) |
| `/api/conversations/{id}/turns` | POST | 턴 저장 |
| `/api/conversations/{id}` | DELETE | 세션 삭제 |

**Request/Response 예시:**

```typescript
// GET /api/conversations
{
  "sessions": [
    {
      "id": "abc123",
      "title": "SUPRA XP PM 절차",
      "preview": "SUPRA XP의 PM 절차에 대해 알려줘",
      "turnCount": 3,
      "createdAt": "2026-01-15T10:30:00Z",
      "updatedAt": "2026-01-15T10:35:00Z"
    }
  ],
  "total": 1
}

// GET /api/conversations/{id}
{
  "session_id": "abc123",
  "title": "SUPRA XP PM 절차",
  "turns": [
    {
      "session_id": "abc123",
      "turn_id": 1,
      "user_text": "SUPRA XP의 PM 절차에 대해 알려줘",
      "assistant_text": "SUPRA XP의 PM 절차는...",
      "doc_refs": [
        {
          "slot": 1,
          "doc_id": "doc_123",
          "title": "SUPRA XP PM Guide",
          "snippet": "PM 절차 개요...",
          "page": 15,
          "score": 0.95
        }
      ],
      "title": "SUPRA XP PM 절차",
      "ts": "2026-01-15T10:30:00Z"
    }
  ],
  "turn_count": 1
}

// POST /api/conversations/{id}/turns
{
  "user_text": "첫 번째 단계에 대해 자세히 설명해줘",
  "assistant_text": "첫 번째 단계는...",
  "doc_refs": [],
  "title": null
}
```

---

## Frontend 구현

### 1. API Client

**파일:** `frontend/src/features/chat/api.ts`

```typescript
export async function fetchSessions(limit?: number, offset?: number): Promise<SessionListResponse>;
export async function fetchSession(sessionId: string): Promise<SessionDetailResponse>;
export async function saveTurn(sessionId: string, turn: SaveTurnRequest): Promise<TurnResponse>;
export async function deleteSession(sessionId: string): Promise<void>;
```

### 2. use-chat-history Hook

**파일:** `frontend/src/features/chat/hooks/use-chat-history.ts`

localStorage 기반에서 API 기반으로 변경:

```typescript
export function useChatHistory() {
  const [history, setHistory] = useState<ChatHistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  // API에서 세션 목록 조회
  const loadSessions = useCallback(async () => { ... }, []);

  // 세션 삭제 (API 호출)
  const deleteChat = useCallback(async (id: string) => { ... }, []);

  // 세션 목록 새로고침
  const refresh = useCallback(() => { loadSessions(); }, [loadSessions]);

  return { history, isLoading, error, deleteChat, getChat, refresh };
}
```

### 3. use-chat-session Hook 확장

**파일:** `frontend/src/features/chat/hooks/use-chat-session.ts`

새로 추가된 기능:

```typescript
// 옵션에 onTurnSaved 콜백 추가
export type UseChatSessionOptions = {
  onSessionChange?: SessionChangeCallback;
  onTurnSaved?: () => void;  // 턴 저장 완료 시 호출
};

// 반환값에 loadSession 추가
return {
  // ... 기존 반환값
  isLoadingSession,  // 세션 로딩 중 여부
  loadSession,       // 세션 로드 함수
};
```

**자동 턴 저장:**
- `handleAgentResponse`에서 대화 완료 시 자동으로 `saveTurn()` 호출
- 저장 완료 시 `onTurnSaved` 콜백 호출 (히스토리 목록 갱신용)

**세션 로드:**
```typescript
const loadSession = useCallback(async (targetSessionId: string) => {
  const session = await fetchSession(targetSessionId);
  // 턴을 메시지 형태로 변환하여 state에 설정
  setMessages(convertTurnsToMessages(session.turns));
  setSessionId(targetSessionId);
  // ...
}, []);
```

### 4. Chat Page

**파일:** `frontend/src/features/chat/pages/chat-page.tsx`

URL 파라미터로 세션 ID 처리:

```typescript
export default function ChatPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const sessionParam = searchParams.get("session");

  // 세션 파라미터가 있으면 해당 세션 로드
  useEffect(() => {
    if (sessionParam && sessionParam !== sessionId) {
      loadSession(sessionParam);
      setSearchParams({}, { replace: true });  // URL 파라미터 제거
    }
  }, [sessionParam, sessionId, loadSession, setSearchParams]);

  // 턴 저장 완료 시 히스토리 목록 갱신
  const handleTurnSaved = useCallback(() => {
    refreshHistory();
  }, [refreshHistory]);

  // ...
}
```

### 5. Left Sidebar

**파일:** `frontend/src/components/layout/left-sidebar.tsx`

클릭 핸들러 수정:

```typescript
const handleHistoryClick = (id: string) => {
  // 세션 파라미터와 함께 채팅 페이지로 이동
  navigate(`/?session=${id}`);
  onClose();
};
```

---

## 인덱스 네이밍 컨벤션

```
Index:  chat_turns_{env}_v{version}   (예: chat_turns_dev_v1)
Alias:  chat_turns_{env}_current      (예: chat_turns_dev_current)
```

- 서비스는 항상 alias를 통해 접근
- 스키마 변경 시 새 버전 인덱스 생성 후 alias 전환

---

## 수정 파일 목록

### Backend
| 파일 | 변경 내용 |
|------|----------|
| `backend/llm_infrastructure/elasticsearch/mappings.py` | `get_chat_turns_mapping()` 추가 |
| `backend/services/chat_history_service.py` | 새 파일 - ES 기반 서비스 |
| `backend/api/routers/conversations.py` | 새 파일 - REST API |
| `backend/api/main.py` | conversations 라우터 등록 |

### Frontend
| 파일 | 변경 내용 |
|------|----------|
| `frontend/src/lib/api-client.ts` | `delete` 메서드 추가 |
| `frontend/src/features/chat/types.ts` | 대화 API 타입 추가 |
| `frontend/src/features/chat/api.ts` | 대화 API 함수로 교체 |
| `frontend/src/features/chat/hooks/use-chat-history.ts` | API 기반으로 변경 |
| `frontend/src/features/chat/hooks/use-chat-session.ts` | `saveTurn`, `loadSession` 추가 |
| `frontend/src/features/chat/context/chat-history-context.tsx` | 인터페이스 업데이트 |
| `frontend/src/features/chat/pages/chat-page.tsx` | URL 세션 파라미터 처리 |
| `frontend/src/components/layout/left-sidebar.tsx` | 클릭 핸들러 수정 |

---

## 향후 확장

1. **RLM 스타일 히스토리 선별**: 가벼운 LLM으로 관련 턴만 선택하여 컨텍스트에 포함
2. **"이전 1번 문서" 참조 기능**: doc_refs의 slot을 활용한 이전 대화 문서 참조
3. **턴별 요약 자동 생성**: summary 필드 활용
4. **검색 기능**: user_text, assistant_text에 nori 분석기 적용됨
