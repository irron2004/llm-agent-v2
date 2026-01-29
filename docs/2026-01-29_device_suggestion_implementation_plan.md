# Device Suggestion Flow 구현 계획

## 개요
스펙 문서: `docs/2026-01-29_device_suggestion_flow_spec.md`

---

## Phase 1: Backend API 스키마 확장

### Task 1.1: AgentRequest 확장
**파일**: `backend/api/routers/agent.py`

```python
class AgentRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    filter_devices: Optional[List[str]] = None  # 장비 필터
    skip_auto_parse: bool = False               # auto_parse 건너뛰기
```

### Task 1.2: AgentResponse 확장
**파일**: `backend/api/routers/agent.py`

```python
class SuggestedDevice(BaseModel):
    name: str
    count: int

class AgentResponse(BaseModel):
    # 기존 필드...
    suggested_devices: Optional[List[SuggestedDevice]] = None
```

---

## Phase 2: Backend 집계 로직

### Task 2.1: device_name 집계 함수 추가
**파일**: `backend/llm_infrastructure/llm/langgraph_agent.py`

```python
def _collect_suggested_devices(docs: List[RetrievalResult]) -> List[Dict[str, Any]]:
    """검색 결과에서 device_name 집계 (count 내림차순)."""
    from collections import Counter

    EXCLUDE_NAMES = {"", "ALL", "etc", "ETC", "all"}

    device_counts = Counter()
    for doc in docs:
        device_name = doc.metadata.get("device_name", "")
        if device_name and device_name.strip() not in EXCLUDE_NAMES:
            device_counts[device_name.strip()] += 1

    return [
        {"name": name, "count": count}
        for name, count in device_counts.most_common()
    ]
```

### Task 2.2: answer_node에서 suggested_devices 생성
**파일**: `backend/llm_infrastructure/llm/langgraph_agent.py`

- `answer_node` 반환값에 `suggested_devices` 추가
- 조건: `auto_parsed_device`가 없을 때만 생성

```python
def answer_node(state, ...):
    # 기존 로직...

    # suggested_devices 집계 (장비 미지정 시에만)
    suggested_devices = None
    if not state.get("auto_parsed_device"):
        docs = state.get("docs", [])
        if docs:
            suggested_devices = _collect_suggested_devices(docs)

    return {
        "answer": answer,
        "suggested_devices": suggested_devices,
        # 기존 필드...
    }
```

---

## Phase 3: Backend 필터 적용

### Task 3.1: filter_devices 처리 로직
**파일**: `backend/api/routers/agent.py`

```python
@router.post("/chat")
async def chat(request: AgentRequest, ...):
    state_overrides = {}

    # filter_devices가 있으면 state에 전달
    if request.filter_devices:
        state_overrides["filter_devices"] = request.filter_devices
        state_overrides["skip_auto_parse"] = True

    if request.skip_auto_parse:
        state_overrides["skip_auto_parse"] = True
```

### Task 3.2: auto_parse 조건부 실행
**파일**: `backend/services/agents/langgraph_rag_agent.py`

- `skip_auto_parse=True`면 auto_parse 노드 건너뛰기
- 또는 auto_parse_node 내부에서 조건 체크

```python
def auto_parse_node(state, ...):
    if state.get("skip_auto_parse"):
        return {}  # 건너뛰기
    # 기존 로직...
```

### Task 3.3: retrieve_node에서 filter_devices 적용
**파일**: `backend/llm_infrastructure/llm/langgraph_agent.py`

```python
def retrieve_node(state, ...):
    filter_devices = state.get("filter_devices")

    # retriever에 filter 전달
    if filter_devices:
        # ES query에 device_name 필터 추가
        docs = retriever.retrieve(queries, filter_devices=filter_devices)
    else:
        docs = retriever.retrieve(queries)
```

### Task 3.4: SearchService/Retriever 필터 지원
**파일**: `backend/services/search_service.py`

- `search()` 메서드에 `filter_devices` 파라미터 추가
- ES query에 `terms` 필터 적용

```python
def search(self, queries, filter_devices=None, ...):
    query_body = {...}

    if filter_devices:
        query_body["query"]["bool"]["filter"].append({
            "terms": {"metadata.device_name": filter_devices}
        })
```

---

## Phase 4: Backend 응답 포함

### Task 4.1: SSE 응답에 suggested_devices 포함
**파일**: `backend/api/routers/agent.py`

```python
async def _stream_response(...):
    # 최종 응답에 suggested_devices 포함
    final_data = {
        "answer": result.get("answer"),
        "suggested_devices": result.get("suggested_devices"),
        # ...
    }
```

---

## Phase 5: Frontend 렌더링

### Task 5.1: 타입 정의
**파일**: `examples/frontend/src/features/chat/types.ts`

```typescript
interface SuggestedDevice {
  name: string;
  count: number;
}

interface Message {
  // 기존 필드...
  suggestedDevices?: SuggestedDevice[];
}
```

### Task 5.2: 추천 장비 UI 컴포넌트
**파일**: `examples/frontend/src/features/chat/components/device-suggestions.tsx`

```typescript
function DeviceSuggestions({ devices, onSelect }) {
  if (!devices?.length) return null;

  return (
    <div className="device-suggestions">
      <p>검색된 문서의 장비 목록:</p>
      {devices.map((d, i) => (
        <button key={d.name} onClick={() => onSelect(i + 1, d.name)}>
          {i + 1}. {d.name} ({d.count}건)
        </button>
      ))}
    </div>
  );
}
```

### Task 5.3: MessageItem에 추천 장비 표시
**파일**: `examples/frontend/src/features/chat/components/message-item.tsx`

- assistant 메시지 하단에 `DeviceSuggestions` 렌더링

---

## Phase 6: Frontend 숫자 선택 처리

### Task 6.1: 상태 추가
**파일**: `examples/frontend/src/features/chat/hooks/use-chat-session.ts`

```typescript
const [lastQuery, setLastQuery] = useState<string>("");
const [lastSuggestedDevices, setLastSuggestedDevices] = useState<SuggestedDevice[]>([]);
```

### Task 6.2: 숫자 입력 감지 및 재요청
**파일**: `examples/frontend/src/features/chat/hooks/use-chat-session.ts`

```typescript
const send = async ({ text }) => {
  // 숫자만 입력인지 확인
  const numMatch = text.match(/^\d+$/);

  if (numMatch && lastSuggestedDevices.length > 0) {
    const index = parseInt(text) - 1;
    if (index >= 0 && index < lastSuggestedDevices.length) {
      // 장비 선택 재요청
      const selectedDevice = lastSuggestedDevices[index].name;
      await sendChatMessage({
        message: lastQuery,  // 직전 질문 재사용
        filter_devices: [selectedDevice],
        skip_auto_parse: true,
      });
      return;
    }
  }

  // 일반 질문 처리
  setLastQuery(text);
  // ...
};
```

### Task 6.3: API 호출 함수 수정
**파일**: `examples/frontend/src/features/chat/api.ts`

```typescript
export async function sendChatMessage(
  payload: {
    message: string;
    sessionId?: string;
    filter_devices?: string[];
    skip_auto_parse?: boolean;
  },
  handlers: {...}
) {
  const body = {
    message: payload.message,
    session_id: payload.sessionId,
    filter_devices: payload.filter_devices,
    skip_auto_parse: payload.skip_auto_parse,
  };
  // ...
}
```

---

## Phase 7: 테스트

### Task 7.1: Backend 단위 테스트
**파일**: `backend/tests/test_device_suggestion.py`

- `_collect_suggested_devices` 함수 테스트
- filter_devices 적용 테스트
- suggested_devices 응답 포함 테스트

### Task 7.2: 통합 테스트
- 장비 미지정 질문 → 추천 리스트 표시
- 장비 지정 질문 → 추천 리스트 미표시
- 숫자 선택 → 재검색 수행
- 범위 초과 숫자 → 일반 질문 처리

---

## 구현 순서 (권장)

| 순서 | Task | 설명 | 의존성 |
|------|------|------|--------|
| 1 | 1.1, 1.2 | API 스키마 확장 | - |
| 2 | 2.1, 2.2 | 집계 로직 추가 | 1 |
| 3 | 4.1 | 응답에 포함 | 2 |
| 4 | 5.1, 5.2, 5.3 | FE 렌더링 | 3 |
| 5 | 3.1~3.4 | 필터 적용 로직 | 1 |
| 6 | 6.1~6.3 | FE 숫자 선택 | 4, 5 |
| 7 | 7.1, 7.2 | 테스트 | 6 |

---

## 예상 작업량
- Backend: ~4-5개 파일 수정
- Frontend: ~4-5개 파일 수정
- 테스트: 1개 파일 추가

---

## 롤아웃 전략
1. **Stage 1**: Backend만 배포 (suggested_devices 응답 추가)
2. **Stage 2**: Frontend 렌더링 배포 (표시만)
3. **Stage 3**: 숫자 선택 기능 활성화
