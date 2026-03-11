# 필터 바 UI 리디자인 — 스크린샷 스타일 반영

## 요구사항

고객사 myBOT 스크린샷과 동일한 하단 필터 바 UI로 변경:
- Doc / Model / Equip 드롭다운을 입력창 위에 항상 노출
- Doc 드롭다운에 **작업조회** (SOP+setup 묶음)와 **이슈조회** (myservice+gcb+pems+ts 묶음) 프리셋 추가
- Model 드롭다운에 검색 기능 추가
- 필터가 비어있고 쿼리에서 모델명 감지 시 → 기존 guided panel로 확인
- 필터에서 이미 선택된 값이 있으면 guided panel 스킵

## 영향 받는 파일

### FE 변경
1. `frontend/src/features/chat/components/chat-input.tsx` — 라디오 버튼 제거, 필터 바 추가
2. `frontend/src/features/chat/pages/chat-page.tsx` — searchMode state 제거, 필터 state 추가, fetchDeviceCatalog 활용
3. `frontend/src/features/chat/hooks/use-chat-session.ts` — filter_devices/filter_doc_types를 필터 바 값에서 전달
4. `frontend/src/features/chat/types.ts` — AgentRequest에 filter_equip_ids 이미 있음, task_mode_override 제거 가능
5. `frontend/src/features/chat/components/guided-selection-panel.tsx` — 유지 (필터 미선택 시 fallback으로 사용)

### BE 변경
6. `backend/api/routers/agent.py` — 선택된 doc_types 기반으로 task_mode 자동 판별 로직
7. `backend/api/routers/devices.py` — doc_types 목록에 preset 그룹 정보 추가 (선택적)

---

## Phase 1: ChatInput 리디자인 — 필터 바 추가

### Step 1-1. ChatInput에서 라디오 버튼 제거 + 필터 바 props 추가

**파일**: `frontend/src/features/chat/components/chat-input.tsx`

**변경**:
1. 기존 `searchMode`/`onSearchModeChange` props 제거
2. 새 props 추가:
   ```typescript
   type FilterBarProps = {
     // Doc type filter
     docTypeOptions: { label: string; value: string; isPreset?: boolean }[];
     selectedDocTypes: string[];
     onDocTypesChange: (types: string[]) => void;
     // Model filter (searchable)
     modelOptions: { label: string; value: string }[];
     selectedModel: string | null;
     onModelChange: (model: string | null) => void;
     // Equip filter
     equipOptions: { label: string; value: string }[];
     selectedEquip: string | null;
     onEquipChange: (equip: string | null) => void;
   };

   type ChatInputProps = {
     onSend: (message: string) => void;
     onStop?: () => void;
     isStreaming?: boolean;
     disabled?: boolean;
     placeholder?: string;
   } & FilterBarProps;
   ```

3. 필터 바 UI 렌더링 (textarea 위):
   ```
   ┌─────────────────────────────────────────────────┐
   │ Doc: [▼ All Documents]  Model: [▼ All Models]  Equip: [▼ All Equip IDs] │
   │ [질문을 입력하세요...                              ] [전송] │
   └─────────────────────────────────────────────────┘
   ```

4. Doc 드롭다운 구조:
   ```
   ── 프리셋 ──
   작업조회 (SOP + setup)
   이슈조회 (myservice + gcb + pems + ts)
   ── 개별 선택 ──
   □ SOP
   □ setup
   □ myservice
   □ gcb
   □ pems
   □ ts (trouble-shooting)
   ```

5. Model 드롭다운: `showSearch: true` (Ant Design Select의 검색 기능)

6. 전송 조건: `!searchMode` 가드 제거 — 필터 없어도 전송 가능 (기존 guided flow가 처리)

### Step 1-2. chat-page에서 필터 state 관리

**파일**: `frontend/src/features/chat/pages/chat-page.tsx`

**변경**:
1. `searchMode` state 제거
2. 필터 state 추가:
   ```typescript
   const [selectedDocTypes, setSelectedDocTypes] = useState<string[]>([]);
   const [selectedModel, setSelectedModel] = useState<string | null>(null);
   const [selectedEquip, setSelectedEquip] = useState<string | null>(null);
   ```

3. 페이지 로드 시 `fetchDeviceCatalog()` 호출하여 옵션 목록 로드:
   - 기존: `pendingRegeneration?.reason === "missing_device_parse"` 일 때만 호출
   - 변경: 항상 호출하여 Model/Equip 옵션 목록 구성
   ```typescript
   const [catalog, setCatalog] = useState<DeviceCatalogResponse | null>(null);
   useEffect(() => {
     fetchDeviceCatalog().then(setCatalog).catch(() => {});
   }, []);
   ```

4. Doc type 옵션 목록 구성 (프리셋 포함):
   ```typescript
   const docTypeOptions = useMemo(() => {
     const presets = [
       { label: "작업조회", value: "__preset_sop", isPreset: true },
       { label: "이슈조회", value: "__preset_issue", isPreset: true },
     ];
     const individual = (catalog?.doc_types ?? []).map(dt => ({
       label: dt.name, value: dt.name,
     }));
     return [...presets, ...individual];
   }, [catalog]);
   ```

5. 프리셋 선택 시 개별 doc types로 확장:
   ```typescript
   const PRESET_SOP = ["sop", "setup"];
   const PRESET_ISSUE = ["myservice", "gcb", "pems", "ts"];

   const handleDocTypesChange = (types: string[]) => {
     const expanded = types.flatMap(t => {
       if (t === "__preset_sop") return PRESET_SOP;
       if (t === "__preset_issue") return PRESET_ISSUE;
       return [t];
     });
     setSelectedDocTypes([...new Set(expanded)]);
   };
   ```

6. Model 옵션: `catalog?.devices` 에서 생성
7. Equip 옵션: 별도 API 필요하거나, 기존 guided flow에서 제공되는 equip_id 목록 사용

8. `handleSend`에서 필터 값을 `send()`에 전달:
   ```typescript
   await send({
     text,
     overrides: {
       filterDevices: selectedModel ? [selectedModel] : undefined,
       filterDocTypes: selectedDocTypes.length > 0 ? selectedDocTypes : undefined,
     },
   });
   ```

9. ChatInput에 필터 props 전달

### Step 1-3. use-chat-session에서 필터 연동 강화

**파일**: `frontend/src/features/chat/hooks/use-chat-session.ts`

**변경**:
1. `searchMode` 관련 코드 제거 (`task_mode_override`)
2. `overrides`의 `filterDevices`/`filterDocTypes`가 있을 때:
   - `auto_parse` 비활성화 (이미 필터 지정됨)
   - `guided_confirm` 조건 변경: 필터가 이미 있으면 스킵
   ```typescript
   const hasFilters = Boolean(overrides?.filterDevices?.length || overrides?.filterDocTypes?.length);
   const autoParseEnabled = !hasFilters && (overrides?.autoParse ?? !Boolean(overrides));
   const guidedConfirmEnabled = !isResume && !hasFilters && autoParseEnabled;
   ```

3. 필터가 있을 때 payload에 filter 값 포함 (기존 로직 유지)

---

## Phase 2: Backend — doc_types 기반 task_mode 자동 판별

### Step 2-1. task_mode 자동 결정 로직

**파일**: `backend/api/routers/agent.py`

**변경**:
1. `_build_state_overrides` 또는 요청 처리 시:
   ```python
   SOP_DOC_TYPES = {"sop", "setup"}
   ISSUE_DOC_TYPES = {"myservice", "gcb", "pems", "ts"}

   def _infer_task_mode(doc_types: list[str] | None) -> str | None:
       if not doc_types:
           return None  # auto_parse에서 결정
       types_set = set(dt.lower() for dt in doc_types)
       if types_set <= SOP_DOC_TYPES:
           return "sop"
       if types_set <= ISSUE_DOC_TYPES:
           return "issue"
       return None  # 혼합 → 기본 동작
   ```

2. 요청 처리 시 `filter_doc_types`에서 task_mode 추론:
   ```python
   if not overrides.get("task_mode"):
       inferred = _infer_task_mode(req.filter_doc_types)
       if inferred:
           overrides["task_mode"] = inferred
   ```

3. `task_mode_override` 필드는 유지 (하위 호환) — 우선순위: task_mode_override > inferred

### Step 2-2. filter_doc_types → selected_doc_types 연동

**파일**: `backend/api/routers/agent.py`

**현재 동작 확인**: `filter_doc_types`가 `state["selected_doc_types"]`에 매핑되는지 확인
- 이미 `_build_state_overrides`에서 처리되고 있다면 추가 변경 불필요
- 없다면 `selected_doc_types_strict` 플래그와 함께 주입

---

## Phase 3: Guided panel과 필터 바 통합

### Step 3-1. 필터 선택 시 guided panel 스킵

**동작 규칙**:
| 필터 상태 | 쿼리 감지 | 동작 |
|-----------|-----------|------|
| Model 선택됨 | - | guided panel의 device 단계 스킵 |
| Model null | 모델명 감지 | guided panel 표시 (기존 1~N 선택) |
| Model null | 감지 없음 | 전체 모델 검색 |
| Equip 선택됨 | - | guided panel의 equip 단계 스킵 |
| Doc 선택됨 | - | auto_parse의 doc_type 감지 스킵, 선택값 사용 |

**구현**: `use-chat-session.ts`에서 `overrides`에 필터값이 있으면 `guided_confirm: false`로 설정.
백엔드에서 `filter_devices`/`filter_doc_types`가 있으면 `auto_parse_node`에서 감지 로직 스킵.

### Step 3-2. 기존 guided panel 유지 (fallback)

- 필터가 비어있을 때만 기존 guided selection panel 표시
- `pendingGuidedSelection`은 기존대로 동작
- 라디오 제거로 인해 guided panel에서 language/task 단계가 이미 제거됨 (Phase 2-4 완료)

---

## 수용 기준

| # | 기준 | 검증 |
|---|------|------|
| 1 | 하단에 Doc/Model/Equip 드롭다운 항상 표시 | 수동 테스트 |
| 2 | Doc에서 "작업조회" 클릭 → SOP+setup 자동 체크 | 수동 테스트 |
| 3 | Doc에서 "이슈조회" 클릭 → myservice+gcb+pems+ts 체크 | 수동 테스트 |
| 4 | Model 드롭다운에서 검색 가능 | 수동 테스트 |
| 5 | Model 선택 후 질문 → guided panel 안 뜸 | 수동 테스트 |
| 6 | Model 미선택 + 쿼리에 모델명 → guided panel 뜸 | 수동 테스트 |
| 7 | SOP doc types만 선택 → 작업절차 고정 양식 답변 | 품질 테스트 |
| 8 | 라디오 버튼 완전히 제거됨 | 수동 테스트 |
| 9 | Frontend 테스트 69건 통과 | `npm run test` |
| 10 | Backend 테스트 441+ 통과 | `uv run pytest` |

## 리스크

| 리스크 | 대응 |
|--------|------|
| fetchDeviceCatalog 페이지 로드 시 호출 → 느린 응답 | 로딩 상태 + 에러 핸들링, catalog 캐시 |
| Equip 목록이 많을 수 있음 | Model과 동일하게 검색 가능 드롭다운 |
| 혼합 doc_type 선택 시 task_mode 불명확 | null → 기본 동작 (issue 프롬프트) |
| 기존 테스트에서 searchMode prop 기대 | 테스트 업데이트 필요 |

## 작업 순서

1. **Phase 1** — FE 필터 바 UI (chat-input.tsx, chat-page.tsx)
2. **Phase 2** — BE task_mode 자동 판별 (agent.py)
3. **Phase 3** — guided panel 통합 + 테스트 업데이트
