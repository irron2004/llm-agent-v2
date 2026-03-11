# SOP 질문 개선 구현 계획

요구사항: `docs/2026-03-11-sop질문개선.md`

---

## Phase 1. 참고 문서 뷰어 개선 (FE only)

### Step 1-1. 우측 bar 페이지 클릭 버그 수정

**파일**: `frontend/src/components/layout/index.tsx`

**현재 문제** (line 1238-1265):
- `previewImages`가 `docs.map()` → 문서당 1개만 생성 (`expanded_page_urls` 무시)
- `handleImageClick(docIndex, _pageIndex)` → `_pageIndex` 무시

**변경**:
1. `previewImages`를 `docs.flatMap()`으로 변경하여 페이지 단위 배열 생성 (line 1238-1255)
   ```typescript
   const previewImages = useMemo(() => {
     return docs.flatMap((doc) => {
       const urls = doc.expanded_page_urls?.filter(hasValidImageUrl) ??
                    (hasValidImageUrl(doc.page_image_url) ? [doc.page_image_url] : []);
       const pages = doc.expanded_pages ?? (doc.page != null ? [doc.page] : []);
       const displayTitle = /* 기존 로직 유지 */;
       if (urls.length === 0) {
         return [{ content: doc.snippet, title: displayTitle, page: doc.page, docId: doc.id }];
       }
       return urls.map((url, i) => ({
         url, title: displayTitle, page: pages[i], docId: doc.id,
       }));
     });
   }, [docs]);
   ```

2. `docIndex → globalPreviewIndex` 매핑 테이블 추가:
   ```typescript
   const docPageOffsets = useMemo(() => {
     const offsets: number[] = [];
     let acc = 0;
     for (const doc of docs) {
       offsets.push(acc);
       const count = doc.expanded_page_urls?.filter(hasValidImageUrl).length
                     || (hasValidImageUrl(doc.page_image_url) ? 1 : 1);
       acc += count;
     }
     return offsets;
   }, [docs]);
   ```

3. `handleImageClick` 수정 (line 1262-1265):
   ```typescript
   const handleImageClick = (docIndex: number, pageIndex: number) => {
     setPreviewIndex((docPageOffsets[docIndex] ?? 0) + pageIndex);
     setPreviewVisible(true);
   };
   ```

4. `handleDocClick` 수정 (line 1257-1260):
   ```typescript
   const handleDocClick = (docIndex: number) => {
     setPreviewIndex(docPageOffsets[docIndex] ?? 0);
     setPreviewVisible(true);
   };
   ```

**검증**: 우측 bar에서 p.3 클릭 → 모달에서 p.3이 표시되는지 확인

### Step 1-2. 인라인 이미지 크기 제한

**파일**: `frontend/src/features/chat/components/message-item.tsx`

**변경** (line 633-634):
```typescript
// Before:
maxWidth: pageUrls.length > 1 ? 150 : "100%",
maxHeight: pageUrls.length > 1 ? 200 : 300,

// After:
maxWidth: pageUrls.length > 1 ? 150 : 300,
maxHeight: pageUrls.length > 1 ? 200 : 200,
```

**파일**: `frontend/src/components/layout/index.tsx` - `.retrieved-doc-image` CSS

**변경**: 우측 bar 이미지도 동일하게 제한
```css
.retrieved-doc-image {
  max-width: 200px;
  max-height: 200px;
  object-fit: contain;
}
```

**검증**: 노트북 화면(1366x768)에서 이미지가 잘리지 않고 전체 페이지 윤곽이 보이는지 확인

---

## Phase 2. task_mode 라디오 버튼 + 언어 선택 제거 (FE+BE)

### Step 2-1. ChatInput에 searchMode prop 추가

**파일**: `frontend/src/features/chat/components/chat-input.tsx`

**변경**:
1. Props에 `searchMode` 추가:
   ```typescript
   type ChatInputProps = {
     onSend: (message: string) => void;
     onStop?: () => void;
     isStreaming?: boolean;
     disabled?: boolean;
     placeholder?: string;
     searchMode: "sop" | "issue" | null;
     onSearchModeChange: (mode: "sop" | "issue") => void;
   };
   ```

2. 라디오 버튼 UI 추가 (textarea 위):
   ```tsx
   <div className="search-mode-selector">
     <label className={searchMode === "sop" ? "active" : ""}>
       <input type="radio" name="searchMode" value="sop"
              checked={searchMode === "sop"}
              onChange={() => onSearchModeChange("sop")} />
       절차검색
     </label>
     <label className={searchMode === "issue" ? "active" : ""}>
       <input type="radio" name="searchMode" value="issue"
              checked={searchMode === "issue"}
              onChange={() => onSearchModeChange("issue")} />
       이슈검색
     </label>
   </div>
   ```

3. `searchMode === null`일 때 전송 비활성화:
   ```typescript
   const handleSend = () => {
     const trimmed = value.trim();
     if (!trimmed || disabled || isStreaming || !searchMode) return;
     onSend(trimmed);
     setValue("");
   };
   ```

4. placeholder 연동:
   ```typescript
   const effectivePlaceholder = !searchMode
     ? "검색 모드를 선택하세요"
     : searchMode === "sop"
       ? "절차/SOP 관련 질문을 입력하세요..."
       : "이슈/트러블 관련 질문을 입력하세요...";
   ```

### Step 2-2. chat-page에 searchMode state 관리

**파일**: `frontend/src/features/chat/pages/chat-page.tsx`

**변경**:
1. state 추가 (line ~46):
   ```typescript
   const [searchMode, setSearchMode] = useState<"sop" | "issue" | null>(null);
   ```

2. ChatInput에 prop 전달 (line ~569-582):
   ```tsx
   <ChatInput
     onSend={handleSend}
     onStop={stop}
     isStreaming={isStreaming}
     searchMode={searchMode}
     onSearchModeChange={setSearchMode}
     disabled={...}
   />
   ```

3. `handleSend`에서 `searchMode`를 `send()`에 전달 (line ~245):
   - `send({ text, searchMode })` 형태로 전달

### Step 2-3. use-chat-session에서 searchMode → task_mode 연동

**파일**: `frontend/src/features/chat/hooks/use-chat-session.ts`

**변경**:
1. `send()` 함수 signature에 `searchMode` 추가
2. API 요청 시 `searchMode`를 직접 payload에 포함:
   ```typescript
   const payload = {
     message: text,
     guided_confirm: true,
     task_mode_override: searchMode,  // "sop" | "issue"
     // ... 기타 필드
   };
   ```

3. guided_confirm resume 시 `task_mode`를 `searchMode`에서 가져오기:
   - 기존: `draftDecision.task_mode` (guided panel에서 선택)
   - 변경: `searchMode` (라디오에서 미리 선택된 값)

4. language 단계 제거:
   - `draftDecision`에서 `target_language` 제거 (자동 감지 사용)

### Step 2-4. guided-selection-panel에서 language/task 단계 제거

**파일**: `frontend/src/features/chat/components/guided-selection-panel.tsx`

**변경**:
1. 기본 steps에서 `"language"`와 `"task"` 제거 (line 70-79):
   ```typescript
   // Before: ["language", "device", "equip_id", "task"]
   // After:  ["device", "equip_id"]
   ```

2. `applyOption()`에서 language/task 분기 제거 (line 164-204)
3. `onComplete` 호출 시 `target_language`와 `task_mode`를 외부에서 주입받도록 변경

### Step 2-5. Backend — task_mode_override 지원

**파일**: `backend/api/routers/agent.py`

**변경**:
1. `AgentRequest` 모델에 `task_mode_override` 필드 추가:
   ```python
   task_mode_override: Optional[Literal["sop", "issue"]] = None
   ```

2. 요청 처리 시 `task_mode_override`를 state에 주입:
   ```python
   if req.task_mode_override:
       state_overrides["task_mode"] = req.task_mode_override
   ```

3. `auto_parse_confirm_node`에서 `task_mode`가 이미 state에 있으면 그 값을 유지

### Step 2-6. Backend — target_language fallback 강화

**파일**: `backend/llm_infrastructure/llm/langgraph_agent.py`

**변경** (line 3488-3491):
- `target_language`가 resume_decision에 없을 때 `detected_language` fallback은 이미 구현됨 (line 2251)
- `auto_parse_confirm_node`에서 `target_language`를 decision에서 제거해도 `answer_node`의 fallback chain이 동작:
  ```python
  answer_language = state.get("target_language") or state.get("detected_language") or "ko"
  ```
- 변경 불필요 — 기존 fallback chain이 정확히 원하는 동작

**검증**:
- 라디오 미선택 → 전송 버튼 비활성화
- "절차검색" 선택 → 질문 → `task_mode: "sop"`로 요청
- 한국어 질문 → 한국어 답변 (언어 선택 UI 없이)
- 영어 질문 → 영어 답변 (자동 감지)

---

## Phase 3. SOP 답변 프롬프트 개선 (BE only)

### Step 3-1. setup_ans_v2.yaml 개선

**파일**: `backend/llm_infrastructure/llm/prompts/setup_ans_v2.yaml`

**변경**: system 프롬프트를 work procedure/workflow 중심 + 고정 양식으로 교체:
```yaml
name: setup_ans
version: v2
description: Installer-style answer with refs for setup/installation questions (Korean).
system: |-
  당신은 설치/셋업 RAG 어시스턴트입니다.

  ## 규칙
  - REFS 라인만 증거로 사용하세요 (추측 금지).
  - REFS가 비어있으면 "관련 절차 문서를 찾지 못했습니다."라고만 답변하세요.
  - **중요: 반드시 한국어로 답변하세요.**
  - REFS에 서로 다른 장비(device_name)의 문서가 섞여 있으면, 장비별로 구분하여 답변하세요.
  - 서로 다른 장비의 절차나 수치를 하나로 합치지 마세요.

  ## 답변 우선순위
  - REFS에서 **Work Procedure**, **Workflow**, **작업 절차** 섹션을 최우선으로 참조하세요.
  - 절차가 있으면 반드시 번호 목록(1. 2. 3.)으로 제시하세요.
  - 배경 설명, 목적, 범위 등은 간략히만 언급하세요.

  ## 답변 양식 (반드시 이 순서와 형식을 따르세요)

  ### 작업 목적
  - (이 절차의 목적을 1~2줄로)

  ### 작업 전 확인
  - (REFS에 사전 조건/준비물이 있으면 bullet으로)
  - (없으면 이 섹션 생략)

  ### 작업 절차
  1. (첫 번째 단계) [출처번호]
  2. (두 번째 단계) [출처번호]
  3. ...

  ### 작업 후 확인
  - (REFS에 후속 확인 사항이 있으면 bullet으로)
  - (없으면 이 섹션 생략)

  ### 주의사항
  - (REFS에 Warning, Caution, Note가 있으면 bullet으로)
  - (없으면 이 섹션 생략)

  ### 참고문헌
  [1] doc_id (device_name)
  [2] doc_id (device_name)
messages:
  - role: user
    content: |
      질문: {sys.query}
      REFS:
      {ref_text}
```

### Step 3-2. 다국어 프롬프트 동일 적용

**파일**:
- `backend/llm_infrastructure/llm/prompts/setup_ans_en_v2.yaml`
- `backend/llm_infrastructure/llm/prompts/setup_ans_ja_v2.yaml`
- `backend/llm_infrastructure/llm/prompts/setup_ans_zh_v2.yaml`

**변경**: 동일한 구조를 각 언어로 번역 적용
- 양식 헤더: Purpose / Pre-check / Procedure / Post-check / Cautions / References
- Work Procedure/Workflow 우선 참조 규칙 동일

**검증**:
- 절차검색 → SOP 문서 검색됨 → 답변이 "작업 목적 / 작업 전 확인 / 작업 절차 / ..." 양식
- REFS 빈 경우 → "관련 절차 문서를 찾지 못했습니다."
- 영어 질문 → 영어 양식으로 동일 구조 답변

---

## 수용 기준

| # | 기준 | 검증 방법 |
|---|------|-----------|
| 1 | 우측 bar에서 p.N 클릭 시 해당 페이지가 모달에 표시 | 수동 테스트 — 다중 페이지 문서에서 p.3 클릭 |
| 2 | 인라인 이미지가 노트북 화면에서 잘리지 않음 | 1366x768 해상도에서 확인 |
| 3 | 라디오 미선택 시 전송 버튼 비활성화 | 수동 테스트 |
| 4 | "절차검색" 선택 → task_mode: "sop"로 API 요청 | Network 탭 확인 |
| 5 | 언어 선택 UI가 guided panel에서 사라짐 | 수동 테스트 |
| 6 | 한국어 질문 → 한국어 답변 (자동) | 품질 테스트 |
| 7 | 절차 답변이 고정 양식을 따름 | 품질 테스트 5건 |
| 8 | SOP 답변에서 Work Procedure 내용이 "작업 절차" 섹션에 표시 | 품질 테스트 |
| 9 | 기존 프론트엔드 테스트 69건 통과 | `npm run test` |
| 10 | 기존 백엔드 테스트 440+ 통과 | `uv run pytest tests/ -v` |

---

## 리스크

| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| 언어 자동 결정 실패 (혼합/짧은 입력) | 낮음 | 중간 | fallback chain: detected → 직전 턴 → "ko" |
| task_mode radio와 guided selection 충돌 | 중간 | 중간 | radio를 authoritative source로 지정, guided panel에서 task 단계 제거 |
| 프롬프트 고정 시 빈약 문서에서 어색한 양식 | 낮음 | 낮음 | "없으면 이 섹션 생략" 규칙으로 대응 |
| previewImages flatMap 변경 시 이미지 없는 문서 처리 | 낮음 | 낮음 | 이미지 없는 경우 snippet 텍스트로 fallback 유지 |

---

## 권장 작업 순서

1. **Phase 1** (FE) — 뷰어 버그 수정 + 이미지 크기 제한 (독립적, 바로 검증 가능)
2. **Phase 3** (BE) — 프롬프트 개선 (독립적, YAML만 수정)
3. **Phase 2** (FE+BE) — 라디오 버튼 + 언어 제거 (가장 큰 변경, 의존성 있음)
