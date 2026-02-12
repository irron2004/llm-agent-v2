# 사내 테스트 점검 Task 목록

## Task 1: auto_parse 프롬프트 영어 번역
- **파일**: `backend/llm_infrastructure/llm/prompts/auto_parse_v1.yaml`
- **내용**: 한국어로 작성된 프롬프트를 영어로 번역
- **상태**: ✅ 완료
- **수행 내용**:
  - 역할/입력/출력 형식 섹션 영어화
  - 파싱 규칙 (장비명, 문서 종류, 언어 감지) 영어화
  - 예시 섹션의 레이블 영어화
  - messages 섹션의 한글 레이블 영어화 (질문 → Query, 장비 목록 → Device list 등)

## Task 2: multi query 생성 확인
- **내용**: multi query가 잘 생성되는지 검증
- **상태**: ✅ 완료
- **수행 내용**:
  - 로그 분석으로 문제 케이스 발견:
    - `"query1"`, `"query2"`, `"query3"` placeholder 누출
    - `"..."` ellipsis 누출
    - `"Let's produce 4 queries. Example:"` meta 텍스트 누출
    - 잘린 쿼리 (문장 중간에서 끊긴 출력)
  - **수정 파일**:
    - `backend/llm_infrastructure/llm/prompts/st_mq_v1.yaml`: placeholder 예시 제거, 형식 설명으로 교체
    - `backend/llm_infrastructure/llm/langgraph_agent.py`: `_is_garbage_query` 함수에 필터 추가
      - `query1`, `q1` 같은 placeholder 패턴
      - `...`, `---` 같은 ellipsis/punctuation-only
      - `"let's produce queries"` 같은 meta 텍스트
      - `"example:"`, `"output format"`, `"output only"` 감지
      - 영어 stopword/한국어 조사로 끝나는 잘린 쿼리 감지
    - `backend/llm_infrastructure/llm/langgraph_agent.py`: `st_mq_node` LLM 호출 `max_tokens` 상향
      - 기존: 기본값 256 (분류용 `MAX_TOKENS_CLASSIFICATION`)
      - 변경: 1024 (쿼리 생성에 충분한 토큰 확보)
  - **잘린 쿼리 원인 분석**:
    - `_invoke_llm` 기본 `max_tokens=256`이 너무 낮아서 output 도중 잘림 발생
    - 예: `"알람이 표시되는 코드나 메시지가 있"` ← 256 토큰 한도 도달로 중간 절단
  - **추가 작업**:
    - MQ 파싱 누출 방지(placeholder/ellipsis/meta/truncation) 중심으로 보정

## Task 3: myservice에서 p0: 제거
- **내용**: myservice 문서 타입에서 `p0:` 접두어 제거
- **상태**: ⏳ 미확인

## Task 4: 관련 문서에 번호 표시
- **내용**: 챗 화면의 관련 문서 목록에 1~20 번호 추가
- **상태**: ✅ 완료
- **수행 내용**:
  - **파일**: `frontend/src/features/chat/components/message-item.tsx`
  - 관련 문서 제목 앞에 `1.` 형식 번호 표시 (회색 텍스트)
  - `idx + 1`로 1부터 시작하는 번호 생성
  - 예시:
    ```tsx
    const displayIndex = idx + 1;
    <span style={{ marginRight: 6, color: "var(--color-text-secondary)" }}>
      {displayIndex}.
    </span>
    <span>{displayTitle}</span>
    ```

## Task 5: 취소선 markdown 문제 수정
- **내용**: 확장 문서/참고 문서에서 의도하지 않은 `~~취소선~~` markdown이 렌더링되는 문제 해결
- **상태**: ✅ 완료 (이미 구현됨)
- **확인 내용**:
  - **파일**: `frontend/src/features/chat/components/message-item.tsx`
  - `preprocessSnippet` 함수에서 markdown 특수문자 이스케이프 처리:
    ```typescript
    // Escape tildes for strikethrough (~~text~~)
    processed = processed.replace(/~~/g, "\\~\\~");
    // Escape underscores that are part of words (e.g., file_name, __init__)
    processed = processed.replace(/(\w)_(\w)/g, "$1\\_$2");
    // Escape asterisks that might trigger bold/italic
    processed = processed.replace(/(\w)\*(\w)/g, "$1\\*$2");
    ```
  - 처리 대상: `~~` (취소선), `_` (이탤릭), `*` (볼드/이탤릭)

## Task 6: UI 영어화 (마이크론 대상)
- **내용**: 화면에 표시되는 한글 텍스트를 영어로 변경
- **상태**: ✅ 완료
- **수행 내용**:
  - 채팅 페이지, 디바이스 제안, 레이아웃 컴포넌트 영어화 완료
  - 한글 텍스트 잔여 없음 확인

| 한글 (원래) | 영어 (변경 완료) |
|-------------|-----------------|
| PE Agent에 오신 것을 환영합니다 | Welcome to PE Agent |
| 실행 로그 | Activity Log |
| 특정 기기에 대해서 검색을 할까요? | Should I search for a specific equipment? |
| ESC를 누르면 채팅 화면으로 돌아갑니다 | Press ESC to go back to the chat |
| 다음의 기기 중에서 어떤 기기를 중심으로 다시 검색할까요? | Which equipment would you like to search for? |
| 추가 검색을 원하지 않으면 ESC를 눌러주세요. | Press ESC if you don't want additional search. |

## Task 7: Judge Node JSON Parse Error 수정
- **내용**: Reasoning 모델 사용 시 judge_node에서 JSON 파싱 실패 문제 해결
- **상태**: ✅ 완료
- **발생 현상**:
  - RAG trace 로그에서 `parse_error` 발생
  ```json
  {
    "node": "judge_node",
    "event": "UNFAITHFUL",
    "issues": ["parse_error"],
    "hint": "judge JSON parse failed",
    "raw_llm_output": "We need to judge if the answer is faithful to evidence. The answer contains many claims..."
  }
  ```
- **원인 분석**:
  1. **Reasoning 모델 특성**: vLLM에 reasoning 모델(예: DeepSeek-R1) 사용 중
  2. **응답 구조 차이**: Reasoning 모델은 응답을 두 부분으로 분리
     - `reasoning`: 사고 과정 텍스트
     - `text`: 실제 JSON 응답
  3. **기존 코드 문제**: `judge_node`가 `_invoke_llm` 사용 → reasoning이 text에 섞여서 반환되는 경우 JSON 파싱 실패
- **해결 방법**:
  - **파일**: `backend/llm_infrastructure/llm/langgraph_agent.py`
  - **수정 1**: `_extract_json_from_text()` 헬퍼 함수 추가
    ```python
    def _extract_json_from_text(text: str) -> dict | None:
        """Extract JSON object from text that may contain reasoning/explanations."""
        # Strategy 1: Direct parse (if text is pure JSON)
        # Strategy 2: Find JSON object pattern {...}
        # Strategy 3: Find JSON in code blocks ```json ... ```
    ```
  - **수정 2**: `judge_node`에서 `_invoke_llm` → `_invoke_llm_with_reasoning` 변경
    ```python
    # Before
    raw = _invoke_llm(llm, sys, user)
    judge = json.loads(raw)

    # After
    raw, reasoning = _invoke_llm_with_reasoning(llm, sys, user)
    judge = _extract_json_from_text(raw)
    ```
- **`_invoke_llm` vs `_invoke_llm_with_reasoning` 차이**:
  | 함수 | 반환값 | 설명 |
  |------|--------|------|
  | `_invoke_llm` | `str` | `out.text`만 반환 |
  | `_invoke_llm_with_reasoning` | `tuple[str, str\|None]` | `(out.text, out.reasoning)` 분리 반환 |
- **수정 3**: `max_tokens` 증가 (핵심 수정)
  ```python
  # 상수 추가
  MAX_TOKENS_JUDGE = 1024  # 기존 256 → 1024

  # judge_node에서 명시적 사용
  raw, reasoning = _invoke_llm_with_reasoning(llm, sys, user, max_tokens=MAX_TOKENS_JUDGE)
  ```
- **실제 원인 (로그 분석 결과)**:
  ```
  raw output:
  We need to judge whether answer faithful to references...
  - Causes: Z-axis vibration, teaching position, end-e  ← 여기서 잘림!
  ```
  - 모델이 reasoning을 출력하다가 256 토큰 한도에 도달
  - JSON 출력 전에 응답이 잘려서 `parse_error` 발생
- **효과**:
  - `max_tokens=1024`로 증가하여 reasoning + JSON 모두 출력 가능
  - `_extract_json_from_text()`로 reasoning 텍스트 사이에서 JSON 추출
  - 세 가지 전략으로 JSON 추출 시도 (직접 파싱 → 패턴 매칭 → 코드 블록)
