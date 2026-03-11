# 122B MoE(Q4) 모델 품질 저하 대응 (2026-03-10)

## TL;DR
> **Summary**: `ollama(qwen3.5:122b-a10b-q4_K_M)` 교체 이후 발생한 환각/반복/프롬프트 무시/오판정(judge) 문제를 "LLM 의존"에서 "결정적(deterministic) 게이트"로 이동시켜 fail-closed로 안전화한다.
> **Deliverables**:
> - `task_mode`→`route` 정합성 고정 (task_mode 우선 라우팅)
> - Retrieval 약할 때 fail-closed(LLM 호출 스킵) + 1회 조건변경 재시도 정책
> - URL/도메인 문자열 금지(프롬프트 + 결정적 스캐너) + judge override
> - Judge 강화(결정적 override 우선순위 + 테스트)
> - MQ/동의어 확장 트리거 개선 (특히 "마노미터" 케이스)
> - 회귀 테스트/증거(evidence) 번들화
> **Effort**: Large
> **Parallel**: YES - 3 waves
> **Critical Path**: Deterministic gate 설계/테스트 → 그래프 라우팅/게이트 삽입 → judge override → MQ/동의어 트리거

## Context
### Original Request
- `docs/2026-03-10-120B-model성능.md`에 기반해 대응 플랜 수립.

### Interview Summary
- 기본 전략: 모델 변경 없이 P0/P1(가드레일/분기/게이트)부터 적용 후, P2에서 검색/모델 비교를 수행.
- `task_mode`/`route` 불일치 해소는 기존 문서(`docs/2026-03-09-agent-분기-개선.md`)의 Phase A와 동일 방향으로 본 플랜에 포함.

### Repo Grounding (facts)
- `route_node`는 `task_mode`를 고려하지 않고 router LLM만 호출: `backend/llm_infrastructure/llm/langgraph_agent.py:997`.
- `prepare_retrieve`는 기본적으로 `skip_mq=True`로 1차 시도에서 MQ를 건너뜀: `backend/services/agents/langgraph_rag_agent.py:435`.
- judge system 프롬프트는 YAML이 아니라 코드 상수(`DEFAULT_JUDGE_*`): `backend/llm_infrastructure/llm/langgraph_agent.py:194`.

### Metis Review (gaps addressed)
- Deterministic decision lattice(게이트 우선순위)를 명시하고, "LLM judge는 advisory"로 취급.
- URL 정책/재시도 정책/사용자 UX(거절 메시지)를 플랜에서 결정(기본값)하고 테스트로 고정.
- Streaming 경로에서 금지 콘텐츠가 먼저 흘러나오지 않도록(버퍼링/사전 게이트) 가드.

## Work Objectives
### Core Objective
- Retrieval 실패/저연관 REFS 상황에서 122B Q4 MoE가 환각으로 빈 공간을 채우는 경로를 차단하고, judge 오판정으로 불량 답변이 노출되는 것을 막는다.

### Deliverables
- 결정적(evidence-based) fail-closed 정책 + 재시도 매트릭스(조건이 바뀔 때만 1회 재시도)
- URL/도메인 문자열 정책(프롬프트 + 결정적 스캐너 + judge override)
- `task_mode`가 선언된 경우 라우팅/템플릿 선택에서 router LLM을 우회
- MQ/동의어 확장 트리거 개선 (마노미터↔manometer↔압력계)
- 회귀 테스트(재현 케이스 포함) + evidence 번들

### Definition of Done (verifiable)
- Backend:
  - `uv run pytest -q tests/api/test_agent_response_metadata_contract.py`
  - 새로 추가되는 회귀 테스트 파일들이 모두 통과
- Frontend(필요 시): `npm -C frontend test`
- Evidence:
  - `.sisyphus/evidence/122b-moe-quality-regression/` 아래에 재현/검증 결과가 파일로 남음

### Must Have
- "증거 부족" 상태에서 LLM answer 호출을 수행하지 않는다(fail-closed).
- URL/도메인 문자열이 답변에 포함되면 judge가 반드시 실패 처리(override)한다.
- `task_mode=issue` 선택이 route/setup 프롬프트로 빠지지 않는다.

### Must NOT Have
- LLM-only judge가 최종 신뢰성 판정을 독점 (override 없는 구조 금지).
- 동일 조건(동일 refs/동일 모델/동일 프롬프트)로 무의미한 재시도 반복.
- B4(HTTP gate)와 결합된 before/after 품질 게이트 작업.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: tests-after (Pytest + Vitest)
- QA policy: 모든 task는 재현/실패 케이스를 포함
- Evidence root: `.sisyphus/evidence/122b-moe-quality-regression/`

## Execution Strategy
### Parallel Execution Waves
Wave 1: 결정적 게이트/검출 유틸 + 회귀 테스트 골격
Wave 2: 그래프 라우팅/게이트 삽입 + answer/judge 동작 변경
Wave 3: MQ/동의어 트리거 + 평가 하네스 + 문서/evidence 정리

### Dependency Matrix (high level)
- Gate 유틸/정책이 먼저 확정되어야 answer/judge/graph 변경이 안정적으로 테스트된다.

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. 결정적 정책(Decision Lattice) 고정 + 공용 유틸 추가 (URL/도메인 스캐너, 증거충분도)

  **What to do**:
  - `backend/llm_infrastructure/llm/langgraph_agent.py`에 아래 유틸을 추가한다.
    - `_extract_url_like_tokens(text: str) -> list[str]`
    - `_has_forbidden_url(answer: str, refs_text: str) -> tuple[bool, list[str]]`
      - 금지 조건(결정 완료):
        - 무조건 금지: `http://`, `https://`, `www.` 포함
        - bare-domain 금지: `\b[a-z0-9-]+\.(com|net|org|io|ai|co|kr|jp|cn|dev|app|site|info)\b` (case-insensitive)
        - 단, 토큰이 `refs_text`에 (case-insensitive) 그대로 포함되면 allow
      - 반환: `(is_forbidden, tokens)`
    - `_evidence_is_weak(state: AgentState) -> tuple[bool, str]`
      - 기준(결정 완료):
        - `ref_text == "EMPTY"` 또는 `len(ref_json)==0` → weak_reason=`"empty_refs"`
        - `guardrail_final_count <= 1` AND `mq_used == False` → weak_reason=`"single_query_no_mq"`
        - 그 외 → not weak
  - 유틸의 단위 테스트를 추가한다: `backend/tests/test_quality_policy_utils.py`

  **Must NOT do**:
  - URL 금지 규칙을 "점 하나라도 있으면" 같은 과도한 정규식으로 만들지 않는다(부품명/버전 false positive 방지).
  - 게이트 정책을 프롬프트에만 의존하지 않는다(결정적 override가 필수).

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: 순수 유틸 + 단위 테스트
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 2,3,4 | Blocked By: none

  **References**:
  - 문제 정의: `docs/2026-03-10-120B-model성능.md:7`
  - 기존 judge/answer 노드: `backend/llm_infrastructure/llm/langgraph_agent.py:2085`, `backend/llm_infrastructure/llm/langgraph_agent.py:2175`
  - guardrail_final_count 의미(최종 search_queries 개수 fallback): `backend/api/routers/agent.py:727`

  **Acceptance Criteria**:
  - [ ] `uv run pytest -q backend/tests/test_quality_policy_utils.py` 통과

  **QA Scenarios**:
  ```
  Scenario: URL 스캐너가 http(s) 링크를 차단
    Tool: Bash
    Steps: uv run pytest -q backend/tests/test_quality_policy_utils.py -k http
    Expected: pass
    Evidence: .sisyphus/evidence/122b-moe-quality-regression/task-1-url-scan-http.txt

  Scenario: bare-domain은 TLD allowlist 기반으로만 차단
    Tool: Bash
    Steps: uv run pytest -q backend/tests/test_quality_policy_utils.py -k bare_domain
    Expected: pass
    Evidence: .sisyphus/evidence/122b-moe-quality-regression/task-1-url-scan-domain.txt
  ```

  **Commit**: YES | Message: `feat(agent): add deterministic url/evidence policy utils` | Files: `backend/llm_infrastructure/llm/langgraph_agent.py`, `backend/tests/test_quality_policy_utils.py`

- [ ] 2. Evidence Gate 노드 추가(LLM answer 호출 전 fail-closed) + 그래프 wiring

  **What to do**:
  - 목표: "증거가 약하면 answer/judge LLM 호출 없이" 안전 응답으로 종료하거나(또는 1회 조건변경 재시도)한다.
  - `backend/llm_infrastructure/llm/langgraph_agent.py`에 노드를 추가한다.
    - `evidence_gate_node(state) -> Command[Literal["retry_mq","abstain","expand_related"]]`
      - `weak, reason = _evidence_is_weak(state)`
      - `weak==False` → goto=`"expand_related"`
      - `weak==True` AND `state.attempts==0` AND `state.mq_mode != "off"` → goto=`"retry_mq"`
      - 그 외 weak → goto=`"abstain"`
    - `abstain_node(state) -> dict`:
      - `answer`를 아래 고정 포맷으로 반환(결정 완료):
        - ko: `"관련 문서를 찾지 못했습니다. 장비/모듈/알람코드/수치(로그)를 더 알려주시면 다시 검색해볼게요."`
        - en: `"I couldn't find relevant documents to answer safely. Please provide device/module/alarm code and any values/logs."`
        - zh/ja도 동일 톤으로 1문장(번역은 고정 문자열로 작성)
      - `judge = {"faithful": false, "issues": ["insufficient_evidence", reason], "hint": ""}`
      - `retry_strategy = "abstain"`로 설정
  - `backend/services/agents/langgraph_rag_agent.py` 그래프에 `evidence_gate`와 `abstain` 노드를 삽입한다.
    - Edge 변경(결정 완료):
      - 기존: `retrieve -> (ask_user?) -> expand_related -> answer -> judge`
      - 변경: `retrieve -> evidence_gate -> (retry_mq | expand_related | abstain)`
      - `abstain -> done(END)` 로 직접 연결하여 judge conditional을 타지 않게 한다.

  **Must NOT do**:
  - abstain 경로에서 `answer_node` 또는 `judge_node`를 호출하지 않는다.
  - weak 증거에서 동일 조건(answer LLM 호출)로 재시도하지 않는다.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: 그래프 wiring 변경 + Command 분기
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 4,5 | Blocked By: 1

  **References**:
  - prepare_retrieve가 기본 skip_mq인 사실: `backend/services/agents/langgraph_rag_agent.py:435`
  - retry_mq edge: `backend/services/agents/langgraph_rag_agent.py:687`
  - answer/judge node 위치: `backend/llm_infrastructure/llm/langgraph_agent.py:2085`, `backend/llm_infrastructure/llm/langgraph_agent.py:2175`
  - 문제 체크포인트: `docs/2026-03-10-120B-model성능.md:217`

  **Acceptance Criteria**:
  - [ ] `uv run pytest -q tests/api/test_agent_fail_closed_on_weak_evidence.py` 통과

  **QA Scenarios**:
  ```
  Scenario: 검색 결과가 비면 LLM 호출 없이 fail-closed
    Tool: Bash
    Steps: uv run pytest -q tests/api/test_agent_fail_closed_on_weak_evidence.py -k empty_refs
    Expected: pass; 응답 answer는 고정 문구; Stub LLM의 answer-system이 호출되지 않음
    Evidence: .sisyphus/evidence/122b-moe-quality-regression/task-2-empty-refs.txt

  Scenario: guardrail_final_count=1 + mq_used=false이면 retry_mq 1회 후에도 약하면 abstain
    Tool: Bash
    Steps: uv run pytest -q tests/api/test_agent_fail_closed_on_weak_evidence.py -k single_query
    Expected: pass; retry_mq 노드가 1회만 실행; 무한 루프 없음
    Evidence: .sisyphus/evidence/122b-moe-quality-regression/task-2-single-query.txt
  ```

  **Commit**: YES | Message: `feat(agent): add evidence gate and fail-closed abstain path` | Files: `backend/llm_infrastructure/llm/langgraph_agent.py`, `backend/services/agents/langgraph_rag_agent.py`, tests

- [ ] 3. task_mode 우선 라우팅(issue→ts, sop→setup) + router LLM fallback

  **What to do**:
  - `backend/llm_infrastructure/llm/langgraph_agent.py:997`의 `route_node`를 수정한다.
    - `task_mode == "sop"`이면 route="setup"으로 고정하고 router LLM 호출을 스킵
    - `task_mode == "issue"`이면 route="ts"로 고정하고 router LLM 호출을 스킵
    - 그 외(`task_mode` 없음/"all")만 기존 router LLM 사용
  - `parsed_query.route`도 함께 업데이트한다.
  - API 테스트로 "issue task_mode인데 route가 setup으로 나오지 않음"을 고정한다.

  **Must NOT do**:
  - Route literal을 추가하지 않는다.
  - `task_mode`가 설정된 경우에도 router LLM이 route를 덮어쓰게 두지 않는다.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: 국소 변경 + API 테스트
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 5 | Blocked By: 1

  **References**:
  - task_mode가 state에 저장되는 위치: `backend/llm_infrastructure/llm/langgraph_agent.py:3320`
  - 재현 케이스(불일치): `docs/2026-03-10-120B-model성능.md:15`
  - guided confirm 테스트 패턴: `tests/api/test_agent_autoparse_confirm_interrupt_resume.py:234`

  **Acceptance Criteria**:
  - [ ] `uv run pytest -q tests/api/test_agent_task_mode_route_alignment.py` 통과

  **QA Scenarios**:
  ```
  Scenario: task_mode=issue이면 route=ts로 고정
    Tool: Bash
    Steps: uv run pytest -q tests/api/test_agent_task_mode_route_alignment.py -k issue
    Expected: pass
    Evidence: .sisyphus/evidence/122b-moe-quality-regression/task-3-issue-route.txt
  ```

  **Commit**: YES | Message: `fix(agent): make task_mode override route` | Files: `backend/llm_infrastructure/llm/langgraph_agent.py`, tests

- [ ] 4. Judge 강화: 결정적 override(URL/REFS/인용 범위/언어혼합) + 테스트

  **What to do**:
  - `backend/llm_infrastructure/llm/langgraph_agent.py:2175`의 `judge_node`를 강화한다.
    - LLM judge JSON 파싱 이후, 아래 결정적 override를 적용(우선순위 고정):
      1) `_has_forbidden_url(answer, ref_text)`가 true이면 → `faithful=false`, issues에 `url_violation` + 토큰 목록(최대 3개) 추가
      2) `ref_text == "EMPTY"`인데 답안이 비어있지 않으면 → `faithful=false`, issues에 `empty_refs_answered` 추가
      3) 답안의 인용이 REFS 범위를 벗어나면 → `faithful=false`, issues에 `citation_out_of_range` 추가
         - 규칙(결정 완료): `r"\[(\d+)\]"`로 숫자 인용을 수집하고, `max(citation) > len(ref_items)`면 실패
      4) `issues`가 비어있지 않으면 `faithful`은 반드시 false (LLM이 faithful=true라도 override)
      5) (선택적이지만 본 플랜에서는 P1로 포함) 언어 혼합 감지:
         - `metadata.target_language`가 ko인데 답안에 `参考文献` 또는 CJK(중국어 한자 범위 `\u4e00-\u9fff`)가 포함되면 `language_mixed` 추가 + faithful=false
  - `DEFAULT_JUDGE_*`(코드 문자열)에도 "URL/도메인 생성 금지" 및 "REFS 외 근거 금지"를 명시적으로 추가한다.
  - 단위 테스트: `backend/tests/test_judge_deterministic_overrides.py`를 추가하여 LLM judge가 faithful=true를 반환해도 override로 false가 되는 케이스를 고정한다.

  **Must NOT do**:
  - override 우선순위를 케이스마다 다르게 두지 않는다(항상 동일한 lattice).
  - URL 검출 규칙을 prompt-only로 두지 않는다.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: 정책/정규식/테스트 경계값
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: none | Blocked By: 1

  **References**:
  - judge_node: `backend/llm_infrastructure/llm/langgraph_agent.py:2175`
  - DEFAULT_JUDGE_*: `backend/llm_infrastructure/llm/langgraph_agent.py:194`
  - 증상 예시: `docs/2026-03-10-120B-model성능.md:21`

  **Acceptance Criteria**:
  - [ ] `uv run pytest -q backend/tests/test_judge_deterministic_overrides.py` 통과

  **QA Scenarios**:
  ```
  Scenario: judge가 faithful=true를 반환해도 URL 위반이면 override로 false
    Tool: Bash
    Steps: uv run pytest -q backend/tests/test_judge_deterministic_overrides.py -k url_violation
    Expected: pass
    Evidence: .sisyphus/evidence/122b-moe-quality-regression/task-4-judge-url-override.txt

  Scenario: 인용 범위 초과([99])는 반드시 실패
    Tool: Bash
    Steps: uv run pytest -q backend/tests/test_judge_deterministic_overrides.py -k citation_out_of_range
    Expected: pass
    Evidence: .sisyphus/evidence/122b-moe-quality-regression/task-4-judge-citation-range.txt
  ```

  **Commit**: YES | Message: `feat(agent): add deterministic judge overrides for urls and citations` | Files: `backend/llm_infrastructure/llm/langgraph_agent.py`, tests

- [ ] 5. Answer 프롬프트 P0 강화: URL/도메인 생성 금지 + REFS 준수 재강조 (v1/v2)

  **What to do**:
  - 아래 프롬프트 파일들에 공통 문구를 추가한다(결정 완료):
    - "절대로 URL(예: http(s)://, www., digikey.com 같은 도메인)을 생성하거나 출력하지 마세요. REFS에 없는 출처/링크를 만들지 마세요."
    - "REFS에 없는 정보를 추측하지 말고, 근거가 부족하면 '관련 문서를 찾지 못했습니다'라고 말하세요."
  - 대상 파일(최소):
    - `backend/llm_infrastructure/llm/prompts/setup_ans_v1.yaml`
    - `backend/llm_infrastructure/llm/prompts/ts_ans_v1.yaml`
    - `backend/llm_infrastructure/llm/prompts/general_ans_v1.yaml`
    - `backend/llm_infrastructure/llm/prompts/setup_ans_v2.yaml`
    - `backend/llm_infrastructure/llm/prompts/ts_ans_v2.yaml`
    - `backend/llm_infrastructure/llm/prompts/general_ans_v2.yaml`
  - 언어별 v1 프롬프트가 실제 런타임에 존재한다면(이미 존재): en/zh/ja v1에도 동일 문구를 추가한다.
  - 프롬프트 로딩 스모크 테스트를 추가한다: `backend/tests/test_prompt_files_load.py`

  **Must NOT do**:
  - 프롬프트만으로 안전을 보장하려 하지 않는다(본 플랜의 핵심은 evidence_gate + judge override).

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: 다수 YAML 일관 편집
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: none | Blocked By: none

  **References**:
  - 프롬프트 구조 예시: `backend/llm_infrastructure/llm/prompts/setup_ans_v1.yaml:4`
  - 문제 증상(URL 환각): `docs/2026-03-10-120B-model성능.md:21`

  **Acceptance Criteria**:
  - [ ] `uv run pytest -q backend/tests/test_prompt_files_load.py` 통과

  **QA Scenarios**:
  ```
  Scenario: v1/v2 answer 프롬프트가 모두 로드된다
    Tool: Bash
    Steps: uv run pytest -q backend/tests/test_prompt_files_load.py
    Expected: pass
    Evidence: .sisyphus/evidence/122b-moe-quality-regression/task-5-prompts-load.txt
  ```

  **Commit**: YES | Message: `docs(prompt): harden answer prompts against url hallucination` | Files: `backend/llm_infrastructure/llm/prompts/*_ans_*_v*.yaml`, tests

- [ ] 6. Retrieval 안전망: 마노미터 동의어 확장 + MQ 트리거(조건 변경 재시도)

  **What to do**:
  - `backend/services/agents/langgraph_rag_agent.py:435`의 `_prepare_retrieve_node`에 "도메인 동의어"를 추가한다.
    - 규칙(결정 완료): query에 아래 토큰이 포함되면 search_queries에 동의어를 추가 (최대 4개까지, 중복 제거)
      - 트리거: `마노미터` 또는 `manometer`
      - 추가 후보(고정): `manometer`, `mano meter`, `pressure gauge`, `압력계`
  - Evidence gate(Task 2)에서 `single_query_no_mq`일 때 `retry_mq`로 1회만 조건변경 재시도하도록 유지한다.
  - 테스트: API 수준에서 `message`에 "마노미터"가 들어가면 `metadata.search_queries_final`에 동의어가 포함되는지 확인.

  **Must NOT do**:
  - 동의어 확장을 전역 무제한으로 하지 않는다(키워드 드리프트/노이즈 증가 방지).

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: 작은 규칙 추가 + 테스트
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: none | Blocked By: 2

  **References**:
  - 재현 키워드: `docs/2026-03-10-120B-model성능.md:14`
  - prepare_retrieve 기본 bilingual queries: `backend/services/agents/langgraph_rag_agent.py:435`

  **Acceptance Criteria**:
  - [ ] `uv run pytest -q tests/api/test_agent_domain_synonym_expansion.py` 통과

  **QA Scenarios**:
  ```
  Scenario: "마노미터" 입력 시 search_queries에 pressure gauge 동의어가 포함
    Tool: Bash
    Steps: uv run pytest -q tests/api/test_agent_domain_synonym_expansion.py
    Expected: pass
    Evidence: .sisyphus/evidence/122b-moe-quality-regression/task-6-synonyms.txt
  ```

  **Commit**: YES | Message: `feat(retrieval): add manometer synonym expansion and mq trigger gate` | Files: `backend/services/agents/langgraph_rag_agent.py`, tests

- [ ] 7. 런타임 진단 메타데이터 추가(llm_method/model/prompt_spec_version) + 계약 테스트 업데이트

  **What to do**:
  - `backend/api/routers/agent.py`의 metadata에 아래 키를 추가한다(결정 완료, 비밀값 금지):
    - `llm_method`: `rag_settings.llm_method`
    - `llm_model`: method==ollama면 `ollama_settings.model_name`, 아니면 `vllm_settings.model_name`
    - `prompt_spec_version`: `rag_settings.prompt_spec_version`
  - 계약 테스트 업데이트:
    - `tests/api/test_agent_response_metadata_contract.py`의 required_keys에 위 3개를 추가하고 타입을 검증한다.

  **Must NOT do**:
  - base_url 같은 내부 주소는 metadata에 노출하지 않는다.

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: API metadata + 테스트
  - Skills: []

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: none | Blocked By: none

  **References**:
  - get_default_llm 분기: `backend/api/dependencies.py:162`
  - metadata 빌더: `backend/api/routers/agent.py:727`

  **Acceptance Criteria**:
  - [ ] `uv run pytest -q tests/api/test_agent_response_metadata_contract.py` 통과

  **QA Scenarios**:
  ```
  Scenario: metadata에 llm_method/llm_model/prompt_spec_version 포함
    Tool: Bash
    Steps: uv run pytest -q tests/api/test_agent_response_metadata_contract.py
    Expected: pass
    Evidence: .sisyphus/evidence/122b-moe-quality-regression/task-7-metadata-contract.txt
  ```

  **Commit**: YES | Message: `chore(api): expose llm and prompt spec metadata for diagnosis` | Files: `backend/api/routers/agent.py`, `tests/api/test_agent_response_metadata_contract.py`

- [ ] 8. 재현 케이스 회귀 테스트 + Evidence 번들 생성(문서 업데이트 포함)

  **What to do**:
  - 테스트를 추가한다: `tests/api/test_agent_122b_moe_regression_case.py`
    - guided_confirm → resume_decision(task_mode=issue, device skip) 경로로 실행
    - FakeSearchService를 "마노미터"에 대해 빈 결과/저연관 결과로 세팅
    - 기대: 최종 answer는 fail-closed(또는 retry_mq 1회 후 abstain)이며 URL/도메인이 포함되지 않음
  - Evidence 문서 작성(실행 로그/결과 요약):
    - `.sisyphus/evidence/122b-moe-quality-regression/repro-case.md`
    - 포함: 입력 payload, 최종 metadata(선택된 route/task_mode/guardrail_final_count/mq_used), 최종 answer/judge
  - `docs/2026-03-10-120B-model성능.md` 하단 체크포인트에 "완료" 상태와 변경 커밋/테스트 링크를 추가한다.

  **Must NOT do**:
  - 품질 지표를 사람 수동 판단으로만 남기지 않는다(테스트로 최소한의 안전 보장).

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: API 회귀 테스트 + evidence 문서화
  - Skills: []

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: none | Blocked By: 2,4

  **References**:
  - 재현 케이스 설명: `docs/2026-03-10-120B-model성능.md:12`
  - guided_confirm 테스트 패턴: `tests/api/test_agent_autoparse_confirm_interrupt_resume.py:234`

  **Acceptance Criteria**:
  - [ ] `uv run pytest -q tests/api/test_agent_122b_moe_regression_case.py` 통과
  - [ ] evidence 파일 존재: `.sisyphus/evidence/122b-moe-quality-regression/repro-case.md`

  **QA Scenarios**:
  ```
  Scenario: 122B Q4 회귀 케이스에서 URL 환각이 노출되지 않음
    Tool: Bash
    Steps: uv run pytest -q tests/api/test_agent_122b_moe_regression_case.py
    Expected: pass; answer에 http(s) 또는 bare-domain 토큰 없음
    Evidence: .sisyphus/evidence/122b-moe-quality-regression/task-8-regression-case.txt
  ```

  **Commit**: YES | Message: `test(agent): add regression fixture for 122b moe hallucination case` | Files: `tests/api/test_agent_122b_moe_regression_case.py`, evidence/doc updates


## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA (agent-executed) — unspecified-high (+ playwright if UI)
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit 1: `feat(agent): add deterministic evidence gates and url scanner`
- Commit 2: `feat(agent): align task_mode routing and fail-closed answer/judge`
- Commit 3: `feat(retrieval): improve mq triggers and add domain synonyms`

## Success Criteria
- `docs/2026-03-10-120B-model성능.md` 재현 케이스에서:
  - URL/도메인 환각이 사용자 출력으로 노출되지 않음
  - 증거 부족 시 fail-closed 메시지로 안전 종료
  - `task_mode=issue`가 setup 형식으로 잘못 라우팅되지 않음
  - judge가 환각/규칙 위반을 faithful로 통과시키지 않음
