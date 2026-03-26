# Task: SOP/Setup 문서 선택 시 비절차 질문 라우팅 편향 개선

Status: Draft
Owner: hskim
Branch or worktree: TBD
Created: 2026-03-26

## Goal

SOP/Setup 문서를 선택한 상태에서 절차(교체, 설치 등) 이외의 질문("work sheet 조회해줘", "tool list 보여줘", "scope 알려줘" 등)을 했을 때, 시스템이 절차 답변으로 편향되지 않고 사용자 의도에 맞는 답변을 생성하도록 개선한다.

## Why

현재 시스템은 SOP/Setup 문서를 선택하면 `_infer_task_mode_from_doc_types()`가 `task_mode="sop"`로 강제 설정한다. 이후 LLM 라우터의 의도 분류 결과와 무관하게 전체 파이프라인이 절차 중심으로 동작한다:

1. **MQ 생성**: `setup_mq` 사용 → 절차 키워드(토크, 스펙, 밸리데이션) 위주 검색 쿼리
2. **검색 결과 조정**: `sop_only_predicate=True` → Scope/Contents/목차 페이지에 패널티
3. **답변 생성**: `setup_ans` 템플릿 → "## 작업 절차" 번호 목록 형식 강제

이로 인해 "work sheet를 조회해줘"처럼 정보 조회 의도의 질문에도 절차 형식의 답변이 나오는 문제가 있다.

### SOP 문서의 목차 구조 고려

SOP 문서는 절차(Work Procedure) 외에도 다양한 섹션을 포함한다:

| 섹션 | 성격 | 현재 처리 |
|---|---|---|
| **Scope** | 문서 범위 정의 | scope_penalty로 패널티 |
| **Contents** | 목차 (페이지 참조 포함) | scope_penalty로 패널티 |
| **Safety / Safety Label** | 안전 주의사항 | 절차로 취급될 수 있음 |
| **사고 사례** | 사고 케이스 | 절차로 취급될 수 있음 |
| **환경 안전 보호구 Check Sheet** | 체크리스트 | 절차로 취급될 수 있음 |
| **Worker Location** | 작업자 위치 | 절차로 취급될 수 있음 |
| **Tool List** | 필요 도구/장비 목록 | 절차로 취급될 수 있음 |
| **Part 위치** | 부품 위치도 | 절차로 취급될 수 있음 |
| **Work List** | 작업 목록 (하위 절차 인덱스) | 절차로 취급될 수 있음 |
| **Work Procedure** | 실제 작업 절차 | procedure_boost로 부스트 |
| **Flow Chart** | 작업 흐름도 | procedure_boost로 부스트 |
| **Revision History** | 개정 이력 | scope_penalty로 패널티 |

사용자가 "tool list 보여줘", "check sheet 조회", "part 위치 알려줘" 등을 요청하면 해당 섹션을 직접 찾아 보여줘야 하는데, 현재는 이들이 절차 답변에 묻히거나 패널티를 받을 수 있다.

## 현재 코드 흐름 분석

### 1. task_mode 결정 (`langgraph_agent.py:1479-1516`)

```
route_node() 진입
  ├─ task_mode가 이미 설정? → 그대로 사용
  ├─ 미설정 → _infer_task_mode_from_doc_types(selected_doc_types)
  │   └─ SOP 계열 문서 → return "sop"   ← 여기서 강제
  ├─ task_mode == "ts" → route="ts" (라우터 건너뜀)
  ├─ task_mode == "sop" → LLM 라우터 실행
  │   └─ 라우터 결과: setup/ts/general 중 하나
  └─ 하지만 task_mode="sop"는 이후 MQ/답변에서 setup 경로 강제
```

**핵심 문제**: task_mode="sop"일 때 라우터가 "general"을 반환해도, MQ 생성과 답변 템플릿 선택에서는 `route` 값과 `task_mode` 값이 혼용되어 결국 setup 경로로 흐른다.

### 2. MQ 생성 (쿼리 확장)

- `task_mode="sop"` + `route="setup"` → `setup_mq` 사용
- `task_mode="sop"` + `route="general"` → `general_mq` 사용 **가능하지만**, section expansion과 답변에서 다시 setup 경로로 편향

### 3. 검색 결과 조정 (`langgraph_agent.py:2415-2490`)

```python
_PROCEDURE_KEYWORDS = {"교체", "절차", "작업", "방법", "replacement", "procedure", ...}
_PROCEDURE_CHAPTERS = {"work procedure", "flow chart", "work 절차"}
_SCOPE_MARKERS = {"scope", "contents", "목차", "table of contents", "revision history"}

# sop_only_predicate=True 이면:
# 1) 절차 키워드가 쿼리에 있으면 → Work Procedure 부스트
# 2) Scope/Contents 페이지 → 무조건 패널티
```

**문제점**: `sop_only_predicate`는 문서 타입 기반이므로, SOP를 선택한 순간 비절차 질문에도 scope 패널티가 적용된다.

### 4. 답변 생성 (`langgraph_agent.py:3620-3656`)

```python
templates = {
    "setup": spec.setup_ans,    # "## 작업 절차" + 번호 목록
    "ts": spec.ts_ans,          # 원인/진단/조치
    "general": spec.general_ans, # 유연한 형식
}
tmpl = templates.get(route, spec.general_ans)
```

route="general"이면 유연한 템플릿을 사용하므로, **라우터 판정이 올바르게 전파되면** 답변 형식 문제는 해결된다.

## Contracts To Preserve

- C-API-001 — 응답 메타데이터 키(`route`, `mq_mode` 등) 유지
- C-API-002 — Interrupt/resume `thread_id` 연속성
- C-API-003 — `retrieval_only=true` 동작

## Contracts To Update

- None (라우팅 내부 로직 변경이며 API 응답 shape는 변경 없음)

## 개선 방향 (설계 검토 필요)

### 방안 A: 의도 키워드 기반 오버라이드

`route_node`에서 라우터 호출 전/후에 조회 의도 키워드를 감지하여 `route`를 보정:

```
_INQUIRY_KEYWORDS = {"조회", "보여줘", "알려줘", "확인", "목록", "리스트",
                     "worksheet", "work sheet", "tool list", "check sheet",
                     "scope", "목차", "part 위치", "개요"}
```

- 라우터가 `setup`을 반환해도, 쿼리에 inquiry 키워드가 있으면 → `route="general"` 로 변경
- `sop_only_predicate` 하에서도 scope 패널티를 건너뜀

**장점**: 구현 간단, 기존 절차 질문에 영향 없음
**단점**: 키워드 기반이므로 "교체 절차 조회해줘"처럼 절차+조회가 섞인 경우 오판 가능

### 방안 B: 라우터 판정 우선 정책

`_infer_task_mode_from_doc_types()`의 결과를 `task_mode`의 "hint"로만 사용하고, 실제 파이프라인 경로는 LLM 라우터 판정(`route`)을 따르도록 변경:

- `task_mode="sop"` + `route="general"` → MQ는 `general_mq`, 답변은 `general_ans`, scope 패널티 미적용
- `task_mode="sop"` + `route="setup"` → 현행과 동일 (절차 경로)

**장점**: LLM의 의도 분류를 최대 활용, 키워드 관리 불필요
**단점**: 라우터 오판 시 절차 질문이 general로 빠질 수 있음

### 방안 C: 라우터 프롬프트 개선 + 4번째 route 추가

라우터에 `info` (정보 조회) route를 추가하여 4단계 분류:

- `setup`: 설치/교체/교정 절차
- `ts`: 트러블슈팅
- `info`: 문서 내 특정 섹션 조회 (worksheet, tool list, scope 등)
- `general`: 일반 설명/개요

**장점**: 가장 정확한 의도 분류
**단점**: 라우터 프롬프트 변경, 새 MQ/답변 템플릿 필요, 변경 범위 큼

### 권장: 방안 B (라우터 판정 우선) + 방안 A 일부 (안전망)

1. `route_node`에서 `task_mode="sop"`여도 라우터를 실행하고, `route` 값을 그대로 전파
2. MQ 선택, scope 패널티, 답변 템플릿을 `route` 기준으로 결정 (`task_mode`가 아닌)
3. 안전망으로 조회 의도 키워드가 명확한 경우 scope 패널티를 건너뛰는 로직 추가

## Allowed Files

- `backend/llm_infrastructure/llm/langgraph_agent.py` — route_node, retrieve_node, answer_node
- `backend/llm_infrastructure/llm/prompts/router_v1.yaml` — 라우터 프롬프트 (방안 C 시)
- `backend/config/settings.py` — 새 설정 플래그 (필요 시)
- `tests/` — 라우팅 관련 테스트

## Out Of Scope

- 라우터 프롬프트 전면 재작성
- issue 모드 로직 변경
- 프론트엔드 UI 변경
- 문서 인덱싱/청킹 로직 변경

## Risks

- 기존 절차 질문의 답변 품질 저하 (route 오판으로 general 경로로 빠지는 경우)
- scope 패널티 제거 시 불필요한 목차/개정이력이 상위에 노출
- MQ 템플릿 불일치로 검색 품질 저하

## Verification Plan

```bash
# 기존 계약 테스트
uv run pytest tests/api/test_agent_response_metadata_contract.py -v
uv run pytest tests/api/test_agent_interrupt_resume_regression.py -v
uv run pytest tests/api/test_agent_retrieval_only.py -v

# 수동 검증 시나리오
# 1. SOP 문서 선택 → "work sheet 조회해줘" → 절차가 아닌 워크시트 내용 반환 확인
# 2. SOP 문서 선택 → "tool list 보여줘" → tool list 섹션 내용 반환 확인
# 3. SOP 문서 선택 → "slot valve 교체 절차" → 기존과 동일한 절차 답변 확인
# 4. SOP 문서 선택 → "이 문서 scope 알려줘" → scope 섹션 내용 반환 확인
```

## Verification Results

(구현 후 기록)

## Handoff

- Current status: Draft — 설계 검토 필요
- Remaining TODOs:
  1. 방안 B+A 세부 설계 확정
  2. 라우터가 SOP 컨텍스트에서 general vs setup 판정 정확도 테스트
  3. scope 패널티 조건부 적용 로직 상세 설계
  4. 구현 및 테스트

## Change Log

- 2026-03-26: 태스크 생성, 현황 분석 및 개선 방향 초안

## Final Check

- [ ] Diff stayed inside allowed files, or this doc was updated first
- [ ] Protected contract IDs were re-checked
- [ ] Verification commands were run, or blockers were recorded
- [ ] Any contract changes were reflected in `product-contract.md`
- [ ] Remaining risks and follow-ups were documented
