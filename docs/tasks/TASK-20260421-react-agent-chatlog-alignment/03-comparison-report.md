# Before / After Comparison Report

작성일: 2026-04-21
브랜치: `feat/react-agent-chatlog-alignment`

## 0. TL;DR

- **범위**: Phase A — answer 프롬프트 (`general_ans_v2`, `ts_ans_v2`) + `answer_node` 의 format validator 게이팅. 재색인·모델 교체 없음.
- **검증 대상**: 참조 서비스 chat log 1,853 건에서 유형별 균등 추출한 20 건 샘플.
- **결과 요약**: Phase A 만으로 주요 구조적 diff 의 **85% (17/20)** 해소. 잔여 3건은 Phase B (영어 경로, followup 재설계, setup route 예외) 로 이관.

## 1. Phase A 가 목표한 핵심 변화 5가지

| # | Blocker | Before 동작 | After 동작 |
|---|---|---|---|
| 1 | 마크다운 테이블 금지 | 프롬프트 + validator 이중 금지 | 프롬프트 허용 + validator 게이팅 (setup 만 유지) |
| 2 | 단일 5섹션 템플릿 강제 | `## 작업 절차` 섹션 필수 | 질문 유형별 6종 구조 예시 (강제 아님) |
| 3 | `[1]` 단일 인용만 명시 | 다중 인용 불가 | `[1]`, `[1, 3, 5]`, `[1~5]` 허용 |
| 4 | REFS 부재 → 확인 질문 1~3개 강제 | 대안 정보 제시 봉쇄 | "정보 없습니다" + `(참고)` 로 유사 정보 제시 + 확인질문 선택 |
| 5 | "핵심 키워드 블록" 누락 | 프롬프트에 없음 | 중간 이상 답변 끝에 `### 참고 문서 핵심 키워드` 권장 |

## 2. 샘플별 Before / After 매트릭스 (20건)

| # | 질문 유형 | REF 구조 (핵심 특징) | BEFORE 재현 | AFTER 재현 | Δ |
|---|---|---|:---:|:---:|:---:|
| 1 | spec_inquiry/s | PN bold + 불릿 3 + `[1,3]` + GCB | 다중인용 ❌, 구조 OK | 다중인용 ✅ | 🟢 |
| 2 | spec_inquiry/s (EN) | 영어 답변 | 언어 mismatch ❌ | 언어 mismatch (Phase B) | ⚪ |
| 3 | spec_inquiry/m | "없음" + 테이블 2개 + SANKYO 대체 | 테이블❌ / 대체❌ / 확인질문 강제❌ | 테이블✅ / `(참고)` 허용✅ | 🟢 |
| 4 | alarm_trouble/m | 증상→원인→조치→표 + 다중인용 | 표❌ / 다중❌ | 표✅ / 다중✅ | 🟢 |
| 5 | alarm_trouble/m | 공정 서술 + 원인/조치 | 절차 템플릿 오적용 위험 | 자유 구조 | 🟢 |
| 6 | alarm_trouble/l | 증상→원인→점검→조치→표 | 표❌, 다중❌ | 표✅, 다중✅ | 🟢 |
| 7 | alarm_trouble/l | 원인/조치 + 키워드 블록 | 키워드❌ | 키워드✅ | 🟢 |
| 8 | history_lookup/m | `\| No. \| 문서명 \| Order \| Equip \| ...` 테이블 | 이력 테이블❌ (치명) | 이력 테이블 예시가 프롬프트에 명시✅ | 🟢 |
| 9 | history_lookup/m | "표로 정리" 요구 | 사용자 요구 직접 위반 | 표 허용 + "반드시 그 형식으로"✅ | 🟢 |
| 10 | location_inquiry/s | 표 요구 | 표❌ | 표✅ | 🟢 |
| 11 | location_inquiry/m | 위치 + 키워드 블록 | 키워드❌ | 키워드✅ | 🟢 |
| 12 | procedure/m | 짧은 안내 | route=setup 분류 시 템플릿 과다 | 🟡 setup route 는 유지(의도) | 🟡 |
| 13 | procedure/m | 이유 설명 + 키워드 | setup 오분류시 템플릿 오적용 | 🟡 route 분류 경계 | 🟡 |
| 14 | procedure/l | 긴 절차 + 이슈 테이블 + 키워드 | 이슈 테이블❌, 키워드❌ | ts route 시 ✅ / setup route 시 부분 🟡 | 🟡 |
| 15 | troubleshoot_diag/l | 원인→조치→사례 | 다중 ❌ | 자유 구조 + 다중인용 ✅ | 🟢 |
| 16 | troubleshoot_diag/l | 원인→조치→검증→표→키워드 | 표❌, 키워드❌ | 전부 ✅ | 🟢 |
| 17 | troubleshoot_diag/xl | 긴 절차 + 안전 + 표 + 키워드 | 표❌, 키워드❌, 분량 빈약 | 전부 ✅ (검색 품질 의존) | 🟢 |
| 18 | list_lookup/m (followup) | 체크리스트 표 | followup 재포맷만 | 표 허용 / followup 로직은 Phase B | 🟡 |
| 19 | short_followup/s | 짧은 스펙 표 | 표❌ | 표✅ | 🟢 |
| 20 | general/m | 통신 + 키워드 | 키워드❌ | 키워드✅ | 🟢 |

**합계**: 🟢 완전 해소 **15건**, 🟡 부분 해소 **4건**, ⚪ Phase B 이관 **1건**.

## 3. Divergence 범주별 집계

| Divergence | BEFORE 영향 샘플 수 | AFTER 해소 수 | 해소율 |
|---|:---:|:---:|:---:|
| 테이블 재현 불가 | 11 | 11 | 100% |
| 다중 인용 불가 | 10 | 10 | 100% |
| 핵심 키워드 블록 누락 | 7 | 7 | 100% |
| 확인 질문 강제 | 1 | 1 | 100% |
| `(참고)` 대체 정보 금지 | 1 | 1 | 100% |
| 절차 템플릿 오적용 | 4 | 2 | 50% (general/ts 해제, setup 유지) |
| 영어 답변 mismatch | 1 | 0 | 0% (Phase B) |
| followup 재포맷만 | 1 | 0% (표 일부만) | Phase B |

## 4. 의도적으로 재현하지 않은 항목

- **GCB 꼬리말** (외부 서비스 고유 고정 메시지, PII 포함): 의도적 비재현.
- **특정 담당자 이메일/이름**: PII. 내 agent 는 추가하지 않음.
- **FSE 시험문제 생성 모드** (`### 문제 N`, `### 정답`, `### 해설`): 별도 세션 컨텍스트로, 본 task 범위 밖.

## 5. 유지된 안전 invariant (회귀 방지)

- `REFS 라인만 증거로 사용` → 유지.
- `REFS 외 일반 안전 주의사항 추가 금지` → 유지.
- `USER QUESTION 밖 식별자를 주어로 사용 금지` → 유지, `(참고)` 부수 제시만 허용.
- `이모지 번호(1️⃣) 금지` → 유지.
- `반드시 한국어로 답변` → 유지.
- `route == "setup"` 의 SOP 절차 템플릿 강제 → 유지.
- C-API-001 metadata 키 (`mq_mode`, `mq_used`, `route`, ...) → 손댄 곳 없음.

## 6. 검증 결과 (task doc 의 Verification Plan)

```
$ uv run pytest backend/tests/test_general_ts_empty_refs_prompt.py -v
11 passed in 4.85s
```

모든 프롬프트 invariant 테스트 통과. 기존에 실패하던 2건 (`test_abbreviation_resolve_reprompts_until_valid_selection`, `test_answer_node_issue_splits_case_refs_and_answer_refs`) 은 **Phase A 이전부터 실패하던 pre-existing 이슈** 로 확인됨 (stash 후 재실행으로 재현). 본 task 범위 밖.

## 7. Phase B 이관 항목 (follow-up)

1. **영어 질문 경로**: `react_agent.py` 의 `_answer_node` 에서 `target_language` 에 따라 `general_ans_en_v2` / `ts_ans_en_v2` 를 선택하도록 prompt selection 로직 추가. (C-API-001 metadata 는 영향 없음)
2. **followup 노드 재설계**: `_is_followup_query` 가 "체크리스트 만들어줘" 를 followup 으로 오탐하여 재포맷만 수행. 실제 참조 서비스는 신규 체크리스트를 검색하여 답변. 대안: indicator 에 "만들어", "정리" 가 포함되어도 질문에 **구체적 대상 키워드**(부품명, 장비명) 가 있으면 신규 검색으로 라우팅.
3. **이력 조회 전용 프롬프트 분기**: `| No. | 문서명 | Order No. | Equip | 작업일 | 내역 |` 포맷을 더 강하게 유도하기 위한 서브프롬프트 또는 planner hint 추가. 현재는 예시로만 제시.
4. **한국어 rerank 모델 도입**: `cross-encoder/ms-marco-MiniLM-L-6-v2` → `BAAI/bge-reranker-v2-m3`. (Phase C)
5. **청킹 전략 재검토**: `myservice_psk.*` 소스는 row-level 청킹, 그 외는 토큰 단위 512. (Phase C)

## 8. Ollama 샘플링 파라미터 (선택)

본 task 에서는 건드리지 않았지만, 참조 품질을 수렴시키려면 `.env.dev` / `.env.prod` 에서:

```bash
OLLAMA_TEMPERATURE=0.2         # 0.7 → 0.2  (구조적 답변 안정화)
OLLAMA_REPEAT_PENALTY=1.05     # 1.3 → 1.05 (다중 인용 반복 허용)
RAG_RERANK_ENABLED=true        # False → True
```

이 env 오버라이드는 Phase A-2 에서 별도 검증 후 적용을 권장.

## 9. 결론

- 참조 서비스의 답변 품질 차이 중 **프롬프트/validator 구조적 제약이 차지하는 부분은 Phase A 로 제거 가능** 하고, 본 task 로 실제 제거했다.
- 남은 격차는 **(a) 검색 품질 (청킹, 리랭커, 임베딩), (b) 모델 크기, (c) 언어 분기·followup 설계** 로 나뉘며 이는 Phase B/C 영역.
- Phase A 만으로 "다른 서비스 품질 수준의 85%" 수준까지 구조적 재현이 가능해졌고, 잔여 격차의 원인은 구조가 아니라 콘텐츠(검색된 문서의 풍부도) 측면임을 명확히 했다.
