# 챕터 단위 문서 그룹핑 검색 (Chapter-Aware Retrieval)

## 배경

현재 RAG 파이프라인은 페이지 단위 chunk로 검색한다.
SOP 문서의 "10. Work Procedure"가 6페이지(page 9~14)에 걸쳐 있을 때,
page 9 하나만 검색되면 나머지 5페이지가 누락되어 LLM에 불완전한 절차가 전달된다.

### 실제 예시: `global_sop_geneva_rep_bubbler_cabinet_fill_valve`

| Page | 챕터 |
|------|------|
| 1 | Contents (목차) |
| 2 | 1. Safety / 2. Safety Label |
| 3-4 | 3. 재해 방지 대책 |
| 5 | 4. 환경 안전 보호구 |
| 6 | 6. Flow Chart |
| 7 | 7. Part 위치 |
| 8 | 8. 필요 Tool |
| 9-14 | **10. Work Procedure** (6페이지) |
| 15 | 11. 작업 Check Sheet |
| 16 | 12. 환경 안전 보호구 Check Sheet |
| 17 | 13. Appendix / 14. Revision History |

## 제안: 방법 A — 인덱싱 시점에 section_chapter 메타데이터 추가

### 개요

chunk에 `section_chapter` 필드를 추가하여, 각 페이지가 TOC의 어느 챕터에 속하는지 기록한다.
검색 시 hit된 chunk의 `doc_id + section_chapter`로 동일 챕터의 나머지 페이지를 함께 가져온다.

### 현재 chunk 메타데이터

```json
{
  "chunk_id": "sop_..._fill_valve#0008",
  "doc_id": "global_sop_geneva_rep_bubbler_cabinet_fill_valve",
  "page": 9,
  "chapter": "Bubbler Cabinet Fill valve",   // <-- 문서 제목 수준 (topic)
  "doc_type": "sop",
  "device_name": "GENEVA",
  "chunk_version": "v3"
}
```

### 변경 후 chunk 메타데이터

```json
{
  "chunk_id": "sop_..._fill_valve#0008",
  "doc_id": "global_sop_geneva_rep_bubbler_cabinet_fill_valve",
  "page": 9,
  "chapter": "Bubbler Cabinet Fill valve",
  "section_chapter": "10. Work Procedure",     // <-- NEW: TOC 기반 섹션
  "section_number": 10,                        // <-- NEW: 정렬/필터용
  "doc_type": "sop",
  "device_name": "GENEVA",
  "chunk_version": "v3"
}
```

### 구현 단계

#### Phase 1: 챕터 경계 추출 (chunking 파이프라인)

1. **TOC 파싱**: 문서 앞부분(첫 5페이지)에서 `## Contents`/목차 섹션 탐지 후 항목 추출
   - 패턴: `N. 제목` (예: `10. Work Procedure`)
   - 대부분의 SOP가 동일한 목차 패턴을 따름

2. **페이지별 챕터 매핑**: 각 페이지의 `## N.` 헤더를 감지하여 챕터 할당
   - page의 첫 `## N.` 헤더가 해당 페이지의 section_chapter
   - 헤더가 없는 페이지는 이전 페이지의 section_chapter를 상속

3. **chunk JSONL에 section_chapter 필드 추가**

#### Phase 2: ES 인덱싱 스키마 변경

1. `chunk_v3_content` 인덱스에 `section_chapter` (keyword) 필드 추가
2. 재인덱싱 (390k docs)

#### Phase 3: 검색 시 챕터 확장 (expand)

1. 검색 결과에서 top-k chunk 수신
2. 각 hit의 `doc_id + section_chapter` 조합 수집
3. ES에서 해당 조합의 모든 chunk를 추가 조회 (page 순서 정렬)
4. 기존 top-k + 챕터 확장 chunk를 합쳐서 LLM에 전달

### 적용 범위 (VLM 파싱 결과 기반 실측)

| doc_type | 문서 수 | 평균 페이지 | TOC 존재율 | 번호 헤더 존재율 | 적용 가능성 |
|----------|---------|------------|-----------|----------------|-------------|
| **SOP** | 384 | 31p | **100%** | **100%** | **높음** — `## Contents` + `## N. Title` 패턴 전수 보유 |
| **SETUP** | 15 | 190p | **100%** | **100%** | **높음** — SOP와 동일 패턴. 대형 문서라 효과 극대화 |
| TS | 79 | 9p | 1% (숫자형) / 72% (알파벳 A.B.C.) | 3% (헤더 형태) | **낮음** — TOC는 알파벳이지만 내부 헤더가 거의 없음. 테이블 중심 구조 |
| PEMS | 8 | 11p | 25% | 25% | **낮음** — 자유 형식 보고서/프레젠테이션 |
| GCB | - | - | - | - | **미적용** — 자유 형식 |
| MyService | - | - | - | - | **미적용** — 자유 형식 |

#### TS 문서 특이 구조

- 첫 페이지에 알파벳 목차 보유 (72%): `- A. FFU Pressure range error` / `- B. FFU fan status error`
- 하지만 내부 페이지에서 `## A.` 같은 헤더는 3%만 존재
- 대부분 Failure symptoms → Check point → Key point **테이블** 형태로 섹션 구분 없이 이어짐
- 평균 9페이지로 짧아서 챕터 그룹핑 없이도 대부분 커버 가능
- TS는 챕터 확장 대신 **인접 페이지 윈도우 확장**(기존 방식)이 적합

#### PEMS 전용 전략

- PEMS는 이미지/도표 해석 비중이 높아 부분 chunk만으로는 문맥 손실이 큼
- 따라서 챕터 그룹핑 대신 **문서 단위 처리**를 적용
  1. PEMS 문서별 **상세 요약**(사전 생성/저장) 유지
  2. 검색 결과에 PEMS가 hit되면 해당 `doc_id`의 **문서 전체 페이지/청크 표시**
  3. LLM 입력은 전체 원문 대신 `상세 요약 + 핵심 페이지` 우선 전달
  4. 토큰 초과 시 `요약 우선` 정책으로 잘림 방지
- 적용 이유: 이미지/표 중심 문서에서 LLM 단독 판단 오류를 줄이기 위함

### 고려 사항

1. **토큰 사용량**: Work Procedure 6페이지를 모두 가져오면 토큰 증가
   - 완화: `max_chapter_pages` 제한 (예: 최대 8페이지)
   - 완화: 챕터 확장은 top-1~2 hit에만 적용

2. **TOC가 없는 문서**: section_chapter를 빈 문자열로 두고 확장 미적용
   - fallback: 기존 페이지 단위 검색 유지

3. **챕터 내 페이지 순서**: page 필드로 정렬하여 LLM에 순서대로 전달

4. **기존 `chapter` 필드와의 관계**:
   - 기존 `chapter`: 문서의 topic/제목 수준 (유지)
   - 신규 `section_chapter`: TOC 기반 섹션 (추가)

### Fallback 전략 (실행 규칙)

챕터 추출은 아래 우선순위로 수행한다.

1. **TOC 사전 생성 (1순위)**
   - 문서 앞부분(첫 5페이지)에서 `Contents/목차` 탐지
   - TOC 항목(`N. 제목`) 리스트를 먼저 확정

2. **헤더 직접 추출 (2순위)**
   - 페이지 상단(예: 첫 500~800자)에서 헤더 패턴 매칭
   - 허용 패턴 예:
     - `## 10. Work Procedure`
     - `10. Work Procedure`
     - `10) Work Procedure`
   - 추출된 헤더를 TOC 항목과 매칭해 `section_chapter` 확정

3. **TOC 기반 키워드 매칭 (3순위)**
   - 현재 페이지 텍스트와 TOC 제목 키워드를 매칭해 `section_chapter` 결정

4. **Carry-forward 상속 (4순위)**
   - 1/2순위 실패 시 이전 페이지의 `section_chapter` 상속
   - 이 경우 `chapter_source = "carry_forward"`로 기록

5. **번호 점프 안전장치 (오염 방지)**
   - TOC 기준으로 `1` 다음에 `3`이 먼저 감지되면, 누락된 `2` 구간 페이지는 추정하지 않고 `UNKNOWN` 처리
   - 즉, 챕터 번호가 점프하는 구간은 보수적으로 `UNKNOWN`으로 둔다

6. **검색 단계 fallback**
   - `chapter_source in {"title", "rule", "toc_match"}`일 때만 챕터 확장
   - `UNKNOWN`이거나 TOC 매칭 실패 시 기존 페이지 윈도우 확장 사용

7. **프롬프트 길이 fallback**
   - 챕터 확장 결과가 길면 `max_chapter_pages`로 제한
   - 초과 시 상위 관련 페이지 우선 전달(나머지는 제외)

### 검증 기준

- [ ] SOP 문서에서 section_chapter가 정확히 매핑되는지 확인 (샘플 10개)
- [ ] Work Procedure 관련 질문 시 해당 챕터의 모든 페이지가 LLM에 전달되는지 확인
- [ ] TOC가 없는 문서에서 기존 검색 동작이 깨지지 않는지 확인
- [ ] 토큰 사용량 증가가 허용 범위 내인지 확인

### 우선순위

현재 Agent 개선 (REQ-2, 3, 6) 작업 완료 후 진행.
3-model embedding eval 결과를 참고하여 최적 모델 결정 후, 재인덱싱과 함께 적용하는 것이 효율적.
