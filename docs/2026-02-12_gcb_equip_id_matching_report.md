# GCB Equip_ID 매칭 분석 보고서

## 1. 목적

ES 인덱스 `rag_chunks_dev_v2`의 GCB 문서에 `Equip_ID` 필드를 추가하기 위해,
전처리된 GCB 텍스트 파일과 원본 크롤링 JSON 간의 매칭 가능성을 분석한다.

## 2. 데이터 소스

| 구분 | 경로 | 건수 |
|------|------|------|
| GCB 텍스트 파일 | `/home/llm-share/.../pe_preprocess_data/gcb/GCB_{number}.txt` | 6,114개 |
| 원본 크롤링 JSON | `/home/llm-share/.../gcb_raw/20260126/scraped_gcb.json` | 16,354 entries |
| ES 인덱스 | `rag_chunks_dev_v2` (doc_type=gcb) | 7,264 chunks |

### 2.1 텍스트 파일 구조

```
### GCB 344 | Status: Close | Model: TERA21 | Req:
**Title**: Bistel FDC OOC to Halt Chamber

**Question (원인/문의/초기 가설)**:
...
**Confirmed Resolution (최종 확정 조치)**:
...
**Tags**: model=TERA21; req=; os=N/A; patch=N/A; module=CHAMBER
```

- 파일명: `GCB_{GCB_number}.txt`
- 1번째 줄: 헤더 (GCB 번호, Status, Model)
- `**Title**:` 행에서 Title 
- 추출 가능

### 2.2 JSON 구조

```json
{
  "GCB_number": "344",
  "Status": "Close",
  "Title": "Bistel FDC OOC to Halt Chamber",
  "Model Name": "TERA21",
  "Content": "...",
  "Request_Item2": "N/A",
  "Equip_ID": null
}
```

- `GCB_number`: 문자열 타입 (범위: "100" ~ "68367")
- `Equip_ID`: 매칭 대상 필드 (null 또는 문자열)

## 3. 매칭 키 전략

### 3.1 GCB_number 단독 매칭의 한계

JSON에 동일 `GCB_number`가 여러 건 존재하는 **중복 문제**가 있다.

| 구분 | 건수 |
|------|------|
| JSON 전체 entries | 16,354 |
| 고유 GCB_number | 13,944 |
| 중복 GCB_number | 2,410 |

중복 GCB_number의 entries는 **완전히 다른 GCB 건**이다:

- Model Name이 다른 건: 2,305 / 2,410 (95.6%)
- Equip_ID가 다른 건: 1,942 / 2,410
- Title이 다른 건: 거의 전부

**예시 — GCB_number = 1001:**

| # | Model Name | Equip_ID | Title |
|---|-----------|----------|-------|
| 1 | TERA21i | null | Daily Updates on PICP Power Check... |
| 2 | GENEVA xp | 850-03 | ASEK P16101020 Scheduler malfunction |

→ 크롤링 시 페이지네이션/인덱싱 이슈로 하나의 GCB_number에 서로 다른 건이 매핑된 것으로 추정.

### 3.2 GCB_number + Title 복합 매칭

텍스트 파일의 `**Title**:` 행과 JSON의 `Title` 필드를 함께 비교하면 중복을 정확히 구분할 수 있다.

- 비교 방법: 소문자 변환 + 공백 정규화 후 포함(contains) 비교
- 중복 1,349건 중 **1,348건 매칭 성공** (실패 단 1건)

## 4. 매칭 결과

### 4.1 전체 매칭 현황

| 구분 | 건수 | 비율 |
|------|------|------|
| **매칭 성공** | **5,760** | **94.2%** |
| ├ 비중복 (GCB_number 유일) | 4,412 | |
| └ 중복 → Title로 구분 | 1,348 | |
| **매칭 실패** | **354** | **5.8%** |
| ├ JSON에 없음 | 353 | |
| └ 중복 + Title 불일치 | 1 | |

### 4.2 Equip_ID 현황 (매칭 성공 5,760건 기준)

| 구분 | 건수 | 비율 |
|------|------|------|
| 유효 Equip_ID 있음 | 3,958 | 68.7% |
| Equip_ID 없음/쓰레기 | 1,802 | 31.3% |
| 고유 Equip_ID 수 | 1,464 | — |

쓰레기 값으로 분류한 Equip_ID: `null`, `""`, `"-"`, `"."`, `"/"`, `"1"`, `"NA"`, `"N/A"`

## 5. 요약

- **GCB_number + Title** 복합 키로 6,114개 파일 중 **5,760개 (94.2%)** 를 JSON과 정확히 매칭 가능
- 매칭된 건 중 **3,958개 (68.7%)** 가 유효한 Equip_ID를 보유
- 미매칭 354건은 JSON에 해당 GCB_number 자체가 없음
- ES 업데이트 시 `doc_id` (예: `GCB_344`) → GCB_number 추출 → JSON Title 비교 → Equip_ID 획득 순서로 처리
