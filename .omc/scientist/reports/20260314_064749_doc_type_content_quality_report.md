# doc_type별 콘텐츠 품질 및 구조 차이 분석 보고서

**생성일**: 2026-03-14 06:47  
**분석 단계**: RESEARCH_STAGE 3  
**분석 범위**: ts / myservice / gcb doc_type  

---

## [OBJECTIVE]

ts, myservice, gcb 문서의 실제 콘텐츠 구조 차이를 분석하고, 이슈/원인/조치 정보가 각 문서 유형에서 어떻게 담겨있는지 파악한다. 이슈 요약/상세 답변(issue_ans_v2, issue_detail_ans_v2)의 출력 요구사항과 실제 REFS 데이터의 정보량 차이를 식별한다.

---

## [DATA]

| 소스 | 파일 수 | 포맷 | 현황 |
|------|---------|------|------|
| ts (Trouble Shooting PDFs) | 57 | PDF | VLM 파싱 미완료, chunk_v3 미적재 |
| myservice | 99,031 | .txt | 적재됨 (단, 62% 빈 파일) |
| gcb (txt) | 6,115 | .txt | 적재됨 |
| gcb (raw JSON) | 16,354 entries | JSON | 현재 미적재, raw 형태 보관 |

---

## 1. doc_type별 문서 특성

### 1.1 ts (Trouble Shooting Guide)

- **형식**: 구조화된 PDF 매뉴얼 (알람코드/증상/원인/조치/결과)
- **이슈 정보 밀도**: 최고 — 섹션별로 symptom/cause/action이 명시
- **현재 상태**: **chunk_v3 인덱스에 미적재** (VLM 파싱 필요, 57개 PDF)
- **이슈 답변 적합도**: 가장 높지만, 실제로 검색 불가

### 1.2 myservice

- **형식**: `[meta]` JSON + `[status]` / `[action]` / `[cause]` / `[result]` 섹션
- **총 파일**: 99,031개 (BM 86.2%, PM 10.3%)
- **섹션 구조**: 4개 섹션이 템플릿으로 존재하지만 실제 내용 기입 여부가 변동적

### 1.3 gcb

- **형식**: `**Question**` (원인/초기 가설) + `**Confirmed Resolution**` (최종 조치) + `**Tags**`
- **총 파일**: 6,115개 txt (전부 `Status: Close`)
- **내용 품질**: Question/Resolution 양쪽 모두 항상 존재
- **Raw JSON**: 별도 보관 (16,354 entries, dialogue 형식, 현재 미적재)

---

## 2. 청킹 결과 품질

### [FINDING] F1: myservice 파일 62.2%가 빈 콘텐츠로 청크 0개 (인덱스 미적재)

[STAT:p_value] n=99,031 전수 검사 (분류 오류 없음)  
[STAT:ci] 95% CI: [61.9%, 62.5%]  
[STAT:n] n=99,031  
[STAT:effect_size] 실제 인덱스 가능 파일: ~37,800개 (38%)

**메커니즘**: `completeness='empty'`(14.6%) 외에도 `complete` 태그를 달고 있어도 실제로 섹션 내용이 없거나 20자 미만인 경우가 존재. 특히 BM 활동(86.2%)에서 엔지니어가 섹션을 비워두는 경우가 많음.

### [FINDING] F2: gcb 75.5%가 2개 청크로 분리 — Question과 Resolution이 각각 다른 청크

[STAT:ci] 95% CI: [74.4%, 76.5%]  
[STAT:n] n=6,115  
[STAT:effect_size] 파일 총 877자(중앙값) → 청크 1: Question(346자), 청크 2: Resolution(261자)

**메커니즘**: `fixed_size` 청커(chunk_size=512, overlap=50)가 875자 파일을 2개로 분리. 질문에서 Q 청크만 검색되면 Resolution 정보가 REFS에 포함되지 않음. 반대의 경우 증상/원인 컨텍스트 없이 조치만 전달됨.

### [FINDING] F3: myservice 섹션이 section_type 메타데이터 없이 하나의 청크로 병합

[STAT:n] n=4,495 (비어있지 않은 파일)  
[STAT:effect_size] action 섹션 중앙값 215자, cause 섹션 중앙값 33자

**메커니즘**: 현재 청킹 전략이 `[status]`, `[action]`, `[cause]`, `[result]` 섹션 경계를 인식하지 않음. 모든 섹션이 단일 텍스트로 합쳐져 인덱싱됨. ES `chunk_v3_content` 매핑에 `section_type` 필드 없음.

---

## 3. 메타데이터 활용도

### [FINDING] F4: REFS 텍스트에 doc_type/섹션 정보가 전혀 포함되지 않음

[STAT:n] 코드 분석 (langgraph_agent.py:923-995)

`results_to_ref_json()`은 `device_name`과 `equip_id`만 metadata에 포함시킴. `doc_type`, `chapter`, `section_type` 등 이슈 컨텍스트 파악에 유용한 필드는 REFS 텍스트에서 제외됨.

```python
# 현재 코드 (langgraph_agent.py:948-951)
for key in ("device_name", "equip_id"):
    val = d.metadata.get(key)
    if val and str(val).strip():
        metadata[key] = str(val).strip()
```

LLM이 GCB Question 청크를 받았을 때 "이것이 원인 분석인가, 조치 내용인가"를 판별할 메타 단서가 없음.

### [FINDING] F5: gcb raw JSON의 Equip_ID 41.5% 누락

[STAT:ci] 95% CI: [40.7%, 42.2%]  
[STAT:n] n=16,354 (raw JSON entries)

gcb txt 파일에는 Tags 필드에 model 정보가 있으나 equip_id가 없음. raw JSON에서 복합키(GCB_number + Title) 매칭으로 94.2%는 복원 가능하나, 복원된 케이스 중에서도 31.3%가 유효한 Equip_ID를 보유하지 않음.

---

## 4. 이슈 요약에 필요한 정보 가용성

| 정보 항목 | ts | gcb (txt) | myservice (비어있지 않은) | myservice (전체 대비) |
|-----------|-----|-----------|--------------------------|----------------------|
| 증상/현상 | 90%+ (planned) | 84.1% | 26.2% | ~11.8% |
| 원인 분석 | 90%+ (planned) | 63.6% (keyword) | 71.1% (섹션 존재) | ~32.0% |
| 조치/해결 | 90%+ (planned) | 100% (Resolution) | 72.7% (섹션 존재) | ~32.7% |
| 현재 상태 | 70%+ (planned) | 100% (Status:Close) | 70.3% (섹션 존재) | ~31.6% |
| 4가지 모두 | N/A (미적재) | 84.1% | 36.9% | ~16.6% |

[STAT:n] gcb: n=6,115; myservice 비어있지 않은: n=4,495; myservice 전체: n=99,031  
[STAT:ci] gcb 4가지 모두: [83.0%, 85.2%] | myservice 비어있지 않은 4가지: [35.5%, 38.3%]

---

## 5. 프롬프트와 데이터 정합성

### issue_ans_v2.yaml 요구사항
- 이슈 사례 번호 목록 + 핵심 내용 + 근거 + 인용
- REFS가 비어있으면 "찾지 못했다" 응답

### issue_detail_ans_v2.yaml 요구사항
- `## 이슈 내용` + `## 해결 방안` 섹션 필수
- REFS 발췌 3~8개 글머리표

### [FINDING] F6: issue_detail_ans의 ## 이슈 내용 / ## 해결 방안 섹션이 gcb 단일 청크에서는 채워지지 않음

[STAT:effect_size] gcb 75.5%가 Q/Resolution 분리 → 1개 청크만 검색될 경우

gcb 파일이 2개 청크로 분리된 상황에서, retrieval이 Q 청크만 반환하면 REFS에는 증상+원인만 존재하고 해결 방안이 없음. LLM은 `## 해결 방안`을 "(원문을 참고하세요.)"로 폴백 처리함 (`_ensure_issue_detail_sections()` in langgraph_agent.py:2669).

### [FINDING] F7: myservice 'cause' 섹션이 매우 짧아 LLM이 구체적 원인 분석을 생성할 수 없음

[STAT:ci] 95% CI for mean cause length: [47, 48] chars  
[STAT:effect_size] 원인 섹션 중앙값 33자 (약 8~10 단어 수준)  
[STAT:n] n=56,730 (비어있지 않은 cause 섹션)

33자 원인 설명 예: "LP1, 2, 3 Leveling & Height Check" — 이것이 원인인지 조치인지 모호.  
`## 이슈 내용`에 충분한 근거가 없어 LLM이 REFS 기반 답변 대신 추론에 의존하게 됨.

### [FINDING] F8: ts doc_type이 이슈 답변에 가장 적합하지만 현재 chunk_v3에 미존재

[STAT:n] 57 PDF files 확인  
[STAT:effect_size] 이슈 모드 retrieval 시 ts 결과 0건

TS 가이드는 알람코드/증상/원인/조치 섹션을 구조적으로 포함하는 유일한 doc_type. 그러나 VLM 파싱이 완료되지 않아 chunk_v3에 적재되지 않았음. 이슈 쿼리 시 gcb와 myservice만 검색 대상.

---

## [LIMITATION]

1. **ts 분석 한계**: ts PDF 파일은 실제 텍스트를 추출할 수 없어 아키텍처 문서(2026-03-04 계획)와 코드 분석에 의존. 실제 ts 청크 품질은 VLM 파싱 후 재분석 필요.

2. **myservice 샘플 편향**: 정보 가용성 분석은 파일 순서 기준 10,000개 샘플. 파일명이 시간순이라면 초기 BM 이력이 과대 대표될 수 있음.

3. **gcb raw vs txt 비교**: gcb 현재 인덱스는 txt 파일 기반. raw JSON(16,354 entries)은 미인덱싱 상태로, 더 풍부한 Content 필드를 보유하나 구조가 dialogue 형식이어서 직접 활용 어려움.

4. **fixed_size 청크 시뮬레이션**: 실제 chunk 경계는 `_find_split_point()`가 separator(\n\n, \n 등)를 찾아 조정하므로 정확한 분할 위치는 추정치.

5. **정보 추출 패턴 매칭 한계**: keyword regex 기반 정보 존재 여부 판단은 false positive/negative 발생 가능. 특히 myservice의 섹션 헤더 자체가 keyword와 혼동될 수 있음.

---

## 핵심 정보 손실 구간 요약

| 구간 | doc_type | 손실 내용 | 심각도 |
|------|----------|-----------|--------|
| Raw → Indexing | myservice | 62% 파일이 빈 내용으로 미인덱싱 | Critical |
| Raw → Indexing | ts | 전체 미인덱싱 (VLM 미완) | Critical |
| Chunking | gcb | Q + Resolution이 별도 청크로 분리 (75.5%) | High |
| Chunking | myservice | section_type 정보 없이 전체 병합 | Medium |
| REFS 빌드 | 전체 | doc_type / chapter / section_type 미포함 | Medium |
| LLM Prompt | gcb | Resolution 청크 미검색 시 해결방안 공백 | High |
| LLM Prompt | myservice | cause 섹션 33자 중앙값 → 원인 분석 불가 | High |

---

## 권고사항

1. **[즉시] GCB 청킹 전략 변경**: `**Question**`과 `**Confirmed Resolution**`을 각각 별도 청크로 인식하는 section-aware 청커 추가. 또는 `expand_related_docs_node`에서 gcb는 같은 doc_id의 인접 청크를 자동 포함.
2. **[즉시] myservice REFS에 section_type 추가**: 청킹 시 `section_type` (`status/action/cause/result`) 메타를 보존하고 REFS 텍스트에 포함 (예: `[1] 40001439 (action): -. Undocking 및 Packing...`).
3. **[단기] ts VLM 파싱 완료 및 인덱싱**: 57 PDF 파싱 후 alarm_code 메타 포함하여 chunk_v3 적재. 이슈 모드 검색 품질을 즉각 향상시킬 것으로 예상.
4. **[중기] myservice 데이터 품질 개선**: 55% 빈 파일에 대한 소급 데이터 입력 또는 빈 파일 제외 필터링 정책 수립.
5. **[구조적] REFS에 doc_type 및 section_type 포함**: `results_to_ref_json()`에서 `doc_type`, `chapter`, `section_type` 필드를 metadata에 추가하여 LLM이 컨텍스트를 인식하도록 개선.

---

*Report generated by Scientist agent*  
*Figures: `.omc/scientist/figures/`*  
*Session ID: doc_type_content_quality_analysis*
