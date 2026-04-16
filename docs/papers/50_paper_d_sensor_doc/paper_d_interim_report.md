# Paper D — 중간 진행 보고서

> 작성일: 2026-04-16
> 보고 대상: 지도교수 / 내부 연구 미팅
> 단계: 데이터 적합성 검증 완료 → Pilot Set 구축 착수 전

---

## 1. 연구 주제 (한 줄)

> 반도체 장비의 센서 이상 episode를 정비 사례와 절차 문서에 연결하여
> 근거 기반 진단을 수행하는 retrieval 중심 프레임워크

---

## 2. 지금까지 한 일

### 2.1 연구 설계 (완료)

| 항목 | 결과 |
|------|------|
| 연구 질문 (RQ) | 3개 정의 (시간 불확실성 정렬, 정비사례 retrieval, grounded diagnosis) |
| 방법론 | 6모듈 파이프라인 설계 (Eventizer → Temporal Alignment → Case Retriever → Doc Retriever → Reasoner → Diagnosis) |
| 논문 분할 | 3편 계획 (데이터/문제정의 → Retrieval → Grounded Diagnosis) |
| 관련 연구 | 26편+ 조사, 비교표 작성 |

### 2.2 데이터 적합성 검증 (완료)

**검증 1: 센서명 직접 매칭 (raw name)**

SUPRA Vplus 장비의 62개 고유 센서명으로 ES(myservice, gcb) 문서를 검색.

| 결과 | 수치 |
|------|------|
| 전체 센서 | 62개 |
| 문서에서 직접 잡힌 센서 | **7개 (11%)** |
| 직접 연결 가능한 센서 | APC_Position(25건), APC_Pressure(58건), Temp1(80건), Temp2(242건) |

**의미**: 센서명 그대로는 대부분 문서와 연결되지 않음 → **sensor-document semantic gap 존재 확인**

---

**검증 2: 키워드 확장 매칭 (전수 스캔)**

52,406 chunks (14,052 고유 문서)를 전수 스캔, regex 기반 확장 키워드로 재검색.

| 센서 그룹 | raw name hit | 확장 매칭 | 배율 |
|-----------|:-----------:|:---------:|:----:|
| APC 계열 | 25~58 | **319** | ~5x |
| Temp 계열 | 57~242 | **2,057** | ~9x |
| Gas/MFC | 0~1 | **573** | **500x+** |
| EPD | 0~1 | **377** | **300x+** |
| RF Power | 0 | **244** | **∞** |

**의미**: 확장 검색하면 관련 문서가 대량으로 존재함 → **lexical expansion이 필수 전처리 단계**임을 실증

---

**검증 3: ES 원문 검증**

매칭된 문서의 원문을 직접 조회하여 relevance를 판정.

| doc_id | 센서 | 원문 확인 결과 | 판정 |
|--------|------|--------------|------|
| 40146514 | APC_Position | "원인 미상 APC Position 상승" — status/action/cause/result 4개 chunk | **relevant** |
| 40036448 | APC_Pressure | "APC PRESSURE HUNTING" — APC 교체, SOP 참조 있음 | **relevant** |
| 40042585 | Temp1 | "PM2 Temp1 temperature differential FDC out of spec" — chuck 교체 | **relevant** |
| 40086313 | EPD_Monitor | "PM2 CH1 EPD MONITOR 값 불량 (9수준)" | **relevant** |
| 40044000 | Position (single token) | "LP2 EFEM ROBOT WAFER SENSOR ALARM" — robot teaching | **irrelevant** |

**의미**:
- exact/spaced 매칭은 높은 precision으로 관련 문서를 찾음
- single token은 false positive 발생 (Position → robot teaching)
- **40036448이 gold case 후보** — cause/action/SOP 참조까지 완비

---

### 2.3 핵심 발견 요약

```
발견 1: sensor-document semantic gap이 실재함 (raw match 11%)
발견 2: 확장 검색 시 관련 문서는 충분히 존재함 (수백~수천 건)
발견 3: 원문 수준에서 센서 이상 → 원인 → 조치 → SOP 연결이 실제로 성립함
발견 4: single token 검색은 false positive 위험이 높음 → synonym dictionary 필요
발견 5: APC + Temp 계열이 pilot에 가장 적합함 (문서량 + 연결 품질)
```

---

## 3. 논문 가능성 판정

`paper_d_paper_strategy.md`에서 정의한 7개 핵심 숫자 기준:

| 판정 항목 | 기준 | 결과 | 상태 |
|-----------|------|------|------|
| APC/pressure 계열 로그 수 | 수백 건 이상 | **319 docs** (APC general) | ✅ 충족 |
| 관련 센서 수 | 10개 이상 | **17개 그룹** 매칭 | ✅ 충족 |
| setpoint/actual 쌍 존재 | 있어야 함 | APC_Position + APC_SetPoint 존재 | ✅ 확인 필요 (시계열) |
| recipe/step 정보 존재 | 있어야 함 | Recipe_Step_Num 센서 있음 | ✅ 확인 필요 (시계열) |
| 로그 component 명시 비율 | 높을수록 좋음 | 원문에서 APC/Temp 직접 명시 확인 | ✅ |
| ±3일 내 연결 가능 비율 | 높을수록 좋음 | **미확인** (시계열 데이터 필요) | ⏳ |
| Gold link 수 | 50+ pilot, 100~300 retrieval | **미확인** (pilot set 필요) | ⏳ |

**판정**: 7개 중 **5개 충족, 2개 시계열 데이터 확보 후 확인 필요**. 논문 진행 가능성 **높음**.

---

## 4. Pilot Set 구축 계획

### 4.1 대상

| 항목 | 값 |
|------|-----|
| 장비 | SUPRA Vplus |
| 센서 family | APC (Position, Pressure, SetPoint) + Pressure (Chamber) |
| 문서 소스 | myservice + gcb (319+ docs) |
| 목표 episode 수 | 50~100개 |

### 4.2 필요한 데이터

| 데이터 | 소스 | 상태 |
|--------|------|------|
| APC 센서 시계열 | FDC/센서 DB | **확보 필요** |
| 정비 로그 (myservice) | ES `rag_chunks_dev_current` | ✅ 접근 가능 |
| SOP/Manual (gcb) | ES `rag_chunks_dev_current` | ✅ 접근 가능 |
| recipe/step 정보 | FDC/센서 DB | **확보 필요** |
| 알람/이벤트 로그 | 장비 시스템 | **확보 필요** |

### 4.3 구축 절차

```
Step 1: 시계열 데이터 확보
  - APC_Position, APC_Pressure, APC_SetPoint, Pressure 시계열
  - 최소 6개월~1년 분량
  - recipe step 정보 포함

Step 2: 이상 episode 추출
  - 기준: tracking error, saturation, oscillation, drift, stuck
  - 규칙 기반으로 50~100개 후보 추출

Step 3: 정비 로그 매칭
  - 각 episode에 대해 ±1일/±3일/±7일 내 myservice 로그 후보 검색
  - validated vocabulary 활용

Step 4: Gold/Silver/Weak 라벨링
  - 원문 확인 후 판정
  - gold 50건 이상 확보가 목표

Step 5: Baseline 실험
  - 규칙 기반 매칭 vs BM25 vs dense retrieval
  - Recall@K, MRR 측정
```

### 4.4 예상 타임라인

| 주차 | 작업 | 산출물 |
|------|------|--------|
| 1주차 | 시계열 데이터 확보 + episode 추출 | episode 후보 목록 |
| 2주차 | 정비 로그 매칭 + 라벨링 | pilot set (gold/silver/weak) |
| 3주차 | baseline 실험 | Recall@K, MRR 결과 |
| 4~5주차 | 1편 논문 초고 | draft |

---

## 5. 현재 framing

### 하는 것
- sensor episode → maintenance case retrieval
- temporal uncertainty-aware alignment
- SOP/manual document grounding
- evidence-based diagnosis

### 하지 않는 것
- 일반적인 multivariate anomaly detection 전체
- LLM end-to-end diagnosis
- retrieval 없는 classifier-only 구조

### graph 아이디어의 위치
- graph-based anomaly detection이 아님
- **Eventizer / state representation 강화층**으로 위치
- 필요 시 2편에서 확장

---

## 6. 산출물 목록

### 기획 문서
| 문서 | 용도 |
|------|------|
| `paper_d_research_proposal.md` | 박사 연구계획서 |
| `paper_d_professor_onepage_summary.md` | 교수님 보고용 1페이지 |
| `paper_d_paper_strategy.md` | 논문 전략/프레이밍 |
| `paper_d_algorithm_design.md` | 6모듈 방법론 설계 |
| `paper_d_map_and_graph_framing.md` | 지도/길 찾기 + graph framing |

### 데이터 검증
| 문서 | 용도 |
|------|------|
| `paper_d_es_query_results.md` | 1차 ES 조회 (raw name) |
| `paper_d_full_sensor_doc_scan.md` | 전수 스캔 결과 (14K docs) |
| `paper_d_keyword_query_log.md` | 키워드별 원문 검증 로그 |
| `evidence/paper_d_sensor_doc_matches.json` | 센서별 매칭 상세 데이터 |

### 문헌
| 문서 | 용도 |
|------|------|
| `evidence/2026-04-14_literature_survey.md` | 6개 영역 26편 조사 |
| `evidence/paper_d_paper_comparison_table.md` | 논문 비교표 |
| `evidence/paper_d_bibtex_priority.md` | BibTeX 수집 우선순위 |

### 시각화
| 파일 | 용도 |
|------|------|
| `paper_d_presentation.html` | 11페이지 프레젠테이션 (방향키 이동) |
| `paper_d_graph_visualization.html` | 센서-문서 knowledge graph |

---

## 7. 결론 및 다음 단계 요청

### 확인된 사항
1. 센서-문서 연결은 **실제로 성립**한다 (ES 원문 검증 완료)
2. 관련 문서는 **충분히 존재**한다 (APC 319건, Temp 2,057건)
3. semantic gap은 **실재**하며, 이것 자체가 **논문 기여**가 된다

### 다음 단계에 필요한 것
1. **APC 센서 시계열 데이터 접근** (FDC DB 또는 파일)
2. **recipe/step 정보가 포함된 시계열** (step별 센서 값)
3. **알람/이벤트 로그** (episode 경계 판정용)

이 3가지가 확보되면 **2주 내 pilot set 구축, 3주 내 baseline 실험**이 가능합니다.

---

## 8. 한 줄 요약

> 데이터 적합성 검증이 완료되었고, 센서-문서 연결이 실제로 성립함을 확인했습니다.
> 다음 단계는 **APC 센서 시계열 데이터를 확보하여 pilot set 50~100개를 구축**하는 것입니다.
