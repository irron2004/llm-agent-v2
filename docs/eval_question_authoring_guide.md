# Eval 질문 작성 가이드

> 반도체 에칭 장비 RAG 시스템의 retrieval 성능 평가를 위한 질문 작성 가이드.
> 이 가이드를 따라 ES 인덱스의 실제 문서를 읽고, 해당 문서를 정답으로 하는 질문을 생성한다.

## 1. 목적

- 검색 시스템이 질문에 대해 올바른 문서를 찾아오는지 평가 (hit@k, page_hit@k)
- 다양한 장비/문서유형/난이도를 커버하여 검색 품질을 종합적으로 측정

## 2. 데이터 소스

ES 인덱스 `chunk_v3_content`에 저장된 문서를 사용한다.

| doc_type | 설명 | 예시 |
|----------|------|------|
| `sop` | Global SOP (작업 절차서) | PM 교체, Robot Teaching, Chuck 교체 등 |
| `ts` | Trouble Shooting Guide | Microwave Abnormal, Particle Spec Out 등 |
| `setup` | 설비 설치 매뉴얼 | Cable Hook Up, Docking, Utility On 등 |
| `gcb` | GCB (Global Case Base) 이슈 보고서 | 장비별 장애 사례, 원인/대책 |
| `myservice` | myService 작업 이력 | 현장 작업 이력, 부품 교체 기록 |

## 3. 질문 작성 프로세스

### Step 1: 문서 탐색

ES에서 특정 doc_type의 문서 목록을 조회한다:

```python
from elasticsearch import Elasticsearch
es = Elasticsearch(['http://localhost:8002'])

resp = es.search(
    index='chunk_v3_content',
    body={
        'query': {'term': {'doc_type': 'sop'}},  # sop, ts, setup, gcb 등
        'size': 0,
        'aggs': {
            'docs': {
                'terms': {'field': 'doc_id', 'size': 50},
                'aggs': {
                    'sample': {
                        'top_hits': {
                            '_source': ['doc_id', 'content', 'page', 'device_name'],
                            'size': 1,
                            'sort': [{'page': 'asc'}]
                        }
                    }
                }
            }
        }
    }
)
```

### Step 2: 문서 내용 읽기

선택한 문서의 청크들을 페이지 순으로 읽는다:

```python
resp = es.search(
    index='chunk_v3_content',
    body={
        'query': {'term': {'doc_id': '<doc_id>'}},
        '_source': ['content', 'page', 'section_title'],
        'sort': [{'page': 'asc'}],
        'size': 20,
    }
)
```

### Step 3: 질문 생성

문서 내용을 읽고 아래 원칙에 따라 질문을 작성한다.

## 4. 질문 작성 원칙

### 4.1 페르소나

반도체 에칭 장비를 다루는 **현장 엔지니어** 관점으로 질문한다.
- 장비 앞에서 작업 중 모르는 절차를 물어보는 상황
- 알람이 발생해서 대응 방법을 찾는 상황
- 장비 셋업/PM 전에 준비물을 확인하는 상황
- 과거 유사 이슈 사례를 찾는 상황

### 4.2 질문 유형 (반드시 골고루 포함)

| 유형 | 설명 | 예시 |
|------|------|------|
| **절차 질문** | SOP/매뉴얼의 특정 작업 절차 | "SUPRA N TM Robot 교체 절차를 알려줘" |
| **트러블슈팅** | 알람/이상 현상 발생 시 대응 | "Microwave Reflect 발생 시 점검 포인트는?" |
| **스펙/수치** | 특정 파라미터 값, Part number | "Ring Seal Vac Spec은 몇 mmHg인가요?" |
| **안전/보호구** | 작업 시 안전 주의사항 | "Chuck 교체 시 필요한 보호구는?" |
| **이력/사례** | GCB/myService 기반 과거 사례 | "SUPRA Vm에서 Particle NG 발생 사례 알려줘" |
| **비교/차이** | 장비간/모델간 차이점 | "SUPRA N과 SUPRA XP의 APC 센서 차이는?" |

### 4.3 질문 품질 기준

**DO:**
- 문서 내용을 실제로 읽고, 해당 문서에 답이 있는 질문만 작성
- 자연스러운 현장 엔지니어 어투 사용 (격식체/비격식체 혼용 OK)
- 한국어 기본, 영어/일본어 질문도 일부 포함 (다국어 검색 평가)
- 질문 길이: 15자 ~ 120자 (너무 짧으면 검색 어려움, 너무 길면 비현실적)
- 장비명을 포함하되, 때로는 장비명 없이 일반적으로 물어보기도 함

**DON'T:**
- 문서에 없는 내용으로 질문 만들지 않기
- 동일한 문서에서 3개 이상 질문 만들지 않기 (편중 방지)
- "~에 대해 알려줘" 같은 너무 막연한 질문 지양
- 페이지 번호나 문서 ID를 질문에 직접 언급하지 않기

### 4.4 expected_doc / expected_pages 기록 규칙

| 필드 | 규칙 |
|------|------|
| `expected_doc` | ES `doc_id` 값 그대로 기록 (예: `global_sop_supra_n_series_all_tm_robot`) |
| `expected_pages` | 정답이 위치한 페이지 범위 (예: `6-8`, `2`, `3-6`) |

- 정답이 여러 페이지에 걸쳐 있으면 범위로 기록: `6-8`
- 단일 페이지면 숫자만: `2`
- 정답 문서를 특정할 수 없는 질문 (채팅 추출 등)은 빈 값으로 남김

## 5. 출력 CSV 포맷

```csv
qid,question,expected_doc,expected_pages
1,"SUPRA N TM Robot 교체 시 안전 주의사항은?",global_sop_supra_n_series_all_tm_robot,2
2,"GENEVA XP에서 Particle Spec Out 시 CASE 1 절차는?",ts_pdfs_pskh_ts_guide_geneva_xp_particle_spec_out,3-6
```

- 인코딩: UTF-8
- 구분자: 콤마 (CSV 표준)
- 질문에 콤마/줄바꿈이 포함되면 큰따옴표로 감싸기 (CSV 표준)
- 파일 위치: `data/eval_questions_from_chat.csv` (기존 파일에 append 또는 별도 파일)

## 6. 장비/문서 커버리지 체크리스트

질문 세트 전체에서 아래 항목이 골고루 포함되어야 한다:

### 장비 모델
- [ ] SUPRA N / Nm / Np
- [ ] SUPRA XP (ZEDIUS XP)
- [ ] SUPRA Vplus / Vm
- [ ] INTEGER Plus
- [ ] PRECIA
- [ ] GENEVA XP (STP300 XP)
- [ ] ECOLITE 3000 / II 300 / II 400

### 문서 유형
- [ ] SOP (절차서) — 최소 40%
- [ ] Trouble Shooting Guide — 최소 20%
- [ ] Setup Manual — 최소 10%
- [ ] GCB (이슈 보고서) — 최소 10%
- [ ] myService (작업 이력) — 최소 5%

### 질문 언어
- [ ] 한국어 — 최소 70%
- [ ] 영어 — 최소 15%
- [ ] 일본어 — 최소 5%

## 7. 검증

질문 작성 후 아래를 확인:

1. **expected_doc이 ES에 존재하는지**: `es.count(index='chunk_v3_content', body={'query': {'term': {'doc_id': '<doc_id>'}}})`
2. **expected_pages에 실제 정답이 있는지**: 해당 페이지 청크를 읽어서 답이 포함되어 있는지 확인
3. **중복 질문 없는지**: 기존 CSV의 질문과 의미적으로 겹치지 않는지 확인

## 8. 예시 (10개)

| qid | question | expected_doc | pages |
|-----|----------|-------------|-------|
| 240 | SUPRA N series 설비에서 TM Robot 교체 작업 시 안전 주의사항은 무엇인가요? | global_sop_supra_n_series_all_tm_robot | 2 |
| 241 | ZEDIUS XP 설비의 Hook Lifter Pin 교체에 필요한 보호구와 작업 절차를 알려줘 | global_sop_supra_xp_all_pm_pin_assy | 7-8 |
| 242 | INTEGER Plus 설비의 PM Floating Joint 조절 작업 시 필요한 Tool과 절차가 뭐야? | global_sop_integer_plus_all_pm_pin_motor | 8 |
| 243 | PRECIA 설비의 Chuck 교체 시 Worker Location 배치는 어떻게 해야 하나요? | global_sop_precia_all_pm_chuck | 5-6 |
| 244 | SUPRA N에서 Trace Microwave Abnormal 발생 시 Failure symptoms별 Check point와 Key point는? | supra_n_all_trouble_shooting_guide_trace_microwave_abnormal | 2-3 |
| 245 | GENEVA XP에서 IO Response Error (Ring Seal Vac error) 발생 시 Ring seal Spec과 확인 방법은? | ts_pdfs_pskh_ts_guide_geneva_xp_io_response_error_ring_seal_vac_error | 4-6 |
| 246 | SUPRA N 설비 설치 매뉴얼에서 Cable Hook Up 단계의 준비사항은 무엇인가요? | set_up_manual_supra_n | 2 |
| 247 | SUPRA N series 설비의 Manometer 교체 작업에 필요한 보호구와 작업 인원은? | global_sop_supra_n_series_all_sub_unit_manometer | 6-7 |
| 248 | INTEGER Plus EFEM Robot Leveling 작업 절차와 필요 Tool을 알려줘 | global_sop_integer_plus_all_efem_robot | 7-8 |
| 249 | GENEVA XP에서 Particle Spec Out 발생 시 CASE 1의 절차 (Chamber Open ~ Re-PM)는? | ts_pdfs_pskh_ts_guide_geneva_xp_particle_spec_out | 3-6 |
