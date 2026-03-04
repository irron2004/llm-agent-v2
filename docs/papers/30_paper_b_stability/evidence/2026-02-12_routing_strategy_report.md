# 라우팅(Route) 전략 보고서

**작성일**: 2026-02-12
**목적**: ES 데이터 특성 분석을 기반으로, 사용자 질문을 어떤 카테고리로 분류하면 검색 품질이 최적화되는지 전략 도출

---

## 1. 현재 라우터 구조

### 1.1 현재 분류 체계 (router_v1.yaml)

```
setup  → 설치, 교체, 분해, 조립, 초기화, 캘리브레이션, 파라미터 설정, 케이블 연결, 셋업 절차
ts     → 알람/에러/경고/인터록/이상 증상, 원인 분석/트러블슈팅/로그 분석/예방
general → 위 둘에 해당하지 않는 일반 질문
```

**우선순위**: ts > setup > general

### 1.2 라우팅의 역할

현재 라우팅은 **MQ 생성 프롬프트 선택**에만 영향을 줌:
- `setup` → `setup_mq_v1.yaml`로 MQ 생성
- `ts` → `ts_mq_v1.yaml`로 MQ 생성
- `general` → `general_mq_v1.yaml`로 MQ 생성

검색 대상 doc_type 필터링에는 영향을 주지 않음 (모든 doc_type 대상으로 검색).

---

## 2. ES 데이터 현황

### 2.1 문서 규모

| doc_type | chunks | 비중 | unique docs | 언어 |
|---|---|---|---|---|
| myservice | 324,482 | 93.8% | ~81,000 | ko (한영 혼용) |
| SOP | 11,452 | 3.3% | ~339 | ko |
| gcb | 7,264 | 2.1% | 3,632 | **en** |
| set_up_manual | 2,045 | 0.6% | ~12 | ko |
| trouble_shooting_guide | 688 | 0.2% | ~77 | ko |

### 2.2 문서 구조 비교

#### myservice (정비 이력 기록)
```
doc_id: 40015778 (8자리 숫자)
├── status  : "Run中 TM Robot Communication Alarm, Controller Error Code: 6"
├── cause   : "TM Robot Controller Reset → Servo Power On Fail"
├── action  : "TM Robot Controller Reset → Servo Power On → GUI Home OK"
└── result  : "Buzz5000 Alarm Check → Realtime Error Exists"
```
- **4개 chapter**: status(78,933) / cause(81,510) / action(81,326) / result(82,713)
- status/cause는 대부분 50자 미만 단문
- action은 median 176자, 절차 기술

#### SOP (표준 작업 절차서)
```
doc_id: global_sop_supra_n_all_pm_fcip_r3   ← doc_id에 장비명+부품명 포함
chapter: "2. Safety Label"                    ← 번호 기반 절차 구조
content: 절차 표(Flow / Procedure / Tool & Point), 마크다운 형식, 400-700자/chunk
```
- 대표 chapter: "2. Safety Label"(10,466), "0. Safety"(455)
- doc_id 패턴: `global_sop_{장비}_{대상부품}` 또는 `set_up_manual_{장비}`

#### GCB (Global Case Book - 글로벌 기술 문의/해결)
```
doc_id: GCB_10004
├── question   : "After upgrading ATIS software from 2.11 to 2.50.14..."  (영문)
└── resolution : "The issue was identified as a result of..."             (영문)
```
- **전부 영문**, Q&A 쌍 (문서당 정확히 2 chunks)
- 소프트웨어 패치, 부품 번호 확인, 장비 호환성 등 다양한 주제

#### set_up_manual (셋업 매뉴얼)
```
doc_id: set_up_manual_supra_vm
chapter: "2. Module Unpacking & Moving"       ← 설치 순서별 chapter
content: 단계별 설치 절차, 체크리스트, 도면 참조
```
- 대표 chapter: "0. Safety"(991), "1. Template Draw"(316), "2. Tool Fab in"(239)
- 키워드: EFEM, Tool Check, Working Check, Pendent, TM Robot, Servo, Teaching

#### trouble_shooting_guide (트러블슈팅 가이드)
```
doc_id: ts_pdfs_pskh_ts_guide_geneva_xp_pm_wafers_exceed_time_limit  ← 알람명 포함
chapter: "[Trouble Shooting Guide]"
content: "Alarm Description: The wafer stay longer than setting time..."
```
- doc_id에 **알람명/증상명**이 직접 포함
- 특정 알람별 진단 절차, CASE별 대응 방법
- GENEVA XP(268), SUPRA N(184)에 집중

### 2.3 doc_type별 핵심 키워드

| doc_type | 상위 키워드 |
|---|---|
| myservice status | PM1, PM2, PM3, FCIP, CH1, CH2, **alarm**, **power drop**, LP1, **Leak** |
| myservice cause | **root cause**, customer request, FCIP, **power drop**, **STD alarm**, failure |
| myservice action | **leak check**, backup, **FCIP replacement**, aging test, **calibration**, cleaning |
| myservice result | complete, monitoring, SOP, T/S Guide, tool release, pending |
| SOP | Procedure, EFEM, PM, Check, 보호구, assembly, check sheet, Safety |
| gcb | replacement, monitoring, **alarm**, **part number**, **software patch**, installation |
| set_up_manual | EFEM, **Teaching**, TM Robot, **Servo**, Position Arrived, Controller ROM |
| trouble_shooting_guide | I/O list, **Error code table**, GFP, EFEM, replacement |

### 2.4 장비 분포

| doc_type | 상위 장비 |
|---|---|
| myservice | SUPRA N(133K), SUPRA Vplus(50K), SUPRA V(24K), SUPRA XP(18K) |
| SOP | INTEGER plus(2,865), SUPRA N(2,134), PRECIA(1,832), ZEDIUS XP(1,541) |
| gcb | SUPRA Vplus(1,588), SUPRA N(1,472), SUPRA Vm(452), TERA21(410) |
| set_up_manual | SUPRA Vm(316), ECOLITE3000(284), SUPRA Np(266), SUPRA Nm(239) |
| ts_guide | GENEVA XP(268), SUPRA N(184), SUPRA XP(91), PRECIA(70) |

---

## 3. 현재 라우팅의 문제점

### 3.1 myservice는 ts와 setup 모두에 해당

myservice 한 문서 안에 ts 관련 정보(status/cause)와 setup 관련 정보(action)가 공존:

- **status chapter**: alarm(1,711건), power drop(1,669건), Leak(1,243건) → **ts 성격**
- **action chapter**: FCIP replacement(4,126건), calibration(3,484건), cleaning(2,002건) → **setup 성격**
- **cause chapter**: root cause(3,768건), failure(917건) → **ts 성격**

따라서 "FCIP 교체 방법"이라는 질문이 `setup`으로 분류되더라도, myservice의 action chapter에서 실제 교체 사례를 찾는 것이 가장 유용함.

### 3.2 현재 3분류의 한계

| 사용자 질문 유형 | 현재 분류 | 실제 유용한 문서 |
|---|---|---|
| "FCIP alarm 원인" | ts | myservice(status/cause) + ts_guide + gcb |
| "FCIP 교체 절차" | setup | **SOP** + set_up_manual + myservice(action) |
| "FCIP 교체 후 power cal" | setup | **SOP** + myservice(action) |
| "Error Code 252 의미" | ts | myservice(status) + ts_guide |
| "Leak Rate 기준이 뭐야?" | general? setup? | SOP + ts_guide |
| "SUPRA N에서 자주 발생하는 알람" | ts | myservice(status) - 통계적 질문 |
| "ATIS 소프트웨어 업데이트 이슈" | general? ts? | **gcb** (영문 Q&A) |
| "EFEM Robot teaching 절차" | setup | set_up_manual + SOP |

**문제**: 현재 3분류는 "어떤 MQ를 생성할지"만 결정하고, 질문에 가장 적합한 **문서 유형**을 특정하지 못함.

### 3.3 GCB 영문 데이터의 사각지대

현재 라우팅에서 GCB에 대한 고려가 없음. GCB는 전부 영문이므로:
- 한국어 질문 → 한국어 MQ 생성 → GCB 검색 불가
- GCB에는 ts/setup 모두에 해당하는 사례가 혼재

---

## 4. 라우팅 전략 제안

### 4.1 전략 A: 의도 기반 분류 (현재 3분류 개선)

현재 3분류를 유지하되, 각 분류의 **정의를 데이터 특성에 맞게 재정의**:

```yaml
# 제안 분류 정의

ts (증상/원인 분석):
  - 사용자가 "왜 이런 현상이 발생하는지" 알고 싶을 때
  - 키워드 신호: alarm, error, fail, abnormal, 원인, root cause, leak, interlock, warning
  - 주 검색 대상: myservice(status/cause), trouble_shooting_guide, gcb(question)
  - MQ 전략: 에러코드/알람명을 정확히 보존, 짧은 키워드 쿼리

setup (절차/방법):
  - 사용자가 "어떻게 하는지" 절차를 알고 싶을 때
  - 키워드 신호: 교체, replacement, 설치, install, calibration, 절차, procedure, 방법,
                  setting, parameter, teaching, 셋업, bring-up, disassembly
  - 주 검색 대상: SOP, set_up_manual, myservice(action)
  - MQ 전략: 부품명 + 작업동사 조합

general (일반/복합 질문):
  - 위 둘에 명확히 해당하지 않는 질문
  - 장비 일반 정보, 스펙 비교, 정책 문의 등
  - 주 검색 대상: 전체 (필터 없음)
```

**개선 포인트**:
- 분류 기준이 "문서 유형"이 아닌 **"사용자가 원하는 정보의 성격"**에 맞춰짐
- MQ 프롬프트에서 각 분류별 최적 검색 전략을 적용

### 4.2 전략 B: 5분류 (doc_type-aware 라우팅)

doc_type 특성을 반영한 5분류:

```yaml
ts_symptom:       # "이 알람/에러가 뭐야?" → myservice(status) + ts_guide
ts_cause:         # "왜 이런 문제가 생겨?" → myservice(cause) + ts_guide + gcb
procedure:        # "어떻게 교체/설치해?" → SOP + set_up_manual
maintenance_log:  # "이전에 어떻게 조치했어?" → myservice(action/result)
general:          # 위에 해당 안 되는 일반 질문
```

**장점**: 각 분류가 1-2개 doc_type에 밀접하게 대응
**단점**: 분류가 세밀해져 LLM 분류 정확도 하락 가능, 복합 질문 처리 어려움

### 4.3 전략 C: 2축 분류 (의도 × 문서유형)

의도(intent)와 문서유형 힌트(doc_hint)를 별도로 출력:

```yaml
# 출력 형식
intent: ts | setup | general
doc_hint: procedure | case_history | reference | any

# 조합 예시
"FCIP alarm 원인"        → intent=ts,    doc_hint=case_history  (myservice/gcb 우선)
"FCIP 교체 절차"         → intent=setup, doc_hint=procedure     (SOP/set_up_manual 우선)
"이전에 FCIP 교체 어떻게 했어?" → intent=setup, doc_hint=case_history (myservice action 우선)
"Error Code 252 뭐야?"   → intent=ts,    doc_hint=reference     (ts_guide 우선)
```

**장점**: 의도와 문서유형을 분리해서 MQ 전략과 검색 필터를 독립적으로 최적화
**단점**: LLM이 2개 값을 동시에 출력해야 하므로 파싱 복잡도 증가

### 4.4 제안: 전략 A (개선된 3분류) + 보조 신호

**현실적 최적안**: 기존 3분류를 유지하면서, 라우터가 **보조 신호**를 함께 출력하도록 확장.

```yaml
# 출력 형식 (한 줄)
ts          # 기본 케이스
setup+sop   # SOP 절차서가 핵심일 때
ts+gcb      # 영문 GCB 참조가 필요할 때
setup       # 일반 setup
general     # 일반

# 보조 신호가 MQ 전략에 미치는 영향
+sop   → doc_id 패턴 "global_sop_"에 맞는 키워드 조합 생성
+gcb   → 영문 MQ를 반드시 1개 이상 포함
+log   → myservice action/result chapter 키워드 조합 생성
```

---

## 5. 프롬프트 개선안

### 5.1 router_v1 → router_v2 제안

```yaml
name: router
version: v2
description: Route user intent to setup / ts / general with optional doc-type hints.
system: |
  # Role
  You are a routing agent for semiconductor equipment Q&A.

  # Input
  - {sys.query}: User's question (may be in Korean, English, or mixed)

  # Output
  Output exactly ONE label. Choose from:
    ts
    setup
    general

  NO explanation, quotes, code blocks, or periods.

  # How to classify

  ## ts (troubleshooting / symptom / cause analysis)
  The user wants to know WHY something happened or WHAT is wrong.
  Signals:
  - alarm, error, fail, warning, interlock, abnormal
  - 원인, root cause, 문제, 증상, 이상, 발생, 재발
  - specific error codes (e.g., "Code 252", "ALID 420")
  - specific alarm names (e.g., "TM Robot Communication Alarm")
  - "왜", "why", "원인이 뭐야", "어떤 에러"

  ## setup (procedure / how-to / installation / calibration)
  The user wants to know HOW to do something.
  Signals:
  - 교체, 설치, 분해, 조립, replacement, install, disassembly
  - calibration, 캘리브레이션, 파워캘, power cal
  - procedure, 절차, 방법, 순서, 단계, step
  - teaching, parameter, recipe, setting, configuration
  - bring-up, setup, 셋업, 연결, hook-up, cable
  - checklist, 체크리스트

  ## general
  Does not clearly fit ts or setup:
  - General explanations, specifications, comparisons
  - "뭐야", "what is", overview, 설명
  - Part number inquiries, availability

  # Priority
  - If the question contains BOTH symptom AND procedure → choose ts
  - If ambiguous between setup and general → choose setup
messages:
  - role: user
    content: "{sys.query}"
```

### 5.2 핵심 변경 사항

| 항목 | 현재 (v1) | 제안 (v2) |
|---|---|---|
| ts 정의 | 추상적 ("alarm/error/warning") | **구체적 신호어 제시** (에러코드, 알람명, "왜") |
| setup 정의 | 추상적 ("installation, replacement") | **실제 데이터 키워드 반영** (calibration, teaching, power cal) |
| general 정의 | "나머지" | **적극적 신호어 제시** ("뭐야", spec, 설명) |
| 우선순위 | ts > setup > general | ts > setup > general 유지 + **ambiguous → setup** 규칙 추가 |
| 언어 고려 | 없음 | 한/영 혼용 입력 명시 |
| 예시 | 없음 | 분류별 신호 키워드를 프롬프트에 나열 |

---

## 6. 라우팅 결과가 MQ에 미치는 영향 (연동 전략)

### 6.1 route별 MQ 최적화 방향

| route | MQ 핵심 전략 | 이유 |
|---|---|---|
| **ts** | 에러코드/알람명 **정확 보존**, 짧은 키워드 쿼리 (3-5 토큰) | myservice status/cause는 단문(median 36자), 긴 쿼리 시 BM25 점수 희석 |
| **setup** | 부품명 + 작업동사 조합, SOP doc_id 패턴(`global_sop_`) 매칭 | SOP의 doc_id 자체가 검색 키워드 역할, action chapter는 절차 기술 |
| **general** | 원문 키워드 보존, 동의어 확장 최소화 | 다양한 doc_type에서 검색해야 하므로 과도한 확장은 노이즈 |

### 6.2 모든 route에 공통 적용할 규칙

1. **영문 MQ 1개 이상** 포함 (GCB 검색용)
2. **수치/코드 정확 보존** (Error Code 252, 3800W, version 2.50.14)
3. **MQ당 10단어 이내** (myservice 단문 매칭 최적화)
4. **"overview", "summary", "guide" 등 일반 확장어 금지** (myservice에 이런 단어 없음)

---

## 7. 향후 고려: 라우팅 → 검색 필터 연동

현재 라우팅은 MQ 선택에만 사용되지만, 향후 **doc_type 가중치**에도 활용 가능:

| route | doc_type 가중치 제안 |
|---|---|
| ts | myservice(status/cause) ×1.5, trouble_shooting_guide ×2.0, gcb ×1.2 |
| setup | SOP ×2.0, set_up_manual ×2.0, myservice(action) ×1.5 |
| general | 균등 |

이 방식은 doc_type 필터(배제)가 아닌 **boost(가산)**이므로, 관련 없는 문서가 완전히 배제되지 않으면서도 관련 문서가 상위에 올라오는 효과를 기대할 수 있음.

---

## 8. 결론

### 8.1 핵심 인사이트

1. **myservice가 94%를 차지하므로, 모든 라우팅은 결국 myservice 검색 최적화가 핵심**
2. **myservice 내 chapter(status/cause/action/result)가 사실상 ts/setup의 하위 구분과 대응**
3. **현재 3분류 체계는 유지하되, 분류 기준을 추상적 정의에서 실제 데이터 키워드 기반으로 변경**
4. **GCB 영문 데이터를 위한 이중 언어 MQ는 라우팅과 무관하게 항상 적용 필요**

### 8.2 구현 우선순위

| 순위 | 작업 | 난이도 | 기대 효과 |
|---|---|---|---|
| 1 | router 프롬프트 v2 적용 (신호어 기반) | 낮음 | 분류 정확도 향상 |
| 2 | 모든 MQ에 영문 쿼리 1개 보장 | 낮음 | GCB 검색 가능 |
| 3 | MQ 길이 제한 (10단어 이내) | 낮음 | myservice 매칭 향상 |
| 4 | route별 doc_type boost 적용 | 중간 | 관련 문서 상위 노출 |
| 5 | 2축 분류 (intent × doc_hint) | 높음 | 정밀 검색 가능 |
