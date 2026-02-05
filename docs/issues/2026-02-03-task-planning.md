# 태스크 기획 문서

## 날짜
2026-02-03

---

## 태스크 1: 답변 프롬프트 개선 - 트러블슈팅 답변 상세화

### 상태
[x] 완료

### 문제 상황
- 트러블 해결을 위한 답변이 **너무 요약된 형태**로 생성되고 있음
- 사용자가 문제 해결을 위해 필요한 상세 정보가 부족함

### 해결
`retrieval_ans_v1.yaml` 프롬프트 수정:
- 에러/알람/트러블슈팅 질문에 대해 5단계 구조 추가:
  1. 현상 요약
  2. 가능한 원인 (우선순위/빈도순)
  3. 점검 포인트 (체크리스트)
  4. 조치 방법 (단계별)
  5. 추가 참고사항

### 수정된 파일
- `backend/llm_infrastructure/llm/prompts/retrieval_ans_v1.yaml`

---

## 태스크 2: Retrieval Test 기본 질문 리스트 설정

### 상태
[x] 완료

### 목표
- Retrieval Test 페이지에서 사용할 **기본 질문 리스트**를 설정
- 현재는 저장된 평가 데이터만 사용하는데, 미리 정의된 테스트 질문도 제공

### 현재 구조
- `frontend/src/features/retrieval-test/context/retrieval-test-context.tsx`
  - `savedEvaluations`: API에서 불러온 평가 데이터 (Chat/Search에서 평가된 것)
  - 기본 질문 리스트가 없음 (삭제된 `test-questions.ts` 파일 존재)

### 예상 작업
1. `frontend/src/features/retrieval-test/data/default-questions.ts` 파일 생성
2. 기본 질문 리스트 정의 (groundTruthDocIds 없이 질문만)
3. context에서 기본 질문 + 저장된 평가 통합

### 기본 질문 리스트 (68개)

```typescript
export const DEFAULT_TEST_QUESTIONS: string[] = [
  // SOP/문서 관련
  "Global SOP_PRECIA_REP_PM_MFC 문서에서 2. Tool Arrange 내용이 뭔지 알고싶어",
  "global_sop_precia_all_pm_mfc.pdf 문서에는 어떤내용이 있니?",
  "Set_Up_Manual_SUPRA_N문서의 SUPRA N 제품설치 매뉴얼 목차 알려줘",

  // SUPRA N 관련
  "SUPRA N 장비 Setdup시 Cable Hook Up 에서 준비해야 할건 뭐니?",
  "SUPRA N Docking 방법 알려줘",
  "SUPRA N Baffle 장착시 Screw 토크 체결 spec을 알려줄래?",
  "SUPRA N TM ROBOT ENDEFFECTOR 장착시 SCREW 체결 토크 스펙 좀 알려줘",
  "SUPRA N APC Position 이상현상 발생 시 점검 포인트를 알려줘 트러블슈팅가이드 기반으로 알려줘",
  "supra n 설비에서 trace pin move abnormal시, 알람 번호 01, 02번에 대해 설명해주세요",
  "How to replace the PM Devicenet board on the SUPRA N",

  // TM Robot 관련
  "TM Robot - LL1 Pick Wafer Slide 관련내용을 알고있니?",
  "TM ROBOT END EFFECTOR3 VACUUM ALARM(10)이 발생하고 있는데 어떤 Action item이 있을까?",
  "robot teaching 방법을 알려 주세요",
  "ROBOT BEFORE/AFTER Pick Fail Alarm 원인 알려줘",

  // myservice/Order 관련
  "myservice 에서 \"TM Lamp Power Cable 탈착\" 이력있는 장비의 Equip. NO가 뭐니?",
  "Order No이 40161567의 Service Description내용 알려줘",
  "40144674에 대한 내용 알려줘",

  // 이력/문서 조회
  "25년도 FCIP의 교체 이력이 있다면 어떤 문서에서 확인할 수 있는지 리스트업해줘.",
  "SEC SRD 라인의 EPA404 LL 관련해서 점검 이력 좀 정리해줄래?",
  "24년 1월부터 12월까지 각 국가 별 Installation order 건 수, 신규 setup 건만 알려주고 이설/철거는 제외",
  "24년도 월 별 GCB Open 건 수는?",

  // 트러블슈팅
  "MFC9 케이블 해체 시에만 Sycon이 연결되는 현상의 원인과 해결 방법은 무엇인가요?",
  "pmc crevice module sycon 끊김현상, apc ,mfg , 솔레노이드 밸브 연결 끊김",
  "Alarm descrption said Fan2 LCU connection fail",
  "pendent system error 133, return code 50, parameter code -1, 10, 0 이게 무슨 error 인지 알려줘",
  "Temp interlock alarm이 발생하는 이유가 뭔가요?",
  "Source(R3)를 교체하고 나서 Ch1,2 Temp on을 하면 ELCB가 Trip됩니다. 이유가 뭔가요?",
  "SOurce 교체하고도 동일하게 Source on alarm이 발생했을때는 어떻게 해?",

  // EFEM/램프 관련
  "EFEM 램프 교체 절차를 알려주세요",

  // ZEDIUS 관련
  "ZEDIUS XP 설비의 TM Device Net Board 관련 작업 시에 필요한 작업 절차 알려줘",
  "ZEDIUS XP의 PM 절차를 알려줘",

  // Microwave 관련
  "Microwave의 Reflect가 발생 시 부품의 교체 순위를 정해주고 SOP를 알려주세요",

  // Pin 관련
  "새 pin 설치 작업시에, 15. Pin 높이 확인 단계에서 Ch1과 Ch2의 높이는 각각 얼마 인가요?",
  "INTEGER plus PM PIN Motor교체 시 pin높이는 몇으로 설정해? 그리고 S/W은 몇번을 off해?",

  // APC 관련
  "apc밸브 수명은 얼마나 되나요?",
  "SUPRA III 설비의 APC Pressure Hunting시 어떤 포인트를 점검 해야하는지 가이드좀 알려줘",
  "GENEVA XP 설비 APC 관련 내용이 있는 GCB no 좀 알 수 있을까?",

  // GENEVA 관련
  "GENEVA 설비의 IO Response error 발생 시 절차 알려줘",

  // ECOLITE 관련
  "ECOLITE 3000 Uniformity Spec Out 원인 분석 방법은?",
  "ECOLITE 정기 PM 절차 알려줘",
  "ECOLITE II 400 설비에서 Aligner에서 Notch alarm이 떴어. 조치방법을 알려줘.",
  "ECOLITE3000 설비에서 PM Chamver 내부 View Port 쪽에 Local Plasma 및 Arcing이 발생하는 원인을 알려줘",

  // DESCUM 관련
  "DESCUM 장비에서 얼라니어 노치 알람이 뜨면 조치 방법을 알려주세요.",

  // Reflow 관련
  "Reflow 설비에서 formic acid leak 감지가 되었습니다. 조치 방법에 대해서 알려주세요.",
  "RERFLOW 설비에서 insulation over temp 알람이 발생했을 때 조치해야 할 내용 알려줘.",

  // SUPRA V/Vplus 관련
  "SUPRA Vplus APC sensor part number는?",
  "What is the APC sensor part number of SUPRA V?",
  "SUPRA Vplus의 DSM사용하는 설비의 Ashing Rate가 낮은 경우 어떻게 해야지 올릴수 있는지 알려줘",
  "SUPRA Vplus에서 Cooling stage Align Time out이 발생했을 때의 확인방법을 알려줘",
  "SUPRA V에서 SUPRA Np로 개조한 설비의 설비 호기명을 알려줘",

  // SUPRA Np/GCB 관련
  "SUPRA Np Issue 내용들을 GCB 기반으로 알려줘. 각 GCB별 내용을 현상, 원인, 대책, 결과 4가지 컬럼으로 정리해서 표로 만들어줘",
  "GCB 67224에 대해 알려주세요",

  // INTEGER 관련
  "INTEGER model에서 main rack door open interlock 발생 시 해결방법은?",
  "인터락 스위치 교체 절차는 어떻게 되나요?",

  // 일반 기술 질문
  "Flow switch의 교체방법에 대해서 알려주세요",
  "SUPRA XP의 Prevent maintenance의 절차를 알려줘",
  "Baffle의 하는 역할을 알려줘",
  "Ashing rate가 낮을때 Chuck 온도가 관계있는 이유가 뭐야?",
  "SW Patch를 할 때 백업에 필요한 File에는 어떤 것들이 있니?",
  "AR 과 관련있는 파라미터는 뭔가요?",
  "Chamber open 후 backup Process 를 알려 주세요",

  // Particle/품질 관련
  "Particle 문제 시 봐야할 것들과 순서를 알려 주세요",
  "Particle 발생 원인은 뭐 였나요?",
  "AR이 감소하면 어떻게 대처 해야 하나요?",
  "Particle이 발생 하면 어떻게 대처 해야 하나요?",

  // FDC/MFC 관련
  "O2 Gas의 FDC Data spec이 3%이상 차이가 발생하고 있는데 이것에 대해서 MFC를 제외한 다른 원인에 대해서 확인해주세요",

  // 통계/분석
  "각 Segment 별 SOP 개수를 알려줘",
  "EPAGQ03에서 Source Unready Alarm 발생한 이력에 대해 정리하여 알려줘",
];
```

### 참고사항
- 이 질문들은 groundTruthDocIds가 없으므로 **메트릭 계산 불가**
- 단순히 검색 결과 확인용 + 수동으로 관련 문서 평가하는 용도
- 평가 후 저장하면 savedEvaluations에 추가됨

### UI 변경 (예상)
- "기본 질문" 섹션 추가 또는 탭으로 분리
- 기본 질문 선택 → 실행 → 결과 확인 → 관련 문서 평가 → 저장

### 관련 파일
- `frontend/src/features/retrieval-test/context/retrieval-test-context.tsx`
- `frontend/src/features/retrieval-test/pages/retrieval-test-page.tsx`

---

## 태스크 3: 재생성 시 피드백 메시지 표시

### 상태
[x] 완료

### 목표
- "재생성" 버튼 클릭 후 답변 재생성 시, 사용자가 선택한 옵션을 채팅 메시지로 표시
- 사용자가 어떤 조건으로 재생성했는지 명확하게 알 수 있도록 함

### 해결
`chat-page.tsx`의 `submitRegeneration` 함수 수정:
- user 메시지에 선택된 장비/문서 정보를 prefix로 추가
- 형식: `[SUPRA N / SOP로 재생성] 원래 질문...`

### 메시지 형식 예시
- `[SUPRA N / SOP로 재생성] APC 알람 원인이 뭐야?`
- `[전체 장비 / 전체 문서로 재생성] APC 알람 원인이 뭐야?`

### 수정된 파일
- `frontend/src/features/chat/pages/chat-page.tsx`

---

## 태스크 4: myservice 검색 시 "p0:" 표시 이슈

### 상태
[ ] 기획 중

### 문제 상황
- myservice 문서 검색 시 `p0:`가 항상 표시됨
- 원래는 `[action] [state] ...` 형식의 텍스트가 있어야 하는 위치

### 예상 원인 (분석 필요)
- 데이터 파싱 과정에서 필드가 잘못 매핑되었을 가능성
- 원본 데이터의 priority 필드(`p0`, `p1` 등)가 잘못 노출되고 있을 가능성
- myservice 문서 특유의 구조가 일반 파서에서 제대로 처리되지 않는 문제

### 예상 작업
1. 원인 분석
   - [ ] myservice 문서의 원본 데이터 구조 확인
   - [ ] ES에 저장된 실제 데이터 확인
   - [ ] 파싱 로직 확인 (`backend/services/ingest/` 또는 관련 파서)

2. 수정
   - [ ] 원인에 따라 파싱 로직 수정 또는 데이터 재색인

### 관련 파일 (예상)
- `backend/services/es_ingest_service.py`
- `backend/domain/doc_type_mapping.py`
- myservice 관련 파서 파일
