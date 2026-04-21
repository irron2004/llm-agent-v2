# Chat Log Evaluation Samples (n=20)

| # | month | qtype | bucket | q_len | a_lines | has_table | has_kw_block | q 요약 |
|---|---|---|---|---:|---:|:---:|:---:|---|
| 1 | 2026-02 | spec_inquiry | s | 41 | 9 | . | . | SUPRA Vm의 Slit door screw part number를알려줘 |
| 2 | 2026-02 | spec_inquiry | s | 64 | 11 | . | . | What is the part number for the slit door screw on the SUPRA |
| 3 | 2026-01 | spec_inquiry | m | 55 | 23 | Y | . | TM모듈에서 ROBOT PART NUMBER를 모두 알려줘 참고 문서: 사용 안 함 또는 찾지 못함 |
| 4 | 2026-02 | alarm_trouble | m | 88 | 59 | Y | . | SR8250 TM Robot에서 Communication Alarm 떠서 봤더니 Error Code 6번 R |
| 5 | 2026-01 | alarm_trouble | m | 341 | 46 | . | . | -. EFEM LL1 Teaching  -> y target값 -240맞추고 transfer 과정에서 y축  |
| 6 | 2026-03 | alarm_trouble | l | 55 | 70 | Y | . | SUpra설비에서 모든 채널에 EPD COmmunication alarm이 떴는데 어떻게 해결해?? |
| 7 | 2026-01 | alarm_trouble | l | 58 | 78 | . | Y | INTEGER plus에서 Chamber open interlock alarm발생 시 해결 방법을 알려줘 |
| 8 | 2026-01 | history_lookup | m | 36 | 48 | Y | . | INTEGER plus Ball Screw 교체 이력 모두 알려줘 |
| 9 | 2026-01 | history_lookup | m | 40 | 58 | Y | . | supra n의 wafer broken issue 이력을 표로 정리해줘ㅏ |
| 10 | 2026-01 | location_inquiry | s | 1611 | 18 | Y | . | 질문별 문서 위치 및 목적 안내  질문 내용	문서명 및 위치	문서 종류	문서 목적 및 설명  SUPRA XP |
| 11 | 2026-01 | location_inquiry | m | 69 | 45 | . | Y | GENEVA설비에서 2개의 vac Sol valve가 있는데 exhaust 가 각각 어느 port랑 연결되어 |
| 12 | 2026-03 | procedure | m | 79 | 22 | . | . | mySITE에서 Setup service order 내의 Maintenance task에서 사용한 histo |
| 13 | 2026-01 | procedure | m | 62 | 32 | . | Y | INTEGER plus safety controller, safety module교체 작업을 한 이유를 알려 |
| 14 | 2026-01 | procedure | l | 64 | 104 | Y | Y | SUPRA N 에서 TM ROBOT 교체 순서를 자세하게 알려줘. 또한 3개월 간 교체 간 발생한 이슈도 정 |
| 15 | 2026-02 | troubleshoot_diag | l | 23 | 63 | . | . | FCIP Leak 발생시 조치방법을 알려줘 |
| 16 | 2026-02 | troubleshoot_diag | l | 55 | 79 | Y | Y | reflow 설비에서 formic acid leak 감지가 되었습니다. 조치 방법에 대해 알려주세요 |
| 17 | 2026-02 | troubleshoot_diag | xl | 33 | 142 | Y | Y | SUPRA XP Toxic Gas Turn On 방법 알려줘 |
| 18 | 2026-02 | list_lookup | m | 34 | 46 | Y | . | 체크리스트로 만들라고 참고 문서: 사용 안 함 또는 찾지 못함 |
| 19 | 2026-03 | short_followup | s | 10 | 18 | Y | . | FFU 转速spec |
| 20 | 2026-01 | general | m | 27 | 47 | . | Y | SUPRA Q Robot 통신은 어떤 통신을 해? |

---

## Sample 01 — spec_inquiry / s (9줄)

- **Timestamp**: `2026-02-05 15:50:09`
- **Question** (41자):

  ```
  SUPRA Vm의 Slit door screw part number를알려줘
  ```
- **Sources (Top 20)**: first 5 = ['scraped_gcb.json', 'scraped_gcb.json', 'scraped_gcb.json', 'scraped_gcb.json', 'scraped_gcb.json']
- **Structure flags**: table=. kw_block=. summary=. cause=. gcb_footer=Y part_num=Y
- **First line**: `SUPRA Vm의 Slit door screw의 Part Number는 **5900-1000227 (M6*50)** 입니다[3].`

<details><summary>Reference answer (PII masked)</summary>

```markdown
SUPRA Vm의 Slit door screw의 Part Number는 **5900-1000227 (M6*50)** 입니다[3].

- 해당 스크류는 SUS 재질로 사용되며, UMC 고객사에서 2년 이상 교체 없이 사용 시 스크류 파손 사례가 보고되어 정기 교체(1년 주기)를 권장하고 있습니다[1,3].
- Door valve Part Number는 C5700173이며, 이와 연관된 스크류 정보입니다[1,3].
- 스크류 체결 시 허용된 토크 스펙을 준수해야 하며, 과도한 토크 체결은 파손 원인이 될 수 있으니 주의가 필요합니다[1,3].

요약하면, SUPRA Vm Slit door screw Part Number는 **5900-1000227 (M6*50)**이며, SUS 재질이고 정기 교체 및 적정 토크 준수가 중요합니다.

해결되지 않으면 GCB에 등록하세요. 문의: [NAME] ([EMAIL])
```

</details>

## Sample 02 — spec_inquiry / s (11줄)

- **Timestamp**: `2026-02-26 10:24:18`
- **Question** (64자):

  ```
  What is the part number for the slit door screw on the SUPRA Vm?
  ```
- **Sources (Top 20)**: first 5 = ['scraped_gcb.json', 'scraped_gcb.json', 'scraped_gcb.json', 'scraped_gcb.json', 'scraped_gcb.json']
- **Structure flags**: table=. kw_block=. summary=. cause=. gcb_footer=. part_num=Y
- **First line**: `The provided reference materials do not contain information related to this ques`

<details><summary>Reference answer (PII masked)</summary>

```markdown
The provided reference materials do not contain information related to this question.

There are some **slit door–related parts** listed, but none is explicitly defined as “slit door screw” for SUPRA Vm, so the exact screw P/N cannot be confirmed:

- **[Part Name] GUIDE_SCREW(M6), [Part Number] C5700011, [Maker] RS** (Old Part: 5176-05963 / SCREW GUIDE(M6)) – used at slit door bracket area.[16]  
- **[Part Name] PLATE_SLIT DOOR, [Part Number] C5700218, [Maker] MF** – slit door plate itself, not the screw.[17]  

Since the documents do not clearly specify which screw you mean (valve screw, bracket screw, plate mounting screw, etc.), the exact part number cannot be identified from the provided references.

If you need the precise screw P/N, register the case in **GCB** with photos or drawings indicating the exact screw position.  
For further escalation or feedback on this AI answer, contact **GunamDOH ([EMAIL])**.
```

</details>

## Sample 03 — spec_inquiry / m (23줄)

- **Timestamp**: `2026-01-02 10:33:11`
- **Question** (55자):

  ```
  TM모듈에서 ROBOT PART NUMBER를 모두 알려줘
  참고 문서: 사용 안 함 또는 찾지 못함
  ```
- **Sources (Top 0)**: first 5 = []
- **Structure flags**: table=Y kw_block=. summary=. cause=. gcb_footer=Y part_num=Y
- **First line**: `TM 모듈에서 사용되는 ROBOT의 모든 Part Number는 제공된 참고자료 및 이전 대화 기록에 명확히 존재하지 않습니다. 특히, JEL `

<details><summary>Reference answer (PII masked)</summary>

```markdown
TM 모듈에서 사용되는 ROBOT의 모든 Part Number는 제공된 참고자료 및 이전 대화 기록에 명확히 존재하지 않습니다. 특히, JEL 제조사의 TM ROBOT Part Number도 확인 불가합니다.

아래와 같이 정리됩니다.

| 모듈명 | Part Number | Maker | Spec | 비고 |
|--------|-------------|-------|------|------|
| TM (JEL TM ROBOT) | - | - | - | Part Number 미확인, 자료 없음 |

참고로, SANKYO 제조사의 EFEM 모듈 ROBOT Part Number는 별도로 아래와 같이 확인 가능합니다.

| 모듈명 | Part Number | Maker | Spec | 비고 |
|--------|-------------|-------|------|------|
| EFEM | A0400326 | SANKYO | SR8242-0002 (FW 2.08) | EFEM ROBOT |
| EFEM | A0400336 | SANKYO | REU10015050 | PAD ROBOT EFEM |
| EFEM | A0400452 | SANKYO | SR8242PSK01 (FW 2016.08) | EFEM ROBOT SR8242 |
| EFEM | A0500044 | SANKYO | SC5000 | ROBOT CONTROLLER |

따라서 TM 모듈에서 사용되는 ROBOT의 Part Number는 제공된 자료에 없습니다.  
추가 확인이 필요하면 GCB에 등록하여 문의하시기 바랍니다.

해결되지 않으면 GCB에 등록하세요.  
문의: [NAME] ([EMAIL])  
[문서 1,2,5,8]
```

</details>

## Sample 04 — alarm_trouble / m (59줄)

- **Timestamp**: `2026-02-28 10:25:33`
- **Question** (88자):

  ```
  SR8250 TM Robot에서 Communication Alarm 떠서 봤더니 Error Code 6번 Real Time 이 떠 있는데, 어떤 조치 했었어?
  ```
- **Sources (Top 36)**: first 5 = ['myservice_psk.jsonl', 'myservice_psk.jsonl', 'myservice_psk.jsonl', 'myservice_psk.jsonl', 'myservice_psk.jsonl']
- **Structure flags**: table=Y kw_block=. summary=. cause=. gcb_footer=Y part_num=Y
- **First line**: `SR8250 TM Robot에서 Communication Alarm 발생 시 Error Code 6번(Real Time Error)에 대한 조치`

<details><summary>Reference answer (PII masked)</summary>

```markdown
SR8250 TM Robot에서 Communication Alarm 발생 시 Error Code 6번(Real Time Error)에 대한 조치 내용은 다음과 같습니다. 여러 사례를 종합하여 우선순위별로 단계별 조치 방법을 안내드립니다.

---

### 1. **초기 상태 및 원인 파악**
- Error Code 6번은 Real Time Error로, 원인이 다양하며 주로 **로봇 내부 모터 과부하, 케이블 단선, 서보 파워 문제, 엔코더 이상, 또는 로봇 컨트롤러 이상** 등이 있습니다[1,2,4,10].
- 대표적으로 Axis 2 Clamp Error, Z축 모터 과부하, 충돌 감지, 케이블 접속 불량 등이 동반될 수 있습니다[1,4,14].

---

### 2. **Step-by-Step 조치 가이드**

| 단계 | 조치 내용 | 상세 설명 및 참고 사항 |
|-------|------------|-----------------------|
| 1 | **Alarm 상태 및 Error Code 확인** | Pendant 및 Controller에서 Error Code 6번과 관련된 Real Time Error(252) 확인[1,7,10]. |
| 2 | **TM Robot Controller Reset 및 Servo Power On 시도** | Controller Reset 후 Servo Power On 시도. 정상 시 GUI에서 Robot Home 가능 여부 확인[1,7]. 실패 시 다음 단계 진행. |
| 3 | **케이블 상태 점검 및 재접속** | Robot Controller 케이블, 엔코더 케이블, Molex 케이블 등 연결 상태 육안 점검 및 재접속 실시. 접촉 불량 시 Error 지속 가능[4,8,11,26]. |
| 4 | **ABS Home 수행** | ABS Home 동작 수행하여 위치 초기화. ABS Home 실패 시 Servo Power Off 및 Error 발생 가능[4,9,29,36]. |
| 5 | **Robot Speed 조정** | Robot Speed를 100%에서 60~80% 수준으로 낮춰서 사용 권고. 속도 낮출 경우 떨림 심해질 수 있으니 60% 이하로 낮추지 말 것[2,10,15,17]. |
| 6 | **Log 및 Error History 분석** | Pendant 및 Controller Error History 확인, Buzz5000 프로그램으로 Error Log 백업 및 분석[3,7,8,17]. |
| 7 | **Physical 점검** | Robot Arm 및 Blade 체결 상태, Vacuum 상태 점검. 특히 Hand 2번 Vacuum 저하 및 Arm 충돌 여부 확인[4,7]. |
| 8 | **Robot 교체 고려** | 위 조치 후에도 Error 지속 시, Robot 또는 Robot Controller 교체 필요. 특히 Z축 모터 고장, Controller 내부 보드 이상, Encoder Cable 단선 의심 시 교체 권고[1,4,9,12,17,23]. |
| 9 | **고객 및 업체 협의** | ZEUS, Nidec 등 업체 문의 및 협의 진행. 필요 시 Robot Log 반출 및 추가 분석[3,6,8,18]. |

---

### 3. **특이 사례 및 참고**

- **Servo Power On Fail** 현상 발생 시 Controller Reset 반복, Pendant 연결 상태 점검 필요[1,7].
- **Z축 Clamp Error 및 Motor Fail** 관련 Error Code 4, 80, 457 등이 동반될 수 있음[4,10,29].
- **Robot Speed 변경 후 재발 방지** 위해 60~80% 사이 유지 권고[2,10,15].
- **Robot 내부 Cable 단선 또는 Motor Fail** 의심 시 Robot 교체가 최종 해결책[4,9,12,23].
- **Robot Pendant Parameter On/Off 변경**으로 Error Clear 시도 가능하나 근본 원인 해결 아님[8].
- **Robot Log 분석 시 통신 끊김 확인**되면 Controller 또는 Robot 내부 문제 가능성 높음[8,17].

---

### 4. **요약**

- **Error Code 6번 Real Time Error는 다양한 원인으로 발생하므로, 우선 Controller Reset, Servo Power On, 케이블 접속 상태 점검, ABS Home 수행, Robot Speed 조정 순으로 점검하세요.**
- **이후에도 문제가 지속되면 Robot 및 Controller 교체를 검토해야 합니다.**
- **특히 Z축 모터 관련 Error가 동반되면 Robot 교체가 필요합니다.**
- **점검 및 조치 후에도 해결되지 않으면 ZEUS 또는 Robot 제조사에 문의하고, GCB에 등록하여 추가 지원 받으시기 바랍니다.**

---

### 5. **참고 부품 정보**

| 부품명 | Part Number | Maker | 비고 |
|---------|-------------|-------|-------|
| TM Robot Controller (SR8250PSK03) | B117J03 | PSK (Maker 정보 0으로 표기되나 PSK 관련) | SUPRA N 모델용 Controller |

---

해결되지 않으면 GCB에 등록하세요. 문의: [NAME] ([EMAIL])

---

**참고문서:** [Doc 1,2,4,7,8,9,10,12,15,17,23,29]
```

</details>

## Sample 05 — alarm_trouble / m (46줄)

- **Timestamp**: `2026-01-27 16:43:10`
- **Question** (341자):

  ```
  -. EFEM LL1 Teaching
  
  -> y target값 -240맞추고 transfer 과정에서 y축 끝까지 못들어간 상태로 알람 발생
  
  -> Task no:1 stnd_com.txt exec.line:2822 4-0-0 : Area Over
  
  -> Y target값 : -240.364mm
  
  -> Alarm 시 -238.5정도에서 pusher 놓으면서 알람 발생
  
  -> -237.8(임의)로 하면 알람발생 안함
  
  -> 내려놓을때 2mm 오프셋 때문으로 추정
  
  -> 레이저지그 올려놓고 teaching시 똑같이 -240 넘어감
  
  
  
  이런 현상이 발생한적이 있었는지, 그리고 개선하는 방법 구체적으로 설명해
  ```
- **Sources (Top 550)**: first 5 = ['myservice_psk.csv', 'myservice_psk.csv', 'myservice_psk.csv', 'myservice_psk.csv', 'myservice_psk.csv']
- **Structure flags**: table=. kw_block=. summary=Y cause=. gcb_footer=Y part_num=.
- **First line**: `### EFEM LL1 Teaching y target값 -240 알람 발생 사례 및 개선 방법`

<details><summary>Reference answer (PII masked)</summary>

```markdown
### EFEM LL1 Teaching y target값 -240 알람 발생 사례 및 개선 방법

---

#### 1. 원인 분석
- LL1 EFEM Robot Teaching 중 y target값 -240 부근에서 위치 불일치 또는 좌표 오류가 발생하여 Disarray Alarm이 발생한 사례가 있습니다.
- 알람 발생 시점은 y축 위치가 약 -238.5mm에서 pusher가 wafer를 놓으면서 발생하며, -237.8mm로 임의 조정 시 알람이 발생하지 않는 것으로 보아 약 2mm 정도의 오프셋이 문제로 추정됩니다.
- 레이저 지그를 올려놓고 teaching을 진행해도 동일하게 -240mm를 넘어가는 현상이 확인되어 좌표 데이터 자체의 정확성 문제 가능성이 있습니다.
- Dummy Wafer를 이용한 Teaching 점검 시에도 좌표 불일치가 확인되었으며, Load Lock 및 Robot End Effector 상태 점검이 필요합니다.

#### 2. 개선 방법 (Step-by-Step)

1) **Dummy Wafer를 사용한 Teaching 재실시 및 좌표 보정**
   - Dummy Wafer를 장착하여 LL1 EFEM Robot Teaching을 다시 수행합니다.
   - y target값 -240 위치를 중심으로 좌표 정확도를 재확인하고, 필요 시 좌표를 수정하여 Teaching 데이터를 보정합니다.

2) **EFEM Robot End Effector 상태 점검**
   - End Effector의 센서 이상 여부를 확인합니다.
   - Mechanical 상태(예: pusher 동작, 센서 위치, 마모 등)를 점검하여 정상 동작 여부를 확인합니다.

3) **Load Lock 및 관련 설비 상태 점검**
   - Load Lock의 Mechanical 상태 및 위치 센서 상태를 점검하여 Robot 동작에 영향을 주는 문제가 없는지 확인합니다.

4) **Teaching 데이터 저장 및 관리 프로세스 점검**
   - Teaching 좌표 데이터가 올바르게 저장되고 관리되고 있는지 점검하여 데이터 오류 발생 가능성을 차단합니다.

5) **필요 시 Robot Controller 및 관련 부품 점검/교체**
   - 위 점검 후에도 문제가 지속되면 Robot Controller 및 관련 부품의 이상 여부를 점검하고, 필요 시 교체를 진행합니다.

#### 3. 주의사항
- Teaching 작업 시 레이저 지그 및 Dummy Wafer 사용 시 정확한 위치 고정을 위해 설비 전원이 안정적인 상태인지 확인하세요.
- 센서 점검 시 이물질 확인 및 Air blow를 사용하여 센서 상태를 깨끗하게 유지하세요.
- Mechanical 점검 시 안전장비 착용 및 설비 전원 차단 후 작업하세요.

---

### 요약
- y target값 -240 부근에서 발생하는 알람은 Teaching 좌표 불일치 및 Robot End Effector 동작 오류가 주요 원인입니다.
- Dummy Wafer를 이용한 좌표 재보정, End Effector 및 Load Lock 상태 점검, Teaching 데이터 관리 프로세스 점검이 핵심 개선 방법입니다.
- 필요 시 Robot Controller 및 부품 점검/교체를 진행하세요.

---

해결되지 않으면 GCB에 등록하세요. 문의: [NAME] ([EMAIL])

[참고문서: 문서 40060194 DPCB803 LL1 EFEM Robot Teaching Disarray Alarm 발생 사례]
```

</details>

## Sample 06 — alarm_trouble / l (70줄)

- **Timestamp**: `2026-03-09 11:48:38`
- **Question** (55자):

  ```
  SUpra설비에서 모든 채널에 EPD COmmunication alarm이 떴는데 어떻게 해결해??
  ```
- **Sources (Top 37)**: first 5 = ['scraped_gcb.json', 'myservice_psk.jsonl', 'myservice_psk.jsonl', 'scraped_gcb.json', 'scraped_gcb.json']
- **Structure flags**: table=Y kw_block=. summary=Y cause=. gcb_footer=Y part_num=.
- **First line**: `SUPRA 설비에서 모든 채널에 EPD Communication Alarm이 발생한 경우, 다음과 같은 단계별 점검 및 조치 방법을 권장합니다.`

<details><summary>Reference answer (PII masked)</summary>

```markdown
SUPRA 설비에서 모든 채널에 EPD Communication Alarm이 발생한 경우, 다음과 같은 단계별 점검 및 조치 방법을 권장합니다.

---

### Step-by-Step 해결 가이드

1. **케이블 및 커넥터 상태 점검**
   - **EPD Power Cable의 Connector 단 완전 고정 여부 확인**  
     - 과거 사례에서 EPD Power Cable 커넥터 유격으로 인해 전원 On/Off가 반복되어 Communication Alarm이 발생한 적이 있습니다. 커넥터를 완전히 체결하고, 케이블 타이 등으로 고정하여 유격을 제거하세요. [Doc 1, Doc 10]
   - **HUB와 NPORT 간 케이블 연결 상태 확인**  
     - HUB 쪽 커넥터와 Port 간 유격이 심하면 통신이 끊어져 Alarm이 발생합니다. 케이블 탈착 후 재장착, 유격 부분 케이블 타이 고정 조치가 필요합니다. [Doc 2]

2. **EPD 및 HUB 장비 전원 및 리셋**
   - **EPD 전원 Reset 버튼 클릭 또는 Power Cycle 수행**  
     - EPD Display Off 상태 시 전원 버튼 Reset 후 정상 동작 여부 확인하세요. [Doc 1, Doc 10]
   - **Hub 및 EPD LAN Port 상태 점검**  
     - 점멸 상태가 정상인지 확인하고, 필요 시 Hub 및 EPD 간 케이블 재장착 및 Reset을 수행하세요. [Doc 3]

3. **CTC 및 EFEM Controller 점검**
   - **CTC 및 EFEM PC Reset 후 Alarm Clear 시도**  
     - 통신 끊김 현상은 Windows 10 Controller의 네트워크 끊김 문제와 연관될 수 있으므로, Controller 재부팅 후 상태 확인이 필요합니다. [Doc 11, Doc 23]
   - **네트워크 포트 속도 및 절전 모드 설정 확인**  
     - 모든 네트워크 포트에 대해 속도 설정과 절전 모드 해제 설정을 점검하세요. [Doc 11]

4. **Software 및 Driver 점검**
   - **EPD Driver 버전 및 설정 확인**  
     - EPD Driver가 3초 이상 응답 없으면 Communication Alarm 처리하도록 되어 있으므로, Driver 버전 및 설정을 확인하고 필요 시 업데이트하세요. [Doc 4]
   - **EPD Type 설정 확인**  
     - Dual EPD에서 Single EPD로 변경 시 Software 내 EPD Type 변경 누락으로 Alarm 발생할 수 있으니, Setup.ini 파일에서 EPD Type을 정확히 설정하세요. [Doc 19]

5. **장비 내부 부품 및 Firmware 점검**
   - **EPD 내부 Fan 및 Board 상태 확인**  
     - Fan 고장이나 Board 문제로 인해 Communication Alarm이 발생할 수 있으므로, Fan 소음 및 동작 상태 점검 후 필요 시 교체하세요. [Doc 24]
   - **Firmware 업데이트 적용 여부 확인**  
     - 과거 Firmware Patch로 Communication Error 개선 사례가 있으니, 최신 Firmware 적용 여부를 확인하세요. [Doc 13]

6. **기타 점검 사항**
   - **EPD Power Cable 내부 충진 보강 및 Connector 개선 검토**  
     - 케이블 내부 충진 보강 후에도 접촉 불량 사례가 있으므로, 케이블 및 Connector 개선 작업을 검토하세요. [Doc 1, Doc 5]
   - **Dual EPD 제품에 대한 H/W 개선 방안 검토**  
     - Dual EPD 관련 기존 H/W 개선 방안을 검토하고 적용하세요. [Doc 4]

---

### 요약

| 점검 항목                      | 주요 내용 및 조치                                    | 참고 문서 번호 |
|-----------------------------|-------------------------------------------------|------------|
| EPD Power Cable 커넥터 고정        | 커넥터 유격 제거, 케이블 타이 고정                         | Doc 1, Doc 10 |
| HUB-NPORT 케이블 연결 상태 확인    | 케이블 탈착 후 재장착, 유격 부분 고정                       | Doc 2       |
| EPD 전원 Reset 및 Display 상태 확인 | 전원 버튼 Reset, Power Cycle 수행                          | Doc 1, Doc 10 |
| CTC 및 EFEM Controller Reset    | Controller 재부팅, 네트워크 포트 속도 및 절전 모드 설정 확인          | Doc 11, Doc 23 |
| Software EPD Type 설정 확인       | Setup.ini 파일 내 EPD Type 정확히 설정                      | Doc 19      |
| EPD Fan 및 Board 상태 점검       | Fan 소음 및 동작 확인, 필요 시 교체                         | Doc 24      |
| Firmware 업데이트 확인          | 최신 Firmware 적용 여부 확인                              | Doc 13      |
| 케이블 내부 충진 및 Connector 개선 | 케이블 보강 및 Connector 개선 검토                          | Doc 1, Doc 5 |
| Dual EPD H/W 개선 방안 검토      | Dual EPD 관련 H/W 개선 적용                                | Doc 4       |

---

### 주의사항
- 작업 시 **전원 차단 및 안전 절차 준수** 필수입니다.
- 케이블 및 커넥터 점검 시 **이물질 확인 및 Air blow 사용**으로 접촉 불량 방지하세요.
- Controller 재부팅 시 작업 중인 공정 영향 최소화에 주의하세요.

---

해당 조치 후에도 문제가 해결되지 않으면, **GCB에 등록하여 추가 기술 지원을 요청하시기 바랍니다.**

문의 및 AI 답변 품질 관련은 담당자 GunamDOH ([EMAIL])에게 연락 바랍니다.
```

</details>

## Sample 07 — alarm_trouble / l (78줄)

- **Timestamp**: `2026-01-05 17:00:39`
- **Question** (58자):

  ```
  INTEGER plus에서 Chamber open interlock alarm발생 시 해결 방법을 알려줘
  ```
- **Sources (Top 20)**: first 5 = ['setupmanual.json', 'INTEGER plus_VTIS-Ver2.23.0_R11_ALID.csv', 'scraped_gcb.json', 'scraped_sop.json', 'INTEGER plus_VTIS-Ver2.23.0_R11_ALID.csv']
- **Structure flags**: table=. kw_block=Y summary=Y cause=. gcb_footer=Y part_num=.
- **First line**: `INTEGER plus에서 "Chamber open interlock" 알람 발생 시 해결 방법을 단계별로 안내드립니다.`

<details><summary>Reference answer (PII masked)</summary>

```markdown
INTEGER plus에서 "Chamber open interlock" 알람 발생 시 해결 방법을 단계별로 안내드립니다.

---

## 1. 원인 (Cause)
- 챔버 도어가 완전히 닫히지 않았거나  
- 도어 인터락 스위치/센서 이상 또는 배선 문제  
- 도어 씰(오링, 가스켓) 이물질 또는 손상으로 인한 미세 틈 발생  
- 소프트웨어 상 인터락 상태 비트 불일치

---

## 2. 점검 (Check)

### 2-1. 안전 조치
- 공정 중이면 즉시 STOP 또는 ABORT하여 공정을 정지합니다.  
- RF, 가스, 진공 펌프 등 모든 공정 관련 전원을 OFF 상태로 만듭니다.  
- 챔버 압력이 대기압인지 확인 후 작업자 주변 안전 확보합니다.  
- Chamber Open 시 협착 사고 주의하며 1인 작업 실시합니다[1,4,11].

### 2-2. 기계적 점검
- 챔버 도어가 완전히 닫혔는지 확인합니다. 슬라이딩 또는 힌지 도어는 끝까지 밀어 넣어 "딱" 닫히는 느낌이 있어야 합니다.  
- 도어 클램프(락 레버, 핸들)가 완전히 걸려 있는지 확인합니다.  
- 도어 씰부, 오링, 가스켓에 이물질이나 손상, 끼임이 없는지 확인하고 이물질은 제거합니다.  
- 도어 센서(리미트 스위치, 포토 센서) 위치가 어긋나지 않았는지 육안 점검합니다.  
- 센서 커넥터가 빠지거나 케이블이 꺾여 있지 않은지 확인합니다[1,8,16].

### 2-3. 소프트웨어 및 인터락 상태 확인
- HMI 또는 터치패널에서 Alarm 또는 Interlock 화면 진입하여 어떤 챔버, 어떤 도어 인터락인지 확인합니다[2].  
- 도어를 정상 위치로 닫은 후 인터락 상태 비트가 Closed 또는 Safe 상태로 변경되는지 모니터링합니다.  
- 이상 없으면 Alarm Clear 또는 Reset 버튼을 눌러 알람을 해제합니다[2,4].

### 2-4. 센서 및 배선 점검 (필요 시)
- 장비 I/O 모니터에서 도어 인터락 신호가 도어 열림/닫힘에 따라 정상적으로 0↔1 변하는지 확인합니다.  
- 신호 변화가 없으면 센서 고장 또는 배선 문제 가능성이 있으므로 커넥터, 핀 휨, 단선 여부를 점검합니다.  
- 근접센서/포토센서 타입은 센서와 타겟 간격을 매뉴얼 규정값에 맞게 조정합니다.  
- 리미트 스위치 타입은 도어 닫힘 시 스위치 레버가 충분히 눌리는지 확인하고 조정합니다[4].

---

## 3. 조치 (Action)
- 물리적 이상(도어 미닫힘, 이물질, 센서 위치 불량 등) 발견 시 즉시 조치하여 정상 상태로 복구합니다.  
- 소프트웨어 상 인터락 상태가 정상으로 변경되면 알람을 Clear 합니다.  
- 센서 또는 배선 이상 시 교체 또는 수리 진행합니다.  
- 작업 후 간단한 Pump/Vent 동작 테스트로 재발 여부 확인합니다.

---

## 4. 검증 (Verification)
- 알람이 해제되고 도어 인터락 상태가 정상으로 유지되는지 확인합니다.  
- 공정 재개 전 안전 상태 및 인터락 정상 작동 여부를 반드시 확인합니다.

---

## 5. 주의사항 (Caution)
- 인터락 신호를 강제로 점퍼 연결하여 우회하지 마십시오.  
- 도어가 완전히 닫히지 않은 상태에서 공정, RF, 가스 등을 ON 하지 마십시오.  
- 안전커버나 센서 브래킷을 임의로 제거한 상태에서 운전하지 마십시오.  
- Chamber Open/Close 작업 시 협착 사고에 각별히 주의하며 1인 작업을 권장합니다[1,4,11].

---

## 요약
- Chamber door 완전 닫힘 및 클램프 상태 확인  
- 도어 씰 및 센서 상태 점검 및 이물 제거  
- HMI에서 인터락 상태 확인 후 알람 Clear  
- 센서 신호 및 배선 점검, 필요 시 조정 또는 교체  
- 안전 수칙 준수 및 작업자 협착 주의

---

해결되지 않으면 GCB에 등록하고 질문하세요.  
문의: [NAME] ([EMAIL])

---

**참고 문서 핵심 키워드:**  
INTEGER plus, Chamber Open Interlock Alarm, ALID 1704, PM Chamber, Alarm Clear, 도어 센서, 인터락, 안전 주의, SOP, I/O 모니터링[1,2,4,8,11].
```

</details>

## Sample 08 — history_lookup / m (48줄)

- **Timestamp**: `2026-01-14 20:17:31`
- **Question** (36자):

  ```
  INTEGER plus Ball Screw 교체 이력 모두 알려줘
  ```
- **Sources (Top 200)**: first 5 = ['myservice_psk.csv', 'myservice_psk.csv', 'myservice_psk.csv', 'myservice_psk.csv', 'myservice_psk.csv']
- **Structure flags**: table=Y kw_block=. summary=. cause=. gcb_footer=. part_num=.
- **First line**: `| No. | 문서명           | Order No. | Equip ID | 내용                          | Rem`

<details><summary>Reference answer (PII masked)</summary>

```markdown
| No. | 문서명           | Order No. | Equip ID | 내용                          | Remark                          |
|-----|------------------|-----------|----------|-------------------------------|--------------------------------|
| 1   | myservice_psk.csv | 40044502  | DPCEJ01  | LL1 Ball Screw 교체 완료       | INTEGER plus, Ball Screw 교체  |
| 2   | myservice_psk.csv | 40156344  | IPC3715  | LL2 Ball Screw 부하 상승으로 교체 요청 및 변경점 언급 | INTEGER plus, Ball Screw 교체, 변경점 포함 |
| 3   | myservice_psk.csv | 40151744  | DPCHZ11  | LL1 Torque spec out으로 ball screw 교체 요청 | INTEGER plus, Ball Screw 교체  |
| 4   | myservice_psk.csv | 40147068  | DPCB706  | LL2 Torque spec out으로 ball screw 교체 요청 | INTEGER plus, Ball Screw 교체  |
| 5   | myservice_psk.csv | 40151867  | IPC3711  | LL1 INDEXER ABNORMAL ALARM 발생으로 Ball Screw 교체 | INTEGER plus, Ball Screw 교체  |
| 6   | myservice_psk.csv | 40128306  | DPCP816  | LL2 INDEXER MOVE 중 Ball Screw 교체 | INTEGER plus, Ball Screw 교체  |
| 7   | myservice_psk.csv | 40172399  | DPCJL09  | Ball Screw Fail 후 파트 수급 후 교체 요청 | INTEGER plus, Ball Screw 교체  |
| 8   | myservice_psk.csv | 40147335  | DPCHZ07  | LL1 Torque spec out으로 ball screw 교체 요청 | INTEGER plus, Ball Screw 교체  |
| 9   | myservice_psk.csv | 40043188  | DPCG11   | LL Ball Screw Fail로 인한 교체 | INTEGER plus, Ball Screw 교체  |
| 10  | myservice_psk.csv | 40164837  | DPCJL06  | Ball Screw Fail 후 점검 및 교체 | INTEGER plus, Ball Screw 교체  |
| 11  | myservice_psk.csv | 40112705  | DPCP816  | Ball Screw broken으로 인한 교체 | INTEGER plus, Ball Screw 교체  |
| 12  | myservice_psk.csv | 40156494  | DPCJL07  | LL2 Indexer Torque Spec Out으로 점검 및 교체 | INTEGER plus, Ball Screw 교체  |
| 13  | myservice_psk.csv | 40164204  | DPCB809  | Torque값 Spec Out으로 LL1 Ball Screw 교체 | INTEGER plus, Ball Screw 교체  |
| 14  | myservice_psk.csv | 40149085  | DPCP811  | Torque Spec Over로 인한 Ball Screw 교체 | INTEGER plus, Ball Screw 교체  |
| 15  | myservice_psk.csv | 40163187  | DPCP816  | Torque값 Spec Out으로 LL2 Ball Screw 교체 및 Teaching 점검 | INTEGER plus, Ball Screw 교체  |
| 16  | myservice_psk.csv | 40154717  | DPCJL04  | LL1 Torque 200%로 LL1, LL2 Ball Screw 교체 및 변경점 언급 | INTEGER plus, Ball Screw 교체, 변경점 포함 |
| 17  | myservice_psk.csv | 40141937  | DPCB707  | FDC INTERLOCK, LL1 TORQUE 문제로 Ball Screw 교체 | INTEGER plus, Ball Screw 교체  |
| 18  | myservice_psk.csv | 40127428  | DPCJF03  | LL2 Ball Screw 파손으로 인한 교체 | INTEGER plus, Ball Screw 교체  |
| 19  | myservice_psk.csv | 40149621  | IPC3713  | LL1 부하로 인한 Ball Screw 교체 | INTEGER plus, Ball Screw 교체  |
| 20  | myservice_psk.csv | 40148523  | IPC3715  | Part Fail로 인한 LL2 Ball Screw 교체 | INTEGER plus, Ball Screw 교체  |
| 21  | myservice_psk.csv | 40147287  | DPCB712  | 고객 요청으로 LL2 Ball Screw 교체 | INTEGER plus, Ball Screw 교체  |
| 22  | myservice_psk.csv | 40140139  | DPCB801  | LL2 Moving TIME OUT으로 Ball Screw 교체 | INTEGER plus, Ball Screw 교체  |
| 23  | myservice_psk.csv | 40087861  | DPCJF03  | LL2 Elevator Assy 소음 점검, 변경점 없음 | INTEGER plus, Ball Screw 관련, 변경점 없음 |
| 24  | myservice_psk.csv | 40135625  | IPC3801  | Ball screw pulley 전수 검사 | INTEGER plus, Ball Screw 관련  |
| 25  | myservice_psk.csv | 40143046  | IPC3801  | Part 수급 후 Ball Screw 교체 진행 | INTEGER plus, Ball Screw 교체  |
| 26  | myservice_psk.csv | 40135626  | IPC3801  | Ball screw pulley 전수 검사 | INTEGER plus, Ball Screw 관련  |
| 27  | myservice_psk.csv | 40136044  | DPCB701  | P2 Line 전 설비 LL1,2 Ball Screw PM 교체 | INTEGER plus, Ball Screw PM 교체 |
| 28  | myservice_psk.csv | 40178931  | DPCP805  | LL2 INDEXER ABNORMAL POSITION 발생으로 Ball Screw 교체 | INTEGER plus, Ball Screw 교체  |
| 29  | myservice_psk.csv | 40135134  | DPCB809  | ALARM 발생으로 Ball Screw 교체 | INTEGER plus, Ball Screw 교체  |
| 30  | myservice_psk.csv | 40135136  | DPCB809  | Part Fail로 인한 Ball Screw 교체 | INTEGER plus, Ball Screw 교체  |
| 31  | myservice_psk.csv | 40164224  | DPCP803  | Part Fail로 LL1 Ball Screw 교체 | INTEGER plus, Ball Screw 교체  |
| 32  | myservice_psk.csv | 40082665  | DPCHZ07  | DPCHZ11호기 part 차용 후 Ball Screw 장착 | INTEGER plus, Ball Screw 교체  |
| 33  | myservice_psk.csv | 40136050  | IPC3701  | LL2 Indexer Move Time Out으로 Ball Screw 교체 | INTEGER plus, Ball Screw 교체  |
| 34  | myservice_psk.csv | 40137040  | IPC3710  | Ball Screw Part Fail 관련 교체 | INTEGER plus, Ball Screw 교체  |
| 35  | myservice_psk.csv | 40124953  | IPC3712  | Ball Screw 현업 선 대응 후 재고 요청 | INTEGER plus, Ball Screw 교체  |
| 36  | myservice_psk.csv | 40147095  | DPCB706  | 고객 요청으로 Ball Screw Part 교체 | INTEGER plus, Ball Screw 교체  |
| 37  | myservice_psk.csv | 40141936  | DPCB707  | FDC INTERLOCK, LL1 TORQUE 문제로 Ball Screw 점검 | INTEGER plus, Ball Screw 관련  |
| 38  | myservice_psk.csv | 40143046  | IPC3801  | Ball Screw 교체 후 작업 진행 | INTEGER plus, Ball Screw 교체  |
| 39  | myservice_psk.csv | 40135625  | IPC3801  | Ball Screw pulley 전수 검사 | INTEGER plus, Ball Screw 관련  |
| 40  | myservice_psk.csv | 40164837  | DPCJL06  | Ball Screw Fail 점검 및 교체 | INTEGER plus, Ball Screw 교체  |

요약:  
- 총 40건의 INTEGER plus Ball Screw 교체 관련 이력 확인됨  
- 대부분 Torque Spec Out, Fail, Alarm 발생 후 교체 진행  
- 일부 문서에서 변경점(변경) 언급 포함됨 (예: 문서 4, 16)  
- 교체는 LL1, LL2 Ball Screw 대상이며, 부하 상승, 파손, 알람 등이 주요 원인임
```

</details>

## Sample 09 — history_lookup / m (58줄)

- **Timestamp**: `2026-01-19 15:19:36`
- **Question** (40자):

  ```
  supra n의 wafer broken issue 이력을 표로 정리해줘ㅏ
  ```
- **Sources (Top 50)**: first 5 = ['scraped_gcb.json', 'scraped_gcb.json', 'myservice_psk.csv', 'scraped_gcb.json', 'scraped_gcb.json']
- **Structure flags**: table=Y kw_block=. summary=. cause=. gcb_footer=Y part_num=.
- **First line**: `아래는 SUPRA N 모델의 wafer broken 이력들을 주요 정보별로 정리한 표입니다.`

<details><summary>Reference answer (PII masked)</summary>

```markdown
아래는 SUPRA N 모델의 wafer broken 이력들을 주요 정보별로 정리한 표입니다.

| 번호 | 날짜       | Equip ID | Order No. | 주요 이슈 내용 요약                          | 조치 및 상태 요약                                  | 문서 출처          |
|-------|------------|----------|-----------|---------------------------------------------|--------------------------------------------------|--------------------|
| 1     | 2020-02-06 | EPR081   | -         | PM3 Wafer Pick Up 문제                       | 추가 문의사항 없음, Close                          | scraped_gcb.json   |
| 2     | 2020-09-11 | EPR081   | -         | Chuck Fail로 인한 Wafer Broken 추정         | 모니터링 후 현상 해결로 Close                      | scraped_gcb.json   |
| 3     | 2021-06-19 | EPAP809  | 40054560  | PM3 CH1 Wafer Broken 발생                    | Process Kit 교체, Wafer 조각 회수, 내부 Clean 진행 | myservice_psk.csv  |
| 4     | 2016-04-27 | EPRE09   | -         | S/W 또는 H/W 알람 발생 확인 중               | -                                                | scraped_gcb.json   |
| 5     | 2023-01-19 | -        | -         | D1A/D1Y Wafer 막질 문제                      | Recipe 변경 검토 중, 고객사 내부 보안 확인 중      | scraped_gcb.json   |
| 6     | 2020-08-18 | EPAP812  | 40038208  | PM1 Wafer Broken으로 인한 Clean 진행         | Baffle/Focus 탈착, 내부 및 Top Lid Clean 진행      | myservice_psk.csv  |
| 7     | 2016-12-28 | -        | -         | 17L View Port Quartz Broken                  | 고객 검토 후 PM업체 공유 예정                       | scraped_gcb.json   |
| 8     | 2020-06-23 | EPAP333  | 40035050  | Cooling 중 Wafer Broken 발생                  | Alarm 미발생, Broken Wafer LP2 이동                 | myservice_psk.csv  |
| 9     | 2020-08-06 | EPAP811  | 40037564  | PM2 CH2 Wafer Broken 후 Pin 점검 요청        | Pin 높이 조정, Chuck Clean 등 진행                  | myservice_psk.csv  |
| 10    | 2021-05-06 | EPAM33   | -         | Wafer broken sensor NG Issue                  | Sensor cable path 점검                              | scraped_gcb.json   |
| 11    | 2020-08-07 | EPAP812  | 40037704  | PM2 CH1 Wafer Broken으로 인한 Clean 요청     | VAC HOSE로 Clean, Process Kit 탈착 및 Clean 진행    | myservice_psk.csv  |
| 12    | 2019-09-17 | -        | -         | EFEM Robot End Effector Broken Issue          | 동일 이슈 통합 관리, 고객 요청 PSK 개선             | scraped_gcb.json   |
| 13    | 2025-04-23 | EPAL116  | 40155662  | PM2 Wafer Broken, 고객 요청 Chamber Clean    | EQ 다운, Wafer Clear 및 PM2 Clean 진행              | myservice_psk.csv  |
| 14    | 2017-09-27 | EPAJF11  | -         | OP Miss로 S/W 개발 미진행, 안건 Close         | Wafer Alignment 기능 강화, Robot Pick Up Error 대응 | scraped_gcb.json   |
| 15    | 2022-12-22 | EPAP817  | 40092101  | PM1 Processkit 교체 (Wafer Broken)            | Pin 점검 이상 없음, Processkit 교체                  | myservice_psk.csv  |
| 16    | 2021-05-06 | EPAM33   | -         | Wafer broken sensor NG Issue                  | Sensor cable path 점검                              | scraped_gcb.json   |
| 17    | 2019-07-14 | EPAP814  | 40008962  | PM3 Wafer Broken, Thermal Stress 추정         | Wafer Broken 후 Robot Upper Arm으로 이동             | myservice_psk.csv  |
| 18    | 2021-02-10 | WASE101  | 40048672  | 고객 지원 Wafer Broken                         | 고객 지원 완료                                      | myservice_psk.csv  |
| 19    | 2023-01-30 | EPAP825  | 40094836  | PM1 ch2 Wafer Broken 발생                      | Chamber Clean 완료, Pin 이상 없음                    | myservice_psk.csv  |
| 20    | 2022-12-21 | EPAP811  | 40092071  | PM2 CH1 Wafer Broken 재발                       | Recipe 및 broken 시점 상이, 점검 진행                | myservice_psk.csv  |
| 21    | 2022-11-03 | EPAP720  | 40088224  | PM2 Wafer Broken으로 Pin 점검 및 Process Kit Rep | Pin 정상, Broken Wafer 회수 완료                      | myservice_psk.csv  |
| 22    | 2022-10-25 | EPAP804  | 40087639  | PM2 Wafer Broken으로 Pin 점검 및 Clean         | Pin 정상, Process Kit 교체 완료                       | myservice_psk.csv  |
| 23    | 2023-03-28 | EPRC80   | 40099521  | PM1 CH1 Wafer Broken 발생                       | 세부 사항 확인 불가, Lot Flow History 확인 중         | myservice_psk.csv  |
| 24    | 2020-07-21 | EPR081   | 40036573  | PM3 ST1 Wafer Broken 관련 Chuck 교체           | Wafer Broken Log 확인, Chuck 교체 진행               | myservice_psk.csv  |
| 25    | 2023-03-29 | EPAP811  | 40099662  | PM1 Process Kit 교체 공정 Issue (Wafer Broken) | PM1 Disable, Process Kit 탈착 및 Clean 진행           | myservice_psk.csv  |
| 26    | 2024-09-11 | EPAP811  | 40137678  | PM1 Process Kit 교체 공정 Issue (Wafer Broken) | PM1 Disable, Process Kit 탈착 및 Clean 진행           | myservice_psk.csv  |
| 27    | 2024-09-11 | EPAP813  | 40137680  | PM1 Process Kit 교체 공정 Issue (Wafer Broken) | PM1 Disable, Process Kit 탈착 및 Clean 진행           | myservice_psk.csv  |
| 28    | 2024-12-12 | EPAJF12  | 40144988  | Wafer Broken Log 분석 요청                      | Lot Flow, OP Log 확인, 고객 미팅 예정                 | myservice_psk.csv  |
| 29    | 2022-11-29 | EPAP820  | 40090233  | PM1 Wafer Broken으로 Pin 점검 및 Process Kit Rep | Pin 정상, Process Kit 교체 진행                       | myservice_psk.csv  |
| 30    | 2025-01-19 | EPA3702  | 40148215  | PM3 Wafer Broken                                | Pin 정상, TM 공통부 Clean, 설비 정상화                | myservice_psk.csv  |
| 31    | 2021-08-30 | EPAM03   | 40058808  | Wafer Broken, 고객 요청 점검                    | Wafer 회수, PM3 Disable, Broken Wafer Pick 진행       | myservice_psk.csv  |
| 32    | 2022-11-06 | EPAP805  | 40088396  | PM3 Wafer Broken으로 Pin 점검 및 Process Kit Rep | Pin 정상, Broken Wafer 회수 완료                       | myservice_psk.csv  |
| 33    | 2019-04-17 | EPAG21   | 40002471  | PM1 CH2 Chamber 내부 Wafer Broken 점검          | Thermal Stress 추정, Wafer Broken 발생                 | myservice_psk.csv  |
| 34    | 2025-12-12 | EPAM18   | 40174619  | TM robot place fail alarm, PM3 Wafer Broken     | PM3 Disable, Chamber Open, Wafer Broken 2매 확인       | myservice_psk.csv  |
| 35    | -          | EPAB328  | -         | SUPRA N Baffle Screw Broken 발생 건             | -                                                | scraped_gcb.json   |
| 36    | 2020-09-12 | EPAP701  | 40039684  | PM2 Wafer Broken으로 Process Kit 교체 및 Clean  | Baffle/Focus 탈착, 내부 및 Top Lid Clean 진행          | myservice_psk.csv  |
| 37    | 2022-03-21 | EPAP804  | 40070570  | PM2 Process Kit 교체 (Wafer Broken)              | Chamber, Top Lid Clean, Process Kit 교체, Chuck Clean  | myservice_psk.csv  |
| 38    | 2024-10-03 | WASB147  | 40139253  | PM1 CH2 Wafer Broken                             | 고객 매뉴얼 Clean 지원, TM robot teaching 점검         | myservice_psk.csv  |
| 39    | 2025-11-20 | EPAJL53  | 40172353  | PM3 CH2 Wafer Broken 2매 발생                    | 설비 점검 및 Clean 요청                               | myservice_psk.csv  |
| 40    | 2020-05-28 | EPR303   | 40033724  | PM2 CH1 Wafer Broken 발생                         | Door 쪽 Pin 높이 확인, Wafer Broken 발생               | myservice_psk.csv  |
| 41    | 2025-05-19 | EPAP801  | 40157570  | PM1 Process Kit 교체 공정 Issue (Wafer Broken)   | PM1 Disable, Process Kit 탈착 및 Clean 진행             | myservice_psk.csv  |
| 42    | 2023-04-30 | EPAP841  | 40102046  | PM2 Wafer Broken 발생                             | Wafer 회수, Pin 및 Process Kit 정상, Leak check 완료    | myservice_psk.csv  |
| 43    | 2023-04-20 | EPAP820  | 40101378  | PM2 CH1 Wafer Broken 발생                         | Wafer 회수, Pin 및 Process Kit 정상, Leak check 완료    | myservice_psk.csv  |
| 44    | 2023-01-26 | EPAMU1   | 40094720  | PM1 to BM2 이동 중 Wafer Broken                   | EQ 다운, TM 모듈 Wafer Broken, 내부 Clean 진행          | myservice_psk.csv  |
| 45    | 2023-02-20 | EPAP802  | 40096604  | PM3 ch2 Wafer Broken 발생                         | Wafer 회수, Chamber Clean, Process Kit 교체             | myservice_psk.csv  |
| 46    | 2019-06-23 | EPAP331  | 40007160  | PM1 Wafer Broken, Thermal Stress 추정             | Wafer Broken 발생, 고객사 테스트 중                      | myservice_psk.csv  |
| 47    | 2022-10-15 | EPAP801  | 40086883  | PM3 Wafer Broken으로 인한 Clean                   | Wafer 회수 및 Clean, Process Kit 교체, Chuck Clean       | myservice_psk.csv  |
| 48    | 2021-11-18 | EPAM04   | 40063533  | Cooling stage1에서 Wafer Broken 발생              | Broken wafer Clean, TM 및 Cooling stage1 Clean 진행     | myservice_psk.csv  |
| 49    | 2022-01-06 | WASB102  | 40066537  | PM1-1 Wafer Broken                                | PM1 PM, Pin Height 및 Teaching Position 조정            | myservice_psk.csv  |
| 50    | 2019-12-28 | EPAP821  | 40024877  | PM2 WAFER BROKEN으로 인한 점검                    | PM2 공정 중 Wafer Broken 발생, Clean 및 점검 요청        | myservice_psk.csv  |

필요시 특정 이력에 대해 상세 내용 추가 가능합니다.

해결되지 않으면 GCB에 등록하세요. 문의: [NAME] ([EMAIL])
```

</details>

## Sample 10 — location_inquiry / s (18줄)

- **Timestamp**: `2026-01-22 13:19:43`
- **Question** (1611자):

  ```
  질문별 문서 위치 및 목적 안내
  
  질문 내용	문서명 및 위치	문서 종류	문서 목적 및 설명
  
  SUPRA XP PRISM Source 교체 시 주의사항 중 옳지 않은 것	global sop_supra xp_all_pm_prism source.pdf (PM Open 주의사항)	SOP	PRISM Source 교체 시 작업 절차, 안전 주의사항 및 환경 안전 확보 목적
  
  Matcher 관련 설명 중 올바르지 않은 것	global sop_supra xp_all_pm_prism source.pdf (Matcher Cable, power 절차)	SOP	Matcher 전기 연결 및 분리 절차, 안전 및 장비 보호 목적
  
  TM End-effector O-ring 교체 시 TM Maint mode 변경 없이 TM Toplid open 진행 시 문제점	Set_Up_Manual_ZEDIUS XP(SUPRA XP).pdf (3.4 TM Robot End Effector 장착)	Set Up Manual	TM Robot End Effector 교체 절차 및 안전 작업 방법 안내
  
  TM Robot AWC Log Data 해석	문서 내 별도 정보 없음	-	The provided reference materials do not contain information related to this question.
  
  SUPRA XP OP Log 설명 중 올바르지 않은 것	문서 내 별도 정보 없음	-	The provided reference materials do not contain information related to this question.
  
  ZEDIUS XP PM Parameter 설명 중 옳은 것 2개 선택	global sop_supra xp_all_pm_prism source 3000qc.pdf (PM Maint Mode, Leak Check)	SOP	PM 유지보수 모드 전환 및 Leak Rate Spec 준수, 장비 정상 작동 및 안전 확보 목적
  
  SUPRA XP EFEM Manual Screen 설명 중 올바른 것	global sop_supra xp_all_pm_prism source 3000qc.pdf (PM Manual Operation Screen)	SOP	EFEM Manual Screen 사용법 및 작업 전 상태 확인 절차 안내
  
  SUPRA XP PM Pendulum V/V PM 절차 올바른 순서	global sop_supra xp_all_pm_prism source.pdf (Flow Chart 및 절차)	SOP	Pendulum V/V PM 절차 순서 및 작업 포인트 안내, 안전 및 품질 확보 목적
  
  SUPRA XP PM Pin 구동 설명 중 옳지 않은 것	global sop_supra xp_all_pm_prism source.pdf (PM Pin 분리 절차)	SOP	PM Pin 구동 관련 분해 조립 절차 및 안전 온도 조건 안내
  
  SUPRA XP PM 설명 중 옳지 않은 것	global sop_supra xp_all_pm_prism source.pdf (PM Disable 및 Maint Mode 절차)	SOP	PM 작업 시 안전 및 장비 보호를 위한 절차 및 모드 변경 안내
  
  
  
  이부분을 영어로 번역해줘
  참고 문서: 사용 안 함 또는 찾지 못함
  ```
- **Sources (Top 0)**: first 5 = []
- **Structure flags**: table=Y kw_block=. summary=. cause=. gcb_footer=Y part_num=.
- **First line**: `| Question Summary                                                              `

<details><summary>Reference answer (PII masked)</summary>

```markdown
| Question Summary                                                                                  | Document Name and Location                                             | Document Type    | Document Purpose and Description                                                                                   |
|-------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|------------------|--------------------------------------------------------------------------------------------------------------------|
| Incorrect caution when replacing SUPRA XP PRISM Source                                          | global sop_supra xp_all_pm_prism source.pdf (PM Open Caution)         | SOP              | Work procedure, safety precautions, and environmental safety assurance during PRISM Source replacement             |
| Incorrect explanation related to Matcher                                                        | global sop_supra xp_all_pm_prism source.pdf (Matcher Cable, power procedure) | SOP              | Procedures for Matcher electrical connection and disconnection, safety and equipment protection                    |
| Issues when opening TM Toplid without changing TM Maint mode during TM End-effector O-ring replacement | Set_Up_Manual_ZEDIUS XP(SUPRA XP).pdf (3.4 TM Robot End Effector mounting) | Set Up Manual    | TM Robot End Effector replacement procedure and safety work instructions                                           |
| Interpretation of TM Robot AWC Log Data                                                        | -                                                                    | -                | No information available in the provided reference materials                                                        |
| Incorrect explanation of SUPRA XP OP Log                                                       | -                                                                    | -                | No information available in the provided reference materials                                                        |
| Correct explanations about ZEDIUS XP PM Parameters (select 2 correct)                           | global sop_supra xp_all_pm_prism source 3000qc.pdf (PM Maint Mode, Leak Check) | SOP              | PM maintenance mode switching and Leak Rate Spec compliance for normal operation and safety assurance             |
| Correct explanation of SUPRA XP EFEM Manual Screen                                             | global sop_supra xp_all_pm_prism source 3000qc.pdf (PM Manual Operation Screen) | SOP              | Instructions for using EFEM Manual Screen and pre-operation status check procedure                                  |
| Correct sequence of SUPRA XP PM Pendulum V/V PM procedure                                      | global sop_supra xp_all_pm_prism source.pdf (Flow Chart and procedure) | SOP              | Guidance on PM Pendulum V/V procedure sequence, work points, safety, and quality assurance                          |
| Incorrect explanation about SUPRA XP PM Pin operation                                          | global sop_supra xp_all_pm_prism source.pdf (PM Pin disassembly procedure) | SOP              | Disassembly and assembly procedures related to PM Pin operation and safety temperature conditions                  |
| Incorrect explanations about SUPRA XP PM                                                       | global sop_supra xp_all_pm_prism source.pdf (PM Disable and Maint Mode procedure) | SOP              | Procedures and mode changes for safety and equipment protection during PM work                                      |

---

This table is a direct English translation of the previously provided “Question Summary and Document Location and Purpose” table, reflecting the exact document names, types, and purposes as per the reference materials and previous conversation context.

해결되지 않으면 GCB에 등록하세요. 문의: [NAME] ([EMAIL])
```

</details>

## Sample 11 — location_inquiry / m (45줄)

- **Timestamp**: `2026-01-09 13:53:41`
- **Question** (69자):

  ```
  GENEVA설비에서 2개의 vac Sol valve가 있는데 exhaust 가 각각 어느 port랑 연결되어 있는지 확인해줘
  ```
- **Sources (Top 20)**: first 5 = ['setupmanual.json', 'mySERVICE_pskh.csv', 'mySERVICE_pskh.csv', 'mySERVICE_pskh.csv', 'setupmanual.json']
- **Structure flags**: table=. kw_block=Y summary=Y cause=. gcb_footer=Y part_num=.
- **First line**: `GENEVA STP300 xp 설비에서 2개의 vac Solenoid valve의 exhaust 포트 연결 위치에 대해 문의주셨습니다.`

<details><summary>Reference answer (PII masked)</summary>

```markdown
GENEVA STP300 xp 설비에서 2개의 vac Solenoid valve의 exhaust 포트 연결 위치에 대해 문의주셨습니다.

현재 제공된 참고자료 내에는 해당 vac Solenoid valve 각각의 exhaust 포트가 구체적으로 어느 포트와 연결되어 있는지에 대한 명확한 정보가 포함되어 있지 않습니다. 즉, 각 vac Sol valve의 exhaust가 어느 포트(번호)와 연결되는지에 대한 설비 도면, 밸브 모델별 포트 기능, 또는 배관 연결 상세 내용이 문서에 명시되어 있지 않습니다.

---

### 참고자료 기반 안내

- 일반적으로 solenoid valve의 포트 번호는 다음과 같이 표기됩니다:
  - 1 (P) : 압축공기 공급 포트
  - 2 (A), 4 (B) : 진공 대상(척, 패드 등) 연결 포트
  - 3 (R), 5 (R) : Exhaust (배기) 포트

- 하지만 GENEVA 설비 vac sol valve의 경우, 밸브 제조사 및 모델명, 포트별 배관 연결 상태에 따라 exhaust 포트가 달라질 수 있으므로, 정확한 확인을 위해서는 다음 정보가 필요합니다:
  1. Solenoid valve 본체에 적힌 **제조사 및 모델명**
  2. 밸브 몸체에 새겨진 포트 번호 및 기호 (예: 1, 2, 3, 5 또는 P, A, B, R)
  3. 현재 배관 연결 상태 (어느 포트에 어떤 배관이 연결되어 있는지)

- 참고자료 내 GENEVA STP300 xp 설비 관련 solenoid valve 점검 및 교체 사례는 있으나, exhaust 포트 연결 위치에 대한 구체적 언급은 없습니다[1,8,12].

---

### 결론 및 권고

- 현재 자료만으로는 각 vac Solenoid valve의 exhaust 포트가 어느 포트와 연결되어 있는지 정확히 안내드리기 어렵습니다.
- 설비 내 solenoid valve 본체의 제조사 및 모델명, 포트 번호를 확인하시고, 설비 공압 배관 도면 또는 밸브 사양서를 참조하시기 바랍니다.
- 만약 도면이나 밸브 라벨 정보가 확보되면, 추가로 알려주시면 상세 분석 및 안내가 가능합니다.

---

### 추가 확인 요청 사항

1. vac Solenoid valve 본체에 부착된 제조사 및 모델명 확인
2. 각 포트에 새겨진 번호 및 기호 확인
3. 배관 연결 상태(어느 포트에 어떤 배관이 연결되어 있는지) 간단 설명

---

해결되지 않으면 GCB에 등록하고 질문하세요.  
문의: [NAME] ([EMAIL])

---

**참고문서 핵심 키워드:**  
GENEVA STP300 xp, vac Solenoid valve, exhaust port, port 번호, sol valve 점검, sol valve 교체, pneumatic valve, solenoid valve port function[1,8,12]
```

</details>

## Sample 12 — procedure / m (22줄)

- **Timestamp**: `2026-03-08 15:39:53`
- **Question** (79자):

  ```
  mySITE에서 Setup service order 내의 Maintenance task에서 사용한 history part를 보는 방법을 알려줘
  ```
- **Sources (Top 11)**: first 5 = ['myservice_psk.jsonl', 'myservice_psk.jsonl', 'myservice_psk.jsonl', 'myservice_psk.jsonl', 'myservice_psk.jsonl']
- **Structure flags**: table=. kw_block=. summary=. cause=. gcb_footer=Y part_num=.
- **First line**: `mySITE에서 Setup service order 내 Maintenance task에서 사용한 history part를 확인하는 방법은 다음과`

<details><summary>Reference answer (PII masked)</summary>

```markdown
mySITE에서 Setup service order 내 Maintenance task에서 사용한 history part를 확인하는 방법은 다음과 같습니다.

1. **Setup service order 내 Maintenance task 진입**  
   - mySITE에서 해당 Setup service order를 열고, Maintenance task 목록을 확인합니다.

2. **History part 확인 방법**  
   - 각 Maintenance task 내에서 Part 입력 시, 과거 교체 이력(History part)이 기록되어 있습니다.  
   - 예를 들어, Doc 1, Doc 2 사례처럼 Part N.G로 인한 교체 미진행 시에도 과거 Replacement Task로 재상신하며, Part 입력 시 History로 입력하는 방식으로 관리됩니다.  
   - 따라서, Maintenance task 상세 내 Part 관련 항목을 보면 사용한 History part 정보를 확인할 수 있습니다.

3. **구체적 확인 절차**  
   - 작업 상세 내 Replacement, Repair, Adjustment 등 Task Code별로 Part No.와 Part Name이 기록되어 있습니다.  
   - 작업 내역(Action) 및 결과(Result) 항목에서 Part 교체 또는 사용 내역이 기술되어 있으므로, 이를 통해 History part 사용 여부를 파악할 수 있습니다.  
   - 예: Doc 6 AEASK01 작업에서는 Part No. C1600051, Part Name ASTEX, 3.0kw SMARTMATCHER가 History part로 사용된 사례가 명확히 기록되어 있습니다.

4. **주의사항**  
   - Part code가 없는 3rd party part는 History에 작성 불가할 수 있으니 참고하세요(Doc 10 사례).  
   - Task Code 수정 시 작업일자 기입 불가 문제로 기존 Replacement Task를 재상신하는 경우도 있으니, History part 확인 시 Task Code와 작업일자도 함께 확인하는 것이 좋습니다.

요약하면, **mySITE Setup service order 내 Maintenance task 상세 화면에서 Part No.와 Part Name, Task Code, 작업 내역(Action) 및 결과(Result) 항목을 확인하면 사용한 History part를 볼 수 있습니다**[1,2,6].

해결되지 않으면 GCB에 등록하세요. 문의: [NAME] ([EMAIL])
```

</details>

## Sample 13 — procedure / m (32줄)

- **Timestamp**: `2026-01-02 13:20:48`
- **Question** (62자):

  ```
  INTEGER plus safety controller, safety module교체 작업을 한 이유를 알려줘.
  ```
- **Sources (Top 20)**: first 5 = ['myservice_psk.csv', 'scraped_sop.json', 'scraped_sop.json', 'myservice_psk.csv', 'scraped_sop.json']
- **Structure flags**: table=. kw_block=Y summary=Y cause=. gcb_footer=Y part_num=Y
- **First line**: `INTEGER plus 설비의 Safety Controller 및 Safety Module 교체 작업은 주로 설비의 안전 관련 알람 및 Faul`

<details><summary>Reference answer (PII masked)</summary>

```markdown
INTEGER plus 설비의 Safety Controller 및 Safety Module 교체 작업은 주로 설비의 안전 관련 알람 및 Fault 발생, 그리고 기존 Safety Controller의 단종 및 내부 Logic 문제 대응을 위해 수행되었습니다.

### 교체 작업 이유

1. **Safety Module Fail 및 Fault 발생**
   - INTEGER plus 설비에서 Safety Module Fail, Safety Module Fault 알람이 발생하여 설비가 Shut Down 되는 현상이 보고되었습니다.
   - 예를 들어, AM Safety Module Fault로 설비 Turn On 불가, Safety Module Fail 추정으로 교체 예정인 사례가 있었습니다[4].
   - PM Safety Module Fault가 반복적으로 발생하여 Safety Module 교체가 필요했던 사례도 있습니다[7,8].

2. **Safety Controller 단종 및 대체품 적용**
   - 기존 Pilz Safety Controller가 단종되어 OMRON Safety Controller로 대체하는 작업이 진행되었습니다.
   - 이 과정에서 PM ELT Box 내부 적용 부품 번호가 업데이트 되었고, 전체 Set 교체가 권장되었습니다.
   - 단일 Unit Fail 판단이 어려워 전체 Safety Controller Set 교체가 가장 적절한 방법으로 판단되었습니다[2,10].

3. **내부 Logic 문제 및 통신 오류 대응**
   - Safety Controller 내부 Logic에 Fault가 있어 이를 개선하기 위한 교체 작업이 필요했습니다.
   - Safety Module 조립 순서 오류, 통신 설정 오류(예: FSoE 슬레이브 주소 에러), 안전 입력 불일치 에러 등이 발생하여 정상 동작을 위해 교체 및 재조립이 요구되었습니다[10].

4. **Safety Module Chattering 현상 개선**
   - Pilz에서 OMRON으로 교체 시 Safety Module Chattering 현상이 발생하여 개선품으로 교체하는 작업이 있었습니다[10].

### 요약
- INTEGER plus 설비에서 Safety Module 및 Safety Controller 교체는 설비 안전 관련 Fault 및 알람 발생, 단종에 따른 대체품 적용, 내부 Logic 및 통신 오류 문제 해결, 그리고 Chattering 현상 개선을 위해 수행되었습니다.
- 교체 시에는 OMRON 제조사의 Safety Controller 부품(NX-SL3300, NX-SID800, NX-SOD400 등)을 사용하며, 전체 Set 교체가 권장됩니다[2,10].
- 교체 작업은 Global SOP에 따라 안전 보호구 착용, 작업 공간 확보, Cable 분리 시 Short 주의, 고정 브라켓 탈착 등의 절차를 준수하여 진행합니다[1,3,4,8,9].

해결되지 않으면 GCB에 등록하세요. 문의: [NAME] ([EMAIL])

---

**참고 문서 핵심 키워드:**  
INTEGER plus, Safety Controller, Safety Module, Fail, Fault, OMRON, Pilz, 단종, 교체, 내부 Logic, 통신 오류, Chattering, Global SOP, NX-SL3300, NX-SID800, NX-SOD400[1,2,4,10]
```

</details>

## Sample 14 — procedure / l (104줄)

- **Timestamp**: `2026-01-29 13:05:23`
- **Question** (64자):

  ```
  SUPRA N 에서 TM ROBOT 교체 순서를 자세하게 알려줘. 또한 3개월 간 교체 간 발생한 이슈도 정리해줘.
  ```
- **Sources (Top 508)**: first 5 = ['scraped_sop.json', 'scraped_sop.json', 'scraped_sop.json', 'scraped_sop.json', 'scraped_sop.json']
- **Structure flags**: table=Y kw_block=Y summary=. cause=. gcb_footer=Y part_num=.
- **First line**: `## 1. SUPRA N TM ROBOT 교체 순서 (Global SOP 기준)`

<details><summary>Reference answer (PII masked)</summary>

```markdown
## 1. SUPRA N TM ROBOT 교체 순서 (Global SOP 기준)

### 1) 작업 준비 및 안전사항
- 작업자 간 Inform Miss로 인한 Robot 구동 시 충돌 및 손가락·신체 협착 주의
- 고온에 의한 화상 주의, 중량물(5kg 이상) 취급 시 2인 이상 작업 권장
- SFEM 내부 작업 시 Robot 구동에 의한 충돌 및 협착 주의
- 안전모, 보안경, 토시, 앞치마, 방독마스크, 보호가운 등 환경안전보호구 착용 필수
- 장비 동작 중 신체 일부 끼임, 충격 위험 경고 라벨 준수

### 2) 작업 인원 및 시간
- 작업 전: 3명, 30분 (Back up, Home)
- 교체 작업: 3명, 540분 (약 9시간)
- 작업 후: 3명, 30분 (Home)

### 3) 필요 Tool
- 드라이버(+,-), L 렌치, 레이저 지그, 버퍼 지그, 세정용 와이퍼, 레벨러 등

### 4) 교체 작업 상세 순서 (Flow Chart 및 Work Procedure)
1. Global SOP 및 안전 사항 확인
2. 설비 Arrange (고객사 Maint에 요청, Host Communication 상태 변경: Enable → Disable, AMHS Port PIO 상태 변경: Auto → Manual, EFEM Part Home 클릭)
3. Teaching Data 백업 (Robot Controller 내 데이터 백업)
4. EFEM Robot Home 위치 이동
5. SFEM Side Door 오픈 및 BKT 제거
6. EFEM Robot Controller 전원 OFF 및 MCB OFF (CP 5-2 Turn OFF 확인 필수)
7. EFEM Robot End Effector 분리 (Screw 분실 주의, M3x4 4ea)
8. Punching Panel 제거 (Screw 6ea M4x10)
9. Robot Controller Cable 분리 (Ethernet, Serial, Pendant, AC Power, Interlock, DC24V I/O, Encoder 등)
10. 기존 Robot Controller 분리 (Screw 분실 주의, M8x30 Wrench 4ea, 중량물 2인 작업)
11. EFEM Robot Undocking (Cable, Air Hose 분리, 중량물 2인 작업)
12. Robot Docking (Docking 부 청소 후 진행)
13. 신규 Robot Controller 설치 및 고정 (Screw 분실 주의)
14. Cable 연결 및 설치 (Cable 명칭 및 연결 위치 확인 필수)
15. Punching Panel 설치 및 Screw 체결
16. EFEM Robot End Effector 장착 및 간격 확인 (End Effector Gap 10 ± 1mm, M3x4 4ea)
17. EFEM Robot Controller 전원 ON
18. Robot Body Leveling (플랜지 및 베이스 기준 Level Spec 확인, Level Screw 조정, M8x30 Wrench 4ea, Level Spec 1 칸 = 0.5mm)
19. End Effector Level 조정 (상부 및 하부 볼트 분리 후 Level 조정, Spec 10 ± 1mm, M3x10, M4x12, M5x4 Wrench 사용)
20. Load Port 별 Robot Leveling (LP Wafer Pick Up Ready Position Level 확인)
21. SFEM Side Door 닫고 BKT 설치
22. Teaching Data 입력 (Back-Up 데이터 복원)
23. Load Port Foup Docking
24. Pendant Teach Mode 진입 및 Teaching 수행 (Z-Axis Teaching 시 End Effector 간섭 주의)
25. Buffer Stage Teaching, Cooling Stage Teaching 진행
26. Auto Moving Test (Pick & Place, Buffer, LP Place Test)
27. Aging Test 진행 (Buffer/Cooler 선택)
28. Pendant 제거
29. Servo On 및 EFEM Interlock Reset
30. 설비 정상화 및 주변 정리
31. 작업 완료 후 고객 Inform 및 Aging Test 완료 확인

### 5) 작업 Check Sheet 주요 항목
- 작업 전 고객 Inform 및 설비 Arrange 여부
- End Effector Gap 10 ± 1mm 준수 여부
- Level Spec In 여부 (1 칸 = 0.5mm)
- Z-Axis Teaching 시 End Effector 간섭 주의 준수 여부
- 작업 후 주변 정리 및 Aging Test 진행 여부
- 작업 완료 후 고객 Inform 및 설비 정상화 여부

---

## 2. 최근 3개월 간 TM ROBOT 관련 주요 이슈 정리

| 발생일자 | 설비명 | 이슈 제목 | 주요 내용 및 상태 |
|----------|--------|-----------|-------------------|
| 12/22 | EPAJB10 (17L) | B00003 ADJ TM ROBOT - 점검.1 | TM Robot ETC Alarm 발생 후 설비 Power 점검 |
| 12/22 | EPAJB10 (17L) | B00003 ADJ TM ROBOT - HOME 동작 불가.1 | 고객측 ELT Box 관련 문제 발생 |
| 11/13 | EPAJF12 (17L) | B00003 ADJ TM ROBOT - SPEED 조정.1 | TM Robot Speed 조정 요청, 조정 완료 |
| 11/18 | EPAJF14 (17L) | B00003 ADJ TM ROBOT - SPEED 원복.1 | TM Robot Speed 원복 작업 수행 |
| 11/12 | EPAG18 (15L) | B00003 ADJ TM ROBOT - GRIP ERR.1 | Yaskawa Grip Error 발생, 점검 및 조치 |
| 11/18 | EPAP346 | B00003 ADJ ROBOT - COMM ALARM.1 | TM Robot Communication Alarm 다발 발생, 점검 |
| 12/23 | EPAJB10 (17L) | B00003 REP ROBOT - FAIL.1 | TM Robot 교체 및 Passive PAD 교체 진행 |
| 11/17 | EPAHZA2 (16L) | B00003 ADJ TM RB - TEACHING.1 | TM Robot Passive Pad 교체 후 Teaching 검수 |
| 12/16 | EPAJL59 (17L) | B00003 ADJ ROBOT - VACUUM LEAK.2 | TM Robot Vacuum Alarm 점검 및 조치 |
| 11/21 | EPAJL53 (17L) | B00003 REP PASSIVE PICK - MDY.2 | TM Robot Passive Pick Modify 작업 진행 |
| 12/22 | EPAJM36 (17L) | B00003 ADJ ROBOT 점검 - TENSION ISSUE.1 | TM Robot REP 및 Sub Unit1 Fan REP 진행 |
| 11/12 | EPAG22, EPAG25, EPAG68 (15L) | B00003 ADJ TM ROBOT SPEED CTR - 고객 요청.1 | TM Robot Speed 조정 완료 |
| 11/18 | EPAP317 외 다수 | B00003 REP PASSIVE PAD - 정기교체.1 | TM Robot Upper, Lower Passive Pad 정기 교체 및 식각 문제로 교체 |

- 주요 이슈는 Passive Pad 정기 교체, Grip Error, Vacuum Leak, Communication Alarm, ETC Alarm 등 다양한 알람 및 장애 발생
- Robot Speed 조정 및 원복, Teaching 점검 및 검수, Cable 단선 및 교체, Sensor Alarm 점검 등 유지보수 및 성능 개선 작업 활발
- 작업 중 발생 문제에 대해 신속한 점검과 조치, 작업 완료 후 Aging Test 및 고객 Inform 절차로 품질 확보

---

## 3. 종합 안내

- SUPRA N TM ROBOT 교체 작업은 안전과 품질 확보를 위해 엄격한 절차와 점검을 준수해야 하며, 작업 전후 고객 Inform 및 Aging Test가 필수입니다.
- 작업 중 협착, 감전, 고온, 중량물 취급 등 안전 위험 요소가 많으므로 반드시 보호구 착용과 작업자 간 사전 Inform을 철저히 해야 합니다.
- 최근 3개월 간 TM ROBOT 관련 이슈는 주로 Passive Pad 교체, 알람 발생, Speed 조정, Teaching 점검 등 유지보수 작업이 중심입니다.
- 작업 중 문제 발생 시 즉시 점검하고, 해결되지 않으면 GCB에 등록하여 문의하시기 바랍니다.

---

### 참고 문서 핵심 키워드
- SUPRA N, TM ROBOT, EFEM Robot, Robot Controller, End Effector, Leveling, Teaching, Aging Test, Passive Pad, Alarm, Speed 조정, Safety, 작업 보호구, 작업 인원 및 시간, 작업 절차, 작업 Check Sheet

---

해결되지 않으면 GCB에 등록하세요. 문의: [NAME] ([EMAIL])

---
📊 **총 508건의 이력이 검색되었습니다.** (위 표는 상위 20건)

전체 508건의 이력을 표로 생성해드릴까요?
```

</details>

## Sample 15 — troubleshoot_diag / l (63줄)

- **Timestamp**: `2026-02-23 10:44:50`
- **Question** (23자):

  ```
  FCIP Leak 발생시 조치방법을 알려줘
  ```
- **Sources (Top 20)**: first 5 = ['myservice_psk.jsonl', 'myservice_psk.jsonl', 'myservice_psk.jsonl', 'myservice_psk.jsonl', 'myservice_psk.jsonl']
- **Structure flags**: table=. kw_block=. summary=. cause=. gcb_footer=Y part_num=.
- **First line**: `FCIP Leak 발생 시 조치 방법은 다음과 같습니다. 단계별로 점검 및 조치 절차를 안내합니다.`

<details><summary>Reference answer (PII masked)</summary>

```markdown
FCIP Leak 발생 시 조치 방법은 다음과 같습니다. 단계별로 점검 및 조치 절차를 안내합니다.

---

### 1. 원인 파악 및 초기 상태 확인
- Leak Rate Over Alarm 발생 시, PM별 Leak Rate 수치를 확인합니다. (예: PM3 Leak Rate 8mT, PM2 Leak Rate 8.36mT 등)
- Auto Leak Check 결과 및 Leak Check 로그를 확인하여 Leak 발생 위치 및 정도를 파악합니다.
- He Detector를 사용하여 Leak 위치를 탐지합니다. (예: Door Plate O-Ring, FCIP Gas Line Joint, LAN Port 등)  
- Leak 발생 부위가 FCIP 내부로 추정되는 경우가 많습니다.

### 2. 점검 및 조치 절차 (Step-by-Step)

#### Step 1. He Detector를 이용한 Leak 위치 확인
- FCIP Gas Line, Door Valve, Isolation Valve, LAN Port 등 주요 부위에 He Detector 분사하여 Leak 여부 확인
- Leak 반응이 있는 부위에 대해 Jointing(조임) 작업을 실시 후 재검사 (He Detector 3회 이상 실시 권장)
- Leak 반응이 지속되면 FCIP 내부 Leak 가능성 높음

#### Step 2. Pumping 및 Leak Check 수행
- Pumping 실시 후 Baratron Pressure 및 Leak Check 수치 확인 (예: 5 Torr, Leak Rate 8~11.5 mTorr/min)
- Leak Rate가 Spec 이상이면 FCIP 교체 필요성 검토

#### Step 3. FCIP 교체 준비 및 진행
- FCIP 교체 시 기존 FCIP 탈착 후, 새 FCIP 장착
- FCIP 교체 시 Toplid O-Ring 교체도 함께 진행 권장
- FCIP 교체 후 Leak Check 및 Power 정상 작동 여부 확인
- 작업 시 설비 MAIN 전원 OFF 및 안전장비(예: 방독면) 착용 필수 (특히 NBL 설비)

#### Step 4. Leak 발생 부위 주변 부품 점검 및 교체
- PCW Out단 Elbow, PCW In 단 Elbow 등 Leak 발생 가능 부위 점검 및 교체
- Jointing 강도 조절 후 Leak 재발 여부 확인 (30~40분 Leak 확인 권장)
- Air Tube, Sol Valve 등 주변 부품 노후화 여부 점검 및 교체 고려

#### Step 5. 고객 및 현업 인폼 및 모니터링 요청
- Leak 발생 및 조치 결과를 고객에게 Inform
- FCIP 교체 후에도 Leak 재발 시 현업과 협의하여 추가 조치 요청
- 모니터링을 통해 Leak 재발 여부 지속 확인

---

### 3. 주의사항 및 안전
- 작업 전 설비 MAIN 전원 OFF 필수
- NBL 설비의 경우 Chamber Open 시 반드시 방독면 착용
- Door Valve interlock 등 안전장치 확인 및 해제 후 작업 진행
- Torque Wrench 사용하여 조임 토크 준수
- 작업 완료 후 Check Sheet 작성 및 기록 유지

---

### 4. 요약
- FCIP Leak는 주로 FCIP 내부 또는 LAN Port, Gas Line Joint 부위에서 발생
- He Detector를 이용한 Leak 위치 확인 및 Jointing 후 재검사 필수
- Pumping 및 Leak Check로 Leak Rate 확인
- FCIP 및 관련 O-Ring, Elbow 등 부품 교체가 최종 조치
- 작업 시 안전 수칙 준수 및 고객과 현업에 결과 Inform 및 모니터링 요청

---

참고 문서:  
- 문서 1, 2, 3, 4, 5, 6, 7, 10, 16 (FCIP Leak 점검 및 교체 사례)  
- 문서 8 (Lip Seal O-Ring 교체 Jig 관련)  
- 문서 13 (Leak 관련 Shut off Valve 작업)  

해결되지 않으면 GCB에 등록하세요. 문의: [NAME] ([EMAIL])
```

</details>

## Sample 16 — troubleshoot_diag / l (79줄)

- **Timestamp**: `2026-02-03 12:43:35`
- **Question** (55자):

  ```
  reflow 설비에서 formic acid leak 감지가 되었습니다. 조치 방법에 대해 알려주세요
  ```
- **Sources (Top 20)**: first 5 = ['mySERVICE_pskh.csv', 'mySERVICE_pskh.csv', 'mySERVICE_pskh.csv', 'mySERVICE_pskh.csv', 'mySERVICE_pskh.csv']
- **Structure flags**: table=Y kw_block=Y summary=. cause=. gcb_footer=Y part_num=.
- **First line**: `Reflow 설비에서 Formic acid leak 감지 시 조치 방법은 다음과 같이 단계별로 진행하시기 바랍니다.`

<details><summary>Reference answer (PII masked)</summary>

```markdown
Reflow 설비에서 Formic acid leak 감지 시 조치 방법은 다음과 같이 단계별로 진행하시기 바랍니다.

---

## 1. 원인 파악 및 초기 점검

- **Formic gas leak alarm 발생 시 우선 확인할 사항**
  - Bubbler cabinet 내부 gas detector alarm 발생 여부 확인
  - Leak detector port 구성 및 튜브 연결 상태 점검  
    → Input tube가 Exhaust port에 오장착되어 있으면 alarm 발생 가능성 있음  
    → Input port는 sampling port로서 cabinet 내부에 노출되어 있어야 함[문서 1, 9]

- **Formic acid 공급 라인 및 배관부 점검**
  - 배관부 leak 여부 확인 및 재조임 (테프론 Tape 사용 권장)  
  - Pump 및 Valve 정상 동작 여부 확인  
  - Bubbler 내부 Valve 및 Vent 부위 점검[문서 2, 8]

- **Formic gas detector 상태 점검**
  - Detector cartridge 오염 또는 고장 여부 확인 및 교체 필요 시 교체  
  - Detector의 Warning/Alarm range 조정 가능 (예: Warning 60ppm, Alarm 80ppm)  
  - Detector 배선 및 전원 상태 점검[문서 4, 6, 17]

---

## 2. Leak 점검 및 조치 방법 (Step-by-Step)

### Step 1: Leak 감지 및 위치 확인
- 휴대용 Formic gas detector를 사용하여 leak 발생 위치 확인  
- Bubbler cabinet, 배관부, Chamber, Loadlock door, Elbow heater clamp 등 주요 부위 점검  
- Facility panel 및 Filter 오염 여부 확인 (오염 시 Filter 교체 권장)[문서 7, 11, 14]

### Step 2: Leak 부위 조치
- Leak 발생 부위 재조임 및 테프론 Tape 감아 체결  
- Valve, Pump, 배관 연결부 재점검 및 재조임  
- 필요 시 Valve, Cartridge, PFA tube 등 부품 교체  
- Bubbler cabinet 내부 가압 테스트 및 압력 유지 확인 (예: 20psi 가압 후 압력 유지)[문서 2, 8, 13]

### Step 3: Leak 감지기 재설정 및 모니터링
- Formic gas detector manual reset 또는 Dip switch, Parameter, Setup.ini 항목 조정  
- Leak 감지기 교체 후 재점검 및 모니터링  
- Alarm range 조정으로 불필요한 알람 최소화 가능[문서 4, 6, 17]

### Step 4: 공조 및 배기 시스템 점검
- Tool cabinet exhaust pressure 조정 (예: 0.01inWC → 0.2inWC)  
- Damper 조절 및 APC calibration 진행  
- Chamber 및 Loadlock door seal 상태 확인 및 필요 시 교체[문서 2, 16, 15]

---

## 3. 검증 및 안전 주의사항

- Leak 조치 후 Formic recipe cycle test 진행하여 leak alarm 재발생 여부 확인  
- 휴대용 detector로 Leak 감지 재확인  
- Formic acid 취급 시 온도 50℃ 이하 유지 권장 (안전관리자 환경안전 GUIDE 준수)[문서 10]

- Bubbler cabinet door 및 유리 개방 시 leak 수치 변화 확인하여 leak 여부 판단  
- Leak 감지 시 즉시 Formic acid 공급 차단 (Inlet 단 Manual valve Close)[문서 18]

---

## 4. 요약 및 권장 조치

| 조치 항목                          | 상세 내용 및 참고 문서                      |
|----------------------------------|-----------------------------------------|
| Leak detector port 연결 상태 확인 | Input tube가 Exhaust port에 꽂혀있지 않은지 확인 [문서 1, 9] |
| 배관부 및 Valve 재조임            | 테프론 Tape 사용, Pump 및 Valve 정상 동작 확인 [문서 2, 8] |
| Formic gas detector 점검 및 교체  | Cartridge 교체, Alarm range 조정 [문서 4, 6, 17] |
| 휴대용 detector로 Leak 위치 확인  | Bubbler, Chamber, Loadlock door, Elbow heater clamp 등 점검 [문서 7, 14] |
| 공조 및 배기 시스템 조정          | Exhaust pressure 조정, Damper 및 APC calibration [문서 2, 16] |
| Leak 조치 후 cycle test 및 모니터링 | Leak 재발생 여부 확인, 안전 온도 유지 [문서 10, 18] |

---

만약 위 조치 후에도 문제가 해결되지 않으면, GCB에 등록하고 질문하세요.  
AI 답변 품질 문의는 담당자 GunamDOH ([EMAIL])에게 연락 바랍니다.

---

**참고 문서 핵심 키워드:** Bubbler cabinet, Input tube 오장착, 배관부 재조임, Formic gas detector cartridge 교체, 휴대용 detector leak 확인, Exhaust pressure 조정, APC calibration, Formic recipe cycle test, 안전 온도 50℃ 이하 유지[1,2,4,6,7,8,9,10,14,16,17,18]
```

</details>

## Sample 17 — troubleshoot_diag / xl (142줄)

- **Timestamp**: `2026-02-02 12:56:48`
- **Question** (33자):

  ```
  SUPRA XP Toxic Gas Turn On 방법 알려줘
  ```
- **Sources (Top 60)**: first 5 = ['setupmanual.json', 'setupmanual.json', 'setupmanual.json', 'setupmanual.json', 'setupmanual.json']
- **Structure flags**: table=Y kw_block=Y summary=. cause=. gcb_footer=Y part_num=.
- **First line**: `# SUPRA XP 설비 Toxic Gas Turn On 절차 (단계별 상세 안내)`

<details><summary>Reference answer (PII masked)</summary>

```markdown
# SUPRA XP 설비 Toxic Gas Turn On 절차 (단계별 상세 안내)

---

## 1. 사전 조건 및 준비사항

- **안전 보호구 착용**  
  - 안전모, 안전화 착용 필수  
  - Toxic Gas Turn On 작업 시 반드시 방독면 착용 (문서 1, 10.2 Toxic Gas Turn On)  

- **고객사 공급단 Toxic Gas Line 가압 확인**  
  - 설비단 Gas Box 내 Manual V/V를 Close 시킨다.  
  - 가압 상태로 24시간 유지 후, Line 내 가압이 유지되어야 Line Check 진행 가능 (문서 3, 10.1 Toxic Gas Line Check)  

- **Gas Line Leak 및 Chamber Leak Check**  
  - Manual V/V Close 후 Gas Line Leak Check 진행  
  - Chamber Leak은 0.5 mtorr 이하로 확인  
  - Leak Check 완료 후 Manual V/V를 Open (문서 4, 10.2 Toxic Gas Turn On)  

- **Process Gas Spec에 맞게 Regulator 조절**  
  - H2(4%)N2 : 35±2 psi  
  - H2 : 35±2 psi  
  - 고객사마다 상이할 수 있음 (문서 5, 10.1 / 문서 7, 10.2)  

- **24시간 Pumping 진행**  
  - Regulator 조절 후 NC V/V를 Open 시킨 상태에서 24시간 Pumping 진행 (문서 5, 10.1)  

---

## 2. MCB 및 밸브 조작 순서

### 2.1 MCB 조작 (Pump 및 전원 관련)

- **PM Pump MCB On**  
  - Pump 업체와 함께 각 Pump Power Cable 장착 확인  
  - Pump 업체 신호에 따라 각 PM별 Pump MCB On  
  - MCB0-9 : PM1, MCB0-10 : PM2, MCB0-11 : PM3 (문서 12, 6.4 PM Pump Turn on)  

- **TM Pump MCB On**  
  - Pump 업체와 함께 Power Cable 장착 확인  
  - Pump 업체 신호에 따라 TM Pump MCB On  
  - MCB0-7 : TM (문서 20, 6.6 TM Pump Turn on)  

- **LL Pump MCB On**  
  - Pump 업체와 함께 Power Cable 장착 확인  
  - Pump 업체 신호에 따라 LL Pump MCB On  
  - MCB0-8 : LL (문서 16, 6.5 LL Pump Turn on)  

- **EFEM Electric Box MCB On**  
  - EFEM 하단 Cover를 열고 EFEM Electric Box 내 MCB On (문서 17, 5.3 Controller Check)  

- **Main Rack MCB On**  
  - Main Power rack Door 닫고 MCB On (Source power 제외)  
  - 모든 MCB 올림 (문서 15, 5.1 Main Rack)  

### 2.2 밸브 조작 순서

- **고객사 Main 공급단 Manual V/V Close**  
  - Toxic Gas Line 가압 Check 위해 설비단 Gas Box 내 Manual V/V Close (문서 3, 10.1 Toxic Gas Line Check)  

- **Gas Line Leak Check 후 Manual V/V Open**  
  - Leak Check 완료 후 Manual V/V Open (문서 4, 10.2 Toxic Gas Turn On)  

- **유틸업체 신호에 따라 Manual V/V Open**  
  - 유틸업체가 Toxic Gas Line Turn On 신호 주면 Manual V/V Open (문서 2, 10.1 Toxic Gas Line Check)  

- **GUI PM Manual Screen에서 해당 Gas NC V/V Open**  
  - Manual V/V Open 후 GUI에서 NC V/V Open (문서 2, 10.1 Toxic Gas Line Check)  

- **NC V/V Open 후 Gas Flow Full Scale로 조절 및 확인**  
  - NC V/V Open 후 유량 Full Scale로 놓고 Flow 정상 확인 (문서 1, 10.2 / 문서 2, 10.1)  

---

## 3. 인터락 및 안전 확인 절차

- **Gas Line Leak 및 Chamber Leak Check**  
  - Leak Check 화면 우측 하단 Gas Line Leak Check Usage 클릭  
  - Gas Box Manual V/V Close 후 Gas Line Leak Check 실시  
  - Leak Check 기준 : 0.5 mtorr 이하 (문서 6, 10.1 Toxic Gas Line Check)  

- **Pump Alarm 및 Warning Signal 확인**  
  - Pump Alarm 발생 시 설비 Alarm ON 확인  
  - Signal 미발생 시 TM Multiport Rx, Tx 점등 및 Connection 상태 확인  
  - Set Up.ini File Maker 확인 (문서 12, 6.4 / 문서 13, 6.5 / 문서 14, 6.5 / 문서 20, 6.6)  

- **Pump 방치압 확인**  
  - Local Mode Pumping 여부 및 방치압 0.5 mtorr 이하 확인 (고객사마다 상이) (문서 15, 6.4 PM Pump Turn on)  

- **Gas Line Leak 정상 여부 확인**  
  - 고객사 Manual V/V Open 시 Leak 정상 여부 반드시 확인 (문서 9, 9.1 O2, N2 Gas Turn on)  

- **EHS 주의사항**  
  - Valve Open 간 Grating 하부에 Valve 위치 시 Grating Open에 의한 추락 사고 주의 (문서 9, 9.1)  

- **작업 시 온도 및 기계적 주의사항**  
  - 공정 중 내부 온도 250℃, 작업 시 Display 온도 50℃ 이하 확인  
  - Utility Box Open/Close 시 신체 일부 협착 주의 (문서 20 Safety)  

---

## 4. 종합 단계별 절차 요약

| 단계 | 작업 내용 | 상세 내용 |
|-------|-----------|-----------|
| 1     | 안전 보호구 착용 | 안전모, 안전화, 방독면 착용 필수 |
| 2     | 고객사 공급단 Manual V/V Close | Toxic Gas Line 가압 Check 위해 |
| 3     | 24시간 가압 유지 | Line 내 가압 유지 확인 |
| 4     | Gas Line Leak 및 Chamber Leak Check | Manual V/V Close 후 Leak Check 실시, 0.5 mtorr 이하 확인 |
| 5     | Leak Check 완료 후 Manual V/V Open | Leak Check 완료 후 Manual V/V Open |
| 6     | Regulator 조절 | Process Gas Spec에 맞게 조절 (35±2 psi 등) |
| 7     | 24시간 Pumping 진행 | Regulator 조절 후 NC V/V Open 상태에서 Pumping |
| 8     | 유틸업체 신호 수신 후 Manual V/V Open | 유틸업체 신호에 따라 Manual V/V Open |
| 9     | GUI PM Manual Screen에서 NC V/V Open | GUI에서 NC V/V Open |
| 10    | NC V/V Open 후 Gas Flow Full Scale 조절 및 확인 | Flow 정상 확인 |
| 11    | MCB On (Pump 및 EFEM 등) | PM Pump, TM Pump, LL Pump MCB On 및 EFEM Electric Box MCB On |
| 12    | Pump Alarm 및 Warning Signal 확인 | Alarm 발생 시 설비 Alarm ON 확인, Signal 미발생 시 Multiport 및 설정 확인 |
| 13    | Pump 방치압 확인 | Local Mode Pumping 여부 및 방치압 0.5 mtorr 이하 확인 |
| 14    | Gas Line Leak 정상 여부 확인 | 고객사 Manual V/V Open 시 Leak 정상 여부 필수 확인 |
| 15    | EHS 및 안전 주의사항 준수 | Grating 추락 사고 주의, 온도 및 기계적 주의사항 준수 |

---

# 안전 주의사항

- 작업 전 반드시 안전모, 안전화, 방독면 착용  
- Valve Open 시 Grating 하부 위치 및 추락 사고 주의  
- 작업 중 설비 동작 상태 및 인터락 상태 확인  
- 전기 및 가스 관련 감전 및 누출 위험 주의  
- Utility Box Open/Close 시 신체 협착 주의  

---

위 절차는 SUPRA XP 설비의 Toxic Gas Turn On 작업 시 안전하고 정확한 수행을 위한 필수 지침입니다. 작업 중 이상 발생 시 즉시 작업 중지 후 담당자에게 보고하세요.

---

해결되지 않으면 GCB에 등록하세요. 문의: [NAME] ([EMAIL])

---

**참고 문서 핵심 키워드:** SUPRA XP, Toxic Gas Turn On, Manual V/V, NC V/V, Gas Line Leak Check, Chamber Leak, Regulator 조절, Pump MCB On, Pump Alarm, Safety PPE, EHS, Gas Flow Full Scale, MCB 번호 (MCB0-7, MCB0-8, MCB0-9~11) [문서 1~20]
```

</details>

## Sample 18 — list_lookup / m (46줄)

- **Timestamp**: `2026-02-19 11:34:45`
- **Question** (34자):

  ```
  체크리스트로 만들라고
  참고 문서: 사용 안 함 또는 찾지 못함
  ```
- **Sources (Top 0)**: first 5 = []
- **Structure flags**: table=Y kw_block=. summary=. cause=. gcb_footer=Y part_num=.
- **First line**: `# EPA3761 SUPRA N 장비 LP3 TP5 Illegal Sequence Alarm 정상화 사례 기반 점검 및 조치 체크리스트`

<details><summary>Reference answer (PII masked)</summary>

```markdown
# EPA3761 SUPRA N 장비 LP3 TP5 Illegal Sequence Alarm 정상화 사례 기반 점검 및 조치 체크리스트

| 단계 | 점검 항목 | 확인 내용 | 체크 (✓/✗) | 비고 |
|-------|--------------------------|------------------------------------------------------------|-------------|-------|
| 1 | 초기 알람 상태 확인 | EFEM GUI에서 Error_Valid_Off 및 Err_AMHS_Time_Out_TP4 알람 상태 확인 | | |
| 1 | OHT 신호 수신 확인 | OHT COMPT On 신호 수신 여부 로그 확인 | | |
| 2 | 신호 상태 점검 | OHT에서 설비로 전달되는 Valid, CS_0, Transfer Request 신호 상태 점검 | | |
| 2 | Present Sensor 상태 점검 | LP3 TP5 구간 Present Sensor 및 관련 케이블 접촉 상태 확인 | | |
| 2 | 케이블 접촉 불량 테스트 | 케이블 흔들기 테스트로 접촉 불량 여부 확인 | | |
| 3 | 센서 정상 동작 확인 | Present Sensor 정상 동작 여부 확인 및 필요 시 교체 | | |
| 3 | 보드 상태 점검 | Joint Board 및 관련 I/O 보드 상태 점검 및 교체 고려 | | |
| 3 | 메인 보드 연결 확인 | Main Board 및 관련 보드 케이블 연결 상태 확인 | | |
| 4 | OHT 통신 점검 | OHT 측 신호 송수신 정상 여부 점검 | | |
| 4 | 자동화 시스템 통신 점검 | 자동화 시스템과 싸이맥스 간 통신 상태 확인 | | |
| 4 | Transfer Request 신호 확인 | Transfer Request 신호 정상 발생 여부 확인 | | |
| 5 | 로그 분석 | 알람 발생 시점의 로그 상세 분석 | | |
| 5 | Load/Unload Sequence 확인 | FOUP Load/Unload Sequence 정상 흐름 확인 | | |
| 5 | 재현성 테스트 | 재현성 테스트를 통한 문제 재현 및 원인 파악 | | |
| 6 | 부품 교체 | 문제 부품(센서, 케이블, 보드) 교체 | | |
| 6 | 신호선 재배선 | 신호선 재배선 및 커넥터 재장착 | | |
| 6 | 시스템 초기화 | 시스템 재부팅 및 초기화 후 정상 동작 확인 | | |
| 7 | 고객 보고 | 점검 및 조치 결과 고객사에 상세 보고 | | |
| 7 | SOP/Troubleshooting Guide 활용 | SOP 및 Troubleshooting Guide 활용 권장 | | |
| 8 | 예방 점검 계획 | 정기적인 케이블 및 센서 점검 계획 수립 | | |
| 8 | 통신 상태 모니터링 | OHT 신호 라인 및 자동화 시스템 통신 상태 주기적 모니터링 | | |

---

# 참고: 정상화 사례 주요 내용 (Order No. 기준)

| Order No. | 접수일 | 현상/원인 요약 | 주요 조치 요약 |
|-----------|---------|----------------|----------------|
| 40120368 | 2024-01-23 | LP3 TP1 Illegal Sequence Alarm, OHT Valid, CS_0 신호 간헐적 수신 실패 | 로그 분석, 케이블 및 신호선 점검 및 교체 테스트 |
| 40107983 | 2023-07-25 | LP3 TP3 Illegal Sequence Alarm, OHT Transfer Request 신호 Off | 알람 로그 및 PIO 로그 분석, OHT 점검 요청 |
| 40162381 | 2025-07-21 | LP3 TP3 Sensor Logic Alarm, Present Sensor 케이블 흔들림에 따른 신호 깜빡임 | 센서 및 케이블 점검, Present Sensor 교체, SOP/Troubleshooting Guide 권고 |
| 40120737 | 2024-01-29 | LP3 Sensor Access Violation Alarm, Present Sensor 신호 Off 간헐 발생 | Joint Board 교체, 케이블 점검, 재현성 테스트, 고객 인폼 |
| 40120488 | 2024-01-25 | LP3 Sensor Access Violation Alarm, FOUP 안착 후 Present Sensor 신호 떨어짐 | Board 및 케이블 수급 후 교체 예정, 현업 인폼 완료 |
| 40120049 | 2024-01-18 | LP3 TP1 Illegal Sequence Alarm, Transfer Request On 신호 미발생 | 로그 확인, OHT 자동화 점검 요청 |
| 40141832 | 2024-11-01 | LP1 VACUUM ON Alarm, Joint 3보드 케이블 반대로 장착, Main Board 고장 의심 | 케이블 정상 체결 및 교체, Joint 3보드 및 I/O 케이블 Swap 테스트, Main Board 교체 예정 |

---

이 체크리스트는 EPA3761 SUPRA N 장비 LP3 TP5 Illegal Sequence Alarm 및 유사 LP3 TPx Illegal Sequence/Timeout/Sensor Logic 알람 정상화 사례를 기반으로 하였으며, 단계별 점검과 조치 항목을 체계적으로 수행할 수 있도록 설계되었습니다.

해결되지 않으면 GCB에 등록하세요. 문의: [NAME] ([EMAIL])  
[1]
```

</details>

## Sample 19 — short_followup / s (18줄)

- **Timestamp**: `2026-03-05 15:49:15`
- **Question** (10자):

  ```
  FFU 转速spec
  ```
- **Sources (Top 34)**: first 5 = ['PSKH_ASBOM.csv', 'PSKH_ASBOM.csv', 'PSKH_ASBOM.csv', 'PSK_ASBOM_2021-2022.csv', 'PSKH_ASBOM.csv']
- **Structure flags**: table=Y kw_block=. summary=. cause=. gcb_footer=Y part_num=.
- **First line**: `根据参考资料，FFU（Fan Filter Unit）转速的具体规格在文档中未直接明确给出统一的数值，但有一处提及实际操作中的转速调整案例：`

<details><summary>Reference answer (PII masked)</summary>

```markdown
根据参考资料，FFU（Fan Filter Unit）转速的具体规格在文档中未直接明确给出统一的数值，但有一处提及实际操作中的转速调整案例：

- 在Doc 33中，EFEM FFU风扇转速检查为**800 rpm**，TM FFU风扇转速由**950 rpm调整到1100 rpm**，且检查确认通过[33]。

此外，FFU相关的主要制造商为**신성 ENG (Shinsung ENG)**和**한국캠브리지필터 (Korea Cambridge Filter)**，型号多为ECOLITE3000、SUPRA系列等，部件号主要是B0500037（FFU FILTER_TM）和B0500003（FFU FILTER_FFU）等。

总结：

| 项目          | 规格/数值           | 备注                         |
|---------------|---------------------|------------------------------|
| FFU转速（EFEM） | 约800 rpm           | 现场实际检查值               |
| FFU转速（TM）  | 950 rpm → 1100 rpm  | 调整后确认通过               |
| 主要Maker     | 신성 ENG, 한국캠브리지필터 | 代表主要FFU滤网制造商         |
| 相关型号      | ECOLITE3000, SUPRA系列 | 多型号对应不同设备           |

**结论**：FFU转速一般在800 rpm至1100 rpm范围内，具体数值依设备型号和现场调整而定。若需详细转速spec，请确认具体设备型号和FFU类型。

해결되지 않으면 GCB에 등록하세요. 문의: [NAME] ([EMAIL])
```

</details>

## Sample 20 — general / m (47줄)

- **Timestamp**: `2026-01-30 08:40:18`
- **Question** (27자):

  ```
  SUPRA Q Robot 통신은 어떤 통신을 해?
  ```
- **Sources (Top 19)**: first 5 = ['scraped_sop.json', 'scraped_sop.json', 'scraped_sop.json', 'scraped_sop.json', 'scraped_sop.json']
- **Structure flags**: table=. kw_block=Y summary=. cause=. gcb_footer=Y part_num=.
- **First line**: `SUPRA Q 설비에서 Robot과 상위/타 모듈 간 통신 방식에 대해 참고자료를 종합하여 답변드립니다.`

<details><summary>Reference answer (PII masked)</summary>

```markdown
SUPRA Q 설비에서 Robot과 상위/타 모듈 간 통신 방식에 대해 참고자료를 종합하여 답변드립니다.

---

## 1. 상위(Host/MES) ↔ 설비/Robot 간 통신 프로토콜 및 인터페이스

- SUPRA Q 설비는 Host Communication을 Remote에서 Local로 변경하는 절차가 여러 SOP에 공통적으로 명시되어 있습니다.  
- Host와 설비 간 통신은 Ethernet 기반 TCP/IP 통신이 기본이며, NPORT 장비를 통해 Ethernet RJ 케이블로 연결되어 있습니다.  
- NPORT Administrator 프로그램을 사용하여 IP 주소 설정 및 통신 상태를 관리하는 점으로 보아, **Ethernet TCP/IP 기반 통신**이 사용됩니다.  
- SECS-II / HSMS 프로토콜 사용 여부는 명시되어 있지 않으나, 반도체 장비 특성상 SECS-II/HSMS가 상위 통신 표준으로 널리 쓰이므로 가능성이 있으나, 참고자료에 구체적 언급은 없습니다.  
- Host Communication 변경 시 반드시 PIO Status Manual → Auto 변경 확인 절차가 포함되어 있어, AMHS(Automated Material Handling System)와의 연동도 Ethernet TCP/IP 기반으로 추정됩니다[16,18].

---

## 2. Robot 컨트롤러 ↔ 타 모듈(PLC, I/O, 기타 장치) 간 통신

- TM Robot, EFEM, PM 모듈 등 각 모듈별 NPORT가 설치되어 있으며, 각 포트별로 Ethernet RJ 케이블이 연결되어 있습니다.  
- TM NPORT에서는 PORT2가 EFEM FFU 통신, PORT3가 TM FFU 통신으로 지정되어 있고, EFEM NPORT에서는 PORT1~4가 Load Port 통신, PORT5~8이 RFID 통신으로 구분되어 있습니다.  
- NPORT를 통한 통신은 Ethernet TCP/IP 기반이며, 각 모듈 간 통신은 NPORT를 통해 Ethernet 네트워크로 연결되어 있습니다.  
- Fieldbus(예: EtherNet/IP, PROFINET, EtherCAT) 사용 여부는 명확히 언급되어 있지 않습니다.  
- TM Robot Teaching, EFEM Robot Teaching 등에서 IP 주소 설정 및 Buzz5000 프로그램을 통한 IP 관리가 이루어지는 점으로 보아, Ethernet TCP/IP 네트워크 기반 통신임을 알 수 있습니다[2,3,16,18,19].

---

## 3. Robot ↔ 비전 시스템 및 기타 모듈 간 통신

- 참고자료 내에 비전 시스템과 Robot 간 통신 프로토콜에 대한 구체적 언급은 없습니다.  
- 다만, EFEM Robot, TM Robot, PM 모듈 간 통신은 Ethernet TCP/IP 기반 NPORT를 통해 이루어지는 것으로 파악됩니다[2,3,16,18,19].

---

## 4. 요약 및 권고

- SUPRA Q 설비 내 Robot과 상위/타 모듈 간 통신은 **Ethernet TCP/IP 기반 NPORT 장비를 통한 통신**이 기본입니다.  
- 상위 통신 프로토콜은 명확히 기술되어 있지 않으나, 반도체 업계 표준인 SECS-II/HSMS 또는 OPC UA 가능성이 있으나, 참고자료에는 구체적 언급이 없습니다.  
- 모듈 간 통신은 NPORT를 통한 Ethernet RJ 케이블 연결로 이루어지며, 각 포트별 통신 대상이 구분되어 있습니다.  
- Fieldbus 사용 여부는 자료에 명시되어 있지 않습니다.  
- 정확한 프로토콜 및 인터페이스 확인을 위해서는 설비의 Interface Specification 문서, 전기/통신 배선도, 또는 네트워크 구성도를 확인하시기 바랍니다.

---

해결되지 않으면 GCB에 등록하세요. 문의: [NAME] ([EMAIL])

---

**참고 문서 핵심 키워드:**  
SUPRA Q, Robot 통신, NPORT, Ethernet TCP/IP, Host Communication, AMHS, EFEM, TM Robot, PM Module, IP Address, NPORT Administrator, RJ45 Cable, Interlock Reset, Home 동작[2,3,16,18,19]
```

</details>
