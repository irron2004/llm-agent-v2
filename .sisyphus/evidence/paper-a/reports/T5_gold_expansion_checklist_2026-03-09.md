# Gold Doc Expansion Candidates (2026-03-09)

## Summary
- Total queries: 472
- Missing gold: 133 (28.2%)
- Candidates in this pack: 133
- ES index: rag_chunks_dev_v2
- Top-k per query: 30

## PE Review Instructions

For each query below, review the ES candidate documents and mark
which doc_ids are relevant (gold). Update `gold_doc_ids` in
`query_gold_master_v0_5.jsonl` accordingly.

## Candidates by Priority

### explicit_device (highest priority)

#### q_id: `A-q001`
- **Question**: ECOLITE3000 설비에서 PM Chamber 내부 View Port 쪽에 Local Plasma 및 Arcing이 발생하는 원인은 무엇인가?
- **Devices**: [ECOLITE_3000]
- **Scope**: explicit_device | **Intent**: troubleshooting
- **ES candidates** (top-26):
  - [ ] `set_up_manual_integer_plus` (score=11.3641, device=INTEGER plus, type=SOP)
    > ```markdown # 17-21. Load port 인증 | Picture | Description | Data | OK | NG | N/A | | :--- | :--- | :--- | :--- | :--- | 
  - [ ] `global_sop_supra_series_all_sw_operation` (score=9.9192, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_ SW OPERATION Global SOP No Revision No: 2 Page: 20/49 ## 4. Leak Check - Work
  - [ ] `global_sop_supra_n_all_pm_fcip_r3` (score=9.8478, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_CLN_PM_FCIP R3 QUARTZ TUBE Global SOP No : Revision No: 3 Page: 83/84 ## 7. 작업 Check Sh
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=9.6657, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 3/124 ## 3. 사고 사례 ### 1) 화상의 정의 불
  - [ ] `global_sop_supra_xp_all_pm_dual_epd` (score=9.6648, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_DUAL EPD Global SOP No: Revision No: 1 Page: 3/40 ## 3. 사고 사례 ### 1) 화상의 정의 불이
  - [ ] `global_sop_precia_all_pm_gap_sensor` (score=9.6642, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_VIEW PORT QUARTZ Global SOP No : Revision No: 1 Page : 59 / 79 ## Scope 이 Global 
  - [ ] `global_sop_geneva_xp_adj_pm_chuck_temp_calibration` (score=9.642, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Chuck temp calibration Global SOP No: Revision No: 0 Page: 3/21 ## 3. 재해 방지 대책
  - [ ] `global_sop_supra_n_series_all_pm_gas_feed_through` (score=9.5531, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS FEED THROUGH Global SOP No: Revision No: 4 Page: 4 / 18 ## 3. 사고 사례 #
  - [ ] `global_sop_supra_n_series_all_pm_heater_chuck` (score=9.5432, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_HEATER CHUCK Global SOP No : Revision No: 2 Page: 3/40 ## 3. 사고 사례 ### 1)
  - [ ] `global_sop_supra_xp_all_pm_baratron_gauge` (score=9.5356, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_BARATRON GAUGE Global SOP No : 0 Revision No : 0 Page : 3/33 ## 3. 사고 사례 ### 1
  - [ ] `global_sop_supra_xp_all_pm_pendulum_valve` (score=9.5335, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PENDULUM VALVE Global SOP No : Revision No: 6 Page: 3/26 ## 3. 사고 사례 ### 1) 화상
  - [ ] `global_sop_precia_all_pm_pendulum_valve` (score=9.5291, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_PENDULUM VALVE Global SOP No : Revision No: 0 Page: 3/32 ## 3. 사고 사례 ### 1) 화상 재해
  - [ ] `global_sop_geneva_xp_rep_pm_heater_chuck_without_jig` (score=9.5282, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA xp_REP_PM_Heater chuck w/o jig | SOP No: 0 | | | |---|---|---| | Revision No: 1 | | | | Page: 3/52 
  - [ ] `global_sop_supra_xp_all_pm_process_kit` (score=9.5273, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PROCESS KIT Global SOP No : Revision No : 3 Page : 4/28 ## 3-2 화상 ### 1) 화상 재해
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=9.521, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 77 / 105 ## 사고 사례 ##
  - [ ] `global_sop_supra_n_series_all_pm_top_lid` (score=9.5208, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N Series_ALL_PM_TOP LID Global SOP No: Revision No: 0 Page: 4 / 48 ## 3-2 화상 ### 1) 화상 재해
  - [ ] `global_sop_geneva_xp_adj_pm_pin_alignment` (score=9.5189, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Pin Alignment | Global SOP No: | S-KG-A003-R0 | | --- | --- | | Revision No: |
  - [ ] `global_sop_supra_xp_all_pm_pirani_gauge` (score=9.5169, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIRANI GAUGE Global SOP No: Revision No: 1 Page: 3/40 ## 3. 사고 사례 ### 1) 화상 재해
  - [ ] `global_sop_supra_xp_all_pm_pressure_gauge` (score=9.5162, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRESSURE GAUGE Global SOP No: Revision No: 1 Page: 3/34 ## 3. 사고 사례 ### 1) 화상 
  - [ ] `global_sop_geneva_xp_rep_pm_support_pin` (score=9.5084, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Support pin | Global SOP No: | 0 | | --- | --- | | Revision No: | 1 | | Page: 
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=9.5079, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 4/47 ## 3. 사고 사례 ### 1) 화상 재해의 
  - [ ] `global_sop_supra_n_series_all_pm_epd` (score=9.4752, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_EPD Global SOP No : Revision No: 3 Page: 3/49 ## 3. 사고 사례 ### 1) 화상 재해의 정
  - [ ] `global_sop_geneva_xp_rep_pm_adapter_ring` (score=9.4616, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Adapter ring Global SOP No: 0 Revision No: 0 Page: 4 / 30 ## 3. 사고 사례 ### 1) 화
  - [ ] `global_sop_geneva_xp_rep_pm_harmonic_drive` (score=9.454, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Harmonic drive | Global SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Pag
  - [ ] `global_sop_supra_n_series_all_pm_process_kit` (score=9.4467, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_PROCESS KIT Global SOP No : Revision No: 0 Page: 3/55 ## 3. 사고 사례 ### 1) 
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=9.4442, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier | Global SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Pag

#### q_id: `A-q008`
- **Question**: SUPRA III 설비에서 APC Pressure Hunting 발생 시 점검해야 할 포인트는 무엇인가?
- **Devices**: [SUPRA]
- **Scope**: explicit_device | **Intent**: troubleshooting
- **ES candidates** (top-22):
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=9.2428, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_TM_CONTROLLER ## BATTERY_Replacement Global SOP No : Revision No: 6 Page : 94 / 
  - [ ] `set_up_manual_supra_nm` (score=8.9975, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.1. Common List ### 12.1.9 Gas Pressure | Picture | De
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=8.8996, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page : 39 / 105 ## 6. APC 
  - [ ] `set_up_manual_supra_n` (score=8.8438, device=SUPRA N, type=SOP)
    > ```markdown Confidential I | 2) Gas Pressure Check | a. BKM Recipe 로 Gas Flow 진행 시 GUI 화면에 출력되는 Gas 압력을 확인한다. | | | :---
  - [ ] `global_sop_supra_series_all_sw_operation` (score=8.4737, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_ SW OPERATION Global SOP No Revision No: 2 Page: 20/49 ## 4. Leak Check - Work
  - [ ] `global_sop_supra_n_series_all_sub_unit_pressure_vacuum_switch` (score=8.4627, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_PRESSURE & VACUUM SWITCH Global SOP No : Revision No: 1 Page: 3/28 
  - [ ] `40051833` (score=8.412, device=SUPRA Vplus, type=myservice)
    > -. Log 확인시 Placement Error로 보여짐
-> 싸이맥스 로그 확인 시 S4(Foup 없는상태) -> S6(Present 감지) -> S7(Placement 감지) 순으로 변해야하나 S6에서 계속 바뀌
  - [ ] `global_sop_geneva_xp_rep_efem_ffu_controller` (score=8.3727, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_FFU Controller | SOP No: 0 | Revision No: 0 | Page: 3/21 | | --- | --- | ---
  - [ ] `global_sop_geneva_xp_rep_efem_ffu_filter` (score=8.3523, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_FFU Filter | SOP No: S-KG-R041-R0 | | --- | | Revision No: 0 | | Page: 3 / 1
  - [ ] `global_sop_geneva_xp_rep_efem_ffu` (score=8.3396, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_FFU | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3 / 14 | 
  - [ ] `set_up_manual_supra_np` (score=8.3102, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I 3) Gas Pressure Adjust <!-- Image (63, 87, 360, 300) --> a. Gas Pressure 압력이 고객 사양과 상이할 경우, G
  - [ ] `global_sop_geneva_xp_rep_pm_pressure_switch` (score=8.2515, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Pressure switch SOP No: 0 Revision No: 0 Page: 11/16 ## 10. Work Procedure | F
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=8.201, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 2/15 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_n_series_all_tm_pressure_relief_valve` (score=8.1065, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_PRESSURE RELIEF VALVE Global SOP No : Revision No: 2 Page: 2/19 ## 1. Saf
  - [ ] `global_sop_geneva_xp_rep_pm_device_net` (score=8.1019, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Device net | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/15
  - [ ] `global_sop_geneva_xp_adj_all_sw_install` (score=8.0972, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_AII_SW INSTALL | SOP No: 0 | | | | --- | --- | --- | | Revision No: 0 | | | | Pag
  - [ ] `global_sop_geneva_xp_rep_pm_o2_analyzer` (score=8.094, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_O2 analyzer(Delta F) | SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Page
  - [ ] `global_sop_supra_xp_all_tm_pressure_gauge` (score=8.0812, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_PRESSURE GAUGE Global SOP No: Revision No: 1 Page: 3/31 ## 3. 사고 사례 ### 3-1 감전
  - [ ] `global_sop_supra_xp_all_ll_pressure_gauge` (score=8.0793, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_LL_PRESSURE GAUGE Global SOP No: Revision No: 0 Page: 3/28 ## 3. 사고 사례 ### 3-1 감전
  - [ ] `global_sop_supra_n_series_all_rack` (score=8.0587, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_RACK Global SOP No : Revision No: 1 Page: 3/84 ## 3. 사고사례 ### 3-1. 감전의 정의 '감
  - [ ] `global_sop_supra_n_series_all_pcw_turn_on` (score=8.0481, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PCW TURN ON Global SOP No: Revision No: 1 Page: 3/31 ## 3. 사고 사례 ### 1) 감전의 
  - [ ] `global_sop_supra_n_series_all_sub_unit_flow_switch` (score=8.0453, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_FLOW SWITCH Global SOP No : Revision No: 1 Page: 3/17 ## 3. 사고 사례 #

### implicit

#### q_id: `A-imp001`
- **Question**: RF Power 출력이 설정값과 다를 때 점검 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA]
- **Scope**: implicit | **Intent**: troubleshooting
- **ES candidates** (top-16):
  - [ ] `global_sop_integer_plus_all_pm_manometer` (score=8.8197, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER Plus_REP_MANOMETER Global SOP No: Revision No: 1 Page: 18 / 20 ## 8. Appendix 계측모드 Mode + Set 3초길게 
  - [ ] `global_sop_precia_all_pm_manometer` (score=8.313, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_MANOMETER Global SOP No: Revision No: 0 Page: 20 / 20 ## 8. Appendix | Flow | Proced
  - [ ] `global_sop_supra_xp_all_pm_manometer` (score=8.1372, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_MANOMETER ADJUST Global SOP No : Revision No: 1 Page: 27/32 | Flow | Procedure
  - [ ] `set_up_manual_supra_np` (score=8.0477, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I # 11. TTTM (※환경안전 보호구: 안전모, 안전화) ## 11.1. Common List ### 11.1.2 Temp Limit Controller Settin
  - [ ] `global_sop_supra_n_series_all_sub_unit_flow_switch` (score=7.8218, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_SUB UNIT_FLOW SWITCH Global SOP No : Revision No: 1 Page: 15 / 17 | Flow | P
  - [ ] `set_up_manual_supra_nm` (score=7.7017, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.4 Device Net Calibration _ PSK Board ### 12.4.1 ATM P
  - [ ] `global_sop_supra_n_series_all_sub_unit_manometer` (score=7.5373, device=SUPRA N, type=SOP)
    > # Global SOP_SUPRA N series_ADJ_SUB UNIT_MANOMETER Global SOP No: Revision No: 0 Page: 31/32 ## 8. Appendix ### 계측모드 Mod
  - [ ] `precia_all_trouble_shooting_guide_pin_motor_abnormal` (score=7.4321, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # PRECIA Trouble Shooting Guide [Pin Motor Abnormal] Confidential II | Alarm Code | LED 점멸 횟수 | Alarm 종류 | 원
  - [ ] `set_up_manual_supra_n` (score=7.4034, device=SUPRA N, type=SOP)
    > ```markdown Confidential I # 11. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 11.1. Common List ### 11.1.2 Temp Limit Controller Setti
  - [ ] `global_sop_supra_n_series_all_pm_device_net_board` (score=7.3581, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_DEVICE NET BOARD Global SOP No : Revision No: 2 Page: 40/44 | Flow | Proc
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=7.1237, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier Global SOP No: 0 Revision No: 0 Page: 28 / 43 | Flow | Procedur
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=7.0976, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `global_sop_supra_xp_all_pm_device_net_board` (score=6.7962, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_DEVICE NET BOARD REPLACEMENT Global SOP No: Revision No: 0 Page: 14 / 34 | Flo
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=6.6972, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_ADJ_EFEM_ROBOT TEACHING Global SOP No: Revision No: 3 Page: 32 / 126 | Flow | Procedure | Tool
  - [ ] `set_up_manual_precia` (score=6.5629, device=PRECIA, type=set_up_manual)
    > | | h. Z-Axis 및 정렬된 상태에서 'TEACH' Key Click i. 'USE ROBOT' Click j. 우측 Teaching Data 값 확인 k. Z-Axis의 경우 Stage1,2를 더한 값에서 
  - [ ] `precia_pm_trouble_shooting_guide_rf_power_abnormal` (score=6.4997, device=PRECIA, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [RF Power Abnormal] Use this guide to diagnose problems with the [RF Power Abnormal]. It descri

#### q_id: `A-imp002`
- **Question**: Chamber 내부 파티클 원인 분석 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, SUPRA_VPLUS, PRECIA]
- **Scope**: implicit | **Intent**: troubleshooting
- **ES candidates** (top-10):
  - [ ] `40043020` (score=6.4554, device=SUPRA Vplus, type=myservice)
    > -. 원인 분석 중
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock_o_ring` (score=5.1365, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_LOAD LOCK O-RING Global SOP No: S-KG-R019-R0 Revision No: 0 Page: 20 / 30 | Fl
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=4.701, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `set_up_manual_supra_np` (score=4.6578, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I | 2) Jig Assy 조립도 확인 | a. Jig Assy 조립도를 확인하여 다음과 같은 방법으로 Jig Assy가 설치되어야 한다. | | | :--- | :--
  - [ ] `global_sop_geneva_xp_rep_bubbler_cabinet_drain_valve` (score=4.5929, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_BUBBLER # CABINET_DRAIN VALVE Global SOP No: S-KG-R034-R0 Revision No: 0 Page: 15
  - [ ] `set_up_manual_supra_nm` (score=4.5651, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.2 EFEM Robot Leveling & Loadport Teaching | Picture
  - [ ] `global_sop_geneva_xp_rep_pm_support_pin` (score=4.3339, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVA xp_REP_PM_Support pin Global SOP No: 0 Revision No: 1 Page: 16 / 25 ## 10. Work Procedure | Flow | P
  - [ ] `global_sop_geneva_xp_adj_pm_pin_alignment` (score=4.312, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Pin Alignment Global SOP No: S-KG-A003-R0 Revision No: 0 Page: 11 / 25 ## 10. 
  - [ ] `set_up_manual_supra_n` (score=4.2907, device=SUPRA N, type=SOP)
    > ```markdown Confidential I 36) LP2,3 Teaching <!-- Image (70, 70, 290, 226) --> a. LP1 Teaching 과 동일한 방법으로 LP2,3 를 진행하여 
  - [ ] `global_sop_geneva_xp_sw_efem_log_backup` (score=4.0195, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVAxp_SW_EFEM_Log backup Global SOP No: Revision No: 0 Page: 5/13 ## 6. Flow Chart Start ↓ 1. SOP 및 안전사항

#### q_id: `A-imp003`
- **Question**: Endpoint 신호 약해질 때 OES Window 점검 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA]
- **Scope**: implicit | **Intent**: troubleshooting
- **ES candidates** (top-27):
  - [ ] `set_up_manual_supra_nm` (score=5.8986, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.4 Device Net Calibration _ PSK Board ### 12.4.1 ATM P
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=5.7936, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 2/15 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_n_series_all_pm_heater_chuck` (score=5.7932, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_HEATER CHUCK Global SOP No : Revision No: 2 Page: 19/40 | Flow | Procedur
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=5.7499, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `global_sop_geneva_xp_rep_efem_ffu_controller` (score=5.5494, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_FFU Controller | SOP No: 0 | Revision No: 0 | Page: 3/21 | | --- | --- | ---
  - [ ] `global_sop_geneva_xp_rep_efem_ffu_filter` (score=5.532, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_FFU Filter | SOP No: S-KG-R041-R0 | | --- | | Revision No: 0 | | Page: 3 / 1
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=5.5194, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) TC Wafer 설치 확인 | ![](https://i.imgur.com/1234567.pn
  - [ ] `global_sop_geneva_xp_rep_efem_ffu` (score=5.5183, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_FFU | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3 / 14 | 
  - [ ] `global_sop_geneva_xp_adj_all_sw_install` (score=5.4161, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_AII_SW INSTALL | SOP No: 0 | | | | --- | --- | --- | | Revision No: 0 | | | | Pag
  - [ ] `global_sop_geneva_xp_rep_pm_device_net` (score=5.403, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Device net | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/15
  - [ ] `global_sop_geneva_xp_rep_pm_elbow_heater` (score=5.4022, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Elbow heater | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/
  - [ ] `global_sop_geneva_xp_rep_pm_insulation_heater` (score=5.4008, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Insulation heater | SOP No: 0 | | | | | :--- | :--- | :--- | :--- | | Revision
  - [ ] `global_sop_geneva_xp_rep_pm_o2_analyzer` (score=5.3807, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_O2 analyzer(Delta F) | SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Page
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=5.3707, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page :101 / 105 ## Safety 
  - [ ] `global_sop_supra_n_series_all_pm_hook_lifter_servo_motor` (score=5.3319, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_HOOK LIFTER ## SERVO MOTOR Global SOP No : Revision No: 2 Page : 4/106 # 
  - [ ] `global_sop_supra_xp_all_pm_manometer` (score=5.2699, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_MANOMETER Global SOP No : Revision No: 1 Page: 2/32 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_precia_all_pm_manometer` (score=5.2641, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_MANOMETER Global SOP No: Revision No: 0 Page: 2/20 ## 1. Safety 1) 안전 및 주의사항 - 장비가 작
  - [ ] `global_sop_integer_plus_all_pm_manometer` (score=5.2607, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER Plus_ALL_MANOMETER Global SOP No: Revision No: 1 Page: 2/20 ## 1. Safety 1) 안전 및 주의사항 -
  - [ ] `global_sop_precia_all_efem_side_storage` (score=5.2226, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_EFEM_SIDE # STORAGE_YEST SOP No : Revision No : 0 Page : 2/37 ## 1. Safety 1) 안전 및 주
  - [ ] `set_up_manual_precia` (score=5.2074, device=PRECIA, type=set_up_manual)
    > # 공통사항 1. 보호구를 사용하지 않아도 근로자가 유해/위험작업으로부터 보호를 받을 수 있도록 설비 개선 등 필요 조치를 진행한다. 2. 필요 조치를 이행하였음에도 유해 / 위험 요인은 제거하기가 어려울 때. 제한
  - [ ] `set_up_manual_supra_np` (score=5.1992, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I 6) ZERO Calibration <!-- Table (58, 69, 928, 339) --> \begin{tabular}{|l|l|l|} \hline \textbf
  - [ ] `global_sop_integer_plus_all_efem_eda_controller` (score=5.1964, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_EFEM_EDA # CONTROLLER Global_SOP No: Revision No: 1 Page: 2/22 ## 1. Safety 1)
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=5.1707, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_EFEM_ROBOT TEACHING Global SOP No: Revision No: 3 Page: 43 / 126 | Flow | Proc
  - [ ] `global_sop_supra_xp_all_tm_dc_power_supply` (score=5.1625, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ ALL_TM_DC POWER SUPPLY Global SOP No: Revision No: 0 Page: 2/18 ## 1. Safety 1) 안전 및
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3000qc` (score=5.1545, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM 3000,3100QC SOURCE IGNITION WINDOW CHECK Global SOP No: Revision No: 2 P
  - [ ] `global_sop_geneva_xp_rep_pm_o2_analyzer_teledyne` (score=5.1378, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_O2 analyzer(Teledyne -> Teledyne) | SOP No: | S-KG-R027-R1 | | --- | --- | | R
  - [ ] `set_up_manual_supra_n` (score=5.1137, device=SUPRA N, type=SOP)
    > ```markdown Confidential I 36) LP2,3 Teaching <!-- Image (70, 70, 290, 226) --> a. LP1 Teaching 과 동일한 방법으로 LP2,3 를 진행하여 

#### q_id: `A-imp004`
- **Question**: Gas 유량 편차 발생 시 MFC 점검 순서는?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, GENEVA_XP]
- **Scope**: implicit | **Intent**: troubleshooting
- **ES candidates** (top-19):
  - [ ] `global_sop_supra_xp_all_sub_unit_igs_block` (score=5.9349, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_SUB UNIT_MFC Global SOP No : Revision No: 2 Page: 20/72 | Flow | Procedure | Tool
  - [ ] `global_sop_supra_n_all_sub_unit_mfc` (score=5.7001, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_SUB UNIT_MFC Global SOP No: 0 Revision No: 3 Page: 13/21 | Flow | Procedure | Tool 
  - [ ] `precia_all_trouble_shooting_guide_trace_gas_abnormal` (score=5.4379, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [GAS Abnormal] Confidential II | | | | | :--- | :--- | :--- | | | | I/O Check | | |
  - [ ] `global_sop_precia_all_pm_mfc` (score=5.4354, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_MFC Global SOP No : Revision No: 1 Page : 16 / 23 ## 6. Work Procedure | Flow | P
  - [ ] `global_sop_geneva_xp_rep_pm_mfc` (score=5.4282, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_MFC | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/18 | | ## 3. 
  - [ ] `supra_n_all_trouble_shooting_guide_trace_gas_flow_abnormal` (score=5.3929, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace gas flow abnormal] Confidential II | | | | | :--- | :--- | :--- | | | | ▶ Sw
  - [ ] `set_up_manual_integer_plus` (score=5.3843, device=INTEGER plus, type=SOP)
    > ```markdown # 17-23. MFC 인증 | Picture | Description | Data | OK | NG | N/A | |---|---|---|---|---|---| | | PM, AM Chambe
  - [ ] `set_up_manual_precia` (score=5.104, device=PRECIA, type=set_up_manual)
    > ```markdown <!-- Image (68, 41, 363, 195) --> # 9. Gas Turn On (환경안전 보호구: 안전모, 안전화, 방독면, 보안경) ## 9.3 Bulk, Toxic Gas Tur
  - [ ] `global_sop_integer_plus_all_pm_igs_block` (score=5.0286, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_IGS BLOCK Global SOP No : Revision No: 0 Page: 14 / 26 | Flow | Procedure |
  - [ ] `global_sop_supra_xp_all_tm_mfc` (score=4.8555, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_MFC Global SOP No: Revision No: 0 Page: 14 / 18 | Flow | Procedure | Tool & Po
  - [ ] `set_up_manual_ecolite_2000` (score=4.8146, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 8. Part Installation | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) Ceramic Parts<br>
  - [ ] `set_up_manual_supra_nm` (score=4.7093, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.1. Common List ### 12.1.9 Gas Pressure | Picture | De
  - [ ] `global_sop_supra_xp_all_sub_unit_gas_box_board` (score=4.6881, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_SUB UNIT_GAS BOX BOARD Global SOP No : Revision No : 1 Page : 17 / 18 | Flow | Pr
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pressure_switch` (score=4.6284, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Vacuum Pressure switch | SOP No: 0 | | |---|---| | Revision No: 0 | | | Page: 3
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pump` (score=4.6154, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Vacuum pump | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/21 | 
  - [ ] `global_sop_geneva_xp_rep_pm_heat_exchanger` (score=4.6062, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Heat exchanger Global SOP No: Revision No: 0 Page: 3 / 14 ## 3. 사고 사례 ### 1. 협착
  - [ ] `global_sop_geneva_xp_rep_pm_pressure_switch` (score=4.6056, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Pressure switch | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/1
  - [ ] `40035345` (score=4.6011, device=SUPRA V, type=myservice)
    > -. LOG 확인 시 FFU 점검 이전 부터 EFEM TO CTC Communication Alarm 발생
-> EFEM TO CTC Communication LOG 끊김 확인
-> 고객 Inform 완료
-. PM
  - [ ] `global_sop_geneva_xp_rep_efem_load_port` (score=4.5859, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Load port | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/18 | | 

#### q_id: `A-imp005`
- **Question**: Heater 온도 Overshoot 발생 시 조치 방법은?
- **Devices**: [SUPRA_N, SUPRA_VPLUS, SUPRA_XP]
- **Scope**: implicit | **Intent**: troubleshooting
- **ES candidates** (top-27):
  - [ ] `global_sop_supra_n_series_all_pm_gas_feed_through` (score=5.7546, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS FEED THROUGH Global SOP No: Revision No: 4 Page: 4 / 18 ## 3. 사고 사례 #
  - [ ] `global_sop_supra_n_series_all_pm_pressure_gauge` (score=5.7462, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_PRESSURE GAUGE Global SOP No: Revision No: 3 Page: 3/46 ## 3. 사고 사례 ### 3
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=5.7407, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_AM_BARATRON GAUGE Global SOP No: Revision No: Page: 8 / 135 ## 3. 사고 사례 ### 1)
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=5.7399, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 77 / 105 ## 사고 사례 ##
  - [ ] `global_sop_supra_xp_all_pm_baratron_gauge` (score=5.7385, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_BARATRON GAUGE Global SOP No : 0 Revision No : 0 Page : 3/33 ## 3. 사고 사례 ### 1
  - [ ] `global_sop_supra_n_series_all_pm_top_lid` (score=5.7344, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N Series_ALL_PM_TOP LID Global SOP No: Revision No: 0 Page: 4 / 48 ## 3-2 화상 ### 1) 화상 재해
  - [ ] `global_sop_supra_xp_all_pm_pirani_gauge` (score=5.7308, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIRANI GAUGE Global SOP No: Revision No: 1 Page: 3/40 ## 3. 사고 사례 ### 1) 화상 재해
  - [ ] `global_sop_supra_xp_all_pm_pressure_gauge` (score=5.7274, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRESSURE GAUGE Global SOP No: Revision No: 1 Page: 3/34 ## 3. 사고 사례 ### 1) 화상 
  - [ ] `global_sop_supra_xp_all_tm_mfc` (score=5.7134, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_MFC Global SOP No: Revision No: 0 Page: 3/18 ## 3. 사고 사례 ### 1) 화상 재해의정의 불이나 뜨
  - [ ] `global_sop_supra_xp_all_pm_process_kit` (score=5.7129, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PROCESS KIT Global SOP No : Revision No : 3 Page : 4/28 ## 3-2 화상 ### 1) 화상 재해
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=5.647, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 3/124 ## 3. 사고 사례 ### 1) 화상의 정의 불
  - [ ] `global_sop_supra_xp_all_pm_dual_epd` (score=5.6456, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_DUAL EPD Global SOP No: Revision No: 1 Page: 3/40 ## 3. 사고 사례 ### 1) 화상의 정의 불이
  - [ ] `global_sop_geneva_xp_rep_pm_heater_chuck_without_jig` (score=5.4936, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA xp_REP_PM_Heater chuck w/o jig | SOP No: 0 | | | |---|---|---| | Revision No: 1 | | | | Page: 3/52 
  - [ ] `global_sop_supra_n_series_all_pm_heater_chuck` (score=5.4858, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_HEATER CHUCK Global SOP No : Revision No: 2 Page: 3/40 ## 3. 사고 사례 ### 1)
  - [ ] `global_sop_geneva_xp_rep_pm_harmonic_drive` (score=5.4362, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Harmonic drive | Global SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Pag
  - [ ] `global_sop_supra_n_series_all_pm_epd` (score=5.4354, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_EPD Global SOP No : Revision No: 3 Page: 3/49 ## 3. 사고 사례 ### 1) 화상 재해의 정
  - [ ] `global_sop_supra_n_series_all_pm_hook_lifter_servo_motor` (score=5.4332, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_HOOK LIFTER ## SERVO MOTOR Global SOP No : Revision No: 2 Page : 3 / 106 
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=5.418, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier | Global SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Pag
  - [ ] `global_sop_geneva_xp_rep_pm_adapter_ring` (score=5.4134, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Adapter ring Global SOP No: 0 Revision No: 0 Page: 4 / 30 ## 3. 사고 사례 ### 1) 화
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=5.387, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 4/47 ## 3. 사고 사례 ### 1) 화상 재해의 
  - [ ] `global_sop_supra_xp_all_pm_heater_chuck` (score=5.322, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP ZEDIUS XP_ALL_PM_HEATER # CHUCK Global SOP No: 0 Revision No: 2 Page: 3/49 ## 3. 사고 사례 ### 1) 화
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock_o_ring` (score=5.3016, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_LOAD LOCK O-RING | Global SOP No: | S-KG-R019-R0 | | --- | --- | | Revision No
  - [ ] `global_sop_precia_all_pm_pendulum_valve` (score=5.2775, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_PENDULUM VALVE Global SOP No : Revision No: 0 Page: 3/32 ## 3. 사고 사례 ### 1) 화상 재해
  - [ ] `global_sop_supra_xp_all_pm_pendulum_valve` (score=5.2705, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PENDULUM VALVE Global SOP No : Revision No: 6 Page: 3/26 ## 3. 사고 사례 ### 1) 화상
  - [ ] `global_sop_geneva_xp_adj_pm_pin_alignment` (score=5.2701, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Pin Alignment | Global SOP No: | S-KG-A003-R0 | | --- | --- | | Revision No: |
  - [ ] `global_sop_supra_n_series_all_pm_process_kit` (score=5.2689, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_PROCESS KIT Global SOP No : Revision No: 0 Page: 3/55 ## 3. 사고 사례 ### 1) 
  - [ ] `global_sop_geneva_xp_rep_pm_support_pin` (score=5.2564, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Support pin | Global SOP No: | 0 | | --- | --- | | Revision No: | 1 | | Page: 

#### q_id: `A-imp006`
- **Question**: Servo Alarm 해제 후 Robot Teaching 재설정 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, INTEGER_PLUS]
- **Scope**: implicit | **Intent**: procedure
- **ES candidates** (top-12):
  - [ ] `set_up_manual_supra_nm` (score=7.3239, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.8 EFEM BM Teaching (Single) | Picture | Description
  - [ ] `set_up_manual_supra_n` (score=7.199, device=SUPRA N, type=SOP)
    > ```markdown Confidential I | 36) EFEM Single Teaching | a. 방법은 동일하지만 Teaching 변경 시 TM Robot 이 아닌 EFEM Robot Teaching 을 재
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=7.073, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_HOOK LIFTER ## SERVO MOTOR CONTROLLER REPLACEMENT Global SOP No: Revision No: 
  - [ ] `set_up_manual_ecolite_3000` (score=7.0076, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 7. Teaching_Cooling Stage | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 14) TM Robot Mo
  - [ ] `global_sop_integer_plus_all_tm_robot` (score=6.9556, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_ALL_TM_ROBOT Global SOP No : Revision No: 4 Page :103 / 103 ## 8. Appendix [EnM_SOP] A.I.D DAT
  - [ ] `set_up_manual_supra_vm` (score=6.7527, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 9. TM Robot Teaching_Cooling Stage 1 | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 14) 
  - [ ] `global_sop_supra_xp_all_pm_heater_chuck` (score=6.4468, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP ZEDIUS XP_ALL_PM_HEATER # CHUCK Global SOP No: 0 Revision No: 2 Page: 21 / 49 | Flow | Procedur
  - [ ] `set_up_manual_supra_np` (score=6.3851, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I 36) LP2,3 Teaching <!-- Image (60, 70, 364, 227) --> a. LP1 Teaching 과 동일한 방법으로 LP2,3 를 진행하여 
  - [ ] `global_sop_precia_all_efem_ctc` (score=6.3281, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_EFEM_CTC Global SOP No: Revision No: 1 Page: 12/51 ## 6. Work Procedure | Flow | Pro
  - [ ] `global_sop_supra_n_all_efem_robot_m124` (score=6.2782, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_EFEM ROBOT_M124 Global SOP No : Revision No: 1 Page: 41/45 | Flow | Procedure | Too
  - [ ] `global_sop_supra_xp_all_tm_robot` (score=6.0553, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_ROBOT # TEACHING Global SOP No: Revision No : 3 Page : 41 / 47 | Flow | Proced
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=6.009, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced

#### q_id: `A-imp007`
- **Question**: Throttle Valve Position Offset 조정 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, INTEGER_PLUS]
- **Scope**: implicit | **Intent**: procedure
- **ES candidates** (top-10):
  - [ ] `set_up_manual_supra_np` (score=5.5138, device=SUPRA Np, type=set_up_manual)
    > Confidential I | 3) BM1 Ready Move | a. Position CST_L_R1_RDY[4] Click<br>b. Move to Click<br>c. Edit -> 기존 Z-Axis Data에
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=5.2762, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_ADJ_EFEM_ROBOT TEACHING Global SOP No: Revision No: 3 Page: 32 / 126 | Flow | Procedure | Tool
  - [ ] `set_up_manual_supra_nm` (score=5.2301, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.2 EFEM Robot Leveling & Loadport Teaching | Picture
  - [ ] `set_up_manual_supra_n` (score=5.222, device=SUPRA N, type=SOP)
    > ```markdown Confidential I | 22) LP 진입 전 Offset 적용 | a. Position CST_U_R1_RDY[1] Click. | EFEM Robot Offset Parameter | 
  - [ ] `global_sop_supra_n_series_all_pm_pressure_gauge` (score=4.9236, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_PIRANI GAUGE Global SOP No: Revision No: 3 Page: 32/46 | Flow | Procedure
  - [ ] `global_sop_precia_all_pm_prevent_maintenance` (score=4.8782, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_PM_GAP TEACHING Global SOP No: Revision No: 5 Page :100 / 108 ## 6. Work Procedure |
  - [ ] `global_sop_supra_n_series_all_pm_device_net_board` (score=4.8436, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_DEVICE NET BOARD Global SOP No : Revision No: 2 Page: 38/44 | Flow | Proc
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=4.694, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=4.6858, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_HOOK LIFTER ## SERVO MOTOR CONTROLLER REPLACEMENT Global SOP No: Revision No: 
  - [ ] `global_sop_supra_xp_all_pm_heater_chuck` (score=4.6733, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP ZEDIUS XP_ALL_PM_HEATER # CHUCK Global SOP No: 0 Revision No: 2 Page: 21 / 49 | Flow | Procedur

#### q_id: `A-imp008`
- **Question**: LP Transfer 에러 발생 시 초기 점검 항목은?
- **Devices**: [SUPRA_N, SUPRA_XP, SUPRA_VPLUS, INTEGER_PLUS]
- **Scope**: implicit | **Intent**: troubleshooting
- **ES candidates** (top-21):
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=5.7113, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 14. Customer certification | Picture | Description | Spec | Check | Result | | :--- | :--- | :--- | :--- |
  - [ ] `set_up_manual_supra_xq` (score=4.7498, device=SUPRA XQ, type=SOP)
    > | 55 | PM | DN131 | D-NET | □Y □N | 95 | PM | DN131 | D-NET | □Y □N | |---|---|---|---|---|---|---|---|---|---| | 56 | P
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=4.6072, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `set_up_manual_integer_plus` (score=4.5506, device=INTEGER plus, type=SOP)
    > ```markdown # 15. Transfer Test (환경안전 보호구: 안전모, 안전화) ## 15.1 Common list ### 15.1.2 Transfer Test | Picture | Descriptio
  - [ ] `set_up_manual_supra_np` (score=4.4443, device=SUPRA Np, type=set_up_manual)
    > | 9) SUB2 I/F Panel - PM Cable Hook up | a. Sub Unit2 의 PM 방향 Interface Panel Inner Cable Hook up 을 진행한다. b. 장착되는 Cable 
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=4.253, device=ZEDIUS XP, type=set_up_manual)
    > | 35) Teaching Data Save Check | a. Pendent 초기 화면에서 CST_L_R1[1] Data Check 후 MDI Bottom Click. | LP2 CST_L_R1[2] Click L
  - [ ] `set_up_manual_supra_n` (score=4.2173, device=SUPRA N, type=SOP)
    > ```markdown # 4. Cable Hook Up (※환경안전 보호구 : 안전모, 안전화, 안전대, 보안경 보호가운, 헤드랜턴) ## 4.4 Sub Unit Cable Hook Up | Picture | Des
  - [ ] `global_sop_supra_n_series_all_pcw_turn_on` (score=4.2059, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_PM PCW TURN ON Global SOP No: Revision No: 1 Page: 12/31 | Flow | Procedure 
  - [ ] `global_sop_precia_all_efem_load_port_leveling` (score=4.1399, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_EFEM_LOAD PORT LEVELING Global SOP No : Revision No: 0 Page: 15 / 19 | Flow | Proced
  - [ ] `set_up_manual_supra_nm` (score=4.1369, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 4. Cable Hook Up (※환경안전 보호구 : 안전모, 안전화, 안전대, 보안경 보호가운, 헤드랜턴) ## 4.4 Sub Unit Cable Hook Up 
  - [ ] `global_sop_geneva_xp_all_8계통_check` (score=4.1336, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA XP_8계통_Check sheet SOP No: 0 Revision No: 0 Page: 18 / 46 ## 10. Work Procedure | Flow | Procedure 
  - [ ] `all_all_trouble_shooting_guide_trace_ffu_abnormal` (score=4.119, device=etc, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace FFU Abnormal] Confidential II | Failure symptoms | Check point | Key point |
  - [ ] `global_sop_integer_plus_all_ll_lifter_assy` (score=4.0894, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_LL_BUSH, BELLOWS, SHAFT Global SOP No : Revision No: 0 Page: 36/81 | 작업 | Chec
  - [ ] `global_sop_geneva_xp_rep_pm_mfc` (score=4.0703, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_MFC SOP No: 0 Revision No: 0 Page: 12 / 18 ## 10. Work Procedure | Flow | Proc
  - [ ] `40038138` (score=4.0435, device=SUPRA N, type=myservice)
    > -. LP1 9,10 슬랏 픽 명령이후 Alarm발생
-. 실제로 pick success상태
-. FM Log확인시 리시브 RS232 String read fail
-. 통신문제 추정
-> TM FFU포트와 SWAP
  - [ ] `global_sop_geneva_xp_adj_post_align_application` (score=4.0314, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_Post align application Global SOP No: 0 Revision No: 1 Page: 10 / 20 | Flow | Procedu
  - [ ] `global_sop_integer_plus_all_pm_manometer` (score=4.0239, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER Plus_REP_MANOMETER Global SOP No: Revision No: 1 Page: 20 / 20 ## 8. Appendix | Flow | 
  - [ ] `global_sop_geneva_xp_rep_pm_insulation_heater` (score=4.0206, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Insulation heater | SOP No: 0 | | | | | :--- | :--- | :--- | :--- | | Revision
  - [ ] `global_sop_geneva_xp_rep_pm_device_net` (score=4.0205, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Device net | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/15
  - [ ] `global_sop_geneva_xp_rep_pm_elbow_heater` (score=4.0099, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Elbow heater | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/
  - [ ] `40052963` (score=3.9994, device=SUPRA Vplus, type=myservice)
    > -. LP2 Door Close Alarm 점검

#### q_id: `A-imp009`
- **Question**: Process Recipe 변경 후 공정 검증 절차는?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA]
- **Scope**: implicit | **Intent**: procedure
- **ES candidates** (top-19):
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=6.7098, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 2) Load | a. Load를 클릭한다. | | | | | | | 3) Set-up Recipe
  - [ ] `global_sop_supra_n_mfg_all_aging_run` (score=5.0995, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_MFG_ALL_AGING RUN Global SOP No: - Revision No: 1 Page: 6 / 14 ## 6. Work Procedure | F
  - [ ] `set_up_manual_supra_n` (score=5.0389, device=SUPRA N, type=SOP)
    > ```markdown Confidential I | 3) Making Recipe | a. 좌측 Operation Menu 하단의 [New]를 선택한다. | | | :--- | :--- | :--- | | | b. 
  - [ ] `global_sop_integer_plus_all_pm_wall_temp_controller` (score=5.0014, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_WALL TEMP CONTROLLER Global SOP No: Revision No: 0 Page: 17 / 21 | Flow | P
  - [ ] `set_up_manual_ecolite_3000` (score=4.8948, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 14. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) Making 
  - [ ] `global_sop_supra_n_series_all_pm_dual_epd` (score=4.8203, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_DUAL EPD REPLACEMENT & CALIBRATION Global SOP No: Revision No: 4 Page: 15
  - [ ] `set_up_manual_supra_np` (score=4.8038, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I # 14. Process Confirm (※환경안전 보호구: 안전모, 안전화) ## 14.1 Aging Test | Picture | Description | Tool
  - [ ] `set_up_manual_ecolite_ii_400` (score=4.7819, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 14. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) Making 
  - [ ] `set_up_manual_supra_xq` (score=4.7134, device=SUPRA XQ, type=SOP)
    > ```markdown # 13. Process Confirm (※환경안전 보호구: 안전모, 안전화) ## 13-1. Aging Test | Picture | Description | Tool & Spec | | :-
  - [ ] `set_up_manual_ecolite_2000` (score=4.687, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 13. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) Making 
  - [ ] `set_up_manual_supra_nm` (score=4.5503, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 14. Process Confirm (※환경안전 보호구 : 안전모, 안전화) ## 14.1 Aging Test | Picture | Description | Too
  - [ ] `global_sop_integer_plus_all_pm_vacuum_line` (score=4.5176, device=INTEGER XP, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_PM_PENDULUM VALVE Global SOP No: Revision No: Page : 58 / 133 | Flow | Procedu
  - [ ] `global_sop_geneva_xp_adj_post_align_application` (score=4.4904, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_Post align application Global SOP No: 0 Revision No: 1 Page: 16 / 20 | Flow | Procedu
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=4.4732, device=ZEDIUS XP, type=set_up_manual)
    > ```markdown # 14. Process Confirm (※환경안전 보호구 : 안전모, 안전화) ## 14.1 Aging Test | Picture | Description | Tool & Spec | | :-
  - [ ] `set_up_manual_integer_plus` (score=4.4729, device=INTEGER plus, type=SOP)
    > ```markdown # 14. Aging Test (환경안전 보호구: 안전모, 안전화) ## 14.1 Common list ### 14.1.2 Aging Test | Picture | Description | To
  - [ ] `global_sop_geneva_xp_adj_pm_pin_alignment` (score=4.4138, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Pin Alignment Global SOP No: S-KG-A003-R0 Revision No: 0 Page: 20 / 25 ## 10. 
  - [ ] `global_sop_integer_plus_all_am_devicenet_board` (score=4.396, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_DEVICENET BOARD Global SOP No : Revision No: 0 Page: 55/58 | Flow | Procedu
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=4.3521, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_AM_PENDULUM VALVE Global SOP No: Revision No: Page : 70 / 135 | Flow | Procedu
  - [ ] `set_up_manual_supra_vm` (score=4.3372, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 16. Process Check | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 6) PR Wafer가 든 Foup 준비 

#### q_id: `A-imp010`
- **Question**: Chamber Idle 시간이 길 때 재가동 전 점검 항목은?
- **Devices**: [SUPRA_N, SUPRA_XP, SUPRA_VPLUS, PRECIA, INTEGER_PLUS]
- **Scope**: implicit | **Intent**: procedure
- **ES candidates** (top-20):
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=5.8953, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 14. Customer certification | Picture | Description | Spec | Check | Result | | :--- | :--- | :--- | :--- |
  - [ ] `global_sop_geneva_xp_rep_pm_disc` (score=5.2865, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc SOP No: 0 Revision No: 0 Page: 8/31 ## 10. Work Procedure | Flow | Proced
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock` (score=5.2779, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Load lock SOP No: 0 Revision No: 1 Page: 8 / 22 ## 10. Work Procedure | Flow |
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pump` (score=5.2686, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Vacuum pump SOP No: 0 Revision No: 0 Page: 8/21 ## 10. Work Procedure | Flow |
  - [ ] `set_up_manual_supra_np` (score=5.1806, device=SUPRA Np, type=set_up_manual)
    > | 9) SUB2 I/F Panel - PM Cable Hook up | a. Sub Unit2 의 PM 방향 Interface Panel Inner Cable Hook up 을 진행한다. b. 장착되는 Cable 
  - [ ] `global_sop_geneva_xp_rep_pm_elbow_heater` (score=5.1622, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Elbow heater SOP No: 0 Revision No: 0 Page: 9 / 22 ## 10. Work Procedure | Flo
  - [ ] `global_sop_geneva_xp_rep_pm_insulation_heater` (score=5.1567, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Insulation heater SOP No: 0 Revision No: 0 Page: 8/21 ## 10. Work Procedure | 
  - [ ] `global_sop_geneva_xp_adj_pm_chuck_temp_calibration` (score=5.1533, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Chuck temp calibration Global SOP No: Revision No: 0 Page: 3/21 ## 3. 재해 방지 대책
  - [ ] `set_up_manual_integer_plus` (score=5.0899, device=INTEGER plus, type=SOP)
    > ```markdown # 17-23. MFC 인증 | Picture | Description | Data | OK | NG | N/A | |---|---|---|---|---|---| | | PM, AM Chambe
  - [ ] `set_up_manual_supra_n` (score=4.9071, device=SUPRA N, type=SOP)
    > ```markdown # 4. Cable Hook Up (※환경안전 보호구 : 안전모, 안전화, 안전대, 보안경 보호가운, 헤드랜턴) ## 4.4 Sub Unit Cable Hook Up | Picture | Des
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=4.9008, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 35/47 ## 10. Work Procedure | F
  - [ ] `global_sop_geneva_xp_rep_pm_heater_chuck_without_jig` (score=4.8473, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA xp_REP_PM_Heater chuck w/o jig SOP No: 0 Revision No: 1 Page: 41 / 52 ## 10. Work Procedure | Flow 
  - [ ] `global_sop_supra_xp_all_pm_cip_chamber` (score=4.8216, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_CIP CHAMBER | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page: 5 
  - [ ] `set_up_manual_ecolite_2000` (score=4.8144, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 1. Installation Preperation (Layout, etc.) | Picture | Description | Tool & Spec | | :--- | :--- | :--- | 
  - [ ] `set_up_manual_supra_nm` (score=4.8068, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 4. Cable Hook Up (※환경안전 보호구 : 안전모, 안전화, 안전대, 보안경 보호가운, 헤드랜턴) ## 4.4 Sub Unit Cable Hook Up 
  - [ ] `global_sop_integer_plus_all_tm_ctc_controller` (score=4.7633, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_CTC # CONTROLLER Global SOP No: Revision No: 1 Page: 23/51 | Flow | Procedu
  - [ ] `global_sop_geneva_xp_rep_pm_device_net` (score=4.7631, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Device net | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/15
  - [ ] `global_sop_geneva_xp_adj_all_sw_install` (score=4.7582, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_AII_SW INSTALL | SOP No: 0 | | | | --- | --- | --- | | Revision No: 0 | | | | Pag
  - [ ] `global_sop_geneva_xp_rep_pm_o2_analyzer` (score=4.7552, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_O2 analyzer(Delta F) | SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Page
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=4.7137, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 27 / 105 ## Safety 1

#### q_id: `A-imp011`
- **Question**: Wafer 틀어짐 발생 시 Aligner 점검 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, INTEGER_PLUS]
- **Scope**: implicit | **Intent**: troubleshooting
- **ES candidates** (top-16):
  - [ ] `40055879` (score=8.0149, device=SUPRA Vplus, type=myservice)
    > -. IB Flow 틀어짐
  - [ ] `40065488` (score=7.7982, device=SUPRA Vplus, type=myservice)
    > -. TCB 교체 후 Temp 틀어짐
  - [ ] `40054656` (score=7.614, device=SUPRA Vplus, type=myservice)
    > -. D-Net Reset후 Pin Position 틀어짐
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=6.5279, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 8) Aligner 이동 | a. EFEM Robot Pendent 하단에 있는 Servo 를 누른
  - [ ] `set_up_manual_ecolite_3000` (score=5.975, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 7. Teaching_Aligner Stage | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 26) Hand Check 
  - [ ] `precia_ll_trouble_shooting_guide_trace_aligner_alarm` (score=5.9106, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Aligner Alarm] Confidential II | Failure symptoms | Check point | Key point 
  - [ ] `global_sop_geneva_xp_adj_post_align_application` (score=5.9063, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_Post align application Global SOP No: 0 Revision No: 1 Page: 13 / 20 | Flow | Procedu
  - [ ] `global_sop_supra_n_series_all_pm_heater_chuck` (score=5.806, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_HEATER CHUCK Global SOP No : Revision No: 2 Page: 21 / 40 | Flow | Proced
  - [ ] `set_up_manual_supra_vm` (score=5.7758, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 8. EFEM Robot Teaching_Cooling Stage 2_Single | Picture | Description | Tool & Spec | | :--- | :--- | :---
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=5.5476, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `40043406` (score=5.2232, device=SUPRA Vplus, type=myservice)
    > -. EPAGQ04 EFEM Robot Teaching 불량
-> Buffer Stage Aligner 걸치면서 Wafer Place
  - [ ] `global_sop_precia_all_ll_aligner` (score=5.1046, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_LL_ALIGNER PAD Global SOP No : Revision No: 1 Page: 17 / 35 ## 6. Work Procedure | F
  - [ ] `set_up_manual_supra_xq` (score=5.0625, device=SUPRA XQ, type=SOP)
    > ```markdown # 3. Docking (※환경안전 보호구: 안전모, 안전화) ## 3-9 TM Robot End Effector | Picture | Description | Tool & Spec | | :-
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=5.0584, device=ZEDIUS XP, type=set_up_manual)
    > ```markdown # 3. Docking (※환경안전 보호구 : 안전모, 안전화) ## 3.4 TM Robot End Effector 장착 | Picture | Description | Tool & Spec | 
  - [ ] `set_up_manual_supra_nm` (score=5.024, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.2 EFEM Robot Leveling & Loadport Teaching | Picture
  - [ ] `set_up_manual_precia` (score=4.9163, device=PRECIA, type=set_up_manual)
    > ```markdown # 10. Teaching (환경안전 보호구: 안전모, 안전화) ## 10.8 TM Robot Teaching (Load Lock1 Aligner Centering) | Picture | Des

#### q_id: `A-imp012`
- **Question**: PM 후 첫 Lot 투입 전 Dummy Run 절차는?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, INTEGER_PLUS]
- **Scope**: implicit | **Intent**: procedure
- **ES candidates** (top-22):
  - [ ] `set_up_manual_supra_n` (score=5.1299, device=SUPRA N, type=SOP)
    > ```markdown Confidential I | 4) Wafer 매수 설정 후 Strat | a. Select All 후 Job Start | | | :--- | :--- | :--- | | | b. pc Che
  - [ ] `set_up_manual_supra_np` (score=5.1249, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I | 4) Wafer 매수 설정 후 Strat | a. Select All 후 Job Start | | | :--- | :--- | :--- | | | b. pc Che
  - [ ] `40034336` (score=5.0298, device=SUPRA Vplus, type=myservice)
    > -. PM1, 2, 3 Chamber 상부 PM 실시(Process Kit / Chuck / Exhuast Ring / Top Lid)
-> PM1, 2 Process Kit는 작일 삼구가 진행하였다 인폼하여 미 진
  - [ ] `global_sop_supra_n_series_all_efem_ffu` (score=4.6973, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_EFEM_FFU | Global SOP No: | | | --- | --- | | Revision No: 4 | | | Page: 2/5
  - [ ] `set_up_manual_supra_nm` (score=4.6674, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.2 EFEM Robot Leveling & Loadport Teaching | Picture
  - [ ] `global_sop_integer_plus_all_pm_vacuum_line` (score=4.4466, device=INTEGER XP, type=SOP)
    > ```markdown # Global SOP_INTEGER plus _REP_PM_Vacuum Line Global SOP No: Revision No: Page : 125 / 133 | Flow | Procedur
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3100qc` (score=4.4383, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PRISM SOURCE 3100QC REPLACEMENT Global SOP No: Revision No : 1 Page : 17/72 | Flo
  - [ ] `global_sop_supra_n_mfg_all_aging_run` (score=4.3343, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_MFG_ALL_AGING RUN | Global SOP No: | - | | --- | --- | | Revision No: | 1 | | Page: | 1
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=4.2529, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM ROBOT ASSY REPLACEMENT Global SOP No : Revision No: 6 Page : 39 / 107 | F
  - [ ] `global_sop_integer_plus_all_ll_vacuum_line` (score=4.2499, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_LL_PIRANI BARATRON Global SOP No: Revision No: Page: 47 / 77 | Flow | Procedur
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=4.2463, device=ZEDIUS XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) PM Moving<br><img src="image_url" alt="PM Moving"> |
  - [ ] `global_sop_supra_n_series_all_tm_ctc` (score=4.1453, device=SUPRA N, type=SOP)
    > Global SOP_SUPRA N series_SW_TM_CTC PATCH | Global SOP No : | | | --- | --- | | Revision No: 6 | | | Page : 53 / 81 | | 
  - [ ] `global_sop_supra_xp_all_sub_unit_igs_block` (score=4.1422, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_SUB UNIT_MFC Global SOP No : Revision No: 2 Page: 22/72 ## 7. 작업 Check Sheet | 작업
  - [ ] `global_sop_supra_vplus_all_pm_controller` (score=4.1403, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_SW_PM_Controller PATCH Global SOP No: Revision No: 3 Page: 26 / 42 ## 3. 사고 사례 1) 고
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=4.1209, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 2) Load | a. Load를 클릭한다. | | | | | | | 3) Set-up Recipe
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=4.0953, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No: 13 Page: 61/75 ## Scope 이 G
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=4.0597, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_BARATRON GAUGE Global SOP No: Revision No: Page : 53 / 135 | Flow | Procedu
  - [ ] `global_sop_precia_all_pm_gap_sensor` (score=4.0098, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_VIEW PORT QUARTZ Global SOP No : Revision No: 1 Page: 65 / 79 ## 6. Work Procedur
  - [ ] `global_sop_supra_n_series_all_pm_process_kit` (score=4.0093, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_CLN_PM_CHAMBER Global SOP No : Revision No: 0 Page: 33 / 55 | Flow | Procedure |
  - [ ] `global_sop_supra_xp_all_pm_chamber_safety_cover` (score=3.9873, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_SAFETY COVER REPLACEMENT | Global SOP No : | | | --- | --- | | Revision No : 1
  - [ ] `global_sop_integer_plus_all_efem_controller` (score=3.9862, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_EFEM_CONTROLLER | Global SOP No: | | |---|---| | Revision No: 1 | | | Page: 3/
  - [ ] `global_sop_supra_xp_all_pm_process_kit` (score=3.9839, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PROCESS KIT Global SOP No : Revision No : 3 Page : 17 / 28 | Flow | 절차 | Tool 

#### q_id: `A-imp013`
- **Question**: Backside He Leak 발생 시 점검 항목은?
- **Devices**: [SUPRA_N, SUPRA_XP, SUPRA_VPLUS]
- **Scope**: implicit | **Intent**: troubleshooting
- **ES candidates** (top-24):
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=5.711, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 14. Customer certification | Picture | Description | Spec | Check | Result | | :--- | :--- | :--- | :--- |
  - [ ] `global_sop_integer_plus_all_ll_vacuum_line` (score=5.6015, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_LL_ISOLATION ## VALVE Global SOP No: Revision No: Page: 61 / 77 | Flow | Proce
  - [ ] `global_sop_integer_plus_all_ll_slot_valve` (score=5.0871, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_LL_SLOT ## VALVE Global SOP No : Revision No : 4 Page : 26 / 50 | Flow | Proce
  - [ ] `set_up_manual_supra_xq` (score=4.7357, device=SUPRA XQ, type=SOP)
    > | 55 | PM | DN131 | D-NET | □Y □N | 95 | PM | DN131 | D-NET | □Y □N | |---|---|---|---|---|---|---|---|---|---| | 56 | P
  - [ ] `global_sop_supra_n_series_all_pm_heater_chuck` (score=4.7207, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_HEATER CHUCK Global SOP No : Revision No: 2 Page: 27 / 40 | Flow | Proced
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=4.5961, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `global_sop_precia_all_pm_wafer_centering` (score=4.5646, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_PM_WAFER CENTERING Global SOP No: Revision No: 0 Page: 16/23 | Flow | Procedure | To
  - [ ] `set_up_manual_supra_np` (score=4.4494, device=SUPRA Np, type=set_up_manual)
    > | 9) SUB2 I/F Panel - PM Cable Hook up | a. Sub Unit2 의 PM 방향 Interface Panel Inner Cable Hook up 을 진행한다. b. 장착되는 Cable 
  - [ ] `global_sop_integer_plus_all_am_pin_motor` (score=4.4113, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_PIN BELLOWS Global SOP No: Revision No: 4 Page: 37 / 84 | Flow | Procedure 
  - [ ] `set_up_manual_precia` (score=4.3282, device=PRECIA, type=set_up_manual)
    > | 9. Front, Backside가 동일한 경우 | a. Front, Backside Etch가 동일한 경우, TM Robot Offset (Parameter변경) 으로 만 조절하여 Centering 진행 | |
  - [ ] `global_sop_integer_plus_all_am_slot_valve` (score=4.235, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_CLN_AM_SLOT VALVE Global SOP No : Revision No : 0 Page : 36 / 39 | Flow | Procedur
  - [ ] `global_sop_integer_plus_all_pm_slot_valve` (score=4.2281, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_SLOT VALVE Global SOP No : Revision No : 2 Page : 20 / 39 | Flow | Procedur
  - [ ] `set_up_manual_supra_n` (score=4.2188, device=SUPRA N, type=SOP)
    > ```markdown # 4. Cable Hook Up (※환경안전 보호구 : 안전모, 안전화, 안전대, 보안경 보호가운, 헤드랜턴) ## 4.4 Sub Unit Cable Hook Up | Picture | Des
  - [ ] `global_sop_integer_plus_all_ll_lifter_assy` (score=4.1548, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_LL_BUSH, BELLOWS, SHAFT Global SOP No : Revision No: 0 Page: 31/81 | Flow | Pr
  - [ ] `all_all_trouble_shooting_guide_trace_ffu_abnormal` (score=4.1461, device=etc, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace FFU Abnormal] Confidential II | Failure symptoms | Check point | Key point |
  - [ ] `global_sop_integer_plus_all_tm_vacuum_line` (score=4.1394, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_ALL_TM_VACUUM LINE Global SOP No: Revision No: 1 Page: 26 / 26 ## 8. Appendix - NTEGER plus - 
  - [ ] `set_up_manual_supra_nm` (score=4.1357, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 4. Cable Hook Up (※환경안전 보호구 : 안전모, 안전화, 안전대, 보안경 보호가운, 헤드랜턴) ## 4.4 Sub Unit Cable Hook Up 
  - [ ] `global_sop_geneva_xp_all_8계통_check` (score=4.1204, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA XP_8계통_Check sheet SOP No: 0 Revision No: 0 Page: 18 / 46 ## 10. Work Procedure | Flow | Procedure 
  - [ ] `40038715` (score=4.0873, device=SUPRA Vplus, type=myservice)
    > -. EFEM Robot Upper Endeffector Rep
-. Upper Arm leak 정상 확인
-. Lower Leak 확인
-> 초당 1kpa 수준 Leak 발생
-. Aging 1Lot 정상 확인
-
  - [ ] `global_sop_supra_n_series_all_pm_process_kit` (score=4.0499, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_PROCESS KIT Global SOP No : Revision No: 0 Page: 21 / 55 | Flow | Procedu
  - [ ] `global_sop_geneva_xp_rep_pm_insulation_heater` (score=4.0189, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Insulation heater | SOP No: 0 | | | | | :--- | :--- | :--- | :--- | | Revision
  - [ ] `global_sop_integer_plus_all_pm_manometer` (score=4.005, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER Plus_REP_MANOMETER Global SOP No: Revision No: 1 Page: 20 / 20 ## 8. Appendix | Flow | 
  - [ ] `global_sop_geneva_xp_rep_pm_elbow_heater` (score=4.0041, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Elbow heater | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/
  - [ ] `global_sop_geneva_xp_rep_pm_device_net` (score=3.9951, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Device net | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/15

#### q_id: `A-imp014`
- **Question**: Remote Plasma Source 점검 및 교체 기준은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA]
- **Scope**: implicit | **Intent**: procedure
- **ES candidates** (top-25):
  - [ ] `set_up_manual_supra_vm` (score=5.3919, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 17. Interlock Check _ S/W | Module | Action | A Prior Condition | 1st Check | 2nd Check | | :--- | :--- | 
  - [ ] `global_sop_geneva_xp_rep_pm_loadlock_apc_valve` (score=4.6843, device=geneva_xp_rep, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Loadlock APC Global SOP No: Revision No: 0 Page: 19 / 22 | Flow | Procedure | 
  - [ ] `global_sop_geneva_xp_rep_pm_differential_gauge` (score=4.6829, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Differential gauge Global SOP No: Revision No: 0 Page: 15 / 18 | Flow | Proced
  - [ ] `global_sop_geneva_xp_rep_pm_chamber_apc_valve` (score=4.6807, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Chamber APC Global SOP No: Revision No: 0 Page: 19 / 22 | Flow | Procedure | T
  - [ ] `global_sop_integer_plus_all_pm_dc_cooling_fan_motor` (score=4.5449, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_DC # COOLING FAN MOTOR Global SOP No: Revision No: 1 Page: 13 / 16 | Flow |
  - [ ] `global_sop_geneva_xp_rep_pm_heater_chuck_without_jig` (score=4.4957, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA xp_REP_PM_ Heater chuck w/o jig SOP No: 0 Revision No: 1 Page: 48 / 52 | Flow | Procedure | Tool & 
  - [ ] `global_sop_geneva_xp_rep_pm_harmonic_drive` (score=4.4955, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Harmonic drive Global SOP No: 0 Revision No: 0 Page: 31 / 31 | Flow | Procedur
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=4.4935, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier Global SOP No: 0 Revision No: 0 Page: 39 / 43 | Flow | Procedur
  - [ ] `global_sop_geneva_xp_rep_pm_adapter_ring` (score=4.4753, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Adapter ring Global SOP No: 0 Revision No: 0 Page: 26 / 30 ## 10. Work Procedu
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=4.4724, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 43 / 47 ## 10. Work Procedure |
  - [ ] `set_up_manual_ecolite_2000` (score=4.394, device=ECOLITE 2000, type=set_up_manual)
    > | | | Pressure < 700,000 mT | | | |---|---|---|---|---| | | | Pressure > 850,000 mT | | | | | Door Valve Open | ATM Sens
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=4.3098, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 1. Install Preparation(※환경안전 보호구 : 안전모, 안전화) ## 1.1 Foot Print Drawing | Picture | Description | Tool & Sp
  - [ ] `global_sop_geneva_xp_rep_efem_robot_sr8240` (score=4.2956, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_ROBOT SR8240 Global SOP No: 0 Revision No: 0 Page: 14 / 17 | Flow | Procedur
  - [ ] `set_up_manual_supra_xq` (score=4.2083, device=SUPRA XQ, type=SOP)
    > | Module | Action | A Prior Condition | 1st Check | 2nd Check | | :--- | :--- | :--- | :--- | :--- | | | Gas1 O2 Valve O
  - [ ] `set_up_manual_ecolite_3000` (score=4.1508, device=ECOLITE3000, type=set_up_manual)
    > ```markdown | 1. Installation Preperation (Layout, etc.) | | | | :--- | :--- | :--- | | Picture | Description | Tool & S
  - [ ] `global_sop_omnis_plus_adj_pm_rf_power_on_test_eng` (score=4.1314, device=OMNIS plus, type=SOP)
    > ```markdown # Global SOP_OMNIS plus_ADJ_PM_RF Power On Test_ENG Global SOP No : 0 Revision No : 0 Page : 17 / 17 ## 12. 
  - [ ] `supra_n_all_trouble_shooting_guide_trace_fcip_abnormal` (score=4.0794, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace FCIP abnormal] Confidential II | Date | Revision | Reviser | Revision conten
  - [ ] `set_up_manual_ecolite_ii_400` (score=4.0604, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 1. Installation Preperation_Template Draw | Picture | Description | Tool & Spec | | :--- | :--- | :--- | |
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=4.0253, device=ZEDIUS XP, type=set_up_manual)
    > ```markdown # 1. Install Preperation (※환경안전 보호구 : 안전모, 안전화) ## 1.1 Foot Print Drawing | Picture | Description | Tool & S
  - [ ] `global_sop_precia_all_efem_ctc` (score=4.012, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_EFEM_CTC Global SOP No: Revision No: 1 Page: 23/51 | Flow | Procedure | Tool & Spec 
  - [ ] `global_sop_integer_plus_all_pm_gas_line` (score=3.997, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_GAS FILTER Global_SOP No: Revision No: 0 Page: 34 / 75 ## 3. Part 위치 | 평면도 
  - [ ] `global_sop_supra_n_series_all_pm_source_box_interface_board` (score=3.9939, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_SOURCE BOX INTERFACE BOARD Global SOP No: Revision No: 4 Page: 7 / 18 ## 
  - [ ] `global_sop_supra_xp_all_pm_prism_source` (score=3.9658, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM SOURCE REPLACEMENT Global SOP No: Revision No : 2 Page : 26 / 69 | Flow 
  - [ ] `global_sop_precia_all_tm_sensor_board` (score=3.9625, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_TM_SENSOR BOARD Global SOP No: Revision No: 1 Page: 16 / 18 | Flow | Procedure | Too
  - [ ] `global_sop_precia_all_ll_relief_valve` (score=3.9583, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_LL_RELIEF VALVE Global SOP No: Revision No: 1 Page: 16 / 18 | Flow | Procedure | Too

#### q_id: `A-imp015`
- **Question**: Recipe Step별 Process Condition 확인 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA]
- **Scope**: implicit | **Intent**: information_lookup
- **ES candidates** (top-17):
  - [ ] `set_up_manual_ecolite_3000` (score=6.4229, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 14. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Making 
  - [ ] `global_sop_precia_all_pm_wafer_centering` (score=6.2323, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_PM_WAFER CENTERING Global SOP No: Revision No: 0 Page: 12/23 | Flow | Procedure | To
  - [ ] `set_up_manual_ecolite_2000` (score=6.187, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 13. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Making 
  - [ ] `set_up_manual_ecolite_ii_400` (score=6.1831, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 14. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Making 
  - [ ] `set_up_manual_supra_xq` (score=6.1818, device=SUPRA XQ, type=SOP)
    > ```markdown # 13. Process Confirm (※환경안전 보호구: 안전모, 안전화) ## 13-1. Aging Test | Picture | Description | Tool & Spec | | :-
  - [ ] `set_up_manual_supra_n` (score=6.1252, device=SUPRA N, type=SOP)
    > ```markdown Confidential I | 3) Making Recipe | a. 좌측 Operation Menu 하단의 [New]를 선택한다. | | | :--- | :--- | :--- | | | b. 
  - [ ] `set_up_manual_supra_np` (score=5.9822, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I # 5) Making Recipe <!-- Image (60, 70, 348, 254) --> a. [Insert Before]를 선택한다. b. 수정할 수 있는 Re
  - [ ] `set_up_manual_supra_vm` (score=5.7917, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 14. Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Making Recipe | a. [Ins
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=5.638, device=ZEDIUS XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Making Recipe<br><img src="image_url" alt="Image 1">
  - [ ] `set_up_manual_supra_nm` (score=5.6113, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 14. Process Confirm (※환경안전 보호구 : 안전모, 안전화) ## 14.1 Aging Test | Picture | Description | Too
  - [ ] `global_sop_precia_all_efem_ctc` (score=5.3121, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_S/W Install_EFEM_CTC Global SOP No: Revision No: 1 Page: 48/51 | Flow | Procedure | Tool
  - [ ] `global_sop_omnis_plus_pm_er_check_eng` (score=5.2629, device=OMNIS plus, type=SOP)
    > ```markdown # Global SOP_OMNIS Plus_PM_ER_CHECK Global SOP No: Revision No: 0 Page: 9 / 13 ## 9. Work Procedure | Flow |
  - [ ] `set_up_manual_integer_plus` (score=5.2432, device=INTEGER plus, type=SOP)
    > ```markdown # 14. Aging Test (환경안전 보호구: 안전모, 안전화) ## 14.1 Common list ### 14.1.2 Aging Test | Picture | Description | To
  - [ ] `global_sop_supra_series_all_sw_operation` (score=4.7884, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_ SW OPERATION Global SOP No Revision No: 2 Page: 30/49 ## 6. Log Back Up - Wor
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=4.7435, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `supra_n_all_trouble_shooting_guide_trace_microwave_abnormal` (score=4.6265, device=SUPRA N, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace Microwave Abnormal] ## 8.5.38 Power Watchdog Timer (Version 2.4 and 2.7 Software ONLY) *
  - [ ] `global_sop_omnis_sw_all_process_check_eng` (score=4.6022, device=OMNIS, type=SOP)
    > ```markdown # Global SOP_OMNIS_SW_ALL_PROCESS CHECK Global SOP No: Revision No: 2 Page: 12 / 19 | Flow | Procedure | Too

#### q_id: `A-imp016`
- **Question**: Gas Line에서 미세 Leak 검출 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, GENEVA_XP]
- **Scope**: implicit | **Intent**: troubleshooting
- **ES candidates** (top-15):
  - [ ] `global_sop_supra_series_all_sw_operation` (score=6.9394, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_ SW OPERATION Global SOP No Revision No: 2 Page: 10/49 ## 2. Gas line Leak Che
  - [ ] `set_up_manual_supra_n` (score=6.7839, device=SUPRA N, type=SOP)
    > Confidential I 2) Leak Check a. Pump Turn On 후 8시간 Full Pumping 후 Leak Check를 한다. b. Leak Check 조건 및 Spec은 고객사마다 상이. | |
  - [ ] `set_up_manual_ecolite_ii_400` (score=6.6864, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 11. Leak Check | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) Leak Check | a. Leak Ch
  - [ ] `set_up_manual_supra_xq` (score=6.6135, device=SUPRA XQ, type=SOP)
    > ```markdown # 10. Toxic Gas Turn On (※환경안전 보호구: 안전모, 안전화) ## 10-1 Toxic Gas Line Check | Picture | Description | Tool & 
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=6.6125, device=ZEDIUS XP, type=set_up_manual)
    > ```markdown # 10. Toxic Gas Turn On (※환경안전 보호구 : 안전모, 안전화) ## 10.1 Toxic Gas Line Check | Picture | Description | Tool &
  - [ ] `set_up_manual_precia` (score=6.2065, device=PRECIA, type=set_up_manual)
    > | | | | |---|---|---| | 3. Gas box manual valve open | a. Gas Regulator Full open<br>b. Gas manual valve lock key 제거<br>
  - [ ] `set_up_manual_ecolite_3000` (score=6.143, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 11. Leak Check | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Pumping Check | a. Loca
  - [ ] `integer_plus_all_trouble_shooting_guide_trace_leak_rate_over` (score=5.9519, device=INTEGER plus, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace leak rate over] ## Appendix #2 ### A. #### 1. PM He Leak Check Point I - Bottom Gas Line
  - [ ] `set_up_manual_ecolite_2000` (score=5.9028, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 10. Leak Check | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Pumping Check | a. Loca
  - [ ] `set_up_manual_supra_np` (score=5.6192, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I # 10. Leak Check (※환경안전 보호구: 안전모, 안전화) ## 10.1 Leak Check | Picture | Description | Tool & Sp
  - [ ] `set_up_manual_supra_nm` (score=5.6116, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 11. Leak Check (※환경안전 보호구 : 안전모, 안전화) ## 11.1 Leak Check | Picture | Description | Tool & S
  - [ ] `set_up_manual_supra_vm` (score=5.607, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 10. Pump Turn On 및 Leak Check | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Pumping 
  - [ ] `set_up_manual_integer_plus` (score=5.5511, device=INTEGER plus, type=SOP)
    > | 4) NF3 Line Leak Check | a. PM Chamber를 NF3 Line Used 로 변경하여 Manual Leak Check 하여 Result 값을 확인한다. | ※ Leak Spec Chambe
  - [ ] `global_sop_supra_xp_all_sub_unit_igs_block` (score=5.3405, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_SUB UNIT_IGS BLOCK Global SOP No : Revision No: 2 Page: 49/71 | Flow | Procedure 
  - [ ] `global_sop_supra_n_all_sub_unit_igs_block` (score=5.3003, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_SUB UNIT_IGS BLOCK Global SOP No : Revision No: 1 Page: 47 / 67 | Flow | Procedure 

#### q_id: `A-imp017`
- **Question**: Pumping Speed 저하 원인 분석 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, GENEVA_XP, INTEGER_PLUS]
- **Scope**: implicit | **Intent**: troubleshooting
- **ES candidates** (top-18):
  - [ ] `40043020` (score=6.4452, device=SUPRA Vplus, type=myservice)
    > -. 원인 분석 중
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=4.7125, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `set_up_manual_supra_np` (score=4.6603, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I | 2) Jig Assy 조립도 확인 | a. Jig Assy 조립도를 확인하여 다음과 같은 방법으로 Jig Assy가 설치되어야 한다. | | | :--- | :--
  - [ ] `set_up_manual_supra_nm` (score=4.5747, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.2 EFEM Robot Leveling & Loadport Teaching | Picture
  - [ ] `set_up_manual_supra_n` (score=4.2969, device=SUPRA N, type=SOP)
    > ```markdown Confidential I 36) LP2,3 Teaching <!-- Image (70, 70, 290, 226) --> a. LP1 Teaching 과 동일한 방법으로 LP2,3 를 진행하여 
  - [ ] `set_up_manual_supra_xq` (score=4.0569, device=SUPRA XQ, type=SOP)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 4-10) Robot Speed 변경 | a. Speed Click. | | | | | | | 4-
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=4.0494, device=ZEDIUS XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 10) Robot Speed 변경 | a. Speed Click. | | | | | | | 11) 
  - [ ] `global_sop_geneva_xp_sw_efem_log_backup` (score=4.0294, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVAxp_SW_EFEM_Log backup Global SOP No: Revision No: 0 Page: 5/13 ## 6. Flow Chart Start ↓ 1. SOP 및 안전사항
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=3.9283, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM ROBOT ASSY REPLACEMENT Global SOP No : Revision No: 6 Page: 49 / 107 ## 8
  - [ ] `40059539` (score=3.8766, device=SUPRA Vplus, type=myservice)
    > -. 원인파악중
  - [ ] `global_sop_precia_all_pm_slot_valve` (score=3.8742, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_SLOT ## VALVE(PRESYS) Global SOP No : 0 Revision No : 0 Page : 18 / 36 | Flow | P
  - [ ] `set_up_manual_supra_vm` (score=3.8583, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 8. EFEM Robot Teaching_Loadport 1 | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 26) Rob
  - [ ] `global_sop_geneva_xp_adj_efem_robot_teaching` (score=3.858, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_EFEM_Robot teaching Global SOP No: S-KG-R045-R0 Revision No: 0 Page: 16/28 ## 10.
  - [ ] `global_sop_geneva_xp_rep_bubbler_cabinet_drain_valve` (score=3.7955, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp REP_BUBBLER # CABINET_DRAIN VALVE Global SOP No: S-KG-R034-R0 Revision No: 0 Page: 17
  - [ ] `global_sop_geneva_xp_rep_bubbler_cabinet_feed_valve` (score=3.7904, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_BUBBLER # CABINET_FEED VALVE Global SOP No: S-KG-R030-R0 Revision No: 0 Page: 18 
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock_door_o_ring` (score=3.7534, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_LOAD LOCK DOOR O-RING Global SOP No: S-KG-R020-R0 Revision No: 0 Page: 16/23 |
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock_o_ring` (score=3.7486, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_LOAD LOCK O-RING Global SOP No: S-KG-R019-R0 Revision No: 0 Page: 23 / 30 | Fl
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=3.7411, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_EFEM_ROBOT TEACHING Global SOP No: Revision No: 3 Page: 43 / 126 | Flow | Proc

#### q_id: `A-imp018`
- **Question**: Electrode 수명 판단 기준은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, INTEGER_PLUS]
- **Scope**: implicit | **Intent**: information_lookup
- **ES candidates** (top-20):
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=4.0966, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 1. Install Preparation(※환경안전 보호구 : 안전모, 안전화) ## 1.1 Foot Print Drawing | Picture | Description | Tool & Sp
  - [ ] `set_up_manual_ecolite_3000` (score=3.8955, device=ECOLITE3000, type=set_up_manual)
    > ```markdown | 1. Installation Preperation (Layout, etc.) | | | | :--- | :--- | :--- | | Picture | Description | Tool & S
  - [ ] `set_up_manual_ecolite_2000` (score=3.8946, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 1. Installation Preperation (Layout, etc.) | Picture | Description | Tool & Spec | | :--- | :--- | :--- | 
  - [ ] `set_up_manual_ecolite_ii_400` (score=3.8237, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 1. Installation Preperation_Template Draw | Picture | Description | Tool & Spec | | :--- | :--- | :--- | |
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=3.6715, device=ZEDIUS XP, type=set_up_manual)
    > ```markdown # 1. Install Preperation (※환경안전 보호구 : 안전모, 안전화) ## 1.1 Foot Print Drawing | Picture | Description | Tool & S
  - [ ] `set_up_manual_supra_vm` (score=3.6484, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 1. Template Draw | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) Template Drawing 사전 준
  - [ ] `global_sop_precia_all_pm_mfc` (score=3.5969, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_MFC Global SOP No : Revision No: 1 Page : 9/23 ## 4. 필요 Tool | | Name | 미세 Driver
  - [ ] `global_sop_integer_plus_all_pm_igs_block` (score=3.5226, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_IGS BLOCK Global SOP No : Revision No: 0 Page: 9/26 ## 4. 필요 Tool | | Name 
  - [ ] `set_up_manual_supra_xq` (score=3.3606, device=SUPRA XQ, type=SOP)
    > ```markdown # 1. Install Preparation (Layout, etc) (※환경안전 보호구: 안전모, 안전화) | Picture | Description | Tool & Spec | | :--- 
  - [ ] `set_up_manual_supra_nm` (score=3.2574, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I | 7) Sub Unit2 PCW V/V Open | a. Sub Unit2 에 위치한 PCW IN / OUT Manual V/V를 시계방향으로 돌려 Open 한다. 
  - [ ] `set_up_manual_integer_plus` (score=3.2369, device=INTEGER plus, type=SOP)
    > ```markdown # 17-8. Cable Hook up | Picture | Description | Data | OK | NG | N/A | | :--- | :--- | :--- | :--- | :--- | 
  - [ ] `set_up_manual_precia` (score=3.1921, device=PRECIA, type=set_up_manual)
    > # 6. Part Installation ## 6.4 Top Process Kit Install | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1
  - [ ] `40034602` (score=3.1049, device=SUPRA V, type=myservice)
    > -. 설비단 EFEM, TM Signal Tower 교체 완료
  - [ ] `global_sop_supra_xp_all_pm_manometer` (score=3.0965, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_MANOMETER ADJUST Global SOP No : Revision No: 1 Page: 27/32 | Flow | Procedure
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=3.0624, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM ROBOT ASSY REPLACEMENT Global SOP No : Revision No: 6 Page : 22 / 107 | F
  - [ ] `set_up_manual_supra_n` (score=3.0477, device=SUPRA N, type=SOP)
    > ```markdown # 안전가이드 ## 1. 작업 전 안전가이드 - 1 작업위치, 주의사항 확인 - 2 전도 방지 장치 (트리거) 설치 - 3 고정장치, 미끌림 방지대, 안전 걸이줄 확인 必 - 4 벨트식 안전벨트
  - [ ] `global_sop_supra_vplus_all_pm_controller` (score=3.0397, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_SW_PM_Controller PATCH Global SOP No: Revision No: 3 Page: 38 / 42 | Flow | Procedu
  - [ ] `global_sop_precia_all_pm_prevent_maintenance` (score=3.0248, device=PRECIA, type=SOP)
    > # Global SOP_PRECIA_REP_PM_PROCESS KIT (TOP MOUNT TYPE) Global SOP No: Revision No: 5 Page: 24 / 108 ## 6. Work Procedur
  - [ ] `set_up_manual_supra_np` (score=3.0089, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I Analog Output Tap은 조작하지 않는다. # 12. TTTM (※환경안전 보호구: 안전모, 안전화) ## 12.5 Device Net Calibration 
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=2.9997, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced

#### q_id: `A-imp019`
- **Question**: Power Ramp Up 시 Arcing 방지 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, OMNIS]
- **Scope**: implicit | **Intent**: procedure
- **ES candidates** (top-13):
  - [ ] `40046755` (score=5.0351, device=SUPRA Nm, type=myservice)
    > -. 고온 방지 Cover 장착
-. Temp Limit Setting
-. 정기 PM 중이라 Baffle, Focus Adaptor 장착방법 고객 및 PM업체 공유
-. 고객측에서 Applicator Tube 교체
  - [ ] `set_up_manual_supra_nm` (score=4.9918, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.2 EFEM Robot Leveling & Loadport Teaching | Picture
  - [ ] `set_up_manual_integer_plus` (score=4.8001, device=INTEGER plus, type=SOP)
    > ```markdown # 17-8. Cable Hook up | Picture | Description | Data | OK | NG | N/A | | :--- | :--- | :--- | :--- | :--- | 
  - [ ] `set_up_manual_precia` (score=4.791, device=PRECIA, type=set_up_manual)
    > ```markdown # 5. Cable Hook Up (※환경안전 보호구: 안전모, 안전화, 안전대, 보안경 보호가운, 헤드랜턴) ## 5.1 Rack - Module Cable Hookup | Picture | 
  - [ ] `set_up_manual_supra_np` (score=4.7359, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I <!-- Table (58, 69, 928, 731) --> \begin{tabular}{|l|l|l|} \hline \textbf{4) Analog IO Calibr
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=4.6898, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `set_up_manual_supra_n` (score=4.6128, device=SUPRA N, type=SOP)
    > ```markdown Confidential I 36) LP2,3 Teaching <!-- Image (70, 70, 290, 226) --> a. LP1 Teaching 과 동일한 방법으로 LP2,3 를 진행하여 
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=4.5949, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_HOOK LIFTER ## SERVO MOTOR CONTROLLER REPLACEMENT Global SOP No: Revision No: 
  - [ ] `global_sop_supra_n_series_all_pm_heater_chuck` (score=4.4398, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_HEATER CHUCK Global SOP No : Revision No: 2 Page: 19/40 | Flow | Procedur
  - [ ] `global_sop_supra_xp_all_pm_heater_chuck` (score=4.3394, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP ZEDIUS XP_ALL_PM_HEATER # CHUCK Global SOP No: 0 Revision No: 2 Page: 21 / 49 | Flow | Procedur
  - [ ] `global_sop_supra_n_series_all_pm_top_lid` (score=4.266, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N Series_CLN_PM_TOP LID Global SOP No: Revision No: 0 Page: 17 / 48 | Flow | | Tool & Poi
  - [ ] `global_sop_supra_n_series_all_pm_isolation_valve` (score=4.0835, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_ISOLATION VALVE Global SOP No : Revision No: 2 Page: 5/25 ## 5. Worker Lo
  - [ ] `global_sop_geneva_xp_rep_efem_ffu_controller` (score=4.061, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_FFU Controller | SOP No: 0 | Revision No: 0 | Page: 3/21 | | --- | --- | ---

#### q_id: `A-q001_masked`
- **Question**: [DEVICE] PM Chamber 내부 View Port 쪽에 Local Plasma 및 Arcing이 발생하는 원인은 무엇인가?
- **Devices**: [ECOLITE_3000]
- **Scope**: implicit | **Intent**: troubleshooting
- **ES candidates** (top-26):
  - [ ] `set_up_manual_integer_plus` (score=10.7803, device=INTEGER plus, type=SOP)
    > ```markdown # 17-21. Load port 인증 | Picture | Description | Data | OK | NG | N/A | | :--- | :--- | :--- | :--- | :--- | 
  - [ ] `global_sop_supra_n_all_pm_fcip_r3` (score=9.6628, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_CLN_PM_FCIP R3 QUARTZ TUBE Global SOP No : Revision No: 3 Page: 83/84 ## 7. 작업 Check Sh
  - [ ] `supra_n_all_trouble_shooting_guide_trace_device_net_abnormal` (score=9.5291, device=SUPRA XP, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Device Net Abnormal] Use this guide to diagnose problems with the [Trace Dev
  - [ ] `supra_xp_all_trouble_shooting_guide_trace_device_net_abnormal` (score=9.527, device=SUPRA XP, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Device Net Abnormal] Use this guide to diagnose problems with the [Trace Dev
  - [ ] `global_sop_geneva_xp_rep_pm_device_net` (score=9.4705, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Device net | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/15
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=9.3904, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 3/124 ## 3. 사고 사례 ### 1) 화상의 정의 불
  - [ ] `global_sop_supra_xp_all_pm_dual_epd` (score=9.3757, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_DUAL EPD Global SOP No: Revision No: 1 Page: 3/40 ## 3. 사고 사례 ### 1) 화상의 정의 불이
  - [ ] `global_sop_supra_n_series_all_pm_gas_feed_through` (score=9.2797, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS FEED THROUGH Global SOP No: Revision No: 4 Page: 4 / 18 ## 3. 사고 사례 #
  - [ ] `global_sop_supra_n_series_all_pm_heater_chuck` (score=9.2377, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_HEATER CHUCK Global SOP No : Revision No: 2 Page: 3/40 ## 3. 사고 사례 ### 1)
  - [ ] `global_sop_supra_xp_all_pm_baratron_gauge` (score=9.2358, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_BARATRON GAUGE Global SOP No : 0 Revision No : 0 Page : 3/33 ## 3. 사고 사례 ### 1
  - [ ] `global_sop_precia_all_pm_pendulum_valve` (score=9.2292, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_PENDULUM VALVE Global SOP No : Revision No: 0 Page: 3/32 ## 3. 사고 사례 ### 1) 화상 재해
  - [ ] `global_sop_supra_xp_all_pm_pendulum_valve` (score=9.2286, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PENDULUM VALVE Global SOP No : Revision No: 6 Page: 3/26 ## 3. 사고 사례 ### 1) 화상
  - [ ] `global_sop_geneva_xp_adj_pm_pin_alignment` (score=9.225, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Pin Alignment | Global SOP No: | S-KG-A003-R0 | | --- | --- | | Revision No: |
  - [ ] `global_sop_geneva_xp_rep_pm_heater_chuck_without_jig` (score=9.2248, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA xp_REP_PM_Heater chuck w/o jig | SOP No: 0 | | | |---|---|---| | Revision No: 1 | | | | Page: 3/52 
  - [ ] `global_sop_supra_n_series_all_pm_top_lid` (score=9.2207, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N Series_ALL_PM_TOP LID Global SOP No: Revision No: 0 Page: 4 / 48 ## 3-2 화상 ### 1) 화상 재해
  - [ ] `global_sop_geneva_xp_rep_pm_support_pin` (score=9.2191, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Support pin | Global SOP No: | 0 | | --- | --- | | Revision No: | 1 | | Page: 
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=9.2146, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 77 / 105 ## 사고 사례 ##
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=9.2135, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 4/47 ## 3. 사고 사례 ### 1) 화상 재해의 
  - [ ] `global_sop_supra_xp_all_pm_process_kit` (score=9.2122, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PROCESS KIT Global SOP No : Revision No : 3 Page : 4/28 ## 3-2 화상 ### 1) 화상 재해
  - [ ] `global_sop_supra_xp_all_pm_pirani_gauge` (score=9.2089, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIRANI GAUGE Global SOP No: Revision No: 1 Page: 3/40 ## 3. 사고 사례 ### 1) 화상 재해
  - [ ] `global_sop_supra_xp_all_pm_pressure_gauge` (score=9.205, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRESSURE GAUGE Global SOP No: Revision No: 1 Page: 3/34 ## 3. 사고 사례 ### 1) 화상 
  - [ ] `global_sop_geneva_xp_adj_pm_chuck_temp_calibration` (score=9.1992, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Chuck temp calibration Global SOP No: Revision No: 0 Page: 3/21 ## 3. 재해 방지 대책
  - [ ] `global_sop_supra_n_series_all_pm_epd` (score=9.1796, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_EPD Global SOP No : Revision No: 3 Page: 3/49 ## 3. 사고 사례 ### 1) 화상 재해의 정
  - [ ] `global_sop_geneva_xp_rep_pm_adapter_ring` (score=9.1737, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Adapter ring Global SOP No: 0 Revision No: 0 Page: 4 / 30 ## 3. 사고 사례 ### 1) 화
  - [ ] `global_sop_geneva_xp_rep_pm_harmonic_drive` (score=9.1721, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Harmonic drive | Global SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Pag
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=9.1621, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier | Global SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Pag

#### q_id: `A-q003_masked`
- **Question**: SEC SRD 라인의 [EQUIP] LL 관련해서 점검 이력을 정리해줄 수 있을까?
- **Devices**: [(none)]
- **Scope**: implicit | **Intent**: information_lookup
- **ES candidates** (top-18):
  - [ ] `set_up_manual_supra_n` (score=11.552, device=SUPRA N, type=SOP)
    > Confidential I 2) Hand Check | | a. BM 을 Teaching 하기 위해서 Hand Type 을 Right 로 설정. | | |---|---|---| | | **Caution** | | |
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=10.5895, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) TC Wafer 설치 확인 | ![](https://i.imgur.com/1234567.pn
  - [ ] `global_sop_supra_n_series_all_pm_gas_feed_through` (score=10.4275, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS FEED THROUGH Global SOP No: Revision No: 4 Page: 4 / 18 ## 3. 사고 사례 #
  - [ ] `global_sop_supra_xp_all_pm_dual_epd` (score=10.4255, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_DUAL EPD Global SOP No: Revision No: 1 Page: 3/40 ## 3. 사고 사례 ### 1) 화상의 정의 불이
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=10.4176, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 3/124 ## 3. 사고 사례 ### 1) 화상의 정의 불
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=10.1617, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 8 / 105 ## Safety 1)
  - [ ] `40054699` (score=10.0474, device=SUPRA Vplus, type=myservice)
    > -. 6/18 교체 했던 TM Robot Controller로 재교체
-. 교체 후 TM Robot ETC Alarm 발생
-> Controller Cable 체결 상태 양호
-> TM Dnet Cable 체결 상태
  - [ ] `set_up_manual_supra_np` (score=9.9457, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구: 안전모, 안전화) ## 12.5 Device Net Calibration _ PSK Board ### 12.5.1 ATM Pr
  - [ ] `set_up_manual_supra_nm` (score=9.9455, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.4 Device Net Calibration _ PSK Board ### 12.4.1 ATM P
  - [ ] `set_up_manual_supra_vm` (score=9.7661, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 8. EFEM Robot Teachin_Cooling Stage 1_Dual | Picture | Description | Tool & Spec | | :--- | :--- | :--- | 
  - [ ] `set_up_manual_ecolite_3000` (score=9.7162, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 7. Teaching_Cooling Stage_Dual | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 11) EFEM R
  - [ ] `set_up_manual_precia` (score=9.5888, device=PRECIA, type=set_up_manual)
    > # 공통사항 1. 보호구를 사용하지 않아도 근로자가 유해/위험작업으로부터 보호를 받을 수 있도록 설비 개선 등 필요 조치를 진행한다. 2. 필요 조치를 이행하였음에도 유해 / 위험 요인은 제거하기가 어려울 때. 제한
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=9.4819, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_EFEM_ROBOT TEACHING Global SOP No: Revision No: 3 Page: 40 / 126 | Flow | Proc
  - [ ] `global_sop_supra_xp_all_ll_flow_switch` (score=9.2392, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_LL_FLOW SWITCH Global SOP No: Revision No: 2 Page: 3/30 ## 3. 사고 사례 ### 1) 감전의 정의
  - [ ] `global_sop_integer_plus_all_ll_disarray_sensor` (score=9.2289, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_LL_DISARRAY SENSOR Global_SOP No: Revision No: 1 Page: 3 / 18 ## 3. 사고 사례 ### 
  - [ ] `global_sop_supra_xp_all_ll_pressure_gauge` (score=9.2172, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_LL_PRESSURE GAUGE Global SOP No: Revision No: 0 Page: 3/28 ## 3. 사고 사례 ### 3-1 감전
  - [ ] `global_sop_supra_n_series_all_pm_epd` (score=9.2163, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_EPD Global SOP No : Revision No: 3 Page: 3/49 ## 3. 사고 사례 ### 1) 화상 재해의 정
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=9.2066, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier | Global SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Pag

#### q_id: `A-q004_masked`
- **Question**: mySERVICE 이력 중 SEC SRD 라인의 [EQUIP] LL 점검 이력을 찾을 수 있을까?
- **Devices**: [(none)]
- **Scope**: implicit | **Intent**: information_lookup
- **ES candidates** (top-21):
  - [ ] `global_sop_supra_n_series_all_pm_epd` (score=8.0456, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_EPD Global SOP No : Revision No: 3 Page: 3/49 ## 3. 사고 사례 ### 1) 화상 재해의 정
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=8.0414, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 16 / 105 ## 사고 사례 ##
  - [ ] `global_sop_supra_n_series_all_pm_hook_lifter_servo_motor` (score=7.8941, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_HOOK LIFTER ## SERVO MOTOR Global SOP No : Revision No: 2 Page : 3 / 106 
  - [ ] `global_sop_geneva_xp_rep_pm_harmonic_drive` (score=7.8644, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Harmonic drive | Global SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Pag
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=7.8527, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier | Global SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Pag
  - [ ] `global_sop_geneva_xp_rep_pm_adapter_ring` (score=7.8464, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Adapter ring Global SOP No: 0 Revision No: 0 Page: 4 / 30 ## 3. 사고 사례 ### 1) 화
  - [ ] `set_up_manual_supra_n` (score=7.8442, device=SUPRA N, type=SOP)
    > ```markdown Confidential I | 7) Read Temp Calibration | a. Channel 1,2 선택 후, Up/Down Click 시 CH1,2 Limit Controller PV (
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=7.8431, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 4/47 ## 3. 사고 사례 ### 1) 화상 재해의 
  - [ ] `global_sop_supra_xp_all_pm_dual_epd` (score=7.8407, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_DUAL EPD Global SOP No: Revision No: 1 Page: 3/40 ## 3. 사고 사례 ### 1) 화상의 정의 불이
  - [ ] `global_sop_supra_n_series_all_pm_gas_feed_through` (score=7.8382, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS FEED THROUGH Global SOP No: Revision No: 4 Page: 4 / 18 ## 3. 사고 사례 #
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=7.8369, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 3/124 ## 3. 사고 사례 ### 1) 화상의 정의 불
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=7.7234, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) TC Wafer 설치 확인 | ![](https://i.imgur.com/1234567.pn
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock_o_ring` (score=7.6757, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_LOAD LOCK O-RING | Global SOP No: | S-KG-R019-R0 | | --- | --- | | Revision No
  - [ ] `set_up_manual_supra_nm` (score=7.3523, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.4 Device Net Calibration _ PSK Board ### 12.4.2 Micro
  - [ ] `set_up_manual_supra_np` (score=7.3074, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구: 안전모, 안전화) ## 12.5 Device Net Calibration _ PSK Board ### 12.5.1 ATM Pr
  - [ ] `global_sop_geneva_xp_rep_pm_support_pin` (score=7.0636, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Support pin | Global SOP No: | 0 | | --- | --- | | Revision No: | 1 | | Page: 
  - [ ] `global_sop_supra_xp_all_tm_mfc` (score=7.0621, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_MFC Global SOP No: Revision No: 0 Page: 3/18 ## 3. 사고 사례 ### 1) 화상 재해의정의 불이나 뜨
  - [ ] `global_sop_geneva_xp_adj_pm_pin_alignment` (score=7.0619, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Pin Alignment | Global SOP No: | S-KG-A003-R0 | | --- | --- | | Revision No: |
  - [ ] `global_sop_supra_n_series_all_pm_top_lid` (score=7.0602, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N Series_ALL_PM_TOP LID Global SOP No: Revision No: 0 Page: 4 / 48 ## 3-2 화상 ### 1) 화상 재해
  - [ ] `global_sop_supra_n_series_all_pm_heater_chuck` (score=7.0588, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_HEATER CHUCK Global SOP No : Revision No: 2 Page: 3/40 ## 3. 사고 사례 ### 1)
  - [ ] `global_sop_supra_xp_all_pm_pendulum_valve` (score=7.0561, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PENDULUM VALVE Global SOP No : Revision No: 6 Page: 3/26 ## 3. 사고 사례 ### 1) 화상

#### q_id: `A-q005_masked`
- **Question**: SEC SRD 라인의 [EQUIP] LL 관련해서 MYSERVICE 점검 이력을 정리해줄 수 있을까?
- **Devices**: [(none)]
- **Scope**: implicit | **Intent**: information_lookup
- **ES candidates** (top-18):
  - [ ] `set_up_manual_supra_n` (score=11.5512, device=SUPRA N, type=SOP)
    > Confidential I 2) Hand Check | | a. BM 을 Teaching 하기 위해서 Hand Type 을 Right 로 설정. | | |---|---|---| | | **Caution** | | |
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=10.5679, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) TC Wafer 설치 확인 | ![](https://i.imgur.com/1234567.pn
  - [ ] `global_sop_supra_n_series_all_pm_gas_feed_through` (score=10.4033, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS FEED THROUGH Global SOP No: Revision No: 4 Page: 4 / 18 ## 3. 사고 사례 #
  - [ ] `global_sop_supra_xp_all_pm_dual_epd` (score=10.4025, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_DUAL EPD Global SOP No: Revision No: 1 Page: 3/40 ## 3. 사고 사례 ### 1) 화상의 정의 불이
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=10.3967, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 3/124 ## 3. 사고 사례 ### 1) 화상의 정의 불
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=10.1418, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 8 / 105 ## Safety 1)
  - [ ] `40054699` (score=10.0549, device=SUPRA Vplus, type=myservice)
    > -. 6/18 교체 했던 TM Robot Controller로 재교체
-. 교체 후 TM Robot ETC Alarm 발생
-> Controller Cable 체결 상태 양호
-> TM Dnet Cable 체결 상태
  - [ ] `set_up_manual_supra_nm` (score=9.9304, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.4 Device Net Calibration _ PSK Board ### 12.4.1 ATM P
  - [ ] `set_up_manual_supra_np` (score=9.93, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구: 안전모, 안전화) ## 12.5 Device Net Calibration _ PSK Board ### 12.5.1 ATM Pr
  - [ ] `set_up_manual_supra_vm` (score=9.7456, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 8. EFEM Robot Teachin_Cooling Stage 1_Dual | Picture | Description | Tool & Spec | | :--- | :--- | :--- | 
  - [ ] `set_up_manual_ecolite_3000` (score=9.7047, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 7. Teaching_Cooling Stage_Dual | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 11) EFEM R
  - [ ] `set_up_manual_precia` (score=9.566, device=PRECIA, type=set_up_manual)
    > # 공통사항 1. 보호구를 사용하지 않아도 근로자가 유해/위험작업으로부터 보호를 받을 수 있도록 설비 개선 등 필요 조치를 진행한다. 2. 필요 조치를 이행하였음에도 유해 / 위험 요인은 제거하기가 어려울 때. 제한
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=9.4647, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_EFEM_ROBOT TEACHING Global SOP No: Revision No: 3 Page: 40 / 126 | Flow | Proc
  - [ ] `global_sop_supra_xp_all_ll_flow_switch` (score=9.2254, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_LL_FLOW SWITCH Global SOP No: Revision No: 2 Page: 3/30 ## 3. 사고 사례 ### 1) 감전의 정의
  - [ ] `global_sop_integer_plus_all_ll_disarray_sensor` (score=9.2057, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_LL_DISARRAY SENSOR Global_SOP No: Revision No: 1 Page: 3 / 18 ## 3. 사고 사례 ### 
  - [ ] `global_sop_supra_xp_all_ll_pressure_gauge` (score=9.1954, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_LL_PRESSURE GAUGE Global SOP No: Revision No: 0 Page: 3/28 ## 3. 사고 사례 ### 3-1 감전
  - [ ] `global_sop_supra_n_series_all_pm_epd` (score=9.1928, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_EPD Global SOP No : Revision No: 3 Page: 3/49 ## 3. 사고 사례 ### 1) 화상 재해의 정
  - [ ] `global_sop_geneva_xp_rep_pm_harmonic_drive` (score=9.1871, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Harmonic drive | Global SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Pag

#### q_id: `A-q008_masked`
- **Question**: [DEVICE] APC Pressure Hunting 발생 시 점검해야 할 포인트는 무엇인가?
- **Devices**: [SUPRA]
- **Scope**: implicit | **Intent**: troubleshooting
- **ES candidates** (top-26):
  - [ ] `set_up_manual_supra_nm` (score=8.6364, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.1. Common List ### 12.1.9 Gas Pressure | Picture | De
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=8.4385, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_TM_CONTROLLER ## BATTERY_Replacement Global SOP No : Revision No: 6 Page : 94 / 
  - [ ] `set_up_manual_supra_n` (score=8.4126, device=SUPRA N, type=SOP)
    > ```markdown Confidential I | 2) Gas Pressure Check | a. BKM Recipe 로 Gas Flow 진행 시 GUI 화면에 출력되는 Gas 압력을 확인한다. | | | :---
  - [ ] `global_sop_geneva_xp_rep_pm_pressure_switch` (score=8.2408, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Pressure switch SOP No: 0 Revision No: 0 Page: 11/16 ## 10. Work Procedure | F
  - [ ] `global_sop_geneva_xp_rep_pm_device_net` (score=8.1666, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Device net | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/15
  - [ ] `global_sop_geneva_xp_rep_efem_ffu_controller` (score=8.0213, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_FFU Controller | SOP No: 0 | Revision No: 0 | Page: 3/21 | | --- | --- | ---
  - [ ] `global_sop_geneva_xp_rep_efem_ffu_filter` (score=8.0024, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_FFU Filter | SOP No: S-KG-R041-R0 | | --- | | Revision No: 0 | | Page: 3 / 1
  - [ ] `global_sop_geneva_xp_rep_efem_ffu` (score=7.992, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_FFU | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3 / 14 | 
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pressure_switch` (score=7.9325, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Vacuum Pressure switch SOP No: 0 Revision No: 0 Page: 11 / 18 ## 11. Work Proce
  - [ ] `set_up_manual_supra_np` (score=7.9141, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I 3) Gas Pressure Adjust <!-- Image (63, 87, 360, 300) --> a. Gas Pressure 압력이 고객 사양과 상이할 경우, G
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=7.8478, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_PENDULUM VALVE Global SOP No: Revision No: Page : 90 / 135 | Flow | Procedu
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=7.8259, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page : 39 / 105 ## 6. APC 
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3000qc` (score=7.825, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM 3000,3100QC SOURCE IGNITION WINDOW CHECK Global SOP No: Revision No: 2 P
  - [ ] `global_sop_precia_all_tm_pressure_relief_valve` (score=7.8229, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_TM_PRESSURE RELIEF VALVE Global SOP No: Revision No: 1 Page: 2/17 ## 1. Safety 1) 안전
  - [ ] `global_sop_precia_all_efem_pressure_relief_valve` (score=7.8096, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_EFEM_PRESSURE RELIEF VALVE Global SOP No: Revision No: 1 Page: 2/16 ## 1. Safety 1) 
  - [ ] `40052189` (score=7.8037, device=SUPRA V, type=myservice)
    > -. 공정중 Pressure 1500유지하는 도중 갑자기 2000까지 튀었다가 0으로 떨어졌다가 물결치는 현상 발생 (발생빈도 한시프트당 1~2회 발생)
-. 전일 APC Pendant 물리고 Monitoring 중
  - [ ] `global_sop_supra_xp_all_tm_pressure_gauge` (score=7.7825, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_PRESSURE GAUGE Global SOP No: Revision No: 1 Page: 3/31 ## 3. 사고 사례 ### 3-1 감전
  - [ ] `global_sop_supra_xp_all_ll_pressure_gauge` (score=7.7818, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_LL_PRESSURE GAUGE Global SOP No: Revision No: 0 Page: 3/28 ## 3. 사고 사례 ### 3-1 감전
  - [ ] `global_sop_precia_all_efem_device_net_board` (score=7.7224, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_EFEM_DEVICE NET BOARD Global SOP No: Revision No: 0 Page: 2/17 ## 1. Safety 1) 안전 및 
  - [ ] `40051833` (score=7.7171, device=SUPRA Vplus, type=myservice)
    > -. Log 확인시 Placement Error로 보여짐
-> 싸이맥스 로그 확인 시 S4(Foup 없는상태) -> S6(Present 감지) -> S7(Placement 감지) 순으로 변해야하나 S6에서 계속 바뀌
  - [ ] `set_up_manual_ecolite_ii_400` (score=7.7037, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 19-5. Power & Utility Turn on | Picture | Description | Data | OK | NG | N/A | | :--- | :--- | :--- | :---
  - [ ] `global_sop_geneva_xp_adj_all_sw_install` (score=7.6726, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_AII_SW INSTALL | SOP No: 0 | | | | --- | --- | --- | | Revision No: 0 | | | | Pag
  - [ ] `set_up_manual_ecolite_3000` (score=7.6692, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 7. Teaching_Loadport 3 | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 21) Hand Check | a
  - [ ] `global_sop_geneva_xp_rep_pm_o2_analyzer` (score=7.6591, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_O2 analyzer(Delta F) | SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Page
  - [ ] `global_sop_integer_plus_all_pm_pcw_manual_valve` (score=7.629, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_PCW ## MANUAL VALVE Global_SOP No: Revision No: 1 Page: 3 / 19 # 3. 사고 사례 #
  - [ ] `global_sop_integer_plus_all_am_temp_controller` (score=7.6276, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_AM_TEMP CONTROLLER Global SOP No: Revision No: 1 Page: 3 / 27 ## 3. 사고 사례 ### 

#### q_id: `A-q012_masked`
- **Question**: [EQUIP] Source Unready Alarm이 발생한 이력에 대해 정리해줄 수 있을까?
- **Devices**: [(none)]
- **Scope**: implicit | **Intent**: information_lookup
- **ES candidates** (top-27):
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=11.2442, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) TC Wafer 설치 확인 | ![](https://i.imgur.com/1234567.pn
  - [ ] `global_sop_supra_xp_all_pm_dual_epd` (score=11.08, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_DUAL EPD Global SOP No: Revision No: 1 Page: 3/40 ## 3. 사고 사례 ### 1) 화상의 정의 불이
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=11.0769, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 3/124 ## 3. 사고 사례 ### 1) 화상의 정의 불
  - [ ] `global_sop_precia_all_efem_ctc` (score=11.0393, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_S/W Install_EFEM_CTC Global SOP No: Revision No: 1 Page: 33/51 | Flow | Procedure | Tool
  - [ ] `global_sop_supra_n_series_all_pm_gas_feed_through` (score=11.0266, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS FEED THROUGH Global SOP No: Revision No: 4 Page: 4 / 18 ## 3. 사고 사례 #
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=10.7154, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_EFEM_ROBOT TEACHING Global SOP No: Revision No: 3 Page: 31 / 126 | Flow | Proc
  - [ ] `set_up_manual_integer_plus` (score=10.4475, device=INTEGER plus, type=SOP)
    > ```markdown # 17-5. System Template Drawing | Picture | Description | Data | OK | NG | N/A | | :--- | :--- | :--- | :---
  - [ ] `global_sop_supra_n_series_all_pm_hook_lifter_servo_motor` (score=10.2629, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_HOOK LIFTER ## SERVO MOTOR Global SOP No : Revision No: 2 Page : 4/106 # 
  - [ ] `set_up_manual_supra_n` (score=10.2302, device=SUPRA N, type=SOP)
    > ```markdown # 3. Docking (※환경안전 보호구 : 안전모, 안전화) ## 3.13 Signal Tower 장착 | Picture | Description | Tool & Spec | | :--- |
  - [ ] `set_up_manual_supra_np` (score=10.229, device=SUPRA Np, type=set_up_manual)
    > ```markdown # 3. Docking (※환경안전 보호구: 안전모, 안전화) ## 3.13 Signal Tower 장착 | Picture | Description | Tool & Spec | | :--- | 
  - [ ] `set_up_manual_supra_nm` (score=10.2264, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 3. Docking (※환경안전 보호구 : 안전모, 안전화) ## 3.14 Signal Tower 장착 | Picture | Description | Tool & 
  - [ ] `global_sop_supra_n_series_all_sub_unit_elt_box_assy` (score=10.1894, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_SUB ## UNIT_PDB Global SOP No: Revision No: 0 Page :89 / 95 | Flow | Procedu
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=10.1877, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 8 / 105 ## Safety 1)
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=10.1051, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier | Global SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Pag
  - [ ] `global_sop_geneva_xp_rep_pm_harmonic_drive` (score=10.0989, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Harmonic drive | Global SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Pag
  - [ ] `global_sop_geneva_xp_rep_bubbler_cabinet_formic_detector` (score=10.0893, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_Bubbler ## Cabinet_Formic detector Global SOP No : S-KG-R038-R0 Revision No: 1 Pa
  - [ ] `global_sop_geneva_xp_rep_sub_bubbler_pt_sensor` (score=10.0878, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA_REP_Sub_Bubbler PT sensor | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page
  - [ ] `global_sop_supra_n_series_all_pm_epd` (score=10.0859, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_EPD Global SOP No : Revision No: 3 Page: 3/49 ## 3. 사고 사례 ### 1) 화상 재해의 정
  - [ ] `global_sop_geneva_xp_rep_pm_adapter_ring` (score=10.085, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Adapter ring Global SOP No: 0 Revision No: 0 Page: 4 / 30 ## 3. 사고 사례 ### 1) 화
  - [ ] `global_sop_geneva_xp_rep_bubbler_cabinet_formic_detector_cartridge` (score=10.0784, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_Bubbler ## Cabinet_Formic detector cartridge Global SOP No : S-KG-R039-R0 Revisio
  - [ ] `global_sop_geneva_rep_bubbler_cabinet_safety_valve` (score=10.0766, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA_REP_Bubbler Cabinet_Safety Valve | Global SOP No: | S-KG-R032-R0 | | --- | --- | | Revis
  - [ ] `global_sop_geneva_rep_bubbler_cabinet_vent_valve` (score=10.072, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA_REP_Bubbler Cabinet_Vent Valve | Global SOP No: | S-KG-R029-R0 | | --- | --- | | Revisio
  - [ ] `global_sop_geneva_rep_bubbler_cabinet_fill_valve` (score=10.0673, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA_REP_Bubbler Cabinet_Fill Valve | Global SOP No: | S-KG-R028-R0 | | --- | --- | | Revisio
  - [ ] `global_sop_geneva_xp_rep_bubbler_cabinet_delivery_valve` (score=10.0669, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_Bubbler ## Cabinet_Bubbler Delivery valve Global SOP No: S-KG-R031-R0 Revision No
  - [ ] `global_sop_geneva_xp_rep_bubbler_cabinet_relief_valve` (score=10.0662, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_Bubbler ## Cabinet_Relief valve Global SOP No : S-KG-R033-R0 Revision No : 0 Page
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=10.0651, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 4/47 ## 3. 사고 사례 ### 1) 화상 재해의 
  - [ ] `global_sop_supra_xp_all_pm_prism_source` (score=10.0622, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM SOURCE | Global SOP No: | | | --- | --- | | Revision No: 2 | | | Page: 3

### explicit_equip

#### q_id: `A-eq505`
- **Question**: JWRL05 장비에서 통신 에러가 발생하는 원인은?
- **Devices**: [GENEVA_STP300_XP]
- **Scope**: explicit_equip | **Intent**: troubleshooting
- **ES candidates** (top-27):
  - [ ] `set_up_manual_integer_plus` (score=7.4226, device=INTEGER plus, type=SOP)
    > ```markdown | 5) Pump & AGV Valve Turn on | a. Pump가 Turn on되면 AGV Valve Controller가 켜졌는지 확인한다. | ※ 통신 연결 확인여부는 History 
  - [ ] `global_sop_supra_n_series_all_tm_ffu_mcu` (score=7.2031, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_FFU_MCU | Global SOP No: | | | --- | --- | | Revision No: 1 | | | Page: 3
  - [ ] `global_sop_supra_n_series_all_tm_ffu` (score=7.1974, device=SUPRA N, type=set_up_manual)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_FFU | Global SOP No : | | | --- | --- | | Revision No: 1 | | | Page : 3/3
  - [ ] `global_sop_geneva_xp_rep_pm_insulation_heater` (score=7.1707, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Insulation heater | SOP No: 0 | | | | | :--- | :--- | :--- | :--- | | Revision
  - [ ] `global_sop_geneva_xp_rep_pm_device_net` (score=7.1675, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Device net | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/15
  - [ ] `global_sop_geneva_xp_rep_pm_elbow_heater` (score=7.1639, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Elbow heater | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/
  - [ ] `global_sop_geneva_xp_adj_all_sw_install` (score=7.1556, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_AII_SW INSTALL | SOP No: 0 | | | | --- | --- | --- | | Revision No: 0 | | | | Pag
  - [ ] `global_sop_geneva_xp_rep_pm_o2_analyzer` (score=7.1405, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_O2 analyzer(Delta F) | SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Page
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=7.0808, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_TM_CONTROLLER ## BATTERY_Replacement Global SOP No : Revision No: 6 Page : 94 / 
  - [ ] `global_sop_supra_n_series_all_efem_ffu` (score=7.025, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_EFEM_FFU | Global SOP No: | | | --- | --- | | Revision No: 4 | | | Page: 6/5
  - [ ] `global_sop_supra_xp_all_efem_ffu` (score=7.0158, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_EFEM_FFU | Global SOP No: | | | --- | --- | | Revision No: 1 | | | Page: 5/60 | |
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=6.9051, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 2/15 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_geneva_xp_rep_pm_o2_analyzer_teledyne` (score=6.8575, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_O2 analyzer(Teledyne -> Teledyne) | SOP No: | S-KG-R027-R1 | | --- | --- | | R
  - [ ] `global_sop_supra_xp_all_efem_controller` (score=6.8159, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_EFEM_CONTROLLER REPLACEMENT Global SOP No: Revision No: 1 Page: 11/46 ## 3. 사고 사례
  - [ ] `precia_all_trouble_shooting_guide_pin_motor_abnormal` (score=6.8145, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # PRECIA Trouble Shooting Guide [Pin Motor Abnormal] Confidential II | Alarm Code | LED 점멸 횟수 | Alarm 종류 | 원
  - [ ] `global_sop_geneva_xp_adj_pm_chuck_temp_calibration` (score=6.7348, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Chuck temp calibration Global SOP No: Revision No: 0 Page: 3/21 ## 3. 재해 방지 대책
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=6.7024, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page :101 / 105 ## Safety 
  - [ ] `global_sop_supra_n_series_all_sub_unit_pressure_vacuum_switch` (score=6.578, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_PRESSURE & VACUUM SWITCH Global SOP No : Revision No: 1 Page: 3/28 
  - [ ] `global_sop_geneva_xp_rep_efem_ffu_controller` (score=6.5757, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_FFU Controller | SOP No: 0 | Revision No: 0 | Page: 3/21 | | --- | --- | ---
  - [ ] `global_sop_geneva_xp_rep_efem_ffu_filter` (score=6.5674, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_FFU Filter | SOP No: S-KG-R041-R0 | | --- | | Revision No: 0 | | Page: 3 / 1
  - [ ] `global_sop_precia_all_tm_sensor_board` (score=6.5625, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_TM_SENSOR BOARD Global SOP No: Revision No: 1 Page: 3/18 ## 3. 사고 사례 ### 1) 전기 재해의 정
  - [ ] `global_sop_precia_all_tm_interface_board` (score=6.5605, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_TM_INTERFACE BOARD Global SOP No: Revision No: 0 Page: 3/19 ## 3. 사고 사례 ### 1) 전기 재해
  - [ ] `global_sop_geneva_xp_rep_efem_ffu` (score=6.5576, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_FFU | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3 / 14 | 
  - [ ] `global_sop_integer_plus_all_efem_pio_sensor_board` (score=6.5498, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_EFEM_PIO SENSOR BOARD Global SOP No: Revision No: 1 Page: 3/16 ## 3. 사고 사례 ###
  - [ ] `global_sop_precia_all_efem_sensor_board` (score=6.5464, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_EFEM_SENSOR BOARD Global SOP No: Revision No: 1 Page: 3/17 ## 3. 사고 사례 ### 1) 전기 재해의
  - [ ] `global_sop_precia_all_efem_device_net_board` (score=6.545, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_EFEM_DEVICE NET BOARD Global SOP No: Revision No: 0 Page: 3/17 ## 3. 사고 사례 ### 1) 전기
  - [ ] `global_sop_precia_all_ll_relief_valve` (score=6.5446, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_LL_RELIEF VALVE Global SOP No: Revision No: 1 Page: 3/18 ## 3. 사고 사례 ### 1) 전기 재해의 정

#### q_id: `A-eq511`
- **Question**: SGGVLF0300에서 lot 처리 중 주의사항은?
- **Devices**: [GENEVA_STP300_XP]
- **Scope**: explicit_equip | **Intent**: operation
- **ES candidates** (top-28):
  - [ ] `global_sop_integer_plus_all_pm_source_box_board` (score=7.3781, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_SOURCE BOX BOARD Global_SOP No: Revision No: 1 Page: 2 / 16 ## 1. Safety 1)
  - [ ] `global_sop_integer_plus_all_tm_u_til_online_program` (score=7.2689, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_U-TIL_ONLINE PROGRAM Global_SOP No: Revision No: Page: 2/17 ## 1. Safety 1)
  - [ ] `global_sop_supra_n_series_all_sub_unit_water_shut_off_valve` (score=7.2541, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_WATER # SHUT OFF VALVE Global SOP No : Revision No: 2 Page: 2/16 ##
  - [ ] `global_sop_integer_plus_all_efem_o2_gas_leak_detector` (score=7.1832, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_EFEM_O2 GAS LEAK DETECTOR Global_SOP No: 0 Revision No: 0 Page: 2 / 17 ## 1. S
  - [ ] `global_sop_integer_plus_all_pm_h2_gas_leak_detector` (score=7.1798, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_H2 GAS LEAK DETECTOR Global_SOP No: 0 Revision No: 1 Page: 2 / 17 ## 1. Saf
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=6.9523, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 2/15 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_n_series_all_tm_ctc` (score=6.805, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_SW_TM_CTC PATCH | Global SOP No : | | |---|---| | Revision No: 6 | | | Page : 52
  - [ ] `global_sop_supra_n_series_all_pcw_turn_on` (score=6.7825, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PCW TURN ON Global SOP No: Revision No: 1 Page: 2/31 ## 1. Safety 1) 안전 및 주의
  - [ ] `global_sop_supra_xp_all_tm_dc_power_supply` (score=6.7589, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ ALL_TM_DC POWER SUPPLY Global SOP No: Revision No: 0 Page: 2/18 ## 1. Safety 1) 안전 및
  - [ ] `global_sop_supra_n_series_all_pm_source_box_interface_board` (score=6.7305, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_SOURCE BOX INTERFACE BOARD Global SOP No: Revision No: 4 Page: 2 / 18 ## 
  - [ ] `global_sop_supra_xp_all_ll_pin` (score=6.7057, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_LL_PIN Global SOP No: Revision No : 0 Page: 2/19 ## 1. Safety 1) 안전 및 주의사항 - Cham
  - [ ] `global_sop_supra_n_series_all_tm_fluorescent_lamp` (score=6.6958, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_FLUORESCENT LAMP Global SOP No : Revision No:2 Page: 2/23 ## 1. Safety 1)
  - [ ] `global_sop_geneva_rep_ctc` (score=6.5967, device=GENEVA XP, type=SOP)
    > ```markdown # Global_SOP_GENEVA STP300 XP_REP_CTC | Global SOP No: | | |---|---| | Revision No: 0 | | | Page: 3/18 | | #
  - [ ] `global_sop_supra_n_series_all_sub_unit_manometer` (score=6.5945, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT _MANOMETER Global SOP No: Revision No: 0 Page: 2/32 ## 1. Safety 1)
  - [ ] `global_sop_supra_vplus_all_pm_controller` (score=6.5731, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_SW_PM_Controller PATCH Global SOP No: Revision No: 3 Page: 25 / 42 ## 1. Safety 1) 
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=6.5497, device=SUPRA N, type=SOP)
    > # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 21 / 105 ## Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=6.4163, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_AM_BARATRON GAUGE Global SOP No: Revision No: Page: 7 / 135 ## 1. Safety 1) 안전
  - [ ] `set_up_manual_supra_nm` (score=6.3875, device=SUPRA Nm, type=set_up_manual)
    > Confidential 1 # 0. Safety ## Picture 8) 위험지역 출입 <!-- Image (73, 482, 758, 725) --> - 방문목적 외의 지역은 출입을 금지하며, 중장비 작업지역 구간 
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=6.3705, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 2/124 ## 1. Safety 1) 안전 및 주의사항 -
  - [ ] `global_sop_integer_plus_all_pm_manometer` (score=6.3691, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER Plus_ALL_MANOMETER Global SOP No: Revision No: 1 Page: 2/20 ## 1. Safety 1) 안전 및 주의사항 -
  - [ ] `global_sop_precia_all_pm_manometer` (score=6.3681, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_MANOMETER Global SOP No: Revision No: 0 Page: 2/20 ## 1. Safety 1) 안전 및 주의사항 - 장비가 작
  - [ ] `global_sop_supra_xp_all_pm_manometer` (score=6.3618, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_MANOMETER Global SOP No : Revision No: 1 Page: 2/32 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_integer_plus_all_tm_32_multi_port` (score=6.3466, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM 32 MULTI PORT Global SOP No: Revision No: 0 Page: 2/24 ## 1. Safety 1) 안전 및
  - [ ] `global_sop_supra_xp_all_tm_multi_port` (score=6.3318, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDISU XP_ALL_TM_MULTI PORT Global SOP No: Revision No: 1 Page: 2/24 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_integer_plus_all_efem_eda_controller` (score=6.2695, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_EFEM_EDA # CONTROLLER Global_SOP No: Revision No: 1 Page: 2/22 ## 1. Safety 1)
  - [ ] `global_sop_precia_adj_all_utility_turn_onoff` (score=6.2602, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_ALL_UTILITY TURN ON/OFF Global SOP No : Revision No: 0 Page: 2/44 ## 1. Safety 1) 안전
  - [ ] `global_sop_precia_all_pm_prevent_maintenance` (score=6.2464, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_PREVENT MAINTENANCE Global SOP No: Revision No: 5 Page: 2/108 ## 1. Safety 1) 안전 
  - [ ] `global_sop_precia_all_pm_lift_pin` (score=6.2338, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_LIFT PIN Global SOP No : Revision No : 0 Page : 2 / 57 ## 1. Safety 1) 안전 및 주의사항 

#### q_id: `A-eq515`
- **Question**: FRCRFSG01 장비의 PM chamber 관련 이슈 해결 방법은?
- **Devices**: [GENEVA_STP300_XP]
- **Scope**: explicit_equip | **Intent**: troubleshooting
- **ES candidates** (top-27):
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=5.7282, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `set_up_manual_supra_np` (score=5.6926, device=SUPRA Np, type=set_up_manual)
    > ```markdown # 0. Safety ## Picture ### 3) 설비 안전 - **300mm 라인 물류 자동반송(OHT) 종류별 위험성** | 사전 | | | | | | :--- | :--- | :--- 
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=5.6876, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 2/124 ## 1. Safety 1) 안전 및 주의사항 -
  - [ ] `set_up_manual_supra_n` (score=5.6872, device=SUPRA N, type=SOP)
    > ```markdown # 0. Safety ## Picture ### 3) 설비 안전 - **300mm 라인 물류 자동반송(OHT) 종류별 위험성** | 사진 | | | | | :--- | :--- | :--- | 
  - [ ] `set_up_manual_integer_plus` (score=5.6848, device=INTEGER plus, type=SOP)
    > ```markdown # 0. Safety ## Picture ### 3) 설비 안전 - **300mm 라인 물류 자동반송(OHT) 종류별 위험성** | 사진 | | | | | :--- | :--- | :--- | 
  - [ ] `set_up_manual_supra_nm` (score=5.6847, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential 1 # 0. Safety ## Picture ### 3) 설비 안전 - **300mm 라인 물류 자동반송(OHT) 종류별 위험성** | 사전 | | | | | :--- |
  - [ ] `global_sop_precia_all_pm_prevent_maintenance` (score=5.5303, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_PREVENT MAINTENANCE Global SOP No: Revision No: 5 Page: 2/108 ## 1. Safety 1) 안전 
  - [ ] `global_sop_precia_all_pm_lift_pin` (score=5.521, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_LIFT PIN Global SOP No : Revision No : 0 Page : 2 / 57 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `set_up_manual_supra_xq` (score=5.5139, device=SUPRA XQ, type=SOP)
    > ```markdown # 0. Safety ## Picture ### 3) 설비 안전 - **300mm 라인 물류 자동반송(OHT) 종류별 위험성** | 사전 | | | | | :--- | :--- | :--- | 
  - [ ] `set_up_manual_ecolite_3000` (score=5.5136, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 0. Safety ## Picture ### 3) 설비 안전 - **300mm 라인 물류 자동반송(OHT) 종류별 위험성** | 사전 | | | | | :--- | :--- | :--- | 
  - [ ] `set_up_manual_ecolite_ii_400` (score=5.4377, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 0. Safety ## Picture ### 3) 설비 안전 - **400mm 라인 물류 자동반송(OHT) 종류별 위험성** | 사전 | OHT | Foup 자장 Stocker | STB |
  - [ ] `global_sop_supra_n_series_all_pm_gas_spring` (score=5.3112, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS SPRING Global SOP No: Revision No: 4 Page: 2/22 ## 1. Safety 1) 안전 및 
  - [ ] `global_sop_precia_all_ll_aligner` (score=5.2151, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_LL_ALIGNER Global SOP No : Revision No: 1 Page: 6/35 ## 5. Worker Location <!-- Imag
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=5.1488, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 71 / 105 ## Safety 1
  - [ ] `set_up_manual_ecolite_2000` (score=5.1222, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 3) 설비 안전 1. 기계적 위험 - 협착, 절단, 말림, 감김 등 2. 전기적 위험 - 감전, 정전기, 아크, 전자기파 3. 열/저온 및 소음 - 고온, 저온, 청각손실 4. 방사선 위험 
  - [ ] `global_sop_integer_plus_all_pm_pin_motor` (score=5.1066, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_LIFTER PIN ASSY Global SOP No: Revision No: 5 Page: 40 / 126 ## 1. Safety 1
  - [ ] `global_sop_integer_plus_all_tm_ctc_controller` (score=5.1053, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_CTC # CONTROLLER Global SOP No: Revision No: 1 Page: 2/51 ## 1. Safety 1) 안
  - [ ] `global_sop_integer_plus_all_pm_gas_spring` (score=5.1017, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_GAS SPRING Global SOP No: 0 Revision No: 1 Page: 2 / 19 ## 1. SAFETY 1) 안전 
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=5.074, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 2/15 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_integer_plus_all_ll_vacuum_line` (score=5.0735, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_LL_VACUUM LINE Global SOP No: Revision No: Page: 2 / 77 ## 1. Safety 1) 안전 및 주
  - [ ] `global_sop_integer_plus_all_tm_mototr_controller` (score=5.0645, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_TM_MOTOR # CONTROLLER Global SOP No: Revision No: Page : 11 / 21 ## 6. Work Pr
  - [ ] `global_sop_integer_plus_all_pm_vacuum_line` (score=5.0512, device=INTEGER XP, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_VACUUM LINE Global SOP No: Revision No: Page: 2 / 133 ## 1. Safety ### 1) 안
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=5.0441, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 3) 설비 안전 1. 기계적 위험 - 협착, 절단, 말림, 감김 등 2. 전기적 위험 - 감전, 정전기, 아크, 전자기파 3. 열/저온 및 소음 - 고온, 저온, 청각손실 4. 방사선 위험 
  - [ ] `global_sop_precia_all_pm_slot_valve` (score=5.014, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_SLOT VALVE Global SOP No : 0 Revision No : 0 Page : 2/36 ## 1. SAFETY 1) 안전 및 주의사
  - [ ] `global_sop_geneva_xp_rep_pm_harmonic_drive` (score=5.0138, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Harmonic drive Global SOP No: 0 Revision No: 0 Page : 27 / 31 | Flow | Procedu
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=5.0034, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier Global SOP No: 0 Revision No: 0 Page: 35 / 43 | Flow | Procedur
  - [ ] `global_sop_integer_plus_all_pm_quick_connector` (score=4.9827, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_ INTEGER plus_ALL_PM_QUICK CONNECTOR Global SOP No: Revision No: 1 Page: 2/17 ## 1. Safety 1) 안

#### q_id: `A-eq520`
- **Question**: NWRL07 장비의 파츠 교체 주기와 방법은?
- **Devices**: [GENEVA_STP300_XP]
- **Scope**: explicit_equip | **Intent**: maintenance
- **ES candidates** (top-21):
  - [ ] `global_sop_integer_plus_all_tm_ctc_controller` (score=6.3445, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_CTC # CONTROLLER Global SOP No: Revision No: 1 Page: 2/51 ## 1. Safety 1) 안
  - [ ] `global_sop_supra_xp_all_efem_controller` (score=5.8767, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_EFEM_CONTROLLER Global SOP No: Revision No: 1 Page: 2/46 ## 1. Safety 1) 안전 및 주의사
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=5.7531, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `global_sop_supra_n_series_all_efem_controller` (score=5.6897, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_EFEM_CONTROLLER Global SOP No : Revision No: 1 Page: 2/40 ## 1. Safety 1) 안전
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=5.6207, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_EFEM_ROBOT TEACHING Global SOP No: Revision No: 3 Page: 43 / 126 | Flow | Proc
  - [ ] `global_sop_integer_plus_all_tm_32_multi_port` (score=5.5848, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM 32 MULTI PORT Global SOP No: Revision No: 0 Page: 2/24 ## 1. Safety 1) 안전 및
  - [ ] `global_sop_supra_xp_all_tm_multi_port` (score=5.5811, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDISU XP_ALL_TM_MULTI PORT Global SOP No: Revision No: 1 Page: 2/24 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_integer_plus_all_am_solenoid_valve` (score=5.5806, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_AM_SOLENOID VALVE Global SOP No: Revision No: 1 Page: 2 / 19 ## 1. Safety 1) 안
  - [ ] `global_sop_supra_vplus_all_pm_controller` (score=5.5388, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_REP_PM_Controller Global SOP No: Revision No: 3 Page: 12/42 ## 8. 필요 Tool | | Name 
  - [ ] `global_sop_supra_series_all_sw_operation` (score=5.5304, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_ SW OPERATION Global SOP No Revision No: 2 Page: 2/49 ## Safety 1) 안전 및 주의사항 -
  - [ ] `global_sop_supra_v_modify_all_air_tube` (score=5.5117, device=SUPRA V, type=SOP)
    > ```markdown # Global SOP_SUPRA V_MODIFY_ALL_AIR TUBE Global SOP No: Revision No: 0 Page: 2/29 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=5.5004, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier Global SOP No: 0 Revision No: 0 Page: 19 / 43 | Flow | Procedur
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=5.4803, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 2/15 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_integer_plus_all_ll_mfc` (score=5.4771, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_LL_MFC Global SOP No: Revision No: 1 Page: 2/20 ## 1. Safety 1) 안전 및 주의사항 - MF
  - [ ] `set_up_manual_ecolite_3000` (score=5.3865, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 3. Docking (EFEM Robot Pick 장착) | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) EFEM R
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=5.3522, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page :101 / 105 ## Safety 
  - [ ] `global_sop_supra_n_series_all_tm_multi_port` (score=5.3512, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_MULTI PORT Global SOP No : Revision No: 1 Page: 2/22 ## 1. Safety 1) 안전 및
  - [ ] `set_up_manual_supra_np` (score=5.2476, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I # 11. TTTM (※환경안전 보호구: 안전모, 안전화) ## 11.1. Common List ### 11.1.2 Temp Limit Controller Settin
  - [ ] `global_sop_supra_n_series_all_tm_pressure_relief_valve` (score=5.2362, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_PRESSURE RELIEF VALVE Global SOP No : Revision No: 2 Page: 2/19 ## 1. Saf
  - [ ] `global_sop_integer_plus_all_pm_pcw_manual_valve` (score=5.2308, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_PCW MANUAL VALVE Global_SOP No: Revision No: 1 Page: 2 / 19 ## 1. Safety 1)
  - [ ] `set_up_manual_supra_nm` (score=5.2186, device=SUPRA Nm, type=set_up_manual)
    > Confidential 1 # 0. Safety ## Picture 8) 위험지역 출입 <!-- Image (73, 482, 758, 725) --> - 방문목적 외의 지역은 출입을 금지하며, 중장비 작업지역 구간 

#### q_id: `A-eq522`
- **Question**: RFL001 장비의 recipe 설정 방법은?
- **Devices**: [GENEVA_STP300]
- **Scope**: explicit_equip | **Intent**: operation
- **ES candidates** (top-26):
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=7.088, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `set_up_manual_ecolite_3000` (score=6.0786, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 14. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Making 
  - [ ] `set_up_manual_supra_n` (score=5.9555, device=SUPRA N, type=SOP)
    > ```markdown Confidential I | 3) Making Recipe | a. 좌측 Operation Menu 하단의 [New]를 선택한다. | | | :--- | :--- | :--- | | | b. 
  - [ ] `global_sop_integer_plus_all_pm_manometer` (score=5.8832, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER Plus_REP_MANOMETER Global SOP No: Revision No: 1 Page: 18 / 20 ## 8. Appendix 계측모드 Mode + Set 3초길게 
  - [ ] `set_up_manual_ecolite_2000` (score=5.854, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 13. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Making 
  - [ ] `set_up_manual_supra_xq` (score=5.8539, device=SUPRA XQ, type=SOP)
    > ```markdown # 13. Process Confirm (※환경안전 보호구: 안전모, 안전화) ## 13-1. Aging Test | Picture | Description | Tool & Spec | | :-
  - [ ] `set_up_manual_supra_vm` (score=5.8503, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 14. Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Making Recipe | a. [Ins
  - [ ] `set_up_manual_ecolite_ii_400` (score=5.8499, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 14. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Making 
  - [ ] `set_up_manual_supra_np` (score=5.7984, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I Analog Output Tap은 조작하지 않는다. # 12. TTTM (※환경안전 보호구: 안전모, 안전화) ## 12.5 Device Net Calibration 
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=5.7079, device=ZEDIUS XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Making Recipe<br><img src="image_url" alt="Image 1">
  - [ ] `global_sop_supra_n_series_all_sub_unit_manometer` (score=5.6543, device=SUPRA N, type=SOP)
    > # Global SOP_SUPRA N series_ADJ_SUB UNIT_MANOMETER Global SOP No: Revision No: 0 Page: 31/32 ## 8. Appendix ### 계측모드 Mod
  - [ ] `global_sop_geneva_xp_rep_pm_mfc` (score=5.469, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_MFC SOP No: 0 Revision No: 0 Page: 14 / 18 ## 10. Work Procedure | Flow | Proc
  - [ ] `global_sop_precia_all_pm_manometer` (score=5.4633, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_MANOMETER Global SOP No: Revision No: 0 Page: 20 / 20 ## 8. Appendix | Flow | Proced
  - [ ] `global_sop_integer_plus_all_pm_pcw_manual_valve` (score=5.4288, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_PCW ## MANUAL VALVE Global_SOP No: Revision No: 1 Page: 11 / 19 # 6. Work P
  - [ ] `global_sop_integer_plus_all_am_temp_controller` (score=5.3457, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_TEMP # CONTROLLER Global SOP No: Revision No: 1 Page: 12 / 27 ## 6. Work Pr
  - [ ] `set_up_manual_supra_nm` (score=5.2675, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.4 Device Net Calibration _ PSK Board ### 12.4.1 ATM P
  - [ ] `global_sop_integer_plus_all_tm_ctc_controller` (score=5.1044, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_CTC # CONTROLLER Global SOP No: Revision No: 1 Page: 2/51 ## 1. Safety 1) 안
  - [ ] `set_up_manual_precia` (score=5.0256, device=PRECIA, type=set_up_manual)
    > ```markdown # 7) 중량물 취급 작업 ## 중량물 취급 시 주의사항 - 발은 어깨 너비로, 허리는 반듯이 세우고 무릎의 힘으로 일어섭니다. <!-- Image (127, 161, 594, 247) --> 
  - [ ] `global_sop_integer_plus_all_tm_32_multi_port` (score=4.9048, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM 32 MULTI PORT Global SOP No: Revision No: 0 Page: 2/24 ## 1. Safety 1) 안전 및
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=4.8995, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 2/15 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_xp_all_tm_multi_port` (score=4.8973, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDISU XP_ALL_TM_MULTI PORT Global SOP No: Revision No: 1 Page: 2/24 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=4.8769, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM ROBOT ASSY REPLACEMENT Global SOP No : Revision No: 6 Page: 49 / 107 ## 8
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=4.8515, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page :101 / 105 ## Safety 
  - [ ] `global_sop_supra_n_series_all_pm_pcw_valve_switch` (score=4.811, device=SUPRA N series, type=SOP)
    > # Global SOP_SUPRA N series_ALL_PM_FLOW ## SWITCH REPLACEMENT Global SOP No: Revision No: 2 Page : 18 / 47 | Flow | Proc
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=4.7927, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_EFEM_ROBOT ## PARAMETER Global SOP No: Revision No: 3 Page: 106 / 126 | Flow |
  - [ ] `global_sop_integer_plus_all_am_solenoid_valve` (score=4.7911, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_SOLENOID VALVE Global SOP No: Revision No: 1 Page: 11 / 19 ## 6. Work Proce

#### q_id: `A-eq524`
- **Question**: SP201 장비의 스펙 한계값은 얼마인가?
- **Devices**: [ECOLITE_II_T]
- **Scope**: explicit_equip | **Intent**: spec_lookup
- **ES candidates** (top-19):
  - [ ] `set_up_manual_integer_plus` (score=6.9445, device=INTEGER plus, type=SOP)
    > ```markdown # 17-21. Load port 인증 | Picture | Description | Data | OK | NG | N/A | | :--- | :--- | :--- | :--- | :--- | 
  - [ ] `global_sop_supra_n_all_pm_fcip_r3` (score=6.6951, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_CLN_PM_FCIP R3 QUARTZ TUBE Global SOP No : Revision No: 3 Page: 83/84 ## 7. 작업 Check Sh
  - [ ] `set_up_manual_supra_n` (score=6.6205, device=SUPRA N, type=SOP)
    > ```markdown Confidential I | 7) Read Temp Calibration | a. Channel 1,2 선택 후, Up/Down Click 시 CH1,2 Limit Controller PV (
  - [ ] `set_up_manual_supra_np` (score=6.5243, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I 8) Set Temp Calibration <!-- Image (60, 70, 290, 266) --> a. Temp Controller 와 Limit Controll
  - [ ] `set_up_manual_precia` (score=6.448, device=PRECIA, type=set_up_manual)
    > | | h. Z-Axis 및 정렬된 상태에서 'TEACH' Key Click i. 'USE ROBOT' Click j. 우측 Teaching Data 값 확인 k. Z-Axis의 경우 Stage1,2를 더한 값에서 
  - [ ] `global_sop_supra_n_series_all_sub_unit_temp_controller` (score=6.2963, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_ TEMP CONTROLLER Global SOP No : Revision No: 2 Page: 21 / 56 | Flo
  - [ ] `global_sop_supra_n_all_sub_unit_mfc` (score=6.0653, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_SUB UNIT_MFC Global SOP No: 0 Revision No: 3 Page: 19 / 21 ## 11. 작업 Check Sheet | 
  - [ ] `set_up_manual_supra_xq` (score=5.9933, device=SUPRA XQ, type=SOP)
    > ```markdown # 15. Set Up Check Sheet (※환경안전 보호구 : 안전모, 안전화) ## 15-3 Docking | Picture | Description | Spec | Check | Res
  - [ ] `set_up_manual_supra_nm` (score=5.8756, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.1. Common List ### 12.1.2 Temp Limit Controller Setti
  - [ ] `global_sop_precia_all_pm_pirani_gauge` (score=5.8108, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_PM_PIRANI GAUGE Global SOP No : 0 Revision No : 0 Page : 28 / 36 | Flow | Procedure 
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=5.7546, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 2/15 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=5.7468, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page :101 / 105 ## Safety 
  - [ ] `global_sop_integer_plus_all_pm_manometer` (score=5.6822, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER Plus_REP_MANOMETER Global SOP No: Revision No: 1 Page: 20 / 20 ## 8. Appendix | Flow | 
  - [ ] `global_sop_integer_plus_all_tm_32_multi_port` (score=5.6707, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM 32 MULTI PORT Global SOP No: Revision No: 0 Page: 2/24 ## 1. Safety 1) 안전 및
  - [ ] `global_sop_supra_xp_all_tm_multi_port` (score=5.6634, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDISU XP_ALL_TM_MULTI PORT Global SOP No: Revision No: 1 Page: 2/24 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=5.6316, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_PENDULUM VALVE Global SOP No: Revision No: Page : 94 / 135 ## 11. 작업 Check 
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=5.5642, device=ZEDIUS XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | |---|---|---| | 4) EFEM 측면 EMO | a. PM 방향에 위치한 EFEM Side Door에 위치한 EMO 스위치를 OFF시
  - [ ] `global_sop_supra_n_series_all_sub_unit_manometer` (score=5.5499, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT _MANOMETER Global SOP No: Revision No: 0 Page: 2/32 ## 1. Safety 1)
  - [ ] `global_sop_integer_plus_all_tm_ctc_controller` (score=5.5207, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_CTC # CONTROLLER Global SOP No: Revision No: 1 Page: 2/51 ## 1. Safety 1) 안

#### q_id: `A-eq525`
- **Question**: URFL94 장비에서 발생하는 알람의 원인과 조치 방법은?
- **Devices**: [GENEVA_STP300_XP]
- **Scope**: explicit_equip | **Intent**: troubleshooting
- **ES candidates** (top-26):
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=8.2748, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_TM_CONTROLLER ## BATTERY_Replacement Global SOP No : Revision No: 6 Page : 94 / 
  - [ ] `global_sop_geneva_xp_rep_pm_insulation_heater` (score=8.0667, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Insulation heater | SOP No: 0 | | | | | :--- | :--- | :--- | :--- | | Revision
  - [ ] `global_sop_geneva_xp_rep_pm_elbow_heater` (score=8.0529, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Elbow heater | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/
  - [ ] `global_sop_geneva_xp_rep_pm_device_net` (score=7.9907, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Device net | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/15
  - [ ] `global_sop_geneva_xp_rep_pm_o2_analyzer` (score=7.969, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_O2 analyzer(Delta F) | SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Page
  - [ ] `global_sop_geneva_xp_adj_all_sw_install` (score=7.9544, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_AII_SW INSTALL | SOP No: 0 | | | | --- | --- | --- | | Revision No: 0 | | | | Pag
  - [ ] `global_sop_geneva_xp_rep_efem_ffu_filter` (score=7.7569, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_FFU Filter | SOP No: S-KG-R041-R0 | | --- | | Revision No: 0 | | Page: 3 / 1
  - [ ] `global_sop_geneva_xp_rep_efem_ffu_controller` (score=7.7454, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_FFU Controller | SOP No: 0 | Revision No: 0 | Page: 3/21 | | --- | --- | ---
  - [ ] `global_sop_geneva_xp_rep_efem_ffu` (score=7.7433, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_FFU | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3 / 14 | 
  - [ ] `global_sop_geneva_xp_rep_pm_o2_analyzer_teledyne` (score=7.6455, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_O2 analyzer(Teledyne -> Teledyne) | SOP No: | S-KG-R027-R1 | | --- | --- | | R
  - [ ] `set_up_manual_precia` (score=7.4593, device=PRECIA, type=set_up_manual)
    > | | h. Z-Axis 및 정렬된 상태에서 'TEACH' Key Click i. 'USE ROBOT' Click j. 우측 Teaching Data 값 확인 k. Z-Axis의 경우 Stage1,2를 더한 값에서 
  - [ ] `global_sop_supra_n_series_all_tm_ffu` (score=7.4254, device=SUPRA N, type=set_up_manual)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_FFU | Global SOP No : | | | --- | --- | | Revision No: 1 | | | Page : 3/3
  - [ ] `global_sop_supra_n_series_all_tm_ffu_mcu` (score=7.4232, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_FFU_MCU | Global SOP No: | | | --- | --- | | Revision No: 1 | | | Page: 3
  - [ ] `global_sop_geneva_xp_adj_pm_chuck_temp_calibration` (score=7.3956, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Chuck temp calibration Global SOP No: Revision No: 0 Page: 3/21 ## 3. 재해 방지 대책
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=7.3208, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 2/15 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=7.3098, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 29 / 105 ## 사고 사례 ##
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3000qc` (score=7.3016, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM SOURCE 3000QC Global SOP No: Revision No : 2 Page : 5/82 ## 3. 사고 사례 ###
  - [ ] `global_sop_supra_xp_all_pm_cip_chamber` (score=7.3014, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_CIP CHAMBER | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page: 5 
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=7.3007, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No: 13 Page: 5/75 ## 3. 사고 사례 #
  - [ ] `global_sop_supra_xp_all_pm_prism_source` (score=7.2964, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM SOURCE | Global SOP No: | | | --- | --- | | Revision No: 2 | | | Page: 5
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3100qc` (score=7.2922, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM SOURCE 3100QC Global SOP No: Revision No: 1 Page: 5/72 ## 3. 사고 사례 ### 1
  - [ ] `global_sop_supra_n_series_all_efem_ffu` (score=7.28, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_EFEM_FFU | Global SOP No: | | | --- | --- | | Revision No: 4 | | | Page: 6/5
  - [ ] `global_sop_supra_xp_all_efem_ffu` (score=7.2795, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_EFEM_FFU | Global SOP No: | | | --- | --- | | Revision No: 1 | | | Page: 5/60 | |
  - [ ] `precia_all_trouble_shooting_guide_pin_motor_abnormal` (score=7.1951, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # PRECIA Trouble Shooting Guide [Pin Motor Abnormal] Confidential II | Alarm Code | LED 점멸 횟수 | Alarm 종류 | 원
  - [ ] `set_up_manual_supra_np` (score=7.1824, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I # 11. TTTM (※환경안전 보호구: 안전모, 안전화) ## 11.1. Common List ### 11.1.2 Temp Limit Controller Settin
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=7.1759, device=ZEDIUS XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) Seasoning | a. [START]를 누른다. | | | | | | | 14) Seas

#### q_id: `A-eq531`
- **Question**: LWRFL07의 chamber cleaning 절차를 알려줘
- **Devices**: [GENEVA_XP]
- **Scope**: explicit_equip | **Intent**: maintenance
- **ES candidates** (top-11):
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=8.644, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 26 / 47 ## 10. Work Procedure |
  - [ ] `global_sop_geneva_xp_rep_pm_heater_chuck_without_jig` (score=8.6001, device=GENEVA XP, type=SOP)
    > # GENEVA xp_REP_PM_Heater chuck w/o jig SOP No: 0 Revision No: 1 Page: 36 / 52 ## 10. Work Procedure | Flow | Procedure 
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=8.4992, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 5. Install Accessory ## 5.5 Install APC | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1
  - [ ] `set_up_manual_ecolite_3000` (score=7.8706, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 7. Teaching_Process Module 1 | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 21) TM Robot
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=7.6668, device=ZEDIUS XP, type=set_up_manual)
    > | | | | |---|---|---| | 16) Teaching Data Save | a. Servo Power Switch와 Enter Key 를 동시에 눌러 Data를 옮겨준다. | | | | | | | 17)
  - [ ] `global_sop_geneva_xp_rep_pm_adapter_ring` (score=7.629, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Adapter ring Global SOP No: 0 Revision No: 0 Page: 11 / 30 ## 10. Work Procedu
  - [ ] `global_sop_geneva_xp_rep_pm_harmonic_drive` (score=7.6241, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Harmonic drive Global SOP No: 0 Revision No: 0 Page: 11 / 31 | Flow | Procedur
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=7.6201, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier Global SOP No: 0 Revision No: 0 Page: 11 / 43 | Flow | Procedur
  - [ ] `set_up_manual_ecolite_2000` (score=7.6092, device=ECOLITE 2000, type=set_up_manual)
    > | | b. 4Leveler를 Stage에 올려 놓는다<br>c. 3mm Wrench로 5Level Screw를<br>시계 방향 또는 반 시계 방향으로<br>돌려가며 Leveler의 물방울이 Spec<br>Zone에
  - [ ] `set_up_manual_supra_vm` (score=7.4634, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 9. TM Robot Teaching_Process Module 1 | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 21)
  - [ ] `global_sop_geneva_xp_rep_pm_disc` (score=7.1892, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc SOP No: 0 Revision No: 0 Page: 26 / 31 ## 10. Work Procedure | Flow | Pro

#### q_id: `A-eq533`
- **Question**: WLR 201에서 센서 캘리브레이션 절차는?
- **Devices**: [GENEVA_STP300_XP]
- **Scope**: explicit_equip | **Intent**: maintenance
- **ES candidates** (top-23):
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=7.3622, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 23/25 ## 10. Trouble Sh
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=5.4738, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 2/15 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=5.2605, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page :101 / 105 ## Safety 
  - [ ] `set_up_manual_supra_vm` (score=5.2525, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 9. TM Robot Teaching_Buffer Stage | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 23) Sem
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=5.1286, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 2/124 ## 1. Safety 1) 안전 및 주의사항 -
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=5.1252, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) TC Wafer 설치 확인 | ![](https://i.imgur.com/1234567.pn
  - [ ] `global_sop_supra_n_series_all_pm_pcw_valve_switch` (score=5.0526, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_PCW VALVE & SWITCH Global SOP No: Revision No: 2 Page: 4/47 ## 3. 사고 사례 #
  - [ ] `global_sop_precia_all_efem_side_storage` (score=5.0126, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_EFEM_SIDE # STORAGE_YEST SOP No : Revision No : 0 Page : 2/37 ## 1. Safety 1) 안전 및 주
  - [ ] `global_sop_precia_all_pm_mfc` (score=5.0082, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_MFC Global SOP No : Revision No: 1 Page: 2/23 ## 1. Safety 1) 안전 및 주의사항 - Main Ga
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=4.9599, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_TM_CONTROLLER ## BATTERY_Replacement Global SOP No : Revision No: 6 Page : 94 / 
  - [ ] `global_sop_supra_n_series_all_pm_gas_spring` (score=4.9534, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS SPRING Global SOP No: Revision No: 4 Page: 3/22 ## 3. 사고 사례 ### 1) 화상
  - [ ] `set_up_manual_supra_n` (score=4.916, device=SUPRA N, type=SOP)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.1. Common List ### 12.1.7 Pumping & Venting Time Adju
  - [ ] `global_sop_integer_plus_all_pm_igs_block` (score=4.9057, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_IGS BLOCK Global SOP No : Revision No: 0 Page: 2/26 ## 1. Safety ### 1) 안전 
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=4.8774, device=ZEDIUS XP, type=set_up_manual)
    > ```markdown 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.4 EFEM Robot Storage Teaching | Picture | Description | Tool & Spec 
  - [ ] `global_sop_integer_plus_all_pm_wall_temp_controller` (score=4.8645, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_WALL TEMP CONTROLLER Global SOP No: Revision No: 0 Page: 2/21 ## 1. SAFETY 
  - [ ] `set_up_manual_supra_np` (score=4.8597, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I | 30) Teaching Position Check | a. EFEM Robot Pendent를 조작하여 End Effector가 Wafer Pad위에 올 때까지 조
  - [ ] `global_sop_supra_n_series_all_tm_pressure_relief_valve` (score=4.8259, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_PRESSURE RELIEF VALVE Global SOP No : Revision No: 2 Page: 2/19 ## 1. Saf
  - [ ] `global_sop_supra_xp_all_tm_mfc` (score=4.8232, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_MFC Global SOP No: Revision No: 0 Page: 2 / 18 ## 1. Safety ### 1) 안전 및 주의사항 -
  - [ ] `global_sop_supra_n_series_all_pm_hook_lifter_servo_motor` (score=4.7868, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_HOOK LIFTER ## SERVO MOTOR Global SOP No : Revision No: 2 Page : 4/106 # 
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=4.7618, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No: 13 Page: 61/75 ## Scope 이 G
  - [ ] `set_up_manual_supra_xq` (score=4.759, device=SUPRA XQ, type=SOP)
    > ```markdown # 7. Teaching (※환경안전 보호구: 안전모, 안전화) ## 7-3 EFEM Side Storage Teaching | Picture | Description | Tool & Spec 
  - [ ] `global_sop_supra_n_series_all_pm_top_lid` (score=4.7455, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N Series_ALL_PM_TOP LID Global SOP No: Revision No: 0 Page: 4 / 48 ## 3-2 화상 ### 1) 화상 재해
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=4.7415, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 2/47 ## 1. Safety ### 1) Safety

#### q_id: `A-eq543`
- **Question**: EPRS753의 chamber cleaning 절차를 알려줘
- **Devices**: [TERA_21]
- **Scope**: explicit_equip | **Intent**: maintenance
- **ES candidates** (top-11):
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=8.6429, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 26 / 47 ## 10. Work Procedure |
  - [ ] `global_sop_geneva_xp_rep_pm_heater_chuck_without_jig` (score=8.6005, device=GENEVA XP, type=SOP)
    > # GENEVA xp_REP_PM_Heater chuck w/o jig SOP No: 0 Revision No: 1 Page: 36 / 52 ## 10. Work Procedure | Flow | Procedure 
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=8.4974, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 5. Install Accessory ## 5.5 Install APC | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1
  - [ ] `set_up_manual_ecolite_3000` (score=7.8649, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 7. Teaching_Process Module 1 | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 21) TM Robot
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=7.6565, device=ZEDIUS XP, type=set_up_manual)
    > | | | | |---|---|---| | 16) Teaching Data Save | a. Servo Power Switch와 Enter Key 를 동시에 눌러 Data를 옮겨준다. | | | | | | | 17)
  - [ ] `global_sop_geneva_xp_rep_pm_adapter_ring` (score=7.6437, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Adapter ring Global SOP No: 0 Revision No: 0 Page: 11 / 30 ## 10. Work Procedu
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=7.6286, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier Global SOP No: 0 Revision No: 0 Page: 11 / 43 | Flow | Procedur
  - [ ] `global_sop_geneva_xp_rep_pm_harmonic_drive` (score=7.6273, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Harmonic drive Global SOP No: 0 Revision No: 0 Page: 11 / 31 | Flow | Procedur
  - [ ] `set_up_manual_ecolite_2000` (score=7.6015, device=ECOLITE 2000, type=set_up_manual)
    > | | b. 4Leveler를 Stage에 올려 놓는다<br>c. 3mm Wrench로 5Level Screw를<br>시계 방향 또는 반 시계 방향으로<br>돌려가며 Leveler의 물방울이 Spec<br>Zone에
  - [ ] `set_up_manual_supra_vm` (score=7.4651, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 9. TM Robot Teaching_Process Module 3 | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 21)
  - [ ] `global_sop_geneva_xp_rep_pm_disc` (score=7.1931, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc SOP No: 0 Revision No: 0 Page: 26 / 31 ## 10. Work Procedure | Flow | Pro

#### q_id: `A-eq545`
- **Question**: AEPR03에서 센서 캘리브레이션 절차는?
- **Devices**: [EVOLITE_II]
- **Scope**: explicit_equip | **Intent**: maintenance
- **ES candidates** (top-23):
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=7.3703, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 23/25 ## 10. Trouble Sh
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=5.4604, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 2/15 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=5.2538, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page :101 / 105 ## Safety 
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=5.1272, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 2/124 ## 1. Safety 1) 안전 및 주의사항 -
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=5.1165, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) TC Wafer 설치 확인 | ![](https://i.imgur.com/1234567.pn
  - [ ] `global_sop_supra_n_series_all_pm_pcw_valve_switch` (score=5.0563, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_PCW VALVE & SWITCH Global SOP No: Revision No: 2 Page: 4/47 ## 3. 사고 사례 #
  - [ ] `global_sop_precia_all_efem_side_storage` (score=5.0179, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_EFEM_SIDE # STORAGE_YEST SOP No : Revision No : 0 Page : 2/37 ## 1. Safety 1) 안전 및 주
  - [ ] `global_sop_precia_all_pm_mfc` (score=4.9923, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_MFC Global SOP No : Revision No: 1 Page: 2/23 ## 1. Safety 1) 안전 및 주의사항 - Main Ga
  - [ ] `global_sop_supra_n_series_all_pm_gas_spring` (score=4.9551, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS SPRING Global SOP No: Revision No: 4 Page: 3/22 ## 3. 사고 사례 ### 1) 화상
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=4.9529, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_TM_CONTROLLER ## BATTERY_Replacement Global SOP No : Revision No: 6 Page : 94 / 
  - [ ] `set_up_manual_supra_n` (score=4.9081, device=SUPRA N, type=SOP)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.1. Common List ### 12.1.7 Pumping & Venting Time Adju
  - [ ] `global_sop_integer_plus_all_pm_igs_block` (score=4.9011, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_IGS BLOCK Global SOP No : Revision No: 0 Page: 2/26 ## 1. Safety ### 1) 안전 
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=4.8968, device=ZEDIUS XP, type=set_up_manual)
    > ```markdown 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.4 EFEM Robot Storage Teaching | Picture | Description | Tool & Spec 
  - [ ] `set_up_manual_supra_np` (score=4.8937, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I | 30) Teaching Position Check | a. EFEM Robot Pendent를 조작하여 End Effector가 Wafer Pad위에 올 때까지 조
  - [ ] `global_sop_integer_plus_all_pm_wall_temp_controller` (score=4.8453, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_WALL TEMP CONTROLLER Global SOP No: Revision No: 0 Page: 2/21 ## 1. SAFETY 
  - [ ] `global_sop_supra_n_series_all_tm_pressure_relief_valve` (score=4.8148, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_PRESSURE RELIEF VALVE Global SOP No : Revision No: 2 Page: 2/19 ## 1. Saf
  - [ ] `global_sop_supra_xp_all_tm_mfc` (score=4.7953, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_MFC Global SOP No: Revision No: 0 Page: 2 / 18 ## 1. Safety ### 1) 안전 및 주의사항 -
  - [ ] `global_sop_supra_n_series_all_pm_hook_lifter_servo_motor` (score=4.7918, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_HOOK LIFTER ## SERVO MOTOR Global SOP No : Revision No: 2 Page : 4/106 # 
  - [ ] `set_up_manual_supra_xq` (score=4.7777, device=SUPRA XQ, type=SOP)
    > ```markdown # 7. Teaching (※환경안전 보호구: 안전모, 안전화) ## 7-3 EFEM Side Storage Teaching | Picture | Description | Tool & Spec 
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=4.7592, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No: 13 Page: 61/75 ## Scope 이 G
  - [ ] `global_sop_geneva_xp_adj_pm_pin_alignment` (score=4.7553, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Pin Alignment | Global SOP No: | S-KG-A003-R0 | | --- | --- | | Revision No: |
  - [ ] `global_sop_supra_n_series_all_pm_process_kit` (score=4.7463, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_PROCESS KIT Global SOP No : Revision No: 0 Page: 3/55 ## 3. 사고 사례 ### 1) 
  - [ ] `global_sop_supra_n_series_all_pm_top_lid` (score=4.7449, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N Series_ALL_PM_TOP LID Global SOP No: Revision No: 0 Page: 4 / 48 ## 3-2 화상 ### 1) 화상 재해

#### q_id: `A-eq550`
- **Question**: WRFCW01에서 wafer 이송 중 에러가 발생할 때 점검 사항은?
- **Devices**: [GENEVA_STP300_XP]
- **Scope**: explicit_equip | **Intent**: troubleshooting
- **ES candidates** (top-25):
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=8.9327, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 2/15 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=8.4682, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 27 / 105 ## Safety 1
  - [ ] `global_sop_supra_xp_all_pm_manometer` (score=8.3931, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_MANOMETER Global SOP No : Revision No: 1 Page: 2/32 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_precia_all_pm_manometer` (score=8.3812, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_MANOMETER Global SOP No: Revision No: 0 Page: 2/20 ## 1. Safety 1) 안전 및 주의사항 - 장비가 작
  - [ ] `global_sop_integer_plus_all_pm_manometer` (score=8.3798, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER Plus_ALL_MANOMETER Global SOP No: Revision No: 1 Page: 2/20 ## 1. Safety 1) 안전 및 주의사항 -
  - [ ] `set_up_manual_supra_vm` (score=8.3274, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 8. EFEM Robot Teaching_Aligner Stage_Single | Picture | Description | Tool & Spec | | :--- | :--- | :--- |
  - [ ] `global_sop_precia_all_efem_side_storage` (score=8.3084, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_EFEM_SIDE # STORAGE_YEST SOP No : Revision No : 0 Page : 2/37 ## 1. Safety 1) 안전 및 주
  - [ ] `global_sop_supra_n_series_all_tm_pressure_relief_valve` (score=8.299, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_PRESSURE RELIEF VALVE Global SOP No : Revision No: 2 Page: 2/19 ## 1. Saf
  - [ ] `global_sop_integer_plus_all_efem_eda_controller` (score=8.264, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_EFEM_EDA # CONTROLLER Global_SOP No: Revision No: 1 Page: 2/22 ## 1. Safety 1)
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=7.9359, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) TC Wafer 설치 확인 | ![](https://i.imgur.com/1234567.pn
  - [ ] `set_up_manual_supra_xq` (score=7.9327, device=SUPRA XQ, type=SOP)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 4-10) Robot Speed 변경 | a. Speed Click. | | | | | | | 4-
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=7.9207, device=ZEDIUS XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 10) Robot Speed 변경 | a. Speed Click. | | | | | | | 11) 
  - [ ] `global_sop_supra_vplus_all_pm_controller` (score=7.8781, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_REP_PM_Controller Global SOP No: Revision No: 3 Page: 7/42 ## 1. Safety 1) 안전 및 주의사
  - [ ] `global_sop_integer_plus_all_efem_controller` (score=7.8734, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_EFEM_CONTROLLER Global SOP No : Revision No: 1 Page: 2/28 ## 1. Safety 1) 안전 및
  - [ ] `global_sop_precia_all_tm_pressure_switch` (score=7.8673, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_TM_PRESSURE SWITCH Global SOP No : Revision No: 0 Page: 2/27 ## 1. Safety 1) 안전 및 주의
  - [ ] `global_sop_integer_plus_all_ll_cassette` (score=7.8525, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER PLUS_ALL_LL_CASSETTE Global SOP No : Revision No: 1 Page: 2/86 ## 1. Safety 1) 안전 및 주의사
  - [ ] `global_sop_precia_all_pm_prevent_maintenance` (score=7.7999, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_PREVENT MAINTENANCE Global SOP No: Revision No: 5 Page: 2/108 ## 1. Safety 1) 안전 
  - [ ] `global_sop_precia_all_pm_lift_pin` (score=7.7945, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_LIFT PIN Global SOP No : Revision No : 0 Page : 2 / 57 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=7.7604, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 2/124 ## 1. Safety 1) 안전 및 주의사항 -
  - [ ] `global_sop_integer_plus_all_tm_32_multi_port` (score=7.7412, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM 32 MULTI PORT Global SOP No: Revision No: 0 Page: 2/24 ## 1. Safety 1) 안전 및
  - [ ] `global_sop_supra_xp_all_tm_multi_port` (score=7.7333, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDISU XP_ALL_TM_MULTI PORT Global SOP No: Revision No: 1 Page: 2/24 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_n_series_all_pm_pressure_gauge` (score=7.7232, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_PRESSURE GAUGE Global SOP No: Revision No: 3 Page: 2/46 ## 1. SAFETY 1) 안
  - [ ] `global_sop_supra_n_series_all_sub_unit_solenoide_valve` (score=7.7073, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_SOLENOID VALVE Global SOP No : Revision No: 5 Page: 2/17 ## 1. Safe
  - [ ] `set_up_manual_ecolite_3000` (score=7.6703, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 7. Teaching_Loadport 2 | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 11) Robot Speed 변경
  - [ ] `global_sop_integer_plus_all_am_pin_motor` (score=7.5871, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_AM_PIN MOTOR Global SOP No: Revision No: 4 Page: 2 / 84 ## 1. Safety 1) 안전 및 주

#### q_id: `A-eq555`
- **Question**: NWRL08의 chamber cleaning 절차를 알려줘
- **Devices**: [GENEVA_STP300_XP]
- **Scope**: explicit_equip | **Intent**: maintenance
- **ES candidates** (top-11):
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=8.6542, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 26 / 47 ## 10. Work Procedure |
  - [ ] `global_sop_geneva_xp_rep_pm_heater_chuck_without_jig` (score=8.6054, device=GENEVA XP, type=SOP)
    > # GENEVA xp_REP_PM_Heater chuck w/o jig SOP No: 0 Revision No: 1 Page: 36 / 52 ## 10. Work Procedure | Flow | Procedure 
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=8.5035, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 5. Install Accessory ## 5.5 Install APC | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1
  - [ ] `set_up_manual_ecolite_3000` (score=7.871, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 7. Teaching_Process Module 1 | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 21) TM Robot
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=7.6742, device=ZEDIUS XP, type=set_up_manual)
    > | | | | |---|---|---| | 16) Teaching Data Save | a. Servo Power Switch와 Enter Key 를 동시에 눌러 Data를 옮겨준다. | | | | | | | 17)
  - [ ] `global_sop_geneva_xp_rep_pm_adapter_ring` (score=7.6354, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Adapter ring Global SOP No: 0 Revision No: 0 Page: 11 / 30 ## 10. Work Procedu
  - [ ] `global_sop_geneva_xp_rep_pm_harmonic_drive` (score=7.6263, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Harmonic drive Global SOP No: 0 Revision No: 0 Page: 11 / 31 | Flow | Procedur
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=7.62, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier Global SOP No: 0 Revision No: 0 Page: 11 / 43 | Flow | Procedur
  - [ ] `set_up_manual_ecolite_2000` (score=7.6172, device=ECOLITE 2000, type=set_up_manual)
    > | | b. 4Leveler를 Stage에 올려 놓는다<br>c. 3mm Wrench로 5Level Screw를<br>시계 방향 또는 반 시계 방향으로<br>돌려가며 Leveler의 물방울이 Spec<br>Zone에
  - [ ] `set_up_manual_supra_vm` (score=7.4722, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 9. TM Robot Teaching_Process Module 1 | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 21)
  - [ ] `global_sop_geneva_xp_rep_pm_disc` (score=7.1924, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc SOP No: 0 Revision No: 0 Page: 26 / 31 ## 10. Work Procedure | Flow | Pro

#### q_id: `A-eq559`
- **Question**: LWRL35에서 lot 처리 중 주의사항은?
- **Devices**: [GENEVA_STP300_XP]
- **Scope**: explicit_equip | **Intent**: operation
- **ES candidates** (top-28):
  - [ ] `global_sop_integer_plus_all_pm_source_box_board` (score=7.3582, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_SOURCE BOX BOARD Global_SOP No: Revision No: 1 Page: 2 / 16 ## 1. Safety 1)
  - [ ] `global_sop_integer_plus_all_tm_u_til_online_program` (score=7.2474, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_U-TIL_ONLINE PROGRAM Global_SOP No: Revision No: Page: 2/17 ## 1. Safety 1)
  - [ ] `global_sop_supra_n_series_all_sub_unit_water_shut_off_valve` (score=7.2387, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_WATER # SHUT OFF VALVE Global SOP No : Revision No: 2 Page: 2/16 ##
  - [ ] `global_sop_integer_plus_all_pm_h2_gas_leak_detector` (score=7.157, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_H2 GAS LEAK DETECTOR Global_SOP No: 0 Revision No: 1 Page: 2 / 17 ## 1. Saf
  - [ ] `global_sop_integer_plus_all_efem_o2_gas_leak_detector` (score=7.1508, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_EFEM_O2 GAS LEAK DETECTOR Global_SOP No: 0 Revision No: 0 Page: 2 / 17 ## 1. S
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=6.946, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 2/15 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_n_series_all_tm_ctc` (score=6.802, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_SW_TM_CTC PATCH | Global SOP No : | | |---|---| | Revision No: 6 | | | Page : 52
  - [ ] `global_sop_supra_n_series_all_pcw_turn_on` (score=6.7666, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PCW TURN ON Global SOP No: Revision No: 1 Page: 2/31 ## 1. Safety 1) 안전 및 주의
  - [ ] `global_sop_supra_xp_all_tm_dc_power_supply` (score=6.7234, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ ALL_TM_DC POWER SUPPLY Global SOP No: Revision No: 0 Page: 2/18 ## 1. Safety 1) 안전 및
  - [ ] `global_sop_supra_n_series_all_pm_source_box_interface_board` (score=6.7165, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_SOURCE BOX INTERFACE BOARD Global SOP No: Revision No: 4 Page: 2 / 18 ## 
  - [ ] `global_sop_supra_xp_all_ll_pin` (score=6.6972, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_LL_PIN Global SOP No: Revision No : 0 Page: 2/19 ## 1. Safety 1) 안전 및 주의사항 - Cham
  - [ ] `global_sop_supra_n_series_all_tm_fluorescent_lamp` (score=6.6817, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_FLUORESCENT LAMP Global SOP No : Revision No:2 Page: 2/23 ## 1. Safety 1)
  - [ ] `global_sop_supra_n_series_all_sub_unit_manometer` (score=6.5808, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT _MANOMETER Global SOP No: Revision No: 0 Page: 2/32 ## 1. Safety 1)
  - [ ] `global_sop_geneva_rep_ctc` (score=6.565, device=GENEVA XP, type=SOP)
    > ```markdown # Global_SOP_GENEVA STP300 XP_REP_CTC | Global SOP No: | | |---|---| | Revision No: 0 | | | Page: 3/18 | | #
  - [ ] `global_sop_supra_vplus_all_pm_controller` (score=6.5559, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_SW_PM_Controller PATCH Global SOP No: Revision No: 3 Page: 25 / 42 ## 1. Safety 1) 
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=6.543, device=SUPRA N, type=SOP)
    > # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 21 / 105 ## Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_xp_all_pm_temp_controller` (score=6.5124, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_TEMP # CONTROLLER Global SOP No: Revision No : 0 Page : 2/35 ## 1. Safety 1) 안
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=6.4233, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_AM_BARATRON GAUGE Global SOP No: Revision No: Page: 7 / 135 ## 1. Safety 1) 안전
  - [ ] `global_sop_supra_xp_all_tm_device_net_board` (score=6.3789, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_DEVICE NET BOARD Global SOP No : Revision No: 2 Page: 2/35 ## 1. Safety 1) 안전 
  - [ ] `global_sop_precia_all_efem_rfid` (score=6.3736, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_EFEM_RFID Global SOP No : Revision No: 0 Page: 2/35 ## 1. Safety 1) 안전 및 주의사항 - 작업자간
  - [ ] `set_up_manual_supra_nm` (score=6.3691, device=SUPRA Nm, type=set_up_manual)
    > Confidential 1 # 0. Safety ## Picture 8) 위험지역 출입 <!-- Image (73, 482, 758, 725) --> - 방문목적 외의 지역은 출입을 금지하며, 중장비 작업지역 구간 
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=6.3689, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 2/124 ## 1. Safety 1) 안전 및 주의사항 -
  - [ ] `global_sop_supra_xp_all_pm_manometer` (score=6.3423, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_MANOMETER Global SOP No : Revision No: 1 Page: 2/32 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_integer_plus_all_pm_manometer` (score=6.3415, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER Plus_ALL_MANOMETER Global SOP No: Revision No: 1 Page: 2/20 ## 1. Safety 1) 안전 및 주의사항 -
  - [ ] `global_sop_precia_all_pm_manometer` (score=6.341, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_MANOMETER Global SOP No: Revision No: 0 Page: 2/20 ## 1. Safety 1) 안전 및 주의사항 - 장비가 작
  - [ ] `global_sop_integer_plus_all_tm_32_multi_port` (score=6.3279, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM 32 MULTI PORT Global SOP No: Revision No: 0 Page: 2/24 ## 1. Safety 1) 안전 및
  - [ ] `global_sop_supra_xp_all_tm_multi_port` (score=6.319, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDISU XP_ALL_TM_MULTI PORT Global SOP No: Revision No: 1 Page: 2/24 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_integer_plus_all_efem_eda_controller` (score=6.2439, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_EFEM_EDA # CONTROLLER Global_SOP No: Revision No: 1 Page: 2/22 ## 1. Safety 1)

#### q_id: `A-eq560`
- **Question**: NWRL05 장비의 스펙 한계값은 얼마인가?
- **Devices**: [GENEVA_STP300_XP]
- **Scope**: explicit_equip | **Intent**: spec_lookup
- **ES candidates** (top-20):
  - [ ] `set_up_manual_integer_plus` (score=6.9642, device=INTEGER plus, type=SOP)
    > ```markdown # 17-21. Load port 인증 | Picture | Description | Data | OK | NG | N/A | | :--- | :--- | :--- | :--- | :--- | 
  - [ ] `global_sop_supra_n_all_pm_fcip_r3` (score=6.6978, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_CLN_PM_FCIP R3 QUARTZ TUBE Global SOP No : Revision No: 3 Page: 83/84 ## 7. 작업 Check Sh
  - [ ] `set_up_manual_precia` (score=6.4384, device=PRECIA, type=set_up_manual)
    > | | h. Z-Axis 및 정렬된 상태에서 'TEACH' Key Click i. 'USE ROBOT' Click j. 우측 Teaching Data 값 확인 k. Z-Axis의 경우 Stage1,2를 더한 값에서 
  - [ ] `global_sop_supra_n_all_sub_unit_mfc` (score=6.0578, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_SUB UNIT_MFC Global SOP No: 0 Revision No: 3 Page: 19 / 21 ## 11. 작업 Check Sheet | 
  - [ ] `set_up_manual_supra_xq` (score=5.9777, device=SUPRA XQ, type=SOP)
    > ```markdown # 15. Set Up Check Sheet (※환경안전 보호구 : 안전모, 안전화) ## 15-3 Docking | Picture | Description | Spec | Check | Res
  - [ ] `set_up_manual_supra_n` (score=5.7706, device=SUPRA N, type=SOP)
    > ```markdown Confidential I # 11. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 11.1. Common List ### 11.1.2 Temp Limit Controller Setti
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=5.7557, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 2/15 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=5.7547, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page :101 / 105 ## Safety 
  - [ ] `set_up_manual_ecolite_3000` (score=5.7377, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 15-5. Power & Utility Turn on | Picture | Description | Data | OK | NG | N/A | |---|---|---|---|---|---| |
  - [ ] `set_up_manual_supra_nm` (score=5.7057, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.4 Device Net Calibration _ PSK Board ### 12.4.1 ATM P
  - [ ] `global_sop_integer_plus_all_tm_32_multi_port` (score=5.6643, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM 32 MULTI PORT Global SOP No: Revision No: 0 Page: 2/24 ## 1. Safety 1) 안전 및
  - [ ] `global_sop_supra_xp_all_tm_multi_port` (score=5.6624, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDISU XP_ALL_TM_MULTI PORT Global SOP No: Revision No: 1 Page: 2/24 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_integer_plus_all_pm_manometer` (score=5.6418, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER Plus_REP_MANOMETER Global SOP No: Revision No: 1 Page: 20 / 20 ## 8. Appendix | Flow | 
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=5.6229, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_PENDULUM VALVE Global SOP No: Revision No: Page : 94 / 135 ## 11. 작업 Check 
  - [ ] `set_up_manual_supra_np` (score=5.5824, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I # 11. TTTM (※환경안전 보호구: 안전모, 안전화) ## 11.1. Common List ### 11.1.2 Temp Limit Controller Settin
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=5.57, device=ZEDIUS XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | |---|---|---| | 4) EFEM 측면 EMO | a. PM 방향에 위치한 EFEM Side Door에 위치한 EMO 스위치를 OFF시
  - [ ] `global_sop_supra_n_series_all_sub_unit_manometer` (score=5.5543, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT _MANOMETER Global SOP No: Revision No: 0 Page: 2/32 ## 1. Safety 1)
  - [ ] `global_sop_integer_plus_all_tm_ctc_controller` (score=5.5201, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_CTC # CONTROLLER Global SOP No: Revision No: 1 Page: 2/51 ## 1. Safety 1) 안
  - [ ] `global_sop_integer_plus_all_pm_vacuum_line` (score=5.5071, device=INTEGER XP, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_CLN_PM_VACUUM LINE Global SOP No: Revision No: Page : 93 / 133 | Flow | Procedure 
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=5.4668, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 3) PM EMO Check | a. 각 PM 의 위치한 EMO Switch (3ea) Off 시킨

#### q_id: `A-eq569`
- **Question**: FRP202에서 센서 캘리브레이션 절차는?
- **Devices**: [GENEVA_STP300_XP]
- **Scope**: explicit_equip | **Intent**: maintenance
- **ES candidates** (top-23):
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=7.3623, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 23/25 ## 10. Trouble Sh
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=5.4665, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 2/15 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=5.2584, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page :101 / 105 ## Safety 
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=5.1347, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 2/124 ## 1. Safety 1) 안전 및 주의사항 -
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=5.0981, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) TC Wafer 설치 확인 | ![](https://i.imgur.com/1234567.pn
  - [ ] `global_sop_supra_n_series_all_pm_pcw_valve_switch` (score=5.0636, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_PCW VALVE & SWITCH Global SOP No: Revision No: 2 Page: 4/47 ## 3. 사고 사례 #
  - [ ] `global_sop_precia_all_efem_side_storage` (score=5.0259, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_EFEM_SIDE # STORAGE_YEST SOP No : Revision No : 0 Page : 2/37 ## 1. Safety 1) 안전 및 주
  - [ ] `global_sop_precia_all_pm_mfc` (score=5.0188, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_MFC Global SOP No : Revision No: 1 Page: 2/23 ## 1. Safety 1) 안전 및 주의사항 - Main Ga
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=4.9628, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_TM_CONTROLLER ## BATTERY_Replacement Global SOP No : Revision No: 6 Page : 94 / 
  - [ ] `global_sop_supra_n_series_all_pm_gas_spring` (score=4.9603, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS SPRING Global SOP No: Revision No: 4 Page: 3/22 ## 3. 사고 사례 ### 1) 화상
  - [ ] `set_up_manual_supra_n` (score=4.9067, device=SUPRA N, type=SOP)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.1. Common List ### 12.1.7 Pumping & Venting Time Adju
  - [ ] `global_sop_integer_plus_all_pm_igs_block` (score=4.9006, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_IGS BLOCK Global SOP No : Revision No: 0 Page: 2/26 ## 1. Safety ### 1) 안전 
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=4.8905, device=ZEDIUS XP, type=set_up_manual)
    > ```markdown 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.4 EFEM Robot Storage Teaching | Picture | Description | Tool & Spec 
  - [ ] `set_up_manual_supra_np` (score=4.8781, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I | 30) Teaching Position Check | a. EFEM Robot Pendent를 조작하여 End Effector가 Wafer Pad위에 올 때까지 조
  - [ ] `global_sop_integer_plus_all_pm_wall_temp_controller` (score=4.857, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_WALL TEMP CONTROLLER Global SOP No: Revision No: 0 Page: 2/21 ## 1. SAFETY 
  - [ ] `global_sop_supra_n_series_all_tm_pressure_relief_valve` (score=4.8321, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_PRESSURE RELIEF VALVE Global SOP No : Revision No: 2 Page: 2/19 ## 1. Saf
  - [ ] `global_sop_supra_xp_all_tm_mfc` (score=4.826, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_MFC Global SOP No: Revision No: 0 Page: 2 / 18 ## 1. Safety ### 1) 안전 및 주의사항 -
  - [ ] `global_sop_supra_n_series_all_pm_hook_lifter_servo_motor` (score=4.7898, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_HOOK LIFTER ## SERVO MOTOR Global SOP No : Revision No: 2 Page : 4/106 # 
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=4.7729, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No: 13 Page: 61/75 ## Scope 이 G
  - [ ] `set_up_manual_supra_xq` (score=4.7713, device=SUPRA XQ, type=SOP)
    > ```markdown # 7. Teaching (※환경안전 보호구: 안전모, 안전화) ## 7-3 EFEM Side Storage Teaching | Picture | Description | Tool & Spec 
  - [ ] `global_sop_supra_n_series_all_pm_top_lid` (score=4.7615, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N Series_ALL_PM_TOP LID Global SOP No: Revision No: 0 Page: 4 / 48 ## 3-2 화상 ### 1) 화상 재해
  - [ ] `global_sop_geneva_xp_adj_pm_pin_alignment` (score=4.7558, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Pin Alignment | Global SOP No: | S-KG-A003-R0 | | --- | --- | | Revision No: |
  - [ ] `global_sop_supra_n_series_all_pm_pressure_gauge` (score=4.7554, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_PRESSURE GAUGE Global SOP No: Revision No: 3 Page: 3/46 ## 3. 사고 사례 ### 3

#### q_id: `A-eq570`
- **Question**: BMRFP01 장비의 recipe 설정 방법은?
- **Devices**: [GENEVA_STP300]
- **Scope**: explicit_equip | **Intent**: operation
- **ES candidates** (top-20):
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=7.0944, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `set_up_manual_ecolite_3000` (score=6.0659, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 14. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Making 
  - [ ] `set_up_manual_supra_n` (score=5.9157, device=SUPRA N, type=SOP)
    > ```markdown Confidential I | 3) Making Recipe | a. 좌측 Operation Menu 하단의 [New]를 선택한다. | | | :--- | :--- | :--- | | | b. 
  - [ ] `global_sop_integer_plus_all_pm_manometer` (score=5.8962, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER Plus_REP_MANOMETER Global SOP No: Revision No: 1 Page: 18 / 20 ## 8. Appendix 계측모드 Mode + Set 3초길게 
  - [ ] `set_up_manual_supra_xq` (score=5.8402, device=SUPRA XQ, type=SOP)
    > ```markdown # 13. Process Confirm (※환경안전 보호구: 안전모, 안전화) ## 13-1. Aging Test | Picture | Description | Tool & Spec | | :-
  - [ ] `set_up_manual_ecolite_2000` (score=5.8369, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 13. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Making 
  - [ ] `set_up_manual_ecolite_ii_400` (score=5.8307, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 14. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Making 
  - [ ] `set_up_manual_supra_vm` (score=5.8304, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 14. Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Making Recipe | a. [Ins
  - [ ] `set_up_manual_supra_np` (score=5.819, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I Analog Output Tap은 조작하지 않는다. # 12. TTTM (※환경안전 보호구: 안전모, 안전화) ## 12.5 Device Net Calibration 
  - [ ] `set_up_manual_supra_nm` (score=5.6893, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential 1 # 0. Safety ## Picture ### 3) 설비 안전 - **300mm 라인 물류 자동반송(OHT) 종류별 위험성** | 사전 | | | | | :--- |
  - [ ] `set_up_manual_integer_plus` (score=5.6879, device=INTEGER plus, type=SOP)
    > ```markdown # 0. Safety ## Picture ### 3) 설비 안전 - **300mm 라인 물류 자동반송(OHT) 종류별 위험성** | 사진 | | | | | :--- | :--- | :--- | 
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=5.6774, device=ZEDIUS XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Making Recipe<br><img src="image_url" alt="Image 1">
  - [ ] `global_sop_supra_n_series_all_sub_unit_manometer` (score=5.6582, device=SUPRA N, type=SOP)
    > # Global SOP_SUPRA N series_ADJ_SUB UNIT_MANOMETER Global SOP No: Revision No: 0 Page: 31/32 ## 8. Appendix ### 계측모드 Mod
  - [ ] `global_sop_geneva_xp_rep_pm_mfc` (score=5.4851, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_MFC SOP No: 0 Revision No: 0 Page: 14 / 18 ## 10. Work Procedure | Flow | Proc
  - [ ] `global_sop_precia_all_pm_manometer` (score=5.4691, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_MANOMETER Global SOP No: Revision No: 0 Page: 20 / 20 ## 8. Appendix | Flow | Proced
  - [ ] `global_sop_integer_plus_all_pm_pcw_manual_valve` (score=5.4378, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_PCW ## MANUAL VALVE Global_SOP No: Revision No: 1 Page: 11 / 19 # 6. Work P
  - [ ] `global_sop_integer_plus_all_am_temp_controller` (score=5.3363, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_TEMP # CONTROLLER Global SOP No: Revision No: 1 Page: 12 / 27 ## 6. Work Pr
  - [ ] `global_sop_supra_n_series_all_tm_ffu` (score=5.3302, device=SUPRA N, type=set_up_manual)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_FFU Global SOP No : Revision No: 1 Page: 15/32 | Flow | Procedure | Tool 
  - [ ] `global_sop_supra_xp_all_efem_ffu` (score=5.2556, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_EFEM_FFU # CONTROLLER ADJUSTMENT Global SOP No: Revision No : 1 Page : 30 / 60 | 
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=5.099, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 3) 설비 안전 1. 기계적 위험 - 협착, 절단, 말림, 감김 등 2. 전기적 위험 - 감전, 정전기, 아크, 전자기파 3. 열/저온 및 소음 - 고온, 저온, 청각손실 4. 방사선 위험 

#### q_id: `A-eq571`
- **Question**: DREFW4에서 lot 처리 중 주의사항은?
- **Devices**: [GENEVA_STP300]
- **Scope**: explicit_equip | **Intent**: operation
- **ES candidates** (top-27):
  - [ ] `global_sop_integer_plus_all_pm_source_box_board` (score=7.6862, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_SOURCE BOX BOARD Global_SOP No: Revision No: 1 Page: 2 / 16 ## 1. Safety 1)
  - [ ] `global_sop_integer_plus_all_efem_o2_gas_leak_detector` (score=7.4822, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_EFEM_O2 GAS LEAK DETECTOR Global_SOP No: 0 Revision No: 0 Page: 2 / 17 ## 1. S
  - [ ] `global_sop_integer_plus_all_pm_h2_gas_leak_detector` (score=7.4641, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_H2 GAS LEAK DETECTOR Global_SOP No: 0 Revision No: 1 Page: 2 / 17 ## 1. Saf
  - [ ] `global_sop_integer_plus_all_tm_u_til_online_program` (score=7.2574, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_U-TIL_ONLINE PROGRAM Global_SOP No: Revision No: Page: 2/17 ## 1. Safety 1)
  - [ ] `global_sop_supra_n_series_all_sub_unit_water_shut_off_valve` (score=7.2355, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_WATER # SHUT OFF VALVE Global SOP No : Revision No: 2 Page: 2/16 ##
  - [ ] `global_sop_supra_n_series_all_pm_source_box_interface_board` (score=7.2189, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_SOURCE BOX INTERFACE BOARD Global SOP No: Revision No: 4 Page: 2 / 18 ## 
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=7.2066, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 2/15 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=7.1312, device=SUPRA N, type=SOP)
    > # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 21 / 105 ## Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_xp_all_tm_dc_power_supply` (score=7.0931, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ ALL_TM_DC POWER SUPPLY Global SOP No: Revision No: 0 Page: 2/18 ## 1. Safety 1) 안전 및
  - [ ] `global_sop_supra_n_series_all_tm_fluorescent_lamp` (score=7.0809, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_FLUORESCENT LAMP Global SOP No : Revision No:2 Page: 2/23 ## 1. Safety 1)
  - [ ] `global_sop_supra_n_series_all_sub_unit_manometer` (score=7.0122, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT _MANOMETER Global SOP No: Revision No: 0 Page: 2/32 ## 1. Safety 1)
  - [ ] `global_sop_supra_xp_all_ll_pin` (score=7.0115, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_LL_PIN Global SOP No: Revision No : 0 Page: 2/19 ## 1. Safety 1) 안전 및 주의사항 - Cham
  - [ ] `global_sop_geneva_rep_ctc` (score=7.0114, device=GENEVA XP, type=SOP)
    > ```markdown # Global_SOP_GENEVA STP300 XP_REP_CTC | Global SOP No: | | |---|---| | Revision No: 0 | | | Page: 3/18 | | #
  - [ ] `global_sop_supra_vplus_all_pm_controller` (score=7.0032, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_SW_PM_Controller PATCH Global SOP No: Revision No: 3 Page: 25 / 42 ## 1. Safety 1) 
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=6.8367, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_AM_BARATRON GAUGE Global SOP No: Revision No: Page: 7 / 135 ## 1. Safety 1) 안전
  - [ ] `global_sop_supra_n_series_all_tm_ctc` (score=6.7941, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_SW_TM_CTC PATCH | Global SOP No : | | |---|---| | Revision No: 6 | | | Page : 52
  - [ ] `global_sop_supra_xp_all_pm_manometer` (score=6.766, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_MANOMETER Global SOP No : Revision No: 1 Page: 2/32 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_n_series_all_pcw_turn_on` (score=6.766, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PCW TURN ON Global SOP No: Revision No: 1 Page: 2/31 ## 1. Safety 1) 안전 및 주의
  - [ ] `global_sop_integer_plus_all_pm_manometer` (score=6.7638, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER Plus_ALL_MANOMETER Global SOP No: Revision No: 1 Page: 2/20 ## 1. Safety 1) 안전 및 주의사항 -
  - [ ] `global_sop_precia_all_pm_manometer` (score=6.7619, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_MANOMETER Global SOP No: Revision No: 0 Page: 2/20 ## 1. Safety 1) 안전 및 주의사항 - 장비가 작
  - [ ] `global_sop_supra_n_series_all_pm_gas_feed_through` (score=6.6876, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS # FEED THROUGH Global SOP No: Revision No: 4 Page: 2 / 18 ## 1. Safet
  - [ ] `global_sop_integer_plus_all_efem_eda_controller` (score=6.6713, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_EFEM_EDA # CONTROLLER Global_SOP No: Revision No: 1 Page: 2/22 ## 1. Safety 1)
  - [ ] `global_sop_integer_plus_all_tm_32_multi_port` (score=6.6602, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM 32 MULTI PORT Global SOP No: Revision No: 0 Page: 2/24 ## 1. Safety 1) 안전 및
  - [ ] `global_sop_supra_xp_all_tm_multi_port` (score=6.6559, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDISU XP_ALL_TM_MULTI PORT Global SOP No: Revision No: 1 Page: 2/24 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_precia_adj_all_utility_turn_onoff` (score=6.6515, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_ALL_UTILITY TURN ON/OFF Global SOP No : Revision No: 0 Page: 2/44 ## 1. Safety 1) 안전
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=6.6337, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 2/124 ## 1. Safety 1) 안전 및 주의사항 -
  - [ ] `global_sop_integer_plus_all_ll_slot_valve` (score=6.6235, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_LL_SLOT VALVE Global SOP No : Revision No : 4 Page : 2/50 ## 1. Safety 1) 안전 및

#### q_id: `A-eq573`
- **Question**: W716IR 장비에서 발생하는 알람의 원인과 조치 방법은?
- **Devices**: [GENEVA_STP300_XP]
- **Scope**: explicit_equip | **Intent**: troubleshooting
- **ES candidates** (top-26):
  - [ ] `global_sop_geneva_xp_rep_pm_insulation_heater` (score=8.0412, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Insulation heater | SOP No: 0 | | | | | :--- | :--- | :--- | :--- | | Revision
  - [ ] `global_sop_geneva_xp_rep_pm_elbow_heater` (score=8.0397, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Elbow heater | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/
  - [ ] `global_sop_geneva_xp_rep_pm_device_net` (score=7.9547, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Device net | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/15
  - [ ] `global_sop_geneva_xp_adj_all_sw_install` (score=7.9441, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_AII_SW INSTALL | SOP No: 0 | | | | --- | --- | --- | | Revision No: 0 | | | | Pag
  - [ ] `global_sop_geneva_xp_rep_pm_o2_analyzer` (score=7.9419, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_O2 analyzer(Delta F) | SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Page
  - [ ] `global_sop_geneva_xp_rep_efem_ffu_controller` (score=7.724, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_FFU Controller | SOP No: 0 | Revision No: 0 | Page: 3/21 | | --- | --- | ---
  - [ ] `global_sop_geneva_xp_rep_efem_ffu_filter` (score=7.7168, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_FFU Filter | SOP No: S-KG-R041-R0 | | --- | | Revision No: 0 | | Page: 3 / 1
  - [ ] `global_sop_geneva_xp_rep_efem_ffu` (score=7.6995, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_FFU | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3 / 14 | 
  - [ ] `global_sop_geneva_xp_rep_pm_o2_analyzer_teledyne` (score=7.6211, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_O2 analyzer(Teledyne -> Teledyne) | SOP No: | S-KG-R027-R1 | | --- | --- | | R
  - [ ] `set_up_manual_precia` (score=7.4556, device=PRECIA, type=set_up_manual)
    > | | h. Z-Axis 및 정렬된 상태에서 'TEACH' Key Click i. 'USE ROBOT' Click j. 우측 Teaching Data 값 확인 k. Z-Axis의 경우 Stage1,2를 더한 값에서 
  - [ ] `set_up_manual_supra_np` (score=7.4158, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I Analog Output Tap은 조작하지 않는다. # 12. TTTM (※환경안전 보호구: 안전모, 안전화) ## 12.5 Device Net Calibration 
  - [ ] `global_sop_supra_n_series_all_tm_ffu_mcu` (score=7.4031, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_FFU_MCU | Global SOP No: | | | --- | --- | | Revision No: 1 | | | Page: 3
  - [ ] `global_sop_supra_n_series_all_tm_ffu` (score=7.3941, device=SUPRA N, type=set_up_manual)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_FFU | Global SOP No : | | | --- | --- | | Revision No: 1 | | | Page : 3/3
  - [ ] `global_sop_geneva_xp_adj_pm_chuck_temp_calibration` (score=7.3654, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Chuck temp calibration Global SOP No: Revision No: 0 Page: 3/21 ## 3. 재해 방지 대책
  - [ ] `set_up_manual_supra_n` (score=7.3221, device=SUPRA N, type=SOP)
    > ```markdown Confidential I # 5) Program Description <!-- Image (74, 80, 370, 557) --> a. 상기 화면에서 각 항목에 대한 내용은 다음과 같다 (Ch
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=7.3172, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 2/15 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=7.2952, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 29 / 105 ## 사고 사례 ##
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=7.2887, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No: 13 Page: 5/75 ## 3. 사고 사례 #
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3000qc` (score=7.2793, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM SOURCE 3000QC Global SOP No: Revision No : 2 Page : 5/82 ## 3. 사고 사례 ###
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3100qc` (score=7.275, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM SOURCE 3100QC Global SOP No: Revision No: 1 Page: 5/72 ## 3. 사고 사례 ### 1
  - [ ] `global_sop_supra_xp_all_pm_cip_chamber` (score=7.2693, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_CIP CHAMBER | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page: 5 
  - [ ] `global_sop_supra_xp_all_pm_prism_source` (score=7.2648, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM SOURCE | Global SOP No: | | | --- | --- | | Revision No: 2 | | | Page: 5
  - [ ] `global_sop_supra_n_series_all_efem_ffu` (score=7.2474, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_EFEM_FFU | Global SOP No: | | | --- | --- | | Revision No: 4 | | | Page: 6/5
  - [ ] `global_sop_supra_xp_all_efem_ffu` (score=7.2392, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_EFEM_FFU | Global SOP No: | | | --- | --- | | Revision No: 1 | | | Page: 5/60 | |
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=7.2272, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_ROBOT Global SOP No : Revision No: 6 Page : 9/107 ## 3. 사고 사례 ### 1) 추락재해
  - [ ] `precia_all_trouble_shooting_guide_pin_motor_abnormal` (score=7.2046, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # PRECIA Trouble Shooting Guide [Pin Motor Abnormal] Confidential II | Alarm Code | LED 점멸 횟수 | Alarm 종류 | 원

#### q_id: `A-eq580`
- **Question**: JWRL04 장비의 파츠 교체 주기와 방법은?
- **Devices**: [GENEVA_STP300_XP]
- **Scope**: explicit_equip | **Intent**: maintenance
- **ES candidates** (top-21):
  - [ ] `global_sop_integer_plus_all_tm_ctc_controller` (score=6.3461, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_CTC # CONTROLLER Global SOP No: Revision No: 1 Page: 2/51 ## 1. Safety 1) 안
  - [ ] `global_sop_supra_xp_all_efem_controller` (score=5.8791, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_EFEM_CONTROLLER Global SOP No: Revision No: 1 Page: 2/46 ## 1. Safety 1) 안전 및 주의사
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=5.747, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `global_sop_supra_n_series_all_efem_controller` (score=5.6935, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_EFEM_CONTROLLER Global SOP No : Revision No: 1 Page: 2/40 ## 1. Safety 1) 안전
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=5.6072, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_EFEM_ROBOT TEACHING Global SOP No: Revision No: 3 Page: 43 / 126 | Flow | Proc
  - [ ] `global_sop_integer_plus_all_am_solenoid_valve` (score=5.5924, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_AM_SOLENOID VALVE Global SOP No: Revision No: 1 Page: 2 / 19 ## 1. Safety 1) 안
  - [ ] `global_sop_integer_plus_all_tm_32_multi_port` (score=5.5913, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM 32 MULTI PORT Global SOP No: Revision No: 0 Page: 2/24 ## 1. Safety 1) 안전 및
  - [ ] `global_sop_supra_xp_all_tm_multi_port` (score=5.5839, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDISU XP_ALL_TM_MULTI PORT Global SOP No: Revision No: 1 Page: 2/24 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_vplus_all_pm_controller` (score=5.5395, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_REP_PM_Controller Global SOP No: Revision No: 3 Page: 12/42 ## 8. 필요 Tool | | Name 
  - [ ] `global_sop_supra_series_all_sw_operation` (score=5.5272, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_ SW OPERATION Global SOP No Revision No: 2 Page: 7/49 ## Safety 1) 안전 및 주의사항 -
  - [ ] `global_sop_supra_v_modify_all_air_tube` (score=5.514, device=SUPRA V, type=SOP)
    > ```markdown # Global SOP_SUPRA V_MODIFY_ALL_AIR TUBE Global SOP No: Revision No: 0 Page: 2/29 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=5.4954, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 2/15 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=5.4931, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier Global SOP No: 0 Revision No: 0 Page: 19 / 43 | Flow | Procedur
  - [ ] `global_sop_integer_plus_all_ll_mfc` (score=5.4929, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_LL_MFC Global SOP No: Revision No: 1 Page: 2/20 ## 1. Safety 1) 안전 및 주의사항 - MF
  - [ ] `set_up_manual_ecolite_3000` (score=5.372, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 3. Docking (EFEM Robot Pick 장착) | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) EFEM R
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=5.3693, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page :101 / 105 ## Safety 
  - [ ] `global_sop_supra_n_series_all_tm_multi_port` (score=5.3563, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_MULTI PORT Global SOP No : Revision No: 1 Page: 2/22 ## 1. Safety 1) 안전 및
  - [ ] `set_up_manual_supra_np` (score=5.2613, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I # 11. TTTM (※환경안전 보호구: 안전모, 안전화) ## 11.1. Common List ### 11.1.2 Temp Limit Controller Settin
  - [ ] `global_sop_integer_plus_all_pm_pcw_manual_valve` (score=5.2542, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_PCW MANUAL VALVE Global_SOP No: Revision No: 1 Page: 2 / 19 ## 1. Safety 1)
  - [ ] `global_sop_supra_n_series_all_tm_pressure_relief_valve` (score=5.2532, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_PRESSURE RELIEF VALVE Global SOP No : Revision No: 2 Page: 2/19 ## 1. Saf
  - [ ] `set_up_manual_supra_nm` (score=5.2257, device=SUPRA Nm, type=set_up_manual)
    > Confidential 1 # 0. Safety ## Picture 8) 위험지역 출입 <!-- Image (73, 482, 758, 725) --> - 방문목적 외의 지역은 출입을 금지하며, 중장비 작업지역 구간 

#### q_id: `A-eq583`
- **Question**: DES202에서 lot 처리 중 주의사항은?
- **Devices**: [ECOLITE_XPL]
- **Scope**: explicit_equip | **Intent**: operation
- **ES candidates** (top-28):
  - [ ] `global_sop_integer_plus_all_pm_source_box_board` (score=7.3665, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_SOURCE BOX BOARD Global_SOP No: Revision No: 1 Page: 2 / 16 ## 1. Safety 1)
  - [ ] `global_sop_integer_plus_all_tm_u_til_online_program` (score=7.2547, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_U-TIL_ONLINE PROGRAM Global_SOP No: Revision No: Page: 2/17 ## 1. Safety 1)
  - [ ] `global_sop_supra_n_series_all_sub_unit_water_shut_off_valve` (score=7.2468, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_WATER # SHUT OFF VALVE Global SOP No : Revision No: 2 Page: 2/16 ##
  - [ ] `global_sop_integer_plus_all_efem_o2_gas_leak_detector` (score=7.1844, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_EFEM_O2 GAS LEAK DETECTOR Global_SOP No: 0 Revision No: 0 Page: 2 / 17 ## 1. S
  - [ ] `global_sop_integer_plus_all_pm_h2_gas_leak_detector` (score=7.1787, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_H2 GAS LEAK DETECTOR Global_SOP No: 0 Revision No: 1 Page: 2 / 17 ## 1. Saf
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=6.9361, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 2/15 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_n_series_all_tm_ctc` (score=6.7928, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_SW_TM_CTC PATCH | Global SOP No : | | |---|---| | Revision No: 6 | | | Page : 52
  - [ ] `global_sop_supra_n_series_all_pcw_turn_on` (score=6.7681, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PCW TURN ON Global SOP No: Revision No: 1 Page: 2/31 ## 1. Safety 1) 안전 및 주의
  - [ ] `global_sop_supra_xp_all_tm_dc_power_supply` (score=6.7501, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ ALL_TM_DC POWER SUPPLY Global SOP No: Revision No: 0 Page: 2/18 ## 1. Safety 1) 안전 및
  - [ ] `global_sop_supra_n_series_all_pm_source_box_interface_board` (score=6.7195, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_SOURCE BOX INTERFACE BOARD Global SOP No: Revision No: 4 Page: 2 / 18 ## 
  - [ ] `global_sop_supra_xp_all_ll_pin` (score=6.7008, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_LL_PIN Global SOP No: Revision No : 0 Page: 2/19 ## 1. Safety 1) 안전 및 주의사항 - Cham
  - [ ] `global_sop_supra_n_series_all_tm_fluorescent_lamp` (score=6.6697, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_FLUORESCENT LAMP Global SOP No : Revision No:2 Page: 2/23 ## 1. Safety 1)
  - [ ] `global_sop_supra_n_series_all_sub_unit_manometer` (score=6.5768, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT _MANOMETER Global SOP No: Revision No: 0 Page: 2/32 ## 1. Safety 1)
  - [ ] `global_sop_geneva_rep_ctc` (score=6.5637, device=GENEVA XP, type=SOP)
    > ```markdown # Global_SOP_GENEVA STP300 XP_REP_CTC | Global SOP No: | | |---|---| | Revision No: 0 | | | Page: 3/18 | | #
  - [ ] `global_sop_supra_vplus_all_pm_controller` (score=6.5584, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_SW_PM_Controller PATCH Global SOP No: Revision No: 3 Page: 25 / 42 ## 1. Safety 1) 
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=6.5411, device=SUPRA N, type=SOP)
    > # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 21 / 105 ## Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=6.4001, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_AM_BARATRON GAUGE Global SOP No: Revision No: Page: 7 / 135 ## 1. Safety 1) 안전
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=6.3737, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 2/124 ## 1. Safety 1) 안전 및 주의사항 -
  - [ ] `set_up_manual_supra_nm` (score=6.3582, device=SUPRA Nm, type=set_up_manual)
    > Confidential 1 # 0. Safety ## Picture 8) 위험지역 출입 <!-- Image (73, 482, 758, 725) --> - 방문목적 외의 지역은 출입을 금지하며, 중장비 작업지역 구간 
  - [ ] `global_sop_supra_xp_all_pm_manometer` (score=6.3555, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_MANOMETER Global SOP No : Revision No: 1 Page: 2/32 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_precia_all_pm_manometer` (score=6.3458, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_MANOMETER Global SOP No: Revision No: 0 Page: 2/20 ## 1. Safety 1) 안전 및 주의사항 - 장비가 작
  - [ ] `global_sop_integer_plus_all_pm_manometer` (score=6.3445, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER Plus_ALL_MANOMETER Global SOP No: Revision No: 1 Page: 2/20 ## 1. Safety 1) 안전 및 주의사항 -
  - [ ] `global_sop_integer_plus_all_tm_32_multi_port` (score=6.327, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM 32 MULTI PORT Global SOP No: Revision No: 0 Page: 2/24 ## 1. Safety 1) 안전 및
  - [ ] `global_sop_supra_xp_all_tm_multi_port` (score=6.3243, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDISU XP_ALL_TM_MULTI PORT Global SOP No: Revision No: 1 Page: 2/24 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_integer_plus_all_efem_eda_controller` (score=6.2605, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_EFEM_EDA # CONTROLLER Global_SOP No: Revision No: 1 Page: 2/22 ## 1. Safety 1)
  - [ ] `global_sop_precia_adj_all_utility_turn_onoff` (score=6.2386, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_ALL_UTILITY TURN ON/OFF Global SOP No : Revision No: 0 Page: 2/44 ## 1. Safety 1) 안전
  - [ ] `global_sop_precia_all_pm_prevent_maintenance` (score=6.233, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_PREVENT MAINTENANCE Global SOP No: Revision No: 5 Page: 2/108 ## 1. Safety 1) 안전 
  - [ ] `global_sop_precia_all_pm_lift_pin` (score=6.2122, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_LIFT PIN Global SOP No : Revision No : 0 Page : 2 / 57 ## 1. Safety 1) 안전 및 주의사항 

#### q_id: `A-eq587`
- **Question**: WEPR0801 장비의 PM chamber 관련 이슈 해결 방법은?
- **Devices**: [TERA_21]
- **Scope**: explicit_equip | **Intent**: troubleshooting
- **ES candidates** (top-24):
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=5.7312, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=5.6825, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 2/124 ## 1. Safety 1) 안전 및 주의사항 -
  - [ ] `global_sop_precia_all_pm_prevent_maintenance` (score=5.528, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_PREVENT MAINTENANCE Global SOP No: Revision No: 5 Page: 2/108 ## 1. Safety 1) 안전 
  - [ ] `global_sop_precia_all_pm_lift_pin` (score=5.5221, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_LIFT PIN Global SOP No : Revision No : 0 Page : 2 / 57 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_supra_n_series_all_pm_gas_spring` (score=5.3033, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS SPRING Global SOP No: Revision No: 4 Page: 2/22 ## 1. Safety 1) 안전 및 
  - [ ] `set_up_manual_supra_nm` (score=5.1999, device=SUPRA Nm, type=set_up_manual)
    > Confidential 1 # 0. Safety ## Picture 8) 위험지역 출입 <!-- Image (73, 482, 758, 725) --> - 방문목적 외의 지역은 출입을 금지하며, 중장비 작업지역 구간 
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=5.1479, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 71 / 105 ## Safety 1
  - [ ] `global_sop_integer_plus_all_pm_pin_motor` (score=5.1141, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_LIFTER PIN ASSY Global SOP No: Revision No: 5 Page: 40 / 126 ## 1. Safety 1
  - [ ] `global_sop_integer_plus_all_tm_ctc_controller` (score=5.0987, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_CTC # CONTROLLER Global SOP No: Revision No: 1 Page: 2/51 ## 1. Safety 1) 안
  - [ ] `global_sop_integer_plus_all_pm_gas_spring` (score=5.0971, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_GAS SPRING Global SOP No: 0 Revision No: 1 Page: 2 / 19 ## 1. SAFETY 1) 안전 
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=5.0805, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 2/15 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_integer_plus_all_ll_vacuum_line` (score=5.0762, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_LL_VACUUM LINE Global SOP No: Revision No: Page: 2 / 77 ## 1. Safety 1) 안전 및 주
  - [ ] `global_sop_integer_plus_all_tm_mototr_controller` (score=5.0697, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_TM_MOTOR # CONTROLLER Global SOP No: Revision No: Page : 11 / 21 ## 6. Work Pr
  - [ ] `global_sop_integer_plus_all_pm_vacuum_line` (score=5.0611, device=INTEGER XP, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_VACUUM LINE Global SOP No: Revision No: Page: 2 / 133 ## 1. Safety ### 1) 안
  - [ ] `global_sop_precia_all_pm_slot_valve` (score=5.0177, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_SLOT VALVE Global SOP No : 0 Revision No : 0 Page : 2/36 ## 1. SAFETY 1) 안전 및 주의사
  - [ ] `global_sop_geneva_xp_rep_pm_harmonic_drive` (score=5.0128, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Harmonic drive Global SOP No: 0 Revision No: 0 Page : 27 / 31 | Flow | Procedu
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=5.0048, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier Global SOP No: 0 Revision No: 0 Page: 35 / 43 | Flow | Procedur
  - [ ] `global_sop_integer_plus_all_pm_quick_connector` (score=4.9931, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_ INTEGER plus_ALL_PM_QUICK CONNECTOR Global SOP No: Revision No: 1 Page: 2/17 ## 1. Safety 1) 안
  - [ ] `set_up_manual_precia` (score=4.972, device=PRECIA, type=set_up_manual)
    > ```markdown # 7) 중량물 취급 작업 ## 중량물 취급 시 주의사항 - 발은 어깨 너비로, 허리는 반듯이 세우고 무릎의 힘으로 일어섭니다. <!-- Image (127, 161, 594, 247) --> 
  - [ ] `global_sop_supra_xp_all_pm_heater_chuck` (score=4.9697, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP ZEDIUS XP_ALL_PM_HEATER # CHUCK Global SOP No: 0 Revision No: 2 Page: 21 / 49 | Flow | Procedur
  - [ ] `global_sop_integer_plus_all_tm_purge_line_regulator` (score=4.969, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_PURGE LINE REGULATOR Global SOP No: Revision No: 0 Page: 2/19 ## 1. Safety 
  - [ ] `global_sop_integer_plus_all_tm_coded_sensor` (score=4.9618, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_CODED SENSOR Global SOP No: Revision No: 0 Page: 2 / 17 ## 1. Safety ### 1)
  - [ ] `global_sop_integer_plus_all_pm_cooling_chuck` (score=4.9618, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ REP_PM_COOLING CHUCK Global_SOP No: Revision No: 0 Page: 11/23 ## 6. Work Procedu
  - [ ] `global_sop_supra_n_series_all_pm_pressure_gauge` (score=4.9612, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_PRESSURE GAUGE Global SOP No: Revision No: 3 Page: 2/46 ## 1. SAFETY 1) 안

#### q_id: `A-q003`
- **Question**: SEC SRD 라인의 EPA404 LL 관련해서 점검 이력을 정리해줄 수 있을까?
- **Devices**: [(none)]
- **Scope**: explicit_equip | **Intent**: information_lookup
- **ES candidates** (top-18):
  - [ ] `set_up_manual_supra_n` (score=11.5374, device=SUPRA N, type=SOP)
    > Confidential I 2) Hand Check | | a. BM 을 Teaching 하기 위해서 Hand Type 을 Right 로 설정. | | |---|---|---| | | **Caution** | | |
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=10.5686, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) TC Wafer 설치 확인 | ![](https://i.imgur.com/1234567.pn
  - [ ] `global_sop_supra_n_series_all_pm_gas_feed_through` (score=10.4426, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS FEED THROUGH Global SOP No: Revision No: 4 Page: 4 / 18 ## 3. 사고 사례 #
  - [ ] `global_sop_supra_xp_all_pm_dual_epd` (score=10.4359, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_DUAL EPD Global SOP No: Revision No: 1 Page: 3/40 ## 3. 사고 사례 ### 1) 화상의 정의 불이
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=10.4201, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 3/124 ## 3. 사고 사례 ### 1) 화상의 정의 불
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=10.1753, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 8 / 105 ## Safety 1)
  - [ ] `40054699` (score=10.0227, device=SUPRA Vplus, type=myservice)
    > -. 6/18 교체 했던 TM Robot Controller로 재교체
-. 교체 후 TM Robot ETC Alarm 발생
-> Controller Cable 체결 상태 양호
-> TM Dnet Cable 체결 상태
  - [ ] `set_up_manual_supra_nm` (score=9.9477, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.4 Device Net Calibration _ PSK Board ### 12.4.1 ATM P
  - [ ] `set_up_manual_supra_np` (score=9.9441, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구: 안전모, 안전화) ## 12.5 Device Net Calibration _ PSK Board ### 12.5.1 ATM Pr
  - [ ] `set_up_manual_supra_vm` (score=9.7605, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 8. EFEM Robot Teachin_Cooling Stage 1_Dual | Picture | Description | Tool & Spec | | :--- | :--- | :--- | 
  - [ ] `set_up_manual_ecolite_3000` (score=9.7031, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 7. Teaching_Cooling Stage_Dual | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 11) EFEM R
  - [ ] `set_up_manual_precia` (score=9.594, device=PRECIA, type=set_up_manual)
    > # 공통사항 1. 보호구를 사용하지 않아도 근로자가 유해/위험작업으로부터 보호를 받을 수 있도록 설비 개선 등 필요 조치를 진행한다. 2. 필요 조치를 이행하였음에도 유해 / 위험 요인은 제거하기가 어려울 때. 제한
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=9.4699, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_EFEM_ROBOT TEACHING Global SOP No: Revision No: 3 Page: 40 / 126 | Flow | Proc
  - [ ] `global_sop_supra_n_series_all_pm_epd` (score=9.2348, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_EPD Global SOP No : Revision No: 3 Page: 3/49 ## 3. 사고 사례 ### 1) 화상 재해의 정
  - [ ] `global_sop_integer_plus_all_ll_disarray_sensor` (score=9.2321, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_LL_DISARRAY SENSOR Global_SOP No: Revision No: 1 Page: 3 / 18 ## 3. 사고 사례 ### 
  - [ ] `global_sop_supra_xp_all_ll_flow_switch` (score=9.2289, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_LL_FLOW SWITCH Global SOP No: Revision No: 2 Page: 3/30 ## 3. 사고 사례 ### 1) 감전의 정의
  - [ ] `global_sop_supra_xp_all_ll_pressure_gauge` (score=9.2166, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_LL_PRESSURE GAUGE Global SOP No: Revision No: 0 Page: 3/28 ## 3. 사고 사례 ### 3-1 감전
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=9.211, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier | Global SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Pag

#### q_id: `A-q004`
- **Question**: mySERVICE 이력 중 SEC SRD 라인의 EPA404호기 LL 점검 이력을 찾을 수 있을까?
- **Devices**: [(none)]
- **Scope**: explicit_equip | **Intent**: information_lookup
- **ES candidates** (top-21):
  - [ ] `global_sop_supra_n_series_all_pm_epd` (score=8.0798, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_EPD Global SOP No : Revision No: 3 Page: 3/49 ## 3. 사고 사례 ### 1) 화상 재해의 정
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=8.077, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 16 / 105 ## 사고 사례 ##
  - [ ] `global_sop_supra_n_series_all_pm_hook_lifter_servo_motor` (score=7.9188, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_HOOK LIFTER ## SERVO MOTOR Global SOP No : Revision No: 2 Page : 3 / 106 
  - [ ] `global_sop_geneva_xp_rep_pm_harmonic_drive` (score=7.8901, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Harmonic drive | Global SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Pag
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=7.8712, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier | Global SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Pag
  - [ ] `global_sop_supra_n_series_all_pm_gas_feed_through` (score=7.8698, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS FEED THROUGH Global SOP No: Revision No: 4 Page: 4 / 18 ## 3. 사고 사례 #
  - [ ] `global_sop_supra_xp_all_pm_dual_epd` (score=7.8656, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_DUAL EPD Global SOP No: Revision No: 1 Page: 3/40 ## 3. 사고 사례 ### 1) 화상의 정의 불이
  - [ ] `global_sop_geneva_xp_rep_pm_adapter_ring` (score=7.8647, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Adapter ring Global SOP No: 0 Revision No: 0 Page: 4 / 30 ## 3. 사고 사례 ### 1) 화
  - [ ] `set_up_manual_supra_n` (score=7.8619, device=SUPRA N, type=SOP)
    > ```markdown Confidential I | 7) Read Temp Calibration | a. Channel 1,2 선택 후, Up/Down Click 시 CH1,2 Limit Controller PV (
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=7.8562, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 3/124 ## 3. 사고 사례 ### 1) 화상의 정의 불
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=7.855, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 4/47 ## 3. 사고 사례 ### 1) 화상 재해의 
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=7.7278, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) TC Wafer 설치 확인 | ![](https://i.imgur.com/1234567.pn
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock_o_ring` (score=7.6974, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_LOAD LOCK O-RING | Global SOP No: | S-KG-R019-R0 | | --- | --- | | Revision No
  - [ ] `set_up_manual_supra_nm` (score=7.3662, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.4 Device Net Calibration _ PSK Board ### 12.4.2 Micro
  - [ ] `set_up_manual_supra_np` (score=7.3241, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구: 안전모, 안전화) ## 12.5 Device Net Calibration _ PSK Board ### 12.5.1 ATM Pr
  - [ ] `global_sop_supra_n_series_all_pm_top_lid` (score=7.0912, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N Series_ALL_PM_TOP LID Global SOP No: Revision No: 0 Page: 4 / 48 ## 3-2 화상 ### 1) 화상 재해
  - [ ] `global_sop_supra_n_series_all_pm_heater_chuck` (score=7.0887, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_HEATER CHUCK Global SOP No : Revision No: 2 Page: 3/40 ## 3. 사고 사례 ### 1)
  - [ ] `global_sop_precia_all_pm_pendulum_valve` (score=7.0845, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_PENDULUM VALVE Global SOP No : Revision No: 0 Page: 3/32 ## 3. 사고 사례 ### 1) 화상 재해
  - [ ] `global_sop_supra_n_series_all_pm_process_kit` (score=7.0834, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_PROCESS KIT Global SOP No : Revision No: 0 Page: 3/55 ## 3. 사고 사례 ### 1) 
  - [ ] `global_sop_geneva_xp_adj_pm_pin_alignment` (score=7.0826, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Pin Alignment | Global SOP No: | S-KG-A003-R0 | | --- | --- | | Revision No: |
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=7.0807, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_AM_BARATRON GAUGE Global SOP No: Revision No: Page: 8 / 135 ## 3. 사고 사례 ### 1)

#### q_id: `A-q005`
- **Question**: SEC SRD 라인의 EPA404 LL 관련해서 MYSERVICE 점검 이력을 정리해줄 수 있을까?
- **Devices**: [(none)]
- **Scope**: explicit_equip | **Intent**: information_lookup
- **ES candidates** (top-18):
  - [ ] `set_up_manual_supra_n` (score=11.5316, device=SUPRA N, type=SOP)
    > Confidential I 2) Hand Check | | a. BM 을 Teaching 하기 위해서 Hand Type 을 Right 로 설정. | | |---|---|---| | | **Caution** | | |
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=10.5501, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) TC Wafer 설치 확인 | ![](https://i.imgur.com/1234567.pn
  - [ ] `global_sop_supra_n_series_all_pm_gas_feed_through` (score=10.4214, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS FEED THROUGH Global SOP No: Revision No: 4 Page: 4 / 18 ## 3. 사고 사례 #
  - [ ] `global_sop_supra_xp_all_pm_dual_epd` (score=10.4149, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_DUAL EPD Global SOP No: Revision No: 1 Page: 3/40 ## 3. 사고 사례 ### 1) 화상의 정의 불이
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=10.4027, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 3/124 ## 3. 사고 사례 ### 1) 화상의 정의 불
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=10.1549, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 8 / 105 ## Safety 1)
  - [ ] `40054699` (score=10.0276, device=SUPRA Vplus, type=myservice)
    > -. 6/18 교체 했던 TM Robot Controller로 재교체
-. 교체 후 TM Robot ETC Alarm 발생
-> Controller Cable 체결 상태 양호
-> TM Dnet Cable 체결 상태
  - [ ] `set_up_manual_supra_nm` (score=9.9347, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.4 Device Net Calibration _ PSK Board ### 12.4.1 ATM P
  - [ ] `set_up_manual_supra_np` (score=9.9304, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구: 안전모, 안전화) ## 12.5 Device Net Calibration _ PSK Board ### 12.5.1 ATM Pr
  - [ ] `set_up_manual_supra_vm` (score=9.7387, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 8. EFEM Robot Teachin_Cooling Stage 1_Dual | Picture | Description | Tool & Spec | | :--- | :--- | :--- | 
  - [ ] `set_up_manual_ecolite_3000` (score=9.69, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 7. Teaching_Cooling Stage_Dual | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 11) EFEM R
  - [ ] `set_up_manual_precia` (score=9.5692, device=PRECIA, type=set_up_manual)
    > # 공통사항 1. 보호구를 사용하지 않아도 근로자가 유해/위험작업으로부터 보호를 받을 수 있도록 설비 개선 등 필요 조치를 진행한다. 2. 필요 조치를 이행하였음에도 유해 / 위험 요인은 제거하기가 어려울 때. 제한
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=9.4537, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_EFEM_ROBOT TEACHING Global SOP No: Revision No: 3 Page: 40 / 126 | Flow | Proc
  - [ ] `global_sop_supra_xp_all_ll_flow_switch` (score=9.2163, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_LL_FLOW SWITCH Global SOP No: Revision No: 2 Page: 3/30 ## 3. 사고 사례 ### 1) 감전의 정의
  - [ ] `global_sop_supra_n_series_all_pm_epd` (score=9.2129, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_EPD Global SOP No : Revision No: 3 Page: 3/49 ## 3. 사고 사례 ### 1) 화상 재해의 정
  - [ ] `global_sop_integer_plus_all_ll_disarray_sensor` (score=9.2117, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_LL_DISARRAY SENSOR Global_SOP No: Revision No: 1 Page: 3 / 18 ## 3. 사고 사례 ### 
  - [ ] `global_sop_supra_xp_all_ll_pressure_gauge` (score=9.1971, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_LL_PRESSURE GAUGE Global SOP No: Revision No: 0 Page: 3/28 ## 3. 사고 사례 ### 3-1 감전
  - [ ] `global_sop_geneva_xp_rep_pm_harmonic_drive` (score=9.1967, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Harmonic drive | Global SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Pag

#### q_id: `A-q012`
- **Question**: EPAGQ03에서 Source Unready Alarm이 발생한 이력에 대해 정리해줄 수 있을까?
- **Devices**: [(none)]
- **Scope**: explicit_equip | **Intent**: information_lookup
- **ES candidates** (top-27):
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=12.0117, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) TC Wafer 설치 확인 | ![](https://i.imgur.com/1234567.pn
  - [ ] `global_sop_supra_xp_all_pm_dual_epd` (score=11.3888, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_DUAL EPD Global SOP No: Revision No: 1 Page: 3/40 ## 3. 사고 사례 ### 1) 화상의 정의 불이
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=11.3822, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 3/124 ## 3. 사고 사례 ### 1) 화상의 정의 불
  - [ ] `global_sop_supra_n_series_all_pm_gas_feed_through` (score=11.3461, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS FEED THROUGH Global SOP No: Revision No: 4 Page: 4 / 18 ## 3. 사고 사례 #
  - [ ] `global_sop_precia_all_efem_ctc` (score=11.0744, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_S/W Install_EFEM_CTC Global SOP No: Revision No: 1 Page: 33/51 | Flow | Procedure | Tool
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=10.7457, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_EFEM_ROBOT TEACHING Global SOP No: Revision No: 3 Page: 31 / 126 | Flow | Proc
  - [ ] `set_up_manual_supra_n` (score=10.6756, device=SUPRA N, type=SOP)
    > ```markdown # 3. Docking (※환경안전 보호구 : 안전모, 안전화) ## 3.13 Signal Tower 장착 | Picture | Description | Tool & Spec | | :--- |
  - [ ] `set_up_manual_supra_np` (score=10.675, device=SUPRA Np, type=set_up_manual)
    > ```markdown # 3. Docking (※환경안전 보호구: 안전모, 안전화) ## 3.13 Signal Tower 장착 | Picture | Description | Tool & Spec | | :--- | 
  - [ ] `set_up_manual_supra_nm` (score=10.6716, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 3. Docking (※환경안전 보호구 : 안전모, 안전화) ## 3.14 Signal Tower 장착 | Picture | Description | Tool & 
  - [ ] `global_sop_supra_n_series_all_pm_hook_lifter_servo_motor` (score=10.5945, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_HOOK LIFTER ## SERVO MOTOR Global SOP No : Revision No: 2 Page : 4/106 # 
  - [ ] `set_up_manual_integer_plus` (score=10.4658, device=INTEGER plus, type=SOP)
    > ```markdown # 17-5. System Template Drawing | Picture | Description | Data | OK | NG | N/A | | :--- | :--- | :--- | :---
  - [ ] `global_sop_geneva_xp_rep_sub_bubbler_pt_sensor` (score=10.4326, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA_REP_Sub_Bubbler PT sensor | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page
  - [ ] `global_sop_geneva_xp_rep_bubbler_cabinet_formic_detector` (score=10.4291, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_Bubbler ## Cabinet_Formic detector Global SOP No : S-KG-R038-R0 Revision No: 1 Pa
  - [ ] `global_sop_geneva_rep_bubbler_cabinet_safety_valve` (score=10.4274, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA_REP_Bubbler Cabinet_Safety Valve | Global SOP No: | S-KG-R032-R0 | | --- | --- | | Revis
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=10.427, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier | Global SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Pag
  - [ ] `global_sop_supra_vplus_adj_all_undocking` (score=10.4264, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_UNDOCKING Global SOP No: Revision No: 0 Page: 12 / 19 | Flow | Procedure | 
  - [ ] `global_sop_supra_n_adj_all_undocking` (score=10.4245, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ADJ_ALL_UNDOCKING Global SOP No: Revision No: 3 Page: 12 / 19 | Flow | Procedure | Tool
  - [ ] `global_sop_geneva_xp_rep_pm_harmonic_drive` (score=10.4239, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Harmonic drive | Global SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Pag
  - [ ] `global_sop_geneva_xp_rep_bubbler_cabinet_formic_detector_cartridge` (score=10.4216, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_Bubbler ## Cabinet_Formic detector cartridge Global SOP No : S-KG-R039-R0 Revisio
  - [ ] `global_sop_supra_n_series_all_pm_epd` (score=10.4206, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_EPD Global SOP No : Revision No: 3 Page: 3/49 ## 3. 사고 사례 ### 1) 화상 재해의 정
  - [ ] `global_sop_geneva_rep_bubbler_cabinet_vent_valve` (score=10.4163, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA_REP_Bubbler Cabinet_Vent Valve | Global SOP No: | S-KG-R029-R0 | | --- | --- | | Revisio
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=10.4144, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 16 / 105 ## 사고 사례 ##
  - [ ] `global_sop_geneva_xp_rep_bubbler_cabinet_delivery_valve` (score=10.4116, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_Bubbler ## Cabinet_Bubbler Delivery valve Global SOP No: S-KG-R031-R0 Revision No
  - [ ] `global_sop_geneva_rep_bubbler_cabinet_fill_valve` (score=10.4115, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA_REP_Bubbler Cabinet_Fill Valve | Global SOP No: | S-KG-R028-R0 | | --- | --- | | Revisio
  - [ ] `global_sop_geneva_xp_rep_pm_adapter_ring` (score=10.4089, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Adapter ring Global SOP No: 0 Revision No: 0 Page: 4 / 30 ## 3. 사고 사례 ### 1) 화
  - [ ] `global_sop_geneva_xp_rep_bubbler_cabinet_relief_valve` (score=10.4066, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_Bubbler ## Cabinet_Relief valve Global SOP No : S-KG-R033-R0 Revision No : 0 Page
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=10.3962, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 4/47 ## 3. 사고 사례 ### 1) 화상 재해의 

### ambiguous

#### q_id: `A-amb001`
- **Question**: FFU 교체 작업 시 주의사항과 절차는?
- **Devices**: [GENEVA_XP, INTEGER_PLUS, PRECIA, SUPRA_N, SUPRA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-11):
  - [ ] `global_sop_supra_xp_all_efem_ffu` (score=10.7556, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_EFEM_FFU Global SOP No: Revision No : 1 Page : 2/60 ## 1. Safety 1) 안전 및 주의사항 - F
  - [ ] `global_sop_supra_n_series_all_efem_ffu` (score=9.9902, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_EFEM_FFU | Global SOP No: | | | --- | --- | | Revision No: 4 | | | Page: 2/5
  - [ ] `global_sop_precia_all_efem_ffu` (score=9.7409, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_EFEM_FFU, FFU FILTER Global SOP No : Revision No: 0 Page : 24 / 50 ## Scope 이 Global
  - [ ] `global_sop_integer_plus_all_efem_ffu` (score=9.5882, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_EFEM_FFU MOTER Global SOP No : Revision No: 0 Page: 6/46 ## Scope 이 Global SOP
  - [ ] `global_sop_supra_n_series_all_tm_ffu` (score=9.5045, device=SUPRA N, type=set_up_manual)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_FFU Global SOP No : Revision No: 1 Page: 2/32 ## 1. Safety ### 1) 안전 및 주의
  - [ ] `global_sop_supra_n_series_all_tm_ffu_mcu` (score=9.497, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_FFU_MCU Global SOP No : Revision No: 1 Page: 2/57 ## 1. Safety ### 1) 안전 
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=8.7747, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No: 13 Page: 61/75 ## Scope 이 G
  - [ ] `global_sop_integer_plus_all_ll_mfc` (score=8.6416, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_LL_MFC Global SOP No: Revision No: 1 Page: 6/20 ## Scope 이 Global SOP는 INTEGER
  - [ ] `global_sop_integer_plus_all_ll_sensor_board` (score=8.6098, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_REP_LL_SENSOR BOARD Global SOP No: Revision No: 1 Page: 6 / 17 ## Scope 이 Global SOP는 INTEGER 
  - [ ] `global_sop_integer_plus_all_pm_gas_line` (score=8.5158, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_GAS FILTER Global_SOP No: Revision No: 0 Page: 32 / 75 ## Scope 이 Global SO
  - [ ] `global_sop_integer_plus_all_am_gas_line` (score=8.5067, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_GAS FILTER | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page: 

#### q_id: `A-amb002`
- **Question**: PM Controller 교체 후 초기 설정 절차는?
- **Devices**: [INTEGER_PLUS, PRECIA, SUPRA_N, SUPRA_VPLUS, SUPRA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-22):
  - [ ] `global_sop_supra_nm_all_pm_microwave_local_interface` (score=5.2765, device=SUPRA Nm, type=SOP)
    > ```markdown # Global SOP_SUPRA Nm_ADJ_PM_MICROWAVE_LOCAL INTERFACE Global_SOP No: Revision No: Page :16/21 | Flow | Proc
  - [ ] `global_sop_supra_xp_all_pm_temp_controller` (score=5.2172, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_TEMP # CONTROLLER Global SOP No: Revision No : 0 Page : 2/35 ## 1. Safety 1) 안
  - [ ] `global_sop_integer_plus_all_pm_temp_controller` (score=5.2066, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_REP_PM_TEMP CONTROLLER Global_SOP No: Revision No: 1 Page: 6 / 27 ## Scope 이 Global_SOP는 INTEG
  - [ ] `global_sop_integer_plus_all_pm_pin_motor` (score=5.1883, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_REP_PM_PIN MOTOR CONTROLLER Global SOP No: Revision No: 5 Page: 91 / 126 ## Scope 이 Global SOP
  - [ ] `global_sop_supra_n_series_all_tm_ffu` (score=5.073, device=SUPRA N, type=set_up_manual)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_FFU Global SOP No : Revision No: 1 Page: 15/32 | Flow | Procedure | Tool 
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=5.0472, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 3) Bubbler pressure gauge 위치 | a. Bubbler cabinet 의 Dig
  - [ ] `global_sop_integer_plus_all_pm_wall_temp_controller` (score=5.0118, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_REP_PM_WALL TEMP CONTROLLER Global SOP No: Revision No: 0 Page: 6/21 ## Scope 이 Global SOP는 IN
  - [ ] `global_sop_supra_n_series_all_pm_hook_lifter_servo_motor` (score=4.9892, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_BELLOWS Global SOP No : Revision No: 2 Page: 64 / 106 | Flow | Procedure 
  - [ ] `set_up_manual_precia` (score=4.9741, device=PRECIA, type=set_up_manual)
    > | 5. Parameter setting | a. 좌측 Menu 분류6(특수) 선택 b. 분류6 - 번호28 특수 기능 선택 -> 설정값 : 1:블록 동작 유효 변경 | | | :--- | :--- | :--- | 
  - [ ] `global_sop_integer_plus_all_pm_safety_controller` (score=4.9728, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_SAFETY CONTROLLER | Global_SOP No: | | | --- | --- | | Revision No: | 0 | |
  - [ ] `set_up_manual_supra_np` (score=4.9326, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I 4) FFU Controller Power Check a. EFEM ELT Box Cover Open 후 EFEM FFU MCB On 이 되어 있는지 확인한다. 4mm
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=4.8563, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_HOOK LIFTER ## SERVO MOTOR CONTROLLER REPLACEMENT Global SOP No: Revision No: 
  - [ ] `global_sop_supra_xp_all_pm_dual_epd` (score=4.8496, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_DUAL EPD REPLACEMENT Global SOP No: Revision No: 1 Page: 16 / 40 | Flow | Proc
  - [ ] `global_sop_integer_plus_all_am_temp_controller` (score=4.8395, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_REP_AM_TEMP CONTROLLER Global SOP No: Revision No: 1 Page: 6 / 27 ## Scope 이 Global SOP는 INTEG
  - [ ] `set_up_manual_supra_xq` (score=4.8333, device=SUPRA XQ, type=SOP)
    > ```markdown # 7. Teaching (※환경안전 보호구: 안전모, 안전화) ## 7-4 PM Pin Height Teaching(PM,2,3) | Picture | Description | Tool & S
  - [ ] `set_up_manual_ecolite_3000` (score=4.827, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 7. Teaching_Aligner Stage | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 16) Teaching Da
  - [ ] `global_sop_precia_all_efem_ffu` (score=4.8208, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_EFEM_FFU CONTROLLER Global SOP No : Revision No: 0 Page : 7/50 ## Scope 이 Global SOP
  - [ ] `global_sop_integer_plus_all_efem_controller` (score=4.8083, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_EFEM_CONTROLLER Global SOP No : Revision No: 1 Page: 6/28 ## Scope 이 Global SO
  - [ ] `global_sop_precia_all_ll_vision` (score=4.7965, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_LL_VISION CONTROLLER Global SOP No : Revision No: 0 Page : 19 / 82 ## Scope 이 Global
  - [ ] `global_sop_integer_plus_all_tm_ctc_controller` (score=4.7848, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_CTC CONTROLLER Global SOP No: Revision No: 1 Page: 7/51 ## Scope 이 Global S
  - [ ] `global_sop_geneva_xp_rep_pm_controller` (score=4.75, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVAxp_REP_PM_Controller | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page: 1 / 12 | | ## Sc
  - [ ] `global_sop_precia_all_efem_controller` (score=4.7392, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_EFEM CONTROLLER | Global SOP No : | | |---|---| | Revision No: 0 | | | Page : 1/42 |

#### q_id: `A-amb003`
- **Question**: MFC 교체 및 가스 라인 퍼지 절차는?
- **Devices**: [GENEVA_XP, INTEGER_PLUS, PRECIA, SUPRA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-20):
  - [ ] `global_sop_precia_all_pm_mfc` (score=7.6744, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_MFC Global SOP No : Revision No: 1 Page: 3/23 ## 3. 사고 사례 ### 1) 가스 노출 재해의 정의 “근로
  - [ ] `global_sop_supra_n_all_sub_unit_mfc` (score=7.6537, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_SUB UNIT_MFC Global SOP No: 0 Revision No: 3 Page: 3/21 ## 3. 사고 사례 ### 1) 가스 노출 재해
  - [ ] `global_sop_integer_plus_all_ll_mfc` (score=7.5874, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_LL_MFC Global SOP No: Revision No: 1 Page: 3/20 ## 3. 사고 사례 ### 1) 가스 노출 재해의 정
  - [ ] `global_sop_integer_plus_all_tm_epc` (score=7.0574, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_ INTEGER plus_ALL_TM_EPC Global_SOP No: Revision No: 0 Page: 3/21 ## 3. 사고 사례 ### 1) 가스 노출 재해의 
  - [ ] `global_sop_integer_plus_all_tm_filter` (score=7.055, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_FILTER Global_SOP No: Revision No: 0 Page: 3/20 ## 3. 사고 사례 ### 1) 가스 노출 재해
  - [ ] `global_sop_precia_all_pm_pneumatic_valve` (score=7.0508, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_PNEUMATIC VALVE Global SOP No: Revision No: 0 Page: 3/20 ## 3. 사고 사례 ### 1) 가스 노출
  - [ ] `global_sop_integer_plus_all_pm_pneumatic_valve` (score=7.0496, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_PNEUMATIC VALVE Global SOP No: Revision No: 1 Page: 3/20 ## 3. 사고 사례 ### 1)
  - [ ] `global_sop_supra_n_series_all_sub_unit_gas_box_board` (score=7.0357, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_GAS BOX BOARD | Global SOP No: 0 | | | --- | --- | | Revision No: 4
  - [ ] `global_sop_integer_plus_all_pm_igs_block` (score=7.0345, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_IGS BLOCK | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page: 3
  - [ ] `global_sop_integer_plus_all_ll_sensor_board` (score=7.0317, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_LL_SENSOR BOARD Global SOP No: Revision No: 1 Page: 3/17 ## 3. 사고 사례 ### 1) 가스
  - [ ] `global_sop_supra_xp_all_sub_unit_igs_block` (score=7.0302, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_SUB UNIT_IGS BLOCK | Global SOP No: | | | --- | --- | | Revision No: 2 | | | Page
  - [ ] `global_sop_supra_n_all_sub_unit_igs_block` (score=7.02, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_SUB UNIT_IGS BLOCK Global SOP No : Revision No: 1 Page: 3/67 ## 3. 사고 사례 ### 1) 가스 
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=7.0081, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_BARATRON GAUGE Global SOP No: Revision No: Page: 40 / 135 ## 3. 사고 사례 ### 1
  - [ ] `global_sop_supra_n_series_all_sub_unit_pressure_vacuum_switch` (score=7.0, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_PRESSURE & VACUUM SWITCH Global SOP No : Revision No: 1 Page: 3/28 
  - [ ] `global_sop_integer_plus_all_pm_gas_line` (score=6.7227, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_GAS LINE Global_SOP No: Revision No: 0 Page: 3/75 ## 3. 사고 사례 ### 1) 협착 재해의
  - [ ] `set_up_manual_supra_n` (score=5.8156, device=SUPRA N, type=SOP)
    > ```markdown # 2. Tool Fab in (Packing List Check) (※환경안전 보호구 : 안전모, 안전화) ## 2.1 Unpacking | Picture | Description | Tool
  - [ ] `set_up_manual_supra_np` (score=5.7058, device=SUPRA Np, type=set_up_manual)
    > ```markdown # 2. Tool Fab in (Packing List Check) (※환경안전 보호구: 안전모, 안전화) ## 2.1 Unpacking | Picture | Description | Tool 
  - [ ] `global_sop_supra_xp_all_pm_prism_source` (score=5.3152, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM SOURCE REPLACEMENT Global SOP No: Revision No : 2 Page : 29 / 69 | Flow 
  - [ ] `global_sop_supra_xp_all_tm_mfc` (score=5.2011, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_MFC Global SOP No: Revision No: 0 Page: 2 / 18 ## 1. Safety ### 1) 안전 및 주의사항 -
  - [ ] `global_sop_supra_n_all_pm_fcip_r5` (score=5.1815, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_PM_FCIP R5 Global SOP No : Revision No: 3 Page : 23/45 | Flow | Procedure | Tool & 

#### q_id: `A-amb004`
- **Question**: Device Net Board 교체 방법과 점검 사항은?
- **Devices**: [INTEGER_PLUS, PRECIA, SUPRA_N, SUPRA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-13):
  - [ ] `global_sop_supra_xp_all_pm_device_net_board` (score=8.2502, device=ZEDIUS XP, type=SOP)
    > # Global SOP_ZEDIUS XP_ALL_PM_DEVICE NET BOARD REPLACEMENT Global SOP No: Revision No: 0 Page: 10/34 ## 5. Flow Chart St
  - [ ] `supra_xp_all_trouble_shooting_guide_trace_device_net_abnormal` (score=8.0664, device=SUPRA XP, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Device Net Abnormal] Confidential II | | | | | :--- | :--- | :--- | | | | ▶ 
  - [ ] `supra_n_all_trouble_shooting_guide_trace_device_net_abnormal` (score=8.0036, device=SUPRA XP, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Device Net Abnormal] Confidential II | | | | | :--- | :--- | :--- | | | | ▶ 
  - [ ] `global_sop_supra_xp_all_tm_device_net_board` (score=7.9175, device=ZEDIUS XP, type=SOP)
    > # Global SOP_ZEDIUS XP_ALL_TM_DEVICE NET BOARD REPLACEMENT Global SOP No : Revision No: 2 Page : 10/33 ## 5. Flow Chart 
  - [ ] `global_sop_supra_n_series_all_tm_devicenet_board` (score=7.7413, device=SUPRA N, type=SOP)
    > # Global SOP_SUPRA N series_REP_TM_DEVICENET BOARD Global SOP No : Revision No: 2 Page : 10/31 ## 5. Flow Chart Start 1.
  - [ ] `global_sop_precia_all_tm_device_net_board` (score=7.6309, device=PRECIA, type=SOP)
    > # Global SOP_PRECIA_REP_TM_DEVICE NET BOARD Global SOP No : 0 Revision No : 0 Page : 10 / 37 ## 5. Flow Chart Start 1. G
  - [ ] `global_sop_precia_all_pm_device_net_board` (score=7.6077, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_DEVICE NET BOARD | Global SOP No : 0 | | | --- | --- | | Revision No : 0 | | | Pa
  - [ ] `supra_n_all_trouble_shooting_guide_trace_pcw_abnormal` (score=7.5281, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace PCW abnormal] Confidential II | | B-5. Device net board | ▶ Device net reset
  - [ ] `global_sop_geneva_xp_rep_pm_device_net` (score=7.4082, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Device net | SOP No: 0 | | | | | :--- | :--- | :--- | :--- | | Revision No: 0 
  - [ ] `supran_all_trouble_shooting_guide_trace_pcw_abnormal` (score=7.3798, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace PCW abnormal] Confidential II | | B-5. Device net board | ▶ Device net reset
  - [ ] `global_sop_precia_all_efem_device_net_board` (score=7.3739, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_EFEM_DEVICE NET BOARD | Global SOP No: | | |---|---| | Revision No: 0 | | | Page: 1/
  - [ ] `supra_xp_pm_trouble_shooting_guide_trace_prism_abnormal` (score=7.1765, device=SUPRA XP, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace PRISM abnormal] Confidential II | | | | |---|---|---| | | C-4. Manual Valve 
  - [ ] `global_sop_supra_n_series_all_pm_device_net_board` (score=7.1386, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_DEVICE NET BOARD | Global SOP No : | | | --- | --- | | Revision No: 2 | |

#### q_id: `A-amb005`
- **Question**: TM Robot End Effector 교체 및 티칭 절차는?
- **Devices**: [INTEGER_PLUS, PRECIA, SUPRA_N, SUPRA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-16):
  - [ ] `set_up_manual_integer_plus` (score=8.2111, device=INTEGER plus, type=SOP)
    > ```markdown # 5. Accessory Install (※환경안전 보호구: 안전모, 안전화) ## 5.6 TM Robot End Effector | Picture | Description | Tool & S
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=7.6977, device=SUPRA N, type=SOP)
    > # Global SOP_SUPRA N series_ALL_TM ROBOT TEACHING Global SOP No : Revision No: 6 Page : 76 / 107 | Flow | Procedure | To
  - [ ] `global_sop_precia_all_tm_robot` (score=7.506, device=PRECIA, type=SOP)
    > # Global SOP_PRECIA_REP_TM_TM ROBOT Global SOP No : Revision No: 0 Page : 11/56 ## 5. Flow Chart Start 1. Global SOP 및 안
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=7.5031, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_EFEM_END ## EFFECTOR REAR PAD Global SOP No: Revision No: 3 Page: 80 / 126 | F
  - [ ] `global_sop_integer_plus_all_tm_robot` (score=7.4162, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_TM_END EFFECTOR Global SOP No : Revision No: 4 Page : 64 / 103 ## Scope 이 Glob
  - [ ] `global_sop_supra_xp_all_tm_robot` (score=7.3099, device=ZEDIUS XP, type=SOP)
    > # Global SOP_ZEDIUS XP_ALL_TM_ROBOT TEACHING Global SOP No: Revision No : 3 Page : 29 / 47 | Flow | Procedure | Tool & S
  - [ ] `global_sop_supra_n_all_efem_robot_m124` (score=7.3019, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_EFEM ROBOT_M124 Global SOP No : Revision No: 1 Page: 40/45 | Flow | Procedure | Too
  - [ ] `40034833` (score=7.1721, device=SUPRA Vplus, type=myservice)
    > -. 4EPR0505 TM Robot Upper Left End Effector Broken
  - [ ] `set_up_manual_precia` (score=7.1456, device=PRECIA, type=set_up_manual)
    > # 6. Part Installation ## 6.7 TM Robot End-Effector Install | Picture | Description | Tool & Spec | | :--- | :--- | :---
  - [ ] `set_up_manual_supra_xq` (score=7.1323, device=SUPRA XQ, type=SOP)
    > ```markdown # 3. Docking (※환경안전 보호구: 안전모, 안전화) ## 3-9 TM Robot End Effector | Picture | Description | Tool & Spec | | :-
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=7.1288, device=ZEDIUS XP, type=set_up_manual)
    > ```markdown # 3. Docking (※환경안전 보호구 : 안전모, 안전화) ## 3.4 TM Robot End Effector 장착 | Picture | Description | Tool & Spec | 
  - [ ] `global_sop_supra_vplus_adj_all_undocking` (score=7.0274, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_UNDOCKING Global SOP No: Revision No: 0 Page: 8 / 19 ## 10. Work Procedure 
  - [ ] `global_sop_supra_n_adj_all_undocking` (score=7.0232, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ADJ_ALL_UNDOCKING Global SOP No: Revision No: 3 Page: 8 / 19 ## 10. Work Procedure | Fl
  - [ ] `set_up_manual_ecolite_3000` (score=6.9109, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 3. Docking (TM Robot Pick 장착) | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) TM Robot
  - [ ] `set_up_manual_supra_vm` (score=6.869, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 5. Accessory Install (TM Robot Pick) | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) T
  - [ ] `global_sop_genevaxp_rep_efem_robot_end_effector` (score=6.8238, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_EFEM_Robot end effector | Global SOP No: | | | --- | --- | | Revision No: 0 | | | 

#### q_id: `A-amb006`
- **Question**: Heater Chuck 교체 후 레벨링 및 캘리브레이션 절차는?
- **Devices**: [INTEGER_PLUS, SUPRA_N, SUPRA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-19):
  - [ ] `global_sop_supra_n_series_all_pm_heater_chuck` (score=6.8855, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_HEATER CHUCK | Global SOP No : | | | --- | --- | | Revision No: 2 | | | P
  - [ ] `global_sop_integer_plus_all_am_heater_chuck` (score=6.7711, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_HEATER CHUCK Global SOP No: Revision No: 1 Page: 6 / 25 ## Scope 이 Global S
  - [ ] `global_sop_supra_xp_all_pm_heater_chuck` (score=6.5694, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP ZEDIUS XP_ALL_PM_HEATER CHUCK Global SOP No: 0 Revision No: 2 Page: 1/49 ## Scope 이 Global SOP는
  - [ ] `global_sop_geneva_xp_rep_pm_heater_chuck_without_jig` (score=6.5232, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA xp_REP_PM_Heater chuck w/o jig | SOP No: 0 | | | |---|---|---| | Revision No: 1 | | | | Page: 3/52 
  - [ ] `global_sop_integer_plus_all_pm_baffle_heater` (score=6.2179, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_BAFFLE HEATER | Global SOP No : | | | --- | --- | | Revision No : 2 | | | P
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=6.1376, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) TC Wafer 설치 확인 | ![](https://i.imgur.com/1234567.pn
  - [ ] `global_sop_geneva_xp_adj_heater_chuck_leveling` (score=6.1283, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVA_ADJ_Heater chuck leveling | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page: 1/16 | | #
  - [ ] `global_sop_integer_plus_all_tm_top_lid` (score=6.0495, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_TOP LID Global SOP No: Revision No: 0 Page: 2/72 ## 1. Safety 1) 안전 및 주의사항 
  - [ ] `global_sop_geneva_xp_adj_pm_chuck_temp_calibration` (score=6.0234, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Chuck temp calibration Global SOP No: Revision No: 0 Page: 4/21 ## 3. 재해 방지 대책
  - [ ] `global_sop_geneva_xp_adj_pm_pin_alignment` (score=5.9777, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Pin Alignment | Global SOP No: | S-KG-A003-R0 | | --- | --- | | Revision No: |
  - [ ] `global_sop_supra_n_series_all_pm_process_kit` (score=5.9729, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_PROCESS KIT Global SOP No : Revision No: 0 Page: 3/55 ## 3. 사고 사례 ### 1) 
  - [ ] `40055962` (score=5.9699, device=SUPRA Vplus, type=myservice)
    > -. 고객측 heater Chuck Connector 탈착 후 확인 시 Ch2 그을림 확인
-> 기존 CPC Type Connector로 Harting으로 변경 요청
-> MCB1-4, 1-5 off
-. CH1, 
  - [ ] `global_sop_geneva_xp_rep_pm_support_pin` (score=5.9642, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Support pin | Global SOP No: | 0 | | --- | --- | | Revision No: | 1 | | Page: 
  - [ ] `global_sop_precia_all_pm_pendulum_valve` (score=5.9607, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_PENDULUM VALVE Global SOP No : Revision No: 0 Page: 3/32 ## 3. 사고 사례 ### 1) 화상 재해
  - [ ] `global_sop_supra_xp_all_pm_pendulum_valve` (score=5.9589, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PENDULUM VALVE Global SOP No : Revision No: 6 Page: 3/26 ## 3. 사고 사례 ### 1) 화상
  - [ ] `global_sop_supra_n_all_pm_fcip_r3` (score=5.813, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_PM_FCIP R3 Global SOP No : Revision No: 3 Page: 4/84 ## 3. 사고 사례 ### 1) 화상 재해의 정의 불
  - [ ] `global_sop_supra_xp_all_pm_pirani_gauge` (score=5.7965, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIRANI GAUGE Global SOP No: Revision No: 1 Page: 3/40 ## 3. 사고 사례 ### 1) 화상 재해
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=5.7927, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 77 / 105 ## 사고 사례 ##
  - [ ] `global_sop_supra_xp_all_pm_baratron_gauge` (score=5.7926, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_BARATRON GAUGE Global SOP No : 0 Revision No : 0 Page : 3/33 ## 3. 사고 사례 ### 1

#### q_id: `A-amb007`
- **Question**: Gas Spring 교체 기준 및 절차는?
- **Devices**: [INTEGER_PLUS, PRECIA, SUPRA_N]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-9):
  - [ ] `40055385` (score=7.4355, device=SUPRA Vplus, type=myservice)
    > -. Chamber Open
-. Old Gas Spring 탈착
-> 고객 요청으로 파트 폐기
-. New Gas Spring 장착
-> Cap 없는 Type
-. Gas Spring 동작 이상 없음 확인
-. C
  - [ ] `global_sop_precia_all_pm_gas_spring` (score=7.0185, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_GAS SPRING | Global SOP No : 0 | | | --- | --- | | Revision No : 0 | | | Page : 1
  - [ ] `global_sop_integer_plus_all_pm_gas_spring` (score=6.9246, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_GAS SPRING Global SOP No: 0 Revision No: 1 Page: 6 / 19 ## Scope 이 Global S
  - [ ] `global_sop_integer_plus_all_am_gas_spring` (score=6.9125, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_GAS SPRING | Global_SOP No: | | |---|---| | Revision No: 1 | | | Page: 6 / 
  - [ ] `global_sop_supra_n_series_all_pm_gas_spring` (score=6.903, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_GAS SPRING Global SOP No: Revision No: 4 Page: 7/22 ## Scope 이 Global SOP
  - [ ] `global_sop_integer_plus_all_pm_gas_line` (score=5.2454, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_GAS FILTER Global_SOP No: Revision No: 0 Page: 32 / 75 ## Scope 이 Global SO
  - [ ] `global_sop_integer_plus_all_am_gas_line` (score=5.2437, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_GAS FILTER | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page: 
  - [ ] `global_sop_supra_n_series_all_pm_gas_feed_through` (score=5.0751, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_GAS FEED THROUGH | Global SOP No: | | | --- | --- | | Revision No: 4 | | 
  - [ ] `global_sop_integer_plus_all_pm_gas_box_door_sensor` (score=5.0624, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_GAS BOX DOOR SENSOR Global SOP No: Revision No: 1 Page: 6 / 18 ## Scope 이 G

#### q_id: `A-amb008`
- **Question**: Slot Valve 교체 시 Leak Check 절차는?
- **Devices**: [INTEGER_PLUS, PRECIA, SUPRA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-9):
  - [ ] `global_sop_supra_xp_all_tm_slot_valve` (score=6.8722, device=ZEDIUS XP, type=SOP)
    > # Global SOP_ZEDIUS XP_ALL_TM_SLOT VALVE REPLACEMENT Global SOP No : Revision No : 5 Page : 11 / 20 ## 5. Flow Chart Sta
  - [ ] `global_sop_precia_all_tm_slot_valve` (score=6.8214, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_TM_SLOT VALVE | Global SOP No: | | |---|---| | Revision No: 0 | | | Page: 1/39 | | #
  - [ ] `global_sop_integer_plus_all_pm_slot_valve` (score=6.8183, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_SLOT VALVE | Global SOP No : | | | --- | --- | | Revision No : 2 | | | Page
  - [ ] `global_sop_integer_plus_all_am_slot_valve` (score=6.8147, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_AM_SLOT VALVE | Global SOP No : | | | --- | --- | | Revision No : 0 | | | Page
  - [ ] `global_sop_integer_plus_all_ll_slot_valve` (score=6.8142, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_LL_SLOT VALVE Global SOP No : Revision No : 4 Page : 1/50 ## Scope 이 Global SO
  - [ ] `integer_plus_all_trouble_shooting_guide_trace_slot_valve_move_abnormal` (score=6.8048, device=INTEGER plus, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Slot valve Move Abnormal] Confidential II | Failure symptoms | Check point |
  - [ ] `global_sop_precia_all_pm_slot_valve` (score=6.7408, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_SLOT VALVE(VTEX) Global SOP No : 0 Revision No : 0 Page : 29/36 | Flow | Procedur
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=6.4897, device=ZEDIUS XP, type=SOP)
    > # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No: 13 Page: 65/75 ## 5. Flow Chart Start 1
  - [ ] `integer_plus_all_trouble_shooting_guide_trace_leak_rate_over` (score=6.3395, device=INTEGER plus, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace leak rate over] Confidential II | | | | |---|---|---| | | | ▶ Baratron Gauge

#### q_id: `A-amb009`
- **Question**: Solenoid Valve 교체 및 기밀 테스트 방법은?
- **Devices**: [INTEGER_PLUS, PRECIA, SUPRA_N]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-11):
  - [ ] `global_sop_supra_n_series_all_tm_solenoid_valve` (score=6.8372, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_SOLENOID VALVE REPLACEMENT Global SOP No : Revision No: 2 Page : 13 / 18 
  - [ ] `global_sop_precia_all_tm_solenoid_valve` (score=6.122, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_TM_SOLENOID VALVE Global SOP No: Revision No: 1 Page: 13/17 | Flow | Procedure | Too
  - [ ] `global_sop_supra_n_series_all_sub_unit_solenoide_valve` (score=6.0975, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_SUB UNIT_SOLENOID VALVE Global SOP No : Revision No: 5 Page: 14 / 17 | Flow 
  - [ ] `global_sop_integer_plus_all_am_solenoid_valve` (score=5.9999, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_SOLENOID VALVE Global SOP No: Revision No: 1 Page: 14 / 19 | Flow | Procedu
  - [ ] `40046083` (score=5.3695, device=SUPRA Vplus, type=myservice)
    > -. 교체 완료
--- Document Info.---
SOP Title : Global SOP_SUPRA Vplus_REP_SUB UNIT_SOLENOID VALVE
Trouble Shooting Guide Tit
  - [ ] `global_sop_precia_all_pm_solenoid_valve` (score=5.1136, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_SOLENOID VALVE Global SOP No: 0 Revision No: 0 Page: 6 / 19 ## Scope 이 Global SOP
  - [ ] `global_sop_integer_plus_all_tm_solenoid_valve` (score=5.0818, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_TM_ # SOLENOID VALVE Global SOP No: 0 Revision No: 0 Page: 13 / 19 | Flow | Pr
  - [ ] `supra_n_all_trouble_shooting_guide_trace_vacuum_vent_abnormal` (score=5.0389, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Vacuum-Vent abnormal] Confidential II | | | | | :--- | :--- | :--- | | | | ▶
  - [ ] `global_sop_integer_plus_all_pm_solenoid_valve` (score=4.9712, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_SOLENOID VALVE Global SOP No: 0 Revision No: 0 Page: 11 / 19 ## 6. Work Pro
  - [ ] `integer_plus_all_trouble_shooting_guide_trace_slot_valve_move_abnormal` (score=4.8713, device=INTEGER plus, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Slot valve Move Abnormal] Confidential II | Failure symptoms | Check point |
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=4.8371, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | a. Solenoid bank position check<br>Gas Pneumatic<br>![i

#### q_id: `A-amb010`
- **Question**: Sensor Board 교체 후 통신 확인 절차는?
- **Devices**: [INTEGER_PLUS, PRECIA, SUPRA_N]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-15):
  - [ ] `global_sop_precia_all_efem_sensor_board` (score=6.0192, device=PRECIA, type=SOP)
    > # Global SOP_PRECIA_ALL_EFEM_SENSOR BOARD | Global SOP No: | | | --- | --- | | Revision No: 1 | | | Page: 1/17 | | ## Sc
  - [ ] `global_sop_supra_xp_all_efem_pio_sensor_board` (score=6.0188, device=ZEDIUS XP, type=SOP)
    > # Global SOP_ZEDIUS XP_ REP_EFEM_PIO SENSOR BOARD Global SOP No: Revision No: 0 Page: 12 / 14 | Flow | Procedure | Tool 
  - [ ] `global_sop_integer_plus_all_ll_sensor_board` (score=5.8714, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_REP_LL_SENSOR BOARD Global SOP No: Revision No: 1 Page: 6 / 17 ## Scope 이 Global SOP는 INTEGER 
  - [ ] `global_sop_supra_n_all_tm_sensor_board` (score=5.816, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_TM_SENSOR BOARD | Global SOP No: 0 | | --- | | Revision No: 1 | | Page: 1/15 | ## S
  - [ ] `global_sop_integer_plus_all_efem_pio_sensor_board` (score=5.8144, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_EFEM_PIO SENSOR BOARD | Global SOP No: | | |---|---| | Revision No: 1 | | | Pa
  - [ ] `global_sop_precia_all_tm_sensor_board` (score=5.8118, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_TM_SENSOR BOARD | Global SOP No: | | |---|---| | Revision No: 1 | | | Page: 1/18 | |
  - [ ] `global_sop_precia_all_efem_device_net_board` (score=5.7385, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_EFEM_DEVICE NET BOARD Global SOP No: Revision No: 0 Page: 13 / 17 | Flow | Procedure
  - [ ] `global_sop_integer_plus_all_pm_safety_limit_switch` (score=5.3371, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER Plus_REP_PM_SAFETY LIMIT SWITCH Global SOP No: Revision No: 1 Page: 14 / 19 | Flow | Pr
  - [ ] `global_sop_supra_xp_all_sub_unit_gas_box_board` (score=5.3143, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_SUB UNIT_GAS BOX BOARD | Global SOP No : | | | --- | --- | | Revision No : 1 | | 
  - [ ] `global_sop_supra_series_all_sw_operation` (score=5.0913, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_ SW OPERATION Global SOP No Revision No: 2 Page: 30/49 ## 6. Log Back Up - Wor
  - [ ] `global_sop_supra_n_series_all_pm_device_net_board` (score=5.0834, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_DEVICE NET BOARD Global SOP No : Revision No: 2 Page: 29/44 | Flow | Proc
  - [ ] `set_up_manual_ecolite_3000` (score=5.0011, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 12. TTTM (Tool Matching)_D-net Calibration_(PSK Board) | Picture | Description | Tool & Spec | | :--- | :-
  - [ ] `set_up_manual_supra_nm` (score=4.949, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.4 Device Net Calibration _ PSK Board ### 12.4.2 Micro
  - [ ] `set_up_manual_supra_n` (score=4.9445, device=SUPRA N, type=SOP)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.5 Device Net Calibration _ PSK Board ### 12.5.2 FCIP 
  - [ ] `global_sop_supra_xp_all_pm_pirani_gauge` (score=4.8746, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_REP_PM_PIRANI GAUGE Global SOP No: Revision No: 1 Page: 32 / 40 | Flow | Procedure | 

#### q_id: `A-amb011`
- **Question**: APC Valve 교체 후 AutoTune 및 Pressure Calibration 절차는?
- **Devices**: [GENEVA_XP, SUPRA_N, SUPRA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-20):
  - [ ] `supra_n_all_trouble_shooting_guide_trace_apc_abnormal` (score=6.4126, device=supra_n, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace APC Abnormal] Confidential II | Failure symptoms | Check point | Key point |
  - [ ] `global_sop_supra_n_series_all_tm_pressure_relief_valve` (score=6.2978, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_TM_PRESSURE RELIEF VALVE Global SOP No : Revision No: 2 Page: 6/19 ## Scope 
  - [ ] `global_sop_precia_all_efem_pressure_relief_valve` (score=5.9299, device=PRECIA, type=SOP)
    > # Global SOP_PRECIA_ALL_EFEM_PRESSURE RELIEF VALVE | Global SOP No: | | | --- | --- | | Revision No: 1 | | | Page: 6/16 
  - [ ] `global_sop_geneva_xp_rep_pm_apc` (score=5.8954, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA_REP_PM_APC SOP No: 0 Revision No: 0 Page: 11 / 15 | Flow | Procedure | Tool & Point | | 
  - [ ] `supra_xp_tm_trouble_shooting_guide_trace_slot_valve_abnormal` (score=5.7895, device=SUPRA XP, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Slot valve abnormal] Confidential II | | B-2. Pirani gauge | ▶ Pirani gauge 
  - [ ] `supra_xp_all_trouble_shooting_guide_trace_slot_valve_abnormal` (score=5.7895, device=SUPRA XP, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Slot valve abnormal] Confidential II | | B-2. Pirani gauge | ▶ Pirani gauge 
  - [ ] `global_sop_precia_all_tm_pressure_relief_valve` (score=5.7392, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_TM_PRESSURE RELIEF VALVE | Global SOP No: | | | --- | --- | | Revision No: 1 | | | P
  - [ ] `global_sop_supra_n_series_all_pm_pressure_gauge` (score=5.7182, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_PRESSURE GAUGE | Global SOP No: | | | --- | --- | | Revision No: 3 | | | 
  - [ ] `40036506` (score=5.6519, device=SUPRA Vplus, type=myservice)
    > - Convectron Guage Test
-> Ignition Test 30회 재현안됨
- Leak 강제 생성(Vent Port 부)
-> APC Position 및 Pressure 상승. FCIP Power 변화
  - [ ] `set_up_manual_supra_nm` (score=5.6486, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.3. APC Auto Learn | Picture | Description | Tool & Sp
  - [ ] `global_sop_geneva_xp_adj_apc_auto_calibration` (score=5.6315, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVA_ADJ_APC Auto calibration Global SOP No: Revision No: 0 Page: 5 / 15 ## 6. Flow Chart Start ↓ 1. Glob
  - [ ] `set_up_manual_supra_xq` (score=5.4668, device=SUPRA XQ, type=SOP)
    > | Module | Action | A Prior Condition | 1st Check | 2nd Check | | :--- | :--- | :--- | :--- | :--- | | PM | Slow Vacuum 
  - [ ] `set_up_manual_supra_np` (score=5.4411, device=SUPRA Np, type=set_up_manual)
    > Confidential 1 | 13) APC Learn & Parameter Check | a. APC Auto Learn 과 Set-up Parameter Check 진행 하였는가? | Yes | | | | :--
  - [ ] `global_sop_geneva_xp_rep_pm_chamber_apc_valve` (score=5.4314, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Chamber APC | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page: 1/
  - [ ] `global_sop_geneva_xp_rep_pm_loadlock_apc_valve` (score=5.4233, device=geneva_xp_rep, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Loadlock APC | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page: 1
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=5.4132, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No: 13 Page: 61/75 ## Scope 이 G
  - [ ] `global_sop_supra_n_series_all_pm_dual_epd` (score=5.3868, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_DUAL EPD REPLACEMENT & CALIBRATION | Global SOP No: | | |---|---| | Revis
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=5.3639, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 9. Adjust Component ## 9.5 APC Auto Learn | Picture | Description | Tool & Spec | | :--- | :--- | :--- | |
  - [ ] `global_sop_precia_all_efem_switch` (score=5.3134, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_EFEM_PRESSURE, VACUUM SWITCH REPLACEMENT & ADJUST Global SOP No : Revision No: 0 Pag
  - [ ] `global_sop_supra_n_series_all_efem_pressure_vacuum_switch` (score=5.2659, device=SUPRA N series, type=SOP)
    > # Global SOP_SUPRA N series_ALL_EFEM_PRESSURE SWITCH REPLACEMENT & ADJUST Global SOP No : Revision No : 2 Page : 6/29 ##

#### q_id: `A-amb012`
- **Question**: Baratron Gauge 교체 후 Zero Calibration 방법은?
- **Devices**: [INTEGER_PLUS, SUPRA_N, SUPRA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-15):
  - [ ] `set_up_manual_supra_nm` (score=7.9052, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.4 Device Net Calibration _ PSK Board ### 12.4.1 ATM P
  - [ ] `40039531` (score=7.5909, device=SUPRA Vplus, type=myservice)
    > -. EPAHZ14 PM3 Pirani Gauge Calibration 요청
-> 고객 Pirani 교체 후 810,000mTorr Reading
-> Zero 돌려봤으나 변화 없음
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=7.2126, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_BARATRON GAUGE Global SOP No: Revision No: Page: 47 / 135 | Flow | Procedur
  - [ ] `supra_n_all_trouble_shooting_guide_trace_apc_abnormal` (score=6.8603, device=supra_n, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace APC Abnormal] Confidential II | Failure symptoms | Check point | Key point |
  - [ ] `global_sop_precia_all_pm_baratron_gauge` (score=6.4794, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_BARATRON GAUGE Global SOP No : 0 Revision No : 0 Page : 26 / 55 | Flow | Procedur
  - [ ] `global_sop_supra_n_series_all_pm_pressure_gauge` (score=6.2492, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_BARATRON GAUGE Global SOP No: Revision No: 3 Page: 10/46 ## 4. 필요 Tool | 
  - [ ] `global_sop_supra_xp_all_pm_pressure_gauge` (score=6.075, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRESSURE GAUGE Global SOP No: Revision No: 1 Page: 14 / 34 | Flow | Procedure 
  - [ ] `set_up_manual_supra_n` (score=5.931, device=SUPRA N, type=SOP)
    > ```markdown # 3. Docking (※환경안전 보호구 : 안전모, 안전화) ## 3.10 Baratron Gauge, ATM Assy 장착 | Picture | Description | Tool & Spe
  - [ ] `set_up_manual_supra_np` (score=5.931, device=SUPRA Np, type=set_up_manual)
    > ```markdown # 3. Docking (※환경안전 보호구: 안전모, 안전화) ## 3.10 Baratron Gauge, ATM Assy 장착 | Picture | Description | Tool & Spec
  - [ ] `global_sop_precia_all_pm_pirani_gauge` (score=5.8829, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_PM_PIRANI GAUGE Global SOP No : 0 Revision No : 0 Page : 34 / 36 | Flow | Procedure 
  - [ ] `integer_plus_all_trouble_shooting_guide_trace_leak_rate_over` (score=5.8605, device=INTEGER plus, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace leak rate over] ## 1. AM He Leak Check Point I - Gas Line VCR - Gas Filter(1) - Gas Filt
  - [ ] `global_sop_supra_xp_all_pm_baratron_gauge` (score=5.8567, device=ZEDIUS XP, type=SOP)
    > # Global SOP_ZEDIUS XP_REP_PM_BARATRON GAUGE Global SOP No : 0 Revision No : 0 Page : 23/33 ## 5. Flow Chart Start 1. Gl
  - [ ] `global_sop_integer_plus_all_ll_vacuum_line` (score=5.794, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_LL_SIDE VACUUM LINE Global SOP No: Revision No: Page: 16 / 77 | Flow | Procedu
  - [ ] `global_sop_integer_plus_all_pm_vacuum_line` (score=5.7027, device=INTEGER XP, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_PM_BARATRON GAUGE Global SOP No: Revision No: Page : 14 / 133 | Flow | Procedu
  - [ ] `global_sop_supra_xp_all_tm_pressure_gauge` (score=5.701, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_PRESSURE GAUGE Global SOP No: Revision No: 1 Page: 14 / 31 | Flow | Procedure 

#### q_id: `A-amb013`
- **Question**: Chamber O-Ring 교체 시 규격 확인 및 교체 절차는?
- **Devices**: [GENEVA_XP, SUPRA_N, SUPRA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-16):
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=8.1364, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No : 13 Page : 21/75 | Flow | 절
  - [ ] `global_sop_supra_xp_all_pm_cip_chamber` (score=8.0191, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_CIP CHAMBER Global SOP No: Revision No : 0 Page : 20 /37 | Flow | 절차 | Tool & 
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3000qc` (score=7.716, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PRISM SOURCE 3000QC REPLACEMENT Global SOP No: Revision No : 2 Page : 20/82 | Flo
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3100qc` (score=7.0204, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PRISM SOURCE 3100QC REPLACEMENT Global SOP No: Revision No : 1 Page : 19/72 | Flo
  - [ ] `global_sop_supra_xp_all_pm_prism_source` (score=6.8315, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM SOURCE REPLACEMENT Global SOP No: Revision No : 2 Page : 33 / 69 | Flow 
  - [ ] `set_up_manual_supra_vm` (score=6.7411, device=SUPRA Vm, type=set_up_manual)
    > | 11) Chamber Close | a) ATM Transfer 확인 후 Chamber Close 를 실시 한다. | | | :--- | :--- | :--- | | | b) Chamber Close시 O-rin
  - [ ] `global_sop_integer_plus_all_am_gas_line` (score=6.703, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_BT_GAS FEED THROUGH Global SOP No: Revision No: 0 Page: 49 / 67 | Flow | Pr
  - [ ] `global_sop_supra_xp_all_pm_dual_epd` (score=6.4914, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_VIEW QUARTZ REPLACEMENT Global SOP No: Revision No: 1 Page: 33/40 | Flow | Pro
  - [ ] `global_sop_supra_xp_all_pm_process_kit` (score=6.271, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PROCESS KIT Global SOP No : Revision No : 3 Page : 18 / 28 | Flow | 절차 | Tool 
  - [ ] `global_sop_supra_xp_all_tm_slot_vv_housing_o_ring` (score=6.2657, device=SUPRA XP, type=SOP)
    > # Global SOP_ZEDIUS XP_ALL_TM_SLOT V/V HOUSING O-RING REPLACEMENT | Global SOP No: | | | --- | --- | | Revision No: 0 | 
  - [ ] `global_sop_integer_plus_all_am_pin_motor` (score=6.2521, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_PIN BELLOWS Global SOP No: Revision No: 4 Page: 40 / 84 ## 7. 작업 Check Shee
  - [ ] `global_sop_genevaxp_rep_efem_robot_end_effector` (score=6.2157, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_EFEM_Robot end effector Global SOP No: Revision No: 0 Page: 10 / 15 ## 10. Work Pr
  - [ ] `global_sop_integer_plus_all_tm_top_lid` (score=6.205, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_TM REFLECTOR QUARTS Global SOP No: Revision No: 0 Page: 65/72 | Flow | Procedu
  - [ ] `set_up_manual_ecolite_2000` (score=6.1258, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 7. Teaching_ATM Transfer | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 9) Foup Unload |
  - [ ] `set_up_manual_integer_plus` (score=6.0648, device=INTEGER plus, type=SOP)
    > | 5) Bush CLN | 1) Load Lock Bush 분해하여 D.I CLN 및 N2 Blowing (Ball Part 사이 흡착된 이물질 N2 Blowing) 2) Ball Bush CLN 간 Ball 사이
  - [ ] `global_sop_supra_n_series_all_pm_process_kit` (score=5.9973, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_PROCESS KIT Global SOP No : Revision No: 0 Page: 23/55 ## 7. 작업 Check She

#### q_id: `A-amb014`
- **Question**: Lift Pin 교체 기준 및 높이 설정 방법은?
- **Devices**: [PRECIA, SUPRA_N, SUPRA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-7):
  - [ ] `global_sop_integer_plus_all_pm_pin_motor` (score=7.4327, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_ADJ_PM_LIFT PIN LOCKTITE Global SOP No: Revision No: 5 Page: 26 / 126 ## 5. Flow Chart Start 1
  - [ ] `global_sop_precia_all_pm_lift_pin` (score=7.3201, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_PM_LIFT PIN TEACHING Global SOP No : Revision No : 0 Page : 51 / 57 | Flow | Procedu
  - [ ] `global_sop_supra_n_series_all_pm_heater_chuck` (score=7.0926, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_HEATER CHUCK Global SOP No : Revision No: 2 Page: 19/40 | Flow | Procedur
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=6.6494, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_HOOK LIFTER ## SERVO MOTOR CONTROLLER REPLACEMENT Global SOP No: Revision No: 
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=6.0255, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `global_sop_integer_plus_all_am_pin_motor` (score=6.019, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_PIN MOTOR CONTROLLER Global SOP No: Revision No: 4 Page: 78 / 84 | Flow | P
  - [ ] `global_sop_integer_plus_all_pm_swap_kit` (score=5.8521, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_SWAP KIT Global SOP No: Revision No: 3 Page: 29 / 37 | Flow | Procedure | T

#### q_id: `A-amb015`
- **Question**: Endpoint Detection 이상 시 점검 포인트는?
- **Devices**: [PRECIA, SUPRA_N, SUPRA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-27):
  - [ ] `40064646` (score=4.0166, device=SUPRA Vplus, type=myservice)
    > -. Temp up 안되는 상태로 점검 요청
  - [ ] `integer_plus_all_trouble_shooting_guide_trace_tm_robot_abnormal` (score=3.9808, device=INTEGER plus, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace TM Robot Abnormal] Confidential II ## Appendix #2 ### A. #### 11.5. Recovery
  - [ ] `set_up_manual_precia` (score=3.9209, device=PRECIA, type=set_up_manual)
    > ```markdown # 7) 중량물 취급 작업 ## 중량물 취급 시 주의사항 - 발은 어깨 너비로, 허리는 반듯이 세우고 무릎의 힘으로 일어섭니다. <!-- Image (127, 161, 594, 247) --> 
  - [ ] `global_sop_integer_plus_all_ll_disarray_sensor` (score=3.8894, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_LL_DISARRAY SENSOR Global_SOP No: Revision No: 1 Page: 3 / 18 ## 3. 사고 사례 ### 
  - [ ] `global_sop_geneva_rep_ctc` (score=3.8828, device=GENEVA XP, type=SOP)
    > ```markdown # Global_SOP_GENEVA STP300 XP_REP_CTC Global SOP No: Revision No: 0 Page: 4/18 ## 3. 사고 사례 ### 1) 감전의 정의 '감전
  - [ ] `global_sop_supra_n_series_all_rack` (score=3.8705, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_RACK Global SOP No : Revision No: 1 Page: 3/84 ## 3. 사고사례 ### 3-1. 감전의 정의 '감
  - [ ] `global_sop_supra_xp_all_ll_flow_switch` (score=3.8643, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_LL_FLOW SWITCH Global SOP No: Revision No: 2 Page: 3/30 ## 3. 사고 사례 ### 1) 감전의 정의
  - [ ] `global_sop_integer_plus_all_pm_flow_switch` (score=3.864, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_FLOW SWITCH Global SOP No: Revision No: 1 Page: 3/19 ## 3. 사고 사례 ### 1) 감전의
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=3.8637, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 14. Customer certification | Picture | Description | Spec | Check | Result | | :--- | :--- | :--- | :--- |
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=3.8633, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 22 / 105 ## 사고 사례 ##
  - [ ] `40079239` (score=3.8631, device=SUPRA Vplus, type=myservice)
    > -. All PM Pump Down, Temp 정상 Reading X
-> Temp CTR 정상 확인
-. Rack 확인 시 ELCB0-1 Trip 확인
-. ELCB DVM Check 시 Input 220V 정상

  - [ ] `global_sop_supra_xp_all_pm_flow_switch` (score=3.8612, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_FLOW SWITCH Global SOP No: Revision No: 2 Page: 3/32 ## 3. 사고 사례 ### 1) 감전의 정의
  - [ ] `global_sop_integer_plus_all_efem_eda_controller` (score=3.8597, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_EFEM_EDA # CONTROLLER Global_SOP No: Revision No: 1 Page: 3/22 ## 3. 사고 사례 ###
  - [ ] `global_sop_supra_xp_all_pm_manometer` (score=3.8581, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_MANOMETER Global SOP No : Revision No: 1 Page: 3/32 ## 3. 사고 사례 ### 1) 감전의 정의 
  - [ ] `global_sop_supra_n_series_all_pcw_turn_on` (score=3.857, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PCW TURN ON Global SOP No: Revision No: 1 Page: 3/31 ## 3. 사고 사례 ### 1) 감전의 
  - [ ] `global_sop_integer_plus_all_efem_push_button_switch` (score=3.8563, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_EFEM_PUSH BUTTON SWITCH Global SOP No: Revision No: 1 Page: 3/18 ## 3. 사고 사례 #
  - [ ] `global_sop_supra_n_series_all_sub_unit_manometer` (score=3.853, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_MANOMETER Global SOP No: Revision No: 0 Page: 3/32 ## 3. 사고 사례 ### 
  - [ ] `global_sop_precia_all_pm_manometer` (score=3.8496, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_MANOMETER Global SOP No: Revision No: 0 Page: 3/20 ## 3. 사고 사례 ### 1) 감전의 정의 ‘감전’이란 
  - [ ] `global_sop_supra_n_series_all_sub_unit_flow_switch` (score=3.8495, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_FLOW SWITCH Global SOP No : Revision No: 1 Page: 3/17 ## 3. 사고 사례 #
  - [ ] `global_sop_supra_xp_all_ll_pressure_gauge` (score=3.8444, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_LL_PRESSURE GAUGE Global SOP No: Revision No: 0 Page: 3/28 ## 3. 사고 사례 ### 3-1 감전
  - [ ] `global_sop_integer_plus_all_pm_manometer` (score=3.8443, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER Plus_ALL_MANOMETER Global SOP No: Revision No: 1 Page: 3/20 ## 3. 사고 사례 ### 1) 감전의 정의 ‘
  - [ ] `global_sop_supra_xp_all_tm_pressure_gauge` (score=3.8416, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_PRESSURE GAUGE Global SOP No: Revision No: 1 Page: 3/31 ## 3. 사고 사례 ### 3-1 감전
  - [ ] `global_sop_geneva_xp_rep_pm_disc` (score=3.8147, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/31 | | ## 3.
  - [ ] `global_sop_geneva_xp_rep_efem_load_port` (score=3.8134, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Load port | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/18 | | 
  - [ ] `global_sop_geneva_xp_adj_efem_efem_robot_leveling` (score=3.8121, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_EFEM_EFEM Robot leveling Global SOP No: Revision No: 0 Page: 3/15 ## 3. 사고 사례 ###
  - [ ] `global_sop_geneva_xp_adj_post_align_application` (score=3.8064, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_Post align application Global SOP No: 0 Revision No: 1 Page: 3/20 ## 3. 재해 방지 대책 1) 협
  - [ ] `global_sop_geneva_xp_rep_pm_pressure_switch` (score=3.7988, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Pressure switch | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/1

#### q_id: `A-amb016`
- **Question**: Chamber Vent Valve 교체 후 Leak Check 절차는?
- **Devices**: [INTEGER_PLUS, PRECIA, SUPRA_N, SUPRA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-17):
  - [ ] `set_up_manual_supra_vm` (score=7.2369, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 17. Interlock Check _ S/W | Module | Action | A Prior Condition | 1st Check | 2nd Check | | :--- | :--- | 
  - [ ] `set_up_manual_ecolite_ii_400` (score=7.1437, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 13. Customer Certification_Interlock Check_S/W | Module | Action | A Prior Condition | 1st Check | 2nd Che
  - [ ] `supra_n_all_trouble_shooting_guide_trace_vacuum_vent_abnormal` (score=7.1129, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Vacuum-Vent abnormal] Confidential II | | | | | :--- | :--- | :--- | | | | (
  - [ ] `set_up_manual_ecolite_3000` (score=6.8952, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 13. Customer Certification_Interlock Check _ S/W | PM | Module | Action | A Prior Condition | 1st Check | 
  - [ ] `global_sop_supra_series_all_sw_operation` (score=6.7467, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_ SW OPERATION Global SOP No Revision No: 2 Page: 10/49 ## 2. Gas line Leak Che
  - [ ] `set_up_manual_ecolite_2000` (score=6.5414, device=ECOLITE 2000, type=set_up_manual)
    > | | | Slow VAC Valve Open | | | |---|---|---|---|---| | | | Fast VAC Valve Open | | | | | Door Valve Close | TM Side Doo
  - [ ] `global_sop_geneva_rep_bubbler_cabinet_vent_valve` (score=6.0968, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA_REP_Bubbler Cabinet_Vent Valve | Global SOP No: | S-KG-R029-R0 | | --- | --- | | Revisio
  - [ ] `global_sop_integer_plus_all_pm_vacuum_line` (score=5.5445, device=INTEGER XP, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_PENDULUM VALVE Global SOP No: Revision No: Page: 69 / 133 | Flow | Procedur
  - [ ] `set_up_manual_precia` (score=5.5117, device=PRECIA, type=set_up_manual)
    > | | | | |---|---|---| | 3. Gas box manual valve open | a. Gas Regulator Full open<br>b. Gas manual valve lock key 제거<br>
  - [ ] `global_sop_precia_all_tm_isolation_valve` (score=5.4095, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_TM_ISOLATION VALVE | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page: 1
  - [ ] `global_sop_supra_n_series_all_pm_pressure_gauge` (score=5.3765, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_PNEUMATIC VALVE | Global SOP No: | | |---|---| | Revision No: 3 | | | Pag
  - [ ] `global_sop_geneva_xp_cln_bubbler_cabinet_canister` (score=5.3719, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Bubbler cabinet_Canister Global SOP No: 0 Revision No: 0 Page : 17 / 20 ## 10. Wo
  - [ ] `global_sop_integer_plus_all_pm_water_shut_off_valve` (score=5.361, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_WATER SHUT OFF VALVE Global SOP No: Revision No: 0 Page: 6 / 19 ## Scope 이 
  - [ ] `global_sop_integer_plus_all_ll_slot_valve` (score=5.3567, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_CLN_LL_SLOT VALVE Global SOP No : Revision No : 4 Page : 34 / 50 ## 5. Flow Chart Start -> 1. 
  - [ ] `global_sop_geneva_xp_rep_bubbler_cabinet_delivery_valve` (score=5.305, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_Bubbler Cabinet_Bubbler Delivery valve Global SOP No: S-KG-R031-R0 Revision No: 0
  - [ ] `integer_plus_all_trouble_shooting_guide_trace_leak_rate_over` (score=5.2909, device=INTEGER plus, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace leak rate over] Confidential II | | | | |---|---|---| | | | ▶ Baratron Gauge
  - [ ] `global_sop_precia_all_efem_door_valve` (score=5.2895, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_EFEM_DOOR VALVE | Global SOP No: | | |---|---| | Revision No: 0 | | | Page: 1/18 | |

#### q_id: `A-amb017`
- **Question**: Turbo Pump Overhaul 주기 및 교체 기준은?
- **Devices**: [SUPRA_N, SUPRA_XP, SUPRA_VPLUS, GENEVA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-21):
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=4.7088, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 1. Install Preparation(※환경안전 보호구 : 안전모, 안전화) ## 1.1 Foot Print Drawing | Picture | Description | Tool & Sp
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=4.6584, device=ZEDIUS XP, type=set_up_manual)
    > ```markdown # 6. Utility Turn On (PCW/GAS/PUMP)(※환경안전 보호구 : 안전모, 안전화) ## 6.6 TM Pump Turn on | Picture | Picture | Pictu
  - [ ] `set_up_manual_ecolite_3000` (score=4.6192, device=ECOLITE3000, type=set_up_manual)
    > ```markdown | 1. Installation Preperation (Layout, etc.) | | | | :--- | :--- | :--- | | Picture | Description | Tool & S
  - [ ] `set_up_manual_ecolite_2000` (score=4.6141, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 1. Installation Preperation (Layout, etc.) | Picture | Description | Tool & Spec | | :--- | :--- | :--- | 
  - [ ] `set_up_manual_ecolite_ii_400` (score=4.4827, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 1. Installation Preperation_Template Draw | Picture | Description | Tool & Spec | | :--- | :--- | :--- | |
  - [ ] `40097177` (score=4.4469, device=SUPRA Vplus, type=myservice)
    > ISO VALVE HOUSING SCREW BROKEN DURING OVERHAUL
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=4.3829, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier Global SOP No: 0 Revision No: 0 Page: 21 / 43 | Flow | Procedur
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=4.3164, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 8/15 ## 10. Work Procedure | Fl
  - [ ] `set_up_manual_supra_vm` (score=4.2325, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 1. Template Draw | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) Template Drawing 사전 준
  - [ ] `set_up_manual_integer_plus` (score=4.1346, device=INTEGER plus, type=SOP)
    > ```markdown # 17-8. Cable Hook up | Picture | Description | Data | OK | NG | N/A | | :--- | :--- | :--- | :--- | :--- | 
  - [ ] `global_sop_geneva_xp_adj_pm_aio_calibration` (score=4.0615, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_AIO CALIBRATION Global SOP No: 0 Revision No: 0 Page: 8/16 ## 10. Work Procedu
  - [ ] `global_sop_supra_vplus_all_pm_controller` (score=4.0392, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_REP_PM_Controller Global SOP No: Revision No: 3 Page: 12/42 ## 8. 필요 Tool | | Name 
  - [ ] `global_sop_supra_xp_all_pm_prism_source` (score=4.0264, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM SOURCE REPLACEMENT Global SOP No: Revision No : 2 Page : 26 / 69 | Flow 
  - [ ] `global_sop_precia_all_efem_dc_cooling_fan_motor` (score=4.0236, device=PRECIA, type=SOP)
    > # Global SOP_PRECIA_REP_EFEM_DC # COOLING FAN MOTOR Global SOP No: Revision No: 0 Page: 10 / 15 ## 5. Flow Chart Start -
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pump` (score=4.0079, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Vacuum pump SOP No: 0 Revision No: 0 Page: 11/21 ## 10. Work Procedure | Flow 
  - [ ] `global_sop_supra_v_modify_all_air_tube` (score=3.9944, device=SUPRA V, type=SOP)
    > ```markdown # Global SOP_SUPRA V_MODIFY_ALL_AIR TUBE OVERHAUL Global SOP No: Revision No: 0 Page: 7/29 ## 1. 환경 안전 보호구 |
  - [ ] `set_up_manual_supra_xq` (score=3.9883, device=SUPRA XQ, type=SOP)
    > ```markdown # 1. Install Preparation (Layout, etc) (※환경안전 보호구: 안전모, 안전화) | Picture | Description | Tool & Spec | | :--- 
  - [ ] `40079239` (score=3.9769, device=SUPRA Vplus, type=myservice)
    > -. All PM Pump Down 및 Temp Reading X
  - [ ] `global_sop_geneva_xp_all_8계통_check` (score=3.9762, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA XP_8계통_Check sheet SOP No: 0 Revision No: 0 Page: 8/46 ## 10. Work Procedure | Flow | Procedure | T
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=3.9707, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 95 / 105 ## 15. RF I
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock` (score=3.9182, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Load lock SOP No: 0 Revision No: 1 Page: 9/22 ## 10. Work Procedure | Flow | P

#### q_id: `A-amb018`
- **Question**: ESC Chuck Voltage 교정 절차는?
- **Devices**: [SUPRA_N, SUPRA_XP, SUPRA_VPLUS, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-16):
  - [ ] `global_sop_supra_n_series_all_pm_device_net_board` (score=5.0129, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_DEVICE NET BOARD Global SOP No : Revision No: 2 Page: 39/44 | Flow | Proc
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=4.9948, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No : 13 Page : 56 / 75 | Flow |
  - [ ] `global_sop_supra_n_all_pm_fcip_r5` (score=4.896, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_PM_FCIP R5 | Global SOP No: | | |---|---| | Revision No: 3 | | | Page: 9/45 | | ## 
  - [ ] `global_sop_supra_nm_all_pm_microwave_local_interface` (score=4.8943, device=SUPRA Nm, type=SOP)
    > ```markdown # Global SOP_SUPRA Nm_ADJ_PM_MICROWAVE_LOCAL INTERFACE Global_SOP No : Revision No: Page : 7/21 ## Scope 이 G
  - [ ] `global_sop_supra_xp_all_tm_ctc` (score=4.8075, device=ZEDIUS XP, type=SOP)
    > # Global SOP_ZEDIUS XP_ALL_TM_CTC REPLACEMENT Global SOP No : Revision No: 1 Page : 23 / 45 | Flow | Procedure | Tool & 
  - [ ] `global_sop_supra_n_series_all_pm_epd` (score=4.7946, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_DUAL EPD REPLACEMENT & ADJUSTMENT Global SOP No : Revision No: 3 Page: 46
  - [ ] `global_sop_supra_n_series_all_pm_dual_epd` (score=4.5523, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_DUAL EPD REPLACEMENT & CALIBRATION Global SOP No: Revision No: 4 Page: 20
  - [ ] `global_sop_precia_all_pm_chuck` (score=4.3911, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_CHUCK | Global SOP No : | | | --- | --- | | Revision No: 1 | | | Page : 1/132 | |
  - [ ] `global_sop_supra_n_series_all_pm_heater_chuck` (score=4.3056, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_HEATER CHUCK | Global SOP No : | | | --- | --- | | Revision No: 2 | | | P
  - [ ] `global_sop_supra_xp_all_pm_heater_chuck` (score=4.2664, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP ZEDIUS XP_ALL_PM_HEATER CHUCK Global SOP No: 0 Revision No: 2 Page: 1/49 ## Scope 이 Global SOP는
  - [ ] `global_sop_precia_all_tm_robot` (score=4.2621, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_TM_TM ROBOT TEACHING Global SOP No : Revision No: 0 Page : 54 / 56 ## 6. Work Proced
  - [ ] `global_sop_supra_n_all_pm_fcip_r3` (score=4.2173, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_PM_FCIP R3 REPLACEMENT Global SOP No : Revision No: 3 Page: 45/84 ## 8. Appendix | 
  - [ ] `global_sop_integer_plus_all_am_heater_chuck` (score=4.1976, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_HEATER CHUCK Global SOP No: Revision No: 1 Page: 6 / 25 ## Scope 이 Global S
  - [ ] `supra_n_all_trouble_shooting_guide_trace_tool_shut_down` (score=4.1302, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace tool shut down] Confidential II | | A-7. TR | ▶ In/Output AC voltage check<b
  - [ ] `supra_n_all_trouble_shooting_guide_trace_microwave_abnormal` (score=4.113, device=SUPRA N, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace Microwave Abnormal] ## 8.5.20 DC Bus High Voltage - DC Bus High Voltage - AC Module has 
  - [ ] `global_sop_integer_plus_all_pm_cooling_chuck` (score=4.0871, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ ALL_PM_COOLING CHUCK | Global_SOP No: | | |---|---| | Revision No: 0 | | | Page: 

#### q_id: `A-amb019`
- **Question**: RF Power Supply 교체 후 Calibration 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-11):
  - [ ] `set_up_manual_supra_np` (score=6.415, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I <!-- Table (58, 69, 928, 731) --> \begin{tabular}{|l|l|l|} \hline \textbf{4) Analog IO Calibr
  - [ ] `set_up_manual_precia` (score=6.3715, device=PRECIA, type=set_up_manual)
    > ```markdown # 6. Power Turn On (환경안전 보호구: 안전모, 안전화) ## 6.6 RF Generator Calibration | Picture | Description | Tool & Spe
  - [ ] `global_sop_supra_n_series_all_rack` (score=5.6468, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_RACK_DC POWER SUPPLY REPLACEMENT Global SOP No : Revision No: 1 Page: 62/84 
  - [ ] `set_up_manual_supra_nm` (score=5.3476, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 4) Set Power Check ※ Power 측정은 최대 출력 Power의 범위에 대한 비율로 표기한다. ※ Loss Factor (%) = (SET Power
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3100qc` (score=5.2991, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_PM_RF GENERATOR 2-POINTS CALIBRATION Global SOP No: Revision No : 1 Page : 69 / 72 | 
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3000qc` (score=5.2991, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_PM_RF GENERATOR 2-POINTS CALIBRATION Global SOP No: Revision No : 2 Page : 67 / 82 | 
  - [ ] `global_sop_supra_xp_all_tm_dc_power_supply` (score=5.299, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ REP_TM_DC POWER SUPPLY Global SOP No: Revision No: 0 Page: 14 / 18 ## 6. Work Proced
  - [ ] `global_sop_supra_xp_all_pm_prism_source` (score=4.9584, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_RF GENERATOR ## 2-POINTS CALIBRATION Global SOP No: Revision No : 2 Page : 67 
  - [ ] `global_sop_supra_n_series_all_sub_unit_elt_box_assy` (score=4.95, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_SUB ## UNIT_DC POWER SUPPLY Global SOP No: Revision No: 0 Page :71 / 95 | Fl
  - [ ] `global_sop_supra_n_series_all_tm_ctc` (score=4.9146, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_CTC REPLACEMENT Global SOP No : Revision No: 6 Page : 25/81 | Flow | Proc
  - [ ] `40080973` (score=4.8539, device=INTEGER IVr, type=myservice)
    > -. Power Supply(PS0-5) 수리 완료
--- Document Info.---
SOP Title : Global SOP_SUPRA Vplus_REP_RACK_DC POWER SUPPLY
Trouble S

#### q_id: `A-amb020`
- **Question**: Dry Pump 이상 진동 발생 시 점검 절차는?
- **Devices**: [SUPRA_N, SUPRA_XP, GENEVA_XP, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-26):
  - [ ] `40096229` (score=6.9289, device=TIGMA Vplus, type=myservice)
    > -. Dry pump power cable replacement needed due to 'Dry pump' change into new model
  - [ ] `40079239` (score=5.8789, device=SUPRA Vplus, type=myservice)
    > -. All PM Pump Down, Temp 정상 Reading X
-> Temp CTR 정상 확인
-. Rack 확인 시 ELCB0-1 Trip 확인
-. ELCB DVM Check 시 Input 220V 정상

  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pump` (score=5.6116, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Vacuum pump SOP No: 0 Revision No: 0 Page: 16 / 21 ## 10. Work Procedure | Flo
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pressure_switch` (score=4.5981, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Vacuum Pressure switch | SOP No: 0 | | |---|---| | Revision No: 0 | | | Page: 3
  - [ ] `global_sop_geneva_xp_rep_pm_mfc` (score=4.5834, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_MFC | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/18 | | ## 3. 
  - [ ] `global_sop_geneva_xp_rep_pm_pressure_switch` (score=4.5825, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Pressure switch | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/1
  - [ ] `global_sop_geneva_xp_rep_pm_heat_exchanger` (score=4.58, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Heat exchanger Global SOP No: Revision No: 0 Page: 3 / 14 ## 3. 사고 사례 ### 1. 협착
  - [ ] `global_sop_geneva_xp_adj_efem_efem_robot_leveling` (score=4.5793, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_EFEM_EFEM Robot leveling Global SOP No: Revision No: 0 Page: 3/15 ## 3. 사고 사례 ###
  - [ ] `global_sop_geneva_xp_rep_efem_load_port` (score=4.5712, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Load port | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/18 | | 
  - [ ] `40035345` (score=4.5648, device=SUPRA V, type=myservice)
    > -. LOG 확인 시 FFU 점검 이전 부터 EFEM TO CTC Communication Alarm 발생
-> EFEM TO CTC Communication LOG 끊김 확인
-> 고객 Inform 완료
-. PM
  - [ ] `global_sop_geneva_xp_rep_pm_disc` (score=4.5613, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/31 | | ## 3.
  - [ ] `global_sop_geneva_xp_all_8계통_check` (score=4.5445, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA XP_8계통_Check sheet | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/46 | | ## 3. 사고 사례 
  - [ ] `global_sop_geneva_xp_adj_post_align_application` (score=4.5438, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_Post align application Global SOP No: 0 Revision No: 1 Page: 3/20 ## 3. 재해 방지 대책 1) 협
  - [ ] `global_sop_geneva_xp_rep_pm_chamber_lock` (score=4.5401, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Chamber lock Global SOP No: Revision No: 0 Page: 3 / 14 ## 3. 사고 사례 ### 1. 협착 위
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock` (score=4.466, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Load lock | SOP No: 0 | | | | | :--- | :--- | :--- | :--- | | Revision No: 1 |
  - [ ] `global_sop_supra_n_series_all_sub_unit_pressure_vacuum_switch` (score=4.4497, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_PRESSURE & VACUUM SWITCH Global SOP No : Revision No: 1 Page: 3/28 
  - [ ] `global_sop_integer_plus_all_ll_mfc` (score=4.4395, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_LL_MFC Global SOP No: Revision No: 1 Page: 3/20 ## 3. 사고 사례 ### 1) 가스 노출 재해의 정
  - [ ] `global_sop_integer_plus_all_ll_sensor_board` (score=4.428, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_LL_SENSOR BOARD Global SOP No: Revision No: 1 Page: 3/17 ## 3. 사고 사례 ### 1) 가스
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=4.4173, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_BARATRON GAUGE Global SOP No: Revision No: Page: 40 / 135 ## 3. 사고 사례 ### 1
  - [ ] `precia_all_trouble_shooting_guide_pin_motor_abnormal` (score=4.3269, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # PRECIA Trouble Shooting Guide [Pin Motor Abnormal] Confidential II | Alarm Code | LED 점멸 횟수 | Alarm 종류 | 원
  - [ ] `global_sop_integer_plus_all_pm_gas_line` (score=4.3259, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_GAS LINE Global_SOP No: Revision No: 0 Page: 3/75 ## 3. 사고 사례 ### 1) 협착 재해의
  - [ ] `40051833` (score=4.2481, device=SUPRA Vplus, type=myservice)
    > -. Log 확인시 Placement Error로 보여짐
-> 싸이맥스 로그 확인 시 S4(Foup 없는상태) -> S6(Present 감지) -> S7(Placement 감지) 순으로 변해야하나 S6에서 계속 바뀌
  - [ ] `global_sop_supra_n_series_all_efem_ffu` (score=4.2474, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_EFEM_FFU | Global SOP No: | | | --- | --- | | Revision No: 4 | | | Page: 2/5
  - [ ] `global_sop_supra_xp_all_ll_pressure_gauge` (score=4.2339, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_LL_PRESSURE GAUGE Global SOP No: Revision No: 0 Page: 18 / 28 ## 7. 작업 Check Shee
  - [ ] `set_up_manual_integer_plus` (score=4.21, device=INTEGER plus, type=SOP)
    > ```markdown | 5) Pump & AGV Valve Turn on | a. Pump가 Turn on되면 AGV Valve Controller가 켜졌는지 확인한다. | ※ 통신 연결 확인여부는 History 
  - [ ] `global_sop_supra_n_series_all_pm_isolation_valve` (score=4.1783, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_ISOLATION VALVE Global SOP No : Revision No: 2 Page: 15/25 | Flow | Proce

#### q_id: `A-amb021`
- **Question**: Load Lock Door O-Ring 교체 절차는?
- **Devices**: [SUPRA_XP, SUPRA_N, GENEVA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-8):
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock` (score=8.4595, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Load lock SOP No: 0 Revision No: 1 Page: 13/22 ## 10. Work Procedure | Flow | 
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock_door_o_ring` (score=8.2611, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_LOAD LOCK DOOR O-RING | Global SOP No: | S-KG-R020-R0 | | --- | --- | | Revisi
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock_o_ring` (score=7.7145, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVA xp_REP_PM_LOAD LOCK O-RING | Global SOP No: | S-KG-R019-R0 | | --- | --- | | Revision No: | 0 | | Pa
  - [ ] `set_up_manual_supra_xq` (score=7.6601, device=SUPRA XQ, type=SOP)
    > # O. Teaching ## 0.1 Teaching Flow Chart | Position | Internal | | :--- | :--- | | | **Module** | **Finger** | **TH / X*
  - [ ] `set_up_manual_integer_plus` (score=7.6499, device=INTEGER plus, type=SOP)
    > | 5) Bush CLN | 1) Load Lock Bush 분해하여 D.I CLN 및 N2 Blowing (Ball Part 사이 흡착된 이물질 N2 Blowing) 2) Ball Bush CLN 간 Ball 사이
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=6.6428, device=ZEDIUS XP, type=set_up_manual)
    > | | | | |---|---|---| | 6) Load Lock 2 Ready 이동 | a. EFEM Robot Pendent 하단에 있는 Servo를 놀른 후 Pendent의 Enter Button을 Click.
  - [ ] `global_sop_geneva_xp_rep_pm_disc` (score=6.633, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc SOP No: 0 Revision No: 0 Page: 22 / 31 ## 10. Work Procedure | Flow | Pro
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=6.5742, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_EFEM_ROBOT TEACHING Global SOP No: Revision No: 3 Page: 39 / 126 | Flow | Proc

#### q_id: `A-amb022`
- **Question**: Gas Box Manual Valve 동작 확인 방법은?
- **Devices**: [SUPRA_N, PRECIA, INTEGER_PLUS, SUPRA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-17):
  - [ ] `set_up_manual_supra_np` (score=7.9931, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I | 2) Gas Box Manual V/V On | a. Gas Box Manual V/V를 시계방향으로 돌려 On 시킨다. | | | :--- | :--- | :--
  - [ ] `set_up_manual_precia` (score=6.9506, device=PRECIA, type=set_up_manual)
    > | | | | |---|---|---| | 3. Gas box manual valve open | a. Gas Regulator Full open<br>b. Gas manual valve lock key 제거<br>
  - [ ] `set_up_manual_ecolite_2000` (score=6.9402, device=ECOLITE 2000, type=set_up_manual)
    > | | | Pressure < 700,000 mT | | | |---|---|---|---|---| | | | Pressure > 850,000 mT | | | | | Door Valve Open | ATM Sens
  - [ ] `40046083` (score=6.935, device=SUPRA Vplus, type=myservice)
    > -. Sol Valve 교체 후 Manual 동작 정상 확인
-. 백업 예정
  - [ ] `set_up_manual_ecolite_ii_400` (score=6.8689, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 13. Customer Certification_Interlock Check_S/W | Module | Action | A Prior Condition | 1st Check | 2nd Che
  - [ ] `set_up_manual_supra_vm` (score=6.8608, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 17. Interlock Check _ S/W | Module | Action | A Prior Condition | 1st Check | 2nd Check | | :--- | :--- | 
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=6.7736, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM ROBOT ASSY REPLACEMENT Global SOP No : Revision No: 6 Page: 49 / 107 ## 8
  - [ ] `set_up_manual_ecolite_3000` (score=6.6848, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 9. Gas Turn On | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) 고객社 공급단 Process Gas On 
  - [ ] `global_sop_supra_xp_all_sub_unit_igs_block` (score=6.6071, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_SUB UNIT_IGS BLOCK Global SOP No : Revision No: 2 Page: 51/71 | Flow | Procedure 
  - [ ] `set_up_manual_supra_xq` (score=6.4986, device=SUPRA XQ, type=SOP)
    > | Module | Action | A Prior Condition | 1st Check | 2nd Check | | :--- | :--- | :--- | :--- | :--- | | | RF Bias1 On | A
  - [ ] `global_sop_precia_adj_all_utility_turn_onoff` (score=6.4974, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_ALL_UTILITY TURN ON/OFF Global SOP No : Revision No: 0 Page : 20 / 44 ## 10. Work Pr
  - [ ] `set_up_manual_integer_plus` (score=6.4791, device=INTEGER plus, type=SOP)
    > ```markdown # 9. Pump Turn On & Chamber Leak Check (환경안전 보호구: 안전모, 안전화) ## 9.2 Chamber Leak Check | Picture | Descriptio
  - [ ] `global_sop_supra_n_series_all_sub_unit_gas_box_board` (score=6.4506, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_SUB UNIT _GAS BOX BOARD Global SOP No: 0 Revision No: 4 Page: 21 / 23 | Flow
  - [ ] `set_up_manual_supra_nm` (score=6.3818, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 9. Gas Turn On (※환경안전 보호구 : 안전모, 안전화) ## 9.1 Gas Turn On | Picture | Description | Tool & S
  - [ ] `set_up_manual_omnis` (score=6.3703, device=OMNIS, type=set_up_manual)
    > ```markdown # 20. Process Gas Turn On | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1)Turn On the Pro
  - [ ] `set_up_manual_omnis_plus` (score=6.3694, device=OMNIS plus, type=set_up_manual)
    > ```markdown # 19. Process Gas Turn On | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1)Turn On the Pro
  - [ ] `global_sop_geneva_xp_all_8계통_check` (score=6.3582, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA XP_8계통_Check sheet SOP No: 0 Revision No: 0 Page: 29 / 46 ## 10. Work Procedure | Flow | Procedure 

#### q_id: `A-amb023`
- **Question**: Ionizer 교체 및 성능 확인 절차는?
- **Devices**: [SUPRA_XP, SUPRA_VPLUS, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-17):
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=5.1281, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No: 13 Page: 61/75 ## Scope 이 G
  - [ ] `global_sop_supra_xp_all_efem_ionizer` (score=5.0671, device=ZEDIUS XP, type=SOP)
    > # Global SOP_ ZEDIUS XP_ALL_EFEM_IONIZER Global SOP No: Revision No: 1 Page : 13 / 17 | Flow | Procedure | Tool & Point 
  - [ ] `global_sop_supra_xp_all_pm_chamber_safety_cover` (score=4.9664, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_SAFETY COVER REPLACEMENT | Global SOP No : | | | --- | --- | | Revision No : 1
  - [ ] `global_sop_supra_xp_all_pm_cip_chamber` (score=4.9597, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_CIP CHAMBER | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page: 9 
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=4.5343, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_LM GUIDE ## REPLACEMENT & GREASE INJECTION | Global SOP No: | | |---|---| | Re
  - [ ] `global_sop_supra_xp_all_tm_ctc` (score=4.3895, device=ZEDIUS XP, type=SOP)
    > # Global SOP_ZEDIUS XP_ALL_TM_CTC REPLACEMENT | Global SOP No : | | | --- | --- | | Revision No: 1 | | | Page : 5/45 | |
  - [ ] `global_sop_supra_n_series_all_efem_ffu` (score=4.3624, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_EFEM_FFU | Global SOP No: | | | --- | --- | | Revision No: 4 | | | Page: 2/5
  - [ ] `global_sop_supra_n_series_all_sub_unit_temp_controller` (score=4.3292, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_ TEMP CONTROLLER Global SOP No : Revision No: 2 Page : 11 / 56 ## 6
  - [ ] `global_sop_supra_xp_all_sub_unit_gas_box_board` (score=4.3281, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_SUB UNIT_GAS BOX BOARD Global SOP No : Revision No : 1 Page : 6/18 ## Scope 이 Glo
  - [ ] `global_sop_integer_plus_all_pm_wall_temp_controller` (score=4.3069, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_WALL TEMP CONTROLLER Global SOP No: Revision No: 0 Page: 2/21 ## 1. SAFETY 
  - [ ] `global_sop_supra_xp_all_tm_slot_valve` (score=4.3034, device=ZEDIUS XP, type=SOP)
    > # Global SOP_ZEDIUS XP_ALL_TM_SLOT VALVE REPLACEMENT Global SOP No : Revision No : 5 Page : 11 / 20 ## 5. Flow Chart Sta
  - [ ] `global_sop_supra_n_series_all_tm_ctc` (score=4.2665, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_CTC REPLACEMENT | Global SOP No : | | | --- | --- | | Revision No: 6 | | 
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3000qc` (score=4.2448, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PRISM SOURCE 3000QC REPLACEMENT Global SOP No: Revision No : 2 Page : 20/82 | Flo
  - [ ] `global_sop_supra_xp_all_efem_ffu` (score=4.2355, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_EFEM_FFU Global SOP No: Revision No : 1 Page : 2/60 ## 1. Safety 1) 안전 및 주의사항 - F
  - [ ] `global_sop_integer_plus_all_efem_selector_switch` (score=4.1874, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_EFEM_SELECTOR SWITCH Global SOP No: Revision No: 1 Page: 2 / 15 ## 1. SAFETY 1
  - [ ] `global_sop_precia_all_efem_switch` (score=4.1518, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_EFEM_PRESSURE, VACUUM SWITCH REPLACEMENT & ADJUST Global SOP No : Revision No: 0 Pag
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3100qc` (score=4.1513, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PRISM SOURCE ## 3100QC REPLACEMENT Global SOP No: Revision No : 1 Page : 20/72 | 

#### q_id: `A-amb024`
- **Question**: Chiller Temperature 불안정 시 점검 사항은?
- **Devices**: [SUPRA_N, SUPRA_VPLUS, GENEVA_XP, PRECIA]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-12):
  - [ ] `global_sop_precia_all_pm_chuck` (score=6.1375, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_CHUCK Chiller Line Global SOP No : Revision No: 1 Page :109 / 132 ## 6. Work Proc
  - [ ] `set_up_manual_integer_plus` (score=5.0704, device=INTEGER plus, type=SOP)
    > ```markdown # 8. Utility Turn On (환경안전 보호구: 안전모, 안전화) ## 8.3 Chiller & Heat Exchanger Turn on | Picture | Description | 
  - [ ] `global_sop_precia_all_pm_prevent_maintenance` (score=5.0005, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_PM_GAP TEACHING Global SOP No: Revision No: 5 Page: 90 / 108 ## 6. Work Procedure | 
  - [ ] `global_sop_geneva_xp_all_8계통_check` (score=4.9471, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA XP_8계통_Check sheet SOP No: 0 Revision No: 0 Page: 22 / 46 ## 10. Work Procedure | Flow | Procedure 
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3000qc` (score=4.8107, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM 3000,3100QC SOURCE IGNITION WINDOW CHECK Global SOP No: Revision No: 2 P
  - [ ] `global_sop_precia_all_pm_rf_bias_matcher` (score=4.6659, device=PRECIA, type=SOP)
    > # Global SOP_PRECIA_REP_PM_RF ROD Global SOP No: Revision No: 1 Page: 27 / 55 ## 5. Flow Chart Start 1. Global SOP 및 안전 
  - [ ] `supra_n_all_trouble_shooting_guide_trace_temperature_abnormal` (score=4.5613, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Temperature abnormal] Use this guide to diagnose problems with the [Trace Te
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=4.426, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 14. Customer certification | Picture | Description | Spec | Check | Result | | :--- | :--- | :--- | :--- |
  - [ ] `global_sop_integer_plus_all_pm_quick_connector` (score=4.2644, device=INTEGER plus, type=SOP)
    > # Global SOP _ INTEGER plus _ REP _ PM _ QUICK CONNECTOR Global SOP No: Revision No: 1 Page: 10 / 17 ## 5. Flow Chart St
  - [ ] `global_sop_precia_all_pm_lift_pin` (score=4.0499, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_LIFT PIN BASE Global SOP No : Revision No : 0 Page : 12 / 57 | Flow | Procedure |
  - [ ] `global_sop_supra_n_series_all_pm_gas_feed_through` (score=4.0366, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS FEED THROUGH Global SOP No: Revision No: 4 Page: 4 / 18 ## 3. 사고 사례 #
  - [ ] `global_sop_supra_n_series_all_tm_fluorescent_lamp` (score=4.0355, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_FLUORESCENT LAMP Global SOP No : Revision No:2 Page: 2/23 ## 1. Safety 1)

#### q_id: `A-amb025`
- **Question**: Exhaust Pressure Gauge 교체 및 Calibration 방법은?
- **Devices**: [INTEGER_PLUS, PRECIA, SUPRA_N]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-21):
  - [ ] `supra_xp_tm_trouble_shooting_guide_trace_slot_valve_abnormal` (score=7.101, device=SUPRA XP, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Slot valve abnormal] Confidential II | | B-2. Pirani gauge | ▶ Pirani gauge 
  - [ ] `supra_xp_all_trouble_shooting_guide_trace_slot_valve_abnormal` (score=7.101, device=SUPRA XP, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Slot valve abnormal] Confidential II | | B-2. Pirani gauge | ▶ Pirani gauge 
  - [ ] `set_up_manual_supra_np` (score=7.024, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I Analog Output Tap은 조작하지 않는다. # 12. TTTM (※환경안전 보호구: 안전모, 안전화) ## 12.5 Device Net Calibration 
  - [ ] `set_up_manual_supra_nm` (score=7.0095, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.4 Device Net Calibration _ PSK Board ### 12.4.1 ATM P
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=6.0395, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 4) Open PM CDA/VAC Regulator | a. Main CDA Turn on 후 CD
  - [ ] `global_sop_supra_xp_all_pm_pressure_gauge` (score=5.854, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRESSURE GAUGE Global SOP No: Revision No: 1 Page: 14 / 34 | Flow | Procedure 
  - [ ] `set_up_manual_supra_n` (score=5.6588, device=SUPRA N, type=SOP)
    > ```markdown Confidential 1 # 6) Analog IO Calibration <!-- Image (71, 70, 370, 285) --> a. Analog Input Calibration은 프로그
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pressure_switch` (score=5.6131, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVAxp_REP_PM_Vacuum Pressure switch SOP No: 0 Revision No: 0 Page: 7 / 18 ## 8. 필요 Tool | Name | Teflon 
  - [ ] `global_sop_integer_plus_all_pm_vacuum_line` (score=5.6072, device=INTEGER XP, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_PM_BARATRON GAUGE Global SOP No: Revision No: Page : 14 / 133 | Flow | Procedu
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=5.5916, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_AM_BARATRON GAUGE Global SOP No: Revision No: Page : 16 / 135 | Flow | Procedu
  - [ ] `global_sop_supra_n_series_all_pm_pressure_gauge` (score=5.527, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_PRESSURE GAUGE | Global SOP No: | | | --- | --- | | Revision No: 3 | | | 
  - [ ] `global_sop_supra_xp_all_pm_baratron_gauge` (score=5.526, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ADJ_PM_BARATRON GAUGE Global SOP No : 0 Revision No : 0 Page : 14 / 33 | Flow | Proce
  - [ ] `global_sop_geneva_xp_all_8계통_check` (score=5.5138, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA XP_8계통_Check sheet SOP No: 0 Revision No: 0 Page: 24 / 46 ## 10. Work Procedure | Flow | Procedure 
  - [ ] `global_sop_geneva_xp_rep_pm_pressure_switch` (score=5.502, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Pressure switch | SOP No: 0 | | |---|---| | Revision No: 0 | | | Page: 7/16 | 
  - [ ] `global_sop_precia_all_pm_baratron_gauge` (score=5.2592, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_PM_CHUCK # BACK.BARATRON GAUGE Global SOP No : 0 Revision No : 0 Page : 51 / 55 | Fl
  - [ ] `global_sop_supra_xp_all_tm_pressure_gauge` (score=5.2254, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_PRESSURE GAUGE Global SOP No: Revision No: 1 Page: 29 / 31 | Flow | Procedure 
  - [ ] `40039531` (score=5.2118, device=SUPRA Vplus, type=myservice)
    > -. EPAHZ14 PM3 Pirani Gauge Calibration 요청
-> 고객 Pirani 교체 후 810,000mTorr Reading
-> Zero 돌려봤으나 변화 없음
  - [ ] `supra_n_all_trouble_shooting_guide_trace_apc_abnormal` (score=5.1567, device=supra_n, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace APC Abnormal] Confidential II | Failure symptoms | Check point | Key point |
  - [ ] `set_up_manual_precia` (score=5.1222, device=PRECIA, type=set_up_manual)
    > | | | b. Coolant 용액은 적정 Level 수준 으로 Charge되어 있는가? | | | | | :--- | :--- | :--- | :--- | :--- | :--- | | 3) Utility Turn 
  - [ ] `global_sop_geneva_xp_rep_pm_differential_gauge` (score=5.0711, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVA xp_REP_PM_Differential gauge Global SOP No: Revision No: 0 Page: 5/18 ## 6. Flow Chart Start 1. SOP 
  - [ ] `global_sop_supra_xp_all_ll_pressure_gauge` (score=5.0683, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_LL_PRESSURE GAUGE Global SOP No: Revision No: 0 Page: 26 / 28 | Flow | Procedure 

#### q_id: `A-amb026`
- **Question**: Wafer Transfer 실패 시 점검 포인트는?
- **Devices**: [SUPRA_N, SUPRA_XP, INTEGER_PLUS, PRECIA]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-10):
  - [ ] `set_up_manual_ecolite_3000` (score=5.7742, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 7. Teaching_Process Module 2 | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) Semi Tran
  - [ ] `set_up_manual_supra_vm` (score=5.4699, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 9. TM Robot Teaching_Cooling Stage 2 | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) S
  - [ ] `global_sop_supra_n_series_all_efem_load_port_duraport` (score=4.6024, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_EFEM_LOAD PORT_DURAPORT Global SOP No : Revision No: 1 Page: 74/93 | Flow | 
  - [ ] `set_up_manual_ecolite_ii_400` (score=4.502, device=ECOLITE II 400, type=set_up_manual)
    > | 8) Wafer Status 반복 Check | a) PM, 2로 반복 진행 한다. | | | | :--- | :--- | :--- | :--- | | | b) Wafer의 안착 상태가 불량 시 재 Teachin
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=4.4904, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM ROBOT TEACHING Global SOP No : Revision No: 6 Page : 84 / 107 | Flow | Pr
  - [ ] `set_up_manual_ecolite_2000` (score=4.4198, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 7. Teaching_ATM Transfer | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 9) Foup Unload |
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=4.3406, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 11. Robot Teaching ## 11.5 EFEM Loadlock | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 
  - [ ] `set_up_manual_supra_n` (score=4.2408, device=SUPRA N, type=SOP)
    > ```markdown Confidential I 4) Wafer 매수 설정 후 Strat <!-- Image (72, 75, 357, 247) --> a. ATM Transfer를 진행할 Wafer Slot 선택 후
  - [ ] `set_up_manual_supra_np` (score=4.2329, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I 4) Wafer 매수 설정 후 Strat <!-- Image (62, 72, 344, 247) --> a. ATM Transfer를 진행할 Wafer Slot 선택 후
  - [ ] `set_up_manual_precia` (score=4.1979, device=PRECIA, type=set_up_manual)
    > | | | | |---|---|---| | | a. 'JOG' Mode -> 'Up' Key Click<br>b. Stage1 End-Effector와 Wafer가 닿는<br>위치 Z-Axis 확인<br>c. 'Up

#### q_id: `A-amb027`
- **Question**: PM 후 Base Pressure 미달 시 조치 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, SUPRA_VPLUS, PRECIA]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-18):
  - [ ] `set_up_manual_integer_plus` (score=5.7621, device=INTEGER plus, type=SOP)
    > ```markdown # 9. Pump Turn On & Chamber Leak Check (환경안전 보호구: 안전모, 안전화) ## 9.2 Chamber Leak Check | Picture | Descriptio
  - [ ] `global_sop_integer_plus_all_pm_vacuum_line` (score=5.6924, device=INTEGER XP, type=SOP)
    > ```markdown # Global SOP_INTEGER plus _REP_PM_Vacuum Line Global SOP No: Revision No: Page : 129 / 133 | Flow | Procedur
  - [ ] `set_up_manual_supra_nm` (score=5.6407, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.2 EFEM Robot Leveling & Loadport Teaching | Picture
  - [ ] `set_up_manual_supra_np` (score=5.5744, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I Analog Output Tap은 조작하지 않는다. # 12. TTTM (※환경안전 보호구: 안전모, 안전화) ## 12.5 Device Net Calibration 
  - [ ] `global_sop_integer_plus_all_ll_vacuum_line` (score=5.2017, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_LL_PIRANI # BARATRON Global SOP No: Revision No: Page: 45 / 77 | Flow | Proced
  - [ ] `set_up_manual_supra_n` (score=5.1458, device=SUPRA N, type=SOP)
    > ```markdown Confidential I 36) LP2,3 Teaching <!-- Image (70, 70, 290, 226) --> a. LP1 Teaching 과 동일한 방법으로 LP2,3 를 진행하여 
  - [ ] `global_sop_integer_plus_all_pm_flange_adaptor` (score=5.1364, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_FLANGE ADAPTOR Global SOP No : Revision No : 2 Page : 17 / 21 | Flow | Proc
  - [ ] `global_sop_precia_all_pm_baratron_gauge` (score=5.1108, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_BARATRON GAUGE Global SOP No : 0 Revision No : 0 Page : 27 / 55 | Flow | Procedur
  - [ ] `global_sop_integer_plus_all_pm_pin_motor` (score=5.0284, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_PM_PIN TEACHING Global SOP No: Revision No: 5 Page: 123 / 126 | Flow | Procedu
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=4.923, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_BARATRON GAUGE Global SOP No: Revision No: Page : 51 / 135 | Flow | Procedu
  - [ ] `global_sop_supra_xp_all_pm_baratron_gauge` (score=4.8567, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ADJ_PM_BARATRON GAUGE Global SOP No : 0 Revision No : 0 Page : 14 / 33 | Flow | Proce
  - [ ] `global_sop_integer_plus_all_pm_slot_valve` (score=4.8261, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_CLN_PM_SLOT VALVE Global SOP No : Revision No : 2 Page : 35 / 39 | Flow | Procedur
  - [ ] `global_sop_supra_xp_all_ll_pressure_gauge` (score=4.8079, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_LL_PRESSURE GAUGE Global SOP No: Revision No: 0 Page: 26 / 28 | Flow | Procedure 
  - [ ] `global_sop_integer_plus_all_am_pin_motor` (score=4.7932, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER PLUS_ADJ_AM_PIN # TEACHING Global SOP No: Revision No: 4 Page: 16 / 84 | Flow | Procedu
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pressure_switch` (score=4.7502, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVAxp_REP_PM_Vacuum Pressure switch SOP No: 0 Revision No: 0 Page: 7 / 18 ## 8. 필요 Tool | Name | Teflon 
  - [ ] `40046755` (score=4.7253, device=SUPRA Nm, type=myservice)
    > -. 고온 방지 Cover 장착
-. Temp Limit Setting
-. 정기 PM 중이라 Baffle, Focus Adaptor 장착방법 고객 및 PM업체 공유
-. 고객측에서 Applicator Tube 교체
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=4.7221, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `global_sop_geneva_xp_rep_pm_pressure_switch` (score=4.6544, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Pressure switch | SOP No: 0 | | |---|---| | Revision No: 0 | | | Page: 7/16 | 

#### q_id: `A-amb028`
- **Question**: Vacuum Gauge 교체 후 Zero Adjust 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, INTEGER_PLUS, GENEVA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-17):
  - [ ] `set_up_manual_supra_nm` (score=6.956, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.4 Device Net Calibration _ PSK Board ### 12.4.1 ATM P
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=6.2676, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_AM_PIRANI GAUGE Global SOP No: Revision No: Page : 32 / 135 | Flow | Procedure
  - [ ] `40039531` (score=6.1964, device=SUPRA Vplus, type=myservice)
    > -. EPAHZ14 PM3 Pirani Gauge Calibration 요청
-> 고객 Pirani 교체 후 810,000mTorr Reading
-> Zero 돌려봤으나 변화 없음
  - [ ] `global_sop_integer_plus_all_pm_vacuum_line` (score=5.8866, device=INTEGER XP, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_PM_PIRANI GAUGE Global SOP No: Revision No: Page: 27 / 133 | Flow | Procedure 
  - [ ] `global_sop_integer_plus_all_ll_vacuum_line` (score=5.6923, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_LL_PIRANI GAUGE Global SOP No: Revision No: Page: 29 / 77 | Flow | Procedure |
  - [ ] `global_sop_supra_xp_all_ll_cip` (score=5.5785, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_SUPRA XP_REP_LL_PIRANI GAUGE REP Global SOP No: Revision No: 0 Page: 16 / 51 | Flow | Procedure
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pressure_switch` (score=5.4526, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVAxp_REP_PM_Vacuum Pressure switch SOP No: 0 Revision No: 0 Page: 7 / 18 ## 8. 필요 Tool | Name | Teflon 
  - [ ] `global_sop_supra_n_series_all_pm_pressure_gauge` (score=5.4107, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_BARATRON GAUGE Global SOP No: Revision No: 3 Page: 10/46 ## 4. 필요 Tool | 
  - [ ] `global_sop_precia_all_pm_pirani_gauge` (score=5.1634, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_PM_PIRANI GAUGE Global SOP No : 0 Revision No : 0 Page : 34 / 36 | Flow | Procedure 
  - [ ] `global_sop_geneva_xp_rep_pm_differential_gauge` (score=5.1453, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Differential gauge Global SOP No: Revision No: 0 Page: 11 / 18 | Flow | Proced
  - [ ] `global_sop_integer_plus_all_tm_vacuum_line` (score=5.1452, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_VACUUM LINE Global SOP No: Revision No: 1 Page: 18 / 26 | Flow | Procedure 
  - [ ] `set_up_manual_precia` (score=5.0356, device=PRECIA, type=set_up_manual)
    > | | | b. Coolant 용액은 적정 Level 수준 으로 Charge되어 있는가? | | | | | :--- | :--- | :--- | :--- | :--- | :--- | | 3) Utility Turn 
  - [ ] `40081285` (score=4.9868, device=SUPRA Vplus, type=myservice)
    > -. PM2 Undocking
-. Robot replacement
-. PM2 Docking
-> PCW 배관 간섭으로 인해 지방 1EA 미장착
-. Robot Teaching
-. Seasoning Test 중 
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=4.9299, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM ROBOT ASSY REPLACEMENT Global SOP No : Revision No: 6 Page: 49 / 107 ## 8
  - [ ] `set_up_manual_supra_n` (score=4.9257, device=SUPRA N, type=SOP)
    > ```markdown Confidential I 6) ZERO Calibration <!-- Table (68, 69, 938, 319) --> \begin{tabular}{|l|l|l|} \hline \textbf
  - [ ] `set_up_manual_supra_np` (score=4.8749, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I 6) ZERO Calibration <!-- Table (58, 69, 928, 339) --> \begin{tabular}{|l|l|l|} \hline \textbf
  - [ ] `40046755` (score=4.8047, device=SUPRA Nm, type=myservice)
    > -. 고온 방지 Cover 장착
-. Temp Limit Setting
-. 정기 PM 중이라 Baffle, Focus Adaptor 장착방법 고객 및 PM업체 공유
-. 고객측에서 Applicator Tube 교체

#### q_id: `A-amb029`
- **Question**: Source Power Matching 불안정 원인과 조치는?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-24):
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3000qc` (score=5.7051, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM 3000,3100QC SOURCE IGNITION WINDOW CHECK Global SOP No: Revision No: 2 P
  - [ ] `precia_all_trouble_shooting_guide_rf_matching_alarm` (score=5.6191, device=GENEVA XP, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [RF Matching Alarm] Use this guide to diagnose problems with the [RF Matching Alarm]. It descri
  - [ ] `supra_xp_pm_trouble_shooting_guide_trace_prism_abnormal` (score=5.2764, device=SUPRA XP, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace PRISM abnormal] Use this guide to diagnose problems with the [Trace PRISM abnormal]. It 
  - [ ] `global_sop_supra_xp_all_pm_prism_source` (score=4.9882, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM SOURCE | Global SOP No: | | | --- | --- | | Revision No: 2 | | | Page: 5
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3100qc` (score=4.9814, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM SOURCE 3100QC Global SOP No: Revision No: 1 Page: 5/72 ## 3. 사고 사례 ### 1
  - [ ] `precia_all_trouble_shooting_guide_pin_motor_abnormal` (score=4.6749, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # PRECIA Trouble Shooting Guide [Pin Motor Abnormal] Confidential II | Alarm Code | LED 점멸 횟수 | Alarm 종류 | 원
  - [ ] `precia_pm_trouble_shooting_guide_rf_power_abnormal` (score=4.6318, device=PRECIA, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [RF Power Abnormal] Use this guide to diagnose problems with the [RF Power Abnormal]. It descri
  - [ ] `global_sop_geneva_xp_rep_pm_harmonic_drive` (score=4.6281, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Harmonic drive | Global SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Pag
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=4.6118, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier | Global SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Pag
  - [ ] `global_sop_geneva_xp_rep_pm_adapter_ring` (score=4.5992, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Adapter ring Global SOP No: 0 Revision No: 0 Page: 4 / 30 ## 3. 사고 사례 ### 1) 화
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=4.5948, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 66 / 105 ## 사고 사례 ##
  - [ ] `global_sop_supra_n_series_all_pm_epd` (score=4.5944, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_EPD Global SOP No : Revision No: 3 Page: 3/49 ## 3. 사고 사례 ### 1) 화상 재해의 정
  - [ ] `global_sop_supra_n_series_all_pm_hook_lifter_servo_motor` (score=4.5827, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_HOOK LIFTER ## SERVO MOTOR Global SOP No : Revision No: 2 Page : 3 / 106 
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=4.5531, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 4/47 ## 3. 사고 사례 ### 1) 화상 재해의 
  - [ ] `set_up_manual_supra_n` (score=4.5178, device=SUPRA N, type=SOP)
    > ```markdown Confidential I # 11. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 11.1. Common List ### 11.1.2 Temp Limit Controller Setti
  - [ ] `global_sop_supra_n_series_all_pm_dual_epd` (score=4.4884, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_DUAL EPD Global SOP No: Revision No: 4 Page: 3/41 ## 3. 사고 사례 ### 1) 화상의 
  - [ ] `global_sop_supra_xp_all_pm_cip_chamber` (score=4.4815, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_CIP CHAMBER | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page: 5 
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=4.4754, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No: 13 Page: 5/75 ## 3. 사고 사례 #
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock_o_ring` (score=4.4714, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_LOAD LOCK O-RING | Global SOP No: | S-KG-R019-R0 | | --- | --- | | Revision No
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=4.4636, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 3/124 ## 3. 사고 사례 ### 1) 화상의 정의 불
  - [ ] `global_sop_supra_xp_all_pm_dual_epd` (score=4.4622, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_DUAL EPD Global SOP No: Revision No: 1 Page: 3/40 ## 3. 사고 사례 ### 1) 화상의 정의 불이
  - [ ] `40080973` (score=4.4276, device=INTEGER IVr, type=myservice)
    > -. 현상
-> 연속 Run 중 TM TOP LID OPEN INTERLOCK Alarm 7/28 03:27:48 1회 발생
-> Alarm 관련 Safety Module 0-2, 0-3 확인 시 LED 모두 Off
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=4.4231, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_EFEM_ROBOT TEACHING Global SOP No: Revision No: 3 Page: 30 / 126 | Flow | Proc
  - [ ] `global_sop_supra_n_series_all_pm_gas_feed_through` (score=4.419, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS FEED THROUGH Global SOP No: Revision No: 4 Page: 4 / 18 ## 3. 사고 사례 #

#### q_id: `A-amb030`
- **Question**: Chamber Cleaning 후 Seasoning Recipe 설정 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, INTEGER_PLUS, PRECIA]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-12):
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=6.26, device=ZEDIUS XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) Seasoning | a. [START]를 누른다. | | | | | | | 14) Seas
  - [ ] `set_up_manual_ecolite_3000` (score=6.0583, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 14. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) Season
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=5.9914, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `set_up_manual_supra_xq` (score=5.9514, device=SUPRA XQ, type=SOP)
    > ```markdown # 13. Process Confirm (※환경안전 보호구: 안전모, 안전화)) ## 13-1. Aging Test | Picture | Description | Tool & Spec | | :
  - [ ] `set_up_manual_supra_nm` (score=5.7714, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 14. Process Confirm (※환경안전 보호구 : 안전모, 안전화) ## 14.1 Aging Test | Picture | Description | Too
  - [ ] `set_up_manual_supra_vm` (score=5.7665, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 14. Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) Seasoning | a. [START]
  - [ ] `set_up_manual_ecolite_ii_400` (score=5.7519, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 14. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) Season
  - [ ] `set_up_manual_ecolite_2000` (score=5.7503, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 13. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) Season
  - [ ] `set_up_manual_supra_n` (score=5.6766, device=SUPRA N, type=SOP)
    > Confidential 1 | 11) Seasoning | a. 만들어 두었던 Recipe를 선택한다. | | | :--- | :--- | :--- | | ![](https://i.imgur.com/1.png) | 
  - [ ] `set_up_manual_supra_np` (score=5.5617, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I | 12) Seasoning | a. Process Loop를 1회 진행하기 위해 [1]을 기입한 후 [OK]를 선택한다. | | | :--- | :--- | :---
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=5.3801, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 30 / 47 ## 10. Work Procedure |
  - [ ] `global_sop_supra_xp_all_tm_multi_port` (score=5.0937, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDISU XP_ALL_TM_MULTI PORT Global SOP No: Revision No: 1 Page: 17 / 24 | Flow | Procedure | To

#### q_id: `A-amb031`
- **Question**: EMO Alarm 발생 후 복구 절차는?
- **Devices**: [SUPRA_N, SUPRA_XP, SUPRA_VPLUS, INTEGER_PLUS, PRECIA]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-16):
  - [ ] `set_up_manual_integer_plus` (score=5.474, device=INTEGER plus, type=SOP)
    > | 5) AM ELT Box to PM ELT Box EMO Cable Connection | 1) AM ELT Box to PM ELT Box EMO Cable Connection을 진행한다. | Cable Tie
  - [ ] `40043342` (score=5.422, device=SUPRA V, type=myservice)
    > -. Deadzone Patch 되어있음(Overhaul된 Robot 교체품)
-. ZEUS Eng'r(김한울, 박성관) 입실 후 현상 확인
-> Error History 상 EMO 관련 문제는 맞지만 Pendant
  - [ ] `set_up_manual_ecolite_3000` (score=5.2135, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 5. Power Turn On (EMO 확인) | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) EFEM 측면 EMO 
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=5.1996, device=ZEDIUS XP, type=set_up_manual)
    > ```markdown # 5.2 EMO Check | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) EMO(Emergency Machin Off
  - [ ] `set_up_manual_ecolite_ii_400` (score=5.163, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 5. Power Turn On_EMO 확인 | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) EFEM 측면 EMO | 
  - [ ] `set_up_manual_supra_xq` (score=5.1064, device=SUPRA XQ, type=SOP)
    > | 4) PM EMO | a. 각 PM의 Door위에 위치한 EMO 스위치를 OFF시킨 후, 설비 Power를 On하여 Power인가 여부를 확인. | | | :--- | :--- | :--- | | 5) Rack 
  - [ ] `set_up_manual_supra_vm` (score=5.0682, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 6. Power Turn On (EMO 확인) | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) EFEM 측면 EMO 
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=5.046, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 3) PM EMO Check | a. 각 PM 의 위치한 EMO Switch (3ea) Off 시킨
  - [ ] `set_up_manual_supra_n` (score=4.9675, device=SUPRA N, type=SOP)
    > ```markdown # 5. Power Turn On (※환경안전 보호구 : 안전모, 안전화) ## 5.1 EMO Check | Picture | Description | Tool & Spec | | :--- | 
  - [ ] `set_up_manual_supra_np` (score=4.9675, device=SUPRA Np, type=set_up_manual)
    > ```markdown # 5. Power Turn On (※환경안전 보호구: 안전모, 안전화) ## 5.1 EMO Check | Picture | Description | Tool & Spec | | :--- | :
  - [ ] `set_up_manual_supra_nm` (score=4.9073, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 5. Power Turn On (※환경안전 보호구 : 안전모, 안전화) ## 5.1 EMO Check | Picture | Description | Tool & S
  - [ ] `set_up_manual_ecolite_2000` (score=4.7666, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 5. Power Turn On (EQ Power Turn On & EMO 확인) | Picture | Description | Tool & Spec | | :--- | :--- | :--- 
  - [ ] `set_up_manual_precia` (score=4.605, device=PRECIA, type=set_up_manual)
    > ```markdown # 6. Power Turn On (환경안전 보호구: 안전모, 안전화) ## 6.4 EMO Check | Picture | Description | Tool & Spec | | :--- | :-
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=4.5137, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_TM_CONTROLLER ## BATTERY_Replacement Global SOP No : Revision No: 6 Page : 94 / 
  - [ ] `global_sop_integer_plus_all_pm_gas_line` (score=4.4897, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_GAS LINE Global_SOP No: Revision No: 0 Page: 3/75 ## 3. 사고 사례 ### 1) 협착 재해의
  - [ ] `global_sop_integer_plus_all_am_baffle` (score=4.4294, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_ INTEGER plus_ALL_AM_BAFFLE Global_SOP No: Revision No: 1 Page: 3 / 18 ## 3. 사고 사례 ### 1) 협착 재해

#### q_id: `A-amb032`
- **Question**: Chamber Pressure High Alarm 원인 분석 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-21):
  - [ ] `40043020` (score=6.4276, device=SUPRA Vplus, type=myservice)
    > -. 원인 분석 중
  - [ ] `set_up_manual_supra_np` (score=5.5491, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I Analog Output Tap은 조작하지 않는다. # 12. TTTM (※환경안전 보호구: 안전모, 안전화) ## 12.5 Device Net Calibration 
  - [ ] `set_up_manual_supra_xq` (score=5.0842, device=SUPRA XQ, type=SOP)
    > | | | Slot Valve2 Open | | | |---|---|---|---|---| | | | Chamber Pressure ATM State | | | | | | Gas Box Door Opened | | 
  - [ ] `set_up_manual_supra_nm` (score=4.9007, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.4 Device Net Calibration _ PSK Board ### 12.4.1 ATM P
  - [ ] `set_up_manual_ecolite_2000` (score=4.8151, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 12. Customer Certification_Interlock Check _ H/W | Module | Action | A Prior Condition | 1st Check | 2nd C
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=4.7021, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `set_up_manual_supra_n` (score=4.6342, device=SUPRA N, type=SOP)
    > ```markdown Confidential 1 # 6) Analog IO Calibration <!-- Image (71, 70, 370, 285) --> a. Analog Input Calibration은 프로그
  - [ ] `global_sop_integer_plus_all_pm_vacuum_line` (score=4.6089, device=INTEGER XP, type=SOP)
    > ```markdown # Global SOP_INTEGER plus _REP_PM_Vacuum Line Global SOP No: Revision No: Page : 129 / 133 | Flow | Procedur
  - [ ] `global_sop_geneva_xp_sw_efem_log_backup` (score=4.5616, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVAxp_SW_EFEM_Log backup Global SOP No: Revision No: 0 Page: 5/13 ## 6. Flow Chart Start ↓ 1. SOP 및 안전사항
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock_o_ring` (score=4.5176, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_LOAD LOCK O-RING Global SOP No: S-KG-R019-R0 Revision No: 0 Page: 11 / 30 | Fl
  - [ ] `set_up_manual_supra_vm` (score=4.489, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 17. Interlock Check _ H/W | Module | Action | A Prior Condition | 1st Check | 2nd Check | | :--- | :--- | 
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pressure_switch` (score=4.475, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVAxp_REP_PM_Vacuum Pressure switch SOP No: 0 Revision No: 0 Page: 7 / 18 ## 8. 필요 Tool | Name | Teflon 
  - [ ] `global_sop_supra_n_series_all_tm_ffu_mcu` (score=4.4132, device=SUPRA N, type=SOP)
    > # Global SOP_SUPRA N series_ADJ_TM_FFU_MCU Global SOP No : Revision No: 1 Page : 14 / 57 ## 5. Flow Chart Start -> 1. Gl
  - [ ] `set_up_manual_ecolite_ii_400` (score=4.3817, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 13. Customer Certification_Interlock Check_H/W | Module | Action | A Prior Condition | 1st Check | 2nd Che
  - [ ] `global_sop_geneva_xp_rep_pm_pressure_switch` (score=4.3797, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Pressure switch | SOP No: 0 | | |---|---| | Revision No: 0 | | | Page: 7/16 | 
  - [ ] `global_sop_geneva_xp_rep_pm_support_pin` (score=4.3399, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVA xp_REP_PM_Support pin Global SOP No: 0 Revision No: 1 Page: 16 / 25 ## 10. Work Procedure | Flow | P
  - [ ] `global_sop_geneva_xp_adj_pm_pin_alignment` (score=4.3127, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Pin Alignment Global SOP No: S-KG-A003-R0 Revision No: 0 Page: 11 / 25 ## 10. 
  - [ ] `global_sop_precia_all_pm_chuck` (score=4.3049, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_CHUCK Chiller Line Global SOP No : Revision No: 1 Page :110 / 132 ## 6. Work Proc
  - [ ] `precia_pm_trouble_shooting_guide_process_stable_time_out_alarm` (score=4.2799, device=PRECIA, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Process Stable Time out Alarm] Use this guide to diagnose problems with the [Process Stable Ti
  - [ ] `global_sop_integer_plus_all_pm_slot_valve` (score=4.2798, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_CLN_PM_SLOT VALVE Global SOP No : Revision No : 2 Page : 35 / 39 | Flow | Procedur
  - [ ] `global_sop_precia_all_tm_pressure_switch` (score=4.2793, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_TM_PRESSURE SWITCH Global SOP No : Revision No: 0 Page: 24 / 27 | Flow | Procedure |

#### q_id: `A-amb033`
- **Question**: Interlock Reset 후에도 설비 Run 안 되는 경우 조치는?
- **Devices**: [SUPRA_N, SUPRA_XP, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-27):
  - [ ] `global_sop_supra_n_series_all_pm_gas_spring` (score=7.8874, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS SPRING Global SOP No: Revision No: 4 Page: 3/22 ## 3. 사고 사례 ### 1) 화상
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=7.7131, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_AM_PIRANI GAUGE Global SOP No: Revision No: Page: 24 / 135 ## 3. 사고 사례 ### 1) 
  - [ ] `global_sop_geneva_xp_adj_pm_chuck_temp_calibration` (score=7.6475, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Chuck temp calibration Global SOP No: Revision No: 0 Page: 3/21 ## 3. 재해 방지 대책
  - [ ] `set_up_manual_precia` (score=7.5442, device=PRECIA, type=set_up_manual)
    > # 공통사항 1. 보호구를 사용하지 않아도 근로자가 유해/위험작업으로부터 보호를 받을 수 있도록 설비 개선 등 필요 조치를 진행한다. 2. 필요 조치를 이행하였음에도 유해 / 위험 요인은 제거하기가 어려울 때. 제한
  - [ ] `global_sop_supra_n_all_pm_chamber_open_interlock_change` (score=7.1482, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_MODIFY_PM_CHAMBER_OPEN_INTERLOCK_CHANGE Global SOP No: Revision No: 3 Page: 3/24 ## 3. 
  - [ ] `global_sop_supra_nm_all_pm_microwave_local_interface` (score=7.1475, device=SUPRA Nm, type=SOP)
    > ```markdown # Global SOP_SUPRA Nm_ALL_PM_MICROWAVE_LOCAL INTERFACE Global_SOP No: Revision No: Page : 3/21 ## 3. 사고 사례 #
  - [ ] `40064646` (score=7.1444, device=SUPRA Vplus, type=myservice)
    > -. Temp up 안되는 상태로 점검 요청
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=7.0625, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 2) Load | a. Load를 클릭한다. | | | | | | | 3) Set-up Recipe
  - [ ] `global_sop_precia_all_tm_branch_tap` (score=6.9003, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_TM_BRANCH TAP Global SOP No : Revision No: 0 Page : 3/23 ## 3. 사고 사례 ### 1) 감전의 정의 -
  - [ ] `set_up_manual_ecolite_3000` (score=6.8616, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 4. Cable Hook Up_Module Connection(긴급 차단 Valve Air Tube 연결) | Picture | Description | Tool & Spec | | :---
  - [ ] `global_sop_supra_n_series_all_tm_ctc` (score=6.8534, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_CTC REPLACEMENT Global SOP No : Revision No: 6 Page: 17/81 | Flow | Proce
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=6.843, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_TM_CONTROLLER ## BATTERY_Replacement Global SOP No : Revision No: 6 Page :101 / 
  - [ ] `global_sop_supra_n_series_all_pm_dual_epd` (score=6.8145, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_DUAL EPD Global SOP No: Revision No: 4 Page: 3/41 ## 3. 사고 사례 ### 1) 화상의 
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=6.8097, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 10 / 105 ## 사고 사례 ##
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=6.8083, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No: 13 Page: 5/75 ## 3. 사고 사례 #
  - [ ] `global_sop_supra_xp_all_pm_cip_chamber` (score=6.8067, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_CIP CHAMBER | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page: 5 
  - [ ] `global_sop_supra_xp_all_pm_prism_source` (score=6.8016, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM SOURCE | Global SOP No: | | | --- | --- | | Revision No: 2 | | | Page: 5
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3000qc` (score=6.7879, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM SOURCE 3000QC Global SOP No: Revision No : 2 Page : 5/82 ## 3. 사고 사례 ###
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3100qc` (score=6.7847, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM SOURCE 3100QC Global SOP No: Revision No: 1 Page: 5/72 ## 3. 사고 사례 ### 1
  - [ ] `set_up_manual_supra_n` (score=6.7781, device=SUPRA N, type=SOP)
    > ```markdown Confidential 1 # 13. Customer Certification (※환경안전 보호구 : 안전모, 안전화) ## 13.1 Interlock Check (H/W, S/W) | Pict
  - [ ] `global_sop_integer_plus_all_gr_shut_off_valve` (score=6.7657, device=INTEGER plus, type=SOP)
    > ```markdown # [Global_SOP] INTEGER plus_ALL_GENERATOR SHUT OFF V/V Global SOP No: Revision No: 0 Page: 3/28 ## 3. 사고 사례 
  - [ ] `global_sop_integer_plus_all_am_devicenet_board` (score=6.763, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_AM_DEVICENET BOARD | Global SOP No: | | | --- | --- | | Revision No: 0 | | | P
  - [ ] `global_sop_integer_plus_all_tm_safety_controller` (score=6.7578, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_SAFETY CONTROLLER Global_SOP No: Revision No: 0 Page: 3/19 ## 3. 사고 사례 ### 
  - [ ] `global_sop_integer_plus_all_tm_devicenet_board` (score=6.7567, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP _ INTEGER plus _ ALL _ TM _ DEVICENET BOARD Global_SOP No: Revision No: 3 Page: 3/21 ## 3. 사고 사
  - [ ] `global_sop_integer_plus_all_pm_safety_controller` (score=6.7552, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_SAFETY # CONTROLLER Global_SOP No: Revision No: 0 Page: 3/19 ## 3. 사고 사례 ##
  - [ ] `global_sop_precia_all_tm_robot` (score=6.7522, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_TM_ROBOT | Global SOP No: | | |---|---| | Revision No: 0 | | | Page: 4/56 | | ## 3. 
  - [ ] `global_sop_supra_n_series_all_tm_devicenet_board` (score=6.7495, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_DEVICENET BOARD | Global SOP No: | | | --- | --- | | Revision No: 2 | | |

#### q_id: `A-amb034`
- **Question**: Temperature Over Alarm 발생 시 Heater 점검 순서는?
- **Devices**: [SUPRA_N, SUPRA_VPLUS, SUPRA_XP, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-26):
  - [ ] `supra_n_all_trouble_shooting_guide_trace_temperature_abnormal` (score=5.8711, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Temperature abnormal] Confidential II | | A-6. Buffer Stage | ▶ Wafer positi
  - [ ] `supra_xp_pm_trouble_shooting_guide_trace_chuck_abnormal` (score=5.6146, device=SUPRA XP, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Chuck Abnormal] Confidential II | Temp Stable Time Out Alarm | B -1. Heater Chuck 
  - [ ] `40035345` (score=5.4853, device=SUPRA V, type=myservice)
    > -. LOG 확인 시 FFU 점검 이전 부터 EFEM TO CTC Communication Alarm 발생
-> EFEM TO CTC Communication LOG 끊김 확인
-> 고객 Inform 완료
-. PM
  - [ ] `set_up_manual_supra_xq` (score=5.3883, device=SUPRA XQ, type=SOP)
    > | | | Water Leak State | | | |---|---|---|---|---| | | | Source Cover Open State | | | | | | Gas Box Exh. Differ. Pressu
  - [ ] `40079239` (score=5.1235, device=SUPRA Vplus, type=myservice)
    > -. All PM Pump Down, Temp 정상 Reading X
-> Temp CTR 정상 확인
-. Rack 확인 시 ELCB0-1 Trip 확인
-. ELCB DVM Check 시 Input 220V 정상

  - [ ] `global_sop_geneva_xp_adj_pm_chuck_temp_calibration` (score=4.8487, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Chuck temp calibration Global SOP No: Revision No: 0 Page: 3/21 ## 3. 재해 방지 대책
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=4.8069, device=GENEVA XP, type=set_up_manual)
    > | | Elbow Heater#1 | Limit Temp | Alarm & Heater#1 Power Off | | | | :--- | :--- | :--- | :--- | :--- | :--- | | | Elbow
  - [ ] `set_up_manual_ecolite_2000` (score=4.7708, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 8. Part Installation | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) Ceramic Parts<br>
  - [ ] `global_sop_geneva_xp_rep_pm_elbow_heater` (score=4.6713, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Elbow heater SOP No: 0 Revision No: 0 Page: 12/22 ## 10. Work Procedure | Flow
  - [ ] `global_sop_precia_all_pm_rf_bias_matcher` (score=4.6314, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_RF ROD Global SOP No: Revision No: 1 Page: 32 / 55 ## 6. Work Procedure | Flow | 
  - [ ] `40044046` (score=4.5891, device=SUPRA Vplus, type=myservice)
    > -. EPAGH4 PM1 Leak Rate Over Alarm
-> Leak Rate : 16mTorr/min
-> 고객측 확인 시 CH1 Bellows 확인으로 교체 요청
  - [ ] `global_sop_geneva_xp_rep_pm_heat_exchanger` (score=4.5708, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Heat exchanger Global SOP No: Revision No: 0 Page: 3 / 14 ## 3. 사고 사례 ### 1. 협착
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pressure_switch` (score=4.569, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Vacuum Pressure switch | SOP No: 0 | | |---|---| | Revision No: 0 | | | Page: 3
  - [ ] `global_sop_geneva_xp_rep_pm_pressure_switch` (score=4.5528, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Pressure switch | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/1
  - [ ] `global_sop_precia_all_pm_chuck` (score=4.5447, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_CHUCK Chiller Line Global SOP No : Revision No: 1 Page :128 / 132 ## 6. Work Proc
  - [ ] `global_sop_geneva_xp_rep_efem_load_port` (score=4.5437, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Load port | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/18 | | 
  - [ ] `global_sop_geneva_xp_adj_efem_efem_robot_leveling` (score=4.5412, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_EFEM_EFEM Robot leveling Global SOP No: Revision No: 0 Page: 3/15 ## 3. 사고 사례 ###
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pump` (score=4.5365, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Vacuum pump | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/21 | 
  - [ ] `global_sop_geneva_xp_rep_pm_disc` (score=4.5312, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/31 | | ## 3.
  - [ ] `global_sop_geneva_xp_rep_pm_chamber_lock` (score=4.5289, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Chamber lock Global SOP No: Revision No: 0 Page: 3 / 14 ## 3. 사고 사례 ### 1. 협착 위
  - [ ] `global_sop_geneva_xp_rep_pm_mfc` (score=4.5265, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_MFC | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/18 | | ## 3. 
  - [ ] `global_sop_geneva_xp_all_8계통_check` (score=4.5149, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA XP_8계통_Check sheet | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/46 | | ## 3. 사고 사례 
  - [ ] `global_sop_geneva_xp_adj_post_align_application` (score=4.5088, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_Post align application Global SOP No: 0 Revision No: 1 Page: 3/20 ## 3. 재해 방지 대책 1) 협
  - [ ] `set_up_manual_supra_vm` (score=4.5012, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 2. Undocking 및 Module 이동 | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Module 반입 순서 
  - [ ] `global_sop_geneva_xp_rep_pm_insulation_heater` (score=4.4849, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Insulation heater | SOP No: 0 | | | | | :--- | :--- | :--- | :--- | | Revision
  - [ ] `global_sop_integer_plus_all_pm_baffle_heater` (score=4.4718, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_BAFFLE HEATER Global SOP No : Revision No : 2 Page : 14 / 22 | Flow | Proce

#### q_id: `A-amb035`
- **Question**: Vacuum Leak Alarm 발생 시 1차 점검 항목은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, GENEVA_XP]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-23):
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=5.7145, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 14. Customer certification | Picture | Description | Spec | Check | Result | | :--- | :--- | :--- | :--- |
  - [ ] `global_sop_supra_n_series_all_pm_heater_chuck` (score=5.6596, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_HEATER CHUCK Global SOP No : Revision No: 2 Page: 27 / 40 | Flow | Proced
  - [ ] `40038715` (score=5.2012, device=SUPRA Vplus, type=myservice)
    > -. EFEM Robot Upper Endeffector Rep
-. Upper Arm leak 정상 확인
-. Lower Leak 확인
-> 초당 1kpa 수준 Leak 발생
-. Aging 1Lot 정상 확인
-
  - [ ] `set_up_manual_precia` (score=5.0447, device=PRECIA, type=set_up_manual)
    > ```markdown # 4. Docking (※환경안전 보호구: 안전모, 안전화) ## 4.2 EFEM - TM Docking | Picture | Description | Tool & Spec | | :--- |
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=4.8919, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `40035345` (score=4.7729, device=SUPRA V, type=myservice)
    > -. LOG 확인 시 FFU 점검 이전 부터 EFEM TO CTC Communication Alarm 발생
-> EFEM TO CTC Communication LOG 끊김 확인
-> 고객 Inform 완료
-. PM
  - [ ] `set_up_manual_supra_xq` (score=4.738, device=SUPRA XQ, type=SOP)
    > | 55 | PM | DN131 | D-NET | □Y □N | 95 | PM | DN131 | D-NET | □Y □N | |---|---|---|---|---|---|---|---|---|---| | 56 | P
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=4.6726, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_CLN_AM_VACUUM LINE Global SOP No: Revision No: Page : 111 / 135 | Flow | Procedure
  - [ ] `global_sop_integer_plus_all_pm_swap_kit` (score=4.6675, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_SWAP KIT Global SOP No: Revision No: 3 Page: 32 / 37 | Flow | Procedure | T
  - [ ] `40064811` (score=4.6243, device=TIGMA Vplus, type=myservice)
    > -. PM1 Slow vacuum time out alarm
  - [ ] `40044046` (score=4.5763, device=SUPRA Vplus, type=myservice)
    > -. EPAGH4 PM1 Leak Rate Over Alarm
-> Leak Rate : 16mTorr/min
-> 고객측 확인 시 CH1 Bellows 확인으로 교체 요청
  - [ ] `global_sop_integer_plus_all_ll_cassette` (score=4.509, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER PLUS_REP_LL_REFLECTOR Global SOP No : Revision No: 1 Page: 82/86 | Flow | Procedure | T
  - [ ] `supra_n_all_trouble_shooting_guide_trace_vacuum_vent_abnormal` (score=4.4746, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Vacuum-Vent abnormal] Use this guide to diagnose problems with the [Trace Va
  - [ ] `set_up_manual_supra_n` (score=4.468, device=SUPRA N, type=SOP)
    > ```markdown # 4. Cable Hook Up (※환경안전 보호구 : 안전모, 안전화, 안전대, 보안경 보호가운, 헤드랜턴) ## 4.4 Sub Unit Cable Hook Up | Picture | Des
  - [ ] `set_up_manual_supra_np` (score=4.4409, device=SUPRA Np, type=set_up_manual)
    > | 9) SUB2 I/F Panel - PM Cable Hook up | a. Sub Unit2 의 PM 방향 Interface Panel Inner Cable Hook up 을 진행한다. b. 장착되는 Cable 
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pump` (score=4.4226, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Vacuum pump SOP No: 0 Revision No: 0 Page: 17 / 21 ## 10. Work Procedure | Flo
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pressure_switch` (score=4.4004, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Vacuum Pressure switch | SOP No: 0 | | |---|---| | Revision No: 0 | | | Page: 3
  - [ ] `40039537` (score=4.3884, device=SUPRA III, type=myservice)
    > -. Log Check
-> Mapping arm Open Alarm 없음
-. H/W 동작 Test 시 정상 확인
-. 고객측 Monitoring 후 재발 시 재점검
  - [ ] `global_sop_integer_plus_all_am_view_quartz` (score=4.3859, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_VIEW QUARTZ Global SOP No: Revision No: 2 Page: 16 / 20 | Flow | Procedure 
  - [ ] `set_up_manual_supra_nm` (score=4.3816, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 4. Cable Hook Up (※환경안전 보호구 : 안전모, 안전화, 안전대, 보안경 보호가운, 헤드랜턴) ## 4.4 Sub Unit Cable Hook Up 
  - [ ] `global_sop_integer_plus_all_ll_vacuum_line` (score=4.3635, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_LL_ISOLATION ## VALVE Global SOP No: Revision No: Page: 61 / 77 | Flow | Proce
  - [ ] `global_sop_integer_plus_all_tm_vacuum_line` (score=4.3607, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_ALL_TM_VACUUM LINE Global SOP No: Revision No: 1 Page: 26 / 26 ## 8. Appendix - NTEGER plus - 
  - [ ] `global_sop_integer_plus_all_ll_lifter_assy` (score=4.3534, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_LL_BUSH, BELLOWS, SHAFT Global SOP No : Revision No: 0 Page: 33/81 | Flow | Pr

#### q_id: `A-amb036`
- **Question**: Endpoint 미검출 시 Recipe 파라미터 확인 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-19):
  - [ ] `set_up_manual_precia` (score=5.8879, device=PRECIA, type=set_up_manual)
    > | | f. '검사' 버튼 Click | | |---|---|---| | | | | | 5. Edge Parameter Setting | a. Edge 검출 영역 최적화 Parameter 값 확인 | | | | | 
  - [ ] `set_up_manual_supra_nm` (score=5.8782, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.2 EFEM Robot Leveling & Loadport Teaching | Picture
  - [ ] `set_up_manual_supra_np` (score=5.6159, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I | 2) Jig Assy 조립도 확인 | a. Jig Assy 조립도를 확인하여 다음과 같은 방법으로 Jig Assy가 설치되어야 한다. | | | :--- | :--
  - [ ] `global_sop_geneva_xp_adj_pm_chuck_motor` (score=5.3138, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_CHUCK MOTOR Global SOP No: 0 Revision No: 0 Page: 22 / 22 ## 13. Appendix 원점 (
  - [ ] `set_up_manual_supra_n` (score=5.298, device=SUPRA N, type=SOP)
    > ```markdown Confidential I 36) LP2,3 Teaching <!-- Image (70, 70, 290, 226) --> a. LP1 Teaching 과 동일한 방법으로 LP2,3 를 진행하여 
  - [ ] `set_up_manual_ecolite_3000` (score=4.8254, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 14. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Making 
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=4.7248, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `set_up_manual_ecolite_2000` (score=4.677, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 13. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Making 
  - [ ] `set_up_manual_ecolite_ii_400` (score=4.6738, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 14. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Making 
  - [ ] `set_up_manual_supra_xq` (score=4.6703, device=SUPRA XQ, type=SOP)
    > ```markdown # 13. Process Confirm (※환경안전 보호구: 안전모, 안전화) ## 13-1. Aging Test | Picture | Description | Tool & Spec | | :-
  - [ ] `set_up_manual_supra_vm` (score=4.6531, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 14. Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Making Recipe | a. [Ins
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=4.6058, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_LM GUIDE ## REPLACEMENT & GREASE INJECTION Global SOP No: Revision No: 1 Page:
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=4.5775, device=ZEDIUS XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Making Recipe<br><img src="image_url" alt="Image 1">
  - [ ] `precia_all_trouble_shooting_guide_pin_motor_abnormal` (score=4.564, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # PRECIA Trouble Shooting Guide [Pin Motor Abnormal] Confidential II | Alarm Code | LED 점멸 횟수 | Alarm 종류 | 원
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=4.4609, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM ROBOT ASSY REPLACEMENT Global SOP No : Revision No: 6 Page: 49 / 107 ## 8
  - [ ] `global_sop_geneva_xp_adj_post_align_application` (score=4.282, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_Post align application Global SOP No: 0 Revision No: 1 Page: 15 / 20 | Flow | Procedu
  - [ ] `global_sop_precia_all_efem_ctc` (score=4.2686, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_EFEM_CTC Global SOP No: Revision No: 1 Page: 24/51 | Flow | Procedure | Tool & Spec 
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=4.2601, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 4) PM Formic leak 확인 | a. Recipe setup 시 PM 에 장착되어 있는 F
  - [ ] `40043020` (score=4.2466, device=SUPRA Vplus, type=myservice)
    > -. F6v Recipe 진행시 타 CH Peak 50~70수준. PM3 CH1의 경우 20~30수준
-. 설비 확인시 PM3 CH1 View Quartz Hume 확인
-> Clean 및 CH1<->CH2 Swap

#### q_id: `A-amb037`
- **Question**: Etch Rate Drift 발생 시 원인 분석 절차는?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-30):
  - [ ] `40043020` (score=6.3956, device=SUPRA Vplus, type=myservice)
    > -. 원인 분석 중
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pressure_switch` (score=4.6829, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Vacuum Pressure switch | SOP No: 0 | | |---|---| | Revision No: 0 | | | Page: 3
  - [ ] `global_sop_geneva_xp_rep_pm_pressure_switch` (score=4.6758, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Pressure switch | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/1
  - [ ] `global_sop_geneva_xp_adj_efem_efem_robot_leveling` (score=4.674, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_EFEM_EFEM Robot leveling Global SOP No: Revision No: 0 Page: 3/15 ## 3. 사고 사례 ###
  - [ ] `global_sop_geneva_xp_rep_efem_load_port` (score=4.6724, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Load port | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/18 | | 
  - [ ] `global_sop_geneva_xp_rep_pm_mfc` (score=4.6719, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_MFC | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/18 | | ## 3. 
  - [ ] `global_sop_geneva_xp_rep_pm_disc` (score=4.67, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/31 | | ## 3.
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pump` (score=4.6696, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Vacuum pump | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/21 | 
  - [ ] `global_sop_geneva_xp_rep_pm_heat_exchanger` (score=4.6664, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Heat exchanger Global SOP No: Revision No: 0 Page: 3 / 14 ## 3. 사고 사례 ### 1. 협착
  - [ ] `global_sop_geneva_xp_rep_pm_chamber_lock` (score=4.6615, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Chamber lock Global SOP No: Revision No: 0 Page: 3 / 14 ## 3. 사고 사례 ### 1. 협착 위
  - [ ] `global_sop_geneva_xp_all_8계통_check` (score=4.6559, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA XP_8계통_Check sheet | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/46 | | ## 3. 사고 사례 
  - [ ] `global_sop_geneva_xp_adj_post_align_application` (score=4.6518, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_Post align application Global SOP No: 0 Revision No: 1 Page: 3/20 ## 3. 재해 방지 대책 1) 협
  - [ ] `precia_all_trouble_shooting_guide_pin_motor_abnormal` (score=4.6299, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # PRECIA Trouble Shooting Guide [Pin Motor Abnormal] Confidential II | Failure symptoms | Check point | Key 
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock` (score=4.5272, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Load lock | SOP No: 0 | | | | | :--- | :--- | :--- | :--- | | Revision No: 1 |
  - [ ] `set_up_manual_precia` (score=4.4736, device=PRECIA, type=set_up_manual)
    > | | f. '검사' 버튼 Click | | |---|---|---| | | | | | 5. Edge Parameter Setting | a. Edge 검출 영역 최적화 Parameter 값 확인 | | | | | 
  - [ ] `precia_pm_trouble_shooting_guide_centering_abnormal` (score=4.4193, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Centering Abnormal] Confidential II | Failure symptoms | Check point | Key point |
  - [ ] `supra_xp_pm_trouble_shooting_guide_trace_leak_rate_over` (score=4.4173, device=ZEDIUS XP, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace leak rate over] Use this guide to diagnose problems with the [Trace leak rate over]. It 
  - [ ] `supra_n_all_trouble_shooting_guide_trace_leak_rate_over` (score=4.4173, device=SUPRA N, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace leak rate over] Use this guide to diagnose problems with the [Trace leak rate over]. It 
  - [ ] `global_sop_supra_xp_all_pm_chamber_safety_cover` (score=4.4067, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_SAFETY COVER REPLACEMENT Global SOP No : Revision No : 1 Page : 12 / 18 | Flow
  - [ ] `global_sop_supra_xp_all_tm_mfc` (score=4.4019, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_MFC Global SOP No: Revision No: 0 Page: 16 / 18 | Flow | 절차 | Tool & Spec | | 
  - [ ] `global_sop_geneva_xp_rep_pm_support_pin` (score=4.3988, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Support pin Global SOP No: 0 Revision No: 1 Page: 18 / 25 ## 10. Work Procedur
  - [ ] `global_sop_geneva_xp_adj_pm_chuck_temp_calibration` (score=4.3665, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Chuck temp calibration Global SOP No: Revision No: 0 Page: 3/21 ## 3. 재해 방지 대책
  - [ ] `global_sop_geneva_xp_sw_efem_log_backup` (score=4.323, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_SW_EFEM_Log backup Global SOP No: Revision No: 0 Page: 9/13 ## 10. Work Procedure | Fl
  - [ ] `global_sop_supra_n_series_all_efem_ffu` (score=4.2319, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_EFEM_FFU | Global SOP No: | | | --- | --- | | Revision No: 4 | | | Page: 2/5
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=4.197, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_TM_CONTROLLER ## BATTERY_Replacement Global SOP No : Revision No: 6 Page : 94 / 
  - [ ] `40044046` (score=4.1696, device=SUPRA Vplus, type=myservice)
    > -. EPAGH4 PM1 Leak Rate Over Alarm
-> Leak Rate : 16mTorr/min
-> 고객측 확인 시 CH1 Bellows 확인으로 교체 요청
  - [ ] `global_sop_geneva_xp_rep_bubbler_cabinet_drain_valve` (score=4.1373, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_BUBBLER # CABINET_DRAIN VALVE Global SOP No: S-KG-R034-R0 Revision No: 0 Page: 15
  - [ ] `global_sop_integer_plus_all_ll_mfc` (score=4.1343, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_LL_MFC Global SOP No: Revision No: 1 Page: 3/20 ## 3. 사고 사례 ### 1) 가스 노출 재해의 정
  - [ ] `global_sop_integer_plus_all_ll_sensor_board` (score=4.1307, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_LL_SENSOR BOARD Global SOP No: Revision No: 1 Page: 3/17 ## 3. 사고 사례 ### 1) 가스
  - [ ] `global_sop_geneva_xp_rep_pm_device_net` (score=4.1126, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Device net | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/15

#### q_id: `A-amb038`
- **Question**: Particle Count 증가 시 Chamber 상태 점검 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, SUPRA_VPLUS, PRECIA]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-21):
  - [ ] `set_up_manual_supra_nm` (score=5.0096, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.2 EFEM Robot Leveling & Loadport Teaching | Picture
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=4.872, device=ZEDIUS XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Chamber Open | | | | | a. ATM Recipe로 Run을 진행 전 Cham
  - [ ] `set_up_manual_supra_n` (score=4.8325, device=SUPRA N, type=SOP)
    > ```markdown Confidential I | 36) EFEM Single Teaching | a. 방법은 동일하지만 Teaching 변경 시 TM Robot 이 아닌 EFEM Robot Teaching 을 재
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=4.7161, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 14. Customer certification | Picture | Description | Spec | Check | Result | | :--- | :--- | :--- | :--- |
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=4.7136, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `set_up_manual_ecolite_ii_400` (score=4.6874, device=ECOLITE II 400, type=set_up_manual)
    > | 5)Chamber Open | a) ATM Recipe로 Run을 진행 전 Chamber Open을 실시한다. | | | :--- | :--- | :--- | | | **Caution** | | | | **Cha
  - [ ] `set_up_manual_supra_np` (score=4.6619, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I | 2) Jig Assy 조립도 확인 | a. Jig Assy 조립도를 확인하여 다음과 같은 방법으로 Jig Assy가 설치되어야 한다. | | | :--- | :--
  - [ ] `set_up_manual_supra_xq` (score=4.649, device=SUPRA XQ, type=SOP)
    > ```markdown # 12. TTTM (※환경안전 보호구: 안전모, 안전화) ## 12-6 ATM Transfer | Picture | Description | Tool & Spec | | :--- | :--- 
  - [ ] `global_sop_geneva_xp_rep_pm_chamber_lock` (score=4.5867, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Chamber lock Global SOP No: Revision No: 0 Page: 11 / 14 ## 10. Work Procedure 
  - [ ] `set_up_manual_precia` (score=4.5296, device=PRECIA, type=set_up_manual)
    > ```markdown # 7) 중량물 취급 작업 ## 중량물 취급 시 주의사항 - 발은 어깨 너비로, 허리는 반듯이 세우고 무릎의 힘으로 일어섭니다. <!-- Image (127, 161, 594, 247) --> 
  - [ ] `global_sop_integer_plus_all_pm_pin_motor` (score=4.4455, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_ADJ_PM_LIFT PIN LOCKTITE Global SOP No: Revision No: 5 Page: 34 / 126 | Flow | Procedure | Too
  - [ ] `precia_all_trouble_shooting_particle_trace` (score=4.3667, device=PRECIA, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Particle Trace] Use this guide to diagnose problems with the [Particle Trace]. It describes th
  - [ ] `global_sop_geneva_xp_adj_pm_chuck_temp_calibration` (score=4.3645, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Chuck temp calibration Global SOP No: Revision No: 0 Page: 9/21 ## 10. Work Pr
  - [ ] `global_sop_integer_plus_all_tm_solenoid_valve` (score=4.2675, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_TM_ SOLENOID VALVE Global SOP No: 0 Revision No: 0 Page: 19 / 19 ## 8. Appendi
  - [ ] `global_sop_supra_n_series_all_pm_heater_chuck` (score=4.2599, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_HEATER CHUCK Global SOP No : Revision No: 2 Page: 19/40 | Flow | Procedur
  - [ ] `global_sop_geneva_xp_rep_pm_disc` (score=4.2559, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc SOP No: 0 Revision No: 0 Page: 10 / 31 ## 10. Work Procedure | Flow | Pro
  - [ ] `global_sop_geneva_xp_all_8계통_check` (score=4.2535, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA XP_8계통_Check sheet SOP No: 0 Revision No: 0 Page: 22 / 46 ## 10. Work Procedure | Flow | Procedure 
  - [ ] `integer_plus_all_trouble_shooting_guide_trace_particle_spec_out` (score=4.2337, device=INTEGER plus, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Particle Spec Out] Use this guide to diagnose problems with the [Trace Parti
  - [ ] `global_sop_geneva_xp_cln_bubbler_cabinet_canister` (score=4.1838, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Bubbler cabinet_Canister | Global SOP No: 0 | | | :--- | :--- | | Revision No: 0 
  - [ ] `set_up_manual_supra_vm` (score=4.1792, device=SUPRA Vm, type=set_up_manual)
    > | 6) Chamber Interlock Jig Install | a) Chamber Open시 Interlock이 발생하기 때문에 Interlock Jig를 설치한다. | a. Tool | | :--- | :---
  - [ ] `set_up_manual_ecolite_2000` (score=4.165, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 7. Teaching_ATM Transfer | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5)Chamber Open |

#### q_id: `A-amb039`
- **Question**: Selectivity 변화 시 Process 조건 재설정 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-18):
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=6.0788, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `set_up_manual_supra_nm` (score=5.6379, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.1. Common List ### 12.1.5 Differential Pressure Gauge
  - [ ] `set_up_manual_supra_np` (score=5.6354, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구: 안전모, 안전화) ## 12.1. Common List ### 12.1.5 Differential Pressure Gauge 
  - [ ] `set_up_manual_supra_n` (score=5.4176, device=SUPRA N, type=SOP)
    > ```markdown Confidential I 4) Manometer Setting (Lower Limit) 7 Segment LCD표시기 <!-- Image (68, 70, 375, 231) --> a. Mode
  - [ ] `global_sop_integer_plus_all_pm_manometer` (score=5.3173, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER Plus_REP_MANOMETER Global SOP No: Revision No: 1 Page: 20 / 20 ## 8. Appendix | Flow | 
  - [ ] `global_sop_precia_all_pm_manometer` (score=5.1229, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_MANOMETER Global SOP No: Revision No: 0 Page: 20 / 20 ## 8. Appendix | Flow | Proced
  - [ ] `global_sop_supra_n_series_all_sub_unit_manometer` (score=4.8942, device=SUPRA N, type=SOP)
    > # Global SOP_SUPRA N series_ADJ_SUB UNIT_MANOMETER Global SOP No: Revision No: 0 Page: 31/32 ## 8. Appendix ### 계측모드 Mod
  - [ ] `global_sop_supra_xp_all_tm_multi_port` (score=4.7699, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDISU XP_ALL_TM_MULTI PORT Global SOP No: Revision No: 1 Page: 17 / 24 | Flow | Procedure | To
  - [ ] `global_sop_supra_xp_all_pm_dual_epd` (score=4.762, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_DUAL EPD REPLACEMENT Global SOP No: Revision No: 1 Page: 16 / 40 | Flow | Proc
  - [ ] `set_up_manual_supra_xq` (score=4.6462, device=SUPRA XQ, type=SOP)
    > ```markdown # 13. Process Confirm (※환경안전 보호구: 안전모, 안전화)) ## 13-1. Aging Test | Picture | Description | Tool & Spec | | :
  - [ ] `global_sop_supra_xp_all_efem_ffu` (score=4.5511, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_EFEM_FFU ## CONTROLLER ADJUSTMENT Global SOP No: Revision No : 1 Page : 31 / 60 |
  - [ ] `40036506` (score=4.4715, device=SUPRA Vplus, type=myservice)
    > - Convectron Guage Test
-> Ignition Test 30회 재현안됨
- Leak 강제 생성(Vent Port 부)
-> APC Position 및 Pressure 상승. FCIP Power 변화
  - [ ] `global_sop_geneva_xp_adj_post_align_application` (score=4.4633, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_Post align application Global SOP No: 0 Revision No: 1 Page: 16 / 20 | Flow | Procedu
  - [ ] `set_up_manual_ecolite_3000` (score=4.4562, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 14. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) Season
  - [ ] `global_sop_integer_plus_all_tm_ctc_controller` (score=4.401, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_CTC # CONTROLLER Global SOP No: Revision No: 1 Page: 30 / 51 | Flow | Proce
  - [ ] `global_sop_precia_all_tm_branch_tap` (score=4.3657, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_TM_BRANCH TAP Global SOP No : Revision No: 0 Page: 18 / 23 ## 6. Work Procedure | Fl
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=4.3595, device=ZEDIUS XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) Seasoning | a. [START]를 누른다. | | | | | | | 14) Seas
  - [ ] `global_sop_precia_all_pm_mfc` (score=4.2823, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_MFC Global SOP No : Revision No: 1 Page: 19/23 ## 6. Work Procedure | Flow | Proc

#### q_id: `A-amb040`
- **Question**: Uniformity 불량 시 Showerhead 및 Gas 분배 점검 방법은?
- **Devices**: [SUPRA_N, PRECIA, INTEGER_PLUS, OMNIS]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-18):
  - [ ] `40035542` (score=5.0185, device=TERA21, type=myservice)
    > -. Monitor 분배기 Fail 추정
  - [ ] `set_up_manual_supra_nm` (score=5.0046, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.2 EFEM Robot Leveling & Loadport Teaching | Picture
  - [ ] `global_sop_geneva_xp_rep_pm_분배기` (score=4.9977, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_분배기 Global SOP No: Revision No: 0 Page: 9/13 ## 10. Work Procedure | Flow | Pro
  - [ ] `set_up_manual_supra_np` (score=4.9811, device=SUPRA Np, type=set_up_manual)
    > ```markdown # 3. Part Installation (※환경안전 보호구: 안전모, 안전화) ## 3.16 Portable Rack 장착 | Picture | Description | Tool & Spec 
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=4.724, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `set_up_manual_supra_n` (score=4.6234, device=SUPRA N, type=SOP)
    > ```markdown Confidential I 36) LP2,3 Teaching <!-- Image (70, 70, 290, 226) --> a. LP1 Teaching 과 동일한 방법으로 LP2,3 를 진행하여 
  - [ ] `global_sop_geneva_xp_rep_pm_disc` (score=4.4499, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc SOP No: 0 Revision No: 0 Page: 11/31 ## 10. Work Procedure | Flow | Proce
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=4.4403, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_LM GUIDE ## REPLACEMENT & GREASE INJECTION Global SOP No: Revision No: 1 Page:
  - [ ] `global_sop_geneva_xp_rep_pm_mfc` (score=4.4159, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_MFC SOP No: 0 Revision No: 0 Page: 8 / 18 ## 10. Work Procedure | Flow | Proce
  - [ ] `set_up_manual_ecolite_ii_400` (score=4.4049, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 11. Leak Check | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) Leak Check | a. Leak Ch
  - [ ] `set_up_manual_precia` (score=4.3601, device=PRECIA, type=set_up_manual)
    > ```markdown # 7) 중량물 취급 작업 ## 중량물 취급 시 주의사항 - 발은 어깨 너비로, 허리는 반듯이 세우고 무릎의 힘으로 일어섭니다. <!-- Image (127, 161, 594, 247) --> 
  - [ ] `global_sop_precia_all_pm_gas_spring` (score=4.3576, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_GAS SPRING Global SOP No : 0 Revision No : 0 Page : 13/36 | Flow | Procedure | To
  - [ ] `global_sop_supra_n_series_all_sub_unit_gas_box_board` (score=4.3146, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_GAS BOX BOARD Global SOP No: 0 Revision No: 4 Page: 2/23 ## 1. Safe
  - [ ] `global_sop_supra_n_series_all_pm_gas_spring` (score=4.2612, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_GAS SPRING Global SOP No: Revision No: 4 Page: 15 / 22 | Flow | Procedure
  - [ ] `40043020` (score=4.2231, device=SUPRA Vplus, type=myservice)
    > -. PM3 CH1 특정 공정 Peak 불량 점검 요청
  - [ ] `global_sop_integer_plus_all_tm_epc` (score=4.1685, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_ INTEGER plus_ALL_TM_EPC Global_SOP No: Revision No: 0 Page: 2/21 ## 1. Safety 1) 안전 및 주의사항 - M
  - [ ] `global_sop_integer_plus_all_tm_devicenet_board` (score=4.1623, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_ INTEGER plus_ALL_TM_DEVICENET BOARD Global_SOP No: Revision No: 3 Page: 2/21 ## 1. Safety 1) 안
  - [ ] `global_sop_integer_plus_all_tm_filter` (score=4.0965, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_FILTER Global_SOP No: Revision No: 0 Page: 2/20 ## 1. Safety 1) 안전 및 주의사항 -

#### q_id: `A-amb041`
- **Question**: PM 작업 시 Chamber Open 전 안전 절차는?
- **Devices**: [SUPRA_N, SUPRA_XP, SUPRA_VPLUS, INTEGER_PLUS, PRECIA]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-24):
  - [ ] `global_sop_integer_plus_all_tm_top_lid` (score=7.5595, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_CLN_TM CHAMBER Global SOP No: Revision No: 0 Page: 7/72 ## 1. 환경 안전 보호구 | 구분 | 상세구
  - [ ] `global_sop_supra_n_all_pm_chamber_open_interlock_change` (score=7.5587, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_MODIFY_PM_CHAMBER_OPEN_INTERLOCK_CHANGE Global SOP No: Revision No: 3 Page: 8/24 ## 1. 
  - [ ] `global_sop_integer_plus_all_tm_robot` (score=7.5586, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_CLN_TM_CHAMBER Global SOP No : Revision No: 4 Page : 75 / 103 ## 1. 환경 안전 보호구 | 구분
  - [ ] `global_sop_supra_n_series_all_pm_chamber_hinge` (score=7.3758, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_CHAMBER HINGE Global SOP No : Revision No: 3 Page: 8/22 ## 1. 환경 안전 보호구 |
  - [ ] `global_sop_supra_n_all_pm_fcip_r3` (score=7.3221, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_PM_FCIP R3 REPLACEMENT Global SOP No : Revision No: 3 Page : 11 / 84 ## 1. 환경 안전 보호
  - [ ] `global_sop_integer_plus_all_pm_gas_line` (score=7.2865, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_GAS FILTER Global_SOP No: Revision No: 0 Page: 33/75 ## 1. 환경 안전 보호구 | 구분 |
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=7.284, device=ZEDIUS XP, type=set_up_manual)
    > Picture 6) 안전 관리자 및 보조 작업 인원 ### 위험작업에서 안전담당자 역할 - 모든 위험작업 시 현장 상주 (등급 무관) ※ 안전담당자 현장 이탈 시 작업 중지 - 작업 승인 부서 및 환경안전부서의 안전
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=7.284, device=GENEVA XP, type=set_up_manual)
    > Picture 6) 안전 관리자 및 보조 작업 인원 ### 위험작업에서 안전담당자 역할 - 모든 위험작업 시 현장 상주 (등급 무관) ※ 안전담당자 현장 이탈 시 작업 중지 - 작업 승인 부서 및 환경안전부서의 안전
  - [ ] `set_up_manual_supra_xq` (score=7.2506, device=SUPRA XQ, type=SOP)
    > ```markdown # 0. Safety ## Picture 6) 안전 관리자 및 보조 작업 인원 ### 위험작업에서 안전담당자 역할 - 모든 위험작업 시 현장 상주 (등급 무관) - ※ 안전담당자 현장 이탈 시 
  - [ ] `set_up_manual_ecolite_ii_400` (score=7.2506, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 0. Safety ## Picture 6) 안전 관리자 및 보조 작업 인원 ### 위험작업에서 안전담당자 역할 - 모든 위험작업 시 현장 상주 (등급 무관) - ※ 안전담당자 현장 이탈 시 
  - [ ] `set_up_manual_ecolite_3000` (score=7.2506, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 0. Safety ## Picture 6) 안전 관리자 및 보조 작업 인원 ### 위험작업에서 안전담당자 역할 - 모든 위험작업 시 현장 상주 (등급 무관) - ※ 안전담당자 현장 이탈 시 
  - [ ] `global_sop_geneva_xp_adj_pm_chuck_temp_calibration` (score=7.2381, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Chuck temp calibration Global SOP No: Revision No: 0 Page: 5/21 ## 4. 환경 안전 보호
  - [ ] `global_sop_supra_n_series_all_pm_process_kit` (score=7.1946, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_CLN_PM_CHAMBER Global SOP No : Revision No: 0 Page: 26 / 55 ## 1. 환경 안전 보호구 | 구분
  - [ ] `set_up_manual_precia` (score=7.1706, device=PRECIA, type=set_up_manual)
    > | 5 | 비상연락망 및 담당자 연락처 | 협력사 | | :--- | :--- | :--- | | 6 | 기타 작업 특성에 따른 추가 서류 | | | | - 밀폐공간작업 시: 가스농도측정결과 (1 년 보관) | | 
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=7.1688, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 5/47 ## 4. 환경 안전 보호구 | 구분 | 상세구
  - [ ] `global_sop_integer_plus_all_am_gas_line` (score=7.1268, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_GAS FILTER Global SOP No: Revision No: 0 Page: 56 / 67 ## 1. 환경 안전 보호구 | 구분
  - [ ] `global_sop_supra_xp_all_pm_cip_chamber` (score=7.1223, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_CIP CHAMBER Global SOP No: Revision No : 0 Page : 10 /37 ## 1. 환경 안전 보호구 | 구분 
  - [ ] `global_sop_supra_n_series_all_sub_unit_temp_controller` (score=7.1128, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_ TEMP CONTROLLER Global SOP No : Revision No: 2 Page: 2/56 ## 1. Sa
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=7.095, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_ROBOT Global SOP No : Revision No: 6 Page : 2 / 107 ## 1. Safety ### 1) 안
  - [ ] `global_sop_supra_xp_all_pm_manometer` (score=7.0826, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_MANOMETER ADJUST Global SOP No : Revision No: 1 Page : 20/32 ## 1. 환경 안전보호구 | 
  - [ ] `global_sop_supra_nm_all_pm_microwave_local_interface` (score=7.0713, device=SUPRA Nm, type=SOP)
    > ```markdown # Global SOP_SUPRA Nm_ADJ_PM_MICROWAVE_LOCAL INTERFACE Global_SOP No: Revision No: Page: 8/21 ## 1. 환경 안전 보호
  - [ ] `global_sop_supra_n_series_all_pm_hook_lifter_servo_motor` (score=7.066, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_PIN MOTOR Global SOP No : Revision No: 2 Page : 24 / 106 ## 1. 환경 안전 보호구 
  - [ ] `set_up_manual_supra_n` (score=7.0449, device=SUPRA N, type=SOP)
    > ```markdown # 0. Safety ## Picture 6) 안전 관리자 및 보조 작업 인원 ### 위험작업에서 안전담당자 역할 - 모든 위험작업 시 현장 상주 (등급 무관) - ※ 안전담당자 현장 이탈 시 
  - [ ] `global_sop_supra_n_series_all_pm_top_lid` (score=7.0448, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N Series_REP_PM_TOP LID Global SOP No: Revision No: 0 Page : 27 / 48 ## 1. 환경 안전 보호구 | 구분

#### q_id: `A-amb042`
- **Question**: Wet Cleaning 후 Part 재장착 순서는?
- **Devices**: [SUPRA_N, SUPRA_XP, INTEGER_PLUS, PRECIA]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-20):
  - [ ] `set_up_manual_ecolite_2000` (score=6.593, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 8. Part Installation | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) Ceramic Parts<br>
  - [ ] `set_up_manual_supra_n` (score=5.9255, device=SUPRA N, type=SOP)
    > ```markdown # 3. Docking (※환경안전 보호구 : 안전모, 안전화) ## 3.10 Baratron Gauge, ATM Assy 장착 | Picture | Description | Tool & Spe
  - [ ] `set_up_manual_supra_np` (score=5.9255, device=SUPRA Np, type=set_up_manual)
    > ```markdown # 3. Docking (※환경안전 보호구: 안전모, 안전화) ## 3.10 Baratron Gauge, ATM Assy 장착 | Picture | Description | Tool & Spec
  - [ ] `set_up_manual_supra_nm` (score=5.726, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential 1 # 3. Docking (※환경안전 보호구 : 안전모, 안전화) ## 3.11 Baratron Gauge, ATM Assy 장착 | Picture | Descripti
  - [ ] `40055238` (score=5.6098, device=SUPRA Vplus, type=myservice)
    > -. A급 Part 수급 후 장착 완료
-> 패럴은 재사용
-> Fail품 고객 전달 완료(김창용)
-. TM DNET Board(EPAH51 Fail품) 고객 전달 완료(김창용)
-. TM Dnet 장착 후 Shu
  - [ ] `set_up_manual_ecolite_ii_400` (score=5.5946, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 8. Part Installation_Process kit | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) Ceram
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=5.4803, device=ZEDIUS XP, type=set_up_manual)
    > ```markdown # 8. Part Installation (※환경안전 보호구 : 안전모, 안전화) ## 8.2 Baratron Gauge, ATM Assy | Picture | Description | Tool
  - [ ] `40080973` (score=5.3849, device=INTEGER IVr, type=myservice)
    > -. 해당 Power Supply 분리 후 고객 측에서 자체 수리 완료
-. 수리된 Part 재장착 후 MCB0-20 On 시 정상 작동
-. Alarm Clear 완료
-. 설비 내 Wafer 수거 완료
  - [ ] `global_sop_supra_n_series_all_pm_pek_kit` (score=5.3354, device=SUPRA N series, type=SOP)
    > ```markdown # Global_SOP_SUPRA_N series_ALL_PM_PEK KIT(D-Net Type) Global SOP No: Revision No: 0 Page: 16 / 43 | Flow | 
  - [ ] `set_up_manual_ecolite_3000` (score=5.2345, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 3. Docking (EPD 장착) | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) EPD 구성 | a. EPD는 그
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=5.1891, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 30 / 47 ## 10. Work Procedure |
  - [ ] `global_sop_supra_xp_all_pm_dual_epd` (score=5.1841, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_VIEW QUARTZ REPLACEMENT Global SOP No: Revision No: 1 Page: 35/40 | Flow | Pro
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock` (score=4.9707, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Load lock SOP No: 0 Revision No: 1 Page: 13/22 ## 10. Work Procedure | Flow | 
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=4.9514, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 5. Install Accessory ## 5.3 Install VSP | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1
  - [ ] `global_sop_supra_n_series_all_pm_process_kit` (score=4.9136, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_CLN_PM_CHAMBER Global SOP No : Revision No: 0 Page: 34 / 55 | Flow | Procedure |
  - [ ] `global_sop_geneva_xp_rep_pm_disc` (score=4.9068, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc SOP No: 0 Revision No: 0 Page: 17/31 ## 10. Work Procedure | Flow | Proce
  - [ ] `set_up_manual_supra_xq` (score=4.8003, device=SUPRA XQ, type=SOP)
    > ```markdown # 3. Docking (※환경안전 보호구: 안전모, 안전화) | 3-11 EPD | | | | :--- | :--- | :--- | | Picture | Description | Tool & 
  - [ ] `global_sop_integer_plus_all_pm_igs_block` (score=4.7715, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_IGS BLOCK Global SOP No : Revision No: 0 Page: 17/26 | Flow | Procedure | T
  - [ ] `global_sop_supra_xp_all_efem_ffu` (score=4.7652, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_EFEM_FFU ## CONTROLLER REPLACEMENT Global SOP No: Revision No : 1 Page : 23/60 ##
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock_door_o_ring` (score=4.6991, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_LOAD LOCK DOOR O-RING Global SOP No: S-KG-R020-R0 Revision No: 0 Page: 7/23 ##

#### q_id: `A-amb043`
- **Question**: 정기 PM 시 교체 부품 리스트 확인 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, INTEGER_PLUS, PRECIA, GENEVA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-16):
  - [ ] `40046755` (score=6.1323, device=SUPRA Nm, type=myservice)
    > -. 고온 방지 Cover 장착
-. Temp Limit Setting
-. 정기 PM 중이라 Baffle, Focus Adaptor 장착방법 고객 및 PM업체 공유
-. 고객측에서 Applicator Tube 교체
  - [ ] `set_up_manual_supra_np` (score=5.6251, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I | 2) Jig Assy 조립도 확인 | a. Jig Assy 조립도를 확인하여 다음과 같은 방법으로 Jig Assy가 설치되어야 한다. | | | :--- | :--
  - [ ] `set_up_manual_supra_nm` (score=5.0697, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.2 EFEM Robot Leveling & Loadport Teaching | Picture
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=4.9032, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_LM GUIDE ## REPLACEMENT & GREASE INJECTION Global SOP No: Revision No: 1 Page:
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=4.7377, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `set_up_manual_supra_n` (score=4.7182, device=SUPRA N, type=SOP)
    > ```markdown Confidential I 36) LP2,3 Teaching <!-- Image (70, 70, 290, 226) --> a. LP1 Teaching 과 동일한 방법으로 LP2,3 를 진행하여 
  - [ ] `40052963` (score=4.5565, device=SUPRA Vplus, type=myservice)
    > -. LP2 Door Close 점검
-. Lot 공정 진행 중 Open 되지 않음.
-. Close 상태 지속되어 점검 요청
-. 마지막 Alarm 5/15 일자 확인 후 현업 측 5/17 자체 조치 완료
-. 고
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=4.517, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM ROBOT ASSY REPLACEMENT Global SOP No : Revision No: 6 Page: 49 / 107 ## 8
  - [ ] `global_sop_precia_all_tm_device_net_board` (score=4.4346, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_TM_DEVICE NET BOARD Global SOP No : 0 Revision No : 0 Page : 31 / 36 | Flow | Proced
  - [ ] `global_sop_supra_xp_all_pm_heater_chuck` (score=4.4299, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP ZEDIUS XP_ALL_PM_HEATER # CHUCK Global SOP No: 0 Revision No: 2 Page: 21 / 49 | Flow | Procedur
  - [ ] `global_sop_integer_plus_all_pm_vacuum_line` (score=4.3968, device=INTEGER XP, type=SOP)
    > ```markdown # Global SOP_INTEGER plus _REP_PM_Vacuum Line Global SOP No: Revision No: Page : 133 / 133 ## 8. Appendix [E
  - [ ] `set_up_manual_precia` (score=4.3685, device=PRECIA, type=set_up_manual)
    > ```markdown # 7) 중량물 취급 작업 ## 중량물 취급 시 주의사항 - 발은 어깨 너비로, 허리는 반듯이 세우고 무릎의 힘으로 일어섭니다. <!-- Image (127, 161, 594, 247) --> 
  - [ ] `global_sop_supra_xp_all_tm_robot` (score=4.3009, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_ROBOT ## TEACHING Global SOP No: Revision No : 3 Page : 35 / 47 | Flow | Proce
  - [ ] `40044046` (score=4.284, device=SUPRA Vplus, type=myservice)
    > -. EPAGH4 PM1 Leak Rate Over Alarm
-> Leak Rate : 16mTorr/min
-> 고객측 확인 시 CH1 Bellows 확인으로 교체 요청
  - [ ] `global_sop_supra_n_series_all_pm_gas_feed_through` (score=4.2776, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS FEED THROUGH Global SOP No: Revision No: 4 Page: 4 / 18 ## 3. 사고 사례 #
  - [ ] `global_sop_supra_vplus_adj_all_power_turn_on_off` (score=4.2435, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_RACK_POWER ## TURN ON/OFF Global SOP No: Revision No: 1 Page: 9/19 ## 10. Work 

#### q_id: `A-amb044`
- **Question**: O-Ring 규격 확인 및 호환 부품 검색 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, GENEVA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-23):
  - [ ] `set_up_manual_supra_np` (score=5.6034, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I | 2) Jig Assy 조립도 확인 | a. Jig Assy 조립도를 확인하여 다음과 같은 방법으로 Jig Assy가 설치되어야 한다. | | | :--- | :--
  - [ ] `global_sop_integer_plus_all_tm_top_lid` (score=5.552, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_TM REFLECTOR QUARTS Global SOP No: Revision No: 0 Page: 65/72 | Flow | Procedu
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=5.5004, device=ZEDIUS XP, type=SOP)
    > # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No: 13 Page: 26/75 | Flow | 절차 | Tool & Spe
  - [ ] `global_sop_geneva_xp_rep_pm_insulation_heater` (score=5.4694, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Insulation heater SOP No: 0 Revision No: 0 Page: 15/21 ## 10. Work Procedure |
  - [ ] `global_sop_integer_plus_all_ll_slot_valve` (score=5.4274, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_LL_SLOT VALVE Global SOP No : Revision No : 4 Page : 27 / 50 ## 7. 작업 Check Sh
  - [ ] `global_sop_integer_plus_all_pm_slot_valve` (score=5.3981, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_SLOT VALVE Global SOP No : Revision No : 2 Page : 22 / 39 ## 7. 작업 Check Sh
  - [ ] `40046555` (score=5.3409, device=SUPRA V, type=myservice)
    > -. MCB Rep
-> 기존 규격 : 2A
-> 현재 규격 : 4A
-. TM FFU Rep
-. 추후 배선 작업 필요
  - [ ] `global_sop_genevaxp_rep_efem_robot_end_effector` (score=5.3128, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_EFEM_Robot end effector Global SOP No: Revision No: 0 Page: 10 / 15 ## 10. Work Pr
  - [ ] `global_sop_integer_plus_all_am_pin_motor` (score=5.2919, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_PIN BELLOWS Global SOP No: Revision No: 4 Page: 40 / 84 ## 7. 작업 Check Shee
  - [ ] `global_sop_integer_plus_all_am_slot_valve` (score=5.2523, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_SLOT VALVE Global SOP No : Revision No : 0 Page : 22 / 39 ## 7. 작업 Check Sh
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3000qc` (score=5.1752, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PRISM SOURCE 3000QC REPLACEMENT Global SOP No: Revision No : 2 Page : 25/82 | Flo
  - [ ] `global_sop_geneva_xp_rep_pm_loadlock_apc_valve` (score=5.1453, device=geneva_xp_rep, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Loadlock APC Global SOP No: Revision No: 0 Page: 20 / 22 ## 11. Check Sheet | 
  - [ ] `global_sop_geneva_xp_rep_pm_chamber_apc_valve` (score=5.1434, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Chamber APC Global SOP No: Revision No: 0 Page: 20 / 22 ## 11. Check Sheet | 구
  - [ ] `global_sop_precia_all_pm_chuck` (score=5.0948, device=PRECIA, type=SOP)
    > # Global SOP_PRECIA_REP_PM_CHUCK ISO PLATE Global SOP No : Revision No: 1 Page: 51 / 132 ## 6. Work Procedure | Flow | P
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3100qc` (score=5.0932, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PRISM SOURCE ## 3100QC REPLACEMENT Global SOP No: Revision No : 1 Page : 25/72 | 
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=5.0847, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page :105 / 105 ## 17. Iso
  - [ ] `global_sop_supra_xp_all_pm_dual_epd` (score=5.0839, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_VIEW QUARTZ REPLACEMENT Global SOP No: Revision No: 1 Page: 33/40 | Flow | Pro
  - [ ] `40052963` (score=5.0795, device=SUPRA Vplus, type=myservice)
    > -. LP2 Door Close 점검
-. Lot 공정 진행 중 Open 되지 않음.
-. Close 상태 지속되어 점검 요청
-. 마지막 Alarm 5/15 일자 확인 후 현업 측 5/17 자체 조치 완료
-. 고
  - [ ] `global_sop_supra_vplus_adj_all_io_check` (score=5.036, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_ALL_IO CHECK Global SOP No Revision No: 0 Page: 12/15 ## 10. Work Procedure | F
  - [ ] `global_sop_supra_series_all_sw_operation` (score=4.9651, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_ SW OPERATION Global SOP No Revision No: 2 Page: 5/49 ## 1. IO Check - Work Pr
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock_o_ring` (score=4.9387, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_LOAD LOCK O-RING Global SOP No: S-KG-R019-R0 Revision No: 0 Page: 8 / 30 ## 8.
  - [ ] `global_sop_geneva_xp_rep_pm_elbow_heater` (score=4.9193, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Elbow heater SOP No: 0 Revision No: 0 Page: 17 / 22 ## 10. Work Procedure | Fl
  - [ ] `global_sop_integer_plus_all_tm_ctc_controller` (score=4.879, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_CTC CONTROLLER Global SOP No: Revision No: 1 Page: 49/51 ## 7. 작업 Check She

#### q_id: `A-amb045`
- **Question**: Screw Torque Spec 확인 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-19):
  - [ ] `global_sop_precia_all_pm_prevent_maintenance` (score=6.0337, device=PRECIA, type=SOP)
    > # Global SOP_PRECIA_REP_PM_PROCESS KIT (BOTTOM MOUNT TYPE) Global SOP No: Revision No: 5 Page: 66 / 108 ## 6. Work Proce
  - [ ] `set_up_manual_precia` (score=5.6094, device=PRECIA, type=set_up_manual)
    > | | c) GDP Edge 4point Shim Jig 장착 | | | :--- | :--- | :--- | | | d) (1),(2),(3),(4) GDP Fix Screw 체결 (Torque Spec: 100k
  - [ ] `set_up_manual_supra_np` (score=5.6003, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I | 2) Jig Assy 조립도 확인 | a. Jig Assy 조립도를 확인하여 다음과 같은 방법으로 Jig Assy가 설치되어야 한다. | | | :--- | :--
  - [ ] `set_up_manual_supra_vm` (score=5.5676, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 12. Component Adjust_ Focus Adaptor & Baffle Install | Picture | Description | Tool & Spec | | :--- | :---
  - [ ] `global_sop_integer_plus_all_pm_igs_block` (score=5.3953, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_IGS BLOCK Global SOP No : Revision No: 0 Page: 17/26 | Flow | Procedure | T
  - [ ] `set_up_manual_ecolite_2000` (score=5.299, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 11. TTTM_ Focus Adaptor & Baffle Install | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 
  - [ ] `global_sop_supra_n_series_all_pm_heater_chuck` (score=5.2939, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_HEATER CHUCK Global SOP No : Revision No: 2 Page: 25/40 | Flow | Procedur
  - [ ] `global_sop_supra_n_all_pm_fcip_r3` (score=5.2524, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_REP_PM_FCIP R3 EXIT FLANGE Global SOP No : Revision No: 3 Page: 56/84 | Flow | Procedur
  - [ ] `global_sop_supra_n_all_sub_unit_mfc` (score=5.204, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_SUB UNIT_MFC Global SOP No: 0 Revision No: 3 Page: 14 / 21 | Flow | Procedure | Too
  - [ ] `global_sop_precia_all_pm_gap_sensor` (score=5.1468, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_VIEW PORT QUARTZ Global SOP No : Revision No: 1 Page: 71/79 ## 6. Work Procedure 
  - [ ] `set_up_manual_supra_nm` (score=5.1367, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.2 EFEM Robot Leveling & Loadport Teaching | Picture
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=5.0831, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No : 13 Page : 70/75 | Flow | 절
  - [ ] `global_sop_supra_xp_all_efem_robot_sr8241` (score=4.9778, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_EFEM_ROBOT_SR8241 Global SOP No : Revision No: 2 Page: 20/34 | Flow | Procedure |
  - [ ] `global_sop_supra_xp_all_tm_slot_valve` (score=4.9332, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_SLOT VALVE REPLACEMENT Global SOP No : Revision No : 5 Page : 16 / 20 | Flow |
  - [ ] `global_sop_supra_n_series_all_pm_process_kit` (score=4.9049, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_PROCESS KIT Global SOP No : Revision No: 0 Page: 18/55 | Flow | Procedure
  - [ ] `global_sop_supra_n_all_efem_robot_sr8240` (score=4.8772, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_EFEM ROBOT_SR # 8240 REPLACEMENT Global SOP No: 0 Revision No: 5 Page: 19 / 44 | Fl
  - [ ] `global_sop_precia_all_tm_robot` (score=4.8566, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_TM_TM ROBOT Global SOP No : Revision No: 0 Page: 20/56 ## 6. Work Procedure | Flow |
  - [ ] `global_sop_precia_all_pm_mfc` (score=4.8475, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_MFC Global SOP No : Revision No: 1 Page: 17/23 ## 6. Work Procedure | Flow | Proc
  - [ ] `global_sop_integer_plus_all_pm_swap_kit` (score=4.8301, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_SWAP KIT Global SOP No: Revision No: 3 Page: 26 / 37 | Flow | Procedure | T

#### q_id: `A-amb046`
- **Question**: Host Communication Error 발생 시 조치 방법은?
- **Devices**: [SUPRA_N, SUPRA_VPLUS, SUPRA_XP, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-26):
  - [ ] `global_sop_geneva_xp_rep_pm_loadlock_apc_valve` (score=6.1817, device=geneva_xp_rep, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Loadlock APC Global SOP No: Revision No: 0 Page: 19 / 22 | Flow | Procedure | 
  - [ ] `global_sop_geneva_xp_rep_pm_differential_gauge` (score=6.1557, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Differential gauge Global SOP No: Revision No: 0 Page: 15 / 18 | Flow | Proced
  - [ ] `global_sop_geneva_xp_rep_pm_chamber_apc_valve` (score=6.1554, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Chamber APC Global SOP No: Revision No: 0 Page: 19 / 22 | Flow | Procedure | T
  - [ ] `global_sop_geneva_xp_rep_pm_harmonic_drive` (score=5.8385, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Harmonic drive Global SOP No: 0 Revision No: 0 Page: 31 / 31 | Flow | Procedur
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=5.8372, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier Global SOP No: 0 Revision No: 0 Page: 39 / 43 | Flow | Procedur
  - [ ] `global_sop_geneva_xp_rep_pm_adapter_ring` (score=5.8268, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Adapter ring Global SOP No: 0 Revision No: 0 Page: 26 / 30 ## 10. Work Procedu
  - [ ] `global_sop_geneva_xp_rep_pm_heater_chuck_without_jig` (score=5.8235, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA xp_REP_PM_ Heater chuck w/o jig SOP No: 0 Revision No: 1 Page: 48 / 52 | Flow | Procedure | Tool & 
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=5.8194, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 43 / 47 ## 10. Work Procedure |
  - [ ] `supra_n_all_trouble_shooting_guide_trace_communication_abnormal` (score=5.6854, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Communication Abnormal] Use this guide to diagnose problems with the [Trace 
  - [ ] `global_sop_precia_all_efem_controller` (score=5.6849, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_EFEM EDA CONTROLLER Global SOP No : Revision No: 0 Page: 25 / 42 ## 6. Work Procedur
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=5.6159, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_EFEM_END # EFFECTOR Global SOP No: Revision No: 3 Page: 66 / 126 ## 6. Work Pr
  - [ ] `global_sop_geneva_xp_rep_pm_mfc` (score=5.5944, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_MFC SOP No: 0 Revision No: 0 Page: 8 / 18 ## 10. Work Procedure | Flow | Proce
  - [ ] `global_sop_geneva_xp_rep_efem_robot_sr8240` (score=5.5085, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_ROBOT SR8240 Global SOP No: 0 Revision No: 0 Page: 14 / 17 | Flow | Procedur
  - [ ] `supra_xp_all_trouble_shooting_guide_trace_ctc_abnormal` (score=5.4964, device=SUPRA XP, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace CTC Abnormal] Use this guide to diagnose problems with the [Trace CTC Abnorm
  - [ ] `global_sop_precia_all_efem_serial_8port` (score=5.3689, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_EFEM_SERIAL 8PORT Global SOP No : Revision No: 0 Page : 16 / 18 ## 6. Work Procedure
  - [ ] `global_sop_geneva_xp_rep_pm_device_net` (score=5.3133, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Device net | SOP No: 0 | | | | | :--- | :--- | :--- | :--- | | Revision No: 0 
  - [ ] `supra_xp_all_trouble_shooting_guide_trace_device_net_abnormal` (score=5.2911, device=SUPRA XP, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Device Net Abnormal] Use this guide to diagnose problems with the [Trace Dev
  - [ ] `global_sop_precia_all_efem_ffu` (score=5.2094, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_EFEM_FFU CONTROLLER Global SOP No : Revision No: 0 Page: 21 / 50 | Flow | Procedure 
  - [ ] `global_sop_precia_all_ll_relief_valve` (score=5.2089, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_LL_RELIEF VALVE Global SOP No: Revision No: 1 Page: 16 / 18 | Flow | Procedure | Too
  - [ ] `global_sop_supra_n_series_all_sub_unit_elt_box_assy` (score=5.2077, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_SUB # UNIT_PDB Global SOP No: Revision No: 0 Page :93 / 95 | Flow | Procedur
  - [ ] `global_sop_precia_all_tm_sensor_board` (score=5.1951, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_TM_SENSOR BOARD Global SOP No: Revision No: 1 Page: 16 / 18 | Flow | Procedure | Too
  - [ ] `global_sop_precia_all_ll_vision` (score=5.1947, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_LL_VISION LIGHT # CONTROLLER Global SOP No : Revision No: 0 Page : 42 / 82 | Flow | 
  - [ ] `global_sop_geneva_xp_rep_pm_o2_analyzer_modify` (score=5.1923, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_O2 ANALYZER Modify Global SOP No: 0 Revision No: 0 Page: 30 / 33 | Flow | Proc
  - [ ] `global_sop_integer_plus_all_efem_pio_sensor_board` (score=5.1797, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_EFEM_PIO SENSOR BOARD Global SOP No: Revision No: 1 Page: 14 / 16 | Flow | Pro
  - [ ] `global_sop_geneva_xp_adj_pm_chuck_temp_calibration` (score=5.1666, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Chuck temp calibration Global SOP No: Revision No: 0 Page: 9/21 ## 10. Work Pr
  - [ ] `global_sop_integer_plus_all_efem_eda_controller` (score=5.1497, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_EFEM_EDA # CONTROLLER Global SOP No: Revision No: 1 Page: 20 / 22 | Flow | Pro

#### q_id: `A-amb047`
- **Question**: Recipe Download Fail 시 점검 사항은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-23):
  - [ ] `global_sop_supra_n_series_all_tm_ctc` (score=5.3337, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_CTC REPLACEMENT Global SOP No : Revision No: 6 Page: 35/81 | Flow | Proce
  - [ ] `global_sop_geneva_xp_all_8계통_check` (score=4.9564, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA XP_8계통_Check sheet SOP No: 0 Revision No: 0 Page: 22 / 46 ## 10. Work Procedure | Flow | Procedure 
  - [ ] `global_sop_supra_xp_all_tm_ctc` (score=4.8063, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_CTC REPLACEMENT Global SOP No : Revision No: 1 Page: 33/45 | Flow | Procedure 
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=4.7189, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier Global SOP No: 0 Revision No: 0 Page: 24 / 43 | Flow | Procedur
  - [ ] `global_sop_precia_all_tm_device_net_board` (score=4.4422, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_TM_DEVICE NET BOARD Global SOP No : 0 Revision No : 0 Page : 32 / 36 | Flow | Proced
  - [ ] `global_sop_precia_all_pm_device_net_board` (score=4.4359, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_PM_DEVICE NET BOARD Global SOP No : 0 Revision No : 0 Page : 32 / 37 | Flow | Proced
  - [ ] `global_sop_supra_xp_sw_all_sw_installation_setting` (score=4.3321, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_SW_TM_CTC SW INSTALL Global SOP No : Revision No: 0 Page: 48/74 | Flow | Procedure | 
  - [ ] `global_sop_precia_all_tm_branch_tap` (score=4.2161, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_TM_BRANCH TAP Global SOP No : Revision No: 0 Page: 2/23 ## 1. Safety 1) 안전 및 주의사항 - 
  - [ ] `global_sop_supra_n_series_all_tm_branch_tap` (score=4.1446, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_BRANCH TAP Global SOP No : Revision No: 2 Page: 2/21 ## 1. Safety 1) 안전 및
  - [ ] `global_sop_supra_n_all_tm_sensor_board` (score=4.1428, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_TM_SENSOR BOARD Global SOP No: 0 Revision No: 1 Page: 2/15 ## 1. Safety 1) 안전 및 주의사
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=4.0643, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM ROBOT ASSY REPLACEMENT Global SOP No : Revision No: 6 Page: 44 / 107 | Fl
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=4.064, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_AM_PENDULUM VALVE Global SOP No: Revision No: Page: 59 / 135 ## 1. SAFETY 1) 안
  - [ ] `global_sop_integer_plus_all_am_interface_board` (score=4.0531, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_AM_INTERFACE BOARD Global_SOP No: Revision No: 0 Page: 2 / 18 ## 1. Safety 1) 
  - [ ] `global_sop_supra_n_series_all_tm_fluorescent_lamp` (score=4.0123, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_FLUORESCENT LAMP Global SOP No : Revision No:2 Page: 2/23 ## 1. Safety 1)
  - [ ] `global_sop_integer_plus_all_tm_mototr_controller` (score=3.9883, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_MOTOR CONTROLLER Global SOP No: Revision No: Page: 2 / 21 ## 1. Safety 1) 안
  - [ ] `global_sop_precia_all_efem_signal_tower` (score=3.9473, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_EFEM_SIGNAL TOWER Global SOP No: Revision No: 1 Page: 2/18 ## 1. Safety 1) 안전 및 주의사항
  - [ ] `global_sop_supra_xp_all_efem_pio_sensor_board` (score=3.9349, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ ALL_EFEM_PIO SENSOR BOARD Global SOP No: Revision No: 0 Page: 2 / 14 ## 1. Safety 1)
  - [ ] `global_sop_integer_plus_all_pm_temp_controller` (score=3.9053, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_TEMP # CONTROLLER Global_SOP No: Revision No: 1 Page: 2 / 27 ## 1. Safety 1
  - [ ] `set_up_manual_supra_n` (score=3.9046, device=SUPRA N, type=SOP)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.13 ATM Transfer | Picture | Description | Tool & Sp
  - [ ] `global_sop_integer_plus_all_efem_signal_tower` (score=3.8974, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_EFEM_SIGNAL TOWER Global SOP No: Revision No: 1 Page: 2/18 ## 1. Safety 1) 안전 
  - [ ] `global_sop_integer_plus_all_am_temp_controller` (score=3.8972, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_AM_TEMP # CONTROLLER Global SOP No: Revision No: 1 Page: 2 / 27 ## 1. Safety 1
  - [ ] `global_sop_integer_plus_all_tm_safety_controller` (score=3.8947, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_SAFETY CONTROLLER Global_SOP No: Revision No: 0 Page: 2/19 ## 1. Safety 1) 
  - [ ] `global_sop_integer_plus_all_pm_safety_controller` (score=3.8832, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_SAFETY CONTROLLER Global_SOP No: Revision No: 0 Page: 2/19 ## 1. Safety 1) 

#### q_id: `A-amb048`
- **Question**: Software Update 후 설비 정상 동작 확인 절차는?
- **Devices**: [SUPRA_N, SUPRA_XP, SUPRA_VPLUS, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-26):
  - [ ] `40046083` (score=5.9338, device=SUPRA Vplus, type=myservice)
    > -. Sol Valve 교체 후 Manual 동작 정상 확인
-. 백업 예정
  - [ ] `global_sop_geneva_xp_rep_pm_분배기` (score=5.6975, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_분배기 Global SOP No: Revision No: 0 Page: 11 / 13 ## 12. 환경 안전 보호구 Check Sheet | 
  - [ ] `global_sop_supra_n_series_all_sub_unit_gas_box_board` (score=5.5906, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_SUB UNIT _GAS BOX BOARD Global SOP No: 0 Revision No: 4 Page: 21 / 23 | Flow
  - [ ] `global_sop_geneva_xp_rep_pm_controller` (score=5.5659, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVAxp_REP_PM_Controller Global SOP No: Revision No: 0 Page: 5 / 12 ## 6. Flow Chart - Start - 1. SOP 및 안
  - [ ] `40052189` (score=5.4439, device=SUPRA V, type=myservice)
    > -. 고객 요청으로 Power Supply 탈착하여 콘덴서 확인
->  콘덴서 확인시 2개 부풀어서 터짐 확인
-. 고객 전자기기 수리실 수리 후 장착
-. 장착 후 정상 동작 확인
-. Monitoring
  - [ ] `40094757` (score=5.3059, device=SUPRA N, type=myservice)
    > -. TM Robot Vacuum Check
-> Wafer가 Pick에 없는 상황에서 4번 값이 74kpa로 Reading
-. TM Robot Reset
-> 현상 동일
-. Main Vacuum Check
->
  - [ ] `global_sop_supra_xp_all_efem_controller` (score=5.2206, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_EFEM_EDA pc REPLACEMENT Global SOP No: Revision No: 1 Page: 45/46 ## 7. 작업 Check 
  - [ ] `40039537` (score=5.2035, device=SUPRA III, type=myservice)
    > -. Log Check
-> Mapping arm Open Alarm 없음
-. H/W 동작 Test 시 정상 확인
-. 고객측 Monitoring 후 재발 시 재점검
  - [ ] `global_sop_geneva_xp_rep_efem_ffu` (score=5.1985, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_EFEM_FFU SOP No: 0 Revision No: 0 Page: 11 / 14 | Flow | Procedure | Tool & Point
  - [ ] `global_sop_geneva_xp_rep_pm_digihelic` (score=5.1193, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVAxp_REP_PM_Digihelic Global SOP No: Revision No: 0 Page: 5/13 ## 6. Flow Chart - Start - 1. SOP 및 안전사항
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=5.0962, device=ZEDIUS XP, type=set_up_manual)
    > ```markdown # 5.2 EMO Check | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) EMO(Emergency Machin Off
  - [ ] `global_sop_precia_all_tm_sensor_board` (score=5.0472, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_TM_SENSOR BOARD Global SOP No: Revision No: 1 Page: 17/18 ## 7. 작업 Check Sheet | 구분 
  - [ ] `set_up_manual_supra_n` (score=5.0311, device=SUPRA N, type=SOP)
    > ```markdown Confidential I | 12) ABS 기능 실행 | a. Menu -> ABS -> Move | | :--- | :--- | | | b. 화면에 안내되는 내용에 따라 | | | ABS 동
  - [ ] `40076066` (score=4.9757, device=SUPRA Vplus, type=myservice)
    > -. PM1 PMC 교체
-. Ignition Test : 정상
-. BM1 CH1 Align 동작 Test
-> Align 동작 시 Pusher 끝까지 밀지 못하고 회귀 동작 Speed 느림
-> Air Line 
  - [ ] `set_up_manual_ecolite_3000` (score=4.9568, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 5. Power Turn On (EQ Power Turn On & EMO 확인) | Picture | Description | Tool & Spec | | :--- | :--- | :--- 
  - [ ] `40079239` (score=4.9199, device=SUPRA Vplus, type=myservice)
    > -. All PM Pump Down, Temp 정상 Reading X
-> Temp CTR 정상 확인
-. Rack 확인 시 ELCB0-1 Trip 확인
-. ELCB DVM Check 시 Input 220V 정상

  - [ ] `global_sop_precia_all_efem_serial_8port` (score=4.8941, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_EFEM_SERIAL 8PORT Global SOP No : Revision No: 0 Page : 17 / 18 ## 7. 작업 Check Sheet
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=4.8278, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No: 13 Page: 61/75 ## Scope 이 G
  - [ ] `set_up_manual_ecolite_ii_400` (score=4.8147, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 5. Power Turn On_EQ Power Turn On & EMO 확인 | Picture | Description | Tool & Spec | | :--- | :--- | :--- | 
  - [ ] `set_up_manual_supra_vm` (score=4.8125, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 6. Power Turn On (EQ Power Turn On & EMO 확인) | Picture | Description | Tool & Spec | | :--- | :--- | :--- 
  - [ ] `set_up_manual_ecolite_2000` (score=4.8074, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 5. Power Turn On (EQ Power Turn On & EMO 확인) | Picture | Description | Tool & Spec | | :--- | :--- | :--- 
  - [ ] `set_up_manual_integer_plus` (score=4.8036, device=INTEGER plus, type=SOP)
    > ```markdown # 7. Power Turn On (※환경안전 보호구: 안전모, 안전화) ## 7.1 EMO Check | Picture | Description | Tool & Spec | | :--- | :
  - [ ] `40052963` (score=4.7852, device=SUPRA Vplus, type=myservice)
    > -. LP2 Door Close 점검
-. Lot 공정 진행 중 Open 되지 않음.
-. Close 상태 지속되어 점검 요청
-. 마지막 Alarm 5/15 일자 확인 후 현업 측 5/17 자체 조치 완료
-. 고
  - [ ] `40035500` (score=4.7411, device=SUPRA Vplus, type=myservice)
    > -. EMO Line(PM1,2,3, TM, EFEM, Subunit, Main Rack) Check : 체결 정상
-. Safety Module Check : 정상 동작
-. EFEM, TM Power On : 정
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=4.6715, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 14 / 105 ## 2. Fast 
  - [ ] `global_sop_supra_xp_all_pm_chamber_safety_cover` (score=4.6683, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_SAFETY COVER REPLACEMENT | Global SOP No : | | | --- | --- | | Revision No : 1

#### q_id: `A-amb049`
- **Question**: Sensor 값 Drift 발생 시 Calibration 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, SUPRA_VPLUS, PRECIA]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-10):
  - [ ] `set_up_manual_supra_nm` (score=6.9348, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.4 Device Net Calibration _ PSK Board ### 12.4.1 ATM P
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=6.7059, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `set_up_manual_supra_np` (score=6.389, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I Analog Output Tap은 조작하지 않는다. # 12. TTTM (※환경안전 보호구: 안전모, 안전화) ## 12.5 Device Net Calibration 
  - [ ] `set_up_manual_supra_n` (score=5.8126, device=SUPRA N, type=SOP)
    > ```markdown Confidential I 6) ZERO Calibration <!-- Table (68, 69, 938, 319) --> \begin{tabular}{|l|l|l|} \hline \textbf
  - [ ] `global_sop_supra_n_all_pm_fcip_r3` (score=5.5192, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_PM_FCIP R3 REPLACEMENT Global SOP No : Revision No: 3 Page: 45/84 ## 8. Appendix | 
  - [ ] `global_sop_geneva_xp_adj_pm_aio_calibration` (score=5.3821, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_AIO CALIBRATION Global SOP No: 0 Revision No: 0 Page: 13 / 16 | Flow | Procedu
  - [ ] `set_up_manual_precia` (score=5.2197, device=PRECIA, type=set_up_manual)
    > ```markdown # 6. Power Turn On (환경안전 보호구: 안전모, 안전화) ## 6.6 RF Generator Calibration | Picture | Description | Tool & Spe
  - [ ] `global_sop_supra_n_all_efem_robot_sr8240` (score=5.0562, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_EFEM ROBOT_SR 8240 REPLACEMENT Global SOP No: 0 Revision No: 5 Page: 22 / 44 ## 6. 
  - [ ] `global_sop_supra_n_series_all_tm_ctc` (score=4.997, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_CTC REPLACEMENT Global SOP No : Revision No: 6 Page : 25/81 | Flow | Proc
  - [ ] `global_sop_supra_xp_all_tm_robot` (score=4.9003, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_ROBOT ## TEACHING Global SOP No: Revision No : 3 Page : 35 / 47 | Flow | Proce

#### q_id: `A-amb050`
- **Question**: FDC Alarm 기준 설정 및 조정 방법은?
- **Devices**: [SUPRA_N, SUPRA_VPLUS, SUPRA_XP, INTEGER_PLUS, PRECIA]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-19):
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=6.0332, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `set_up_manual_supra_np` (score=5.086, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I Analog Output Tap은 조작하지 않는다. # 12. TTTM (※환경안전 보호구: 안전모, 안전화) ## 12.5 Device Net Calibration 
  - [ ] `global_sop_supra_n_series_all_tm_ffu` (score=4.9746, device=SUPRA N, type=set_up_manual)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_FFU Global SOP No : Revision No: 1 Page: 15/32 | Flow | Procedure | Tool 
  - [ ] `global_sop_integer_plus_all_pm_manometer` (score=4.8683, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER Plus_REP_MANOMETER Global SOP No: Revision No: 1 Page: 20 / 20 ## 8. Appendix | Flow | 
  - [ ] `global_sop_supra_xp_all_tm_multi_port` (score=4.7549, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDISU XP_ALL_TM_MULTI PORT Global SOP No: Revision No: 1 Page: 17 / 24 | Flow | Procedure | To
  - [ ] `global_sop_precia_all_pm_manometer` (score=4.7216, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_MANOMETER Global SOP No: Revision No: 0 Page: 20 / 20 ## 8. Appendix | Flow | Proced
  - [ ] `global_sop_supra_xp_all_pm_manometer` (score=4.6502, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_MANOMETER ADJUST Global SOP No : Revision No: 1 Page: 27/32 | Flow | Procedure
  - [ ] `global_sop_geneva_xp_rep_pm_mfc` (score=4.6061, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_MFC SOP No: 0 Revision No: 0 Page: 14 / 18 ## 10. Work Procedure | Flow | Proc
  - [ ] `set_up_manual_supra_nm` (score=4.532, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.2 EFEM Robot Leveling & Loadport Teaching | Picture
  - [ ] `global_sop_supra_xp_all_pm_heater_chuck` (score=4.5172, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP ZEDIUS XP_ALL_PM_HEATER # CHUCK Global SOP No: 0 Revision No: 2 Page: 21 / 49 | Flow | Procedur
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=4.5125, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_HOOK LIFTER PIN 조절 Global SOP No: Revision No: 1 Page :116 / 124 | Flow | Proc
  - [ ] `global_sop_precia_all_efem_ctc` (score=4.5075, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_EFEM_CTC Global SOP No: Revision No: 1 Page: 20/51 | Flow | Procedure | Tool & Spec 
  - [ ] `global_sop_integer_plus_all_am_devicenet_board` (score=4.4751, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_ADJ_AM_AIO ## CALIBRATION [TOS BOARD] Global SOP No : Revision No: 0 Page : 35 / 58 | Flow | P
  - [ ] `global_sop_integer_plus_all_pm_d_net` (score=4.4465, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_ADJ_PM_AIO # CALIBRATION [PSK BOARD] Global SOP No : Revision No: 0 Page : 10 / 56 ## 5. Flow 
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=4.4394, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_ADJ_EFEM_ROBOT TEACHING Global SOP No: Revision No: 3 Page: 32 / 126 | Flow | Procedure | Tool
  - [ ] `global_sop_integer_plus_all_tm_ctc_controller` (score=4.3827, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_CTC # CONTROLLER Global SOP No: Revision No: 1 Page: 30 / 51 | Flow | Proce
  - [ ] `global_sop_precia_all_tm_branch_tap` (score=4.3818, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_TM_BRANCH TAP Global SOP No : Revision No: 0 Page: 18 / 23 ## 6. Work Procedure | Fl
  - [ ] `global_sop_supra_xp_all_tm_robot` (score=4.3733, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_ROBOT ## TEACHING Global SOP No: Revision No : 3 Page : 35 / 47 | Flow | Proce
  - [ ] `global_sop_supra_n_series_all_sub_unit_manometer` (score=4.369, device=SUPRA N, type=SOP)
    > # Global SOP_SUPRA N series_ADJ_SUB UNIT_MANOMETER Global SOP No: Revision No: 0 Page: 31/32 ## 8. Appendix ### 계측모드 Mod

#### q_id: `A-amb051`
- **Question**: 최근 동일 Alarm 반복 발생 이력은?
- **Devices**: [SUPRA_N, SUPRA_XP, SUPRA_VPLUS]
- **Scope**: ambiguous | **Intent**: information_lookup
- **ES candidates** (top-30):
  - [ ] `global_sop_integer_plus_all_ll_slot_valve` (score=4.4169, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_LL_SLOT ## VALVE Global SOP No : Revision No : 4 Page : 25 / 50 | Flow | Proce
  - [ ] `40038715` (score=4.3686, device=SUPRA Vplus, type=myservice)
    > -. EFEM Robot Upper Endeffector Rep
-. Upper Arm leak 정상 확인
-. Lower Leak 확인
-> 초당 1kpa 수준 Leak 발생
-. Aging 1Lot 정상 확인
-
  - [ ] `set_up_manual_supra_vm` (score=3.843, device=SUPRA Vm, type=set_up_manual)
    > | 6) Chamber Interlock Jig Install | a) Chamber Open시 Interlock이 발생하기 때문에 Interlock Jig를 설치한다. | a. Tool | | :--- | :---
  - [ ] `40035345` (score=3.8034, device=SUPRA V, type=myservice)
    > -. 간헐적 Communication Alarm 발생
  - [ ] `precia_all_trouble_shooting_guide_rf_matching_alarm` (score=3.7035, device=GENEVA XP, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [RF Matching Alarm] Use this guide to diagnose problems with the [RF Matching Alarm]. It descri
  - [ ] `precia_pm_trouble_shooting_guide_chuck_abnormal` (score=3.6697, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Chuck Abnormal] Use this guide to diagnose problems with the [Chuck Motor Alarm]. 
  - [ ] `precia_pm_trouble_shooting_guide_gap_sensor_alarm` (score=3.6686, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Gap Sensor Alarm] Use this guide to diagnose problems with the [Gap Sensor Alarm].
  - [ ] `precia_ll_trouble_shooting_guide_trace_aligner_alarm` (score=3.6549, device=PRECIA, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace Aligner Alarm] Use this guide to diagnose problems with the [Trace Aligner Alarm]. It de
  - [ ] `supra_n_all_trouble_shooting_guide_trace_fps_abnormal` (score=3.5288, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace FPS abnormal] Use this guide to diagnose problems with the [Trace FPS Abnorm
  - [ ] `precia_all_trouble_shooting_guide_trace_fps_abnormal` (score=3.5288, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace FPS abnormal] Use this guide to diagnose problems with the [Trace FPS Abnorm
  - [ ] `40054656` (score=3.5191, device=SUPRA Vplus, type=myservice)
    > -. 고객 측 D-Net Reset후 Pin Move Time Out Alarm 발생
  - [ ] `set_up_manual_ecolite_2000` (score=3.4621, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 7. Teaching_ATM Transfer | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5)Chamber Open |
  - [ ] `precia_pm_trouble_shooting_guide_process_stable_time_out_alarm` (score=3.4614, device=PRECIA, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Process Stable Time out Alarm] Use this guide to diagnose problems with the [Process Stable Ti
  - [ ] `supra_n_all_trouble_shooting_guide_trace_microwave_abnormal` (score=3.4292, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Microwave Abnormal] Use this guide to diagnose problems with the [Trace Micr
  - [ ] `supra_xp_sub_unit_trouble_shooting_guide_igs_block_abnormal` (score=3.4258, device=SUPRA XP, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [IGS Block Abnormal] Use this guide to diagnose problems with the [IGS Block Abnorm
  - [ ] `40054699` (score=3.4116, device=SUPRA Vplus, type=myservice)
    > -. 6/18 교체 했던 TM Robot Controller로 재교체
-. 교체 후 TM Robot ETC Alarm 발생
-> Controller Cable 체결 상태 양호
-> TM Dnet Cable 체결 상태
  - [ ] `40051833` (score=3.4096, device=SUPRA Vplus, type=myservice)
    > -. LP1 Light Curtain Alarm 1회 발생
-> 고객도 당시 상황 모른다함
  - [ ] `global_sop_precia_all_pm_slot_valve` (score=3.3988, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_SLOT ## VALVE(PRESYS) Global SOP No : 0 Revision No : 0 Page : 18 / 36 | Flow | P
  - [ ] `supra_xp_pm_trouble_shooting_guide_trace_chuck_abnormal` (score=3.3971, device=SUPRA XP, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Chuck Abnormal] Use this guide to diagnose problems with the [Trace Chuck abnormal
  - [ ] `supra_n_all_trouble_shooting_guide_trace_pcw_abnormal` (score=3.3799, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace PCW abnormal] Use this guide to diagnose problems with the [Trace PCW abnorm
  - [ ] `supran_all_trouble_shooting_guide_trace_pcw_abnormal` (score=3.3799, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace PCW abnormal] Use this guide to diagnose problems with the [Trace PCW abnorm
  - [ ] `global_sop_integer_plus_all_pm_pin_motor` (score=3.377, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_PIN MOTOR CONTROLLER Global SOP No: Revision No: 5 Page : 107 / 126 | Flow 
  - [ ] `global_sop_geneva_xp_rep_pm_insulation_heater` (score=3.3765, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Insulation heater | SOP No: 0 | | | | | :--- | :--- | :--- | :--- | | Revision
  - [ ] `global_sop_geneva_xp_rep_pm_device_net` (score=3.3681, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Device net | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/15
  - [ ] `global_sop_geneva_xp_rep_pm_elbow_heater` (score=3.3663, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Elbow heater | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/
  - [ ] `global_sop_supra_n_series_all_pm_top_lid` (score=3.3647, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N Series_REP_PM_TOP LID Global SOP No: Revision No: 0 Page: 47 / 48 ## 7. 작업 Check Sheet 
  - [ ] `40065885` (score=3.3574, device=SUPRA IV, type=myservice)
    > -. Easy Cluster 삭제 후에도 Main Engine 관련 Error Pop-up 발생
-. S/W 재설치 진행
-. Alarm 발생 당시 System Log 반출 후 SW G 문의 예정
  - [ ] `global_sop_precia_all_efem_ctc` (score=3.3513, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_S/W Install_EFEM_CTC Global SOP No: Revision No: 1 Page: 48/51 | Flow | Procedure | Tool
  - [ ] `global_sop_geneva_xp_adj_all_sw_install` (score=3.3492, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_AII_SW INSTALL | SOP No: 0 | | | | --- | --- | --- | | Revision No: 0 | | | | Pag
  - [ ] `global_sop_geneva_xp_rep_pm_o2_analyzer` (score=3.3288, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_O2 analyzer(Delta F) | SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Page

#### q_id: `A-amb052`
- **Question**: 지난 3개월 PM 교체 부품 이력은?
- **Devices**: [SUPRA_N, SUPRA_XP, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: information_lookup
- **ES candidates** (top-26):
  - [ ] `global_sop_supra_vplus_all_pm_controller` (score=5.2042, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_REP_PM_Controller Global SOP No: Revision No: 3 Page: 12/42 ## 8. 필요 Tool | | Name 
  - [ ] `global_sop_supra_n_series_all_pm_epd` (score=4.7563, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_EPD Global SOP No : Revision No: 3 Page: 3/49 ## 3. 사고 사례 ### 1) 화상 재해의 정
  - [ ] `global_sop_supra_n_series_all_pm_hook_lifter_servo_motor` (score=4.6831, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_HOOK LIFTER ## SERVO MOTOR Global SOP No : Revision No: 2 Page : 3 / 106 
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=4.6331, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 3/105 ## 사고 사례 ### 1
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=4.6064, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 3/124 ## 3. 사고 사례 ### 1) 화상의 정의 불
  - [ ] `global_sop_integer_plus_all_pm_pin_motor` (score=4.5707, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_ADJ_PM_FLOATING JOINT Global SOP No: Revision No: 5 Page: 11 / 126 ## 4. 필요 Tool | | Name | Wr
  - [ ] `global_sop_geneva_xp_rep_pm_harmonic_drive` (score=4.5704, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Harmonic drive | Global SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Pag
  - [ ] `global_sop_supra_xp_all_pm_dual_epd` (score=4.5604, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_DUAL EPD Global SOP No: Revision No: 1 Page: 3/40 ## 3. 사고 사례 ### 1) 화상의 정의 불이
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=4.5551, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier | Global SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Pag
  - [ ] `global_sop_geneva_xp_rep_pm_adapter_ring` (score=4.5453, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Adapter ring Global SOP No: 0 Revision No: 0 Page: 4 / 30 ## 3. 사고 사례 ### 1) 화
  - [ ] `global_sop_geneva_xp_rep_pm_controller` (score=4.5048, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Controller Global SOP No: Revision No: 0 Page: 7 / 12 ## 8. 필요 Tool | | Name | 
  - [ ] `global_sop_geneva_xp_rep_pm_digihelic` (score=4.4908, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Digihelic Global SOP No: Revision No: 0 Page: 7/13 ## 8. 필요 Tool | | Name | Wre
  - [ ] `global_sop_geneva_xp_rep_pm_o2_cell` (score=4.4899, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GNENVA_PM_REP_O2 Cell Global SOP No: Revision No: 0 Page: 7/14 ## 8. 필요 Tool | | Name | - | | N
  - [ ] `global_sop_geneva_xp_rep_pm_elbow_heater` (score=4.4886, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Elbow heater SOP No: 0 Revision No: 0 Page: 8/22 ## 9. Part Maintenance <!-- T
  - [ ] `global_sop_supra_n_series_all_pm_gas_feed_through` (score=4.4394, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS FEED THROUGH Global SOP No: Revision No: 4 Page: 4 / 18 ## 3. 사고 사례 #
  - [ ] `global_sop_supra_n_series_all_pm_pressure_gauge` (score=4.4277, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_BARATRON GAUGE Global SOP No: Revision No: 3 Page: 10/46 ## 4. 필요 Tool | 
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pressure_switch` (score=4.4233, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVAxp_REP_PM_Vacuum Pressure switch SOP No: 0 Revision No: 0 Page: 7 / 18 ## 8. 필요 Tool | Name | Teflon 
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=4.4228, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 4/47 ## 3. 사고 사례 ### 1) 화상 재해의 
  - [ ] `global_sop_geneva_xp_rep_pm_apc` (score=4.4166, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA_REP_PM_APC SOP No: 0 Revision No: 0 Page: 7/15 ## 8. 필요 Tool | | Name | L Wrench | |---|
  - [ ] `global_sop_genevaxp_rep_pm_disc_home_sensor` (score=4.4105, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Disc home sensor Global SOP No: Revision No: 0 Page: 8 / 15 ## 8. 필요 Tool | | N
  - [ ] `global_sop_geneva_xp_rep_pm_chamber_lock` (score=4.4062, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Chamber lock Global SOP No: Revision No: 0 Page: 8 / 14 ## 8. 필요 Tool | | Name 
  - [ ] `global_sop_genevaxp_rep_pm_disc_rotation_sensor` (score=4.406, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Disc rotation sensor Global SOP No: Revision No: 0 Page: 8 / 15 ## 8. 필요 Tool |
  - [ ] `global_sop_geneva_xp_rep_pm_device_net` (score=4.4044, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Device net SOP No: 0 Revision No: 0 Page: 7/15 ## 8. 필요 Tool | | Name | mm wre
  - [ ] `global_sop_geneva_xp_rep_pm_분배기` (score=4.3455, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_분배기 Global SOP No: Revision No: 0 Page: 7 / 13 ## 8. 필요 Tool | | Name | Wrench 
  - [ ] `global_sop_geneva_xp_rep_pm_heat_exchanger` (score=4.3444, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVAxp_REP_PM_Heat exchanger Global SOP No: Revision No: 0 Page: 8 / 14 ## 8. 필요 Tool | | Name | Wrench S
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock_o_ring` (score=4.3426, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_LOAD LOCK O-RING | Global SOP No: | S-KG-R019-R0 | | --- | --- | | Revision No

#### q_id: `A-amb053`
- **Question**: 이전에 같은 문제 해결한 사례는?
- **Devices**: [SUPRA_N, SUPRA_XP, SUPRA_VPLUS, PRECIA]
- **Scope**: ambiguous | **Intent**: information_lookup
- **ES candidates** (top-28):
  - [ ] `precia_all_trouble_shooting_particle_trace` (score=7.7362, device=PRECIA, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Particle Trace] Use this guide to diagnose problems with the [Particle Trace]. It describes th
  - [ ] `supra_n_all_trouble_shooting_guide_trace_temperature_abnormal` (score=7.7186, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Temperature abnormal] Use this guide to diagnose problems with the [Trace Te
  - [ ] `integer_plus_all_trouble_shooting_guide_trace_tm_robot_abnormal` (score=7.7169, device=INTEGER plus, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace TM Robot Abnormal] Use this guide to diagnose problems with the [Trace TM Robot Abnormal
  - [ ] `supra_n_all_trouble_shooting_guide_trace_tool_shut_down` (score=7.7125, device=SUPRA N, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace tool shut down] Use this guide to diagnose problems with the [Trace tool shut down]. It 
  - [ ] `precia_pm_trouble_shooting_guide_centering_abnormal` (score=7.7122, device=PRECIA, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Centering Abnormal] Use this guide to diagnose problems with the [Centering Abnormal]. It desc
  - [ ] `supra_n_all_trouble_shooting_guide_trace_communication_abnormal` (score=7.7107, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Communication Abnormal] Use this guide to diagnose problems with the [Trace 
  - [ ] `integer_plus_all_trouble_shooting_guide_trace_efem_robot_abnormal` (score=7.7104, device=INTEGER plus, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace EFEM Robot Abnormal] Use this guide to diagnose problems with the [Trace EFE
  - [ ] `precia_all_trouble_shooting_guide_rf_matching_alarm` (score=7.7073, device=GENEVA XP, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [RF Matching Alarm] Use this guide to diagnose problems with the [RF Matching Alarm]. It descri
  - [ ] `supra_xp_pm_trouble_shooting_guide_trace_leak_rate_over` (score=7.7037, device=ZEDIUS XP, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace leak rate over] Use this guide to diagnose problems with the [Trace leak rate over]. It 
  - [ ] `supra_n_all_trouble_shooting_guide_trace_leak_rate_over` (score=7.7037, device=SUPRA N, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace leak rate over] Use this guide to diagnose problems with the [Trace leak rate over]. It 
  - [ ] `precia_pm_trouble_shooting_guide_rf_power_abnormal` (score=7.7033, device=PRECIA, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [RF Power Abnormal] Use this guide to diagnose problems with the [RF Power Abnormal]. It descri
  - [ ] `supra_xp_pm_trouble_shooting_guide_trace_chuck_abnormal` (score=7.6979, device=SUPRA XP, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Chuck Abnormal] Use this guide to diagnose problems with the [Trace Chuck abnormal
  - [ ] `precia_pm_trouble_shooting_guide_chuck_abnormal` (score=7.6965, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Chuck Abnormal] Use this guide to diagnose problems with the [Chuck Motor Alarm]. 
  - [ ] `precia_all_trouble_shooting_guide_trace_awc_abnormal` (score=7.6931, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # PRECIA Trouble Shooting Guide [AWC Trace] Use this guide to diagnose problems with the [AWC Trace]. It des
  - [ ] `supra_n_all_trouble_shooting_guide_trace_pin_move_abnormal` (score=7.6886, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace pin move abnormal] Use this guide to diagnose problems with the [Trace pin m
  - [ ] `supra_n_all_trouble_shooting_guide_trace_pcw_abnormal` (score=7.6884, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace PCW abnormal] Use this guide to diagnose problems with the [Trace PCW abnorm
  - [ ] `supran_all_trouble_shooting_guide_trace_pcw_abnormal` (score=7.6884, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace PCW abnormal] Use this guide to diagnose problems with the [Trace PCW abnorm
  - [ ] `supra_n_all_trouble_shooting_guide_trace_process_specification_out` (score=7.6866, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Process spec out] Use this guide to diagnose problems with the [Trace Proces
  - [ ] `integer_plus_all_trouble_shooting_guide_trace_pin_move_abnormal` (score=7.6847, device=INTEGER plus, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Pin Move Abnormal] Use this guide to diagnose problems with the [Trace Pin M
  - [ ] `supra_n_all_trouble_shooting_guide_trace_apc_abnormal` (score=7.6829, device=supra_n, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace APC Abnormal] Use this guide to diagnose problems with the [Trace APC Abnorm
  - [ ] `precia_pm_trouble_shooting_guide_gap_sensor_alarm` (score=7.6825, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Gap Sensor Alarm] Use this guide to diagnose problems with the [Gap Sensor Alarm].
  - [ ] `precia_all_trouble_shooting_guide_pin_motor_abnormal` (score=7.6702, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # PRECIA Trouble Shooting Guide [Pin Motor Abnormal] Use this guide to diagnose problems with the [Pin Motor
  - [ ] `supra_n_all_trouble_shooting_guide_trace_awc_abnormal` (score=7.6684, device=PRECIA, type=trouble_shooting_guide)
    > # PRECIA Trouble Shooting Guide [AWC Trace] Use this guide to diagnose problems with the [AWC Trace]. It describes the f
  - [ ] `global_sop_supra_xp_all_pm_cip_chamber` (score=7.4332, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_CIP CHAMBER | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page: 5 
  - [ ] `global_sop_supra_xp_all_pm_prism_source` (score=7.4311, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM SOURCE | Global SOP No: | | | --- | --- | | Revision No: 2 | | | Page: 5
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=7.4173, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 10 / 105 ## 사고 사례 ##
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=7.4072, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No: 13 Page: 5/75 ## 3. 사고 사례 #
  - [ ] `supra_xp_pm_trouble_shooting_guide_trace_prism_abnormal` (score=7.4057, device=SUPRA XP, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace PRISM abnormal] Use this guide to diagnose problems with the [Trace PRISM abnormal]. It 

#### q_id: `A-amb054`
- **Question**: 설비 가동률 저하 원인 분석 이력은?
- **Devices**: [SUPRA_N, SUPRA_VPLUS, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: information_lookup
- **ES candidates** (top-30):
  - [ ] `40043020` (score=6.5447, device=SUPRA Vplus, type=myservice)
    > -. 원인 분석 중
  - [ ] `global_sop_geneva_xp_rep_pm_mfc` (score=4.4302, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_MFC | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/18 | | ## 3. 
  - [ ] `global_sop_geneva_xp_rep_pm_disc` (score=4.4289, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/31 | | ## 3.
  - [ ] `set_up_manual_integer_plus` (score=4.427, device=INTEGER plus, type=SOP)
    > ```markdown # 17-27. 중간 가동 인증 | Picture | Description | Data | OK | NG | N/A | | :--- | :--- | :--- | :--- | :--- | :---
  - [ ] `global_sop_geneva_xp_adj_efem_efem_robot_leveling` (score=4.4245, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_EFEM_EFEM Robot leveling Global SOP No: Revision No: 0 Page: 3/15 ## 3. 사고 사례 ###
  - [ ] `global_sop_geneva_xp_rep_pm_pressure_switch` (score=4.4122, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Pressure switch | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/1
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pressure_switch` (score=4.4106, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Vacuum Pressure switch | SOP No: 0 | | |---|---| | Revision No: 0 | | | Page: 3
  - [ ] `global_sop_geneva_xp_rep_efem_load_port` (score=4.4103, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Load port | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/18 | | 
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pump` (score=4.4002, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Vacuum pump | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/21 | 
  - [ ] `global_sop_geneva_xp_rep_pm_heat_exchanger` (score=4.3917, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Heat exchanger Global SOP No: Revision No: 0 Page: 3 / 14 ## 3. 사고 사례 ### 1. 협착
  - [ ] `global_sop_geneva_xp_all_8계통_check` (score=4.3889, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA XP_8계통_Check sheet | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/46 | | ## 3. 사고 사례 
  - [ ] `global_sop_geneva_xp_adj_post_align_application` (score=4.3887, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_Post align application Global SOP No: 0 Revision No: 1 Page: 3/20 ## 3. 재해 방지 대책 1) 협
  - [ ] `global_sop_geneva_xp_rep_pm_chamber_lock` (score=4.3757, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Chamber lock Global SOP No: Revision No: 0 Page: 3 / 14 ## 3. 사고 사례 ### 1. 협착 위
  - [ ] `global_sop_geneva_xp_rep_pm_device_net` (score=4.0846, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Device net | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/15
  - [ ] `global_sop_geneva_xp_rep_pm_insulation_heater` (score=4.0823, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Insulation heater | SOP No: 0 | | | | | :--- | :--- | :--- | :--- | | Revision
  - [ ] `global_sop_geneva_xp_adj_all_sw_install` (score=4.0766, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_AII_SW INSTALL | SOP No: 0 | | | | --- | --- | --- | | Revision No: 0 | | | | Pag
  - [ ] `global_sop_geneva_xp_rep_pm_elbow_heater` (score=4.0757, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Elbow heater | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/
  - [ ] `global_sop_geneva_xp_rep_pm_o2_analyzer` (score=4.0587, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_O2 analyzer(Delta F) | SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Page
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock` (score=4.0448, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Load lock | SOP No: 0 | | | | | :--- | :--- | :--- | :--- | | Revision No: 1 |
  - [ ] `global_sop_geneva_xp_sw_efem_log_backup` (score=4.0391, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVAxp_SW_EFEM_Log backup Global SOP No: Revision No: 0 Page: 5/13 ## 6. Flow Chart Start ↓ 1. SOP 및 안전사항
  - [ ] `global_sop_geneva_xp_adj_pm_chuck_temp_calibration` (score=4.0214, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Chuck temp calibration Global SOP No: Revision No: 0 Page: 3/21 ## 3. 재해 방지 대책
  - [ ] `40059539` (score=3.9599, device=SUPRA Vplus, type=myservice)
    > -. 원인파악중
  - [ ] `global_sop_genevaxp_rep_efem_robot_controller` (score=3.8971, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_EFEM_Robot controller Global SOP No: Revision No: 0 Page: 4 / 16 ## 3. 사고 사례 ### 2
  - [ ] `global_sop_geneva_xp_rep_pm_controller` (score=3.8837, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Controller | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page: 3 / 
  - [ ] `global_sop_genevaxp_rep_pm_disc_rotation_sensor` (score=3.8718, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Disc rotation sensor | Global SOP No: | Revision No: 0 | Page: 4 / 15 | | --- |
  - [ ] `global_sop_geneva_xp_rep_pm_o2_analyzer_teledyne` (score=3.871, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_O2 analyzer(Teledyne -> Teledyne) | SOP No: | S-KG-R027-R1 | | --- | --- | | R
  - [ ] `global_sop_geneva_xp_rep_pm_digihelic` (score=3.8707, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Digihelic Global SOP No: Revision No: 0 Page: 3/13 ## 3. 사고 사례 ### 1. 충돌 재해 1) 
  - [ ] `global_sop_genevaxp_rep_efem_robot_end_effector` (score=3.8674, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_EFEM_Robot end effector | Global SOP No: | | | --- | --- | | Revision No: 0 | | | 
  - [ ] `global_sop_geneva_xp_rep_pm_분배기` (score=3.867, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_분배기 Global SOP No: Revision No: 0 Page: 3 / 13 ## 3. 사고 사례 ### 1. 충돌 재해 1) 충돌 재
  - [ ] `global_sop_genevaxp_rep_pm_disc_home_sensor` (score=3.856, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Disc home sensor | Global SOP No: | | |---|---| | Revision No: 0 | | | Page: 4 

#### q_id: `A-amb055`
- **Question**: 동일 모듈에서 반복 교체된 부품 이력은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: information_lookup
- **ES candidates** (top-23):
  - [ ] `global_sop_geneva_xp_rep_pm_harmonic_drive` (score=5.3048, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Harmonic drive | Global SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Pag
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=5.2939, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier | Global SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Pag
  - [ ] `global_sop_supra_n_series_all_pm_epd` (score=5.2926, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_EPD Global SOP No : Revision No: 3 Page: 3/49 ## 3. 사고 사례 ### 1) 화상 재해의 정
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=5.2919, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 66 / 105 ## 사고 사례 ##
  - [ ] `global_sop_supra_n_series_all_pm_hook_lifter_servo_motor` (score=5.277, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_HOOK LIFTER ## SERVO MOTOR Global SOP No : Revision No: 2 Page : 3 / 106 
  - [ ] `global_sop_geneva_xp_rep_pm_adapter_ring` (score=5.27, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Adapter ring Global SOP No: 0 Revision No: 0 Page: 4 / 30 ## 3. 사고 사례 ### 1) 화
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=5.2581, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 4/47 ## 3. 사고 사례 ### 1) 화상 재해의 
  - [ ] `40052963` (score=5.2404, device=SUPRA Vplus, type=myservice)
    > -. LP2 Door Close 점검
-. Lot 공정 진행 중 Open 되지 않음.
-. Close 상태 지속되어 점검 요청
-. 마지막 Alarm 5/15 일자 확인 후 현업 측 5/17 자체 조치 완료
-. 고
  - [ ] `set_up_manual_precia` (score=5.1631, device=PRECIA, type=set_up_manual)
    > | | h. Z-Axis 및 정렬된 상태에서 'TEACH' Key Click i. 'USE ROBOT' Click j. 우측 Teaching Data 값 확인 k. Z-Axis의 경우 Stage1,2를 더한 값에서 
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock_o_ring` (score=5.1535, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_LOAD LOCK O-RING | Global SOP No: | S-KG-R019-R0 | | --- | --- | | Revision No
  - [ ] `global_sop_supra_n_series_all_pm_gas_feed_through` (score=5.1402, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS FEED THROUGH Global SOP No: Revision No: 4 Page: 4 / 18 ## 3. 사고 사례 #
  - [ ] `global_sop_supra_xp_all_pm_dual_epd` (score=5.1397, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_DUAL EPD Global SOP No: Revision No: 1 Page: 3/40 ## 3. 사고 사례 ### 1) 화상의 정의 불이
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=5.1327, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 3/124 ## 3. 사고 사례 ### 1) 화상의 정의 불
  - [ ] `global_sop_geneva_xp_rep_pm_controller` (score=5.0554, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Controller Global SOP No: Revision No: 0 Page: 9 / 12 ## 10. Work Procedure | F
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=5.025, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 7) L/P2 이동_2 | a. Selected slot에서 원하는 Slot을 선택한다. <br> 
  - [ ] `set_up_manual_supra_xq` (score=4.9363, device=SUPRA XQ, type=SOP)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 2-11) Robot Speed 변경 | a. Speed Click 후 변경 | | | | | | 
  - [ ] `set_up_manual_supra_n` (score=4.8844, device=SUPRA N, type=SOP)
    > ```markdown Confidential I | 25) Buffer Stage Data 보상 | a. Cooling2에서 변경한 Data 를 Buffer Stage (Stage_R1[1]) 에 보상하여 Teach
  - [ ] `global_sop_supra_n_all_pm_chamber_open_interlock_change` (score=4.772, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_MODIFY_PM_CHAMBER_OPEN_INTERLOCK_CHANGE Global SOP No: Revision No: 3 Page: 19 / 24 | F
  - [ ] `set_up_manual_supra_nm` (score=4.7234, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.7 EFEM to TM Connection | Picture | Description | T
  - [ ] `global_sop_supra_n_series_sw_all_sw_installation_setting` (score=4.7197, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_SW_TM_CTC SW INSTALL Global SOP No : Revision No: 2 Page: 51/84 | Flow | Procedu
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3000qc` (score=4.7065, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM 3000,3100QC SOURCE IGNITION WINDOW CHECK Global SOP No: Revision No : 2 
  - [ ] `global_sop_supra_n_all_pm_fcip_r3` (score=4.629, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_PM_FCIP R3 REPLACEMENT Global SOP No : Revision No: 3 Page: 45/84 ## 8. Appendix | 
  - [ ] `set_up_manual_supra_np` (score=4.5322, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I Analog Output Tap은 조작하지 않는다. # 12. TTTM (※환경안전 보호구: 안전모, 안전화) ## 12.5 Device Net Calibration 

#### q_id: `A-amb056`
- **Question**: Leak Check 실패 원인과 재시험 절차는?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, GENEVA_XP]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-27):
  - [ ] `supra_xp_pm_trouble_shooting_guide_trace_leak_rate_over` (score=4.9036, device=ZEDIUS XP, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace leak rate over] Use this guide to diagnose problems with the [Trace leak rate over]. It 
  - [ ] `supra_n_all_trouble_shooting_guide_trace_leak_rate_over` (score=4.9036, device=SUPRA N, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace leak rate over] Use this guide to diagnose problems with the [Trace leak rate over]. It 
  - [ ] `set_up_manual_ecolite_2000` (score=4.8927, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 8. Part Installation | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) Ceramic Parts<br>
  - [ ] `integer_plus_all_trouble_shooting_guide_trace_leak_rate_over` (score=4.8871, device=INTEGER plus, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace leak rate over] Use this guide to diagnose problems with the [Trace leak rat
  - [ ] `precia_all_trouble_shooting_guide_leak_abnormal` (score=4.8445, device=INTEGER plus, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace leak rate over] Use this guide to diagnose problems with the [Trace leak rate over]. It 
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pump` (score=4.6288, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Vacuum pump SOP No: 0 Revision No: 0 Page: 17 / 21 ## 10. Work Procedure | Flo
  - [ ] `global_sop_geneva_xp_all_8계통_check` (score=4.6203, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA XP_8계통_Check sheet | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/46 | | ## 3. 사고 사례 
  - [ ] `set_up_manual_supra_vm` (score=4.5017, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 19. 인증_OHT | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) OHT 반송 인증 일정 조율 | a. 고객 OHT
  - [ ] `set_up_manual_ecolite_ii_400` (score=4.4918, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 13. 인증_OHT | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) OHT 반송 인증 일정 조율 | a. 고객 OHT
  - [ ] `supra_n_all_trouble_shooting_guide_trace_pcw_abnormal` (score=4.4347, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace PCW abnormal] Use this guide to diagnose problems with the [Trace PCW abnorm
  - [ ] `supran_all_trouble_shooting_guide_trace_pcw_abnormal` (score=4.4347, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace PCW abnormal] Use this guide to diagnose problems with the [Trace PCW abnorm
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pressure_switch` (score=4.3179, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Vacuum Pressure switch | SOP No: 0 | | |---|---| | Revision No: 0 | | | Page: 3
  - [ ] `global_sop_integer_plus_all_am_gas_line` (score=4.3163, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_AM_GAS LINE Global SOP No: Revision No: 0 Page: 3/67 ## 3. 사고 사례 ### 1) 협착 재해의
  - [ ] `global_sop_geneva_xp_rep_pm_pressure_switch` (score=4.3108, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Pressure switch | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/1
  - [ ] `global_sop_geneva_xp_rep_pm_heat_exchanger` (score=4.2979, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Heat exchanger Global SOP No: Revision No: 0 Page: 3 / 14 ## 3. 사고 사례 ### 1. 협착
  - [ ] `global_sop_geneva_xp_rep_efem_load_port` (score=4.2912, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Load port | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/18 | | 
  - [ ] `global_sop_geneva_xp_adj_efem_efem_robot_leveling` (score=4.2907, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_EFEM_EFEM Robot leveling Global SOP No: Revision No: 0 Page: 3/15 ## 3. 사고 사례 ###
  - [ ] `global_sop_geneva_xp_rep_pm_disc` (score=4.2892, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/31 | | ## 3.
  - [ ] `global_sop_geneva_xp_rep_pm_mfc` (score=4.286, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_MFC | SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 3/18 | | ## 3. 
  - [ ] `global_sop_geneva_xp_rep_pm_chamber_lock` (score=4.2828, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_PM_Chamber lock Global SOP No: Revision No: 0 Page: 3 / 14 ## 3. 사고 사례 ### 1. 협착 위
  - [ ] `global_sop_geneva_xp_adj_post_align_application` (score=4.267, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_Post align application Global SOP No: 0 Revision No: 1 Page: 3/20 ## 3. 재해 방지 대책 1) 협
  - [ ] `global_sop_supra_series_all_sw_operation` (score=4.26, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_ SW OPERATION Global SOP No Revision No: 2 Page: 10/49 ## 2. Gas line Leak Che
  - [ ] `global_sop_integer_plus_all_efem_o2_gas_leak_detector` (score=4.2568, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_EFEM_O2 GAS LEAK DETECTOR | Global_SOP No: | 0 | | --- | --- | | Revision No: 
  - [ ] `supra_xp_pm_trouble_shooting_guide_trace_prism_abnormal` (score=4.2514, device=SUPRA XP, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace PRISM abnormal] Use this guide to diagnose problems with the [Trace PRISM abnormal]. It 
  - [ ] `supra_n_all_trouble_shooting_guide_trace_apc_abnormal` (score=4.2442, device=supra_n, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace APC Abnormal] Use this guide to diagnose problems with the [Trace APC Abnorm
  - [ ] `set_up_manual_ecolite_3000` (score=4.2421, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 13. Customer Certification_인증_OHT | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) OHT 
  - [ ] `global_sop_integer_plus_all_pm_h2_gas_leak_detector` (score=4.2291, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_H2 GAS LEAK DETECTOR | Global_SOP No: | 0 | | --- | --- | | Revision No: | 

#### q_id: `A-amb057`
- **Question**: Gas Line Purge 후 잔류 가스 확인 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-24):
  - [ ] `global_sop_precia_all_pm_mfc` (score=6.6013, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_MFC Global SOP No : Revision No: 1 Page: 14 / 23 ## 6. Work Procedure | Flow | Pr
  - [ ] `set_up_manual_integer_plus` (score=6.3973, device=INTEGER plus, type=SOP)
    > ```markdown # 8. Utility Turn On - N2 Gas (환경안전 보호구: 안전모, 안전화) ## 8.2 N2 Gas Turn on | Picture | Description | Tool & Sp
  - [ ] `global_sop_supra_n_series_all_sub_unit_gas_box_board` (score=6.3574, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_GAS BOX BOARD | Global SOP No: 0 | | | --- | --- | | Revision No: 4
  - [ ] `set_up_manual_supra_n` (score=6.1875, device=SUPRA N, type=SOP)
    > Confidential I 2) Leak Check a. Pump Turn On 후 8시간 Full Pumping 후 Leak Check를 한다. b. Leak Check 조건 및 Spec은 고객사마다 상이. | |
  - [ ] `set_up_manual_supra_xq` (score=6.1676, device=SUPRA XQ, type=SOP)
    > ```markdown # 10. Toxic Gas Turn On (※환경안전 보호구: 안전모, 안전화) ## 10-1 Toxic Gas Line Check | Picture | Description | Tool & 
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=6.166, device=ZEDIUS XP, type=set_up_manual)
    > ```markdown # 10. Toxic Gas Turn On (※환경안전 보호구 : 안전모, 안전화) ## 10.1 Toxic Gas Line Check | Picture | Description | Tool &
  - [ ] `global_sop_supra_series_all_sw_operation` (score=5.9241, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_ SW OPERATION Global SOP No Revision No: 2 Page: 10/49 ## 2. Gas line Leak Che
  - [ ] `global_sop_precia_all_pm_pneumatic_valve` (score=5.8619, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_PNEUMATIC VALVE Global SOP No: Revision No: 0 Page: 3/20 ## 3. 사고 사례 ### 1) 가스 노출
  - [ ] `set_up_manual_ecolite_3000` (score=5.8585, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 11. Leak Check | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Pumping Check | a. Loca
  - [ ] `global_sop_integer_plus_all_pm_pneumatic_valve` (score=5.8582, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_PNEUMATIC VALVE Global SOP No: Revision No: 1 Page: 3/20 ## 3. 사고 사례 ### 1)
  - [ ] `global_sop_supra_n_all_sub_unit_mfc` (score=5.8571, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_SUB UNIT_MFC Global SOP No: 0 Revision No: 3 Page: 3/21 ## 3. 사고 사례 ### 1) 가스 노출 재해
  - [ ] `global_sop_supra_n_all_sub_unit_igs_block` (score=5.8519, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_SUB UNIT_IGS BLOCK Global SOP No : Revision No: 1 Page: 3/67 ## 3. 사고 사례 ### 1) 가스 
  - [ ] `global_sop_integer_plus_all_tm_filter` (score=5.8446, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_FILTER Global_SOP No: Revision No: 0 Page: 3/20 ## 3. 사고 사례 ### 1) 가스 노출 재해
  - [ ] `global_sop_integer_plus_all_tm_epc` (score=5.8391, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_ INTEGER plus_ALL_TM_EPC Global_SOP No: Revision No: 0 Page: 3/21 ## 3. 사고 사례 ### 1) 가스 노출 재해의 
  - [ ] `global_sop_supra_xp_all_sub_unit_igs_block` (score=5.8291, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_SUB UNIT_IGS BLOCK | Global SOP No: | | | --- | --- | | Revision No: 2 | | | Page
  - [ ] `global_sop_integer_plus_all_pm_igs_block` (score=5.8282, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_IGS BLOCK | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page: 3
  - [ ] `global_sop_supra_n_series_all_sub_unit_pressure_vacuum_switch` (score=5.7, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_PRESSURE & VACUUM SWITCH Global SOP No : Revision No: 1 Page: 3/28 
  - [ ] `global_sop_integer_plus_all_pm_gas_line` (score=5.64, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_GAS LINE Global_SOP No: Revision No: 0 Page: 3/75 ## 3. 사고 사례 ### 1) 협착 재해의
  - [ ] `set_up_manual_supra_np` (score=5.6185, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I | 2) Jig Assy 조립도 확인 | a. Jig Assy 조립도를 확인하여 다음과 같은 방법으로 Jig Assy가 설치되어야 한다. | | | :--- | :--
  - [ ] `set_up_manual_ecolite_2000` (score=5.5839, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 10. Leak Check | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Pumping Check | a. Loca
  - [ ] `set_up_manual_precia` (score=5.5147, device=PRECIA, type=set_up_manual)
    > | | | | |---|---|---| | 3. Gas box manual valve open | a. Gas Regulator Full open<br>b. Gas manual valve lock key 제거<br>
  - [ ] `set_up_manual_ecolite_ii_400` (score=5.4438, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 11. Leak Check | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) Leak Check | a. Leak Ch
  - [ ] `global_sop_supra_xp_all_pm_cip_chamber` (score=5.3951, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_CIP CHAMBER Global SOP No: Revision No : 0 Page : 27 /37 | Flow | 절차 | Tool & 
  - [ ] `precia_all_trouble_shooting_particle_trace` (score=5.3501, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Particle Trace] Confidential II | Failure symptoms | Check point | Key point | | :

#### q_id: `A-amb058`
- **Question**: Power Supply Fan Alarm 발생 시 교체 기준은?
- **Devices**: [SUPRA_N, SUPRA_XP, INTEGER_PLUS, PRECIA]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-15):
  - [ ] `40080973` (score=6.2641, device=INTEGER IVr, type=myservice)
    > -. DVM 측정 시 Power Supply(PS0-5) Fail 추정
-> Safety Module 0-2, 0-3 LED Off 상태로 인한 TM TOP LID OPEN ITLK Alarm 발생
  - [ ] `40059539` (score=5.7728, device=SUPRA Vplus, type=myservice)
    > -. EFEM FFU FAN2 RPM DROP Alarm 발생
-. Shingsung Eg'r 내방하여 EFEM FFU1,2 LIU <-> FFU Fan1,2 연결 Cable 2종 교체
-> 교체 진행후 FFU Fa
  - [ ] `global_sop_supra_n_series_all_rack` (score=5.7093, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_RACK_DC POWER SUPPLY REPLACEMENT Global SOP No : Revision No: 1 Page: 62/84 
  - [ ] `40054176` (score=5.6634, device=TIGMA Vplus, type=myservice)
    > "EPD COMMUNICATION ALARM" occurred
EPD module display whites out
POWER and FAN LED off (POWER blinks, FAN lit when norma
  - [ ] `global_sop_supra_xp_all_tm_dc_power_supply` (score=5.2848, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ REP_TM_DC POWER SUPPLY Global SOP No: Revision No: 0 Page: 14 / 18 ## 6. Work Proced
  - [ ] `global_sop_precia_all_efem_dc_cooling_fan_motor` (score=5.2676, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_EFEM_DC # COOLING FAN MOTOR Global SOP No: Revision No: 0 Page: 12 / 15 ## 10. Work 
  - [ ] `global_sop_supra_xp_all_efem_ffu` (score=5.0068, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_EFEM_FFU & FILTER (SAFAS) Global SOP No: Revision No : 1 Page : 55/60 | Flow | Pr
  - [ ] `global_sop_supra_n_series_all_sub_unit_elt_box_assy` (score=4.8903, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_SUB ## UNIT_DC POWER SUPPLY Global SOP No: Revision No: 0 Page :74 / 95 | Fl
  - [ ] `global_sop_supra_n_series_all_efem_ffu` (score=4.8384, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_EFEM_FFU | Global SOP No: | | | --- | --- | | Revision No: 4 | | | Page: 2/5
  - [ ] `set_up_manual_supra_xq` (score=4.7323, device=SUPRA XQ, type=SOP)
    > ```markdown # 5. Power Turn On (※환경안전 보호구: 안전모, 안전화) ## 5-1 Main Rack | Picture | Description | Tool & Spec | | :--- | :
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=4.6415, device=ZEDIUS XP, type=set_up_manual)
    > ```markdown # 5. Power Turn On (※환경안전 보호구 : 안전모, 안전화, 보안경) ## 5.1 Main Rack | Picture | Description | Tool & Spec | | :-
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=4.5597, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 1. Install Preparation(※환경안전 보호구 : 안전모, 안전화) ## 1.1 Foot Print Drawing | Picture | Description | Tool & Sp
  - [ ] `40052189` (score=4.5583, device=SUPRA V, type=myservice)
    > -. 고객 요청으로 Power Supply 탈착하여 콘덴서 확인
->  콘덴서 확인시 2개 부풀어서 터짐 확인
-. 고객 전자기기 수리실 수리 후 장착
-. 장착 후 정상 동작 확인
-. Monitoring
  - [ ] `40035345` (score=4.5324, device=SUPRA V, type=myservice)
    > -. LOG 확인 시 FFU 점검 이전 부터 EFEM TO CTC Communication Alarm 발생
-> EFEM TO CTC Communication LOG 끊김 확인
-> 고객 Inform 완료
-. PM
  - [ ] `supra_n_all_trouble_shooting_guide_trace_pin_move_abnormal` (score=4.4545, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace pin move abnormal] Confidential II ## Appendix #2 ### A-4 Pendant #### 13-2-

#### q_id: `A-amb059`
- **Question**: Chuck Surface Damage 확인 및 교체 판단 기준은?
- **Devices**: [SUPRA_N, SUPRA_VPLUS, SUPRA_XP, PRECIA]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-24):
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=5.3631, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 1. Install Preparation(※환경안전 보호구 : 안전모, 안전화) ## 1.1 Foot Print Drawing | Picture | Description | Tool & Sp
  - [ ] `set_up_manual_ecolite_3000` (score=5.2633, device=ECOLITE3000, type=set_up_manual)
    > ```markdown | 1. Installation Preperation (Layout, etc.) | | | | :--- | :--- | :--- | | Picture | Description | Tool & S
  - [ ] `40061148` (score=5.2529, device=TIGMA Vplus, type=myservice)
    > -.ST2 hearter chuck remove
-.ST2 lift pin 1ea rep(side surface damage)
-.ST2 hearter chuck inatall
-.ST1.2 pin height,po
  - [ ] `set_up_manual_ecolite_2000` (score=5.239, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 1. Installation Preperation (Layout, etc.) | Picture | Description | Tool & Spec | | :--- | :--- | :--- | 
  - [ ] `set_up_manual_ecolite_ii_400` (score=5.1171, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 1. Installation Preperation_Template Draw | Picture | Description | Tool & Spec | | :--- | :--- | :--- | |
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=5.0368, device=ZEDIUS XP, type=set_up_manual)
    > ```markdown # 1. Install Preperation (※환경안전 보호구 : 안전모, 안전화) ## 1.1 Foot Print Drawing | Picture | Description | Tool & S
  - [ ] `set_up_manual_supra_vm` (score=4.8749, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 1. Template Draw | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) Template Drawing 사전 준
  - [ ] `set_up_manual_supra_xq` (score=4.6115, device=SUPRA XQ, type=SOP)
    > ```markdown # 1. Install Preparation (Layout, etc) (※환경안전 보호구: 안전모, 안전화) | Picture | Description | Tool & Spec | | :--- 
  - [ ] `global_sop_geneva_xp_adj_heater_chuck_leveling` (score=4.5982, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA_ADJ_Heater chuck leveling Global SOP No: Revision No: 0 Page: 11/16 ## 10. Work Procedur
  - [ ] `set_up_manual_supra_nm` (score=4.5169, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.13 ATM Transfer | Picture | Description | Tool & Sp
  - [ ] `40044046` (score=4.3551, device=SUPRA Vplus, type=myservice)
    > -. PM1 Bellows 탈착
-. Bellows 교체 후 Leak
-> 26mTorr/min
-> 추가 Pumping 후 확인 요청
-. TM Robot Teaching
-> CH1,2 Teaching 상태 않조
  - [ ] `global_sop_integer_plus_all_ll_stepping_motor` (score=4.3084, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_ADJ_LL STEPPING ## MOTOR Pulley 높이 측정 Jig Global SOP No: Revision No: Page : 57 / 68 ### 5. Fl
  - [ ] `set_up_manual_integer_plus` (score=4.2391, device=INTEGER plus, type=SOP)
    > ```markdown # 17-8. Cable Hook up | Picture | Description | Data | OK | NG | N/A | | :--- | :--- | :--- | :--- | :--- | 
  - [ ] `set_up_manual_supra_n` (score=4.143, device=SUPRA N, type=SOP)
    > ```markdown Confidential I 4) Wafer 매수 설정 후 Strat <!-- Image (72, 75, 357, 247) --> a. ATM Transfer를 진행할 Wafer Slot 선택 후
  - [ ] `set_up_manual_supra_np` (score=4.1348, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I 4) Wafer 매수 설정 후 Strat <!-- Image (62, 72, 344, 247) --> a. ATM Transfer를 진행할 Wafer Slot 선택 후
  - [ ] `global_sop_supra_xp_all_pm_heater_chuck` (score=4.0956, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP ZEDIUS XP_ALL_PM_CERAMIC PLATE Global SOP No: 0 Revision No: 2 Page: 28 / 49 ## Scope 이 Global 
  - [ ] `global_sop_integer_plus_all_am_pin_motor` (score=4.0928, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_PIN BELLOWS Global SOP No: Revision No: 4 Page: 30 / 84 | Flow | Procedure 
  - [ ] `40072912` (score=4.0557, device=SUPRA Vplus, type=myservice)
    > -. CH1 Chuck, Ceramic Plate 탈착
-> Chuck Screw 조금 풀려있음, 고객 확인 시 이력없음
-. Ceramic Plate 및 O-ring 확인 시 특이사항 없음
-. Ceramic Pl
  - [ ] `40042934` (score=4.0539, device=SUPRA Vplus, type=myservice)
    > -. Hand3 Vac Error확인
-. Upper arm Cover open후 확인시 Fitting및 Vac hose 갈림확인
-. 내부Fitting및 Vac hose교체진행
-. Aging 1Lot진행시 Vac
  - [ ] `precia_pm_trouble_shooting_guide_chuck_abnormal` (score=4.0363, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Chuck Abnormal] Use this guide to diagnose problems with the [Chuck Motor Alarm]. 
  - [ ] `global_sop_integer_plus_all_am_heater_chuck` (score=3.9953, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_HEATER CHUCK Global SOP No: Revision No: 1 Page: 15 / 25 | Flow | Procedure
  - [ ] `40091880` (score=3.9942, device=TIGMA Vplus, type=myservice)
    > There is a mark like a white scratch on the surface
  - [ ] `global_sop_supra_vplus_all_pm_controller` (score=3.9817, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_SW_PM_Controller PATCH Global SOP No: Revision No: 3 Page: 38 / 42 | Flow | Procedu
  - [ ] `supra_xp_pm_trouble_shooting_guide_trace_chuck_abnormal` (score=3.9804, device=SUPRA XP, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Chuck Abnormal] Confidential II | Temp Stable Time Out Alarm | B -1. Heater Chuck 

#### q_id: `A-amb060`
- **Question**: Cooling Water Flow 이상 시 점검 순서는?
- **Devices**: [SUPRA_N, SUPRA_XP, SUPRA_VPLUS, GENEVA_XP]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-22):
  - [ ] `supra_n_all_trouble_shooting_guide_trace_microwave_abnormal` (score=6.3635, device=SUPRA N, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace Microwave Abnormal] ## 8.5.13 Water Temperature Low - Water Temperature Low - Generator 
  - [ ] `global_sop_supra_xp_all_tm_ctc` (score=5.6289, device=ZEDIUS XP, type=SOP)
    > # Global SOP_ZEDIUS XP_ALL_TM_CTC REPLACEMENT | Global SOP No : | Revision No: 1 | Page : 24 / 45 | |---|---|---| | Flow
  - [ ] `global_sop_supra_n_series_all_tm_ctc` (score=5.5333, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_CTC REPLACEMENT Global SOP No : Revision No: 6 Page: 26/81 | Flow | Proce
  - [ ] `set_up_manual_supra_vm` (score=5.4238, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 2. Undocking 및 Module 이동 | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Module 반입 순서 
  - [ ] `global_sop_supra_n_series_all_sub_unit_water_shut_off_valve` (score=5.3541, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_SUB UNIT_WATER # SHUT OFF VALVE Global SOP No : Revision No: 2 Page: 14 / 16
  - [ ] `set_up_manual_supra_nm` (score=5.3324, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.1. Common List ### 12.1.3 Cooling Stage Pin Speed Adj
  - [ ] `global_sop_supra_n_series_all_pcw_turn_on` (score=5.2642, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_COOLING STAGE ## PCW TURN ON Global SOP No: Revision No: 1 Page: 24 / 31 | F
  - [ ] `global_sop_precia_all_pm_prevent_maintenance` (score=5.1995, device=PRECIA, type=SOP)
    > # Global SOP_PRECIA_REP_PM_PROCESS KIT (TOP MOUNT TYPE) Global SOP No: Revision No: 5 Page : 33 / 108 ## 6. Work Procedu
  - [ ] `global_sop_integer_plus_all_pm_pin_motor` (score=5.1785, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_LIFTER PIN ASSY Global SOP No: Revision No: 5 Page: 58 / 126 | Flow | Proce
  - [ ] `global_sop_integer_plus_all_tm_ctc_controller` (score=5.121, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_CTC # CONTROLLER Global SOP No: Revision No: 1 Page: 40/51 | Flow | Procedu
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=4.8727, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Module 반입 순서 (Rear) | a. Module의 반입 순서에 따라 해당 Module
  - [ ] `set_up_manual_ecolite_2000` (score=4.8368, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 7. Teaching_Aliner(Cooling) Stage | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | ![](ima
  - [ ] `set_up_manual_supra_np` (score=4.7892, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I | 3) TM 하단 Side Door Open | a. PM 방향에서 TM 하단 Side Door 를 Open 한다. | Wrench | | :--- | :--- | 
  - [ ] `global_sop_supra_n_series_all_sub_unit_manometer` (score=4.7331, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_SUB UNIT _MANOMETER Global SOP No: Revision No: 0 Page: 28/32 | Flow | Proce
  - [ ] `global_sop_precia_adj_all_utility_turn_onoff` (score=4.7275, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_ALL_UTILITY TURN ON/OFF Global SOP No : Revision No: 0 Page : 26 / 44 ## 10. Work Pr
  - [ ] `40066837` (score=4.7079, device=SUPRA Vplus, type=myservice)
    > -. 고객측 점검 시 IB Flow 후단 Speed CTR 연결되어있음
-> Speed CTR 방향 반대로 체결
-> Speed CTR 제거 후 IB Flow Calibration 시 Pin Up 간 튕김 현상
->
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=4.6914, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM ROBOT TEACHING Global SOP No : Revision No: 6 Page : 78 / 107 | Flow | Pr
  - [ ] `set_up_manual_supra_n` (score=4.6771, device=SUPRA N, type=SOP)
    > ```markdown Confidential I | | | | |---|---|---| | | | | | 2) Calibration Pendant 준비 | a. Pin Time 조절 전 IB Flow Pendant 
  - [ ] `set_up_manual_ecolite_3000` (score=4.6635, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 2. Tool Fab In (Packing List Check) | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 5) Mo
  - [ ] `global_sop_supra_n_series_all_tm_buffer_module` (score=4.5942, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_BUFFER MODULE # CLEAN Global SOP No : Revision No: 3 Page: 14 / 33 | Flow
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=4.5869, device=ZEDIUS XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 4) Module 반입 순서 (Front) | a. Module의 반입 순서에 따라 해당 Modul
  - [ ] `set_up_manual_supra_xq` (score=4.5747, device=SUPRA XQ, type=SOP)
    > ```markdown | 2. Tool Fab in (Packing List Check) (※환경안전 보호구: 안전모, 안전화) | | | | :--- | :--- | :--- | | 2-1 Unpacking 및 M

#### q_id: `A-amb061`
- **Question**: Chamber 내 Arcing 흔적 발견 시 조치 절차는?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, OMNIS]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-24):
  - [ ] `global_sop_integer_plus_all_ll_stepping_motor` (score=5.1642, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_LL STEPPING ## MOTOR Pulley 높이 측정 Jig Global SOP No: Revision No: Page: 60 / 6
  - [ ] `40044731` (score=4.6812, device=SUPRA Vplus, type=myservice)
    > -. PM3 Cleaning
-> Baffle Screw 및 Washer 교체
-> CH2 11h 방향 Back Stream 흔적 보임
-> Chamber 내부 Cleaning 및 N2 Blowing 진행
-. Pr
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3100qc` (score=4.6172, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PRISM SOURCE ## 3100QC REPLACEMENT Global SOP No: Revision No : 1 Page : 20/72 | 
  - [ ] `global_sop_supra_xp_all_pm_cip_chamber` (score=4.5742, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_CIP CHAMBER Global SOP No: Revision No : 0 Page : 20 /37 | Flow | 절차 | Tool & 
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=4.5536, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No : 13 Page : 21/75 | Flow | 절
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3000qc` (score=4.398, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PRISM SOURCE 3000QC REPLACEMENT Global SOP No: Revision No : 2 Page : 20/82 | Flo
  - [ ] `global_sop_supra_n_series_all_pm_top_lid` (score=4.3091, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N Series_CLN_PM_TOP LID Global SOP No: Revision No: 0 Page: 17 / 48 | Flow | | Tool & Poi
  - [ ] `global_sop_supra_n_series_all_pm_process_kit` (score=4.2897, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_CLN_PM_CHAMBER Global SOP No : Revision No: 0 Page: 42/55 | Flow | Procedure | T
  - [ ] `global_sop_supra_n_series_all_sub_unit_temp_controller` (score=4.2783, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_ TEMP CONTROLLER Global SOP No : Revision No: 2 Page: 2/56 ## 1. Sa
  - [ ] `global_sop_supra_xp_all_pm_chamber_safety_cover` (score=4.2701, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_SAFETY COVER REPLACEMENT | Global SOP No : | | | --- | --- | | Revision No : 1
  - [ ] `global_sop_integer_plus_all_ll_mfc` (score=4.229, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_LL_MFC | Global SOP No: | | |---|---| | Revision No: 1 | | | Page: 1/20 | | ##
  - [ ] `global_sop_integer_plus_all_am_baffle` (score=4.2239, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_ INTEGER plus_REP_AM_BAFFLE Global_SOP No: Revision No: 1 Page: 6/18 ## Scope 이 Global SOP는 INT
  - [ ] `global_sop_integer_plus_all_ll_sensor_board` (score=4.2143, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_REP_LL_SENSOR BOARD Global SOP No: Revision No: 1 Page: 6 / 17 ## Scope 이 Global SOP는 INTEGER 
  - [ ] `global_sop_supra_n_series_all_pm_gas_spring` (score=4.1309, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS SPRING Global SOP No: Revision No: 4 Page: 2/22 ## 1. Safety 1) 안전 및 
  - [ ] `set_up_manual_supra_n` (score=4.123, device=SUPRA N, type=SOP)
    > ```markdown # 0. Safety ## Picture 5) 환경안전 보호구 - **작업/장소 별 필수 안전보호구** 하기 보호구는 필수적 착용 보호구임./각 작업에 따라 추가적 보호구 착용이 필요함 <!--
  - [ ] `set_up_manual_ecolite_2000` (score=4.1172, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 0. Safety ## Picture ### 5) 환경안전 보호구 - **작업/장소 별 필수 안전보호구** 하기 보호구는 필수적 착용 보호구임./각 작업에 따라 추가적 보호구 착용이 필요함 
  - [ ] `global_sop_integer_plus_all_pm_manometer` (score=4.1073, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER Plus_REP_MANOMETER Global SOP No: Revision No: 1 Page: 20 / 20 ## 8. Appendix | Flow | 
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=4.0943, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All | Global SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Page: 1/4
  - [ ] `set_up_manual_supra_nm` (score=4.0941, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 0. Safety ## Picture ### 5) 환경안전 보호구 - **작업/장소 별 필수 안전보호구** 하기 보호구는 필수적 착용 보호구임./각 작업에 따라 추
  - [ ] `global_sop_supra_n_series_all_pm_chamber_hinge` (score=4.0845, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_CHAMBER HINGE Global SOP No : Revision No: 3 Page: 1/22 ## Scope 이 Global
  - [ ] `set_up_manual_supra_xq` (score=4.075, device=SUPRA XQ, type=SOP)
    > ```markdown # 0. Safety ## Picture ### 5) 환경안전 보호구 - **작업/장소 별 필수 안전보호구** - 하기 보호구는 필수적 착용 보호구임./각 작업에 따라 추가적 보호구 착용이 필요
  - [ ] `set_up_manual_ecolite_3000` (score=4.0737, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 0. Safety ## Picture ### 5) 환경안전 보호구 - **작업/장소 별 필수 안전보호구** - 하기 보호구는 필수적 착용 보호구임./각 작업에 따라 추가적 보호구 착용이 필요
  - [ ] `set_up_manual_supra_np` (score=4.0654, device=SUPRA Np, type=set_up_manual)
    > ```markdown # 0. Safety ## Picture ### 5) 환경안전 보호구 - **작업/장소 별 필수 안전보호구** 하기 보호구는 필수적 착용 보호구임./각 작업에 따라 추가적 보호구 착용이 필요함 
  - [ ] `set_up_manual_integer_plus` (score=4.0642, device=INTEGER plus, type=SOP)
    > ```markdown # 0. Safety ## Picture ### 5) 환경안전 보호구 - **작업/장소 별 필수 안전보호구** 하기 보호구는 필수적 착용 보호구임./각 작업에 따라 추가적 보호구 착용이 필요함 

#### q_id: `A-amb062`
- **Question**: Pin Hole 발생 시 원인 분석 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-18):
  - [ ] `40043020` (score=6.4454, device=SUPRA Vplus, type=myservice)
    > -. 원인 분석 중
  - [ ] `global_sop_supra_n_series_all_pm_heater_chuck` (score=5.6541, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_HEATER CHUCK Global SOP No : Revision No: 2 Page: 19/40 | Flow | Procedur
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=5.4141, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_HOOK LIFTER ## SERVO MOTOR CONTROLLER REPLACEMENT Global SOP No: Revision No: 
  - [ ] `precia_all_trouble_shooting_guide_pin_motor_abnormal` (score=5.3205, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # PRECIA Trouble Shooting Guide [Pin Motor Abnormal] Confidential II | Failure symptoms | Check point | Key 
  - [ ] `set_up_manual_supra_nm` (score=4.9955, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.2 EFEM Robot Leveling & Loadport Teaching | Picture
  - [ ] `set_up_manual_precia` (score=4.9308, device=PRECIA, type=set_up_manual)
    > ```markdown # 4. Docking (※환경안전 보호구: 안전모, 안전화) ## 4.3 TM - PM Docking | Picture | Description | Tool & Spec | | :--- | :
  - [ ] `global_sop_geneva_xp_sw_efem_log_backup` (score=4.8129, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_SW_EFEM_Log backup Global SOP No: Revision No: 0 Page: 9/13 ## 10. Work Procedure | Fl
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=4.7092, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `set_up_manual_supra_np` (score=4.6683, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I | 2) Jig Assy 조립도 확인 | a. Jig Assy 조립도를 확인하여 다음과 같은 방법으로 Jig Assy가 설치되어야 한다. | | | :--- | :--
  - [ ] `supra_n_all_trouble_shooting_guide_trace_pin_move_abnormal` (score=4.6678, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace pin move abnormal] Use this guide to diagnose problems with the [Trace pin m
  - [ ] `set_up_manual_supra_n` (score=4.6357, device=SUPRA N, type=SOP)
    > ```markdown Confidential I 36) LP2,3 Teaching <!-- Image (70, 70, 290, 226) --> a. LP1 Teaching 과 동일한 방법으로 LP2,3 를 진행하여 
  - [ ] `integer_plus_all_trouble_shooting_guide_trace_pin_move_abnormal` (score=4.5616, device=INTEGER plus, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Pin Move Abnormal] Use this guide to diagnose problems with the [Trace Pin M
  - [ ] `supra_xp_pm_trouble_shooting_guide_trace_pin_move_abnormal` (score=4.519, device=SUPRA XP, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace pin move abnormal] Use this guide to diagnose problems with the [Trace pin move abnormal
  - [ ] `global_sop_geneva_xp_rep_pm_support_pin` (score=4.4329, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Support pin Global SOP No: 0 Revision No: 1 Page: 15 / 25 ## 10. Work Procedur
  - [ ] `precia_pm_trouble_shooting_guide_centering_abnormal` (score=4.4181, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Centering Abnormal] Confidential II | Failure symptoms | Check point | Key point |
  - [ ] `global_sop_geneva_xp_adj_pm_pin_alignment` (score=4.3584, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Pin Alignment Global SOP No: S-KG-A003-R0 Revision No: 0 Page: 15/25 ## 10. Wo
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=4.3141, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_EFEM_ROBOT TEACHING Global SOP No: Revision No: 3 Page: 25 / 126 | Flow | Proc
  - [ ] `set_up_manual_integer_plus` (score=4.3134, device=INTEGER plus, type=SOP)
    > ```markdown | | Nut Jointing 과정에서 Chamber Level 이 들어지지 않도록 주의 | | |---|---|---| | | | | | 3. Docking 및 Leveling (※환경안전 보

#### q_id: `A-amb063`
- **Question**: Mass Flow Controller Zero Shift 발생 시 조치는?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, GENEVA_XP]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-23):
  - [ ] `global_sop_integer_plus_all_tm_robot` (score=5.6333, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_TM_ROBOT TEACHING Global SOP No : Revision No: 4 Page: 45 / 103 | Flow | Proce
  - [ ] `global_sop_supra_n_series_all_tm_ib_flow` (score=5.3367, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_IB FLOW Global SOP No : Revision No: 3 Page: 15/23 | Flow | Procedure | T
  - [ ] `global_sop_supra_xp_all_tm_device_net_board` (score=4.9861, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_DEVICE NET BOARD ## ANALOG CALIBRATION Global SOP No : Revision No: 1 Page: 29
  - [ ] `global_sop_integer_plus_all_am_devicenet_board` (score=4.9483, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_AM_AIO # CALIBRATION [PSK BOARD] Global SOP No : Revision No: 0 Page: 18/58 | 
  - [ ] `global_sop_integer_plus_all_pm_d_net` (score=4.9416, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_PM_AIO # CALIBRATION [PSK BOARD] Global SOP No : Revision No: 0 Page: 18/56 | 
  - [ ] `global_sop_supra_n_series_all_pm_device_net_board` (score=4.8466, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_DEVICE NET BOARD Global SOP No : Revision No: 2 Page: 33/44 | Flow | Proc
  - [ ] `global_sop_supra_xp_all_pm_pressure_gauge` (score=4.8118, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRESSURE GAUGE Global SOP No: Revision No: 1 Page: 28 / 34 | Flow | Procedure 
  - [ ] `global_sop_precia_all_pm_manometer` (score=4.8106, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_MANOMETER Global SOP No: Revision No: 0 Page: 20 / 20 ## 8. Appendix | Flow | Proced
  - [ ] `global_sop_supra_xp_all_pm_pirani_gauge` (score=4.8092, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_REP_PM_PIRANI GAUGE Global SOP No: Revision No: 1 Page: 34 / 40 | Flow | Procedure | 
  - [ ] `global_sop_supra_n_all_pm_fcip_r3` (score=4.7743, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_PM_FCIP R3 REPLACEMENT Global SOP No : Revision No: 3 Page: 39/84 | Flow | Procedur
  - [ ] `40050881` (score=4.6828, device=TIGMA Vplus, type=myservice)
    > 4/13 Night shift
-. Removed BM2,3 stage
4/14 Day shift
-. Replaced BM2,3 Align cylinder(8ea.)
*One Speed controller dama
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=4.593, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_EFEM_ROBOT Global SOP No: Revision No: 3 Page : 121 / 126 | Flow | Procedure |
  - [ ] `global_sop_integer_plus_all_pm_pin_motor` (score=4.5861, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_PIN MOTOR CONTROLLER Global SOP No: Revision No: 5 Page : 101 / 126 | Flow 
  - [ ] `global_sop_supra_xp_all_efem_robot_sr8241` (score=4.5176, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_EFEM_ROBOT_SR8241 Global SOP No : Revision No: 2 Page: 14 / 34 | Flow | Procedure
  - [ ] `supra_n_all_trouble_shooting_guide_trace_apc_abnormal` (score=4.4977, device=supra_n, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace APC Abnormal] | Failure | Check | Action | | :--- | :--- | :--- | | Pressure reading is 
  - [ ] `40055879` (score=4.484, device=SUPRA Vplus, type=myservice)
    > -. Air Flow 정상 확인
-. 고객 Speed Controller 조절로 IB Flow 틀어짐 추정
-> IB Flow 조절 후 Pin Down 2.0s
-. 고객사 모니터링 예정
  - [ ] `global_sop_integer_plus_all_tm_safety_controller` (score=4.4121, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus REP_TM_SAFETY CONTROLLER Global_SOP No: Revision No: 0 Page: 14 / 19 | Flow | Proc
  - [ ] `global_sop_integer_plus_all_pm_safety_controller` (score=4.406, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_SAFETY CONTROLLER Global_SOP No: Revision No: 0 Page: 14 / 19 | Flow | Proc
  - [ ] `set_up_manual_integer_plus` (score=4.3991, device=INTEGER plus, type=SOP)
    > ```markdown # 17-23. MFC 인증 | Picture | Description | Data | OK | NG | N/A | |---|---|---|---|---|---| | | PM, AM Chambe
  - [ ] `global_sop_integer_plus_all_am_pin_motor` (score=4.3981, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_PIN MOTOR CONTROLLER Global SOP No: Revision No: 4 Page: 73 / 84 | Flow | P
  - [ ] `global_sop_supra_xp_all_efem_controller` (score=4.3908, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_EFEM_CONTROLLER REPLACEMENT Global SOP No: Revision No: 1 Page: 15 /46 | Flow | P
  - [ ] `global_sop_integer_plus_all_efem_eda_controller` (score=4.3856, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_EFEM_EDA CONTROLLER Global SOP No: Revision No: 1 Page: 6/22 ## Scope 이 Global
  - [ ] `supra_xp_sub_unit_trouble_shooting_guide_igs_block_abnormal` (score=4.3644, device=SUPRA XP, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [IGS Block Abnormal] Confidential II | Failure symptoms | Check point | Key point |

#### q_id: `A-amb064`
- **Question**: Throttle Valve Hunting 발생 시 점검 항목은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-28):
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=5.7277, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 14. Customer certification | Picture | Description | Spec | Check | Result | | :--- | :--- | :--- | :--- |
  - [ ] `set_up_manual_supra_xq` (score=4.7514, device=SUPRA XQ, type=SOP)
    > | 55 | PM | DN131 | D-NET | □Y □N | 95 | PM | DN131 | D-NET | □Y □N | |---|---|---|---|---|---|---|---|---|---| | 56 | P
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=4.6177, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `set_up_manual_supra_np` (score=4.4226, device=SUPRA Np, type=set_up_manual)
    > | 9) SUB2 I/F Panel - PM Cable Hook up | a. Sub Unit2 의 PM 방향 Interface Panel Inner Cable Hook up 을 진행한다. b. 장착되는 Cable 
  - [ ] `global_sop_supra_n_series_all_tm_solenoid_valve` (score=4.3507, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_SOLENOID VALVE REPLACEMENT Global SOP No : Revision No: 2 Page : 13 / 18 
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=4.2778, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_AM_PENDULUM VALVE Global SOP No: Revision No: Page : 70 / 135 | Flow | Procedu
  - [ ] `set_up_manual_supra_n` (score=4.2103, device=SUPRA N, type=SOP)
    > ```markdown # 4. Cable Hook Up (※환경안전 보호구 : 안전모, 안전화, 안전대, 보안경 보호가운, 헤드랜턴) ## 4.4 Sub Unit Cable Hook Up | Picture | Des
  - [ ] `global_sop_integer_plus_all_am_slot_valve` (score=4.2074, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_SLOT VALVE Global SOP No : Revision No : 0 Page : 17 / 39 | Flow | Procedur
  - [ ] `all_all_trouble_shooting_guide_trace_ffu_abnormal` (score=4.1557, device=etc, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace FFU Abnormal] Confidential II | Failure symptoms | Check point | Key point |
  - [ ] `global_sop_geneva_xp_all_8계통_check` (score=4.1526, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA XP_8계통_Check sheet SOP No: 0 Revision No: 0 Page: 18 / 46 ## 10. Work Procedure | Flow | Procedure 
  - [ ] `global_sop_integer_plus_all_am_solenoid_valve` (score=4.1357, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_SOLENOID VALVE Global SOP No: Revision No: 1 Page: 16 / 19 | Flow | Procedu
  - [ ] `global_sop_integer_plus_all_pm_slot_valve` (score=4.1318, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_SLOT VALVE Global SOP No : Revision No : 2 Page : 17 / 39 | Flow | Procedur
  - [ ] `set_up_manual_supra_nm` (score=4.126, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 4. Cable Hook Up (※환경안전 보호구 : 안전모, 안전화, 안전대, 보안경 보호가운, 헤드랜턴) ## 4.4 Sub Unit Cable Hook Up 
  - [ ] `global_sop_integer_plus_all_ll_lifter_assy` (score=4.1196, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_LL_BUSH, BELLOWS, SHAFT Global SOP No : Revision No: 0 Page: 36/81 | 작업 | Chec
  - [ ] `global_sop_integer_plus_all_ll_slot_valve` (score=4.0746, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_LL_SLOT VALVE Global SOP No : Revision No : 4 Page : 20 / 50 | Flow | Procedur
  - [ ] `supra_xp_tm_trouble_shooting_guide_trace_slot_valve_abnormal` (score=4.0411, device=SUPRA XP, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Slot valve abnormal] Use this guide to diagnose problems with the [Trace Slo
  - [ ] `supra_xp_all_trouble_shooting_guide_trace_slot_valve_abnormal` (score=4.0411, device=SUPRA XP, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Slot valve abnormal] Use this guide to diagnose problems with the [Trace Slo
  - [ ] `global_sop_integer_plus_all_pm_manometer` (score=4.0379, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER Plus_REP_MANOMETER Global SOP No: Revision No: 1 Page: 20 / 20 ## 8. Appendix | Flow | 
  - [ ] `supra_n_all_trouble_shooting_guide_trace_door_valve_abnormal` (score=4.0312, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Door valve abnormal] Use this guide to diagnose problems with the [Trace Doo
  - [ ] `global_sop_geneva_xp_rep_pm_disc` (score=4.0262, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc SOP No: 0 Revision No: 0 Page: 12 / 31 ## 10. Work Procedure | Flow | Pro
  - [ ] `global_sop_geneva_xp_rep_pm_elbow_heater` (score=4.0235, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Elbow heater | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/
  - [ ] `global_sop_geneva_xp_adj_post_align_application` (score=4.0192, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_Post align application Global SOP No: 0 Revision No: 1 Page: 10 / 20 | Flow | Procedu
  - [ ] `global_sop_geneva_xp_rep_pm_insulation_heater` (score=4.0185, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Insulation heater | SOP No: 0 | | | | | :--- | :--- | :--- | :--- | | Revision
  - [ ] `global_sop_geneva_xp_adj_all_sw_install` (score=4.016, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_AII_SW INSTALL | SOP No: 0 | | | | --- | --- | --- | | Revision No: 0 | | | | Pag
  - [ ] `global_sop_geneva_xp_rep_pm_device_net` (score=4.0141, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Device net | SOP No: 0 | | | |---|---|---| | Revision No: 0 | | | | Page: 3/15
  - [ ] `global_sop_geneva_xp_rep_pm_o2_analyzer` (score=4.0137, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_O2 analyzer(Delta F) | SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Page
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3100qc` (score=4.0048, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_PM_PRISM SOURCE ## 3100QC O-RING & TUBE REPLACEMENT | Global SOP No: | | |---|---| | 
  - [ ] `global_sop_integer_plus_all_pm_vacuum_line` (score=4.0017, device=INTEGER XP, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_PM_PENDULUM VALVE Global SOP No: Revision No: Page : 58 / 133 | Flow | Procedu

#### q_id: `A-amb065`
- **Question**: Wafer Breakage 발생 시 Chamber 복구 절차는?
- **Devices**: [SUPRA_N, SUPRA_XP, SUPRA_VPLUS, PRECIA]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-27):
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3100qc` (score=5.1744, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PRISM SOURCE ## 3100QC REPLACEMENT Global SOP No: Revision No : 1 Page : 20/72 | 
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=5.0604, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 7) Open Chamber | a. Chamber 상단의 Chamber bolt와 Clamp를 해
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=5.0167, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No : 13 Page : 21/75 | Flow | 절
  - [ ] `global_sop_supra_xp_all_pm_cip_chamber` (score=5.0086, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_CIP CHAMBER Global SOP No: Revision No : 0 Page : 20 /37 | Flow | 절차 | Tool & 
  - [ ] `global_sop_supra_n_series_all_pm_tc_wafer` (score=4.8523, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_TC WAFER Global SOP No: Revision No: 2 Page: 3/21 ## 3. 사고 사례 ### 1) 협착 재
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3000qc` (score=4.8191, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PRISM SOURCE 3000QC REPLACEMENT Global SOP No: Revision No : 2 Page : 20/82 | Flo
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=4.7759, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_TM_CONTROLLER ## BATTERY_Replacement Global SOP No : Revision No: 6 Page : 94 / 
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=4.7329, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 3/47 ## 3. 사고 사례 ### 1) 협착 재해의 
  - [ ] `global_sop_precia_all_pm_wafer_centering` (score=4.7088, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_WAFER CENTERING | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page: 3
  - [ ] `global_sop_integer_plus_all_pm_gas_line` (score=4.6405, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_GAS LINE Global_SOP No: Revision No: 0 Page: 3/75 ## 3. 사고 사례 ### 1) 협착 재해의
  - [ ] `global_sop_supra_n_series_all_pm_chamber_hinge` (score=4.5909, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_CHAMBER HINGE Global SOP No : Revision No: 3 Page: 3/22 ## 3. 사고 사례 ### 1
  - [ ] `global_sop_supra_xp_all_pm_chamber_safety_cover` (score=4.5881, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_CHAMBER SAFETY COVER Global SOP No : Revision No : 1 Page : 3/18 ## 3. 사고 사례 #
  - [ ] `global_sop_geneva_xp_rep_pm_chamber_apc_valve` (score=4.5647, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Chamber APC Global SOP No: Revision No: 0 Page: 3/22 ## 3. 사고 사례 ### 1) 협착 재해의
  - [ ] `global_sop_integer_plus_all_pm_cooling_chuck` (score=4.4527, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ ALL_PM_COOLING CHUCK Global_SOP No: Revision No: 0 Page: 3/23 ## 3. 사고 사례 ### 1) 
  - [ ] `global_sop_supra_n_series_all_pm_isolation_valve` (score=4.447, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_ISOLATION VALVE Global SOP No : Revision No: 2 Page: 3/25 ## 3. 사고 사례 ###
  - [ ] `global_sop_precia_all_pm_slot_valve` (score=4.4469, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_SLOT VALVE Global SOP No : 0 Revision No : 0 Page : 3/36 ## 3. 사고 사례 ### 1) 협착 재해
  - [ ] `global_sop_integer_plus_all_pm_view_quartz` (score=4.4424, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_VIEW # QUARTZ Global SOP No: Revision No: 2 Page: 3/19 ## 3. 사고 사례 ### 1) 협
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=4.4378, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus _REP_AM_VACUUM Line Global SOP No: Revision No: Page : 119 / 135 ## 3. 사고 사례 ### 1
  - [ ] `global_sop_integer_plus_all_am_view_quartz` (score=4.4358, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_AM_VIEW QUARTZ Global SOP No: Revision No: 2 Page: 3 / 20 ## 3. 사고 사례 ### 1) 협
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=4.4353, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 28 / 105 ## 사고 사례 ##
  - [ ] `global_sop_supra_n_series_all_tm_buffer_module` (score=4.4331, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_BUFFER MODULE Global SOP No : Revision No: 3 Page: 3/33 ## 3. 사고 사례 ### 1
  - [ ] `global_sop_integer_plus_all_am_baffle` (score=4.4328, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_ INTEGER plus_ALL_AM_BAFFLE Global_SOP No: Revision No: 1 Page: 3 / 18 ## 3. 사고 사례 ### 1) 협착 재해
  - [ ] `global_sop_integer_plus_all_tm_vacuum_line` (score=4.4292, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_VACUUM LINE Global SOP No: Revision No: 1 Page: 3/26 ## 3. 사고 사례 ### 1) 협착 
  - [ ] `global_sop_integer_plus_all_tm_solenoid_valve` (score=4.4255, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_SOLENOID VALVE Global SOP No: 0 Revision No: 0 Page: 3/19 ## 3. 사고 사례 ### 1
  - [ ] `global_sop_integer_plus_all_pm_solenoid_valve` (score=4.4254, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_SOLENOID VALVE Global SOP No: 0 Revision No: 0 Page: 3/19 ## 3. 사고 사례 ### 1
  - [ ] `global_sop_genevaxp_rep_efem_robot_controller` (score=4.4246, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_EFEM_Robot controller Global SOP No: Revision No: 0 Page: 3 / 16 ## 3. 사고 사례 ### 1
  - [ ] `global_sop_integer_plus_all_pm_gas_spring` (score=4.4245, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_GAS SPRING Global SOP No: 0 Revision No: 1 Page: 3 / 19 ## 3. 사고 사례 ### 1) 

#### q_id: `A-amb066`
- **Question**: Bias Power 이상 시 RF Match Network 점검 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-16):
  - [ ] `precia_pm_trouble_shooting_guide_rf_power_abnormal` (score=6.5296, device=PRECIA, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [RF Power Abnormal] Confidential II ## Appendix #1 | A | GCB - GFP - Global SOP - Global SOP_PR
  - [ ] `precia_all_trouble_shooting_guide_pin_motor_abnormal` (score=6.2353, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # PRECIA Trouble Shooting Guide [Pin Motor Abnormal] Confidential II | Alarm Code | LED 점멸 횟수 | Alarm 종류 | 원
  - [ ] `global_sop_precia_all_pm_rf_bias_matcher` (score=6.2147, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_RF BIAS MATCHER Global SOP No: Revision No: 1 Page: 18 / 35 | Flow | Procedure | 
  - [ ] `set_up_manual_ecolite_2000` (score=6.0757, device=ECOLITE 2000, type=set_up_manual)
    > | | MCB1-6 | PMC | | | |---|---|---|---|---| | | MCB1-7 | Distributer | | | | | MCB1-8 | Portable Monitor | | | | | MCB1
  - [ ] `set_up_manual_supra_np` (score=5.9221, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I <!-- Table (58, 69, 928, 731) --> \begin{tabular}{|l|l|l|} \hline \textbf{4) Analog IO Calibr
  - [ ] `global_sop_omnis_plus_adj_pm_rf_power_on_test_eng` (score=5.7122, device=OMNIS plus, type=SOP)
    > ```markdown # Global SOP_OMNIS plus_ADJ_PM_RF Power On Test_ENG Global SOP No : 0 Revision No : 0 Page : 17 / 17 ## 12. 
  - [ ] `set_up_manual_ecolite_ii_400` (score=5.5153, device=ECOLITE II 400, type=set_up_manual)
    > | Module | Circuit Breaker | Connected Component | 1st Check | 2nd Check | | :--- | :--- | :--- | :--- | :--- | | | MCB1
  - [ ] `set_up_manual_ecolite_3000` (score=5.3982, device=ECOLITE3000, type=set_up_manual)
    > | Module | Circuit Breaker | Connected Component | 1st Check | 2nd Check | | :--- | :--- | :--- | :--- | :--- | | Gen Ra
  - [ ] `set_up_manual_supra_xq` (score=5.297, device=SUPRA XQ, type=SOP)
    > | Module | Action | A Prior Condition | 1st Check | 2nd Check | | :--- | :--- | :--- | :--- | :--- | | | RF Bias1 On | A
  - [ ] `set_up_manual_precia` (score=5.1205, device=PRECIA, type=set_up_manual)
    > ```markdown # 7) 중량물 취급 작업 ## 중량물 취급 시 주의사항 - 발은 어깨 너비로, 허리는 반듯이 세우고 무릎의 힘으로 일어섭니다. <!-- Image (127, 161, 594, 247) --> 
  - [ ] `set_up_manual_supra_nm` (score=4.9526, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.2 EFEM Robot Leveling & Loadport Teaching | Picture
  - [ ] `precia_pm_trouble_shooting_guide_process_stable_time_out_alarm` (score=4.8789, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Process Stable Time out Alarm] Confidential II | Failure symptoms | Check point | 
  - [ ] `set_up_manual_supra_n` (score=4.6945, device=SUPRA N, type=SOP)
    > ```markdown Confidential I # 5) Program Description <!-- Image (74, 80, 370, 557) --> a. 상기 화면에서 각 항목에 대한 내용은 다음과 같다 (Ch
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=4.6667, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `global_sop_supra_series_all_smart_match` (score=4.6652, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_Series_ALL_SMART MATCH (with Nport-5650-8-DT-J/MOXA Setting) Global SOP No : Revision N
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=4.5805, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_LM GUIDE ## REPLACEMENT & GREASE INJECTION Global SOP No: Revision No: 1 Page:

#### q_id: `A-amb067`
- **Question**: Gas Cabinet Safety Interlock 해제 및 복구 절차는?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, GENEVA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-28):
  - [ ] `global_sop_supra_n_series_all_sub_unit_pressure_vacuum_switch` (score=7.7095, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_SUB UNIT_PRESSURE & VACUUM SWITCH Global SOP No : Revision No: 1 Page: 2/28 
  - [ ] `global_sop_precia_adj_all_utility_turn_onoff` (score=6.514, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_ALL_UTILITY TURN ON/OFF Global SOP No : Revision No: 0 Page: 2/44 ## 1. Safety 1) 안전
  - [ ] `global_sop_geneva_xp_rep_bubbler_cabinet_feed_valve` (score=6.445, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVA xp_REP_BUBBLER ## CABINET_FEED VALVE | Global SOP No: | S-KG-R030-R0 | | --- | --- | | Revision No: 
  - [ ] `global_sop_geneva_xp_rep_bubbler_cabinet_drain_valve` (score=6.4429, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_BUBBLER ## CABINET_DRAIN VALVE | Global SOP No: | S-KG-R034-R0 | | --- | --- | | 
  - [ ] `global_sop_supra_n_series_all_pm_gas_feed_through` (score=6.4248, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS # FEED THROUGH Global SOP No: Revision No: 4 Page: 2 / 18 ## 1. Safet
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=6.4064, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 4) PM Formic leak 확인 | a. Recipe setup 시 PM 에 장착되어 있는 F
  - [ ] `global_sop_supra_n_series_all_sub_unit_safety_module` (score=6.3755, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N ## Series_ALL_SUBUNIT_SAFETY CTR (OMRON) Global SOP No: Revision No: 1 Page: 35 / 44 | 
  - [ ] `global_sop_geneva_rep_bubbler_cabinet_safety_valve` (score=6.3071, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVA_REP_Bubbler Cabinet_Safety Valve | Global SOP No: | S-KG-R032-R0 | | --- | --- | | Revision No: | 0 
  - [ ] `global_sop_precia_all_pm_gas_spring` (score=6.1455, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_GAS SPRING | Global SOP No : 0 | | | --- | --- | | Revision No : 0 | | | Page : 1
  - [ ] `global_sop_supra_n_all_pm_chamber_open_interlock_change` (score=6.143, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRAN_MODIFY_PM_CHAMBER_OPEN_INTERLOCK_CHANGE | Global SOP No: | | |---|---| | Revision No: 3 
  - [ ] `global_sop_precia_all_pm_mfc` (score=6.1102, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_MFC Global SOP No : Revision No: 1 Page: 2/23 ## 1. Safety 1) 안전 및 주의사항 - Main Ga
  - [ ] `global_sop_supra_n_all_sub_unit_mfc` (score=6.0613, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_SUB UNIT_MFC Global SOP No: 0 Revision No: 3 Page: 2/21 ## 1. Safety ### 1) 안전 및 주의
  - [ ] `global_sop_integer_plus_all_tm_filter` (score=6.0417, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_FILTER Global_SOP No: Revision No: 0 Page: 2/20 ## 1. Safety 1) 안전 및 주의사항 -
  - [ ] `global_sop_precia_all_pm_pneumatic_valve` (score=6.0305, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_PNEUMATIC VALVE Global SOP No: Revision No: 0 Page: 2/20 ## 1. SAFETY 1) 안전 및 주의사
  - [ ] `global_sop_integer_plus_all_pm_igs_block` (score=5.9691, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_IGS BLOCK Global SOP No : Revision No: 0 Page: 2/26 ## 1. Safety ### 1) 안전 
  - [ ] `global_sop_integer_plus_all_pm_gas_box_door_sensor` (score=5.9641, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_GAS BOX DOOR SENSOR | Global SOP No: | | |---|---| | Revision No: 1 | | | P
  - [ ] `global_sop_supra_n_series_all_sub_unit_gas_box_board` (score=5.9572, device=SUPRA N series, type=SOP)
    > # Global SOP_SUPRA N series_ALL_SUB UNIT_GAS BOX BOARD | Global SOP No: 0 | | | --- | --- | | Revision No: 4 | | | Page:
  - [ ] `global_sop_integer_plus_all_pm_gas_spring` (score=5.9421, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_GAS SPRING | Global SOP No: 0 | | | --- | --- | | Revision No: 1 | | | Page
  - [ ] `global_sop_supra_xp_all_sub_unit_gas_box_board` (score=5.9412, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_SUB UNIT_GAS BOX BOARD | Global SOP No : | | | --- | --- | | Revision No : 1 | | 
  - [ ] `global_sop_supra_n_series_all_pm_gas_spring` (score=5.9385, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS SPRING Global SOP No: Revision No: 4 Page: 1/22 ## Scope 이 Global SOP
  - [ ] `global_sop_integer_plus_all_am_gas_spring` (score=5.9373, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_ INTEGER plus_ALL_AM_Gas Spring Global_SOP No: Revision No: 1 Page: 1/19 ## Scope 이 Global SOP는
  - [ ] `global_sop_integer_plus_all_pm_pneumatic_valve` (score=5.8821, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_PNEUMATIC VALVE Global SOP No: Revision No: 1 Page: 2/20 ## 1. SAFETY 1) 안전
  - [ ] `global_sop_integer_plus_all_am_gas_line` (score=5.8354, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_REP_AM_TOP_GAS FEED THROUGH Global SOP No: Revision No: 0 Page: 24 / 67 ## 5. Flow Chart Start
  - [ ] `global_sop_supra_xp_all_tm_mfc` (score=5.8178, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_MFC Global SOP No: Revision No: 0 Page: 2 / 18 ## 1. Safety ### 1) 안전 및 주의사항 -
  - [ ] `global_sop_integer_plus_all_efem_o2_gas_leak_detector` (score=5.7998, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_EFEM_O2 GAS LEAK DETECTOR | Global_SOP No: | 0 | | --- | --- | | Revision No: 
  - [ ] `global_sop_integer_plus_all_pm_h2_gas_leak_detector` (score=5.782, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_H2 GAS LEAK DETECTOR | Global_SOP No: | 0 | | --- | --- | | Revision No: | 
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=5.78, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 27 / 105 ## Safety 1
  - [ ] `global_sop_integer_plus_all_tm_devicenet_board` (score=5.7699, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_ INTEGER plus_ALL_TM_DEVICENET BOARD Global_SOP No: Revision No: 3 Page: 2/21 ## 1. Safety 1) 안

#### q_id: `A-amb068`
- **Question**: EFEM Robot Teaching 재설정 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, SUPRA_VPLUS, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-15):
  - [ ] `set_up_manual_supra_n` (score=8.0577, device=SUPRA N, type=SOP)
    > ```markdown Confidential I | 36) EFEM Single Teaching | a. 방법은 동일하지만 Teaching 변경 시 TM Robot 이 아닌 EFEM Robot Teaching 을 재
  - [ ] `set_up_manual_supra_nm` (score=7.5627, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.8 EFEM BM Teaching (Single) | Picture | Description
  - [ ] `set_up_manual_supra_np` (score=6.6226, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I 36) EFEM Single Teaching <!-- Image (60, 70, 357, 252) --> a. 방법은 동일하지만 Teaching 변경 시 TM Robo
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=5.9876, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `global_sop_integer_plus_all_tm_robot` (score=5.9388, device=INTEGER plus, type=SOP)
    > # Global SOP_INTEGER plus_ALL_TM_ROBOT Global SOP No : Revision No: 4 Page :103 / 103 ## 8. Appendix [EnM_SOP] A.I.D DAT
  - [ ] `global_sop_integer_plus_all_efem_robot` (score=5.8887, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_EFEM_ROBOT TEACHING Global SOP No: Revision No: 3 Page: 43 / 126 | Flow | Proc
  - [ ] `global_sop_supra_xp_all_pm_heater_chuck` (score=5.8736, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP ZEDIUS XP_ALL_PM_HEATER # CHUCK Global SOP No: 0 Revision No: 2 Page: 21 / 49 | Flow | Procedur
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=5.8654, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_HOOK LIFTER ## SERVO MOTOR CONTROLLER REPLACEMENT Global SOP No: Revision No: 
  - [ ] `40043406` (score=5.5938, device=SUPRA Vplus, type=myservice)
    > -. EFEM Robot Teaching
-. TM Robot Teaching
--- Document Info.---
SOP Title :
Global SOP_SUPRA Vplus_ADJ_EFEM_ROBOT TEAC
  - [ ] `set_up_manual_integer_plus` (score=5.5119, device=INTEGER plus, type=SOP)
    > ```markdown # 17-18. EFEM Robot Teaching | Picture | Description | Data | OK | NG | N/A | | :--- | :--- | :--- | :--- | 
  - [ ] `set_up_manual_supra_vm` (score=5.4758, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 8. EFEM Robot Teaching_Loadport 1 | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 26) Rob
  - [ ] `global_sop_geneva_xp_adj_efem_efem_robot_leveling` (score=5.4174, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_EFEM_EFEM Robot leveling Global SOP No: Revision No: 0 Page: 12 / 15 ## 10. Work 
  - [ ] `set_up_manual_ecolite_3000` (score=5.3028, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 3. Docking (EFEM Robot Pick 장착) | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) EFEM R
  - [ ] `global_sop_supra_xp_all_efem_robot_sr8241` (score=5.2904, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_EFEM_ROBOT_SR8241 Global SOP No : Revision No: 2 Page: 24 / 34 | Flow | Procedure
  - [ ] `global_sop_genevaxp_rep_efem_robot_controller` (score=5.2588, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVAxp_REP_EFEM_Robot controller Global SOP No: Revision No: 0 Page: 12 / 16 | Flow | Procedu

#### q_id: `A-amb069`
- **Question**: UPS 전환 시 설비 동작 영향 및 점검 사항은?
- **Devices**: [SUPRA_N, SUPRA_XP, SUPRA_VPLUS, INTEGER_PLUS, PRECIA]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-23):
  - [ ] `global_sop_supra_n_series_all_rack` (score=5.4385, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_RACK Global SOP No : Revision No: 1 Page: 2/84 ## 1. Safety 1) 안전 및 주의사항 - 감
  - [ ] `global_sop_genevaxp_rep_pm_disc_home_sensor` (score=5.2103, device=GENEVA XP, type=SOP)
    > # Global SOP_GENEVAxp_REP_PM_Disc home sensor Global SOP No: Revision No: 0 Page: 6 / 15 ## 6. Flow Chart Start ↓ 1. SOP
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=5.2011, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page : 73 / 105 ## 환경 안전 보
  - [ ] `global_sop_supra_vplus_adj_all_power_turn_on_off` (score=5.1837, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ADJ_RACK_POWER ## TURN ON/OFF Global SOP No: Revision No: 1 Page: 2/19 # 1. Safety 
  - [ ] `global_sop_supra_vplus_all_pm_controller` (score=5.1781, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_ALL_PM_Controller Global SOP No: Revision No: 3 Page: 4/42 ## 4. 환경 안전 보호구 | 구분 | 상
  - [ ] `global_sop_integer_plus_all_pm_pin_motor` (score=5.1734, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_PM_FLOATING JOINT Global SOP No: Revision No: 5 Page: 9 / 126 ## 1. 환경 안전 보호구 
  - [ ] `global_sop_integer_plus_all_am_baffle` (score=5.1393, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_ INTEGER plus_ALL_AM_BAFFLE Global_SOP No: Revision No: 1 Page: 2 / 18 ## 1. SAFETY 1) 안전 및 주의사
  - [ ] `global_sop_integer_plus_all_am_view_quartz` (score=5.1259, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_AM_VIEW QUARTZ Global SOP No: Revision No: 2 Page: 2 / 20 ## 1. Safety 1) 안전 및
  - [ ] `set_up_manual_omnis` (score=5.1149, device=OMNIS, type=set_up_manual)
    > | | MCB2-3 | Heat Exchanger 2 | | | | :--- | :--- | :--- | :--- | :--- | | | MCB3-1 | Heat Exchanger 1 | | | | | MCB2-3 
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=5.0862, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 3. Docking 및 Leveling_EFEM | Picture | Description | Spec | Check | Result | | :--- | :--- | :--- | :--- |
  - [ ] `global_sop_integer_plus_all_efem_controller` (score=5.0696, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_EFEM_CONTROLLER Global SOP No : Revision No: 1 Page: 7/28 ## 1. 환경 안전 보호구 | 구분
  - [ ] `global_sop_integer_plus_all_am_devicenet_board` (score=5.0578, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_AM_AIO # CALIBRATION [TOS BOARD] Global SOP No : Revision No: 0 Page: 25/58 ##
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=5.0294, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PASSIVE PAD REPLACEMENT Global SOP No : Revision No: 6 Page : 51 / 107 ## 1.
  - [ ] `global_sop_supra_n_series_all_tm_ctc` (score=5.0235, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_CTC REPLACEMENT Global SOP No : Revision No: 6 Page: 8/81 ## 1. 환경 안전 보호구
  - [ ] `global_sop_supra_n_series_all_pm_heater_chuck` (score=5.0179, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_HEATER CHUCK Global SOP No : Revision No: 2 Page: 7/40 ## 1. 환경 안전 보호구 | 
  - [ ] `global_sop_integer_plus_all_tm_mototr_controller` (score=5.0165, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_TM_MOTOR CONTROLLER Global SOP No: Revision No: Page: 7 / 21 ## 1. 환경안전보호구 | 구
  - [ ] `global_sop_geneva_xp_rep_pm_o2_cell` (score=5.0113, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GNENVA_PM_REP_O2 Cell | Global SOP No: | | |---|---| | Revision No: 0 | | | Page: 2/14 | | ## 1
  - [ ] `global_sop_supra_n_series_all_pcw_turn_on` (score=5.0066, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PCW TURN ON Global SOP No: Revision No: 1 Page: 2/31 ## 1. Safety 1) 안전 및 주의
  - [ ] `global_sop_integer_plus_all_ll_vacuum_block` (score=5.0032, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_LL_VACUUM BLOCK Global SOP No : Revision No : 1 Page : 8/20 ## 1. 환경안전보호구 | 구분
  - [ ] `global_sop_supra_n_series_all_tm_branch_tap` (score=4.9911, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_BRANCH TAP REPLACEMENT Global SOP No : Revision No: 2 Page: 7/21 ## 1. 환경
  - [ ] `global_sop_supra_n_all_pm_chamber_open_interlock_change` (score=4.9898, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_MODIFY_PM_CHAMBER_OPEN_INTERLOCK_CHANGE Global SOP No: Revision No: 3 Page: 8/24 ## 1. 
  - [ ] `global_sop_geneva_xp_all_8계통_check` (score=4.9648, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA XP_8계통_Check sheet SOP No: 0 Revision No: 0 Page: 22 / 46 ## 10. Work Procedure | Flow | Procedure 
  - [ ] `global_sop_supra_n_series_all_sub_unit_elt_box_assy` (score=4.9578, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_SUB UNIT_NOISE FILTER Global SOP No: Revision No: 0 Page 23/95 ## 1. 환경 안전 보

#### q_id: `A-amb070`
- **Question**: CDA Pressure Drop 시 설비 영향 및 조치 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, INTEGER_PLUS, PRECIA, GENEVA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-11):
  - [ ] `global_sop_precia_all_tm_pressure_relief_valve` (score=5.9821, device=PRECIA, type=SOP)
    > # Global SOP_PRECIA_ALL_TM_PRESSURE RELIEF VALVE Global SOP No: Revision No: 1 Page: 10/16 ## 5. Flow Chart Start 1. Glo
  - [ ] `global_sop_geneva_xp_rep_pm_pressure_switch` (score=5.8519, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Pressure switch SOP No: 0 Revision No: 0 Page: 8 / 16 ## 10. Work Procedure | 
  - [ ] `global_sop_precia_all_efem_pressure_relief_valve` (score=5.7845, device=PRECIA, type=SOP)
    > # Global SOP_PRECIA_ALL_EFEM_PRESSURE RELIEF VALVE Global SOP No: Revision No: 1 Page: 10/16 ## 5. Flow Chart Start ↓ 1.
  - [ ] `global_sop_supra_n_series_all_efem_pressure_vacuum_switch` (score=5.6131, device=SUPRA N series, type=SOP)
    > # Global SOP_SUPRA N series_ALL_EFEM_PRESSURE SWITCH REPLACEMENT & ADJUST Global SOP No : Revision No : 2 Page : 10 / 29
  - [ ] `set_up_manual_supra_np` (score=5.5613, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I Analog Output Tap은 조작하지 않는다. # 12. TTTM (※환경안전 보호구: 안전모, 안전화) ## 12.5 Device Net Calibration 
  - [ ] `set_up_manual_supra_n` (score=5.2848, device=SUPRA N, type=SOP)
    > ```markdown Confidential I 8) Pendant Disconnect & Door Close a. Cooling Stage Pin Speed 조절 완료 후 Cable Disconnection 을 진
  - [ ] `global_sop_precia_all_tm_pressure_switch` (score=5.284, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ADJ_TM_PRESSURE SWITCH Global SOP No : Revision No: 0 Page: 24 / 27 | Flow | Procedure |
  - [ ] `set_up_manual_supra_nm` (score=5.2256, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 12. TTTM (※환경안전 보호구 : 안전모, 안전화) ## 12.1. Common List ### 12.1.4 CDA/VAC Pressure Switch | P
  - [ ] `global_sop_supra_xp_all_efem_pressure_switch` (score=5.2085, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_EFEM_PRESSURE SWITCH REPLACEMENT Global SOP No : Revision No: 1 Page: 16/31 | Flo
  - [ ] `set_up_manual_precia` (score=5.1053, device=PRECIA, type=set_up_manual)
    > | | | b. Coolant 용액은 적정 Level 수준 으로 Charge되어 있는가? | | | | | :--- | :--- | :--- | :--- | :--- | :--- | | 3) Utility Turn 
  - [ ] `40085764` (score=4.9702, device=TIGMA Vplus, type=myservice)
    > WPSKK5PV00 Recovery work
Since CDA LIMIT ALARM occurred on 114, Swapped CDA PRESSURE SW on 5PV and 114. (mySERVICE : 400

#### q_id: `A-amb071`
- **Question**: Door Open Interlock 반복 발생 시 점검 항목은?
- **Devices**: [SUPRA_N, SUPRA_XP, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: information_lookup
- **ES candidates** (top-13):
  - [ ] `set_up_manual_supra_nm` (score=6.7873, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential 1 # 13. Customer Certification (※환경안전 보호구 : 안전모, 안전화) ## 13.1 Interlock Check (H/W, S/W) | Pict
  - [ ] `set_up_manual_supra_np` (score=6.4799, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I ### 3) H/W Interlock Example | Module | A prior condition | Action | Result ( Hw : Interlock)
  - [ ] `set_up_manual_supra_n` (score=6.0721, device=SUPRA N, type=SOP)
    > ```markdown Confidential 1 # 13. Customer Certification (※환경안전 보호구 : 안전모, 안전화) ## 13.1 Interlock Check (H/W, S/W) | Pict
  - [ ] `set_up_manual_ecolite_3000` (score=5.985, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 13. Customer Certification_Interlock Check _ H/W | Module | Action | A Prior Condition | 1st Check | 2nd C
  - [ ] `global_sop_supra_xp_all_efem_ionizer` (score=5.8777, device=ZEDIUS XP, type=SOP)
    > # Global SOP_ ZEDIUS XP_ALL_EFEM_IONIZER Global SOP No: Revision No: 1 Page: 12 / 17 | Flow | Detail | |---|---| | 4. EF
  - [ ] `set_up_manual_supra_vm` (score=5.7902, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 17. Interlock Check _ H/W | Module | Action | A Prior Condition | 1st Check | 2nd Check | | :--- | :--- | 
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=5.6785, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 14. Customer certification | Picture | Description | Spec | Check | Result | | :--- | :--- | :--- | :--- |
  - [ ] `set_up_manual_ecolite_ii_400` (score=5.451, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 13. Customer Certification_Interlock Check_H/W | Module | Action | A Prior Condition | 1st Check | 2nd Che
  - [ ] `global_sop_integer_plus_all_pm_gas_box_door_sensor` (score=5.2874, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus REP_PM_GAS BOX DOOR SENSOR Global SOP No: Revision No: 1 Page: 15 / 18 | Flow | Pr
  - [ ] `global_sop_geneva_xp_all_8계통_check` (score=5.185, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA XP_8계통_Check sheet SOP No: 0 Revision No: 0 Page: 12/46 ## 10. Work Procedure | Flow | Procedure | 
  - [ ] `set_up_manual_ecolite_2000` (score=5.1443, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 12. Customer Certification_Interlock Check _ H/W | Module | Action | A Prior Condition | 1st Check | 2nd C
  - [ ] `set_up_manual_integer_plus` (score=4.9134, device=INTEGER plus, type=SOP)
    > ```markdown # 16. Interlock Check _ H/W | Module | Action | A prior condition | 1st Check | 2nd Check | | :--- | :--- | 
  - [ ] `global_sop_geneva_xp_rep_pm_mfc` (score=4.8299, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_MFC SOP No: 0 Revision No: 0 Page: 8 / 18 ## 10. Work Procedure | Flow | Proce

#### q_id: `A-amb072`
- **Question**: Process Abort 후 Wafer 복구 절차는?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-14):
  - [ ] `set_up_manual_ecolite_3000` (score=4.8511, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 7. Teaching_Process Module 2 | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) Semi Tran
  - [ ] `set_up_manual_supra_vm` (score=4.5829, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 9. TM Robot Teaching_Process Module 2 | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) 
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=4.552, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 2) Load | a. Load를 클릭한다. | | | | | | | 3) Set-up Recipe
  - [ ] `global_sop_precia_all_pm_wafer_centering` (score=4.4006, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_WAFER CENTERING | Global SOP No: | | | --- | --- | | Revision No: 0 | | | Page: 1
  - [ ] `global_sop_supra_n_series_all_pm_tc_wafer` (score=4.2849, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_TC WAFER Global SOP No: Revision No: 2 Page: 3/21 ## 3. 사고 사례 ### 1) 협착 재
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=4.2625, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM_WAFER SENSOR | Global SOP No : | | | --- | --- | | Revision No: 2 | | | P
  - [ ] `set_up_manual_supra_xq` (score=4.2272, device=SUPRA XQ, type=SOP)
    > ```markdown # 13. Process Confirm (※환경안전 보호구: 안전모, 안전화) ## 13-2 Process Check | Picture | Description | Tool & Spec | | 
  - [ ] `set_up_manual_supra_nm` (score=4.2128, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 14. Process Confirm (※환경안전 보호구 : 안전모, 안전화) ## 14.2 Process Check | Picture | Description | 
  - [ ] `global_sop_precia_all_pm_prevent_maintenance` (score=4.157, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_PREVENT MAINTENANCE | Global SOP No: | | |---|---| | Revision No: 5 | | | Page: 1
  - [ ] `global_sop_supra_n_series_all_pm_process_kit` (score=4.1553, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_PROCESS KIT | Global SOP No : | | | --- | --- | | Revision No: 0 | | | Pa
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=4.1178, device=ZEDIUS XP, type=set_up_manual)
    > ```markdown # 14. Process Confirm (※환경안전 보호구 : 안전모, 안전화) ## 14.2 Process Check | Picture | Description | Tool & Spec | |
  - [ ] `global_sop_supra_xp_all_pm_process_kit` (score=4.1082, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PROCESS KIT Global SOP No : Revision No : 3 Page : 3/28 ## 3. 사고 사례 ### 3-1 협착
  - [ ] `set_up_manual_ecolite_ii_400` (score=4.094, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 14. Process Confirm_Process Check | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 6) PR W
  - [ ] `set_up_manual_ecolite_2000` (score=4.0833, device=ECOLITE 2000, type=set_up_manual)
    > | 5) pc Check 진행 | a. Aging 후 고객社에 pc Check Wafer 요청 후 pc Check 진행. | Spec 45nm 50ea 이하 Spec은 고객社 마다 상이 | | | | :--- | :

#### q_id: `A-amb073`
- **Question**: Endpoint Detection Window 설정 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-21):
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=6.0023, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `set_up_manual_supra_np` (score=5.0708, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I Analog Output Tap은 조작하지 않는다. # 12. TTTM (※환경안전 보호구: 안전모, 안전화) ## 12.5 Device Net Calibration 
  - [ ] `global_sop_supra_xp_all_tm_multi_port` (score=4.7784, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDISU XP_ALL_TM_MULTI PORT Global SOP No: Revision No: 1 Page: 17 / 24 | Flow | Procedure | To
  - [ ] `global_sop_integer_plus_all_pm_manometer` (score=4.6463, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER Plus_REP_MANOMETER Global SOP No: Revision No: 1 Page: 20 / 20 ## 8. Appendix | Flow | 
  - [ ] `set_up_manual_supra_nm` (score=4.5164, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.2 EFEM Robot Leveling & Loadport Teaching | Picture
  - [ ] `global_sop_precia_all_pm_manometer` (score=4.498, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_MANOMETER Global SOP No: Revision No: 0 Page: 20 / 20 ## 8. Appendix | Flow | Proced
  - [ ] `40060043` (score=4.4433, device=SUPRA Vplus, type=myservice)
    > lower than ignition window
  - [ ] `global_sop_precia_all_tm_branch_tap` (score=4.3574, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_TM_BRANCH TAP Global SOP No : Revision No: 0 Page: 18 / 23 ## 6. Work Procedure | Fl
  - [ ] `global_sop_integer_plus_all_tm_ctc_controller` (score=4.3386, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_TM_CTC # CONTROLLER Global SOP No: Revision No: 1 Page: 30 / 51 | Flow | Proce
  - [ ] `global_sop_supra_n_series_all_sub_unit_manometer` (score=4.3232, device=SUPRA N, type=SOP)
    > # Global SOP_SUPRA N series_ADJ_SUB UNIT_MANOMETER Global SOP No: Revision No: 0 Page: 31/32 ## 8. Appendix ### 계측모드 Mod
  - [ ] `set_up_manual_supra_n` (score=4.2789, device=SUPRA N, type=SOP)
    > ```markdown Confidential I 36) LP2,3 Teaching <!-- Image (70, 70, 290, 226) --> a. LP1 Teaching 과 동일한 방법으로 LP2,3 를 진행하여 
  - [ ] `global_sop_supra_n_series_all_tm_ctc` (score=4.2369, device=SUPRA N, type=SOP)
    > # Global SOP_SUPRA N series_ALL_TM_CTC REPLACEMENT Global SOP No : Revision No: 6 Page : 19/81 | Flow | Procedure | Tool
  - [ ] `global_sop_precia_all_pm_mfc` (score=4.2282, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_MFC Global SOP No : Revision No: 1 Page: 19/23 ## 6. Work Procedure | Flow | Proc
  - [ ] `global_sop_supra_xp_all_pm_prism_source_3000qc` (score=4.0432, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRISM 3000,3100QC SOURCE IGNITION WINDOW CHECK Global SOP No: Revision No: 2 P
  - [ ] `integer_plus_all_trouble_shooting_guide_trace_tm_robot_abnormal` (score=3.9829, device=INTEGER plus, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace TM Robot Abnormal] Confidential II ## Appendix #2 ### A. #### 11.5. Recovery
  - [ ] `global_sop_supra_xp_all_tm_ctc` (score=3.9609, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_CTC REPLACEMENT Global SOP No : Revision No: 1 Page: 30/45 | Flow | Procedure 
  - [ ] `global_sop_supra_n_series_sw_all_sw_installation_setting` (score=3.9158, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_SW_TM_DEVICE NET # SETTING Global SOP No : Revision No: 2 Page: 78/84 | Flow | P
  - [ ] `global_sop_geneva_xp_rep_pm_mfc` (score=3.9079, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_MFC SOP No: 0 Revision No: 0 Page: 14 / 18 ## 10. Work Procedure | Flow | Proc
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=3.8636, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM ROBOT ASSY REPLACEMENT Global SOP No : Revision No: 6 Page: 49 / 107 ## 8
  - [ ] `global_sop_supra_n_series_all_tm_ffu_mcu` (score=3.7943, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_FFU_MCU Global SOP No : Revision No: 1 Page: 26 / 57 ## 7. 작업 Check Sheet
  - [ ] `global_sop_supra_n_series_all_tm_devicenet_board` (score=3.7603, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_TM_DEVICE NET BOARD Global SOP No : Revision No: 2 Page: 22/31 | Flow | Proc

#### q_id: `A-amb074`
- **Question**: Heater Zone별 온도 편차 기준 및 조정 방법은?
- **Devices**: [SUPRA_N, SUPRA_VPLUS, SUPRA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-25):
  - [ ] `global_sop_supra_xp_all_pm_heater_chuck` (score=6.116, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP ZEDIUS XP_ALL_PM_HEATER # CHUCK Global SOP No: 0 Revision No: 2 Page: 20/49 | Flow | Procedure 
  - [ ] `global_sop_supra_xp_all_pm_baratron_gauge` (score=5.5011, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_BARATRON GAUGE Global SOP No : 0 Revision No : 0 Page : 3/33 ## 3. 사고 사례 ### 1
  - [ ] `global_sop_supra_xp_all_pm_pirani_gauge` (score=5.4994, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIRANI GAUGE Global SOP No: Revision No: 1 Page: 3/40 ## 3. 사고 사례 ### 1) 화상 재해
  - [ ] `global_sop_supra_n_series_all_pm_pressure_gauge` (score=5.4948, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_PRESSURE GAUGE Global SOP No: Revision No: 3 Page: 3/46 ## 3. 사고 사례 ### 3
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=5.4915, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 77 / 105 ## 사고 사례 ##
  - [ ] `global_sop_integer_plus_all_am_vacuum_line` (score=5.4914, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ADJ_AM_BARATRON GAUGE Global SOP No: Revision No: Page: 8 / 135 ## 3. 사고 사례 ### 1)
  - [ ] `global_sop_supra_xp_all_pm_pressure_gauge` (score=5.491, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PRESSURE GAUGE Global SOP No: Revision No: 1 Page: 3/34 ## 3. 사고 사례 ### 1) 화상 
  - [ ] `global_sop_supra_n_series_all_pm_top_lid` (score=5.4844, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N Series_ALL_PM_TOP LID Global SOP No: Revision No: 0 Page: 4 / 48 ## 3-2 화상 ### 1) 화상 재해
  - [ ] `global_sop_supra_xp_all_tm_mfc` (score=5.4817, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_TM_MFC Global SOP No: Revision No: 0 Page: 3/18 ## 3. 사고 사례 ### 1) 화상 재해의정의 불이나 뜨
  - [ ] `global_sop_supra_xp_all_pm_process_kit` (score=5.4793, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PROCESS KIT Global SOP No : Revision No : 3 Page : 4/28 ## 3-2 화상 ### 1) 화상 재해
  - [ ] `global_sop_supra_n_series_all_pm_epd` (score=5.4404, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_EPD Global SOP No : Revision No: 3 Page: 3/49 ## 3. 사고 사례 ### 1) 화상 재해의 정
  - [ ] `global_sop_geneva_xp_rep_pm_harmonic_drive` (score=5.4395, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Harmonic drive | Global SOP No: 0 | | | --- | --- | | Revision No: 0 | | | Pag
  - [ ] `global_sop_supra_n_series_all_pm_hook_lifter_servo_motor` (score=5.4247, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_HOOK LIFTER ## SERVO MOTOR Global SOP No : Revision No: 2 Page : 3 / 106 
  - [ ] `global_sop_geneva_xp_rep_pm_adapter_ring` (score=5.419, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Adapter ring Global SOP No: 0 Revision No: 0 Page: 4 / 30 ## 3. 사고 사례 ### 1) 화
  - [ ] `global_sop_geneva_xp_rep_pm_disc_amplifier_r1` (score=5.4158, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Disc amplifier | Global SOP No: | 0 | | --- | --- | | Revision No: | 0 | | Pag
  - [ ] `global_sop_geneva_xp_rep_pm_heater_chuck_without_jig` (score=5.4157, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA xp_REP_PM_Heater chuck w/o jig | SOP No: 0 | | | |---|---|---| | Revision No: 1 | | | | Page: 3/52 
  - [ ] `global_sop_supra_n_series_all_pm_heater_chuck` (score=5.4024, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_HEATER CHUCK Global SOP No : Revision No: 2 Page: 3/40 ## 3. 사고 사례 ### 1)
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=5.3975, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 4/47 ## 3. 사고 사례 ### 1) 화상 재해의 
  - [ ] `global_sop_geneva_xp_rep_pm_load_lock_o_ring` (score=5.3073, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_LOAD LOCK O-RING | Global SOP No: | S-KG-R019-R0 | | --- | --- | | Revision No
  - [ ] `global_sop_supra_n_series_all_pm_gas_feed_through` (score=5.3023, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_PM_GAS FEED THROUGH Global SOP No: Revision No: 4 Page: 4 / 18 ## 3. 사고 사례 #
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=5.291, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PIN ASSY Global SOP No: Revision No: 1 Page: 3/124 ## 3. 사고 사례 ### 1) 화상의 정의 불
  - [ ] `global_sop_supra_xp_all_pm_dual_epd` (score=5.2893, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_DUAL EPD Global SOP No: Revision No: 1 Page: 3/40 ## 3. 사고 사례 ### 1) 화상의 정의 불이
  - [ ] `global_sop_supra_xp_all_pm_pendulum_valve` (score=5.1958, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PENDULUM VALVE Global SOP No : Revision No: 6 Page: 3/26 ## 3. 사고 사례 ### 1) 화상
  - [ ] `global_sop_geneva_xp_adj_pm_pin_alignment` (score=5.1932, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_PM_Pin Alignment | Global SOP No: | S-KG-A003-R0 | | --- | --- | | Revision No: |
  - [ ] `global_sop_precia_all_pm_pendulum_valve` (score=5.1908, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_PENDULUM VALVE Global SOP No : Revision No: 0 Page: 3/32 ## 3. 사고 사례 ### 1) 화상 재해

#### q_id: `A-amb075`
- **Question**: Gas Line Integrity Test 절차는?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, GENEVA_XP]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-19):
  - [ ] `global_sop_supra_xp_all_pm_cip_chamber` (score=5.9472, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_CIP CHAMBER Global SOP No: Revision No : 0 Page : 27 /37 | Flow | 절차 | Tool & 
  - [ ] `global_sop_integer_plus_all_pm_gas_line` (score=5.8141, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_PM_GAS LINE | Global_SOP No: | | |---|---| | Revision No: 0 | | | Page: 1/75 |
  - [ ] `global_sop_supra_xp_all_pm_preventive_maintenance` (score=5.5519, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PREVENTIVE MAINTENANCE Global SOP No: Revision No : 13 Page : 25/75 | Flow | 절
  - [ ] `global_sop_integer_plus_all_am_gas_line` (score=5.4317, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_ALL_AM_GAS LINE | Global SOP No: | | | --- | --- | | Revision No: | 0 | | Page: | 
  - [ ] `global_sop_precia_all_pm_mfc` (score=5.2823, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_MFC Global SOP No : Revision No: 1 Page: 14 / 23 ## 6. Work Procedure | Flow | Pr
  - [ ] `integer_plus_all_trouble_shooting_guide_trace_leak_rate_over` (score=5.2811, device=INTEGER plus, type=trouble_shooting_guide)
    > # Trouble Shooting Guide [Trace leak rate over] ## Appendix #2 ### A. #### 1. PM He Leak Check Point I - Bottom Gas Line
  - [ ] `set_up_manual_precia` (score=5.2413, device=PRECIA, type=set_up_manual)
    > | | | | |---|---|---| | 3. Gas box manual valve open | a. Gas Regulator Full open<br>b. Gas manual valve lock key 제거<br>
  - [ ] `global_sop_supra_series_all_sw_operation` (score=5.1582, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_ SW OPERATION Global SOP No Revision No: 2 Page: 10/49 ## 2. Gas line Leak Che
  - [ ] `set_up_manual_supra_xq` (score=4.8073, device=SUPRA XQ, type=SOP)
    > ```markdown # 10. Toxic Gas Turn On (※환경안전 보호구: 안전모, 안전화) ## 10-1 Toxic Gas Line Check | Picture | Description | Tool & 
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=4.8064, device=ZEDIUS XP, type=set_up_manual)
    > ```markdown # 10. Toxic Gas Turn On (※환경안전 보호구 : 안전모, 안전화) ## 10.1 Toxic Gas Line Check | Picture | Description | Tool &
  - [ ] `set_up_manual_supra_np` (score=4.7701, device=SUPRA Np, type=set_up_manual)
    > ```markdown # 3. Docking (※환경안전 보호구: 안전모, 안전화) ## 3.15 PM Gas Line & Cooling PCW Line 장착 | Picture | Description | Tool 
  - [ ] `set_up_manual_integer_plus` (score=4.7431, device=INTEGER plus, type=SOP)
    > ```markdown # 10. Process Gas Turn On (환경안전 보호구: 안전모, 안전화, 방독면, 보안경) ## 10.1 H2 Gas Turn on | Picture | Description | To
  - [ ] `set_up_manual_supra_n` (score=4.7293, device=SUPRA N, type=SOP)
    > Confidential I 2) Leak Check a. Pump Turn On 후 8시간 Full Pumping 후 Leak Check를 한다. b. Leak Check 조건 및 Spec은 고객사마다 상이. | |
  - [ ] `set_up_manual_ecolite_ii_400` (score=4.6873, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 11. Leak Check | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) Leak Check | a. Leak Ch
  - [ ] `global_sop_precia_all_pm_gas_spring` (score=4.6785, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_ALL_PM_GAS SPRING | Global SOP No : 0 | | | --- | --- | | Revision No : 0 | | | Page : 1
  - [ ] `set_up_manual_supra_nm` (score=4.6686, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 3. Docking (※환경안전 보호구 : 안전모, 안전화) ## 3.16 PM Gas Line & Cooling PCW Line 장착 | Picture | Des
  - [ ] `global_sop_supra_xp_all_sub_unit_igs_block` (score=4.654, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_SUB UNIT_IGS BLOCK Global SOP No : Revision No: 2 Page: 49/71 | Flow | Procedure 
  - [ ] `global_sop_supra_xp_all_pm_n2_curtain_eng` (score=4.6458, device=ZEDIUS XP, type=SOP)
    > # Global SOP _ ZEDIUS XP _ ALL _ PM _ N2 _ CURTAIN Global SOP No: 0 Revision No: 1 Page: 3 / 19 ## 3. Accident Case ### 
  - [ ] `global_sop_supra_n_all_sub_unit_igs_block` (score=4.6339, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_SUB UNIT_IGS BLOCK Global SOP No : Revision No: 1 Page: 62/67 | Flow | Procedure | 

#### q_id: `A-amb076`
- **Question**: Chamber Seasoning 후 공정 안정화 확인 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-17):
  - [ ] `set_up_manual_supra_xq` (score=6.6428, device=SUPRA XQ, type=SOP)
    > ```markdown # 13. Process Confirm (※환경안전 보호구: 안전모, 안전화)) ## 13-1. Aging Test | Picture | Description | Tool & Spec | | :
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=6.2743, device=ZEDIUS XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) Seasoning | a. [START]를 누른다. | | | | | | | 14) Seas
  - [ ] `set_up_manual_ecolite_3000` (score=6.0642, device=ECOLITE3000, type=set_up_manual)
    > ```markdown # 14. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) Season
  - [ ] `set_up_manual_supra_nm` (score=5.7991, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.2 EFEM Robot Leveling & Loadport Teaching | Picture
  - [ ] `set_up_manual_supra_vm` (score=5.7771, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 14. Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) Seasoning | a. [START]
  - [ ] `set_up_manual_ecolite_ii_400` (score=5.7564, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 14. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) Season
  - [ ] `set_up_manual_ecolite_2000` (score=5.752, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 13. Process Confirm_Aging Test | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 13) Season
  - [ ] `set_up_manual_integer_plus` (score=5.627, device=INTEGER plus, type=SOP)
    > ```markdown # 13. Component Adjust (환경안전 보호구: 안전모, 안전화) ## 13.1 Common list ### 13.1.3 PM Baffle Heater Auto Tune | Pict
  - [ ] `set_up_manual_supra_np` (score=5.5887, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I | 2) Jig Assy 조립도 확인 | a. Jig Assy 조립도를 확인하여 다음과 같은 방법으로 Jig Assy가 설치되어야 한다. | | | :--- | :--
  - [ ] `set_up_manual_supra_n` (score=5.2381, device=SUPRA N, type=SOP)
    > ```markdown Confidential I 36) LP2,3 Teaching <!-- Image (70, 70, 290, 226) --> a. LP1 Teaching 과 동일한 방법으로 LP2,3 를 진행하여 
  - [ ] `global_sop_supra_n_series_all_tm_multi_port` (score=4.8271, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_TM_MULTI PORT Global SOP No : Revision No: 1 Page: 18/22 | Flow | Procedure 
  - [ ] `global_sop_integer_plus_all_pm_slot_valve` (score=4.6906, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_CLN_PM_SLOT VALVE Global SOP No : Revision No : 2 Page : 35 / 39 | Flow | Procedur
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=4.6512, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `global_sop_geneva_xp_cln_chamber_all` (score=4.6114, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_CLN_Chamber_All Global SOP No: 0 Revision No: 0 Page: 4/47 ## 3. 사고 사례 ### 1) 화상 재해의 
  - [ ] `global_sop_supra_xp_all_pm_heater_chuck` (score=4.576, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP ZEDIUS XP_ALL_PM_HEATER # CHUCK Global SOP No: 0 Revision No: 2 Page: 21 / 49 | Flow | Procedur
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=4.569, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_LM GUIDE ## REPLACEMENT & GREASE INJECTION Global SOP No: Revision No: 1 Page:
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=4.5611, device=GENEVA XP, type=set_up_manual)
    > | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 2) Load | a. Load를 클릭한다. | | | | | | | 3) Set-up Recipe

#### q_id: `A-amb077`
- **Question**: Vacuum Pump Oil Level Check 주기 및 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, GENEVA_XP, INTEGER_PLUS]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-15):
  - [ ] `set_up_manual_supra_vm` (score=5.7045, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 8. EFEM Robot Teaching_Preparing | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 6) EFEM 
  - [ ] `supra_n_all_trouble_shooting_guide_trace_auto_scan_limit_out` (score=5.5394, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Auto Scan Limit Out] Confidential II | | | | | :--- | :--- | :--- | | | E-2. Robot
  - [ ] `global_sop_geneva_xp_rep_pm_vacuum_pump` (score=5.2652, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_REP_PM_Vacuum pump SOP No: 0 Revision No: 0 Page: 18/21 ## 10. Work Procedure | Flow 
  - [ ] `global_sop_supra_n_series_all_pm_isolation_valve` (score=5.1693, device=SUPRA N series, type=SOP)
    > # Global SOP_SUPRA N series_ALL_PM_ISOLATION VALVE Global SOP No : Revision No: 2 Page : 10 / 25 ## 5. Flow Chart Start 
  - [ ] `global_sop_supra_vplus_all_pm_controller` (score=5.0694, device=SUPRA Vplus, type=SOP)
    > ```markdown # Global SOP_SUPRA Vplus_REP_PM_Controller Global SOP No: Revision No: 3 Page: 12/42 ## 8. 필요 Tool | | Name 
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=4.9708, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM ROBOT TEACHING Global SOP No : Revision No: 6 Page : 79 / 107 | Flow | Pr
  - [ ] `set_up_manual_supra_np` (score=4.8671, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I # 10. Leak Check (※환경안전 보호구: 안전모, 안전화) ## 10.1 Leak Check | Picture | Description | Tool & Sp
  - [ ] `set_up_manual_supra_nm` (score=4.8575, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 11. Leak Check (※환경안전 보호구 : 안전모, 안전화) ## 11.1 Leak Check | Picture | Description | Tool & S
  - [ ] `set_up_manual_supra_xq` (score=4.8001, device=SUPRA XQ, type=SOP)
    > | Module | Action | A Prior Condition | 1st Check | 2nd Check | | :--- | :--- | :--- | :--- | :--- | | LL | LL2 Vacuum V
  - [ ] `global_sop_geneva_xp_adj_efem_efem_robot_leveling` (score=4.7606, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp_ADJ_EFEM_EFEM Robot leveling Global SOP No: Revision No: 0 Page: 8 / 15 ## 8. 필요 Tool
  - [ ] `set_up_manual_precia` (score=4.7425, device=PRECIA, type=set_up_manual)
    > ```markdown # 8. Leak Check (환경안전 보호구: 안전모, 안전화) ## 8.3 Module Pumping | Picture | Description | Tool & Spec | | :--- | 
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=4.7109, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `global_sop_supra_n_series_all_pm_heater_chuck` (score=4.7003, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_HEATER CHUCK Global SOP No : Revision No: 2 Page: 19/40 | Flow | Procedur
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=4.664, device=ZEDIUS XP, type=set_up_manual)
    > ```markdown # 6. Utility Turn On (PCW/GAS/PUMP)(※환경안전 보호구 : 안전모, 안전화) ## 6.6 TM Pump Turn on | Picture | Picture | Pictu
  - [ ] `set_up_manual_supra_n` (score=4.6524, device=SUPRA N, type=SOP)
    > Confidential I 2) Leak Check a. Pump Turn On 후 8시간 Full Pumping 후 Leak Check를 한다. b. Leak Check 조건 및 Spec은 고객사마다 상이. | |

#### q_id: `A-amb078`
- **Question**: Servo Motor 이상 진동 발생 시 점검 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, INTEGER_PLUS, PRECIA]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-9):
  - [ ] `global_sop_integer_plus_all_pm_pin_motor` (score=6.4552, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_PM_SERVO MOTOR Global SOP No: Revision No: 5 Page: 77 / 126 | Flow | Procedure
  - [ ] `global_sop_supra_xp_all_pm_pin_assy` (score=6.3886, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_HOOK LIFTER ## SERVO MOTOR CONTROLLER REPLACEMENT Global SOP No: Revision No: 
  - [ ] `global_sop_integer_plus_all_am_pin_motor` (score=5.9387, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_AM_SERVO MOTOR Global SOP No: Revision No: 4 Page: 44 / 84 ## 1. 환경 안전 보호구 | 구
  - [ ] `global_sop_supra_series_all_pm_bottom_structure` (score=5.9177, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_PM_BOTTOM STRUCTURE Global SOP No: 0 Revision No: 4 Page: 18 / 105 ## 3. Hook 
  - [ ] `global_sop_supra_n_series_all_pm_hook_lifter_servo_motor` (score=5.7344, device=SUPRA N, type=SOP)
    > # Global SOP_SUPRA N series_REP_PM_PIN MOTOR Global SOP No : Revision No: 2 Page : 12 / 106 ## 5. Flow Chart Start 1. Gl
  - [ ] `global_sop_precia_all_pm_chuck` (score=5.2568, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_CHUCK MOTOR TEACHING Global SOP No : Revision No: 1 Page: 97 / 132 ## 6. Work Pro
  - [ ] `precia_all_trouble_shooting_guide_pin_motor_abnormal` (score=5.2303, device=PRECIA, type=trouble_shooting_guide)
    > ```markdown # PRECIA Trouble Shooting Guide [Pin Motor Abnormal] Confidential II | Alarm Code | LED 점멸 횟수 | Alarm 종류 | 원
  - [ ] `40079239` (score=5.2264, device=SUPRA Vplus, type=myservice)
    > -. All PM Pump Down, Temp 정상 Reading X
-> Temp CTR 정상 확인
-. Rack 확인 시 ELCB0-1 Trip 확인
-. ELCB DVM Check 시 Input 220V 정상

  - [ ] `set_up_manual_precia` (score=5.1841, device=PRECIA, type=set_up_manual)
    > ```markdown # 7) 중량물 취급 작업 ## 중량물 취급 시 주의사항 - 발은 어깨 너비로, 허리는 반듯이 세우고 무릎의 힘으로 일어섭니다. <!-- Image (127, 161, 594, 247) --> 

#### q_id: `A-amb079`
- **Question**: Process Kit 수명 관리 기준은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, INTEGER_PLUS, OMNIS]
- **Scope**: ambiguous | **Intent**: procedure
- **ES candidates** (top-14):
  - [ ] `set_up_manual_ecolite_ii_400` (score=5.2537, device=ECOLITE II 400, type=set_up_manual)
    > ```markdown # 8. Part Installation_Process kit | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) Ceram
  - [ ] `global_sop_supra_xp_all_pm_process_kit` (score=4.5828, device=SUPRA XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_PROCESS KIT Global SOP No : Revision No : 3 Page : 6/28 ## 5. Worker Location 
  - [ ] `set_up_manual_supra_np` (score=4.4393, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I | | STAGE_R2_RDY[4] | | | 150.000 | | 270.000 | 270.000 | | :--- | :--- | :--- | :--- | :--- 
  - [ ] `set_up_manual_supra_nm` (score=4.3243, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 8. Part Installation (※환경안전 보호구 : 안전모, 안전화) ## 8.1 Process kit Install (Focus Adaptor & Baf
  - [ ] `global_sop_precia_all_pm_prevent_maintenance` (score=4.1763, device=PRECIA, type=SOP)
    > # Global SOP_PRECIA_REP_PM_PROCESS KIT (TOP MOUNT TYPE) Global SOP No: Revision No: 5 Page: 8/108 ## 3. Flow Chart Start
  - [ ] `set_up_manual_geneva_stp300_xp_r6` (score=4.113, device=GENEVA XP, type=set_up_manual)
    > ```markdown # 1. Install Preparation(※환경안전 보호구 : 안전모, 안전화) ## 1.1 Foot Print Drawing | Picture | Description | Tool & Sp
  - [ ] `set_up_manual_ecolite_2000` (score=4.0981, device=ECOLITE 2000, type=set_up_manual)
    > ```markdown # 8. Part Installation | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) Ceramic Parts<br>
  - [ ] `global_sop_supra_n_series_all_efem_controller` (score=4.0631, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_EFEM_CONTROLLER Global SOP No : Revision No: 1 Page : 33 / 40 | Flow | Proce
  - [ ] `set_up_manual_ecolite_3000` (score=3.9214, device=ECOLITE3000, type=set_up_manual)
    > ```markdown | 1. Installation Preperation (Layout, etc.) | | | | :--- | :--- | :--- | | Picture | Description | Tool & S
  - [ ] `global_sop_supra_n_series_all_pm_process_kit` (score=3.9077, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_PROCESS KIT Global SOP No : Revision No: 0 Page: 15/55 | Flow | Procedure
  - [ ] `set_up_manual_integer_plus` (score=3.855, device=INTEGER plus, type=SOP)
    > ```markdown | 17-20. TTTM, 8계통 | | | | | | | :--- | :--- | :--- | :--- | :--- | :--- | | Picture | Description | Data | 
  - [ ] `global_sop_geneva_xp_all_8계통_check` (score=3.7714, device=GENEVA XP, type=SOP)
    > ```markdown # GENEVA XP_8계통_Check sheet SOP No: 0 Revision No: 0 Page: 19 / 46 ## 10. Work Procedure | Flow | Procedure 
  - [ ] `set_up_manual_zedius_xp_supra_xp` (score=3.665, device=ZEDIUS XP, type=set_up_manual)
    > ```markdown # 1. Install Preperation (※환경안전 보호구 : 안전모, 안전화) ## 1.1 Foot Print Drawing | Picture | Description | Tool & S
  - [ ] `set_up_manual_supra_vm` (score=3.6567, device=SUPRA Vm, type=set_up_manual)
    > ```markdown # 1. Template Draw | Picture | Description | Tool & Spec | | :--- | :--- | :--- | | 1) Template Drawing 사전 준

#### q_id: `A-amb080`
- **Question**: N2 Purge Line 누설 점검 방법은?
- **Devices**: [SUPRA_N, SUPRA_XP, PRECIA, GENEVA_XP]
- **Scope**: ambiguous | **Intent**: troubleshooting
- **ES candidates** (top-18):
  - [ ] `global_sop_supra_n_series_all_tm_wafer_sensor` (score=5.6735, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ADJ_TM_WAFER SENSOR Global SOP No : Revision No: 2 Page: 12/25 ## 6. Work Proced
  - [ ] `set_up_manual_integer_plus` (score=5.6345, device=INTEGER plus, type=SOP)
    > ```markdown # 8. Utility Turn On - N2 Gas (환경안전 보호구: 안전모, 안전화) ## 8.2 N2 Gas Turn on | Picture | Description | Tool & Sp
  - [ ] `global_sop_geneva_xp_rep_pm_o2_analyzer_modify` (score=5.3252, device=GENEVA XP, type=SOP)
    > ```markdown # Global SOP_GENEVA xp REP PM O2 ANALYZER Modify Global SOP No: 0 Revision No: 0 Page: 19 / 33 | Flow | Proc
  - [ ] `global_sop_supra_xp_all_ll_cip` (score=5.3121, device=SUPRA XP, type=SOP)
    > # Global SOP_SUPRA XP_MODIFY_LL_N2 ## PURGE LINE DIFFUSER Global SOP No: Revision No: 0 Page: 27 / 51 | Flow | Procedure
  - [ ] `global_sop_precia_all_pm_mfc` (score=5.0134, device=PRECIA, type=SOP)
    > ```markdown # Global SOP_PRECIA_REP_PM_MFC Global SOP No : Revision No: 1 Page: 14 / 23 ## 6. Work Procedure | Flow | Pr
  - [ ] `set_up_manual_supra_np` (score=4.9939, device=SUPRA Np, type=set_up_manual)
    > ```markdown Confidential I | 2) Jig Assy 조립도 확인 | a. Jig Assy 조립도를 확인하여 다음과 같은 방법으로 Jig Assy가 설치되어야 한다. | | | :--- | :--
  - [ ] `global_sop_integer_plus_all_ll_mfc` (score=4.9565, device=INTEGER plus, type=SOP)
    > ```markdown # Global SOP_INTEGER plus_REP_LL_MFC Global SOP No: Revision No: 1 Page: 15 / 20 | Flow | Procedure | Tool &
  - [ ] `precia_all_trouble_shooting_guide_leak_abnormal` (score=4.8793, device=INTEGER plus, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace leak rate over] Confidential II | LL Leak Trace | C-4. Vacuum Line | ▶ Vacuu
  - [ ] `supra_n_all_trouble_shooting_guide_trace_loadport_abnormal` (score=4.8496, device=SUPRA N, type=trouble_shooting_guide)
    > ```markdown # Trouble Shooting Guide [Trace Loadport Abnormal] Confidential II | | F-4. Log | Post mapping log & FA log 
  - [ ] `set_up_manual_supra_n` (score=4.7848, device=SUPRA N, type=SOP)
    > ```markdown Confidential I 36) LP2,3 Teaching <!-- Image (70, 70, 290, 226) --> a. LP1 Teaching 과 동일한 방법으로 LP2,3 를 진행하여 
  - [ ] `global_sop_supra_xp_all_pm_flow_switch` (score=4.7739, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_PM_FLOW # SWITCH REPLACEMENT Global SOP No: Revision No: 2 Page: 13 / 32 | Flow |
  - [ ] `set_up_manual_supra_nm` (score=4.7537, device=SUPRA Nm, type=set_up_manual)
    > ```markdown Confidential I # 7. Teaching (※환경안전 보호구 : 안전모, 안전화) ## 7.2 EFEM Robot Leveling & Loadport Teaching | Picture
  - [ ] `global_sop_supra_n_all_sub_unit_igs_block` (score=4.6727, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N_ALL_SUB UNIT_IGS BLOCK Global SOP No : Revision No: 1 Page: 47 / 67 | Flow | Procedure 
  - [ ] `global_sop_supra_xp_all_ll_flow_switch` (score=4.6074, device=ZEDIUS XP, type=SOP)
    > ```markdown # Global SOP_ZEDIUS XP_ALL_LL_FLOW ## SWITCH REPLACEMENT Global SOP No: Revision No: 2 Page: 12/30 | Flow | 
  - [ ] `40085110` (score=4.5935, device=TIGMA Vplus, type=myservice)
    > WPSKK5PV00 PM3 PURGE N2 REGULATOR LEAK
  - [ ] `global_sop_supra_n_series_all_tm_robot` (score=4.5682, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_ALL_TM ROBOT ASSY REPLACEMENT Global SOP No : Revision No: 6 Page: 49 / 107 ## 8
  - [ ] `global_sop_supra_series_all_sw_operation` (score=4.5448, device=SUPRA N, type=SOP)
    > ```markdown # Global SOP_SUPRA SERIES_ALL_ SW OPERATION Global SOP No Revision No: 2 Page: 11/49 ## 2. Gas Line Leak Che
  - [ ] `global_sop_supra_n_series_all_pm_heater_chuck` (score=4.5434, device=SUPRA N series, type=SOP)
    > ```markdown # Global SOP_SUPRA N series_REP_PM_HEATER CHUCK CONNECTOR Global SOP No : Revision No: 2 Page: 38/40 | Flow 

