# Paper D — 전체 센서-문서 매칭 스캔 결과

> 스캔일: 2026-04-16
> 대상: `rag_chunks_dev_current` (SUPRA Vplus, myservice + gcb)
> 방법: 52,406 chunks → 14,052 고유 문서로 병합 → 센서별 regex 패턴 매칭

---

## 1. 전체 결과

| 센서 그룹 | 매칭 문서 수 | myservice | gcb | Paper D 활용 가능성 |
|-----------|----------:|----------:|----:|-------------------|
| **Temp_Heater** (heater chuck, temperature diff/control) | **2,057** | 2,025 | 32 | ★★★★★ 가장 많음, failure family 후보 |
| **Gas_Flow_MFC** (MFC, mass flow, gas flow/leak/line) | **573** | 536 | 37 | ★★★★☆ 두 번째로 많음 |
| **EPD_General** (EPD 키워드 전체) | **377** | 345 | 32 | ★★★★☆ |
| **APC_General** (APC 키워드 전체) | **319** | 309 | 10 | ★★★★★ 핵심 타겟 |
| **Temp2** | **227** | 225 | 2 | ★★★★☆ |
| **SourcePwr_Forward** (RF power, source power) | **208** | 195 | 13 | ★★★☆☆ |
| **Pressure_Chamber** (chamber pressure, baratron) | **145** | 140 | 5 | ★★★★☆ APC와 연관 높음 |
| **APC_Pressure** | **66** | 66 | 0 | ★★★★★ 핵심 — failure mode 직접 연결 |
| **Temp1** | **57** | 55 | 2 | ★★★★☆ |
| **SourcePwr_Reflect** (reflect, RF reflect) | **36** | 24 | 12 | ★★★☆☆ |
| **APC_SetPoint** | **35** | 33 | 2 | ★★★★☆ APC 계열 보강 |
| **APC_Position** | **23** | 21 | 2 | ★★★★★ 핵심 — ES 원문 검증 완료 |
| **Gas_Pressure** | **16** | 16 | 0 | ★★☆☆☆ |
| **EPD_Monitor** (EPD monitor, endpoint detection) | **10** | 5 | 5 | ★★★☆☆ |
| **Gas_Valve** | **6** | 6 | 0 | ★★☆☆☆ |
| **Gas_Temp** | **3** | 1 | 2 | ★☆☆☆☆ |
| **Recipe_Step** | **1** | 1 | 0 | ★☆☆☆☆ |
| **EPD_Amp** | **0** | 0 | 0 | ☆☆☆☆☆ 문서 없음 |

---

## 2. 핵심 해석

### 2.1 Paper D pilot에 가장 적합한 센서 그룹

**1순위: APC 계열** (총 319문서, 세부 합산 기준)
- APC_General: 319
- APC_Pressure: 66
- APC_Position: 23
- APC_SetPoint: 35
- 이유: 이미 ES 원문 검증 완료, failure mode (pressure hunting, position drift) 직접 확인됨

**2순위: Temp 계열** (총 2,057+ 문서)
- Temp_Heater: 2,057 (heater chuck 관련)
- Temp2: 227
- Temp1: 57
- 이유: 문서량이 압도적으로 많아 학습 데이터 충분. Temp1 FDC out of spec 사례도 검증됨

**3순위: Gas/MFC 계열** (총 573문서)
- Gas_Flow_MFC: 573
- Gas_Pressure: 16
- Gas_Valve: 6
- 이유: MFC 관련 문서가 많으나, raw sensor name과 문서 표현 사이 gap이 큼

### 2.2 semantic gap 정량적 증거

| 센서 raw name | ES exact match (이전 결과) | regex 확장 매칭 (이번 결과) | 배율 |
|--------------|------------------------:|------------------------:|-----:|
| APC_Position | 25 hits | 23 docs | ~1x |
| APC_Pressure | 58 hits | 66 docs | ~1x |
| Temp1 | 80 hits | 57 docs | ~1x |
| Temp2 | 242 hits | 227 docs | ~1x |
| EPD_Monitor1 | 1 hit | 10 docs (EPD_Monitor) | **10x** |
| SourcePwr1_Reflect | 0 hits | 36 docs | **∞** |
| Gas1~6_* | 0~1 hits | 573 docs (MFC/flow) | **500x+** |

**핵심**: raw sensor name으로는 11%만 잡히지만, 확장 키워드로 검색하면 훨씬 더 많은 관련 문서가 존재.
→ **sensor name → document term mapping (synonym dictionary)이 논문의 필수 전처리 단계**임을 실증.

### 2.3 문서가 없는 센서

- **EPD_Amp**: 0건 — 문서에서 전혀 언급되지 않음
- **Recipe_Step_Num**: 1건 — 독립 센서로서보다 다른 문서의 context 정보
- **Gas_Temp**: 3건 — 거의 다루어지지 않음

이 센서들은 Paper D의 초기 pilot에서 **제외하는 것이 안전**.

---

## 3. Paper D 논문 기여 관점

이 스캔 결과는 다음을 실증:

1. **14,052개 문서 중 센서 관련 문서는 충분히 존재** — pilot set 구축 가능
2. **sensor name ≠ document term** — lexical expansion 필요성의 정량적 근거
3. **APC + Temp 계열로 시작하면 수백~수천 건의 문서와 연결 가능** — 데이터 부족 문제 없음
4. **센서 그룹별 문서량 차이가 큼** — 모든 센서를 동일하게 다루면 안 되고, failure family 선택이 중요

---

## 4. 추천 pilot 설계

```
Phase 1: APC 계열 (319문서)
  - APC_Position + APC_Pressure + APC_SetPoint
  - Pressure_Chamber (145문서)와 교차 확인
  - 예상 pilot set: 50~100 episodes

Phase 2: Temp 계열 (2,057문서)
  - Temp1 + Temp2 + Temp_Heater
  - 문서량이 많아 학습 데이터로 확장 가능
  
Phase 3: Gas/MFC 계열 (573문서)
  - synonym dictionary 구축 후 진행
```

---

## 5. 데이터 파일

상세 매칭 결과 (doc_id별):
`evidence/paper_d_sensor_doc_matches.json`

---

## Related Documents
- `paper_d_es_query_results.md` — 첫 번째 ES 조회 (raw name 기준)
- `paper_d_keyword_query_log.md` — 키워드별 원문 검증 로그
- `paper_d_data_verification.md` — 데이터 검증 Agent 지시문
