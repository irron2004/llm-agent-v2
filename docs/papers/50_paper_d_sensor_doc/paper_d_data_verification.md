# Paper D — 데이터 검증 Agent 지시문

> 작성일: 2026-04-14
> 목적: 논문 주제 성립 여부를 데이터로 빠르게 확인

---

## Agent 1: 로그 데이터 담당

### 목표
정비로그가 논문용으로 쓸 수 있을 만큼 구조화 가능한지 확인

### 지시문

```
반도체 장비 정비로그에서 아래 항목을 추출해 표로 정리해줘.

- log id
- equipment id / chamber id
- 작업 생성시각, 작업 시작시각, 작업 종료시각
- free-text 원문
- 언급된 component 명칭
- symptom 표현
- 원인(cause) 표현
- 수행 조치(action) 표현
- 교체 부품(part) 표현
- 알람 코드 또는 에러 코드
- 동일 의미 다른 표현 예시

그리고 아래 통계도 내줘.
- 전체 로그 수
- 장비별 로그 수
- component 명시 비율
- 원인 명시 비율
- 조치 명시 비율
- 시간정보 누락 비율
- free-text 길이 분포

마지막으로, APC / pressure / valve / position / calibration / feedback 관련 로그 100건을 별도로 추려줘.
```

### 확인 포인트
- [ ] 로그가 component / cause / action으로 쪼개질 수 있는가
- [ ] 특정 failure family가 논문 한 편 분량으로 충분한가
- [ ] 시간정보가 어느 정도 신뢰 가능한가

---

## Agent 2: 센서 데이터 담당

### 목표
센서 쪽에서 event-centric episode를 만들 수 있는지 확인

### 지시문

```
반도체 센서 데이터에서 아래를 정리해줘.

- 설비별 센서 목록
- 각 센서 샘플링 주기
- 결측률
- 장비 stop/run 구간 구분 가능 여부
- recipe / step 정보 존재 여부
- 알람/이벤트 로그와 timestamp 기준 병합 가능 여부
- APC 또는 pressure-control 관련 센서 목록
- setpoint와 actual이 동시에 있는 센서 목록

최근 6개월 기준 이상 추정 event 후보 100개 추출
기준 예시: tracking error 지속, saturation, oscillation, drift, stuck pattern

각 후보마다 아래를 포함해줘.
- equipment/chamber
- 시작시각, 종료시각
- 관련 센서명
- 이상 패턴 종류
- 전후 2시간 window 존재 여부
- 대응되는 로그가 ±1일, ±3일, ±7일 내에 있는지
```

### 확인 포인트
- [ ] 센서에서 episode 단위로 자를 수 있는가
- [ ] setpoint/actual 기반의 해석 가능한 이상 정의가 가능한가
- [ ] 로그와 시간적으로 연결 가능한가

---

## Agent 3: 센서-로그 연결 담당

### 목표
실제로 논문 주제가 성립하는지 가장 빨리 검증

### 지시문

```
센서 이상 episode와 정비로그를 연결하는 pilot set을 만들어줘.
대상은 APC/pressure-control 계열로 제한하고, 최근 1년 데이터에서 50~100개 episode를 뽑아줘.

각 episode에 대해 아래를 표로 정리해줘.
- episode id
- equipment/chamber
- 센서 이상 시작/종료 시각
- 핵심 이상 패턴 요약
- ±1일 / ±3일 / ±7일 내 관련 로그 후보
- 후보 로그의 component, symptom, cause, action
- 사람이 봤을 때 link quality를 gold / silver / weak / none으로 판정
- 가장 자주 등장하는 failure mode 상위 10개
- 가장 자주 등장하는 action 상위 10개
- 텍스트 표현의 동의어 예시

마지막으로, 사람이 보기에도 "센서 패턴과 로그가 분명히 연결된다"고 볼 수 있는 대표 사례 10개를 뽑아줘.
```

### 확인 포인트
- [ ] 센서 패턴과 로그가 실제로 연결되는 사례가 존재하는가
- [ ] gold link가 50건 이상 확보 가능한가
- [ ] 반복적으로 등장하는 failure mode/action 패턴이 있는가

---

## 논문 가능성 판정 기준 (Agent 결과로 확인)

| 항목 | 기준 | 결과 |
|------|------|------|
| APC/pressure 계열 로그 수 | 최소 수백 건 | TBD |
| 관련 센서 수 | 10개 이상 | TBD |
| setpoint/actual 쌍 존재 | 있어야 함 | TBD |
| recipe/step 정보 존재 | 있어야 함 | TBD |
| 로그 component 명시 비율 | 높을수록 좋음 | TBD |
| ±3일 내 연결 가능 비율 | 높을수록 좋음 | TBD |
| Gold link 수 | 50+ 파일럿 / 100~300 retrieval / 300+ 박사 안정 | TBD |
