# HVSR 아이디어 노트 (Paper A 확장)

작성일: 2026-03-18  
대상: `Paper A` 코어 방법 확장 초안 (P8/P9 후보)

---

## 1) 문제 재정의

현재 결과(P6/P7)에서 확인된 핵심은 다음이다.

- soft penalty 기반 재점수화는 contamination을 구조적으로 해결하지 못함
- 이유는 contamination이 **rerank 이전**, 즉 candidate generation/scope selection 단계에서 이미 발생하기 때문
- 따라서 해결점은 post-hoc demotion이 아니라 **scope hypothesis를 먼저 만들고 검증한 뒤 hard retrieval** 하는 방식이어야 함

즉 결론은 `P6/P7이 무의미`가 아니라:

> cross-equipment contamination은 ranking 문제가 아니라 scope selection 문제다.

---

## 2) 제안 방법: HVSR

HVSR (Hypothesis-Verified Scope Retrieval)은 장비 스코프를 단일 추정으로 고정하지 않고, 상위 가설들을 병렬 검증한 뒤 최종 스코프를 선택한다.

### 2.1 Stage 0: 정규화

- alias canonicalization (예: `ZEDIUS XP -> SUPRA_XP`, `SUPRA V -> SUPRA_VPLUS`)
- device/equip/doc_type canonical mapping 적용
- explicit device/equip가 있으면 hard evidence로 반영

### 2.2 Stage 1: Device hypothesis proposal

질의 `q`와 문맥 `h`로 device 후보 집합 `H(q)`를 생성:

`H(q) = TopMDevices(q, h)`

제안 점수(간단 버전):

`S_prop(d) = w_alias*S_alias + w_lex*S_lex + w_ctx*S_ctx + w_probe*S_probe`

- `S_alias`: alias/explicit mention 점수
- `S_lex`: query-token과 device profile의 lexical 적합도
- `S_ctx`: 직전 turn/sticky context 점수
- `S_probe`: 가벼운 1차 probe retrieval의 device 분포 점수

권장 시작값: `M=3`

### 2.3 Stage 2: 가설별 hard retrieval

각 가설 device `d`에 대해 독립적으로 hard-filter retrieval 수행:

`R_L(q; d) = Retrieve(q, D_device=d, L)`

핵심:

- global 후보를 먼저 뽑지 않음
- 각 가설에서 후보를 새로 생성하므로 candidate ceiling 문제를 완화

권장 시작값: `L=30`

### 2.4 Stage 3: Hypothesis verification

각 가설의 증거 강도를 계산:

`E(d; q) = a*Mass + b*Coverage + c*DocTypeMatch - g*SharedDominance`

- `Mass`: 상위 문서 점수 총량
- `Coverage`: 질의 핵심 토큰 커버리지
- `DocTypeMatch`: intent/doc_type 적합도
- `SharedDominance`: shared 문서 과다 점유 패널티

### 2.5 Stage 4: 최종 scope 선택

`d* = argmax_d [ eta*S_prop(d) + (1-eta)*E(d; q) ]`

최종 검색 결과:

`R_final(q) = TopK(R_L(q; d*))`

이로써 최종 top-k는 단일 device scope를 유지하여 contamination을 낮춘다.

---

## 3) Shared 문서 처리: SSG (선택)

`B4.5` 역효과(shared flood)를 피하기 위해 shared 문서는 별도 채널로 처리:

- shared index/pool 분리
- query별 gate로 열기 (`g_shared(q) in {0,1}`)
- quota 제한 (`B <= 1~2` docs in top-k)

결합:

`R_final = TopK( R_device(d*) U TopB(R_shared) )`

---

## 4) P6/P7 대비 차이

- P6/P7: global top-k 이후 점수 미세 조정 (post-hoc)
- HVSR: scope 가설별 hard candidate 생성 + evidence 검증 (pre-selection)

메시지 전환:

> soft scoring 실패는 알고리즘 실패가 아니라 문제의 위치를 보여주는 진단이다.  
> contamination은 candidate generation 단계에서 제어해야 한다.

---

## 5) 최소 구현 스펙 (HVSR-v1)

파라미터를 최소화해 과튜닝 리스크를 줄인다.

- `M` (가설 수): 기본 3
- `L` (가설별 retrieval depth): 기본 30
- `eta` (proposal vs evidence 결합): 기본 0.5
- `B` (shared quota): 기본 1

초기 버전에서 제외:

- 복잡한 Bayesian joint state
- user preference prior
- family graph 고급 확장

---

## 6) 실험 설계 (Paper A 반영안)

### 6.1 비교군

- `B3`: Hybrid + Rerank (global)
- `B4`: Hard device filter (oracle upper bound)
- `B4.5`: B4 + naive shared allowance
- `P6/P7`: soft scoring (negative ablation)
- `P8`: HVSR
- `P9`: HVSR + SSG

### 6.2 평가셋

- explicit_device
- explicit_equip
- implicit
- ambiguous/shared challenge

### 6.3 메트릭

- `Strict/Loose Hit@10`
- `Adjusted Cont@10`, `CE@10`
- `MRR`, `NDCG@10`
- `Latency`, candidate count

### 6.4 성공 기준 (초안)

- `P8`이 `B3` 대비 contamination 유의 감소 + strict hit 유의 개선
- `P8`이 `B4` oracle upper bound에 근접 (특히 explicit_device)
- `P9`가 ambiguous/shared에서 `B4.5`보다 안정적 (shared flood 억제)

---

## 7) 리스크와 대응

- 리스크: 2-pass/다중 retrieval로 latency 증가
  - 대응: `M` 고정(3), retrieval depth 제한, 캐시
- 리스크: SSG가 과도하면 recall 하락
  - 대응: quota 작은 값부터 시작, gate 조건 점진 완화
- 리스크: parser/alias 오류가 proposal 단계에서 전파
  - 대응: hard alias 품질 점검 + proposal에서 probe evidence 병합

---

## 8) Paper A에 넣는 방식

권장 포지션:

- 본문 메인: `B4 upper bound` + `P8/P9` (새 알고리즘 기여)
- `P6/P7`: 본문 짧은 실패 분석 또는 부록으로 축소

기여 문장(요지):

1. contamination은 candidate generation-level failure임을 실증
2. post-hoc soft scoring의 구조적 한계를 분석
3. multi-hypothesis hard-scope retrieval(HVSR) 제안
4. shared flood 방지를 위한 selective gate(SSG) 제안

---

## 9) 즉시 실행 TODO

- [ ] `P8` 구현: hypothesis proposal + per-hypothesis hard retrieval + verify
- [ ] `P9` 구현: shared gate + quota
- [ ] 기존 evaluator에 `P8/P9` 조건 추가
- [ ] 4개 eval split에서 재실험
- [ ] 통계 검정(bootstrap/McNemar) + latency 표 추가
- [ ] 원고 Section 3/5 기여 문장 업데이트

