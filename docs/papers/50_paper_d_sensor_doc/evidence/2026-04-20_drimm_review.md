# DriMM 논문 리뷰 — Paper D 관점

> 리뷰일: 2026-04-20
> 논문: DriMM: Drilling Multimodal Model for Time-Series and Text in the Era of Large Models
> 출처: ICML 2025 FMSD Workshop (OpenReview, 2025-06-09)
> 링크: https://openreview.net/pdf?id=NlOwF1b84H

---

## 1. 한 줄 요약

> Drilling sensor window와 Daily Drilling Report 텍스트를 bi-encoder + InfoNCE contrastive learning으로 같은 embedding 공간에 정렬한 **산업용 CLIP** 논문.

---

## 2. 핵심 구조

```
Sensor window (10 sensors, 512 steps)
    → LM4TS encoder (Moirai / MOMENT)
    → linear projector
    → z_ts

DDR text (operation description)
    → RoBERTa-base
    → linear projector
    → z_txt

Loss: InfoNCE (bidirectional)
    sim(z_ts, z_txt) — positive pair 가깝게, negative 멀게
```

## 3. 데이터

- 1,787 multivariate time series
- 10 sensor features (hookload, torque, flow rate 등)
- 1초 간격 → 65,536 step window → 512 step subsampling
- **145,715 paired samples** (sensor window ↔ DDR text)

## 4. 결과

- Cross-modal retrieval: Recall@1은 높지 않지만, class-level F1@10은 유의미
- Zero-shot classification: 3-class 75.7%, 9-class 44.2% (frozen text encoder)
- Linear probing: 3-class 89.3%, 9-class 74.2% (non-frozen)
- 저자 인정 한계: "모든 negative를 dissimilar로 가정하는 InfoNCE는 한계, hard negative mining 필요"

---

## 5. Paper D와의 관계

### DriMM = Paper D의 가장 가까운 선행연구

| 항목 | DriMM | Paper D |
|------|-------|---------|
| 도메인 | Oil & Gas drilling | 반도체 정비 |
| 센서 입력 | **raw sensor window** → LM4TS | **event abstraction** (rule-initialized) |
| 텍스트 대상 | DDR operation ("뭘 했나") | **maintenance cause/action/SOP** ("왜, 어떻게") |
| 목표 | activity retrieval/classification | **evidence/root-cause retrieval** |
| Loss | 기본 InfoNCE | **hierarchy-aware hard negative** |
| Pair 구성 | sensor ↔ operation (시간 매칭) | sensor event ↔ maintenance case (**weak pairing**) |
| Contamination | 다루지 않음 | **핵심 기여** (wrong-device/module) |

### Paper D의 차별점 3가지

**1. Event abstraction (vs raw window)**
- DriMM: raw sensor → LM4TS 직접 입력
- Paper D: 문서에서 정의한 failure vocabulary 기반 event token으로 변환
- 이유: 정비 문서는 "pressure hunting"이라 쓰지 "값이 91.2%"라 쓰지 않음

**2. Maintenance evidence retrieval (vs activity alignment)**
- DriMM: "어떤 drilling 작업이었나" 검색
- Paper D: "왜 문제가 생겼나, 뭘 해야 하나" 검색
- 즉, operation alignment → **diagnostic evidence retrieval**로 확장

**3. Hierarchy-aware hard negative (vs basic InfoNCE)**
- DriMM 저자가 직접 한계로 인정: "semantically similar operations 때문에 hard negative 필요"
- Paper D가 이 gap을 해결: 장비/모듈/부품 계층 반영 hard negative

---

## 6. Related Work에서의 위치 (논문 문장 초안)

> DriMM (ICML 2025 FMSD Workshop)은 drilling 도메인에서 multivariate sensor window와 daily drilling report를 contrastive alignment로 정렬하여, 산업 시계열-텍스트 cross-modal retrieval의 가능성을 보여주었다. 그러나 DriMM은 raw sensor window를 직접 입력으로 사용하여 maintenance-relevant event abstraction을 다루지 않으며, 동일 장비 family 내의 semantically similar negative 처리를 future work으로 남겼다. 본 연구는 이 gap을 (1) 문서 기반 failure taxonomy로 event abstraction을 수행하고, (2) hierarchy-aware hard negative mining으로 contamination을 감소시키며, (3) operation alignment가 아닌 maintenance evidence retrieval로 확장한다.

---

## 7. Novelty Boundary (DriMM 이후)

**Paper D의 novelty를 "산업 시계열-텍스트 alignment"로 잡으면 안 된다.** 그건 DriMM이 이미 했다.

Paper D의 novelty는 반드시 이렇게 좁혀야 한다:

> 산업 시계열-텍스트 alignment 자체가 아니라, **센서 "이상 이벤트"를 정비 근거 문서와 연결하고, temporal uncertainty와 equipment hierarchy contamination을 다루는 retrieval 문제**

### Contribution 재정의 (DriMM 이후)

1. **Sensor event abstraction** — raw window가 아닌 event episode (DriMM과 차별)
2. **Weakly supervised maintenance evidence alignment** — noisy temporal pairing (DriMM은 깔끔한 pair)
3. **Hierarchy-aware hard negative** — DriMM이 future work으로 남긴 것을 해결

### Temporal uncertainty의 위치 변경

- 기존: 제목/중심 키워드
- 수정: **weak supervision 문제로 본문에** ("maintenance logs are not synchronized with sensor anomalies → positive pairs are noisy → we use temporal windows + metadata + symptom cues for weak pairing")

### 실험에 DriMM-style baseline 필수

```
DriMM-style baseline:
  raw sensor summary + maintenance text + standard InfoNCE bi-encoder

Paper D proposed:
  sensor event episode + maintenance evidence + hierarchy-aware hard negative
```

이 비교가 없으면 reviewer가 "DriMM 방식 그대로 하면 안 되나?"라고 물을 것.

## 8. Paper D에 주는 시사점

1. **산업 시계열-텍스트 alignment가 학술적으로 인정받는 방향** — ICML workshop 수준
2. **bi-encoder + InfoNCE가 기본 구조로 충분** — 복잡한 architecture 필요 없음
3. **DriMM이 남긴 hard negative 한계가 Paper D의 기여로 직결**
4. **경쟁 위험**: 누군가 DriMM을 반도체/정비에 적용하면 차별점 약화 → **빨리 1편을 내야 함**
5. **"alignment 자체"는 더 이상 novelty가 아님** — event abstraction + weak pairing + hierarchy가 novelty

---

## Related Documents

- `paper_d_failure_taxonomy.md` — 문서 기반 이벤트 정의 (DriMM과의 차별점 1)
- `daily/2026-04-20--final-strategy-confirmed.md` — 최종 scope (hierarchy-aware = 차별점 3)
- `evidence/2026-04-14_literature_survey.md` — 전체 문헌 조사
- `evidence/paper_d_paper_comparison_table.md` — 논문 비교표 (DriMM 추가 필요)
