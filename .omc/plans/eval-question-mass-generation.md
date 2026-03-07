# Eval 질문 대량 생성 Plan (Updated 2026-03-06)

> 목표: ES 문서 기반으로 document-grounded eval 질문을 대량 생성하여 retrieval 성능 평가 데이터셋 완성

## 1. 현재 상태 (2026-03-06 기준)

### 질문 생성 진행률

| 항목 | 수량 | 상태 |
|------|------|------|
| Main CSV annotated (qid 240-450) | 211개 | DONE |
| 박진우 formal eval (별도 파일) | 79개 (+ 유사질문 3개씩) | DONE |
| Main CSV unannotated (qid 1-239) | 239개 | expected_doc 없음 |
| **Total document-grounded** | **290개** | 목표 200개 초과 달성 |

### doc_type 분포 (Main CSV annotated 211개)

| doc_type | 생성 수 | 목표 수 | 달성률 |
|----------|---------|---------|--------|
| sop | 86 | 80 | 107% |
| ts | 45 | 40 | 112% |
| setup | 21 | 20 | 105% |
| gcb | 39 | 40 | 97% |
| myservice | 20 | 20 | 100% |

### Retrieval Probe 진행률

| 항목 | 수량 | 상태 |
|------|------|------|
| Probe 완료 (50_draft) | 52개 | DONE → `data/eval_questions_with_retrieval_probe.csv` |
| Probe 미완료 (qid 302-450) | 159개 | TODO |
| 박진우 변형 probe | 79개 (+ 237 유사질문) | TODO |
| 50_draft 결과 | hit@5=82.7%, hit@10=82.7% | 9 misses |

### Embedding 파이프라인

| 모델 | 상태 | 비고 |
|------|------|------|
| bge_m3 | DONE | |
| jina_v5 | DONE | 395,686 chunks |
| qwen3_emb_4b | RUNNING | Task b95dbcqux, GPU:0 |

## 2. 남은 작업 (우선순위 순)

### Step A: Retrieval Probe 완료 (나머지 159개)
- **대상**: Main CSV qid 302-450 중 expected_doc이 있는 질문
- **방법**: ES BM25 top10 검색 → hit@5, hit@10, hard_negatives 기록
- **출력**: `data/eval_questions_with_retrieval_probe.csv`에 append
- **소요**: ~5분

### Step B: 박진우 변형 질문 Retrieval Probe
- **대상**: `data/eval_sop_question_list_박진우_변형.csv` 79개 원본 + 237개 유사질문
- **방법**: 원본/유사질문 각각 ES BM25 top10 검색
- **목적**: 유사질문이 원본과 동일한 문서를 retrieve하는지 검증
- **출력**: `data/eval_sop_question_list_박진우_retrieval_probe.csv`
- **소요**: ~5분

### Step C: 커버리지 리포트 생성
- 전체 질문셋 (290개) doc_type/언어/장비/질문유형 분포
- hit@5/hit@10 전체 성능 요약
- hard_negatives 빈도 상위 문서 분석 (자주 혼동되는 문서 식별)
- 부족 영역 식별
- **소요**: ~3분

### Step D: qwen3 Embedding 완료 대기 후 ES Ingest
- qwen3_emb_4b 완료 확인
- 3개 모델 embedding을 ES embed indices에 ingest
- **소요**: embedding 완료 대기 + ingest ~10분

### Step E: 3-Model 비교 Eval 실행
- bge_m3, jina_v5, qwen3_emb_4b 3개 모델로 retrieval eval
- 290개 document-grounded 질문 사용
- hit@1/5/10, MRR 비교
- **소요**: ~15분

### Step F: Content Index Gap 조사 (optional)
- `chunk_v3_content`: 390,385 docs vs embedding: 395,686 chunks
- 5,301개 누락 문서 원인 파악
- **소요**: ~5분

## 3. 파일 현황

```
data/
├── eval_questions_from_chat.csv              # 통합 CSV (450개, 211개 annotated)
├── eval_questions_50_draft.csv               # 52개 draft (merged)
├── eval_questions_250_299_draft.csv          # 50개 draft (merged)
├── eval_questions_302_401_draft.csv          # 100개 draft (merged)
├── eval_questions_sop_draft.csv              # 20개 SOP draft (merged)
├── eval_questions_ts_draft.csv               # 10개 TS draft (merged)
├── eval_questions_with_retrieval_probe.csv   # probe 결과 (52개 완료)
├── eval_sop_question_list_박진우.csv          # 원본 79개
├── eval_sop_question_list_박진우_변형.csv      # 79개 + 유사질문1~3
```

## 4. Acceptance Criteria

- [x] 200개 이상의 신규 질문이 expected_doc과 함께 생성됨 (211개)
- [x] doc_type 분포가 목표 +/-5% 이내
- [ ] 모든 annotated 질문에 대해 ES BM25 top10 retrieval probe 완료
- [ ] 박진우 유사질문 retrieval probe 완료
- [ ] hard_negatives 기록 완료
- [ ] 커버리지 리포트 생성
- [ ] qwen3 embedding 완료 후 3-model 비교 eval 실행
- [ ] CSV 포맷이 `eval_sop_questionlist.py`와 호환 검증

## 5. 리스크 및 대응

| 리스크 | 대응 |
|--------|------|
| qwen3 embedding 장시간 소요 | GPU 사용률 모니터링, 필요시 batch size 조정 |
| ES 다운 시 probe 불가 | ES health check 후 probe 실행 |
| 유사질문 retrieve 실패율 높음 | BM25 한계 → embedding 기반 검색으로 보완 예정 |
| content index gap | eval에 영향 없으면 후순위 처리 |
