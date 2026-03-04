# SOP 질문리스트 정답문서/정답페이지 적중률 보고서

이 문서는 아래 원본 보고서의 동일 내용을 참조하기 위한 alias 문서입니다.

- 원본: [2026-03-01_sop_golden_set_accuracy_report.md](2026-03-01_sop_golden_set_accuracy_report.md)
- 행별 상세 데이터: [evidence/2026-03-01_sop_questionlist_eval_retrieval_rows.csv](evidence/2026-03-01_sop_questionlist_eval_retrieval_rows.csv)

## 핵심 지표 (페이지 포함 기준 우선)

| 지표 | 결과 | 비율 |
|---|---|---|
| page-hit@1 | 53/79 | 67.1% |
| page-hit@3 | 57/79 | 72.2% |
| page-hit@5 | 58/79 | 73.4% |
| page-hit@10 | 61/79 | 77.2% |
| page-hit@20 | 61/79 | 77.2% |
| --- | --- | --- |
| hit@5 | 76/79 | 96.2% |
| hit@10 | 79/79 | 100.0% |
| rank=1 | 71/79 | 89.9% |
| 정답 페이지 포함(=page-hit@10) | 61/79 | 77.2% |

## 주요 발견

1. **문서 정확도는 충분하지만 페이지 정확도는 별도 개선 필요** — 문서 기준 top-10 100%, page-hit@10은 77.2%
2. **운영 서버 인덱스 미스매치** — 현재 서버가 합성 인덱스(`rag_synth`) 사용 중, `rag_chunks_dev` 전환 필요
3. **페이지 미스 18건** — 문서 목차(page 1) chunk가 BM25 최상위 → 동일 문서 다중 페이지 집계로 해결 가능
4. **Top-5 밖 3건** — PRISM SOURCE 유사 문서 4종 경합, Pirani/Pressure Gauge 통합문서 경합

## 개선 우선순위

| 순위 | 조치 | 기대 효과 |
|---|---|---|
| P0 | 서버 인덱스 전환 (`rag_synth` → `rag_chunks`) | 실제 SOP 데이터 접근 |
| P0 | 동일 문서 다중 페이지 집계 | 페이지 포함율 77% → ~95% |
| P1 | 목차 chunk 가중치 조정 + Reranker | Top-5 밖 3건 해결 |
| P1 | `mq_mode=fallback` 배포 | 반복 안정성 확보 |

상세 분석(Top-5 밖 케이스별 원인, 페이지 미포함 18건 목록, 개선 로드맵)은 원본 보고서를 확인해 주세요.
