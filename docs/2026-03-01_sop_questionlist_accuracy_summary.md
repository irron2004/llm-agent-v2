# SOP 질문리스트 정답률 요약

- 상세 보고서: [2026-03-01_sop_golden_set_accuracy_report.md](/home/hskim/work/llm-agent-v2/docs/2026-03-01_sop_golden_set_accuracy_report.md)

요약 지표:

- hit@5: `76/79 = 96.2%`
- hit@10: `79/79 = 100.0%`
- hit@20: `79/79 = 100.0%`
- rank=1: `71/79 = 89.9%`
- 정답 페이지 포함: `61/79 = 77.2%`

핵심 이슈:

1. 서버 인덱스가 합성(`rag_synth_synth_v2`)으로 잡혀 있으면 실제 SOP 성능이 반영되지 않음
2. 정답 페이지 미스는 다수 케이스에서 목차/표지 chunk가 상위로 노출되는 구조적 문제

권장 우선순위:

1. 인덱스 설정을 운영 SOP 인덱스로 고정
2. 동일 문서 다중 페이지 집계 표시
3. 필요 시 reranker/MQ fallback 추가
