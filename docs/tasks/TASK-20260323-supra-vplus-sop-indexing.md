# TASK-20260323: SUPRA Vplus SOP 누락 문서 VLM 파싱 + ES 인덱싱

## 0. 누락 원인

VLM 파싱은 3/4~3/5에 `doc_type=sop_pdf`로 실행되어 **507개 전부 파싱 완료**되었다.
이후 `sop_pdf → sop` 리팩터링(커밋 `7a5ea59`, 3/5 08:11)에서 수동 폴더 이동 시
`data/vlm_parsed/sop_pdf/` → `data/vlm_parsed/sop/` 과정에서 **123개 파일이 누락**되었다.
원본 `sop_pdf/` 디렉토리는 현재 비어있어 복구 불가 → 재파싱 필요.

## 1. 현황 요약

| 항목 | 수량 |
|---|---|
| sop_pdfs/ 내 SUPRA Vplus PDF | **176개** |
| 매니페스트 등록 (chunk_v3_manifest.json) | **176개** (전부 등록됨) |
| VLM 파싱 완료 (data/vlm_parsed/sop/) | **53개** |
| ES 인덱싱 완료 (chunk_v3_content) | **53개** (1,139 chunks) |
| **누락 (미파싱)** | **123개** |

### 현재 ES 인덱스 현황

| 인덱스 | 문서 수 | 크기 |
|---|---|---|
| chunk_v3_content | 390,717 | 257.7 MB |
| chunk_v3_embed_bge_m3_v1 | 390,472 | 7.5 GB |
| chunk_v3_embed_jina_v5_v1 | 390,472 | 3.8 GB |
| chunk_v3_embed_qwen3_emb_4b_v1 | 390,385 | 18.5 GB |

### 누락 문서 특성
- 전부 **SUPRA Vplus** 장비의 SOP 문서
- 작업유형: ADJ(~42개), REP(~43개), CLN(~14개), MODIFY(2개) 등
- 기존 파싱된 53개는 주로 REP/SW/일부 ADJ
- 평균 페이지 수: ~20페이지/문서
- 예상 총 페이지: ~2,434페이지

---

## 2. 파이프라인 구조

```
Phase 1: VLM 파싱     Phase 2: 청킹          Phase 3: 임베딩         Phase 4: ES 적재
PDF ──→ JSON          JSON ──→ JSONL         JSONL ──→ .npy          JSONL+.npy ──→ ES
vlm_parse.py          run_chunking.py        run_embedding.py        run_ingest.py
```

### 핵심 사항
- `run_chunking.py`는 **전체 vlm_parsed/ 디렉토리**를 스캔하여 all_chunks.jsonl 생성
- `run_ingest.py content`는 chunk_id 기준 upsert(동일 _id면 덮어쓰기)이므로 **재실행 안전**
- `run_ingest.py embed`도 동일하게 upsert 방식
- `vlm_parse.py`는 이미 파싱된 파일 자동 스킵 (resume 지원)

---

## 3. 실행 절차

### 사전 조건 확인

```bash
# VLM 서버 상태 확인 (Qwen3-VL-30B)
curl -s ${VLM_BASE_URL:-http://localhost:8004}/v1/models | python3 -m json.tool

# ES 상태 확인
curl -s http://localhost:8002/_cluster/health | python3 -m json.tool

# GPU 상태 확인
nvidia-smi
```

### Phase 1: VLM 파싱 (예상 시간: 15~20시간)

누락 123개 PDF를 VLM으로 파싱하여 JSON 생성.
`vlm_parse.py`는 이미 파싱된 파일을 자동 스킵하므로, sop_pdfs 전체 디렉토리를 입력으로 넣어도 안전.

```bash
cd /home/hskim/work/llm-agent-v2

# 단일 워커 (안전)
python scripts/chunk_v3/vlm_parse.py \
    --input /home/llm-share/datasets/pe_agent_data/pe_preprocess_data/sop_pdfs/ \
    --doc-type sop \
    --output data/vlm_parsed/ \
    --no-review

# 또는 GPU 2개 병렬 (빠름, ~10시간)
python scripts/chunk_v3/vlm_parse.py \
    --input /home/llm-share/datasets/pe_agent_data/pe_preprocess_data/sop_pdfs/ \
    --doc-type sop \
    --output data/vlm_parsed/ \
    --workers 2 \
    --no-review
```

**체크포인트**: 파싱 완료 후 확인
```bash
ls data/vlm_parsed/sop/ | grep -i vplus | wc -l
# 기대값: 176
```

### Phase 2: 청킹 (예상 시간: 수 분)

전체 문서를 다시 청킹하여 all_chunks.jsonl 재생성.
기존 문서 + 신규 123개 모두 포함됨.

```bash
# 기존 all_chunks.jsonl 백업
cp data/chunks_v3/all_chunks.jsonl data/chunks_v3/all_chunks.jsonl.bak.$(date +%Y%m%d)

# 전체 청킹 (VLM + myservice + gcb)
python scripts/chunk_v3/run_chunking.py \
    --vlm-dir data/vlm_parsed \
    --output data/chunks_v3/all_chunks.jsonl \
    --manifest data/chunk_v3_manifest.json
```

**체크포인트**: SOP 청크 수 증가 확인
```bash
grep '"doc_type": "sop"' data/chunks_v3/all_chunks.jsonl | wc -l
# 기존: 13,116 → 예상: ~15,500+ (123개 × 평균 ~20 chunks)

# Vplus SOP 문서 수 확인
grep '"doc_type": "sop"' data/chunks_v3/all_chunks.jsonl | \
    grep -o '"doc_id": "[^"]*vplus[^"]*"' | sort -u | wc -l
# 기대값: 176
```

### Phase 3: 임베딩 (예상 시간: 모델당 30분~1시간)

전체 청크에 대해 3개 모델 임베딩 재생성.

```bash
# 기존 임베딩 백업
for model in bge_m3 jina_v5 qwen3_emb_4b; do
    cp data/chunks_v3/embeddings_${model}.npy data/chunks_v3/embeddings_${model}.npy.bak.$(date +%Y%m%d) 2>/dev/null
    cp data/chunks_v3/chunk_ids_${model}.jsonl data/chunks_v3/chunk_ids_${model}.jsonl.bak.$(date +%Y%m%d) 2>/dev/null
done

# 전체 임베딩 생성 (3개 모델 순차)
python scripts/chunk_v3/run_embedding.py \
    --chunks data/chunks_v3/all_chunks.jsonl \
    --models bge_m3 jina_v5 qwen3_emb_4b \
    --output-dir data/chunks_v3/ \
    --batch-size 64
```

**체크포인트**: 임베딩 shape 확인
```bash
python3 -c "
import numpy as np
for model in ['bge_m3', 'jina_v5', 'qwen3_emb_4b']:
    v = np.load(f'data/chunks_v3/embeddings_{model}.npy')
    print(f'{model}: {v.shape}')
"
```

### Phase 4: ES 적재 (예상 시간: 10~30분)

```bash
# 4a. Content 인덱스 적재 (upsert, 기존 데이터 유지)
python scripts/chunk_v3/run_ingest.py content \
    --chunks data/chunks_v3/all_chunks.jsonl

# 4b. Embed 인덱스 적재 (3개 모델)
python scripts/chunk_v3/run_ingest.py embed \
    --model bge_m3 \
    --embeddings data/chunks_v3/embeddings_bge_m3.npy \
    --chunk-ids data/chunks_v3/chunk_ids_bge_m3.jsonl \
    --chunks data/chunks_v3/all_chunks.jsonl

python scripts/chunk_v3/run_ingest.py embed \
    --model jina_v5 \
    --embeddings data/chunks_v3/embeddings_jina_v5.npy \
    --chunk-ids data/chunks_v3/chunk_ids_jina_v5.jsonl \
    --chunks data/chunks_v3/all_chunks.jsonl

python scripts/chunk_v3/run_ingest.py embed \
    --model qwen3_emb_4b \
    --embeddings data/chunks_v3/embeddings_qwen3_emb_4b.npy \
    --chunk-ids data/chunks_v3/chunk_ids_qwen3_emb_4b.jsonl \
    --chunks data/chunks_v3/all_chunks.jsonl
```

### Phase 5: 검증

```bash
# 5a. 인덱스 문서 수 확인
curl -s "http://localhost:8002/_cat/indices/chunk_v3*?v&h=index,docs.count,store.size"

# 5b. Vplus SOP 문서 수 ES 확인
curl -s "http://localhost:8002/chunk_v3_content/_search" \
    -H 'Content-Type: application/json' -d '{
  "size": 0,
  "query": {"bool": {"filter": [
    {"term": {"doc_type": "sop"}},
    {"term": {"device_name": "SUPRA_VPLUS"}}
  ]}},
  "aggs": {"unique_docs": {"cardinality": {"field": "doc_id"}}}
}'
# 기대값: unique_docs = 176

# 5c. Content ↔ Embed 동기화 검증
python scripts/chunk_v3/run_ingest.py verify --model bge_m3
python scripts/chunk_v3/run_ingest.py verify --model jina_v5
python scripts/chunk_v3/run_ingest.py verify --model qwen3_emb_4b

# 5d. 샘플 검색 테스트
curl -s "http://localhost:8002/chunk_v3_content/_search" \
    -H 'Content-Type: application/json' -d '{
  "size": 3,
  "query": {"bool": {
    "must": [{"match": {"search_text": "SUPRA Vplus heater chuck 교체"}}],
    "filter": [{"term": {"doc_type": "sop"}}]
  }},
  "_source": ["doc_id", "device_name", "chapter", "page"]
}' | python3 -m json.tool
```

---

## 4. 주의 사항

1. **VLM 서버 필수**: Phase 1은 VLM 서버(기본 localhost:8004)가 실행 중이어야 함
2. **GPU 메모리**: Phase 3 임베딩 시 GPU 메모리 필요. OOM 발생 시 `--batch-size` 줄이기
3. **디스크 공간**: 임베딩 .npy 파일이 큼 (bge_m3 ~7.5GB, qwen3 ~18GB). 기존+신규 합산 주의
4. **run_ingest.py는 upsert**: `--recreate` 플래그 없이 실행하면 기존 데이터 유지 + 신규 추가
5. **run_chunking.py는 전체 재생성**: all_chunks.jsonl을 새로 만들므로 반드시 백업 후 실행
6. **Phase 2~4는 전체 데이터 대상**: 증분이 아닌 전체 재처리 방식. 기존 데이터 포함하여 재생성

## 5. 롤백 절차

문제 발생 시:
```bash
# all_chunks.jsonl 복원
cp data/chunks_v3/all_chunks.jsonl.bak.YYYYMMDD data/chunks_v3/all_chunks.jsonl

# 임베딩 복원
for model in bge_m3 jina_v5 qwen3_emb_4b; do
    cp data/chunks_v3/embeddings_${model}.npy.bak.YYYYMMDD data/chunks_v3/embeddings_${model}.npy
    cp data/chunks_v3/chunk_ids_${model}.jsonl.bak.YYYYMMDD data/chunks_v3/chunk_ids_${model}.jsonl
done

# ES 재적재 (기존 데이터로)
python scripts/chunk_v3/run_ingest.py content --chunks data/chunks_v3/all_chunks.jsonl
# + embed 각 모델 재적재
```

## 6. 누락 문서 목록 (123개)

<details>
<summary>전체 목록 펼치기</summary>

### ADJ (~42개)
- Global SOP_SUPRA Vplus_ADJ_All_POWER TURN ON／OFF_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_EFEM_FFU CONTROLLER_EN.pdf
- Global SOP_SUPRA Vplus_ADJ_EFEM_LOAD PORT LEVELING_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_EFEM_LOAD PORT MAPPING SENSOR_SELOP8 0_EN.pdf
- Global SOP_SUPRA Vplus_ADJ_EFEM_LOAD PORT N2 PURGE_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_EFEM_ROBOT LEVELING (Yaskawa)_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_EFEM_ROBOT PENDANT (Sankyo)_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_EFEM_ROBOT PENDANT (Yaskawa)_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_EFEM_UTILITY TURN ON OFF_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_LOAD PORT CERTIFICATION_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_2ND GENERATION DUAL EPD_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_APC VALVE_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_ATM RELAY_EN.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_BARATRON GAUGE_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_CHAMBER LEVELING_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_DEVICE NET BOARD_EN.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_DOCKING_EN.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_DOOR VALVE (Presys)_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_DOOR VALVE (VAT)_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_EPD_EN.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_FCIP R3_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_FLOW SWITCH_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_GAS LINE LEAK CHECK_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_GAS SPRING_EN.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_GAS SPRING_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_HOOK LIFTER LM GUIDE GREASE_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_HOOK LIFTER PIN LEVELING_EN.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_HOOK LIFTER PIN PENDANT_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_HOOK LIFTER PIN_EN.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_LEAK CHECK_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_PCW DRAIN_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_PIN MOTOR PENDANT_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_PIRANI GAUGE_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_PUMP TURN ON OFF_EN.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_TC WAFER_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_PM_UNDOCKING_EN.pdf
- Global SOP_SUPRA Vplus_ADJ_SUB UNIT_FLOW SWITCH_EN.pdf
- Global SOP_SUPRA Vplus_ADJ_SUB UNIT_GAS TURN ONOFF_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_SUB UNIT_PCW TURN ON OFF_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_SUB UNIT_PRESSURE SWITCH_EN.pdf
- Global SOP_SUPRA Vplus_ADJ_SUB UNIT_TEMP CONTORLLER_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_SUB UNIT_VAC TURN ON OFF_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_SUB UNIT_VACUUM SWITCH_EN.pdf
- Global SOP_SUPRA Vplus_ADJ_SUB_MANOMETER_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_TM_CTC BUZZ USB CONNECTION_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_TM_FFU CONTROLLER_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_TM_IB FLOW_EN.pdf
- Global SOP_SUPRA Vplus_ADJ_TM_ROBOT LEVELING_EN.pdf
- Global SOP_SUPRA Vplus_ADJ_TM_ROBOT PENDANT_ENG.pdf
- Global SOP_SUPRA Vplus_ADJ_TM_ROBOT TEACHING_EN.pdf
- Global SOP_SUPRA Vplus_ADJ_TM_ROBOT TEACHING_RND_EN.pdf

### CLN (~14개)
- Global SOP_SUPRA Vplus_CLN_EFEM_LOAD PORT_ENG.pdf
- Global SOP_SUPRA Vplus_CLN_PM_APC_VALVE_ENG.pdf
- Global SOP_SUPRA Vplus_CLN_PM_DOOR VALVE_ENG.pdf
- Global SOP_SUPRA Vplus_CLN_PM_EXHAUST RING_EN.pdf
- Global SOP_SUPRA Vplus_CLN_PM_EXHAUST RING_ENG.pdf
- Global SOP_SUPRA Vplus_CLN_PM_FAST VAC VALVE_ENG.pdf
- Global SOP_SUPRA Vplus_CLN_PM_FOCUS ADAPTOR_EN.pdf
- Global SOP_SUPRA Vplus_CLN_PM_FOCUS ADAPTOR_ENG.pdf
- Global SOP_SUPRA Vplus_CLN_PM_HEATER CHUCK_ENG.pdf
- Global SOP_SUPRA Vplus_CLN_PM_ISOLATION VALVE_EN.pdf
- Global SOP_SUPRA Vplus_CLN_PM_SLOW VACUUM VALVE_ENG.pdf
- Global SOP_SUPRA Vplus_CLN_PM_TOP LID_ENG.pdf
- Global SOP_SUPRA Vplus_CLN_TM_BUFFER STAGE_ENG.pdf
- Global SOP_SUPRA Vplus_CLN_TM_COOLING STAGE_ENG.pdf

### REP (~43개)
- Global SOP_ SUPRA Vplus_REP_EFEM_BCR READER_KEYENCE_ENG.pdf
- Global SOP_SUPRA Vplus_REP_EFEM ROBOT_ENG.pdf
- Global SOP_SUPRA Vplus_REP_EFEM_8-PORT PANEL_EN.pdf
- Global SOP_SUPRA Vplus_REP_EFEM_CONTROLLER_ENG.pdf
- Global SOP_SUPRA Vplus_REP_EFEM_DEVICE NET BOARD_ENG.pdf
- Global SOP_SUPRA Vplus_REP_EFEM_EMO SWITCH 0_EN.pdf
- Global SOP_SUPRA Vplus_REP_EFEM_FFU CONTROLLER_ENG.pdf
- Global SOP_SUPRA Vplus_REP_EFEM_FFU FILTER_ENG.pdf
- Global SOP_SUPRA Vplus_REP_EFEM_LAMP FLUORESCENT_EN.pdf
- Global SOP_SUPRA Vplus_REP_EFEM_LIGHT CURTAIN_ENG.pdf
- Global SOP_SUPRA Vplus_REP_EFEM_LOAD PORT_ENG.pdf
- Global SOP_SUPRA Vplus_REP_EFEM_LP Z-AXIS MODULE_EN.pdf
- Global SOP_SUPRA Vplus_REP_EFEM_PRESSURE SWITCH_ENG.pdf
- Global SOP_SUPRA Vplus_REP_EFEM_RFID ANTENNA_EN.pdf
- Global SOP_SUPRA Vplus_REP_EFEM_RFID READER_ENG.pdf
- Global SOP_SUPRA Vplus_REP_EFEM_ROBOT CONTROLLER_ENG.pdf
- Global SOP_SUPRA Vplus_REP_EFEM_ROBOT END EFFECTOR PAD_ENG.pdf
- Global SOP_SUPRA Vplus_REP_EFEM_ROBOT_SR8240_EN.pdf
- Global SOP_SUPRA Vplus_REP_EFEM_SENSOR BOARD SFEM_EN.pdf
- Global SOP_SUPRA Vplus_REP_EFEM_SIGNAL TOWER 0_EN.pdf
- Global SOP_SUPRA Vplus_REP_PM_APC VALVE_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_ATM RELAY_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_BAFFLE_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_BARATRON GAUGE_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_BELLOWS_EN.pdf
- Global SOP_SUPRA Vplus_REP_PM_CERAMIC PLATE_EN.pdf
- Global SOP_SUPRA Vplus_REP_PM_CHAMBER HINGE_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_CYLINDER SENSOR_EN.pdf
- Global SOP_SUPRA Vplus_REP_PM_DEVICE NET BOARD_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_DOOR VALVE_EN.pdf
- Global SOP_SUPRA Vplus_REP_PM_DOOR VALVE_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_DSM KIT_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_EPD_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_EXHAUST RING_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_FAST VACUUM VALVE_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_FCIP R3_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_FLOW SWITCH_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_FOCUS ADAPTOR_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_GAS FEED THROUGH_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_GAS SPRING_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_HEATER CHUCK CONNECTOR_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_HEATER CHUCK_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_HEATING JACKET_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_HOOK LIFT PIN_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_HOOK LIFTER LM GUIDE_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_HOOK LIFTER SERVO MOTOR CONTROLLER_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_HOOK LIFTER SERVO MOTOR_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_ISOLATION VALVE_EN.pdf
- Global SOP_SUPRA Vplus_REP_PM_PCW MANUAL VALVE_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_PIRANI GAUGE_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_PNEUMATIC VALVE_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_SLOW VAC VALVE_ENG.pdf
- Global SOP_SUPRA Vplus_REP_PM_SOURCE BOX INTERFACE BOARD_ENG.pdf

### MODIFY (2개)
- Global SOP_SUPRA Vplus_MODIFY_ARM LEVELING SENSOR_EN.pdf
- Global SOP_SUPRA Vplus_MODIFY_SU_LOTO COVER_EN.pdf

### 중복 가능 (파일명 변형으로 이미 파싱된 것과 겹칠 수 있음, 3개)
- Global SOP_SUPRA_Vplus_ADJ_TM_Robot(RND)_Encoder Reset_EN.pdf
- Global_SOP_SUPRA_Vplus_REP_EFEM_FFU_FILTER_LV32(SAFAS)_EN.pdf
- global sop_supra vplus_adj_all_power turn on／off.pdf

</details>
