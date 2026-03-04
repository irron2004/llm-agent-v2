#!/usr/bin/env bash
# Phase 1: 전체 VLM 파싱 실행
# Usage: bash scripts/chunk_v3/run_vlm_all.sh [--dry-run]
set -euo pipefail

DATA_ROOT="/home/llm-share/datasets/pe_agent_data/pe_preprocess_data"
OUTPUT_ROOT="data/vlm_parsed"
DRY_RUN="${1:-}"
SCRIPT="scripts/chunk_v3/vlm_parse.py"
WORKERS=2

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$OUTPUT_ROOT/logs"
MASTER_LOG="$LOG_DIR/run_all_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

# 마스터 로그 + 터미널 동시 출력
exec > >(tee -a "$MASTER_LOG") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "============================================"
log " chunk_v3 Phase 1: VLM 파싱 (workers=$WORKERS)"
log " 로그: $MASTER_LOG"
log "============================================"

# VLM 서버 체크
if ! curl -s --connect-timeout 5 http://localhost:8004/v1/models > /dev/null 2>&1; then
    log "ERROR: VLM 서버 (localhost:8004) 응답 없음"; exit 1
fi
log "VLM 서버 OK"

# 파싱 대상 정의: (이름 doc_type 경로 예상건수)
TASKS=(
    "Setup_Manual|set_up_manual|$DATA_ROOT/set_up_manual/|14"
    "Trouble_Shooting|ts|$DATA_ROOT/ts_pdfs/|57"
    "SOP|sop_pdf|$DATA_ROOT/sop_pdfs/|507"
)

TOTAL=${#TASKS[@]}
OVERALL_SUCCESS=0
OVERALL_FAIL=0

for i in "${!TASKS[@]}"; do
    IFS='|' read -r NAME DOC_TYPE INPUT_PATH EXPECTED <<< "${TASKS[$i]}"
    STEP=$((i + 1))
    STEP_LOG="$LOG_DIR/${DOC_TYPE}_${TIMESTAMP}.log"

    log ""
    log "── [$STEP/$TOTAL] $NAME (예상 ${EXPECTED}건) ──"
    log "  input:    $INPUT_PATH"
    log "  doc_type: $DOC_TYPE"
    log "  log:      $STEP_LOG"

    if [ "$DRY_RUN" = "--dry-run" ]; then
        log "  DRY-RUN: skip"
        continue
    fi

    STEP_START=$(date +%s)

    if python -u "$SCRIPT" \
        --input "$INPUT_PATH" \
        --doc-type "$DOC_TYPE" \
        --output "$OUTPUT_ROOT" \
        --workers "$WORKERS" 2>&1 | tee "$STEP_LOG"; then

        STEP_END=$(date +%s)
        STEP_ELAPSED=$(( STEP_END - STEP_START ))
        PARSED_COUNT=$(find "$OUTPUT_ROOT/$DOC_TYPE" -name "*.json" 2>/dev/null | wc -l)
        log "  완료: ${PARSED_COUNT}건 파싱, ${STEP_ELAPSED}s 소요"
        OVERALL_SUCCESS=$((OVERALL_SUCCESS + PARSED_COUNT))
    else
        STEP_END=$(date +%s)
        STEP_ELAPSED=$(( STEP_END - STEP_START ))
        log "  ERROR: $NAME 파싱 실패 (${STEP_ELAPSED}s 소요)"
        OVERALL_FAIL=$((OVERALL_FAIL + 1))
    fi
done

log ""
log "============================================"
log " Phase 1 완료 요약"
log "  총 파싱: ${OVERALL_SUCCESS}건"
log "  실패 doc_type: ${OVERALL_FAIL}개"
log "  마스터 로그: $MASTER_LOG"
log "============================================"
log ""
log "다음 단계 - 검증:"
log "  python scripts/chunk_v3/validate_vlm.py --parsed-dir $OUTPUT_ROOT --source-dir $DATA_ROOT"
