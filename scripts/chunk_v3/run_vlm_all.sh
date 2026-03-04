#!/usr/bin/env bash
# Phase 1: 전체 VLM 파싱 실행 (GPU 2개 병렬)
# Usage: bash scripts/chunk_v3/run_vlm_all.sh [--dry-run]
set -euo pipefail

DATA_ROOT="/home/llm-share/datasets/pe_agent_data/pe_preprocess_data"
OUTPUT_ROOT="data/vlm_parsed"
DRY_RUN="${1:-}"
SCRIPT="scripts/chunk_v3/vlm_parse.py"
WORKERS=2

echo "============================================"
echo " chunk_v3 Phase 1: VLM 파싱 (workers=$WORKERS)"
echo "============================================"

if ! curl -s --connect-timeout 5 http://localhost:8004/v1/models > /dev/null 2>&1; then
    echo "ERROR: VLM 서버 (localhost:8004) 응답 없음"; exit 1
fi
echo "VLM 서버 OK"

# 1) Setup Manual (14건)
echo -e "\n── [1/3] Setup Manual (14건) ──"
if [ "$DRY_RUN" = "--dry-run" ]; then echo "  DRY-RUN: skip"
else python -u $SCRIPT --input "$DATA_ROOT/set_up_manual/" --doc-type set_up_manual --output "$OUTPUT_ROOT" --workers $WORKERS 2>&1 | tee "$OUTPUT_ROOT/log_set_up_manual.txt"
fi

# 2) Trouble Shooting (79건)
echo -e "\n── [2/3] Trouble Shooting (79건) ──"
if [ "$DRY_RUN" = "--dry-run" ]; then echo "  DRY-RUN: skip"
else python -u $SCRIPT --input "$DATA_ROOT/ts_pdfs/" --doc-type ts --output "$OUTPUT_ROOT" --workers $WORKERS 2>&1 | tee "$OUTPUT_ROOT/log_ts.txt"
fi

# 3) SOP (507건)
echo -e "\n── [3/3] SOP (507건) ──"
if [ "$DRY_RUN" = "--dry-run" ]; then echo "  DRY-RUN: skip"
else python -u $SCRIPT --input "$DATA_ROOT/sop_pdfs/" --doc-type sop_pdf --output "$OUTPUT_ROOT" --workers $WORKERS 2>&1 | tee "$OUTPUT_ROOT/log_sop_pdf.txt"
fi

echo -e "\n============================================"
echo " Phase 1 완료! 검증:"
echo "   python scripts/chunk_v3/validate_vlm.py --parsed-dir $OUTPUT_ROOT --source-dir $DATA_ROOT"
echo "============================================"
