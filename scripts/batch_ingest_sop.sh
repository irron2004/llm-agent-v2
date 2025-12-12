#!/bin/bash
# Batch ingest SOP PDFs to Elasticsearch
# Usage: nohup bash scripts/batch_ingest_sop.sh > logs/batch_ingest.log 2>&1 &

set -e

PDF_DIR="/home/llm-share/datasets/pe_agent_data/pe_preprocess_data/sop_pdfs"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== Batch SOP Ingest Started at $(date) ==="
echo "PDF Directory: $PDF_DIR"
echo "Working Directory: $(pwd)"
echo ""

count=0
total=$(find "$PDF_DIR" -maxdepth 1 -type f -name '*.pdf' | wc -l)
echo "Total PDFs to process: $total"
echo ""

find "$PDF_DIR" -maxdepth 1 -type f -name '*.pdf' -print0 | \
while IFS= read -r -d '' pdf; do
  count=$((count + 1))
  doc_id=$(basename "${pdf%.pdf}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr -cs 'a-z0-9_-' '_')

  echo "[$count/$total] Ingesting: $(basename "$pdf")"
  echo "  doc_id: $doc_id"
  echo "  started: $(date '+%Y-%m-%d %H:%M:%S')"

  if python scripts/vlm_es_ingest.py \
    --pdf "$pdf" \
    --doc-id "$doc_id" \
    --doc-type sop \
    --lang ko \
    --tenant-id tenant1 \
    --project-id proj1 \
    --tags sop batch \
    --refresh; then
    echo "  status: SUCCESS"
  else
    echo "  status: FAILED (exit code: $?)"
  fi
  echo ""
done

echo "=== Batch SOP Ingest Completed at $(date) ==="
