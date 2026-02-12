#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${1:-$ROOT_DIR/data/logs/docker}"

mkdir -p "$LOG_DIR"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/compose_${TIMESTAMP}.log"

echo "[log] writing docker compose logs to: $LOG_FILE"
cd "$ROOT_DIR"
docker compose logs --timestamps -f | tee -a "$LOG_FILE"
