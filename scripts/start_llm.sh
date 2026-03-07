#!/bin/bash
# =============================================================================
# LLM 서버 시작 스크립트
# MODEL_LLM 환경변수만 변경하면 나머지는 자동으로 설정됩니다.
# Usage: ./scripts/start_llm.sh [--docker]
#   --docker: docker compose로 실행 (지원 모델만)
#   (기본): 로컬 vllm으로 실행
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# .env 로드
set -a
source "$PROJECT_DIR/.env"
set +a

# 모델별 설정 자동 결정
resolve_model_config() {
    local model="$MODEL_LLM"
    case "$model" in
        *GLM-4.7-Flash*|*glm-4.7*)
            TP_SIZE=1
            CUDA_DEVICES="1"
            MAX_MODEL_LEN=8192
            DOCKER_SUPPORT=false
            ;;
        *deepseek-r1-distill-llama-70b*|*DeepSeek*70*)
            TP_SIZE=2
            CUDA_DEVICES="0,1"
            MAX_MODEL_LEN=8192
            DOCKER_SUPPORT=true
            ;;
        *Qwen3-32B*|*qwen3-32b*)
            TP_SIZE=1
            CUDA_DEVICES="1"
            MAX_MODEL_LEN=8192
            DOCKER_SUPPORT=true
            ;;
        *Qwen2.5-72B*|*qwen2.5-72b*)
            TP_SIZE=2
            CUDA_DEVICES="0,1"
            MAX_MODEL_LEN=8192
            DOCKER_SUPPORT=true
            ;;
        *gpt-oss-20b*)
            TP_SIZE=2
            CUDA_DEVICES="0,1"
            MAX_MODEL_LEN=32768
            DOCKER_SUPPORT=true
            ;;
        *)
            echo "Unknown model: $model"
            echo "Add model config to scripts/start_llm.sh"
            exit 1
            ;;
    esac
}

resolve_model_config

echo "=== LLM Server Config ==="
echo "  Model:       $MODEL_LLM"
echo "  TP Size:     $TP_SIZE"
echo "  GPU:         $CUDA_DEVICES"
echo "  Max Len:     $MAX_MODEL_LEN"
echo "  Port:        $LLM_PORT"
echo "  Docker:      $DOCKER_SUPPORT"
echo "========================="

# 기존 서버 정리
stop_existing() {
    # docker vllm 중지
    docker compose --profile with-llm down vllm 2>/dev/null || true
    # 로컬 vllm 중지
    pkill -f "vllm.entrypoints.openai.api_server.*--port ${LLM_PORT}" 2>/dev/null || true
    sleep 2
}

USE_DOCKER=false
if [[ "${1:-}" == "--docker" ]]; then
    USE_DOCKER=true
fi

# Docker 모드
if [[ "$USE_DOCKER" == true ]]; then
    if [[ "$DOCKER_SUPPORT" == false ]]; then
        echo "WARNING: $MODEL_LLM is not supported in docker vllm image."
        echo "Falling back to local vllm..."
        USE_DOCKER=false
    fi
fi

stop_existing

if [[ "$USE_DOCKER" == true ]]; then
    echo "Starting via docker compose..."
    # docker compose에 환경변수 전달
    VLLM_TP_SIZE=$TP_SIZE \
    VLLM_CUDA_DEVICES=$CUDA_DEVICES \
    VLLM_MAX_MODEL_LEN=$MAX_MODEL_LEN \
    docker compose --profile with-llm up vllm -d

    echo "Waiting for model to load..."
    for i in $(seq 1 60); do
        if curl -s "http://localhost:${LLM_PORT}/v1/models" >/dev/null 2>&1; then
            echo "Server ready on port $LLM_PORT"
            curl -s "http://localhost:${LLM_PORT}/v1/models" | python3 -m json.tool 2>/dev/null
            exit 0
        fi
        sleep 5
    done
    echo "Timeout waiting for server. Check: docker logs rag-vllm"
    exit 1
else
    echo "Starting via local vllm..."
    HF_HOME="$PROJECT_DIR/data/hf_cache" \
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_LLM" \
        --download-dir "$PROJECT_DIR/data/hf_cache" \
        --tensor-parallel-size "$TP_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --port "$LLM_PORT" \
        --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL:-0.9}" \
        --trust-remote-code \
        &

    echo "Waiting for model to load..."
    for i in $(seq 1 60); do
        if curl -s "http://localhost:${LLM_PORT}/v1/models" >/dev/null 2>&1; then
            echo "Server ready on port $LLM_PORT"
            curl -s "http://localhost:${LLM_PORT}/v1/models" | python3 -m json.tool 2>/dev/null
            exit 0
        fi
        sleep 5
    done
    echo "Timeout waiting for server. Check process logs."
    exit 1
fi
