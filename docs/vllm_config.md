**Purpose**
This document explains how to change vLLM settings when the service is deployed with Docker Compose.

**Where To Change**
Docker Compose reads vLLM options from two files.
- `docker-compose.yml`
- `.env`

**How It Works**
The `vllm` service in `docker-compose.yml` defines the runtime flags. The actual values are injected from `.env`.

**Steps**
1. Edit `.env` and set the desired values.
2. Recreate the vLLM container so the new values are applied.
3. Verify the new settings in the vLLM logs.

**Example**
```
VLLM_MAX_MODEL_LEN=65536
VLLM_GPU_MEM_UTIL=0.7
VLLM_TP_SIZE=2
MODEL_LLM=openai/gpt-oss-20b
```

**Apply Changes**
```
docker compose up -d --force-recreate vllm
```

**Verify**
```
docker logs rag-vllm | grep -i "max_model_len"
```

**Key Parameters**
- `VLLM_MAX_MODEL_LEN`: Maximum context length.
- `VLLM_GPU_MEM_UTIL`: GPU memory utilization for KV cache.
- `VLLM_TP_SIZE`: Tensor parallel size.
- `MODEL_LLM` or `VLLM_MODEL_NAME`: Model name.

**Notes**
- `.env` currently defines `VLLM_MAX_MODEL_LEN` more than once. Keep a single value to avoid confusion.
- Changing `VLLM_MAX_MODEL_LEN` may affect latency and throughput.
