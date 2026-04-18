#!/bin/bash
# Local PD Separation: Launch vLLM instance + Proxy
#
# Usage:
#   1. Start vLLM instance:  bash dycp/examples/start_local_pd.sh server
#   2. Start proxy:          bash dycp/examples/start_local_pd.sh proxy
#   3. Run benchmark:        bash dycp/examples/start_local_pd.sh bench
set -x

export MODEL_PATH=${MODEL_PATH:-/tmp/models/DeepSeek-V2-Lite}
export PYTHONPATH=/tmp/vllm:$PYTHONPATH
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
export VLLM_VERSION=0.13.0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=380
export VLLM_ATTENTION_BACKEND=FLASHMLA
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_FORCE_LOAD_BLANCE=1

# Ports
VLLM_PORT=${VLLM_PORT:-8400}
PROXY_PORT=${PROXY_PORT:-9000}

# KV cache storage (use tmpfs for better I/O)
KV_STORAGE_PATH=${KV_STORAGE_PATH:-/tmp/kv_cache}
mkdir -p $KV_STORAGE_PATH

export COMMON_ARGS="
    --trust-remote-code
    --served-model-name auto
    --model-loader-extra-config {\"enable_multithread_load\":true,\"num_threads\":8}
    --disable-log-requests
"

# DP/CP configuration
DP_SIZE=${DP_SIZE:-8}
DP_PER_DOMAIN=${DP_PER_DOMAIN:-8}
NUM_CP_SEQS=${NUM_CP_SEQS:-4}

case "${1:-server}" in
  server)
    echo "=== Starting vLLM instance with LocalPDConnector ==="
    mkdir -p /tmp/logs
    source /tmp/vllm/.venv/bin/activate
    vllm serve ${MODEL_PATH} \
        --port ${VLLM_PORT} \
        $COMMON_ARGS \
        --hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":8.0,"original_max_position_embeddings":163840}}' \
        --distributed-executor-backend dmp \
        --max-model-len 1048576 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.7 \
        --no-enable-prefix-caching \
        --data-parallel-size ${DP_SIZE} \
        --tensor-parallel-size 1 \
        --dp-per-domain ${DP_PER_DOMAIN} \
        --block-size 64 \
        --cp-kv-cache-interleave-size 64 \
        --no-enforce-eager \
        --compilation-config '{"cudagraph_capture_sizes":[4, 8, 16, 24, 32, 64], "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes_for_cp": 4}' \
        --num-cp-seqs ${NUM_CP_SEQS} \
        --enable-expert-parallel \
        --kv-transfer-config "{\"kv_connector\":\"LocalPDConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"storage_path\":\"${KV_STORAGE_PATH}\"}}" \
        2>&1 | tee /tmp/logs/local_pd_vllm.log
    ;;

  proxy)
    echo "=== Starting Local PD Proxy ==="
    source /tmp/vllm/.venv/bin/activate
    python dycp/proxy/local_pd_proxy.py \
        --vllm-url http://localhost:${VLLM_PORT} \
        --port ${PROXY_PORT} \
        --host 0.0.0.0
    ;;

  bench)
    echo "=== Running Benchmark ==="
    MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
    CONCURRENCY=$((MAX_NUM_SEQS * DP_SIZE))
    source /tmp/vllm/.venv/bin/activate
    python benchmarks/benchmark_serving.py \
        --model ${MODEL_PATH} \
        --base-url http://localhost:${PROXY_PORT} \
        --num-prompts 200 \
        --max-concurrency ${CONCURRENCY} \
        --request-rate 2 \
        --backend openai-chat \
        2>&1 | tee /tmp/logs/local_pd_bench.log
    ;;

  *)
    echo "Usage: $0 {server|proxy|bench}"
    exit 1
    ;;
esac
