#!/bin/bash
# Local PD Separation with DyCP enabled
# Usage:
#   1. Start server:  bash scripts/start_vllm_pd_dycp.sh server
#   2. Start proxy:   bash scripts/start_vllm_pd_dycp.sh proxy
#   3. Run benchmark: bash scripts/start_vllm_pd_dycp.sh bench
#   4. Quick test:    bash scripts/start_vllm_pd_dycp.sh test
set -x

export MODEL_PATH=${MODEL_PATH:-/tmp/models/DeepSeek-V2-Lite}
export PYTHONPATH=/tmp/vllm:$PYTHONPATH
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
export VLLM_VERSION=0.13.0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=380
export VLLM_ATTENTION_BACKEND=FLASHMLA
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# VLLM_USE_FORCE_LOAD_BALANCE=1 causes model output degradation on L20Y GPUs
# export VLLM_USE_FORCE_LOAD_BALANCE=1

# KV cache storage for LocalPDConnector
KV_STORAGE_PATH=${KV_STORAGE_PATH:-/tmp/local_pd_kv}
mkdir -p $KV_STORAGE_PATH

# Ports
VLLM_PORT=${VLLM_PORT:-8400}
PROXY_PORT=${PROXY_PORT:-9000}

# DP/CP configuration
DP_SIZE=${DP_SIZE:-8}
DP_PER_DOMAIN=${DP_PER_DOMAIN:-8}
NUM_CP_SEQS=${NUM_CP_SEQS:-2}

# DyCP thresholds: <4K→CP=1, 4K-16K→CP=2, 16K-32K→CP=4, >=32K→CP=8
CP_SIZE_THRESHOLDS=${CP_SIZE_THRESHOLDS:-'[(4096, 1), (16384, 4), (32768, 8)]'}

# Long request threshold for proxy PD routing
export VLLM_LONG_REQUEST_THRESHOLD=${VLLM_LONG_REQUEST_THRESHOLD:-4096}

export COMMON_ARGS="
    --trust-remote-code
    --served-model-name auto
    --model-loader-extra-config {\"enable_multithread_load\":true,\"num_threads\":8}
    --disable-log-requests
"

case "${1:-server}" in
  server)
    echo "=== Starting vLLM with LocalPDConnector + DyCP ==="
    mkdir -p /tmp/logs
    source /tmp/vllm/.venv/bin/activate
    vllm serve ${MODEL_PATH} \
        --port ${VLLM_PORT} \
        $COMMON_ARGS \
        --hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":8.0,"original_max_position_embeddings":163840}}' \
        --distributed-executor-backend dmp \
        --max-model-len 1048576 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.70 \
        --no-enable-prefix-caching \
        --data-parallel-size ${DP_SIZE} \
        --tensor-parallel-size 1 \
        --dp-per-domain ${DP_PER_DOMAIN} \
        --block-size 64 \
        --cp-kv-cache-interleave-size 64 \
        --no-enforce-eager \
        --compilation-config '{"cudagraph_capture_sizes":[2,4,8,16,32,64,128], "cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes_for_cp":2}' \
        --num-cp-seqs ${NUM_CP_SEQS} \
        --enable-expert-parallel \
        --cp-size-thresholds "${CP_SIZE_THRESHOLDS}" \
        --kv-transfer-config "{\"kv_connector\":\"LocalPDConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"shared_storage_path\":\"${KV_STORAGE_PATH}\"}}" \
        2>&1 | tee /tmp/logs/pd_dycp_vllm.log
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
    echo "=== Running PD Benchmark ==="
    RESULT_DIR=${RESULT_DIR:-/tmp/chenao/bench-results/pd-dycp}
    mkdir -p ${RESULT_DIR}
    source /tmp/vllm/.venv/bin/activate
    vllm bench serve \
        --backend openai-chat \
        --dataset-name random \
        --model ${MODEL_PATH} \
        --trust-remote-code \
        --served-model-name auto \
        --random-input-len ${INPUT_LEN:-8192} \
        --random-output-len ${OUTPUT_LEN:-1024} \
        --num-prompts ${NUM_PROMPTS:-32} \
        --max-concurrency ${MAX_CONCURRENCY:-4} \
        --request-rate ${REQUEST_RATE:-2} \
        --ignore-eos \
        --metric-percentiles "50,90,99" \
        --host localhost \
        --port ${PROXY_PORT} \
        --save-result \
        --result-dir ${RESULT_DIR} \
        --endpoint /v1/chat/completions \
        --temperature 0.6 \
        --save-detailed
    ;;

  test)
    echo "=== Quick PD Test ==="
    source /tmp/vllm/.venv/bin/activate
    # Test short request (CP=1, should go direct)
    echo "--- Short request (CP=1, direct forward) ---"
    curl -s http://localhost:${PROXY_PORT}/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"auto","messages":[{"role":"user","content":"Hello, what is 2+2?"}],"max_tokens":20,"temperature":0.6}' | python3 -m json.tool 2>/dev/null || echo "Short request test FAILED"

    # Test long request (CP>1, should go through PD flow)
    echo "--- Long request (CP>1, PD flow) ---"
    python3 -c "
import requests, json
# Generate a long prompt (~5K tokens)
prompt = 'Please explain the following topic in detail: ' + 'The quick brown fox jumps over the lazy dog. ' * 500
resp = requests.post(
    'http://localhost:${PROXY_PORT}/v1/chat/completions',
    json={'model': 'auto', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 50, 'temperature': 0.6},
    timeout=120
)
print(f'Status: {resp.status_code}')
if resp.status_code == 200:
    data = resp.json()
    content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
    print(f'Response: {content[:200]}...')
    print('Long request test PASSED')
else:
    print(f'Error: {resp.text[:500]}')
    print('Long request test FAILED')
"
    ;;

  *)
    echo "Usage: $0 {server|proxy|bench|test}"
    exit 1
    ;;
esac