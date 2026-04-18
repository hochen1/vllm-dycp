#!/bin/bash
# Low threshold for testing long request CP path
# Requests with >100 tokens will use full CP (all dp_per_domain ranks)
set -x

export MODEL_PATH=/tmp/models/DeepSeek-V2-Lite
export PYTHONPATH=/tmp/vllm:$PYTHONPATH
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
export VLLM_VERSION=0.13.0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=380
export VLLM_ATTENTION_BACKEND=FLASHMLA
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_FORCE_LOAD_BLANCE=1
export VLLM_LONG_REQUEST_THRESHOLD=100

mkdir -p /tmp/local_pd_kv /tmp/logs

source /tmp/vllm/.venv/bin/activate

vllm serve ${MODEL_PATH} \
    --port 8400 \
    --trust-remote-code \
    --served-model-name auto \
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":8}' \
    --disable-log-requests \
    --hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":8.0,"original_max_position_embeddings":163840}}' \
    --distributed-executor-backend dmp \
    --max-model-len 1048576 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.7 \
    --no-enable-prefix-caching \
    --data-parallel-size 8 \
    --tensor-parallel-size 1 \
    --dp-per-domain 8 \
    --block-size 64 \
    --cp-kv-cache-interleave-size 64 \
    --no-enforce-eager \
    --compilation-config '{"cudagraph_capture_sizes":[4, 8, 16, 24, 32, 64], "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes_for_cp": 4}' \
    --num-cp-seqs 4 \
    --enable-expert-parallel \
    --kv-transfer-config '{"kv_connector":"LocalPDConnector","kv_role":"kv_both","kv_connector_extra_config":{"shared_storage_path":"/tmp/local_pd_kv"}}' \
    2>&1 | tee /tmp/logs/local_pd_vllm.log
