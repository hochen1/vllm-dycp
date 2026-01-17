set -x

export NCCL_DEBUG=WARN
export MODEL_PATH=/weights/qwen/Qwen3-30B-A3B-Instruct-2507/
export VLLM_USE_V1=1
export COMMON_ARGS="
    --trust-remote-code
    --served-model-name auto
    --model-loader-extra-config {\"enable_multithread_load\":true,\"num_threads\":8}
    --disable-log-requests
"
export VLLM_VERSION=0.13.0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=380
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# export CUDA_LAUNCH_BLOCKING=1

ulimit -n 65536
vllm serve ${MODEL_PATH} \
    --port 8400 \
    $COMMON_ARGS \
    --distributed-executor-backend mp \
    --max-model-len 163840 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.80 \
    --no-enable-prefix-caching \
    --tensor-parallel-size 4 \
    --block-size 64 \
    --cp-kv-cache-interleave-size 64 \
    --no-enforce-eager \
    --max-num-seqs 500 \
    --enable-expert-parallel \
    --compilation-config '{"cudagraph_capture_sizes":[6, 100, 200, 300], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{
        "kv_connector": "ExampleConnector",
    "kv_connector_module_path": "vllm.distributed.kv_transfer.kv_connector.v1.example_connector",
        "kv_role": "kv_consumer",
        "kv_parallel_size": 2,
        "kv_port": "20002",
        "engine_id": "decode-'${NODE_RANK}'",
        "kv_rank": 1,
        "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 1,
                    "tp_size": 16
             },
             "decode": {
                    "dp_size": 1,
                    "tp_size": 4
             }
        }
    }'
