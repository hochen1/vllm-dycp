set -x
rm -rf /root/.cache/vllm/torch_compile_cache/
export NCCL_DEBUG=WARN
# export MODEL_PATH=/input/model_weights/Qwen3-30B-A3B
export MODEL_PATH=/input/model_weights/DeepSeek-V2-Lite
export VLLM_USE_V1=1
export COMMON_ARGS="
    --trust-remote-code
    --served-model-name auto
    --model-loader-extra-config {\"enable_multithread_load\":true,\"num_threads\":8}
"
export VLLM_VERSION=0.13.0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=380
export VLLM_ATTENTION_BACKEND=FLASHMLA
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_TORCH_PROFILER_DIR=${VLLM_TORCH_PROFILER_DIR:-"./profiles"}
rm -rf $VLLM_TORCH_PROFILER_DIR
mkdir -p $VLLM_TORCH_PROFILER_DIR
export VLLM_TORCH_PROFILER_WITH_STACK=0
export VLLM_USE_FORCE_LOAD_BLANCE=1
ulimit -n 65536
vllm serve ${MODEL_PATH} \
    --port 8400 \
    $COMMON_ARGS \
    --max-num-batched-tokens 131072 \
    --distributed-executor-backend mp \
    --hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":8.0,"original_max_position_embeddings":163840}}' \
    --max-model-len 163840 \
    --gpu-memory-utilization 0.6 \
    --no-enable-prefix-caching \
    --data-parallel-size 8 \
    --tensor-parallel-size 1 \
    --block-size 64 \
    --cp-kv-cache-interleave-size 64 \
    --enforce-eager \
    --max-num-seqs 1024 \
    --enable-expert-parallel \
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
                    "tp_size": 8
             }
        }
    }' &> v2lite_${NODE_RANK}.log &