set -x
rm -rf /root/.cache/vllm/torch_compile_cache/
export PYTHONPATH=/input/chenxiao.cx/codebase/vllm-dycp:$PYTHONPATH
export NCCL_DEBUG=WARN
export MODEL_PATH=/input/model_weights/Qwen3-30B-A3B
# export MODEL_PATH=/input/model_weights/DeepSeek-V2-Lite
export VLLM_USE_V1=1
export COMMON_ARGS="
    --trust-remote-code
    --served-model-name auto
    --model-loader-extra-config {\"enable_multithread_load\":true,\"num_threads\":8}
    --disable-log-requests
"
# export CUDA_LAUNCH_BLOCKING=1
export VLLM_VERSION=0.13.0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=380
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_TORCH_PROFILER_DIR=${VLLM_TORCH_PROFILER_DIR:-"./profiles"}
rm -rf $VLLM_TORCH_PROFILER_DIR
mkdir -p $VLLM_TORCH_PROFILER_DIR
export VLLM_TORCH_PROFILER_WITH_STACK=0
export VLLM_USE_FORCE_LOAD_BLANCE=1
ulimit -n 65536
vllm serve ${MODEL_PATH} \
    --port 8400 \
    --async-scheduling \
    $COMMON_ARGS \
    --distributed-executor-backend dmp \
    --hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":8.0,"original_max_position_embeddings":40960}}' \
    --max-model-len 163840 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.8 \
    --no-enable-prefix-caching \
    --data-parallel-size 2 \
    --tensor-parallel-size 4 \
    --dp-per-domain 2 \
    --block-size 64 \
    --cp-kv-cache-interleave-size 64 \
    --enforce-eager \
    --max-num-seqs 320 \
    --num-cp-seqs 4 \
    --enable-expert-parallel \
    --compilation-config '{"cudagraph_capture_sizes":[4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 300], "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes_for_cp": 4}' \
    --kv-transfer-config \
    '{
        "kv_connector": "CrossDPExampleConnector",
    "kv_connector_module_path": "vllm.distributed.kv_transfer.kv_connector.v1.cross_dp_example_connector",
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
                    "dp_size": 2,
                    "tp_size": 4
             }
        }
    }' &> v2dycp_${NODE_RANK}.log &