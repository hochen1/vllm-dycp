set -x
rm -rf /root/.cache/vllm/torch_compile_cache/
#export PYTHONPATH=:$PYTHONPATH
export NCCL_DEBUG=WARN
# export MODEL_PATH=/input/model_weights/Qwen3-30B-A3B
export MODEL_PATH=/input/model_weights/DeepSeek-V2-Lite
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
export VLLM_ATTENTION_BACKEND=FLASHMLA
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_TORCH_PROFILER_DIR=${VLLM_TORCH_PROFILER_DIR:-"./profiles"}
rm -rf $VLLM_TORCH_PROFILER_DIR
mkdir -p $VLLM_TORCH_PROFILER_DIR
export VLLM_TORCH_PROFILER_WITH_STACK=0
export VLLM_USE_FORCE_LOAD_BLANCE=1
# export VLLM_ALL2ALL_BACKEND=allgather_reducescatter
ulimit -n 65536
vllm serve ${MODEL_PATH} \
    --port 8400 \
    $COMMON_ARGS \
    --hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":8.0,"original_max_position_embeddings":163840}}' \
    --distributed-executor-backend dmp \
    --max-model-len 1048576 \
    --max-num-batched-tokens 131072 \
    --gpu-memory-utilization 0.7 \
    --no-enable-prefix-caching \
    --data-parallel-size 8 \
    --tensor-parallel-size 1 \
    --dp-per-domain 8 \
    --block-size 64 \
    --cp-kv-cache-interleave-size 64 \
    --no-enforce-eager \
    --compilation-config '{"cudagraph_capture_sizes":[4, 8, 16, 24, 32, 64], "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes_for_cp": 4 }' \
    --num-cp-seqs 4 \
    --enable-expert-parallel &> v2lite_${NODE_RANK}.txt &
