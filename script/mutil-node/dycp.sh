set -x
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <node_rank>"
    echo "<node_rank> should start from 0 for both prefill and decode."
    exit 1
fi
NODE_RANK=$1
EXTRA_PARAMS=$([ $NODE_RANK -ne 0 ] && echo "--headless" || echo "--api-server-count 1")
EXTRA_PARAMS=$([ $NODE_RANK -ne 0 ] && echo "--headless" || echo "--api-server-count 1")
if [ $NODE_RANK -eq 0 ]; then
    #curl http://node1-ip:port/dycp & \
    curl http://node2-ip:port/dycp & \
    #curl http://node3-ip:port/dycp & \
    wait
fi
rm -rf /root/.cache/vllm/torch_compile_cache/
# export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_DEEPEP_BUFFER_SIZE_MB=0
export VLLM_MOE_DP_CHUNK_SIZE=64
export VLLM_USE_DEEP_GEMM=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/input/chenxiao.cx/codebase/vllm-dycp:$PYTHONPATH
export PYTHONPATH=/input/chenxiao.cx/codebase/temp-code-1-30/tools/ep_kernels/ep_kernels_workspace/DeepEP:$PYTHONPATH
export NCCL_DEBUG=WARN
# export MODEL_PATH=/input/model_weights/Qwen3-30B-A3B
# export MODEL_PATH=/input/model_weights/DeepSeek-V2-Lite
# export MODEL_PATH=/input/model_weights/Qwen3-235B-A22B-Instruct-2507
# export MODEL_PATH=/input/chenxiao.cx/model/DeepSeek-R1/model
export MODEL_PATH=/input/chenxiao.cx/model/Qwen3-235B-Instruct-FP8/model/
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
# export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_TORCH_PROFILER_DIR=${VLLM_TORCH_PROFILER_DIR:-"./profiles"}
# export CUDA_LAUNCH_BLOCKING=1
rm -rf $VLLM_TORCH_PROFILER_DIR
mkdir -p $VLLM_TORCH_PROFILER_DIR
export VLLM_TORCH_PROFILER_WITH_STACK=0
export VLLM_USE_FORCE_LOAD_BLANCE=1
export VLLM_ALL2ALL_BACKEND=deepep_low_latency
ulimit -n 65536
vllm serve ${MODEL_PATH} \
    --port 8400 \
    ${EXTRA_PARAMS} \
    $COMMON_ARGS \
    --async-scheduling \
    --distributed-executor-backend dmp \
    --hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":8.0,"original_max_position_embeddings":131072}}' \
    --max-model-len 524288 \
    --max-num-batched-tokens 64 \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching \
    --data-parallel-size 8 \
    --tensor-parallel-size 2 \
    --data-parallel-size-local 4 \
    --data-parallel-address="10.11.17.5" \
    --data-parallel-rpc-port 8400 \
    --data-parallel-start-rank $((NODE_RANK * 4)) \
    --block-size 64 \
    --cp-kv-cache-interleave-size 64 \
    --no-enforce-eager \
    --max-num-seqs 64 \
    --enable-expert-parallel \
    --dp-per-domain 4 \
    --num-cp-seqs 2 \
    --compilation-config '{"cudagraph_capture_sizes":[2, 4, 8, 10, 16, 18, 32, 34, 64], "cudagraph_mode": "FULL_DECODE_ONLY" , "cudagraph_capture_sizes_for_cp": 2}' \
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
                    "dp_size": 4,
                    "tp_size": 2
             }
        }
    }'  &> ${NODE_RANK}dycp.log & 