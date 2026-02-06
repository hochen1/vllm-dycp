set -x
rm -rf /root/.cache/vllm/torch_compile_cache/
export PYTHONPATH=/input/chenxiao.cx/codebase/vllm-dycp:$PYTHONPATH
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
export VLLM_VERSION=0.13.0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=380
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN_MLA
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_TORCH_PROFILER_DIR=${VLLM_TORCH_PROFILER_DIR:-"./profiles"}
rm -rf $VLLM_TORCH_PROFILER_DIR
mkdir -p $VLLM_TORCH_PROFILER_DIR
export VLLM_TORCH_PROFILER_WITH_STACK=0
export VLLM_USE_FORCE_LOAD_BLANCE=1
ulimit -n 65536
vllm serve ${MODEL_PATH} \
    --async-scheduling \
    --port 8400 \
    $COMMON_ARGS \
    --distributed-executor-backend mp \
    --hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":8.0,"original_max_position_embeddings":163840}}' \
    --max-model-len 1024000 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.8 \
    --no-enable-prefix-caching \
    --data-parallel-size 1 \
    --tensor-parallel-size 8 \
    -dcp=8 \
    --block-size 64 \
    --cp-kv-cache-interleave-size 64 \
    --no-enforce-eager \
    --max-num-seqs 1024 \
    --enable-expert-parallel \
    --compilation-config '{"cudagraph_capture_sizes":[4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304, 312, 320, 328, 336, 344, 352, 360, 368, 376, 384, 392, 400, 408, 416, 424, 432, 440, 448, 456, 464, 472, 480, 488, 496, 504, 512, 516, 520, 524, 528, 532, 536, 540, 544, 548, 552, 556, 560, 564, 568, 572, 576, 580, 584, 588, 592, 596, 600, 604, 608, 612, 616, 620, 624, 628, 632, 636, 640, 644, 648, 652, 656, 660, 664, 668, 672, 676, 680, 684, 688, 692, 696, 700, 704, 708, 712, 716, 720, 724, 728, 732, 736, 740, 744, 748, 752, 756, 760, 764, 768, 772, 776, 780, 784, 788, 792, 796, 800, 804, 808, 812, 816, 820, 824, 828, 832, 836, 840, 844, 848, 852, 856, 860, 864, 868, 872, 876, 880, 884, 888, 892, 896, 900, 904, 908, 912, 916, 920, 924, 928, 932, 936, 940, 944, 948, 952, 956, 960, 964, 968, 972, 976, 980, 984, 988, 992, 996, 1000, 1004, 1008, 1012, 1016, 1020, 1024], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
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