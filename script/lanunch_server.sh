[root@gpuxdn033182141080 deepseek]# cat hunbu.sh 
#!/bin/bash
NODE_RANK=$1
export DATA_PARALLEL_HEAD_ADDRESS=${DATA_PARALLEL_HEAD_ADDRESS:-"master ip"}
export VLLM_TORCH_PROFILER_DIR=./profile
current_dir=$(dirname "$0")
export VLLM_TORCH_PROFILER_DIR=${VLLM_TORCH_PROFILER_DIR:-"./profiles"}
export VLLM_TORCH_PROFILER_WITH_STACK=0
# export ASCEND_LAUNCH_BLOCKING=1
rm -rf $VLLM_TORCH_PROFILER_DIR
mkdir -p $VLLM_TORCH_PROFILER_DIR
export TORCHDYNAMO_DISABLE=1
source ./hccl_buff_size.sh
source ./common.sh
ulimit -n 1048576
vllm serve /<your-model-path> \
    --port 8100 \
    $EXTRA_PARAMS \
    $COMMON_ARGS \
    --max-model-len 163840 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.93 \
    --max-num-seqs 12 \
    --trust-remote-code \
    --enable-prefix-caching \
    --no-enable-prefix-caching \
    --data-parallel-size 1 \
    --no-enforce-eager \
    --tensor-parallel-size 4 \
    -dcp=4 \
    --cp-kv-cache-interleave-size 128 \
    --enable-expert-parallel \
    --compilation-config '{"cudagraph_capture_sizes":[12], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
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
    }' &> v2lite_${NODE_RANK}.log &

