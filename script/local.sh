export PYTHONPATH=/input/chenxiao.cx/codebase/vllm-dycp:$PYTHONPATH
export VLLM_TORCH_PROFILER_WITH_STACK=0

RESULT_DIR="/tmp/chenao/bench-results"

mkdir -p ${RESULT_DIR}
# export MODEL_PATH=/input/model_weights/Qwen3-30B-A3B
# export MODEL_PATH=/input/model_weights/DeepSeek-V2-Lite
# Online - 4k/1.5k/3s/70ms
# export MODEL_PATH=/input/chenxiao.cx/model/DeepSeek-R1/model
export MODEL_PATH=/input/chenxiao.cx/model/Qwen3-235B-Instruct-FP8/model/
export INPUT_LEN=4096
export OUTPUT_LEN=1024
export NUM_PROMPTS=640
export MAX_CONCURRENCY=128
export REQUEST_RATE=inf

ulimit -n 65536
vllm bench serve \
    --backend openai-chat \
    --dataset-name random \
    --trust-remote-code \
    --use-local-json "/input/chenxiao.cx/workspace/qwen16/trace_256k.json" \
    --served-model-name auto \
    --model ${MODEL_PATH} \
    --random-input-len ${INPUT_LEN} \
    --random-output-len ${OUTPUT_LEN} \
    --num-prompts ${NUM_PROMPTS} \
    --max-concurrency ${MAX_CONCURRENCY} \
    --request-rate ${REQUEST_RATE} \
    --ignore-eos \
    --metric-percentiles "50,90,99" \
    --host localhost \
    --port 8400 \
    --save-result \
    --result-dir ${RESULT_DIR} \
    --endpoint /v1/chat/completions \
    --temperature 0.6 \
    --save-detailed
