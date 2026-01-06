export VLLM_USE_V1=1
export VLLM_VERSION=0.13.0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=380

export PROMETHEUS_MULTIPROC_DIR=/tmp/

export COMMON_ARGS="
    --trust-remote-code
    --served-model-name auto
    --distributed-executor-backend mp
    --model-loader-extra-config {\"enable_multithread_load\":true,\"num_threads\":8}
"

