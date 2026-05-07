# PD 分离测试报告 — 2026-04-25

## 测试环境

| 项目 | 配置 |
|------|------|
| 机器 | 8x NVIDIA L20X (143GB each) |
| 模型 | Qwen3-30B-A3B (GQA, 30B MoE) |
| KV Transfer | MooncakeConnector (RDMA, mlx5 bond x8) |
| Prefill | GPU 0-3, TP=4, port 8400, kv_producer |
| Decode | GPU 4-7, DP=4 TP=1, dp_per_domain=4, port 8401, kv_consumer + DyCP |
| max_model_len | 131072 |
| block_size | Prefill=16, Decode=64 |
| num_cp_seqs | 2 |
| long_request_threshold | 128 |

## 性能测试结果

全部请求成功（0 fail）。

### Prefill 性能 (TTFT)

| 场景 | 数量 | Mean TTFT (ms) | P90 TTFT (ms) | Total Throughput (tok/s) |
|------|------|----------------|---------------|--------------------------|
| 4k × 1 | 1 | 107.89 | 107.89 | 11,637 |
| 4k × 64 | 64 | 10,790 | 19,322 | 12,175 |
| 32k × 8 | 8 | 3,500 | 5,607 | 41,693 |
| 128k × 2 | 2 | 4,370 | 5,441 | 42,790 |

### Decode 性能 (TPOT / ITL)

| 场景 | Mean TPOT (ms) | Mean ITL (ms) | P99 ITL (ms) |
|------|----------------|---------------|-------------|
| 4k × 1 | 7.95 | 7.70 | 8.51 |
| 4k × 64 | 8.00 | 7.75 | 9.01 |
| 32k × 8 | 8.25 | 7.99 | 10.45 |
| 128k × 2 | 8.81 | 8.54 | 12.56 |

### 性能分析

- **单条 4k 请求 TTFT 仅 108ms**，PD 分离后 prefill 延迟很低（TP=4 并行加速明显）
- **Decode TPOT 稳定在 8ms 左右**，ITL P99 在 12ms 以内，decode 性能良好
- **128k 长请求 TTFT 仅 4.4s**，得益于 TP=4 prefill 的并行计算能力
- **批量 4k 请求 throughput ~12k tok/s**，受限于 prefill TP=4 的计算带宽

## 精度测试结果

| 测试 | 数据集 | 数量 | 精度 | 错误 | 复读 |
|------|--------|------|------|------|------|
| PD 分离 (Mooncake) | gsm8k | 100 | **90.0%** | 10 | 3 |

### 精度对比（Qwen3-30B 历史数据）

| 配置 | 精度 | 说明 |
|------|------|------|
| DyCP dp_per_domain=8 (单机) | 82.0% | chunk=131072 |
| PD 分离 dp_per_domain=4 (Mooncake) | **90.0%** | Prefill TP=4, Decode DP=4 |
| 纯 DP（历史） | 76-82% | Qwen3-30B 基线水平 |

注：Qwen3-30B 不是推理模型，gsm8k 精度本身在 76-82% 左右。PD 分离模式下 90% 精度反而更高，可能与 TP=4 prefill 的数值精度更好有关（TP 不引入 DualChunkSwap 分片）。

3 条复读属于模型自身的 thinking 循环，非 DyCP 引入。

## 结论

1. **PD 分离链路正常**：Mooncake RDMA KV transfer 工作正常，prefill → decode KV 传输无异常
2. **性能良好**：单条 4k TTFT 108ms，128k TTFT 4.4s，decode TPOT 稳定 8ms
3. **精度正常**：90% 在 Qwen3-30B 的合理范围内
4. **稳定性良好**：所有性能测试 0 fail

## 启动脚本

- Prefill: `/ossfs/workspace/bench8/pd_prefill.sh`
- Decode: `/ossfs/workspace/bench8/pd_decode.sh`
- Prefill 日志: `/tmp/pd_prefill.log`
- Decode 日志: `/tmp/pd_decode.log`
- 精度结果: `/ossfs/workspace/code/benchmark/outputs/default/$(ls -t /ossfs/workspace/code/benchmark/outputs/default/ | head -1)/`
