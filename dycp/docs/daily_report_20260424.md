# DyCP 工作日报 — 2026-04-24

## 分支状态

- **分支**: `dev-dycp-prefill-gqa-2`
- **未提交修改**: 4 个文件（148 行新增，28 行删除）
  - `vllm/v1/worker/gpu_model_runner.py` — 修复 `dycp_real_token_indices` 计算
  - `vllm/v1/attention/backends/flash_attn.py` — GQA DyCP prefill 支持
  - `vllm/v1/attention/backends/utils.py` — 新增 metadata 字段
  - `vllm/v1/core/sched/cross_dp_scheduler.py` — 调度器调整

## 今日完成的工作

### 1. Bug 修复：`n_real` 负数导致服务崩溃

**文件**: `vllm/v1/worker/gpu_model_runner.py:1994-2011`

**现象**: 当 `chunk_size < seq_len` 时（如 chunk=4096 跑 32k+ 请求），所有 worker 报 `ValueError: negative dimensions are not allowed` 并崩溃。

**根因**: `_build_attention_metadata()` 中手动计算 DualChunkSwap padding 分布时，假设"所有 padding 归 rank 0"（第2010行），但实际上 DualChunkSwap 的 padding 分布在多个 rank 的 tail chunk 上。当 `dycp_world_size=8`（dp_per_domain=8）时，`pcp_per_rank = padded / 8` 可能小于总 `pad_counts`，导致 `n_real = n_total - n_pad` 为负数。

**修复**: 将手动的 per-rank padding mask 计算替换为直接使用 `pcp_unpad_mask_cpu`（已在 `update_tokens_for_pcp()` 中正确计算），通过 `pcp_allgather_restore_idx` 映射到 restored 空间后提取 real token indices。

**影响范围**:
- chunk_size=4096: 之前 32k+ 请求全部崩溃 → 修复后全部正常
- chunk_size=32768: 之前 1M 请求崩溃 → 修复后正常
- chunk_size=131072: 无影响（padding 较小，未触发）

### 2. 环境修复

卸载了不兼容的 `flash_attn 2.7.3`（与 PyTorch 2.9 不兼容，`undefined symbol: _ZNK3c106SymInt6sym_neERKS0_`），vLLM 自带的 flash_attn backend 不受影响。

### 3. DyCP vs DCP 性能差异分析

对于单条长请求（如 512k），DyCP 性能不如 DCP 的根本原因：

1. **GPU 利用率不对等**: DyCP `num_cp_seqs=2` 只用 2/8 GPU，DCP 用 8/8
2. **每 rank 工作量差异**: DyCP 每 rank 256k vs DCP 每 rank 64k（4x）
3. **通信模式**: DyCP 每层需 allreduce（weighted output + LSE），DCP 用 allgather+reduce_scatter amortized
4. **DP 同步开销**: 8 个 DP rank 全参与 allreduce（含 6 个空闲）
5. **Batch Phase 锁定**: 6 个空闲 rank 被 prefill phase 锁住

**结论**: DyCP 设计优势在于混合负载（长短请求并存），非单条极长请求。

## 性能测试结果

**环境**: 8x NVIDIA L20X (143GB)
**配置**: DP=8, TP=1, dp_per_domain=8, num_cp_seqs=2, block_size=64

### DSv2Lite (MLA) — 单条 512k TTFT

| Chunk Size | TTFT (ms) | Throughput (tok/s) |
|------------|-----------|-------------------|
| 4,096      | 14,853    | 32,292            |
| 32,768     | 12,198    | 38,788            |
| 131,072    | 12,384    | 41,342            |

### DSv2Lite — 边界测试（chunk=131072，唯一修复前全稳定配置）

| 请求长度 | 数量 | Mean TTFT (ms) | Total Throughput (tok/s) |
|----------|------|----------------|--------------------------|
| 4k × 512 | 512  | 19,481         | 54,205                   |
| 32k × 64 | 64   | 8,053          | 135,979                  |
| 128k × 16| 16   | 11,196         | 106,915                  |
| 1M × 2   | 2    | 54,711         | 28,099                   |

### DSv2Lite — 边界测试（chunk=4096，修复后）

| 请求长度 | 数量 | Mean TTFT (ms) | Total Throughput (tok/s) |
|----------|------|----------------|--------------------------|
| 4k × 512 | 512  | 19,602         | 54,207                   |
| 32k × 64 | 64   | 10,116         | 107,462                  |
| 128k × 16| 16   | 13,159         | 87,650                   |
| 1M × 2   | 2    | 66,447         | 23,081                   |

### Qwen3 30B (GQA) — 边界测试

| Chunk Size | 4k×512 TTFT | 32k×64 TTFT | 128k×16 TTFT | 1M×2 TTFT |
|-----------|-------------|-------------|-------------|-----------|
| 4,096     | 22,364ms    | 14,435ms    | 15,052ms    | 36,240ms  |
| 32,768    | 22,257ms    | 11,995ms    | 20,768ms    | 56,963ms  |
| 131,072   | 15,745ms    | 12,086ms    | 21,506ms    | 132,841ms |

**所有模型、所有 chunk size、所有请求长度均通过测试（0 fail）。**

## 精度测试结果

| 模型 | 数据集 | 数量 | 精度 | 复读 | 正常错误 | 配置 |
|------|--------|------|------|------|---------|------|
| DeepSeek R1 (MLA) | gsm8k | 100 | **96.0%** | 0 | 4 | DP=2,TP=4,chunk=4096,max_len=32k |
| Qwen3 30B (GQA) | gsm8k | 100 | **82.0%** | 4 | 14 | DP=8,TP=1,chunk=131072 |
| Qwen3 235B FP8 (GQA) | gsm8k | 100 | **93.0%** | 0 | 7 | DP=8,TP=1,chunk=4096,max_len=32k |
| Qwen3 235B FP8 (GQA) | gsm8k | 1000 | **91.4%** | 2 | 84 | DP=8,TP=1,chunk=4096,max_len=32k |

- R1 精度 96%，历史水平 95-98%，**正常**
- Qwen3 30B 精度 82%，历史水平 76-82%，**正常**（非推理模型）
- Qwen3 235B 精度 91.4%（1000条），1000 条中 2 条复读（#108, #710），**待排查**

## 待办

- [ ] 排查 Qwen3 235B 的 2 条复读问题（#108, #710）
- [ ] 排查 Qwen3 30B 的 4 条复读问题
- [ ] 提交 `n_real` 负数 bug fix commit
- [ ] 在 DCP 模式下对比 235B 精度基线（确认复读是否 DyCP 引入）

## 修改文件清单

| 文件 | 修改内容 | 行数变化 |
|------|---------|---------|
| `vllm/v1/worker/gpu_model_runner.py` | 修复 `dycp_real_token_indices` padding mask 计算 | +33/-26 |
| `vllm/v1/attention/backends/flash_attn.py` | GQA DyCP prefill DualChunkSwap + cache refill | +104 |
| `vllm/v1/attention/backends/utils.py` | 新增 `dycp_full_slot_mapping`/`dycp_real_token_indices` 字段 | +11 |
| `vllm/v1/core/sched/cross_dp_scheduler.py` | long_request_threshold 调整 | +1/-1 |

## 注意事项

- `flash_attn 2.7.3` 已卸载，与 PyTorch 2.9 不兼容。vLLM 自带 backend 不受影响。
- chunk_size=4096 在长请求（1M）下 TTFT 比 chunk=32k/128k 慢约 20-50%，但稳定性已通过验证。
- Qwen3 30B 的 `original_max_position_embeddings=40960`，跑 1M 上下文需要 YaRN factor=32.0。
- DeepSeek R1 670B 单节点 8x L20X 可用 DP=2,TP=4+EP 部署，需 gpu_memory_utilization=0.9。
