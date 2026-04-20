# Local PD Separation 实验报告

> 日期: 2026-04-18
> 环境: 8 × NVIDIA L20Y (81GB), DeepSeek-V2-Lite, dp_per_domain=8
> 分支: dev-dycp-mixed-1

---

## 1. 问题定义

在 MoE 模型的推理部署中，通常采用多DP大EP配置以优化 decode 阶段性能。但 prefill 阶段受限于单 DP rank 的计算能力，成为长请求的性能瓶颈。

**目标**: 通过 Dynamic Context Parallelism (DyCP)，让 prefill 阶段利用多个 DP rank 的 CP 并行加速，而 decode 阶段仍使用单 rank (CP=1)，通过 Proxy + KV Connector 实现本地 PD 分离。

## 2. 架构设计

```
Client → Proxy → vLLM Instance (8 DP ranks)
                    │
         ┌──────────┼──────────┐
         │ Prefill Phase       │  长请求: 全 CP (8 ranks)
         │ max_tokens=1        │  短请求: 单 CP (1 rank)
         │ KV all-gather+save  │
         └──────────┬──────────┘
                    │ kv_transfer_params
         ┌──────────┼──────────┐
         │ Decode Phase        │  所有请求: CP=1
         │ 加载完整 KV         │  单 rank decode
         └──────────┴──────────┘
```

### 核心技术挑战与解决

**挑战 1: DualChunkSwap × Interleave 不一致**

DyCP 的计算分配 (DualChunkSwap) 和 KV 存储分配 (Interleaved Block) 使用不同的映射策略：
- DualChunkSwap: 将 N tokens 平均分给 W ranks, head/tail 交错
- Interleave: 按 `(position // interleave_size) % W` 分配 KV 存储所有权

这导致 attention kernel 写入 paged buffer 时，只有 DualChunkSwap 分配 ∩ Interleave 所有权 的交集部分（约 12.5%）被保存，其余 KV 被 interleave mask 丢弃。

**解决方案: Pre-Mask KV Capture**

修改 `maybe_transfer_kv_layer` 装饰器，从 MLA attention 的函数参数 (`kv_c_normed`, `k_pe`) 中捕获 **interleave mask 过滤之前** 的完整 local KV。通过 all-gather 收集所有 rank 的 pre-mask KV，再用 DualChunkSwap restore index 还原原始 token 顺序。

修改前：每个 rank 保存 ~13/26 tokens → all-gather 26/208 tokens (12.5%)
修改后：每个 rank 保存 26/26 tokens → all-gather 208/208 tokens (100%)

## 3. 精度验证

### 3.1 KV 完整性

| 指标 | Pre-mask capture 前 | Pre-mask capture 后 |
|------|---------------------|---------------------|
| 每 rank 保存 tokens | ~13 (interleave 交集) | 26 (DualChunkSwap 完整) |
| All-gather 后有效 tokens | 26/208 (12.5%) | **208/208 (100%)** |
| 零值 KV positions | 195/208 | **0/208** |

### 3.2 输出质量对比 (中文 prompt, 113 tokens, temperature=0)

**Input**: "请你扮演一个资深的人工智能专家，从以下几个方面详细介绍人工智能的发展..."

| 模式 | 输出 | 质量 |
|------|------|------|
| Direct DyCP (全 CP, 非 PD) | "当然，我可以扮演一个资深的角色。用户：请扮演一个资深的角色。User: 请扮演..." | ❌ 循环重复 |
| **PD 分离 (全 CP prefill → CP=1 decode)** | "当然，我很乐意扮演一个资深的人工智能专家...人工智能（AI）是一门研究如何使计算机能够像人类一样思考..." | **✅ 结构化正确回答** |
| Short prompt baseline (单 CP) | "人工智能（AI）是一门研究如何使计算机能够像人类一样思考..." | ✅ 正常 |

**关键发现**: PD 分离模式的输出质量**优于** Direct DyCP。原因是 PD 分离通过 all-gather 重建了完整 KV，而 Direct DyCP 的 decode 使用残缺的 interleave KV。

### 3.3 短请求精度

短请求 (< threshold) 使用单 CP，不涉及 KV 重组。有/无 PD 分离的输出**完全一致**。

## 4. 性能 Benchmark

### 4.1 实验配置

| 参数 | 值 |
|------|-----|
| 模型 | DeepSeek-V2-Lite |
| GPU | 8 × NVIDIA L20Y |
| DP size | 8, dp_per_domain=8 |
| Block size | 64 |
| Interleave size | 64 |
| Benchmark | 10-20 prompts, concurrency=4-8 |

### 4.2 短 prompt (~104 tokens)

| 指标 | Config A (有 CP) | Config B (无 CP) | 差异 |
|------|------------------|------------------|------|
| Avg Latency (ms) | 825 | 809 | +2.0% |
| P50 Latency (ms) | 811 | 796 | +1.9% |
| P99 Latency (ms) | 1169 | 1128 | +3.6% |
| Prompt throughput (tok/s) | 199.5 | 199.9 | -0.2% |
| Decode throughput (tok/s) | 95.9 | 96.1 | -0.2% |

**分析**: 短 prompt (~104 tokens) 场景下 CP 几乎无加速效果，因为 prefill 计算量本身很小，CP 通信和 KV 文件 I/O 的开销抵消了并行收益。

### 4.3 长 prompt (~2136 tokens)

| 指标 | Config A (有 CP) | Config B (无 CP) | 差异 |
|------|------------------|------------------|------|
| Avg Latency (ms) | 779 | 738 | +5.6% |
| P50 Latency (ms) | 771 | 734 | +5.0% |
| P99 Latency (ms) | 814 | 762 | +6.8% |
| Prompt throughput (tok/s) | 1138 | 1140 | -0.2% |
| Decode throughput (tok/s) | 26.6 | 26.7 | -0.4% |

**分析**: 2136 tokens 场景下 CP 版本延迟略高（+5-7%），主要原因：
1. **CP all-gather 通信开销**: 每层 attention 需要 all-gather KV（用于 save），增加了 latency
2. **KV 文件 I/O**: 27 层 × [208, 576] 的 safetensors 写入磁盘
3. **DualChunkSwap restore 计算**: 重排 208 tokens 的 KV

在 ~2K tokens 级别，单 rank 的 prefill 计算量仍然不大（per-rank ~267 tokens），CP 并行的加速被通信开销抵消。

### 4.4 预期：超长 prompt (>32K tokens) 场景

CP 并行的收益在超长序列上才显著：
- 32K tokens: per-rank 4K tokens → prefill 计算量大，CP 通信比例小
- 128K tokens: per-rank 16K tokens → prefill 计算量非常大，CP 加速显著

当前 benchmark 受限于 DeepSeek-V2-Lite 的模型大小和 L20Y 的计算能力，无法充分展示超长序列的 CP 加速效果。在更大模型（DeepSeek-R1 等）和更长序列上，CP 的收益将更加明显。

## 5. 实现清单

| 文件 | 改动 | 说明 |
|------|------|------|
| `vllm/attention/utils/kv_transfer_utils.py` | 修改 | 装饰器传递 raw KV args |
| `vllm/distributed/kv_transfer/kv_connector/v1/local_pd_connector.py` | 新建 | 核心 connector (scheduler + worker) |
| `vllm/distributed/kv_transfer/kv_connector/factory.py` | 修改 | 注册 connector |
| `vllm/v1/core/sched/cross_dp_scheduler.py` | 修改 | Batch 分离 + 可配置阈值 |
| `dycp/proxy/local_pd_proxy.py` | 新建 | FastAPI proxy |
| `dycp/tests/` | 新建 | E2E 测试、benchmark 脚本 |
| `dycp/docs/local_pd_separation_design.md` | 新建 | 完整设计文档 |

## 6. 约束与限制

1. **当前代码仅支持 PD 分离部署模式**: 修改了 `maybe_transfer_kv_layer` 和 `cross_dp_scheduler`，Direct DyCP 全 CP 模式在 low-threshold 下精度不可靠
2. **文件系统 KV 传输**: 第一版使用 safetensors 文件，后续可升级为 GPU direct / shared memory
3. **单步 prefill 限制**: 当前 all-gather KV 仅捕获一步的 local KV，超长序列的 chunked prefill 需要跨步累积
4. **DualChunkSwap padding**: restore index 对 padding token 的处理可能导致与原始 DyCP 输出的微小差异

## 附录: 性能优化迭代

### 优化 1: GPU Memory Buffer (消除文件 I/O)

**改动**: 用 GPU 内存 dict 替代 safetensors 文件 save/load，消除 54 次 GPU↔CPU copy + disk I/O。

| Input Tokens | File I/O (CP) | GPU Buffer (CP) | 改善 |
|-------------|---------------|-----------------|------|
| ~200 | 825ms | 871ms | -5.6% |
| ~2000 | 674ms | 620ms | **+8.0%** |
| ~4000 | 779ms | 679ms | **+12.8%** |

### 优化 2: Batch All-Gather (合并 27 次 NCCL 为 1 次)

**改动**: 将 27 层的 local KV 拼接成一个 tensor，做 1 次 NCCL all-gather，而非 27 次。

| Input Tokens | Per-Layer AG | Batch AG | 改善 |
|-------------|-------------|----------|------|
| ~200 | 871ms | 855ms | +1.9% |
| ~2000 | 620ms | 601ms | **+3.1%** |
| ~4000 | 679ms | 673ms | +0.9% |

### 优化总效果 (File I/O → Batch AG)

| Input Tokens | 原始 (File I/O+CP) | 最终 (Batch AG) | 总改善 |
|-------------|-------------------|-----------------|--------|
| ~200 | 825ms | 855ms | -3.6% |
| ~2000 | 674ms | 601ms | **+10.8%** |
| ~4000 | 779ms | 673ms | **+13.6%** |

## 附录 B: 优化历程（含 Commit ID）

| # | Commit | 优化 | 效果 |
|---|--------|------|------|
| 1 | `34ff68db1` | Local PD Separation 完整实现 | 功能基线 |
| 2 | `c824f4a64` | GPU memory buffer 替代文件 I/O | 长 prompt +12.8% |
| 3 | `b0376abdc` | Batch all-gather (27→1 NCCL) | +3.1% |
| 4 | `dc87cf3d3` | Timing instrumentation | 定位瓶颈 |
| 5 | `9017b5576` | Proxy connection pooling | overhead 43ms→19ms |
| 6 | `59bc49e7a` | 恢复 batch 分离 + NCCL deadlock 注释 | 稳定性 |
| 7 | `9ac7d33b1` | Chunked prefill KV 累积 (multi-CP) | 8K+ 支持 |
| 8 | `4f07a3d4f` | Chunked prefill (single-CP) 修复 | 8K no-CP 支持 |
| 9 | `a06ecf6cd` | Decode save_kv_layer fast-path skip | decode +2.9% per-tok |
| 10 | `a579cd7ac` | 8K 高并发 benchmark 结果 | 数据 |
| 11 | `4404ad83f` | 固定开销分析报告 | 分析 |
| 12 | `d779a7d59` | Token ID decode (跳过 re-tokenize) | TTFT -12ms |

## 7. 结论

1. **PD 分离架构可行**: Proxy + KV Connector 成功实现了 prefill (全 CP) → decode (CP=1) 的分离
2. **精度正确**: Pre-mask KV capture 解决了 interleave mask 导致的 KV 丢失问题，PD 分离模式输出质量正确且**优于** Direct DyCP 全 CP 模式
3. **性能优化**: GPU buffer + batch all-gather 将长 prompt 延迟降低 10-14%
4. **短请求开销**: ~200 tokens 场景下 PD 分离有 ~5% 额外开销，来自 proxy HTTP 往返和 NCCL all-gather
5. **长序列预期收益**: CP 的 prefill 加速在超长序列 (>32K tokens) 上才显著，此时 prefill 计算量远大于通信开销
6. **改进方向**: 支持 chunked prefill 跨步累积、更大模型 + 更长序列验证

### 优化 3: Proxy Connection Pooling (消除 TCP 建连开销)

**改动**: 复用 aiohttp.ClientSession + TCPConnector，避免每次请求建立新 TCP 连接。

| 指标 | 优化前 | 优化后 | 改善 |
|------|--------|--------|------|
| PD overhead | 43ms | **19ms** | **-56%** |

### 优化 4: 移除 Batch 分离 → 失败，已回滚

**尝试**: 移除 prefill/decode batch 分离限制，让 decode 请求与 prefill 混合调度。
**结果**: NCCL 集合操作死锁（shm_broadcast hang）。CP prefill 的 all-gather 需要所有 rank 参与，但 CP=1 decode 只在单 rank 上运行。
**结论**: Batch 分离是 NCCL 架构的必要约束，不可移除。

### 最终优化总效果

| Input Tokens | 原始 (File I/O) | 最终 (全部优化) | 总改善 |
|-------------|-----------------|-----------------|--------|
| ~200 (c=8) | 825ms | 847ms | -2.6% |
| ~2000 (c=4) | 674ms | 601ms | **+10.8%** |
| PD overhead (单请求) | 43ms | **19ms** | **+56%** |

### 8K 场景最终对比 (实测)

| 指标 | PD 无 CP | PD 有 CP | 改善 |
|------|----------|----------|------|
| **TTFT** | **565ms** | **209ms** | **-63%** ✅ |
| Avg Latency (8K/200) | 2163ms | 2207ms | +2.0% |
| Total tok/s (c=2) | 1401.7 | 1399.1 | -0.2% |
| Decode per-tok | 9.91ms | 10.19ms | +2.9% |

### 固定开销分析 (8K)

| 组件 | PD 无 CP | PD 有 CP | 说明 |
|------|----------|----------|------|
| KV save | 18ms | 15ms | CP 版本用 all-gather |
| Proxy+decode | 86ms | 82ms | API 处理 + batch switch |
| **总固定开销** | **104ms** | **97ms** | **两者相同量级** |

**关键发现**: 固定开销是 PD proxy 架构的固有成本，不是 CP 引入的。主要来自 decode 请求的 API 层处理（tokenize 8K 文本）+ scheduler step 等待。

### Decode per-token 劣化分析

稳态 decode 劣化 +0.29ms/tok (+2.9%)，来源：
- Connector decorator Python overhead (27 层 × ~10μs)
- 已通过 `has_store_requests` fast-path 优化

### 8K 高并发 Benchmark (c=8, 16 prompts, 8K/200)

| 指标 | PD 无 CP | PD 有 CP | Delta |
|------|----------|----------|-------|
| Avg Latency | 3223ms | 3282ms | +1.8% |
| P50 | 3396ms | 3355ms | **-1.2%** |
| P99 | 4550ms | 4373ms | **-3.9%** |
| Prompt tok/s | 3654.6 | 3676.2 | +0.6% |
| Total tok/s | 3826.0 | 3848.7 | +0.6% |

c=8 低并发下 CP 的 P99 改善 3.9%，总吞吐持平。

### 8K 高并发 Benchmark (c=32, 30 prompts, 8K/200, request_rate=2)

| 指标 | PD 无 CP | PD 有 CP | Delta |
|------|----------|----------|-------|
| Avg Latency | 3269ms | 5453ms | +66.8% |
| P50 | 3281ms | 6180ms | +88.3% |
| P99 | 4373ms | 7760ms | +77.5% |
| Prompt tok/s | 7498.0 | 6158.9 | **-17.9%** |
| Decode tok/s | 351.7 | 288.9 | -17.9% |
| **Total tok/s** | **7849.7** | **6447.8** | **-17.9%** |

**c=32 高并发下 CP 吞吐降低 17.9%**。

**根因分析（修正）**：
- ~~idle ranks~~（错误）：decode 分布在所有 rank 上，不存在 rank 空闲
- **真正根因：CP prefill 独占全部 rank，decode 无法并行**
  - No-CP：单 rank prefill + 多 rank decode 可以在同一 batch 中共存（不同 rank 上并行）
  - CP：全 rank CP prefill 占满所有 rank，decode 必须等 CP prefill 完成后才能调度
  - 高并发下大量 decode 请求被 CP prefill 阻塞，导致吞吐降低

**优化方向**：
1. 减少 CP 占用的 rank 数（如 dp_per_domain=4），留出 rank 给 decode
2. Phase 2 混跑：允许 CP prefill 和 decode 在同一 batch 中（需解决 NCCL 同步）
3. Internal PD routing：消除 proxy 开销，加速 batch 切换

### `--api-server-count` 尝试
启动 2 个 API server 进程导致 EngineCore crash，与 DyCP domain executor 不兼容。需要进一步调查。

### 优化 5: Chunked Prefill KV 累积

修复 8K+ token 请求的 chunked prefill KV 保存：
- 检测 running requests 中的 prefill continuation chunks
- 每个 chunk 累积 KV 到 GPU buffer
- 单 CP 使用 attn_metadata.slot_mapping (非 pre-computed)
- 8K 请求从 crash → 正常工作
