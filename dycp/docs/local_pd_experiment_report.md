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

## 7. 结论

1. **PD 分离架构可行**: Proxy + KV Connector 成功实现了 prefill (全 CP) → decode (CP=1) 的分离
2. **精度正确**: Pre-mask KV capture 解决了 interleave mask 导致的 KV 丢失问题，PD 分离模式输出质量正确
3. **性能**: 短/中等长度 prompt 下 CP 的加速效果被通信开销抵消，预期在超长序列 (>32K) 上收益显著
4. **改进方向**: 减少 KV 传输开销（GPU direct）、支持 chunked prefill 跨步累积、优化 all-gather 通信
