# DyCP Prefill 设计报告

## 1. 概述

### 1.1 背景与动机

**Context Parallelism (CP)** 是一种在多个 GPU 之间切分序列上下文的并行策略，主要用于处理长序列推理。vLLM 已有的 CP 实现包括：

- **DCP (Decode Context Parallelism)**：decode 阶段的上下文并行
- **PCP (Prefill Context Parallelism)**：prefill 阶段的上下文并行

然而，上述 CP 策略是**静态的**——batch 内所有请求要么全部走 CP，要么全部不走 CP。在 PD 分离场景下，这不够灵活。

**DyCP (Dynamic Context Parallelism)** 解决了这个问题：同一 batch 中，**部分请求使用 CP**（DyCP 请求），**部分请求使用 DP**（普通请求）。这使得调度器能够根据每个请求的序列长度动态决定是否启用 CP，从而兼顾长序列的内存和计算效率以及短序列的批处理吞吐。

### 1.2 适用场景

- **PD 分离部署**：Prefill 节点独立运行，batch 中可能混合长短序列
- **长序列推理**：超过单 GPU 内存容量的 KV cache 需要跨 GPU 分布
- **混合负载**：同一 prefill batch 中同时包含需要 CP 的长序列和不需要 CP 的短序列

### 1.3 已支持的后端

| 后端 | DyCP Decode | DyCP Prefill |
|------|:-----------:|:------------:|
| MLA (DeepSeek) | ✅ | ✅ |
| Flash Attention (GQA) | ✅ | ✅ |
| FlashInfer (GQA) | ✅ | ✅ |

---

## 2. DualChunkSwap 策略

### 2.1 核心原理

DyCP prefill 使用 **DualChunkSwap** 策略处理 new-token 注意力。该策略的核心思想是将每个请求的 Q 和 KV 序列分为 **head（前半）** 和 **tail（后半）** 两部分，不同 rank 处理不同组合，实现负载均衡。

### 2.2 分片方式

对于 `dycp_world_size` 个 rank，每个 rank `r` 处理：

**Query 分片**（对称，所有 rank 相同）：
- `q_head`: 每个请求的前 `⌊q_len/2⌋` 个 token
- `q_tail`: 每个请求的后 `q_len - ⌊q_len/2⌋` 个 token

**KV 分片**（非对称，不同 rank 不同）：
- `kv_head` 长度: `⌊q_len × (r+1) / 2⌋`（rank 越大，head 部分越长）
- `kv_tail` 长度: `⌊q_len × (2W - r) / 2⌋`（rank 越小，tail 部分越长）

其中 `W = dycp_world_size`。

### 2.3 负载均衡分析

每个 rank 的总计算量 = `q_head × kv_head + q_tail × kv_tail`。DualChunkSwap 通过让 head 和 tail 部分的 KV 长度互补（一个增大时另一个减小），使得不同 rank 之间的总计算量大致均衡。

### 2.4 输出合并

1. 分别执行 head 和 tail 注意力，各返回 `(output, lse)`
2. 将 head/tail 输出 concat，通过 `output_restore_idx` 恢复为原始 token 顺序
3. 如果存在 context（已缓存的前缀 KV），通过 LSE-aware merge 合并 new-token 和 context 的注意力输出

---

## 3. 整体数据流

```
Prefill Batch (含 DyCP + DP 请求)
│
├─ MetadataBuilder.build()
│  ├─ 识别 DyCP 请求 (num_dycp_reqs, num_dycp_tokens)
│  ├─ 计算 DualChunkSwap 索引:
│  │  ├─ get_pcp_query_indices() → q_head_indices, q_tail_indices
│  │  ├─ get_pcp_kv_indices() → kv_head_indices, kv_tail_indices
│  │  └─ output_restore_idx (恢复原始顺序)
│  ├─ 计算 context KV 长度 (seq_len - query_len)
│  └─ Plan context wrapper (FlashInfer) / 准备 block_table (FA)
│
├─ Forward (DyCP 请求)
│  ├─ Phase A: KV AllGather
│  │  └─ pcp_kv_allgather_and_restore(key, value, restore_idx, dycp_group)
│  │     → gathered_key, gathered_value [全局 new-token KV]
│  │
│  ├─ Phase B: New-Token DualChunkSwap Attention (causal, dense)
│  │  ├─ index_select(q, q_head_indices) × index_select(kv, kv_head_indices)
│  │  │  → head_output, head_lse
│  │  ├─ index_select(q, q_tail_indices) × index_select(kv, kv_tail_indices)
│  │  │  → tail_output, tail_lse
│  │  └─ concat + restore order → new_token_output, new_token_lse
│  │
│  ├─ Phase C: Context Attention (non-causal, paged)
│  │  ├─ 对 local KV cache 执行 attention → context_output, context_lse
│  │  └─ cp_lse_ag_out_ar(context_output, context_lse, dycp_group)
│  │     → all-reduced context_output, context_lse
│  │
│  └─ Phase D: Merge
│     └─ merge_attn_states(output, context_output, context_lse,
│                          new_token_output, new_token_lse)
│
└─ Forward (DP 请求)
   └─ 标准 prefill attention (不涉及跨 rank 通信)
```

---

## 4. 三后端实现对比

### 4.1 MLA 后端 (`mla/common.py`)

MLA（Multi-head Latent Attention）后端的 DyCP prefill 实现与 GQA 后端有显著差异：

| 方面 | MLA 实现 |
|------|---------|
| **混合 batch 处理** | `split_metadata()` 将 batch 分为 DyCP 和 DP 两部分，各自独立走 `forward_common()` |
| **New-token 注意力** | `_run_prefill_new_tokens()` 使用内部 MLA kernel，支持 latent KV |
| **Context 注意力** | `_context_parallel_compute_prefill_context()` 使用分块(chunked)策略：AllGather KV → `reorg_kvcache()` → 分块注意力 → LSE merge |
| **KV 格式** | Compressed latent KV (`kv_c_normed`, `k_pe`)，非标准 KV cache |
| **跨 rank 通信** | `get_dycp_group().all_gather(kv)` 在 workspace 中操作 |

**特点**：
- MLA 使用 chunked context 策略，将 context 分块处理避免 OOM
- KV 格式为压缩 latent 表示，需要 `reorg_kvcache()` 重新组织
- 混合 batch 通过 metadata split 实现，代码路径较清晰

### 4.2 Flash Attention 后端 (`flash_attn.py`)

| 方面 | FA 实现 |
|------|--------|
| **混合 batch 处理** | DyCP 请求排在前面，DP 请求在后；DyCP 部分走 `_forward_with_dycp_prefill()`，DP 部分走 `split_flash_attn_metadata()` + 标准 prefill |
| **New-token 注意力** | `flash_attn_varlen_func(causal=True)` dense 注意力 |
| **Context 注意力** | `flash_attn_varlen_func(causal=False)` + `block_table` + `seqused_k` 实现 paged 非因果注意力 |
| **KV 格式** | 标准 GQA 格式 `[num_tokens, num_kv_heads, head_size]` |
| **跨 rank 通信** | `pcp_kv_allgather_and_restore()` (new-token) + `cp_lse_ag_out_ar()` (context) |

**特点**：
- 所有注意力计算均使用 `flash_attn_varlen_func`，统一且高效
- Context 注意力通过 `seqused_k` 参数限制每个请求只看 local context 部分
- LSE 格式为 `[H, B]` base-e，需要转换后传给 `cp_lse_ag_out_ar()`

### 4.3 FlashInfer 后端 (`flashinfer.py`)

| 方面 | FlashInfer 实现 |
|------|----------------|
| **混合 batch 处理** | 与 FA 相同：DyCP 前 + DP 后，分别处理 |
| **New-token 注意力** | 复用 `flash_attn_varlen_func(causal=True)` dense 注意力（非 FlashInfer 原生 API） |
| **Context 注意力** | `BatchPrefillWithPagedKVCacheWrapper.run(causal=False)` 使用 FlashInfer 原生 paged attention |
| **KV 格式** | 标准 GQA 格式，通过 `kv_cache.permute(stride_order)` 适配 FlashInfer 布局 |
| **跨 rank 通信** | 同 FA |

**特点**：
- **混合 API 策略**：New-token 注意力用 FA（因为 DualChunkSwap 需要 dense causal attention，FA 的 varlen_func 最直接），context 注意力用 FlashInfer wrapper（因为需要 paged attention 访问 KV cache）
- FlashInfer 使用 plan-then-run 模式：metadata build 阶段调用 `wrapper.plan()` 预编排，forward 阶段调用 `wrapper.run()` 执行
- 需要额外的 `dycp_qo_indptr` 字段（FlashInfer metadata 不存储 `query_start_loc`）
- LSE 格式为 `[B, H]` base-2，`cp_lse_ag_out_ar()` 需设置 `is_lse_base_on_e=False`

### 4.4 对比总结

| 特性 | MLA | Flash Attention | FlashInfer |
|------|-----|-----------------|------------|
| New-token kernel | MLA 内部 kernel | `flash_attn_varlen_func` | `flash_attn_varlen_func` |
| Context kernel | 分块 + MLA kernel | `flash_attn_varlen_func` + block_table | `BatchPrefillWithPagedKVCacheWrapper` |
| Context 限制方式 | chunked workspace | `seqused_k` 参数 | wrapper plan 时限制 paged KV 范围 |
| LSE 格式 | [H, B] base-e | [H, B] base-e | [B, H] base-2 |
| 混合 batch | metadata split | index split | index split |
| 额外字段 | 无 | 无 | `dycp_qo_indptr` |

---

## 5. 关键 API 和工具函数

### 5.1 DualChunkSwap 索引计算

```python
# vllm/v1/attention/backends/utils.py

get_pcp_query_indices(
    query_start_loc: torch.Tensor,  # [num_reqs+1] 前缀和
    num_reqs: int,
) -> tuple[torch.Tensor, torch.Tensor]
# 返回 (q_head_indices, q_tail_indices)
# head = 每个请求的前 ⌊q_len/2⌋ 个 token
# tail = 每个请求的后 q_len - ⌊q_len/2⌋ 个 token

get_pcp_kv_indices(
    kv_start_loc: torch.Tensor,
    num_reqs: int,
    pcp_size: int,
    pcp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]
# 返回 (kv_head_indices, kv_tail_indices)
# head_len = ⌊kv_len × (rank+1) / 2⌋
# tail_len = ⌊kv_len × (2W - rank) / 2⌋
```

### 5.2 KV AllGather

```python
# vllm/v1/attention/backends/utils.py

pcp_kv_allgather_and_restore(
    key: torch.Tensor,      # [num_local_tokens, num_kv_heads, head_size]
    value: torch.Tensor,
    num_tokens: int,
    restore_idx: torch.Tensor,  # 恢复原始 token 顺序的索引
    cp_group: GroupCoordinator,
) -> tuple[torch.Tensor, torch.Tensor]
# AllGather 后按 restore_idx 重排，返回全局 KV
```

### 5.3 LSE-Aware AllReduce

```python
# vllm/attention/ops/common.py

cp_lse_ag_out_ar(
    cp_attn_out: torch.Tensor,   # [B, H, D]
    cp_attn_lse: torch.Tensor,   # [B, H]
    cp_group: GroupCoordinator,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,  # True=FA, False=FlashInfer
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor
# 1. AllGather LSE → [N_ranks, B, H]
# 2. 计算全局 max LSE
# 3. 用 exp(local_lse - global_lse) 校正 output
# 4. AllReduce corrected output
```

### 5.4 注意力输出合并

```python
# vllm/attention/ops/merge_attn_states.py

merge_attn_states(
    output: torch.Tensor,          # [B, H, D] 合并结果写入此 tensor
    prefix_output: torch.Tensor,   # context attention 输出
    prefix_lse: torch.Tensor,      # [H, B] context LSE
    suffix_output: torch.Tensor,   # new-token attention 输出
    suffix_lse: torch.Tensor,      # [H, B] new-token LSE
)
# 使用 LSE 权重合并两路 attention: out = softmax_weight(lse_a, lse_b) * (a, b)
```

---

## 6. LSE 格式约定

不同后端和 API 对 LSE 的格式约定不同，这是实现中容易出错的地方：

| 来源 | LSE shape | 底数 | 说明 |
|------|-----------|------|------|
| Flash Attention `varlen_func` | `[H, B]` | e (自然) | `return_softmax_lse=True` |
| FlashInfer `wrapper.run()` | `[B, H]` | 2 | `return_lse=True` |
| `cp_lse_ag_out_ar` 输入 | `[B, H]` | 任意，通过 `is_lse_base_on_e` 指定 |
| `cp_lse_ag_out_ar` 输出 | `[B, H]` | 同输入 |
| `merge_attn_states` 输入 | `[H, B]` | e (自然) | prefix_lse / suffix_lse |

**转换规则**：
- FA → `cp_lse_ag_out_ar`: 需 `.transpose(0, 1)` 将 `[H,B]` → `[B,H]`
- FlashInfer → `cp_lse_ag_out_ar`: 直接传入，设 `is_lse_base_on_e=False`
- `cp_lse_ag_out_ar` 输出 → `merge_attn_states`: 需 `.transpose(0, 1)` 将 `[B,H]` → `[H,B]`
- FA new-token LSE (`[H,B]`) → `merge_attn_states`: 直接传入

---

## 7. 元数据字段

### 7.1 通用字段（三后端共用）

| 字段 | 类型 | 说明 |
|------|------|------|
| `num_dycp_reqs` | int | batch 中 DyCP 请求数量 |
| `num_dycp_tokens` | int | DyCP 请求的总 token 数 |

### 7.2 GQA 后端 DyCP Prefill 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `dycp_prefill_q_head_indices` | Tensor | Q head 部分的 token 索引 |
| `dycp_prefill_q_tail_indices` | Tensor | Q tail 部分的 token 索引 |
| `dycp_prefill_kv_head_indices` | Tensor | KV head 部分的 token 索引 |
| `dycp_prefill_kv_tail_indices` | Tensor | KV tail 部分的 token 索引 |
| `dycp_prefill_output_restore_idx` | Tensor | 恢复 head+tail concat 顺序到原始顺序的索引 |
| `dycp_prefill_context_kv_lens` | Tensor | 每个 DyCP 请求的 context KV 长度 |
| `max_dycp_prefill_context_kv_len` | int | 最大 context KV 长度 |
| `pcp_allgather_restore_idx` | Tensor | AllGather 后恢复 token 顺序的索引 |
| `dycp_qo_indptr` | Tensor | (仅 FlashInfer) DyCP 请求的 query offset |

---

## 8. 实现要点

### 8.1 DyCP 请求排序

在 metadata build 阶段，DyCP 请求必须排在 batch 的前面（排在 decode 之后、DP prefill 之前）。这是由 `CommonAttentionMetadata` 的构建逻辑保证的。

### 8.2 Context Wrapper 规划（FlashInfer 特有）

FlashInfer 的 paged attention 使用 plan-then-run 模式。DyCP context attention 需要在 build 阶段为 DyCP 请求单独规划一个 `BatchPrefillWithPagedKVCacheWrapper`：

- `causal=False`（context attention 是非因果的）
- paged KV indptr/indices 只包含 DyCP 请求
- 规划时使用调整后的 seq_lens（已通过 `get_cp_local_seq_lens` 限制为 local 部分）

### 8.3 FlashInfer 的 `dycp_qo_indptr`

FlashInfer 的 metadata 不像 FA 那样存储 `query_start_loc`。为了在 forward 中计算 per-request query 长度（构建 cu_seqlens 用于 `flash_attn_varlen_func`），需要额外存储 `dycp_qo_indptr` 字段。

### 8.4 混合 API 策略（FlashInfer）

FlashInfer DyCP prefill 采用混合 API 策略：
- **New-token 注意力**：使用 FA 的 `flash_attn_varlen_func`，因为 DualChunkSwap 需要 dense causal attention（index_select 后的连续 tensor），FA 的 varlen 接口最直接
- **Context 注意力**：使用 FlashInfer 的 `BatchPrefillWithPagedKVCacheWrapper`，因为需要 paged attention 访问已缓存的 KV cache

---

## 9. 验证方式

### 9.1 基本验证

```bash
# 语法检查
python3 -m py_compile vllm/v1/attention/backends/flash_attn.py
python3 -m py_compile vllm/v1/attention/backends/flashinfer.py
```

### 9.2 功能验证

1. **DyCP=1 基线**：单 rank 运行作为参考输出
2. **DyCP>1 正确性**：多 rank 运行，对比输出与 DyCP=1 的一致性
3. **混合 batch**：同一 batch 中混合 DyCP 和 DP 请求，验证各自输出正确

### 9.3 测试脚本

使用项目中的测试脚本（`scripts/` 目录下），配合不同模型（Qwen-235B, DeepSeek-V2-Lite 等）进行端到端验证。
