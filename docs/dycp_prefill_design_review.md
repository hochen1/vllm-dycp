# DyCP Prefill for MLA 设计评审报告

> **Commit**: `5147bf2da` — `[feat] support prefill dycp for mla`
> **基线**: `c1ea0cfc9` — `[feat] support base pcp`
> **改动规模**: 14 files changed, +1936 -280 lines

---

## 1. 概述

本次提交在 vLLM 的 MLA (Multi-head Latent Attention) 注意力后端中实现了 **DyCP (Dynamic Context Parallelism)** 的 prefill 支持。DyCP 是一种 CP + DP 的混合并行模式：由 Cross-DP 调度器选出的长序列请求（CP 请求）被分片到所有 DP rank 上协作处理 prefill，而短序列请求（DP 请求）则保留在各自 rank 本地独立执行。

### 1.1 与纯 PCP 的关键区别

| 维度 | PCP (Prefill Context Parallelism) | DyCP (Dynamic Context Parallelism) |
|------|-----------------------------------|-------------------------------------|
| 请求同质性 | 所有 rank 处理相同的请求集 | 各 rank 可混合 CP 请求和本地 DP 请求 |
| 通信分组 | `get_pcp_group()` | `get_dycp_group()` |
| `pcp_world_size` | > 1 | = 1（不使用 PCP 分组） |
| 元数据一致性 | 天然一致 | 需要跨 rank 同步关键元数据 |
| Batch 结构 | 同构 prefill batch | 混合 batch：[DyCP reqs..., DP reqs...] |

---

## 2. 架构设计

### 2.1 整体数据流

```
                     Cross-DP Scheduler
                           │
                   ┌───────▼───────┐
                   │ select_dp()   │  ← 按 per-rank budget 选择目标 rank
                   │ rank_budgets  │
                   └───────┬───────┘
                           │
              ┌────────────▼─────────────┐
              │   gpu_model_runner        │
              │   prepare_inputs()        │
              │                           │
              │  ┌──────────┐ ┌─────────┐ │
              │  │ CP reqs  │ │ DP reqs │ │
              │  │(DyCP)    │ │(local)  │ │
              │  └────┬─────┘ └────┬────┘ │
              │       │            │       │
              │  PCPManager       pass-   │
              │  update_tokens    through  │
              │  for_pcp()                 │
              └───────┬──────────┬─────────┘
                      │          │
            ┌─────────▼──────────▼──────────┐
            │   MLA common.py               │
            │   forward()                   │
            │                               │
            │  ┌──────────────────────────┐  │
            │  │ Mixed batch?             │  │
            │  │  YES → split_metadata()  │  │
            │  │    Phase A: CP → forward │  │
            │  │    Phase B: DP → forward │  │
            │  │  NO → forward_common()   │  │
            │  └──────────────────────────┘  │
            │                               │
            │  forward_common():            │
            │  1. Cache local KV            │
            │  2. DyCP KV allgather         │
            │  3. DualChunkSwap prefill     │
            │  4. Chunked context allgather │
            │  5. Decode LSE allreduce      │
            └───────────────────────────────┘
```

### 2.2 核心设计决策

1. **Batch 分离策略**：混合 batch（DyCP + DP 请求共存）在 attention 层拆分为两个独立的 metadata 对象，分别走 DyCP 集合通信路径和本地计算路径。这避免了让 DP 请求参与不必要的集合通信。

2. **复用 PCP 基础设施**：DyCP 的 DualChunkSwap 分片、token 分配、restore index 等逻辑与 PCP 相同，通过独立的 `PCPManager` 实例 + `get_dycp_group()` 通信分组实现复用。

3. **元数据同步**：在 `_build_prefill` 中通过一次 `all_reduce(MAX)` 同步 `max_context_len_cpu` 和 `num_prefills_with_context_cpu`，确保所有 rank 的 chunked_context 元数据形状一致，替代原本每层 2 次的 consensus 函数。

4. **KV Cache 写入与计算分离**：KV cache 的 `concat_and_cache_mla` 始终使用本地 slot_mapping（只写本 rank 负责的 KV），而 attention 计算使用 allgathered 后的完整 KV。

---

## 3. 逐文件详细分析

### 3.1 `vllm/attention/backends/abstract.py` (+9 -3)

**改动摘要**：修复 AttentionImpl 基类中的 DyCP 相关属性。

| 改动 | 说明 |
|------|------|
| 添加 `total_cp_world_size` / `total_cp_rank` 别名 | 向后兼容旧代码中引用的属性名 |
| `need_to_return_lse_for_decode` 条件修复 | 从 `self.cp_world_size > 1` 改为 `self.dcp_world_size > 1 or self.dycp_world_size > 1` |

**设计意图**：在 DyCP 模式下 `cp_world_size = pcp_world_size * dcp_world_size = 1 * 1 = 1`，但 decode 阶段仍需返回 LSE 用于跨 rank allreduce。原条件 `cp_world_size > 1` 永远不会触发，导致 decode 输出无法正确合并。修复后直接检查 `dcp_world_size` 和 `dycp_world_size`。

### 3.2 `vllm/model_executor/layers/fused_moe/config.py` (+16) & `layer.py` (+8)

**改动摘要**：FusedMoE 层感知 DyCP 并行维度。

- `FusedMoEParallelConfig` 新增 `dycp_size` / `dycp_rank` 字段
- 通过 `get_dycp_group()` 初始化（带 `try/except AssertionError` 回退）
- `FusedMoE` 类暴露 `dycp_size` / `dycp_rank` property

**设计意图**：MoE 层的专家并行需要感知 DyCP 分组，以便在混合 DP 场景下正确路由 token 到专家。

### 3.3 `vllm/v1/attention/backends/mla/common.py` (+988 -82)

这是本次提交的核心文件，改动量最大。按功能模块逐一分析：

#### 3.3.1 `split_metadata()` 函数（新增 ~250 行）

**位置**：L452-700

将混合 DyCP+DP 的 `MLACommonMetadata` 拆分为两个独立的 metadata 对象：

```python
def split_metadata(attn_metadata) -> (dycp_metadata, dp_metadata):
    # 1. 按 num_dycp_reqs 切分 query_start_loc, slot_mapping
    # 2. 拆分 prefill metadata:
    #    - block_table[:n_dycp] vs block_table[n_dycp:]
    #    - query_start_loc 重新 rebase（DP 部分减去偏移量）
    #    - chunked_context 完整拆分（cu_seq_lens, starts, seq_lens, token_to_seq 等）
    # 3. 构造两个完整的 MLACommonMetadata 实例
```

**关键细节**：
- DyCP 部分保留 `pcp_allgather_restore_idx`（用于 KV allgather restore）
- DP 部分的 `query_start_loc` 和 `cu_seq_lens` 需要减去 DyCP 部分的偏移量重新 rebase
- `token_to_seq` 中 DP 部分的请求索引需要减去 `n_dycp`
- chunked_context 的拆分最复杂：需要同时处理 `cu_seq_lens`、`padded_local_cu_seq_lens`、`chunk_total_token` 等多个张量

#### 3.3.2 元数据字段扩展

| 类 | 新增字段 | 用途 |
|----|---------|------|
| `MLACommonPrefillMetadata` | `num_dycp_reqs: int = 0` | prefill 部分中 DyCP 请求数 |
| `MLACommonMetadata` | `num_dycp_tokens: int = 0` | DyCP 请求的 token 总数 |

#### 3.3.3 `MLACommonMetadataBuilder` 修改

**(a) `cp_virtual_block_size` 计算（L839-845）**

```python
if self.dycp_world_size > 1:
    self.cp_virtual_block_size = self.cp_local_block_size * self.dycp_world_size
else:
    self.cp_virtual_block_size = self.cp_local_block_size * self.cp_world_size
```

DyCP 模式下 `cp_world_size = 1`，但实际 KV 分片跨 `dycp_world_size` 个 rank，因此 virtual_block_size 需要用 `dycp_world_size`。

**(b) Chunked prefill workspace 扩大（L859-878）**

DyCP 模式下独立的 `elif` 分支，按 `dycp_world_size` 计算 workspace 扩展量（用于 KV allgather 后的临时存储）。

**(c) `reorder_batch_threshold` 强制为 1（L943-948）**

```python
if self.dycp_world_size > 1:
    self.reorder_batch_threshold = 1
```

FlashMLA 默认阈值为 128，即 token 数 <= 128 的请求会被归类为 "decode"。但 DyCP 的 `update_tokens_for_pcp` 假设 decode 请求在 CP 请求前面（有序），而 `reorder_batch_to_split_cp_and_normal` 不保证这一点。强制阈值为 1 使所有多 token 请求都走 prefill 路径，避免排序假设违反。

**(d) `_build_prefill` 中的 DyCP 同步（L1097-1140）**

```python
if self.dycp_world_size > 1 and prefill_num_dycp_reqs > 0:
    _sync = torch.tensor([max_context_len_cpu, num_prefills_with_context_cpu], ...)
    torch.distributed.all_reduce(_sync, op=ReduceOp.MAX, group=get_dycp_group().device_group)
```

**这是防止 NCCL 死锁的关键同步点**。不同 rank 有不同的 DP 本地请求，导致：
- `max_context_len_cpu` 在 rank 间不同 → 一些 rank 创建 `chunked_context`，另一些不创建
- `num_prefills_with_context_cpu` 不同 → `max_context_chunk` 不同 → allgather padding 大小不同

通过 `all_reduce(MAX)` 取所有 rank 的最大值，确保所有 rank 的 chunked_context 形状一致。代价：每 batch 1 次 all_reduce + 1 次 `.item()`（而非每层）。

**(e) DyCP 强制 prefill 分类（L1105-1115）**

```python
if self.dycp_world_size > 1 and num_dycp_reqs > 0 and kv_role in ("kv_producer", "kv_both"):
    num_decodes = 0
    num_prefills = num_reqs
    num_prefill_tokens = num_tokens
```

在 DyCP batch 中，CP 请求经过 `update_tokens_for_pcp` 分片后，各 rank 的本地 query 长度可能不同。如果用本地 query 长度来判断 decode/prefill 分类，不同 rank 可能得到不同结果 → 条件分支分歧 → NCCL 死锁。强制所有请求走 prefill 路径。

**(f) DyCP chunked context 本地布局计算（L1270-1365）**

对混合 batch，DyCP 请求使用跨 rank 的 local chunk layout（通过 `get_cp_local_seq_lens` 计算每个 rank 的本地 context 长度），DP 请求保持原始本地布局。生成 `local_context_lens_allranks` 矩阵：
- DyCP 请求行：所有 rank 的本地 context 长度
- DP 请求行：只有当前 rank 列有值，其他列为 0

**(g) DyCP PCP 索引构建（L1441-1480）**

仅对 DyCP 请求子集构建 PCP 索引（query head/tail indices, kv head/tail indices, output restore idx），使用 `dycp_world_size` 和 `dycp_rank` 替代 `pcp_world_size` 和 `pcp_rank`。

#### 3.3.4 `MLACommonImpl` 修改

**(a) `_run_prefill_new_tokens_fa` 重构（L1935-2170）**

原始 PCP 路径使用硬编码的 `prefill.query_start_loc // 2` 作为 cu_seqlens，这对奇数长度序列不正确。重构后：

1. 新增 `_safe_index()` 辅助函数：bounds-safe 的索引 clamp，防止 CUDA gather OOB
2. 新增 `_build_cu_seq_lens()` / `_max_seq_len()` 辅助函数
3. 新增 `_run_dual_chunk_attn()` 辅助函数：封装 DualChunkSwap 的单次 attention 调用
4. PCP 路径重写：使用精确的 per-request `q_head_seq_lens` / `q_tail_seq_lens` 和 per-request `kv_head_seq_lens` / `kv_tail_seq_lens`
5. 新增 DyCP `elif` 分支：逻辑与 PCP 相同，但使用 `dycp_rank` / `dycp_world_size`，且有额外的 gate 条件：
   - `prefill.num_dycp_reqs == block_table.shape[0]`（纯 DyCP batch）
   - `k.shape[0] > q.shape[0]`（KV 已 allgathered）

**(b) `_compute_prefill_context` 和 `_context_parallel_compute_prefill_context` 健壮性增强**

- 添加 `toks == 0` 和 `sum_seq_len == 0` 的 skip 检查
- `reorg_kvcache` 添加空 segment 的 fallback 处理
- 两个函数末尾添加 `output is None` 的零初始化兜底

**(c) Chunked context allgather 路径扩展（L2612-2666）**

```python
if self.pcp_world_size > 1:
    gathered = get_pcp_group().all_gather(...)
elif self.dycp_world_size > 1 and attn_metadata.num_dycp_reqs > 0:
    gathered = get_dycp_group().all_gather(...)
else:
    gathered = get_dcp_group().all_gather(...)
```

**(d) `_forward_prefill` 增加 override 参数（L2705-2768）**

```python
def _forward_prefill(self, ..., has_context_override=None, can_use_dycp_context_override=None):
```

在 `forward()` → `split_metadata()` → `forward_common()` 的调用链中，split 后的子 metadata 可能丢失一些判断上下文。通过 override 参数显式传递 `has_context` 和 `can_use_dycp_context` 的决策结果。

**(e) `forward()` / `forward_common()` 拆分（L2812-2920）**

原始的 `forward()` 被拆分为：

- **`forward()`**：入口方法，检测混合 batch 情况：
  - 如果 `num_dycp_reqs > 0` 且 `num_dycp_reqs < num_prefills`（混合 batch）：
    - 调用 `split_metadata()` 拆分
    - Phase A: 对 DyCP 子 batch 调用 `forward_common()`
    - Phase B: 对 DP 子 batch 调用 `forward_common()`
  - 否则：直接调用 `forward_common()`

- **`forward_common()`**：核心 attention 逻辑，包含：
  1. 本地 KV cache 写入（`concat_and_cache_mla`）
  2. DyCP KV allgather（`pcp_kv_allgather_and_restore` + `get_dycp_group()`）
  3. DualChunkSwap prefill attention
  4. Chunked context allgather
  5. Decode LSE allreduce

**(f) KV cache 写入提前到 allgather 前（L2920-2960）**

```python
# 写入本地 KV cache
local_slot_mapping = attn_metadata.slot_mapping.flatten()
ops.concat_and_cache_mla(local_k_c_normed, local_k_pe, kv_cache, local_slot_mapping, ...)

# 然后做 allgather
if full_dycp_prefill:
    k_c_normed, k_pe = pcp_kv_allgather_and_restore(...)
```

这是一个关键的设计选择：KV cache 只存储本 rank 负责的部分，allgather 的结果只用于当前 step 的 attention 计算。

**(g) Decode LSE allreduce 修复（L3146-3160）**

```python
decode_dycp_reqs = min(attn_metadata.num_dycp_reqs, attn_metadata.num_decodes)
```

只对 decode 阶段的 DyCP 请求做 LSE allreduce，避免对 prefill 请求重复操作。

### 3.4 `vllm/v1/attention/backends/utils.py` (+113 -25)

**改动摘要**：

| 改动 | 说明 |
|------|------|
| `dcp_local_seq_lens` / `dcp_local_seq_lens_cpu` 别名 | `CommonAttentionMetadata` 中添加向后兼容字段 |
| `num_dycp_tokens` 字段 | 传递 DyCP token 总数 |
| `_slice()` 传播 DyCP 字段 | 确保 ubatch 切片时保留 DyCP 元数据 |
| `get_dcp_local_seq_lens()` | 向后兼容包装函数 |
| `pcp_kv_allgather_and_restore()` 健壮化 | 添加 bounds checking、空 tensor 处理、restore_idx padding/truncation |
| `get_pcp_query_indices()` 重写 | 替换不正确的 `get_pcp_part_indices` 调用，显式处理奇数长度序列 |
| `reorder_batch_to_split_cp_and_normal()` 清理 | 格式化和注释整理 |

**`pcp_kv_allgather_and_restore` 关键改动**：

```python
# 长度对齐：restore_idx 可能因动态 batch 而与 gathered KV 大小不匹配
if restore_idx.numel() > expected_tokens:
    restore_idx = restore_idx[:expected_tokens]
elif restore_idx.numel() < expected_tokens:
    pad = torch.arange(pad_start, expected_tokens, ...)
    restore_idx = torch.cat([restore_idx, pad], dim=0)
restore_idx = torch.clamp(restore_idx, 0, expected_tokens - 1)
```

**`get_pcp_query_indices` 重写原因**：

原实现通过 `get_pcp_part_indices(cu_num_tokens, 1, 2)` 计算 head/tail，但 `get_pcp_part_indices` 使用整除可能导致奇数长度序列丢失 token。新实现：
- `head_len = floor(len/2)`
- `tail_len = len - head_len`（保留所有 token）

### 3.5 `vllm/v1/core/cross_dp_kv_cache_manager.py` (+15 -46)

**改动摘要**：完全重写 `_calculate_local_seq_lens`。

**重写前（~40 行）**：复杂的 interleave-based 计算，手动分配 remainder 到各 rank。

**重写后（~10 行）**：
```python
num_padded_tokens = cdiv(seq_len, 2 * world_size) * (2 * world_size)
local_seq_len = num_padded_tokens // world_size
return [local_seq_len for _ in range(world_size)]
```

**设计意图**：与 worker 端 `PCPManager.update_tokens_for_pcp` 的 padding 策略对齐。DualChunkSwap 要求 token 数为 `2 * world_size` 的倍数，padding 后等分到各 rank。调度器端的 block 分配必须使用相同的公式，否则 slot_mapping 会不一致。

### 3.6 `vllm/v1/core/kv_cache_utils.py` (+10 -6)

使用 `get_cp_kv_cache_world_size()` 替代硬编码的 `pcp_size * dcp_size`，纳入 `dycp_size` 维度。日志也相应更新。

### 3.7 `vllm/v1/core/sched/cross_dp_scheduler.py` (+70 -42)

**核心改动**：从全局 `token_budget` 升级为 **per-rank `rank_budgets`**。

**(a) Per-rank budget 管理**

```python
rank_budgets = [self.max_num_scheduled_tokens] * self.cp_world_size

def _get_effective_budget(cp_ranks):
    if len(cp_ranks) > 1:  # CP request
        return min(rank_budgets[r] for r in cp_ranks) * cp_size
    return rank_budgets[cp_ranks[0]]  # DP request

def _deduct_budget(cp_ranks, num_tokens):
    per_rank_cost = ceil(num_tokens / cp_size)
    for r in cp_ranks:
        rank_budgets[r] -= per_rank_cost
```

CP 请求的 token 被均分到所有 CP rank，因此实际可调度的 token 数取决于所有相关 rank 中 budget 最小的那个（min × cp_size）。

**(b) `select_dp` 增加 budget 感知**

```python
def select_dp(self, request, is_long, rank_budgets=None):
    if rank_budgets is not None:
        candidates = [i for i in range(cp_world_size)
                      if num_req_per_dp[i] < max_num_seqs and rank_budgets[i] > 0]
        best_dp = max(candidates, key=lambda i: rank_budgets[i])
```

选择剩余 budget 最多的 rank，平衡各 rank 的利用率。

**(c) Waiting queue 调度改进**

CP 请求超出 budget 时不再直接 break，而是 skip 并继续查找较短的 DP 请求：
```python
if not chunked_prefill and num_new_tokens > effective_budget:
    if len(selected_dp) > 1:  # CP request
        skipped_waiting_requests.prepend_request(request)
        continue  # 跳过，继续调度
    break  # DP request，停止调度
```

**(d) 验证改为 per-rank**

```python
for idx in range(self.cp_world_size):
    effective_rank_tokens = sum(tokens/cp_size for each request on rank)
    assert effective_rank_tokens <= self.max_num_scheduled_tokens
```

### 3.8 `vllm/v1/engine/core.py` (+6)

添加 `START_DP_WAVE` 消息类型的 no-op 处理。非 DP core 可能收到此控制消息（通过共享路径），需要安全忽略。

### 3.9 `vllm/v1/kv_cache_interface.py` (+21 -6)

**新增两个全局辅助函数**：

```python
def get_cp_kv_cache_world_size(vllm_config) -> int:
    return dcp_size * pcp_size * dycp_size

def get_cp_kv_cache_model_len(vllm_config) -> int:
    return cdiv(max_model_len, cp_world_size)
```

`FullAttentionSpec.max_memory_usage_bytes` 改用这两个函数，确保 KV cache 内存计算纳入 DyCP 维度。

### 3.10 `vllm/v1/worker/block_table.py` (+30 -20)

**(a) `total_cp_world_size` 计算**

```python
self.total_cp_world_size = pcp_world_size * dcp_world_size * dycp_world_size
self.total_cp_rank = (dycp_rank * pcp_world_size + pcp_rank) * dcp_world_size + dcp_rank
```

**(b) `compute_domain_slot_mapping` 修改**

DyCP 请求使用 `total_cp_world_size` 进行 interleaved slot mapping 计算，DP 请求使用简单的直接映射。双路径通过 `dycp_mask = req_indices < num_dycp_reqs` 分离。

**(c) `MultiGroupBlockTable`**

`total_cp_world_size` 的计算也纳入了 `dycp_world_size`。

### 3.11 `vllm/v1/worker/cp_utils.py` (+418 -2)

**新增完整的 DyCP 版 `PCPManager` 类**（~400 行）。这是对原 PCP `PCPManager` 的 fork，关键差异：

| 方法 | 差异 |
|------|------|
| `__init__` | 接受 `max_pre_division_tokens` 参数，预分配更大的 buffer |
| `update_tokens_for_pcp` | 逻辑相同，但面向 DyCP 的 world_size |
| `get_dycp_restore_slot_mapping` | 使用 `get_dycp_group()` 替代 `get_pcp_group()` |
| `get_dycp_restore_hidden_states` | 使用 `get_dycp_group()` 替代 `get_pcp_group()` |
| `get_padded_slot_mapping` | 增加多种 mismatch fallback 处理 |

**设计考量**：虽然两个 PCPManager 类有大量重复代码，但因为它们使用不同的通信分组（`get_pcp_group` vs `get_dycp_group`），且 DyCP 版本有额外的 buffer sizing 需求和 fallback 逻辑，fork 是当前最清晰的实现方式。

### 3.12 `vllm/v1/worker/gpu_model_runner.py` (+270 -113)

**改动量第二大的文件**，涉及 runner 生命周期的多个阶段：

#### 3.12.1 初始化阶段

**(a) cp_world_size 覆写（L326-330）**

```python
if self.dycp_world_size > 1:
    self.cp_world_size = self.dycp_world_size
    self.cp_rank = self.dycp_rank
```

在 DyCP 模式下，runner 层面的 `cp_world_size` 直接等于 `dycp_world_size`，使下游代码（如 `get_cp_local_seq_lens`）自动使用正确的分片参数。

**(b) Buffer sizing（L441-470, L508-567）**

```python
# 后分片 buffer（GPU tensor 大小）
if self.dycp_world_size > 1:
    max_buffer_num_tokens = max_num_tokens + max_num_reqs * 2 * dycp_world_size

# 前分片 buffer（PCPManager 内部 + arange_np）
max_pre_div_tokens = num_cp_seqs * (max_model_len + 2 * dycp_world_size)
```

DyCP 需要两种 buffer 大小：
- **后分片**（post-division）：padding 后经 DualChunkSwap 分片给当前 rank 的 token 数
- **前分片**（pre-division）：分片前的完整 token 数，用于 PCPManager 内部的 restore index 和 unpad mask

**(c) PCPManager 初始化（L568-580）**

```python
elif self.dycp_world_size > 1:
    self.pcp_manager = PCPManager(
        self.dycp_world_size, self.dycp_rank,
        max_buffer_num_tokens, self.max_num_reqs,
        self.device, self.pin_memory,
        max_pre_division_tokens=max_pre_div_tokens,
    )
```

**(d) `cp_local_seq_lens` buffer 合并（L526-536）**

原来 DyCP 有独立的 `dycp_local_seq_lens` buffer，现改为复用 `cp_local_seq_lens`（条件 `cp_world_size > 1 or dycp_world_size > 1`）。

#### 3.12.2 `prepare_inputs` 阶段

**(a) DyCP 跳过初始 position 计算（L1419-1432）**

当 DyCP 有 CP 请求时，pre-division token 数可能超出 post-division GPU buffer 大小。跳过初始的 positions/req_indices 计算，在 DyCP 分支中用 post-division 值重新计算。

**(b) DyCP token 分片（L1474-1531）**

```python
elif self.dycp_world_size > 1:
    if num_cp_request > 0:
        # 只对 CP 请求做 DualChunkSwap 分片
        num_scheduled_tokens[:num_cp_request], pcp_positions = (
            self.pcp_manager.update_tokens_for_pcp(...)
        )
        # CP 请求用 pcp_positions，DP 请求用标准 arange
        positions_np[:total_num_pcp_scheduled_tokens] = pcp_positions + num_computed_tokens
        positions_np[dp_start:] = num_computed_tokens + arange[dp_start:]
```

**(c) DyCP discard_request_mask（L1639-1660）**

DyCP 和 DP 请求使用不同的 mask 计算逻辑：
- DyCP 请求：考虑 PCP padding (`num_pcp_pads_cpu`)
- DP 请求：标准的 `seq_lens < num_tokens`

**(d) DyCP logits_indices（L1693-1708）**

DyCP 请求的 logits_indices 通过 `pcp_manager.get_logits_indices()` 计算。DP 请求的 indices 需要加上 allgathered DyCP 部分的偏移量：

```python
dycp_allgathered_size = cu_num_tokens[num_dycp_reqs - 1] * self.dycp_world_size
logits_indices[num_dycp_reqs:] += (dycp_allgathered_size - num_dycp_tokens)
```

#### 3.12.3 `_build_attention_metadata` 阶段

**(a) Slot mapping 处理（L1826-1870）**

```python
elif self.dycp_world_size > 1 and num_dycp_tokens > 0:
    # Keep DYCP slot mapping local.
    local_dycp_slot_mapping = slot_mapping[:num_dycp_tokens]
    non_dycp_slot_mapping = slot_mapping[num_dycp_tokens:]
    slot_mapping = torch.cat([local_dycp_slot_mapping, non_dycp_slot_mapping])
```

DyCP 的 slot_mapping 保持本地（不做 allgather），因为 KV cache 更新只写本 rank 负责的部分。

**(b) cp_local_seq_lens 计算（L1888-1941）**

DyCP 请求和 DP 请求使用不同的 local_seq_lens：
- DyCP 请求：通过 `get_cp_local_seq_lens(seq_lens, dycp_world_size, dycp_rank, ...)` 计算
- DP 请求：直接复制 `seq_lens`（不做 CP 分片）

结果统一存入 `cp_local_seq_lens`，并同时设置 `dcp_local_seq_lens` 和 `dycp_local_seq_lens` 别名。

**(c) pcp_allgather_restore_idx（L1923-1943）**

DyCP 模式下 restore_idx 大小为 `num_dycp_tokens * dycp_world_size`（只对 DyCP token 做 allgather）。

#### 3.12.4 `_execute_model_common` 阶段

**(a) max_num_scheduled_tokens 计算（L3319-3337）**

DyCP 和 DP 请求分别计算 max_num_scheduled_tokens：
```python
max_dycp_tokens = int(dycp_tokens.max())
max_non_dycp_tokens = int(non_dycp_tokens.max())
max_num_scheduled_tokens = max(max_dycp_tokens, max_non_dycp_tokens)
```

**(b) Hidden states restore（L3473-3488）**

```python
elif self.dycp_world_size > 1 and num_cp_request > 0:
    dycp_hidden_states = self.pcp_manager.get_dycp_restore_hidden_states(
        hidden_states[:num_dycp_tokens_unpadded], num_dycp_tokens_unpadded
    )
    hidden_states = torch.cat([dycp_hidden_states, non_dycp_hidden_states])
```

DyCP 部分的 hidden states 经 allgather + restore 恢复完整序列顺序，然后与 DP 部分拼接。

#### 3.12.5 Profile / CUDA Graph 阶段

**(a) `_get_profile_num_tokens()` 方法（L4812-4820）**

```python
def _get_profile_num_tokens(self):
    if self.dycp_world_size > 1:
        return self.max_num_tokens  # 不除以 world_size
    if self.pcp_world_size > 1:
        return cdiv(self.max_num_tokens, self.pcp_world_size)
```

DyCP 模式下 profile 使用完整的 `max_num_tokens`，因为短请求可能不经过 CP 分片。

**(b) `get_dycp_allgather_reserve_bytes()` 方法（L4515-4560）**

计算 DyCP KV allgather 所需的额外显存：
```python
pre_div_tokens = num_cp_seqs * max_model_len
extra_tokens = max(0, pre_div_tokens - max_num_tokens)
return extra_tokens * head_size * dtype_bytes
```

### 3.13 `vllm/v1/worker/gpu_worker.py` (+9)

在 KV cache 内存分配前，预留 DyCP allgather 临时 buffer 所需的显存：

```python
dycp_reserve = self.model_runner.get_dycp_allgather_reserve_bytes()
if dycp_reserve > 0:
    self.available_kv_cache_memory_bytes -= dycp_reserve
```

---

## 4. 通信模式汇总

| 通信操作 | 阶段 | 分组 | 频率 | 同步类型 |
|---------|------|------|------|---------|
| `all_reduce(MAX)` | `_build_prefill` 元数据同步 | `get_dycp_group()` | 每 batch 1 次 | CPU-GPU sync (.item()) |
| `all_gather` KV | `forward_common` decode/prefill | `get_dycp_group()` | 每层 1 次 | GPU-only |
| `all_gather` chunked context KV | `_context_parallel_compute_prefill_context` | `get_dycp_group()` | 每层 × num_chunks | GPU-only |
| `cp_lse_ag_out_ar` | `forward_common` decode | `get_dycp_group()` | 每层 1 次（有 decode 时） | GPU-only |
| `all_gather` hidden_states | `_execute_model_common` 输出恢复 | `get_dycp_group()` | 每 step 1 次 | GPU-only |

---

## 5. 关键正确性约束

1. **所有 DyCP rank 必须同时进入或跳过集合通信**：通过 `_build_prefill` 中的 `all_reduce` 同步和 `_forward_prefill` 中的 override 参数保证。

2. **Chunked context 元数据形状必须跨 rank 一致**：`max_context_len_cpu` 和 `num_prefills_with_context_cpu` 取 MAX 值确保。

3. **KV cache 写入使用本地 slot_mapping**：`concat_and_cache_mla` 在 allgather 前执行，使用未 gather 的 slot_mapping。

4. **Batch 排序约束**：DyCP 请求在 DP 请求前面（`reorder_batch_to_split_cp_and_normal`）。

5. **Buffer 大小约束**：post-division buffer 用于 GPU tensor，pre-division buffer 用于 PCPManager 内部状态。

---

## 6. 已知限制与未来优化

1. **PCPManager 代码重复**：DyCP 版本几乎是 PCP 版本的 fork，仅通信分组不同。可以通过参数化通信分组来合并。

2. **mixed batch 拆分开销**：`split_metadata()` 每层执行一次（在 `forward()` 中），涉及大量 tensor 切片操作。可以考虑缓存拆分结果。

3. **强制 prefill 分类**：当前 DyCP batch 中所有请求都被强制归为 prefill，即使实际只有 1 个 token。这可能影响 decode-heavy workload 的性能。

4. **chunked_prefill_workspace_size 硬编码**：从 `64 * 1024` 改为 `1048576`（16x 增大），这个值可能需要根据实际 model_len 和 head_size 动态调整。

5. **all_reduce 同步**：虽然已从每层 2 次优化到每 batch 1 次，但仍涉及 CPU-GPU 同步（`.item()`）。如果能找到不依赖 CPU 值的方式构建 chunked_context 元数据，可以完全消除此同步。

---

## 7. 测试建议

| 场景 | 验证要点 |
|------|---------|
| 纯 DyCP prefill（全 CP 请求） | KV allgather + DualChunkSwap 正确性 |
| 混合 batch（DyCP + DP） | split_metadata 拆分正确性，两 phase 独立执行 |
| DyCP + chunked prefill | 跨 rank 元数据一致性，无 NCCL hang |
| DyCP decode | LSE allreduce 正确，输出合并正确 |
| 奇数长度序列 | DualChunkSwap head/tail 分片不丢失 token |
| 单请求 batch | 边界情况：空 DyCP 或空 DP 子 batch |
| Per-rank budget 调度 | CP 请求和 DP 请求的 budget 分配均衡 |
