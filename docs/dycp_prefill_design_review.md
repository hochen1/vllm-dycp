# DyCP Prefill for MLA — 设计报告

> **Commit**: `a31bda5a9` — `[feat] support base prefill dycp for mla`
> **基线**: `c1ea0cfc9` — `[feat] support base pcp`
> **改动规模**: 15 files changed, +2541 -309 lines

---

## 1. 概述

本次提交在 vLLM 的 MLA（Multi-head Latent Attention）注意力后端中实现了 **DyCP（Dynamic Context Parallelism）** 的 prefill 支持。DyCP 是一种 CP + DP 的混合并行模式：由 Cross-DP Scheduler 选出的长序列请求（CP 请求）被分片到所有 DP rank 上协作处理 prefill，而短序列请求（DP 请求）保留在各自 rank 本地独立执行。

### 1.1 与纯 PCP 的关键区别

| 维度 | PCP（Prefill Context Parallelism） | DyCP（Dynamic Context Parallelism） |
|------|-------------------------------------|--------------------------------------|
| 请求同质性 | 所有 rank 处理相同的请求集 | 各 rank 可混合 CP 请求和本地 DP 请求 |
| 通信分组 | `get_pcp_group()` | `get_dycp_group()` |
| `pcp_world_size` | > 1 | = 1（不使用 PCP 分组） |
| 元数据一致性 | 天然一致 | DyCP 请求由 cross-DP scheduler 统一下发，天然一致；DP 请求各 rank 不同 |
| Batch 结构 | 同构 prefill batch | 混合 batch：`[DyCP reqs..., DP reqs...]` |
| 混合 batch 处理 | 不涉及 | 在 `build()` 中预构建独立的 `_dycp_split` / `_dp_split` metadata |

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
            │   build() → forward()         │
            │                               │
            │  ┌──────────────────────────┐  │
            │  │ Mixed batch?             │  │
            │  │  YES → _build_mixed_     │  │
            │  │    dycp_dp_prefill()     │  │
            │  │    → _dycp_split         │  │
            │  │    → _dp_split           │  │
            │  │  NO → build standard     │  │
            │  └──────────────────────────┘  │
            │                               │
            │  forward():                   │
            │  ┌──────────────────────────┐  │
            │  │ _dycp_split exists?      │  │
            │  │  YES → Phase A: CP fwd   │  │
            │  │        Phase B: DP fwd   │  │
            │  │  NO  → forward_common()  │  │
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

1. **Batch 预拆分策略**：混合 batch（DyCP + DP 请求共存）在 `build()` 阶段即预构建两个独立的 `MLACommonMetadata` 对象（`_dycp_split` / `_dp_split`），存储在 `attn_metadata` 上。`forward()` 直接使用预构建的 metadata，避免每层执行 tensor 切片。

2. **复用 PCP 基础设施**：DyCP 的 DualChunkSwap 分片、token 分配、restore index 等逻辑与 PCP 相同，通过独立的 `PCPManager` 实例 + `get_dycp_group()` 通信分组实现复用。

3. **独立 chunk 构建，无 all_reduce**：DyCP 请求由 cross-DP scheduler 统一下发，所有 rank 收到相同的请求（相同 `context_lens`），元数据天然一致。混合 batch 中 DyCP 和 DP 请求分别构建 chunked_context，各自独占 workspace，避免 chunk 大小被稀释。

4. **KV Cache 写入与计算分离**：KV cache 的 `concat_and_cache_mla` 始终使用本地 slot_mapping（只写本 rank 负责的 KV），而 attention 计算使用 allgathered 后的完整 KV。

---

## 3. 逐文件详细分析

### 3.1 `vllm/v1/core/sched/cross_dp_scheduler.py`（+112 -19）

**核心改动**：从全局 `token_budget` 升级为 **per-rank `rank_budgets`**。

#### 3.1.1 Per-rank budget 管理

```python
rank_budgets = [self.max_num_scheduled_tokens] * self.cp_world_size

def _get_effective_budget(cp_ranks: list[int]) -> int:
    """CP 请求的可用 budget = min(所有相关 rank budget) × cp_size"""
    cp_size = len(cp_ranks)
    if cp_size > 1:
        return min(rank_budgets[r] for r in cp_ranks) * cp_size
    return rank_budgets[cp_ranks[0]]

def _deduct_budget(cp_ranks: list[int], num_tokens: int) -> None:
    """按 per-rank cost = ceil(tokens / cp_size) 扣减"""
    cp_size = len(cp_ranks)
    per_rank_cost = (num_tokens + cp_size - 1) // cp_size
    for r in cp_ranks:
        rank_budgets[r] -= per_rank_cost
```

**设计意图**：CP 请求的 token 被均分到所有 CP rank，因此可调度的 token 数取决于所有相关 rank 中 budget 最小的那个。单独的 DP 请求只消耗目标 rank 的 budget。

#### 3.1.2 `select_dp` 增加 budget 感知

新增 `rank_budgets` 参数。短请求选择剩余 budget 最多 **且** seq slot 可用的 rank，平衡各 rank 利用率：

```python
candidates = [i for i in range(self.cp_world_size)
              if self.num_req_per_dp[i] < self.max_num_seqs
              and rank_budgets[i] > 0]
best_dp = max(candidates, key=lambda i: rank_budgets[i])
```

#### 3.1.3 Waiting queue 调度改进

CP 请求超出 budget 时不再直接 `break`，而是 skip 并继续查找较短的 DP 请求：

```python
if not chunked_prefill and num_new_tokens > effective_budget:
    if len(selected_dp) > 1:  # CP request
        skipped_waiting_requests.prepend_request(request)
        continue  # 跳过，继续调度
    break  # DP request，停止调度
```

#### 3.1.4 验证改为 per-rank

```python
for idx in range(self.cp_world_size):
    effective_rank_tokens = sum(
        (tokens + cp_size - 1) // cp_size
        for each request on rank
    )
    assert effective_rank_tokens <= self.max_num_scheduled_tokens
```

### 3.2 `vllm/v1/worker/gpu_model_runner.py`（+383 -78）

改动量第二大的文件，涉及 runner 生命周期的多个阶段。

#### 3.2.1 初始化阶段

**(a) `cp_world_size` 覆写**

```python
if self.dycp_world_size > 1:
    self.cp_world_size = self.dycp_world_size
    self.cp_rank = self.dycp_rank
```

DyCP 模式下 `pcp_world_size = dcp_world_size = 1`，但 runner 层面需要通过 `cp_world_size` 驱动下游 `get_cp_local_seq_lens` 等函数，因此直接覆写为 `dycp_world_size`。

**(b) Buffer sizing**

DyCP 需要两种 buffer 大小：

| Buffer 类型 | 用途 | 大小计算 |
|------------|------|---------|
| 后分片（post-division） | GPU tensor（positions、slot_mapping 等） | `max_num_tokens + max_num_reqs × 2 × dycp_world_size` |
| 前分片（pre-division） | PCPManager 内部 restore index、unpad mask | `num_cp_seqs × (max_model_len + 2 × dycp_world_size)` |

`InputBatch` 的 `max_num_batched_tokens` 使用后分片大小，确保 `BlockTable.slot_mapping` 足够大。

**(c) PCPManager 初始化**

```python
elif self.dycp_world_size > 1:
    self.pcp_manager = PCPManager(
        self.dycp_world_size, self.dycp_rank,
        max_buffer_num_tokens, self.max_num_reqs,
        self.device, self.pin_memory,
        max_pre_division_tokens=max_pre_div_tokens,
    )
```

**(d) `cp_local_seq_lens` buffer 合并**

原来 DyCP 有独立的 `dycp_local_seq_lens` buffer，现改为复用 `cp_local_seq_lens`（条件 `cp_world_size > 1 or dycp_world_size > 1`），同时设置 `dcp_local_seq_lens` 和 `dycp_local_seq_lens` 别名。

#### 3.2.2 `prepare_inputs` 阶段

**(a) DyCP 跳过初始 position 计算**

当 DyCP 有 CP 请求时，pre-division token 数可能超出 post-division GPU buffer 大小。跳过初始的 `positions`/`req_indices` 计算，在 DyCP 分支中用 post-division 值重新计算：

```python
if not (self.dycp_world_size > 1 and scheduler_output.num_cp_request > 0):
    req_indices = np.repeat(...)
    cu_num_tokens, arange = self._get_cumsum_and_arange(...)
    positions_np = ...
```

**(b) DyCP token 分片**

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

**(c) DyCP `discard_request_mask`**

DyCP 和 DP 请求使用不同的 mask 计算逻辑：
- DyCP 请求：考虑 PCP padding（`num_pcp_pads_cpu`）
- DP 请求：标准的 `seq_lens < num_tokens`

**(d) DyCP `logits_indices`**

DyCP 请求的 logits_indices 通过 `pcp_manager.get_logits_indices()` 计算。DP 请求的 indices 需要加上 allgathered DyCP 部分的偏移量：

```python
dycp_allgathered_size = cu_num_tokens[num_dycp_reqs - 1] * self.dycp_world_size
logits_indices[num_dycp_reqs:] += (dycp_allgathered_size - num_dycp_tokens)
```

#### 3.2.3 `_build_attention_metadata` 阶段

**(a) Slot mapping 处理**

DyCP 的 slot_mapping 保持本地（不做 allgather），因为 KV cache 更新只写本 rank 负责的部分：

```python
elif self.dycp_world_size > 1 and num_dycp_tokens > 0:
    local_dycp_slot_mapping = slot_mapping[:num_dycp_tokens]
    non_dycp_slot_mapping = slot_mapping[num_dycp_tokens:]
    slot_mapping = torch.cat([local_dycp_slot_mapping, non_dycp_slot_mapping])
```

PCP 路径新增 `get_restore_slot_mapping()` 调用，对 slot_mapping 做 allgather + restore 后再 padding，替代原来直接传入本地 slot_mapping 的方式。

**(b) `cp_local_seq_lens` 计算**

DyCP 请求和 DP 请求使用不同的 `local_seq_lens`：
- DyCP 请求：`get_cp_local_seq_lens(seq_lens, dycp_world_size, dycp_rank, ...)`
- DP 请求：直接复制 `seq_lens`（不做 CP 分片）

**(c) `pcp_allgather_restore_idx`**

DyCP 模式下 `restore_idx` 大小为 `num_dycp_tokens × dycp_world_size`（只对 DyCP token 做 allgather）。

#### 3.2.4 `_execute_model_common` 阶段

**(a) `max_num_scheduled_tokens` 计算**

DyCP 和 DP 请求分别计算 `max_num_scheduled_tokens`，取两者最大值。

**(b) Hidden states restore**

DyCP 部分的 hidden states 经 `get_dycp_restore_hidden_states`（allgather + restore）恢复完整序列顺序，然后与 DP 部分拼接：

```python
elif self.dycp_world_size > 1 and num_cp_request > 0:
    dycp_hidden_states = self.pcp_manager.get_dycp_restore_hidden_states(
        hidden_states[:num_dycp_tokens_unpadded], num_dycp_tokens_unpadded
    )
    hidden_states = torch.cat([dycp_hidden_states, non_dycp_hidden_states])
```

#### 3.2.5 Profile / Memory 阶段

**(a) `_get_profile_num_tokens()` 方法**

DyCP 模式下 profile 使用完整的 `max_num_tokens`（不除以 world_size），因为短请求可能不经过 CP 分片。

**(b) `get_dycp_allgather_reserve_bytes()` 方法**

计算 DyCP KV allgather 所需的额外显存：

```python
pre_div_tokens = num_cp_seqs * max_model_len
extra_tokens = max(0, pre_div_tokens - max_num_tokens)
return extra_tokens * head_size * dtype_bytes
```

### 3.3 `vllm/v1/attention/backends/mla/common.py`（+1242 -55）

这是本次提交的核心文件，改动量最大。按功能模块逐一分析。

#### 3.3.1 元数据字段扩展

| 类 | 新增字段 | 用途 |
|----|---------|------|
| `MLACommonPrefillMetadata` | `num_dycp_reqs: int = 0` | prefill 部分中 DyCP 请求数 |
| `MLACommonMetadata` | `num_dycp_tokens: int = 0` | DyCP 请求的 token 总数 |
| `MLACommonMetadata` | `_dycp_split: MLACommonMetadata \| None` | 预构建的 DyCP 子 metadata |
| `MLACommonMetadata` | `_dp_split: MLACommonMetadata \| None` | 预构建的 DP 子 metadata |

#### 3.3.2 `MLACommonMetadataBuilder` 修改

**(a) `cp_virtual_block_size` 计算**

DyCP 模式下 `cp_world_size = 1`，但实际 KV 分片跨 `dycp_world_size` 个 rank，因此 `virtual_block_size` 需要用 `dycp_world_size`：

```python
if self.dycp_world_size > 1:
    self.cp_virtual_block_size = self.cp_local_block_size * self.dycp_world_size
```

**(b) Chunked prefill workspace 扩大**

- workspace 基础大小从 `64 * 1024` 增大到 `1048576`（16×）
- DyCP 模式下独立的 `elif` 分支，按 `dycp_world_size` 计算 workspace 扩展量（用于 KV allgather 后的临时存储）

**(c) `reorder_batch_threshold` 强制为 1**

FlashMLA 默认阈值为 128，即 token 数 ≤ 128 的请求会被归类为 "decode"。但 DyCP 的 `update_tokens_for_pcp` 假设 decode 请求在 CP 请求前面（有序），而 `reorder_batch_to_split_cp_and_normal` 不保证这一点。强制阈值为 1 使所有多 token 请求都走 prefill 路径。

**(d) DyCP 强制 prefill 分类**

```python
if self.dycp_world_size > 1 and num_dycp_reqs > 0:
    num_decodes = 0
    num_prefills = num_reqs
    num_prefill_tokens = num_tokens
```

DyCP batch 中，CP 请求经 `update_tokens_for_pcp` 分片后各 rank 本地 query 长度可能不同。如果用本地 query 长度判断 decode/prefill 分类，不同 rank 可能得到不同结果 → 条件分支分歧 → NCCL 死锁。强制所有请求走 prefill 路径。

#### 3.3.3 `_build_mixed_dycp_dp_prefill()` 方法（新增 ~250 行）

**这是本次提交最核心的新增方法**。当检测到混合 batch（`0 < prefill_num_dycp_reqs < num_prefills`）时，在 `build()` 中调用此方法，分别为 DyCP 和 DP 请求构建独立的 `MLAPrefillMetadata`。

```python
def _build_mixed_dycp_dp_prefill(self, ...):
    # ===== DyCP chunked context (CP local layout) =====
    # 1. DyCP 独占全部 workspace → 更大的 chunk → 更少的 allgather 轮次
    dycp_max_context_chunk = self.chunked_prefill_workspace_size // dycp_n_with_ctx
    # 2. 计算 CP local layout (via get_cp_local_seq_lens)
    # 3. 构建 DyCP 的 ChunkedContextMetadata
    # 4. 构建 DyCP 的 PCPMetadata (query/kv head/tail indices)

    # ===== DP chunked context (standard layout, no CP) =====
    # 1. DP 也独占 workspace（与 DyCP 顺序执行，不冲突）
    dp_max_context_chunk = self.chunked_prefill_workspace_size // dp_n_with_ctx
    # 2. 标准 chunk 构建，无 CP fields

    return dycp_prefill_metadata, dp_prefill_metadata
```

**关键设计**：
- DyCP 和 DP 各自独占 workspace → chunk 大小不被对方稀释
- DyCP 请求由 cross-DP scheduler 统一下发，所有 rank 天然一致 → 无需 `all_reduce`
- 两个 phase 顺序执行，共享同一个 workspace tensor，不存在冲突

#### 3.3.4 `build()` 中预构建拆分 metadata

在 `build()` 末尾，当检测到混合 batch 时，构建两个完整的 `MLACommonMetadata`：

```python
if is_dycp_mixed and dp_prefill_metadata is not None:
    dycp_token_end = int(query_start_loc_cpu[num_decodes + n_dycp_prefill] - ...)

    attn_metadata._dycp_split = self.metadata_cls(
        num_reqs=n_dycp_prefill,
        num_actual_tokens=dycp_token_end,
        slot_mapping=slot_mapping[:dycp_token_end],
        prefill=dycp_prefill_metadata,
        pcp_allgather_restore_idx=pcp_allgather_restore_idx,
        num_dycp_reqs=n_dycp_prefill,
        num_dycp_tokens=dycp_token_end,
        ...
    )

    attn_metadata._dp_split = self.metadata_cls(
        num_reqs=num_prefills - n_dycp_prefill,
        num_actual_tokens=dp_token_num,
        slot_mapping=slot_mapping[dycp_token_end:num_tokens],
        prefill=dp_prefill_metadata,
        pcp_allgather_restore_idx=None,
        num_dycp_reqs=0,
        num_dycp_tokens=0,
        ...
    )
```

#### 3.3.5 纯 DyCP batch 的 chunk 和 PCP 构建

当所有 prefill 都是 DyCP 请求（`prefill_num_dycp_reqs == num_prefills`）时，走 `elif self.dycp_world_size > 1 and prefill_num_dycp_reqs > 0` 分支：

- 计算 DyCP 本地 context layout（`dycp_local_context_lens_allranks`、`dycp_padded_local_chunk_seq_lens`）
- 构建 `ChunkedContextMetadata` with CP fields（`padded_local_cu_seq_lens`、`local_context_lens_allranks`、`chunk_size`）
- 构建 `PCPMetadata`（query/kv head/tail indices）使用 `dycp_rank`/`dycp_world_size`

#### 3.3.6 `MLACommonImpl` 修改

**(a) `_run_prefill_new_tokens_fa` 重构**

新增辅助函数：
- `_safe_index()`：bounds-safe 的索引 clamp，防止 CUDA gather OOB
- `_build_cu_seq_lens()` / `_max_seq_len()`：精确的 per-request cu_seqlens
- `_run_dual_chunk_attn()`：封装 DualChunkSwap 的单次 attention 调用

PCP 路径重写：使用精确的 per-request `q_head_seq_lens` / `q_tail_seq_lens` 替代原来不正确的 `prefill.query_start_loc // 2`（对奇数长度序列不正确）。

新增 DyCP `elif` 分支：逻辑与 PCP 相同，但使用 `dycp_rank` / `dycp_world_size`，且有额外的 gate 条件：
- `prefill.num_dycp_reqs == block_table.shape[0]`（纯 DyCP batch）
- `k.shape[0] > q.shape[0]`（KV 已 allgathered）

**(b) `_compute_prefill_context` 和 `_context_parallel_compute_prefill_context` 健壮性增强**

- 添加 `toks == 0`、`sum_seq_len == 0` 的 skip 检查
- `reorg_kvcache` 添加空 segment 的 fallback 处理
- 两个函数末尾添加 `output is None` 的零初始化兜底

**(c) Chunked context allgather 路径扩展**

```python
if self.pcp_world_size > 1:
    gathered = get_pcp_group().all_gather(...)
elif self.dycp_world_size > 1 and attn_metadata.num_dycp_reqs > 0:
    gathered = get_dycp_group().all_gather(...)
else:
    gathered = get_dcp_group().all_gather(...)
```

**(d) `forward()` / `forward_common()` 拆分**

原始的 `forward()` 被拆分为：

- **`forward()`**：入口方法，检测混合 batch：
  - 如果 `attn_metadata._dycp_split is not None`（混合 batch）：
    - Phase A: 对 DyCP 子 batch 调用 `forward_common()`
    - Phase B: 对 DP 子 batch 调用 `forward_common()`
  - 否则：直接调用 `forward_common()`

- **`forward_common()`**：核心 attention 逻辑：
  1. 本地 KV cache 写入（`concat_and_cache_mla`）
  2. DyCP KV allgather（`pcp_kv_allgather_and_restore` + `get_dycp_group()`）
  3. DualChunkSwap prefill attention
  4. Chunked context allgather
  5. Decode LSE allreduce

**(e) Decode LSE allreduce 修复**

```python
decode_dycp_reqs = min(attn_metadata.num_dycp_reqs, attn_metadata.num_decodes)
```

只对 decode 阶段的 DyCP 请求做 LSE allreduce，避免对 prefill 请求重复操作。

### 3.4 `vllm/v1/attention/backends/utils.py`（+113 -27）

| 改动 | 说明 |
|------|------|
| `dcp_local_seq_lens` / `dcp_local_seq_lens_cpu` 别名 | `CommonAttentionMetadata` 中添加向后兼容字段 |
| `num_dycp_tokens` 字段 | 传递 DyCP token 总数 |
| `_slice()` 传播 DyCP 字段 | 确保 ubatch 切片时保留 DyCP 元数据 |
| `get_dcp_local_seq_lens()` | 向后兼容包装函数，委托给 `get_cp_local_seq_lens` |
| `pcp_kv_allgather_and_restore()` 健壮化 | bounds checking、空 tensor 处理、restore_idx padding/truncation |
| `get_pcp_query_indices()` 重写 | 修复奇数长度序列 token 丢失 |
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

**`get_pcp_query_indices` 重写原因**：原实现通过 `get_pcp_part_indices(cu_num_tokens, 1, 2)` 计算 head/tail，但整除可能导致奇数长度序列丢失 token。新实现：
- `head_len = floor(len/2)`
- `tail_len = len - head_len`（保留所有 token）

### 3.5 `vllm/v1/worker/cp_utils.py`（+149 -24）

`PCPManager` 类的 DyCP 扩展：

#### 新增参数和 buffer

- `__init__` 新增 `max_pre_division_tokens` 参数，预分配更大的 pre-division buffer（`pcp_allgather_restore_idx`、`pcp_unpad_mask_cpu_tensor`）
- 所有关键方法添加空输入守卫（`num_reqs == 0` 或 `len(num_scheduled_tokens) == 0`）

#### 新增方法

| 方法 | 用途 |
|------|------|
| `get_restore_slot_mapping()` | PCP slot_mapping allgather + restore（使用 `get_pcp_group()`） |
| `get_dycp_restore_slot_mapping()` | DyCP slot_mapping allgather + restore（使用 `get_dycp_group()`） |
| `get_dycp_restore_hidden_states()` | DyCP hidden states allgather + restore（使用 `get_dycp_group()`） |

#### 关键修复

- **Position clamping**：PCP padding 后的 token 可能超出原始序列范围。新增 `valid_pos_upper` clamp，防止下游 RoPE/index gather OOB
- **`get_padded_slot_mapping` 改进**：增加多种 mismatch fallback 处理（slot_mapping 可能是 compacted/gathered+padded 两种形状）
- **`get_logits_indices` 修复**：`num_pads` tensor 移到正确的 device
- **`num_pcp_pads_cpu` 切片修复**：添加 `[:num_reqs]` 保护

### 3.6 `vllm/v1/worker/block_table.py`（+50 -17）

**(a) `total_cp_world_size` 计算**

```python
self.total_cp_world_size = pcp_world_size * dcp_world_size * dycp_world_size
self.total_cp_rank = (dycp_rank * pcp_world_size + pcp_rank) * dcp_world_size + dcp_rank
```

**(b) `compute_slot_mapping()` 和 `compute_domain_slot_mapping()` 修改**

- `compute_slot_mapping()`：使用 `self.total_cp_world_size` / `self.total_cp_rank` 替代手动计算
- `compute_domain_slot_mapping()`：DyCP 请求使用 `total_cp_world_size` 进行 interleaved slot mapping 计算，DP 请求使用简单的直接映射。双路径通过 `dycp_mask = req_indices < num_dycp_reqs` 分离

**(c) `MultiGroupBlockTable`**

新增 `dycp_world_size` 获取，`total_cp_world_size = dcp_world_size * pcp_world_size * dycp_world_size`

### 3.7 `vllm/v1/core/cross_dp_kv_cache_manager.py`（+61 -67）

**核心改动**：完全重写 `_avg_distribute_tokens_to_ranks()`。

**重写前（~40 行）**：复杂的 interleave-based 计算，手动分配 remainder 到各 rank。

**重写后（~10 行）**：

```python
def _avg_distribute_tokens_to_ranks(self, world_size, seq_len, ...):
    if world_size <= 1:
        return [seq_len]
    # 与 PCPManager.update_tokens_for_pcp 的 padding 策略对齐
    num_padded_tokens = cdiv(seq_len, 2 * world_size) * (2 * world_size)
    local_seq_len = num_padded_tokens // world_size
    return [local_seq_len for _ in range(world_size)]
```

**设计意图**：DualChunkSwap 要求 token 数为 `2 × world_size` 的倍数，padding 后等分到各 rank。调度器端的 block 分配必须使用相同的公式，否则 slot_mapping 会不一致。

### 3.8 `vllm/v1/kv_cache_interface.py`（+27 -8）

新增两个全局辅助函数：

```python
def get_cp_kv_cache_world_size(vllm_config) -> int:
    return dcp_size * pcp_size * dycp_size

def get_cp_kv_cache_model_len(vllm_config) -> int:
    return cdiv(max_model_len, cp_world_size)
```

`FullAttentionSpec.max_memory_usage_bytes` 改用 `get_cp_kv_cache_model_len()`，确保 KV cache 内存计算纳入 DyCP 维度。消除了原来硬编码的 `dcp_world_size * pcp_world_size`。

### 3.9 `vllm/attention/backends/abstract.py`（+9 -2）

| 改动 | 说明 |
|------|------|
| 添加 `total_cp_world_size` / `total_cp_rank` 别名 | 向后兼容旧代码中引用的属性名 |
| `need_to_return_lse_for_decode` 条件修复 | 从 `self.cp_world_size > 1` 改为 `self.dcp_world_size > 1 or self.dycp_world_size > 1` |

**设计意图**：DyCP 模式下 `cp_world_size = pcp_world_size × dcp_world_size = 1 × 1 = 1`，但 decode 阶段仍需返回 LSE 用于跨 rank allreduce。原条件永远不会触发，修复后直接检查 `dcp_world_size` 和 `dycp_world_size`。

### 3.10 `vllm/model_executor/layers/fused_moe/config.py`（+16） & `layer.py`（+8）

FusedMoE 层感知 DyCP 并行维度：

- `FusedMoEParallelConfig` 新增 `dycp_size` / `dycp_rank` 字段
- 通过 `get_dycp_group()` 初始化（带 `try/except AssertionError` 回退）
- `FusedMoE` 类暴露 `dycp_size` / `dycp_rank` property

### 3.11 `vllm/v1/core/kv_cache_utils.py`（+10 -6）

使用 `get_cp_kv_cache_world_size()` 替代硬编码的 `pcp_size * dcp_size`，纳入 `dycp_size` 维度。日志格式更新为包含三个维度：

```python
"Multiplying the GPU KV cache size by the cp_world_size %d "
"(pcp_world_size %d * dcp_world_size %d * dycp_world_size %d)."
```

### 3.12 `vllm/v1/engine/core.py`（+6）

- `hash_block_size` 乘以 `dp_per_domain`，确保 DyCP 模式下 block hash 粒度正确
- 添加 `START_DP_WAVE` 消息类型的 no-op 处理（非 DP core 可能通过共享路径收到此控制消息）

### 3.13 `vllm/v1/worker/gpu_worker.py`（+9）

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
| `all_gather` KV | `forward_common` prefill | `get_dycp_group()` | 每层 1 次 | GPU-only |
| `all_gather` chunked context KV | `_context_parallel_compute_prefill_context` | `get_dycp_group()` | 每层 × num_chunks | GPU-only |
| `cp_lse_ag_out_ar` | `forward_common` decode | `get_dycp_group()` | 每层 1 次（有 decode 时） | GPU-only |
| `all_gather` hidden_states | `_execute_model_common` 输出恢复 | `get_dycp_group()` | 每 step 1 次 | GPU-only |

**注意**：相比传统方案，本实现 **不需要 `all_reduce(MAX)` 同步**。DyCP 请求由 cross-DP scheduler 统一下发，所有 rank 的 DyCP 元数据天然一致；混合 batch 中 DyCP 和 DP 的 chunked_context 独立构建，互不依赖。

---

## 5. 关键正确性约束

1. **所有 DyCP rank 必须同时进入或跳过集合通信**：通过强制 prefill 分类和 `_forward_prefill` 中的条件守卫保证。

2. **DyCP 元数据跨 rank 天然一致**：Cross-DP scheduler 下发相同的 DyCP 请求集，各 rank 的 `context_lens`、`num_prefills_with_context` 一致。DP 请求各 rank 不同但不参与集合通信。

3. **KV cache 写入使用本地 slot_mapping**：`concat_and_cache_mla` 在 allgather 前执行，使用未 gather 的 slot_mapping。

4. **Batch 排序约束**：DyCP 请求在 DP 请求前面（`reorder_batch_to_split_cp_and_normal`）。

5. **Buffer 大小约束**：post-division buffer 用于 GPU tensor，pre-division buffer 用于 PCPManager 内部状态。

6. **调度器-worker 一致性**：`_avg_distribute_tokens_to_ranks` 与 `PCPManager.update_tokens_for_pcp` 使用相同的 padding 公式（`cdiv(seq_len, 2 × ws) × (2 × ws)`）。

---

## 6. Workspace 共享策略

DyCP 和 DP 共享同一个 `self.chunked_prefill_workspace` tensor：

| Phase | 请求类型 | Workspace 使用 | chunk 大小 |
|-------|---------|---------------|-----------|
| Phase A | DyCP | `local + allgather` 两个区域 | `ws // n_dycp_with_context` |
| Phase B | DP | 全部 workspace | `ws // n_dp_with_context` |

两个 phase 顺序执行，不存在冲突。各自独占 workspace 计算 `max_context_chunk`，避免相互稀释。

---

## 7. 已知限制与未来优化

1. **PCPManager 代码重复**：DyCP 版本（`get_dycp_restore_*`）几乎是 PCP 版本的 fork，仅通信分组不同。可以通过参数化通信分组来合并。

2. **强制 prefill 分类**：当前 DyCP batch 中所有请求都被强制归为 prefill，即使实际只有 1 个 token。这可能影响 decode-heavy workload 的性能。

3. **`chunked_prefill_workspace_size` 硬编码**：从 `64 × 1024` 改为 `1048576`（16× 增大），这个值可能需要根据实际 `model_len` 和 `head_size` 动态调整。

4. **`reorder_batch_threshold` 强制为 1**：禁用了 FlashMLA 的 decode 优化路径，对 DyCP batch 中的短 decode 请求有性能影响。

---

## 8. 测试建议

| 场景 | 验证要点 |
|------|---------|
| 纯 DyCP prefill（全 CP 请求） | KV allgather + DualChunkSwap 正确性 |
| 混合 batch（DyCP + DP） | `_build_mixed_dycp_dp_prefill` 拆分正确性，两 phase 独立执行 |
| DyCP + chunked prefill | 跨 rank 元数据一致性，无 NCCL hang |
| DyCP decode | LSE allreduce 正确，输出合并正确 |
| 奇数长度序列 | DualChunkSwap head/tail 分片不丢失 token |
| 单请求 batch | 边界情况：空 DyCP 或空 DP 子 batch |
| Per-rank budget 调度 | CP 请求和 DP 请求的 budget 分配均衡 |
| 不同 rank 上 DP 请求数不同 | 无 NCCL hang（DP 部分无集合通信） |
| DyCP context_len=0（首次 prefill） | chunked_context=None 路径正常 |
