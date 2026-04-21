# GQA DyCP Prefill 设计文档

## 1. 问题背景

DyCP（Dynamic Context Parallelism）在 prefill 阶段使用 PCP 的 DualChunkSwap 策略将 token 分配给各 rank 并行计算。原始 GQA DyCP prefill 实现存在两个根本问题：

### 1.1 Causal Mask 不兼容

标准 `flash_attn_varlen_func` 的 causal mask 基于 local index 而非 global position。PCP 分片后 query 位置不连续（DualChunkSwap 给出 head+tail 分布），interleave 后 KV 位置也不连续。无论 `causal=True` 还是 `causal=False` 都无法正确处理这种组合：

- `causal=False`：query 看到未来 KV → 非因果 attention → 中间层 hidden states 错误
- `causal=True`：mask 基于 local index → 不匹配 global position 关系

### 1.2 KV Cache 不完整

PCP 的 token 分配和 KV cache 的 interleave 分布不对齐。以 228 tokens、2 ranks、interleave_size=64 为例：

| | PCP 处理 | Interleave 拥有 | 实际写入 cache | 应有 |
|---|---|---|---|---|
| Rank 0 | {0..56, 171..227} | {0..63, 128..191} | {0..56, 171..191} = 78 | 128 |
| Rank 1 | {57..170} | {64..127, 192..227} | {64..113} = 50 | 100 |

100 个 position 在任何 rank 的 cache 中都不存在，decode 时 allreduce 缺失 KV。

## 2. 解决方案

### 2.1 DualChunkSwap Attention（解决 causal mask 问题）

复用 MLA 的 DualChunkSwap 模式：allgather 得到全量 KV 后，将 Q 和 KV 按 head/tail 拆成两个 chunk，每个 chunk 内位置有序，用 `causal=True` 跑 `flash_attn_varlen_func`。

**对齐 MLA 的函数**：
- `get_pcp_query_indices(cu_num_tokens)` — Q 的 head/tail 拆分
- `get_pcp_kv_indices(cu_num_tokens, rank, world_size)` — KV 的 head/tail 拆分
- `pcp_kv_start_loc = pcp_q_start_loc × dycp_world_size` — KV 边界

### 2.2 Cache Refill（解决 KV cache 不完整问题）

Prefill 时 allgather K/V 后，用 `dycp_full_slot_mapping`（覆盖所有 original position 的 interleave slot mapping）将完整 KV 写入 cache。通过 `dycp_real_token_indices` 过滤 PCP padding token，确保 real tokens 和 slot mapping 精确对齐。

### 2.3 Padding 对齐

PCP DualChunkSwap 将 token pad 到 `2×world_size` 的倍数。allgather+restore 后 padding token 混在 real token 中间。`dycp_real_token_indices` 通过纯数学计算（不需要通信）标记哪些是 real token：

- DualChunkSwap 的最后一个 chunk 始终分给 rank 0 的 tail → **所有 padding 归 rank 0**
- Rank 0 每个 request 有 `pads[i]` 个 padding，其他 rank 有 0 个

## 3. 代码修改

### 3.1 `flash_attn.py` — GQA attention backend

**Metadata builder `build()`**：
- 新增 `has_dycp_prefill` 区分 prefill（DualChunkSwap）和 decode（local seq_lens + allreduce）
- Prefill：计算 DualChunkSwap head/tail indices 和 cu_seqlens，不修改 `seq_lens`
- Decode：`seq_lens[:num_dycp_reqs]` 转 local，`causal` 保持 `True`
- Mixed split 路径：传递 `pcp_allgather_restore_idx` 和 `dycp_full_slot_mapping`

**Forward — DyCP prefill**：
1. allgather K/V → restore to global position order
2. 用 `dycp_real_token_indices` 提取 real tokens → 用 `dycp_full_slot_mapping` 写 cache
3. DualChunkSwap：head/tail `flash_attn_varlen_func(causal=True)` → concat → restore order → return
4. 跳过 local write（cache refill 已精确写入所有 real tokens）

**Forward — DyCP decode**：
1. `reshape_and_cache_flash` 写 decode token 到 interleave slot
2. `flash_attn_varlen_func` 从 paged cache 读 local KV
3. `cp_lse_ag_out_ar` allreduce 合并各 rank 部分 attention
4. allreduce 条件从 `self.dycp_world_size > 1` 改为 `attn_metadata.num_dycp_reqs > 0`

**新增 metadata 字段**：`dycp_full_slot_mapping`、`pcp_allgather_restore_idx`、`dycp_real_token_indices`、9 个 DualChunkSwap indices/cu_seqlens。

### 3.2 `gpu_model_runner.py` — 输入准备

- PCP split 前保存 `_dycp_orig_scheduled`（原始 token 数）
- `reorder_batch_threshold` 为 None 时默认 1（GQA 不设此值）
- 调用 `compute_dycp_full_slot_mapping` 计算全量 slot mapping
- 纯数学计算 `dycp_real_token_indices`（零通信）
- 挂载到 `CommonAttentionMetadata` 传递给 attention backend

### 3.3 `block_table.py` — KV cache slot 计算

- 新增 `dycp_full_slot_mapping` buffer
- 新增 `compute_dycp_full_slot_mapping()` 方法：为所有 original position 计算 interleave slot mapping

### 3.4 `utils.py` — 通用 metadata 和 batch 排序

- `CommonAttentionMetadata` 新增 `dycp_full_slot_mapping`、`dycp_real_token_indices` 字段
- `slice_common_attn_metadata` 传递新字段
- 新增 `_deterministic_hash(req_id)` + CP 请求按 hash 排序（保证各 rank batch 内顺序一致）

### 3.5 `mla/common.py` — MLA attention backend

- `MLACommonMetadata` 新增 `dycp_full_slot_mapping` 字段
- Builder 传递 `dycp_full_slot_mapping`（主 metadata 和 `_dycp_split`）
- `forward_common` allgather 后用 `dycp_full_slot_mapping` 写 cache（cache refill）

### 3.6 `cross_dp_scheduler.py`

- `long_request_threshold` 从 128K 调整为 1024

## 4. 数据流

### Prefill（每层 forward）

```
allgather K/V → restore → select real tokens → write cache (dycp_full_slot_mapping)
                       ↘ DualChunkSwap: head flash_attn(causal=True) 
                                       + tail flash_attn(causal=True)
                                       → concat → restore order → output
```

### Decode（每层 forward）

```
reshape_and_cache_flash(decode_token) → flash_attn(paged, local_seq_lens) → cp_lse_ag_out_ar
```

## 5. 解决的 Bug

| Bug | 原因 | 修复 |
|-----|------|------|
| Prefill 输出 garbage（235B） | `causal=False` 导致非因果 attention | DualChunkSwap 正确因果 attention |
| Decode 从第2个 token 开始错 | KV cache 缺失条目 | Cache refill 补全 interleave 分区 |
| Padding 导致 cache 写入错位 | `all_k[:nf]` 包含 padding token | `dycp_real_token_indices` 精确过滤 |
| DP-only batch 死锁 | allreduce 条件用 `self.dycp_world_size > 1` | 改为 `attn_metadata.num_dycp_reqs > 0` |
| `reorder_batch_threshold` None | GQA backend 未设此值 | 默认 1 |
| Mixed split 路径 DualChunkSwap 不触发 | `pcp_allgather_restore_idx` 未传递 | `slice_common_attn_metadata` 传参 |
| CP 请求跨 rank 顺序不一致 | Python `hash()` 跨进程随机 | MD5 确定性 hash 排序 |
