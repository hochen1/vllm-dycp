# DyCP Block Size 各层处理方式对比与正确性验证

**分析日期**: 2026-04-13  
**关联文档**: `docs/dycp_block_size_analysis.md`（变量族谱与单点问题分析）  
**本文重点**: 从端到端数据流角度，梳理 DyCP block size 在 scheduler → KVCacheManager → worker → attention 各层的传递和变换，验证各层之间是否对齐

---

## 1. 整体架构：两条独立路径

DyCP 引入后，block size 的处理分为**两条互斥的路径**。根据 `dp_per_domain` 是否 > 1 选择：

```
                          dp_per_domain > 1?
                         ┌───── YES ─────┐
                         │               │
                   CrossDP 路径      普通 CP 路径
                         │               │
               CrossDPScheduler      Scheduler
                         │               │
            CrossDPKVCacheManager   KVCacheManager
            (N 个独立 BlockPool)   (UnitaryCoordinator)
                         │               │
               ┌─────────┴─────────┐     │
               rank0  rank1 ... rankN    单 manager
               (dcp=1, pcp=1)       (dcp=实际, pcp=实际)
```

**核心设计原则**：
- 普通路径中，`SingleTypeKVCacheManager` 用放大后的 block_size 统一管理
- CrossDP 路径中，scheduler 自己做 token 到 rank 的分配，每个 rank 的 manager 用**原始** block_size 独立管理

---

## 2. 各层 Block Size 详解

### 2.1 Scheduler 层：统一放大

**位置**: `vllm/v1/engine/core.py:142-147`

```python
scheduler_block_size = (
    vllm_config.cache_config.block_size          # 原始值，如 16
    * vllm_config.parallel_config.decode_context_parallel_size   # DCP
    * vllm_config.parallel_config.prefill_context_parallel_size  # PCP
    * max(vllm_config.parallel_config.dp_per_domain, 1)          # DyCP
)
```

**作用**：
1. 传给 `Scheduler.__init__` → `self.block_size`（`scheduler.py:140`）
2. 传给 `get_request_block_hasher`（`engine/core.py:212`）做 prefix cache hash

**设计意图**：scheduler 看到的是**完整请求**的全量 token，需要以一个更大的粒度分配，确保分配出的 token 数能被所有 CP rank 均分。

**DyCP 取值**：假设 base=16, DCP=1, PCP=1, DyCP=4 → scheduler_block_size = 64

---

### 2.2 KVCacheManager 层：两条路径分叉

#### 路径 A: CrossDP 路径（DyCP 启用时）

**选择逻辑**: `cross_dp_scheduler.py:161`

```python
self.cp_world_size = vllm_config.parallel_config.dp_per_domain
```

**CrossDPKVCacheManager 创建**: `cross_dp_scheduler.py:164-174`

```python
self.kv_cache_manager = CrossDPKVCacheManager(
    ...
    cp_world_size=self.cp_world_size,       # = dp_per_domain
    hash_block_size=self.block_size,         # = scheduler_block_size（已放大）
)
```

**CrossDPKVCacheCoordinatorNoPrefixCache 内部**: `cross_dp_kv_cache_manager.py:66-90`

```python
# BlockPool 使用放大后的 hash_block_size（用于 prefix cache hash 粒度）
self.block_pools = [
    DPBlockPool(rank, num_blocks, enable_caching,
                hash_block_size,  # = scheduler_block_size = base × DyCP
                ...)
    for rank in range(cp_world_size)
]

# block_size 使用原始值（用于物理 block 分配）
self.block_size = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size  # = 16

# 每个 rank 的 SingleTypeKVCacheManager：dcp=1, pcp=1 → 不放大
self.corss_dp_single_type_managers = [
    tuple(
        get_manager_for_kv_cache_spec(
            kv_cache_spec=...,
            block_pool=self.block_pools[rank],
            dcp_world_size=1,   # ← 不放大
            pcp_world_size=1,   # ← 不放大
        )
        ...
    ) for rank in range(cp_world_size)
]
```

**关键区分**：

| 变量 | 值 | 用途 |
|------|---|------|
| `hash_block_size`（BlockPool） | base × DyCP = 64 | prefix cache hash 粒度，需要和 request_block_hasher 一致 |
| `self.block_size`（Coordinator） | base = 16 | 物理 block 分配，每个 rank 独立管理 |
| 每 rank manager 的 `block_size` | base = 16 | 单 rank 上的 block 分配计算 |

**正确性验证**：
- hash 粒度 64 = scheduler 的 hash 粒度 64 ✅
- 每 rank 物理 block 16，4 个 rank 合计 64 = scheduler_block_size ✅
- `_avg_distribute_tokens_to_ranks` 均分 token 后，每 rank 的 manager 用 block_size=16 计算所需 block 数 ✅

#### 路径 B: 普通 CP 路径（DyCP 未启用时）

**UnitaryKVCacheCoordinator**: `kv_cache_coordinator.py:294-300`

```python
self.block_size = self.kv_cache_spec.block_size   # 原始值 16
if dcp_world_size > 1:
    self.block_size *= dcp_world_size              # 16 × DCP
if pcp_world_size > 1:
    self.block_size *= pcp_world_size              # × PCP
```

**SingleTypeKVCacheManager**: `single_type_kv_cache_manager.py:44-48`

```python
self.block_size = kv_cache_spec.block_size
if dcp_world_size * pcp_world_size > 1:
    self.block_size *= dcp_world_size * pcp_world_size
```

**注意**：这两处都**没有** DyCP 因子——这是正确的，因为 DyCP 不走这条路径。

---

### 2.3 Worker 层：gpu_model_runner

**位置**: `gpu_model_runner.py:325-331`

```python
self.cp_world_size = self.dcp_world_size * self.pcp_world_size
self.dycp_world_size = self.parallel_config.dp_per_domain
self.dycp_rank = 0 if self.dycp_world_size <= 1 else get_dycp_group().rank_in_group
if self.dycp_world_size > 1:
    self.cp_world_size = self.dycp_world_size    # ← 覆盖！
    self.cp_rank = self.dycp_rank
```

**处理方式**：DyCP 与 DCP/PCP **互斥覆盖**，不是乘法组合。

**隐含约束**：当 `dycp_world_size > 1` 时，`dcp_world_size = 1` 且 `pcp_world_size = 1`。目前代码**未显式 assert**。

---

### 2.4 Worker 层：block_table

**位置**: `block_table.py:105-112`

```python
self.total_cp_world_size = (
    self.pcp_world_size * self.dcp_world_size * self.dycp_world_size
)
self.total_cp_rank = (
    (self.dycp_rank * self.pcp_world_size + self.pcp_rank)
    * self.dcp_world_size
    + self.dcp_rank
)
```

**处理方式**：三维**乘法组合**。

**slot_mapping 计算**: `block_table.py:158-195`

```python
if total_cp_world_size > 1:
    virtual_block_size = self.block_size * total_cp_world_size  # 16 × 4 = 64
    block_table_indices = req_indices * max_blocks + positions // virtual_block_size

    # interleave 分桶：判断当前 token 属于哪个 rank
    virtual_block_offsets = positions % virtual_block_size
    mask = (virtual_block_offsets // interleave_size % total_cp_world_size == total_cp_rank)

    # 本地偏移计算
    block_offsets = (
        virtual_block_offsets // (total_cp_world_size * interleave_size) * interleave_size
        + virtual_block_offsets % interleave_size
    )
    slot_mapping = block_numbers * self.block_size + block_offsets
```

**与 gpu_model_runner 的一致性**：
- 当 DyCP=4, DCP=1, PCP=1 时：
  - block_table: `total_cp_world_size = 1 × 1 × 4 = 4` ✅
  - model_runner: `cp_world_size = 4`（覆盖后） ✅
  - 二者一致

---

### 2.5 Attention 层：MLA metadata

**位置**: `mla/common.py:600-612`

```python
self.cp_world_size = self.dcp_world_size * self.pcp_world_size  # 不含 DyCP
self.cp_local_block_size = parallel_config.cp_kv_cache_interleave_size

if self.dycp_world_size > 1:
    self.cp_virtual_block_size = self.cp_local_block_size * self.dycp_world_size
else:
    self.cp_virtual_block_size = self.cp_local_block_size * self.cp_world_size

# 死代码（从未使用）
self.dycp_local_block_size = parallel_config.cp_kv_cache_interleave_size
self.dycp_virtual_block_size = self.dycp_local_block_size * self.dycp_world_size
```

**处理方式**：DyCP 时用 `dycp_world_size` 替代 `cp_world_size`（与 model_runner 的覆盖逻辑一致）。

**用于 prefill metadata**: `mla/common.py:918-930`

```python
# 将全局 chunk 映射到本地 rank 的大小
dycp_padded_local_max_chunk = (
    cdiv(dycp_max_context_chunk, self.cp_virtual_block_size)
    * self.cp_local_block_size
)
dycp_padded_local_context_lens = (
    cdiv(dycp_context_lens_cpu, self.cp_virtual_block_size)
    * self.cp_local_block_size
)
```

---

### 2.6 KV Cache 接口层

**位置**: `kv_cache_interface.py:19-26`

```python
def get_cp_kv_cache_world_size(vllm_config: VllmConfig) -> int:
    parallel_config = vllm_config.parallel_config
    return (
        max(parallel_config.decode_context_parallel_size, 1)
        * max(parallel_config.prefill_context_parallel_size, 1)
        * max(parallel_config.dp_per_domain, 1)
    )
```

**处理方式**：三维乘法，与 `engine/core.py` 的 scheduler_block_size 计算逻辑一致。

---

## 3. 端到端数据流对比矩阵

以 `base_block_size=16, DyCP=4, DCP=1, PCP=1` 为例：

| 层级 | 组件 | block_size 值 | 放大因子 | 说明 |
|------|------|-------------|---------|------|
| **Scheduler** | `engine/core.py` scheduler_block_size | **64** | ×DCP ×PCP ×DyCP | 全局调度粒度 |
| **Scheduler** | `scheduler.py` self.block_size | **64** | 继承 | 传给 hash_block_size |
| **Hash** | `request_block_hasher` | **64** | 继承 | prefix cache hash 粒度 |
| **KVCacheManager** | `CrossDPKVCacheManager` hash_block_size（BlockPool） | **64** | 继承 | hash 对齐 ✅ |
| **KVCacheManager** | `CrossDPCoordinator` self.block_size | **16** | 无放大 | 物理 block 分配 |
| **KVCacheManager** | 每 rank `SingleTypeManager` block_size | **16** | dcp=1,pcp=1 | 单 rank 独立管理 |
| **Worker** | `gpu_model_runner` cp_world_size | **4** | 覆盖 | DyCP 覆盖 CP |
| **Worker** | `block_table` total_cp_world_size | **4** | 1×1×4 | 三维乘法 |
| **Worker** | `block_table` virtual_block_size | **64** | 16×4 | slot mapping ✅ |
| **Attention** | `mla/common.py` cp_virtual_block_size | interleave×4 | ×DyCP | prefill metadata |

**一致性验证**：
- scheduler hash 粒度 (64) = BlockPool hash 粒度 (64) ✅
- scheduler 分配粒度 (64) = 4 rank × 物理 block (16) ✅
- block_table virtual_block_size (64) = scheduler_block_size (64) ✅
- model_runner cp_world_size (4) = block_table total_cp_world_size (4) ✅

---

## 4. 正确性分析与已知问题

### 4.1 正确的设计决策

| 决策 | 原因 |
|------|------|
| CrossDP 路径的 `SingleTypeManager` 用 dcp=1, pcp=1 | 每个 rank 独立管理自己的 block，不需要放大 |
| 普通路径的 `SingleTypeManager` 不含 DyCP 因子 | DyCP 不走这条路径，两条路径互斥 |
| `hash_block_size` 使用放大后的 scheduler_block_size | 必须和 `request_block_hasher` 的粒度一致 |
| block_table 用 `total_cp_world_size` 三维乘法 | 覆盖和乘法在互斥约束下等价，但乘法更通用 |

### 4.2 风险点：互斥约束缺少显式 assert

**现状**：DyCP 与 DCP/PCP 的互斥是**隐含假设**，多处代码的正确性依赖于此，但无显式校验。

| 位置 | 处理方式 | 互斥时等价 | 非互斥时是否正确 |
|------|---------|----------|----------------|
| `gpu_model_runner.py:329` | DyCP 覆盖 cp_world_size | DyCP = total_cp | ❌ 丢失 DCP/PCP |
| `block_table.py:105` | 三维乘法 | DyCP = total_cp | ✅ 自然支持 |
| `mla/common.py:604` | DyCP 替代 cp_world_size | DyCP = cp_world | ❌ 丢失 DCP/PCP |

**建议**：在 `gpu_model_runner` 和 `mla/common.py` 中添加显式断言：

```python
if self.dycp_world_size > 1:
    assert self.dcp_world_size == 1 and self.pcp_world_size == 1, \
        "DyCP is currently exclusive with DCP/PCP"
```

### 4.3 风险点：`_avg_distribute_tokens_to_ranks` 的 local token 数与 block_size 对齐

**位置**: `cross_dp_kv_cache_manager.py:108-110`

```python
num_padded_tokens = cdiv(seq_len, 2 * world_size) * (2 * world_size)
local_seq_len = num_padded_tokens // world_size
```

分配到单 rank 的 `local_seq_len` 不一定是 `block_size`（16）的倍数：

| world_size | padding 单位 | local_seq_len 示例 | 能整除 16？ |
|-----------|-------------|-------------------|-----------|
| 2 | 4 | seq=100 → pad=100 → local=50 | 50/16=3...2 ❌ |
| 4 | 8 | seq=100 → pad=104 → local=26 | 26/16=1...10 ❌ |
| 8 | 16 | seq=100 → pad=112 → local=14 | 14/16=0...14 ❌ |

当 `local_seq_len` 不整除 `block_size` 时，`_allocate_blocks_to_cp_ranks` 中：

```python
total_blocks = num_tokens // self.block_size  # 截断
```

尾部不足一个 block 的 token 被**静默丢弃**。这些 token 对应的 KV cache 不会被分配 block。

**实际影响**：需要确认 worker 侧是否也做了相同的截断/padding 处理，使两侧一致。如果 worker 侧为这些尾部 token 计算了 slot mapping 但 scheduler 没有分配 block，会导致越界访问。

### 4.4 `dycp_local_block_size` / `dycp_virtual_block_size` 是死代码

**位置**: `mla/common.py:611-612`

定义但从未被读取。实际使用的是 `cp_local_block_size` 和 `cp_virtual_block_size`。应清理。

---

## 5. 总结

| 结论 | 说明 |
|------|------|
| **整体设计正确** | 两条路径互斥且各自自洽，block size 在各层的传递链完整 |
| **hash 粒度一致** | scheduler_block_size → request_block_hasher 和 BlockPool 的 hash_block_size 对齐 |
| **物理分配正确** | CrossDP 每 rank manager 用原始 block_size，不重复放大 |
| **互斥约束需显式化** | 多处覆盖逻辑依赖 DyCP/DCP/PCP 互斥，建议加 assert |
| **尾部 token 对齐需验证** | `_avg_distribute_tokens_to_ranks` 的结果不保证整除 block_size |
| **死代码应清理** | `dycp_local_block_size` / `dycp_virtual_block_size` 未使用 |
