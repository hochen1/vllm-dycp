# DyCP Block Size 对齐分析报告

**分析范围**: commit `d0bb2a4d0` 上所有与 DyCP 相关的 block size 变量  
**重点**: 最近新加入的 block size 变量在处理上的对齐问题  
**平台假设**: 仅考虑 CUDA 平台（`aot_schedule=True`）

---

## 1. Block Size 族谱

下表列出 DyCP 涉及的全部 block size 变量及其层级关系。

### 1.1 变量总览

| 变量名 | 定义位置 | 值 / 计算公式 | 含义 |
|--------|---------|--------------|------|
| `cp_kv_cache_interleave_size` | `config/parallel.py:243` | 用户配置，默认 1 | 连续存放在同一 rank 上的 token 数 |
| `block_size` | `cache_config` | 用户配置，默认 16 | KV cache 的物理 page 大小 |
| `cp_local_block_size` | `mla/common.py:601` | `= cp_kv_cache_interleave_size` | 每个 rank 上连续存放的 token 数（别名） |
| `cp_virtual_block_size` | `mla/common.py:604-607` | DyCP: `cp_local_block_size × dycp_world_size`<br>PCP/DCP: `cp_local_block_size × cp_world_size` | 一个完整 interleave 周期覆盖的 token 数 |
| `dycp_local_block_size` | `mla/common.py:611` | `= cp_kv_cache_interleave_size` | **⚠️ 死代码，从未使用** |
| `dycp_virtual_block_size` | `mla/common.py:612` | `= dycp_local_block_size × dycp_world_size` | **⚠️ 死代码，从未使用** |
| `page_size` | `mla/common.py:616` | `= kv_cache_spec.block_size`（仅 CUDA） | AOT 调度时的对齐单位 |
| `chunked_prefill_workspace_size` | `mla/common.py:531-546` | `min(max(8×max_model_len, 4×max_seqs×block_size), 1048576)` | chunked prefill workspace 大小 |
| `max_context_chunk` | `mla/common.py:1300-1309` | `workspace_size // num_reqs_with_context` | 每个 prefill 请求分到的最大 context chunk |
| `dycp_max_context_chunk` | `mla/common.py:867-872` | `workspace_size // dycp_n_with_ctx` | DyCP 专用路径的最大 context chunk |
| `padded_local_max_chunk` | `mla/common.py:918-920` | `cdiv(max_chunk, cp_virtual_block_size) × cp_local_block_size` | chunk 映射到本地 rank 后的 padded 大小 |

### 1.2 约束关系图

```
cp_kv_cache_interleave_size (配置)
  │
  ├── cp_local_block_size (= interleave_size)
  │     │
  │     └── cp_virtual_block_size (= cp_local_block_size × world_size)
  │           │
  │           ├── padded_local_max_chunk = cdiv(max_chunk, cp_virtual_block_size) × cp_local_block_size
  │           └── padded_local_context_lens = cdiv(context_len, cp_virtual_block_size) × cp_local_block_size
  │
  ├── dycp_local_block_size (= interleave_size)  ← ⚠️ 死代码
  │     └── dycp_virtual_block_size (= dycp_local_block_size × dycp_world_size) ← ⚠️ 死代码
  │
  └── block_size (KV cache page)
        │
        ├── page_size (= block_size, 仅 CUDA)
        │     └── max_context_chunk = round_down(ws // n_with_ctx, page_size) [仅 aot_schedule=True]
        │
        └── virtual_block_size (block_table.py) = block_size × total_cp_world_size
              └── slot_mapping 计算中的 interleave 分桶
```

### 1.3 `world_size` 的多重语义

| 上下文 | 变量名 | 值 | 定义位置 |
|--------|--------|---|---------|
| MLA metadata 构建 | `cp_world_size` | `dcp × pcp` | `mla/common.py:600` |
| MLA metadata 构建 (DyCP) | `dycp_world_size` | `dp_per_domain` | `mla/common.py:588` |
| gpu_model_runner | `cp_world_size` | DyCP 时**覆盖**为 `dycp_world_size` | `gpu_model_runner.py:330` |
| block_table | `total_cp_world_size` | `pcp × dcp × dycp` | `block_table.py:105-106` |
| cross_dp_kv_cache_manager | 局部 `world_size` | `dycp_world_size` | 调用方传入 |

---

## 2. 问题分析

### 问题 1 [P2/仅非 CUDA]: `dycp_max_context_chunk % dycp_world_size` 对齐依赖 `aot_schedule`

**位置**: `mla/common.py:867-875` (`_build_mixed_dycp_dp_prefill`) 及 `mla/common.py:1405-1409`（标准路径）

```python
dycp_max_context_chunk = (
    self.chunked_prefill_workspace_size // dycp_n_with_ctx
)
if self.aot_schedule:                         # ← 仅 CUDA
    dycp_max_context_chunk = round_down(
        dycp_max_context_chunk, self.page_size
    )
assert dycp_max_context_chunk % self.dycp_world_size == 0  # ← L875
```

**CUDA 上安全**: `aot_schedule=True` 时，`round_down(chunk, page_size)` 使 chunk 成为 `block_size`（默认 16）的倍数。只要 `block_size % dycp_world_size == 0`（16 对 2/4/8/16 均满足），assertion 必然成立。

**非 CUDA 上会崩溃**: 但因为只考虑 CUDA，此问题可忽略。标准路径 L1409 同理。

**隐含前提**: 此对齐依赖 `block_size % dycp_world_size == 0`，但该约束没有在配置层面显式校验（见问题 5）。

---

### 问题 3 [P1/死代码]: `dycp_local_block_size` / `dycp_virtual_block_size` 从未使用

**位置**: `mla/common.py:611-612`

```python
self.dycp_local_block_size = parallel_config.cp_kv_cache_interleave_size
self.dycp_virtual_block_size = self.dycp_local_block_size * self.dycp_world_size
```

**验证**: 全局搜索 `dycp_local_block_size` 和 `dycp_virtual_block_size`，仅在 L611-612 出现定义，无任何读取。

**分析**: 从命名看，这两个变量本意是为 DyCP 提供独立的 block size 参数。但实际代码中，DyCP 使用的是 `self.cp_local_block_size` 和 `self.cp_virtual_block_size`（L601-607）。这说明两组变量在语义上完全重复：

| 变量 | 值 | DyCP 时 |
|------|----|---------|
| `cp_local_block_size` | `interleave_size` | ✓ 正在使用 |
| `cp_virtual_block_size` | `interleave_size × dycp_world_size` | ✓ 正在使用 |
| `dycp_local_block_size` | `interleave_size` | ✗ 死代码 |
| `dycp_virtual_block_size` | `interleave_size × dycp_world_size` | ✗ 死代码 |

**修复建议**: 删除 L611-612。

---

### 问题 4 [P1/设计缺陷]: `_avg_distribute_tokens_to_ranks` 忽略 `interleave_size`

**位置**: `cross_dp_kv_cache_manager.py:92-111`

```python
def _avg_distribute_tokens_to_ranks(
    self,
    world_size: int,
    seq_len: int,
    cp_kv_cache_interleave_size: int = 1,  # ← 接受但不用
) -> list[int]:
    # 实际使用的是 2 * world_size 对齐（DualChunkSwap 策略）
    num_padded_tokens = cdiv(seq_len, 2 * world_size) * (2 * world_size)
    local_seq_len = num_padded_tokens // world_size
    return [local_seq_len for _ in range(world_size)]
```

**问题**: 函数签名声称接受 `cp_kv_cache_interleave_size`，但函数体完全忽略该参数，始终使用 `2 * world_size` 对齐（即 DualChunkSwap 策略，假定 `interleave_size = 1`）。

**影响**: 当 `cp_kv_cache_interleave_size > 1` 时：
- `get_cp_local_seq_lens`（utils.py:1199）使用 `interleave_size` 计算每 rank 实际持有的 token 数
- `_avg_distribute_tokens_to_ranks` 忽略 `interleave_size`，用 `2 * world_size` 计算 block 分配
- 两者**可能不一致**：`get_cp_local_seq_lens` 产生的 local len 可能超过 `_avg_distribute_tokens_to_ranks` 分配的 block 数

**当前安全条件**: 只要 `interleave_size = 1`（默认值），`2 * world_size` 对齐等价于 `interleave_size * 2 * world_size` 对齐。但如果用户配置 `interleave_size > 1`，可能触发 slot 越界。

**修复建议**: 要么删除该参数（如果确认 DyCP 永远用 `interleave_size=1`），要么实际使用它：
```python
align_unit = cp_kv_cache_interleave_size * 2 * world_size
num_padded_tokens = cdiv(seq_len, align_unit) * align_unit
```

---

### 问题 5 [P2/配置校验缺失]: DyCP 模式缺少 `block_size` 校验

**位置**: `config/vllm.py:788-812`

```python
# If DCP, ensure the block size is right.
if self.parallel_config.decode_context_parallel_size > 1:
    assert (
        self.parallel_config.cp_kv_cache_interleave_size
        <= self.cache_config.block_size
        and self.cache_config.block_size
        % self.parallel_config.cp_kv_cache_interleave_size
        == 0
    )
```

**问题**: 这段校验**仅**在 `decode_context_parallel_size > 1`（即 DCP 模式）下执行。DyCP 模式（`dp_per_domain > 1`）完全跳过此校验。

**但 DyCP 同样依赖这些约束**：
1. `block_table.py:179`: `virtual_block_offsets // self.cp_kv_cache_interleave_size % total_cp_world_size` — 要求 `block_size >= interleave_size`
2. `mla/common.py:627`: `chunked_prefill_workspace_size % dycp_world_size == 0` — workspace 要被 `dycp_world_size` 整除
3. `mla/common.py:875`: `dycp_max_context_chunk % dycp_world_size == 0` — chunk 要被 `dycp_world_size` 整除

**修复建议**: 将校验条件扩展到 DyCP：
```python
if (self.parallel_config.decode_context_parallel_size > 1
    or self.parallel_config.dp_per_domain > 1):
    assert (
        self.parallel_config.cp_kv_cache_interleave_size
        <= self.cache_config.block_size
        and self.cache_config.block_size
        % self.parallel_config.cp_kv_cache_interleave_size
        == 0
    )
```

---

### 问题 6 [P2/语义不一致]: `total_cp_world_size` 乘法 vs `cp_world_size` 覆盖

**位置**:
- `block_table.py:105-106`:
  ```python
  self.total_cp_world_size = self.pcp_world_size * self.dcp_world_size * self.dycp_world_size
  ```
- `gpu_model_runner.py:329-331`:
  ```python
  if self.dycp_world_size > 1:
      self.cp_world_size = self.dycp_world_size   # ← 覆盖
      self.cp_rank = self.dycp_rank
  ```

**语义冲突**: 
- `block_table` 认为 CP 维度是**乘法组合** (`pcp × dcp × dycp = total_cp_world_size`)，slot mapping 使用 `total_cp_world_size` 做 interleave
- `gpu_model_runner` 认为 DyCP 与 PCP/DCP **互斥**，直接用 `dycp_world_size` 覆盖 `cp_world_size`

**当前安全条件**: 当 DyCP 启用时，`pcp_world_size = 1` 且 `dcp_world_size = 1`（互斥约束），所以 `total_cp_world_size = 1 × 1 × dycp_world_size = dycp_world_size`，与覆盖后的 `cp_world_size` 一致。

**风险**: 如果未来放开互斥约束（允许 DyCP + PCP 共存），`block_table` 使用乘法组合的 `total_cp_world_size` 而 `gpu_model_runner` 只使用 `dycp_world_size`，二者将不一致，导致 slot mapping 与 chunked prefill 的 local layout 错位。

**修复建议**: 在 `gpu_model_runner` 中添加显式断言：
```python
if self.dycp_world_size > 1:
    assert self.pcp_world_size == 1 and self.dcp_world_size == 1, \
        "DyCP is currently exclusive with PCP/DCP"
    self.cp_world_size = self.dycp_world_size
    self.cp_rank = self.dycp_rank
```

同时在 `mla/common.py` 的 `__init__` 中也加上相同断言（目前缺失）。

---

### 问题 7 [P2/潜在风险]: workspace 硬编码 1048576 对非 2 的幂 `dycp_world_size` 的整除性

**位置**: `mla/common.py:531-546`

```python
chunked_prefill_workspace_size = min(
    max(
        8 * model_config.max_model_len,
        4 * scheduler_config.max_num_seqs * cache_config.block_size,
    ),
    1048576,  # ← 硬编码上限
)
```

**以及**: `mla/common.py:627`
```python
assert self.chunked_prefill_workspace_size % self.dycp_world_size == 0
```

**分析**: `1048576 = 2^20`，其因子包含所有 2 的幂。常见的 `dycp_world_size` 值（2, 4, 8, 16）都是 2 的幂，所以 `1048576 % dycp_world_size == 0` 始终成立。

**但**：如果 `dycp_world_size` 不是 2 的幂（如 3, 5, 6, 7...），assertion 会失败。虽然当前实际部署中 `dycp_world_size` 一般等于 DP 数（2 的幂），但代码没有强制这一约束。

另外，`max(8 * max_model_len, 4 * max_seqs * block_size)` 的结果也不保证被 `dycp_world_size` 整除。当这个值小于 1048576 时，workspace 取 `max(...)` 分支，同样可能不满足整除。

**修复建议**: 在 `determine_chunked_prefill_workspace_size` 末尾，或在 L627 之前，对 workspace 做 `round_down`：
```python
if dycp_world_size > 1:
    chunked_prefill_workspace_size = round_down(
        chunked_prefill_workspace_size, dycp_world_size
    )
```

---

## 3. 总结矩阵

| # | 严重度 | 类型 | 位置 | 问题 | CUDA 影响 |
|---|--------|------|------|------|----------|
| 1 | P2 | 隐含约束 | `common.py:875` / `common.py:1409` | chunk 对齐依赖 `block_size % dycp_ws == 0`，无显式校验 | CUDA 上安全（默认 block_size=16），但依赖未校验的前提 |
| 3 | **P1** | 死代码 | `common.py:611-612` | `dycp_local/virtual_block_size` 定义但从未使用 | 代码混淆，维护负担 |
| 4 | **P1** | 设计缺陷 | `cross_dp_kv_cache_manager.py:92` | `interleave_size` 参数被忽略 | `interleave_size>1` 时 block 分配与 local len 可能不一致 |
| 5 | **P1** | 校验缺失 | `config/vllm.py:788` | DyCP 模式跳过 `block_size % interleave_size` 校验 | 问题 1 的对齐前提无保障 |
| 6 | P2 | 语义不一致 | `block_table.py:105` vs `gpu_model_runner.py:330` | 乘法 vs 覆盖 | 当前安全（互斥保证），未来扩展有风险 |
| 7 | P2 | 潜在风险 | `common.py:546` + `common.py:627` | workspace 硬编码对非 2^n world_size 不整除 | 当前安全（world_size 均为 2 的幂） |

### 优先级建议（CUDA Only）

- **应清理**: 问题 3 — 删除死代码，避免维护时混淆
- **应修复**: 问题 5 — 为 DyCP 补充 `block_size % interleave_size` 校验，让问题 1 的隐含前提变为显式保障
- **评估修复**: 问题 4 — 取决于是否支持 `interleave_size > 1`
- **防御性加固**: 问题 6、7 — 加断言防止未来回归
