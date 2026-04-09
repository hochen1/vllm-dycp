# DyCP Prefill MLA Code Review Report

## 1. Review Scope

- Branch: `dev-dycp-prefill-mla`
- Review baseline: current `HEAD -> working tree` diff
- Scope includes:
  - tracked code changes
  - untracked design/test artifacts
  - one local runtime log file

This report focuses on three questions:

1. What functionality this patch is trying to implement.
2. What each major code change is doing.
3. Which parts look sound, and which parts still carry risk.

## 2. Executive Summary

This patch is mainly implementing **Dynamic CP (DyCP) prefill support for MLA models** under the v1 engine path.

The central design idea is:

- **KV cache residency** should be sized by the **local per-rank token residency** after CP sharding.
- **profile run / activation / temporary workspace** should still be sized by a more conservative upper bound, because short non-CP requests may still run locally on one rank.

In practice, the patch tries to align the following modules to one DyCP-aware model:

- KV cache sizing
- block table width and slot mapping
- cross-DP scheduler token budgeting
- worker-side token reshaping and restore
- attention metadata
- MLA prefill execution path
- profile memory reservation

The overall direction is coherent and matches the design note in `docs/design/dynamic_cp_kv_cache_memory.md`. The strongest parts of the patch are the KV cache accounting changes and the worker-side buffer/profile separation. The highest-risk parts are still in scheduler thresholding, KV restore safety, and mixed DyCP+DP batch handling.

## 2.1 Chinese Summary

这批改动的主目标，是把 **MLA 场景下的 DyCP prefill 路径** 从零散兼容，补成一套比较完整的运行时链路。

可以把它理解成 4 条主线：

1. **KV cache 常驻容量重算**
   - 以前很多地方只按 `pcp * dcp` 计算。
   - 现在开始把 `dycp` 也纳入进去，目标是让 KV residency 真正反映本 rank 本地会常驻多少 token。

2. **Scheduler 从全局 token budget 改成 per-rank budget**
   - CP 请求不是只消耗一个全局 token 池，而是会同时占用多个 rank 的预算。
   - 这次改动尝试让 scheduler 的预算模型更接近真实执行模型。

3. **Worker 侧补齐 DyCP/PCP 的 token reshape、restore、slot mapping**
   - 新增 `PCPManager` 统一管理 padding、DualChunkSwap、all-gather restore、padded slot mapping。
   - 这是这批改动里最像“基础设施”的部分。

4. **MLA attention 路径补齐 metadata 和 mixed-batch 处理**
   - `mla/common.py` 现在能感知 DyCP 请求前缀、普通 DP 请求后缀，以及 PCP/DyCP 的 query/KV restore 信息。
   - 同时也引入了这批 patch 最大的行为风险，因为 mixed batch 的语义被改动了。

这批改动里，我认为**方向最对**的部分是：

- KV residency 口径统一到 `pcp * dcp * dycp`
- block table 跟着缩小
- profile run 和常驻 KV 分开处理
- `PCPManager` 把 worker 侧 reshape/restore 逻辑集中起来

我认为**风险最高**的部分是：

- `cross_dp_scheduler.py` 把长请求阈值硬编码成 `4 * 1024`
- `pcp_kv_allgather_and_restore()` 在 restore index 不匹配时选择“静默修补”而不是 fail fast
- `mla/common.py` 只要 batch 里出现 DyCP 请求，就把整个 batch 都按 prefill 语义处理

## 3. Diff Overview

### 3.1 Changed Files

Tracked files in the current diff:

- `vllm/attention/backends/abstract.py`
- `vllm/benchmarks/datasets.py`
- `vllm/model_executor/layers/fused_moe/config.py`
- `vllm/model_executor/layers/fused_moe/layer.py`
- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/flashinfer.py`
- `vllm/v1/attention/backends/mla/common.py`
- `vllm/v1/attention/backends/mla/flashattn_mla.py`
- `vllm/v1/attention/backends/mla/flashmla.py`
- `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`
- `vllm/v1/attention/backends/utils.py`
- `vllm/v1/core/cross_dp_kv_cache_manager.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/core/sched/cross_dp_scheduler.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/kv_cache_interface.py`
- `vllm/v1/worker/block_table.py`
- `vllm/v1/worker/cp_utils.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/gpu_worker.py`

Untracked files observed during review:

- `docs/design/dynamic_cp_kv_cache_memory.md`
- `tests/v1/test_dynamic_cp_memory.py`
- `chenxiao/v2lite_.txt`

### 3.2 Size of the Patch

`git diff --stat HEAD`:

- 20 tracked files changed
- 2676 insertions
- 341 deletions

The patch is not a narrow single-bug fix. It is a cross-layer runtime change touching scheduling, memory planning, worker input preparation, metadata construction, and the MLA attention kernel path.

## 4. Design Intent

The design note already states the intended model clearly:

- KV cache is a **resident state** and should use `ceil(max_model_len / effective_cp_world_size)`.
- profile run is a **peak execution estimate** and should remain conservative under DyCP.

The patch is effectively translating that design into code.

The intended effective CP factor is:

```text
effective_cp_world_size =
    max(1, pcp_world_size)
  * max(1, dcp_world_size)
  * max(1, dycp_world_size)
```

and the intended local KV upper bound is:

```text
local_max_model_len = ceil(max_model_len / effective_cp_world_size)
```

That is the main architectural theme tying many files together.

## 5. Detailed Changes by Area

### 5.1 KV Cache Residency Is Unified to `pcp * dcp * dycp`

**Files**

- `vllm/v1/kv_cache_interface.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/worker/block_table.py`
- `vllm/v1/engine/core.py`

**Representative code**

```python
def get_cp_kv_cache_world_size(vllm_config: VllmConfig) -> int:
    parallel_config = vllm_config.parallel_config
    return (
        max(parallel_config.decode_context_parallel_size, 1)
        * max(parallel_config.prefill_context_parallel_size, 1)
        * max(parallel_config.dp_per_domain, 1)
    )


def get_cp_kv_cache_model_len(vllm_config: VllmConfig) -> int:
    max_model_len = vllm_config.model_config.max_model_len
    cp_world_size = get_cp_kv_cache_world_size(vllm_config)
    if cp_world_size > 1:
        return cdiv(max_model_len, cp_world_size)
    return max_model_len
```

```python
class FullAttentionSpec(AttentionSpec):
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = get_cp_kv_cache_model_len(vllm_config)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes
```

**What changed**

- Full-attention KV memory sizing is no longer based only on `pcp * dcp`.
- DyCP factor `dp_per_domain` is now part of the residency model.
- KV cache logging now multiplies the local number back by the full CP factor, so the reported global-equivalent token count remains understandable.
- `block_table.py` now uses a `total_cp_world_size` and `total_cp_rank` that include DyCP as well.
- `engine/core.py` uses `block_size * dcp * pcp * dycp` as the CP hash block size.

**Why it matters**

This is the most important correctness change in the patch. Before this, the runtime could shrink local KV layout for some paths but still size memory or block indexing with an older CP factor. That creates subtle mismatches between:

- how many KV tokens the rank thinks it can residently store
- how many blocks the block table thinks each request may need
- how many tokens the scheduler thinks can run

This patch moves those pieces toward one shared accounting model.

### 5.2 Cross-DP Scheduler Starts Tracking Per-Rank Token Budgets

**File**

- `vllm/v1/core/sched/cross_dp_scheduler.py`

**Representative code**

```python
rank_budgets = [self.max_num_scheduled_tokens] * self.cp_world_size

def _get_effective_budget(cp_ranks: list[int]) -> int:
    cp_size = len(cp_ranks)
    if cp_size > 1:
        return min(rank_budgets[r] for r in cp_ranks) * cp_size
    return rank_budgets[cp_ranks[0]]

def _deduct_budget(cp_ranks: list[int], num_tokens: int) -> None:
    cp_size = len(cp_ranks)
    per_rank_cost = (num_tokens + cp_size - 1) // cp_size
    for r in cp_ranks:
        rank_budgets[r] -= per_rank_cost
```

**What changed**

- Scheduler no longer relies on a single monolithic `token_budget`.
- Each CP rank tracks its own remaining token budget.
- CP requests consume token budget on every participating rank.
- Short DP requests can be assigned to the rank with the largest remaining budget instead of the rank with the fewest requests.
- End-of-round validation now checks effective per-rank scheduled tokens.

**Why it matters**

This is a necessary change once requests may be split across multiple ranks. A global budget can hide overload on a single rank even if the aggregate number still looks valid. Per-rank budgeting makes the scheduling model match the actual execution model more closely.

**Review note**

This same file also hardcodes:

```python
long_request_threshold = 4 * 1024
```

That is the highest-severity issue found in this review, because the KV residency model now depends on when requests switch into CP mode. If the threshold is later than the local KV residency upper bound, some requests can still enter the single-rank path while local KV memory has already been sized as if they would be split.

### 5.3 Worker-Side PCP/DyCP Token Shaping Is Centralized in `PCPManager`

**Files**

- `vllm/v1/worker/cp_utils.py`
- `vllm/v1/worker/gpu_model_runner.py`

**Representative code**

```python
class PCPManager:
    def update_tokens_for_pcp(
        self,
        num_scheduled_tokens: np.ndarray,
        arange_np: np.ndarray,
        num_reqs: int,
        reorder_batch_threshold: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
```

```python
if self.pcp_world_size > 1:
    self.pcp_manager = PCPManager(...)
elif self.dycp_world_size > 1:
    self.pcp_manager = PCPManager(
        self.dycp_world_size,
        self.dycp_rank,
        max_buffer_num_tokens,
        self.max_num_reqs,
        self.device,
        self.pin_memory,
        max_pre_division_tokens=max_pre_div_tokens,
    )
```

**What changed**

- A new manager class now owns the logic for:
  - DualChunkSwap-style token padding
  - local token positions
  - all-gather restore indices
  - padded slot mappings
  - hidden-state restore after all-gather
- `gpu_model_runner.py` now allocates larger buffers in DyCP mode:
  - post-division buffers for actual local execution
  - pre-division buffers for restore/index construction
- During `_prepare_inputs()`, the worker rewrites:
  - `num_scheduled_tokens`
  - `positions`
  - `slot_mapping`
  - `discard_request_mask`
  - `logits_indices`
  - `cp_local_seq_lens`

**Why it matters**

Without a shared owner for these transforms, the runtime would end up duplicating shape logic across:

- scheduler assumptions
- worker-side CPU preprocessing
- metadata builder
- post-forward restore

The new `PCPManager` is one of the patch's strongest changes because it gathers those concerns into one place.

### 5.4 Attention Metadata Is Generalized from DCP-Only to CP-Aware

**Files**

- `vllm/v1/attention/backends/utils.py`
- `vllm/attention/backends/abstract.py`
- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/flashinfer.py`

**Representative code**

```python
class CommonAttentionMetadata:
    cp_local_seq_lens: torch.Tensor | None = None
    cp_local_seq_lens_cpu: torch.Tensor | None = None
    dcp_local_seq_lens: torch.Tensor | None = None
    dcp_local_seq_lens_cpu: torch.Tensor | None = None
    pcp_allgather_restore_idx: torch.Tensor | None = None
    dycp_local_seq_lens: torch.Tensor | None = None
    dycp_local_seq_lens_cpu: torch.Tensor | None = None
    num_dycp_reqs: int = 0
    num_dycp_tokens: int = 0
```

```python
def get_cp_local_seq_lens(
    seq_lens: torch.Tensor,
    cp_world_size: int = 1,
    cp_rank: int | None = None,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor:
```

**What changed**

- Metadata now has explicit CP-aware fields instead of relying only on DCP naming.
- `get_dcp_local_seq_lens()` is kept as a compatibility wrapper, but the real logic is now in `get_cp_local_seq_lens()`.
- FlashAttention and FlashInfer builders now consume the generalized helper.
- The base abstract attention impl exposes `cp_world_size` / `cp_rank` aliases and enables decode LSE return for DyCP as well.

**Why it matters**

This is how the patch moves DyCP support from being an MLA-only local hack into a model that other attention backends can at least understand at the metadata layer.

### 5.5 MLA Metadata Builder Gains DyCP/PCP-Specific Layout Logic

**File**

- `vllm/v1/attention/backends/mla/common.py`

**Representative code**

```python
class MLACommonPrefillMetadata:
    @dataclass
    class PCPMetadata:
        kv_head_indices: torch.Tensor | None = None
        kv_tail_indices: torch.Tensor | None = None
        query_head_indices: torch.Tensor | None = None
        query_tail_indices: torch.Tensor | None = None
        output_restore_idx: torch.Tensor | None = None
```

```python
def split_metadata(
    attn_metadata: MLACommonMetadata,
) -> tuple[MLACommonMetadata | None, MLACommonMetadata | None]:
```

**What changed**

- MLA prefill metadata now carries extra PCP/DyCP information:
  - query/KV split indices
  - output restore indices
  - DyCP request counts
- New builder logic computes:
  - mixed DyCP + DP prefill layouts
  - local chunk starts / chunk lengths for DyCP prefixes
  - per-rank context lengths across CP ranks
- A new `split_metadata()` helper splits a prefill-only mixed batch into:
  - DyCP prefix metadata
  - DP suffix metadata

**Why it matters**

MLA is the real execution hotspot of this patch. The runtime needs not only the local token count, but also a consistent description of:

- which requests belong to the DyCP prefix
- how those requests are chunked locally
- how query/KV slices are restored after gather

This file contains the bridge between worker-side preprocessing and kernel-side attention execution.

### 5.6 MLA Forward Path Separates "Cache Write" from "Attention Compute"

**Files**

- `vllm/v1/attention/backends/mla/common.py`
- `vllm/v1/attention/backends/mla/flashattn_mla.py`
- `vllm/v1/attention/backends/mla/flashmla.py`
- `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`

**Representative code**

```python
local_k_c_normed = k_c_normed[:num_actual_toks, ...]
local_k_pe = k_pe[:num_actual_toks, ...]
local_slot_mapping = attn_metadata.slot_mapping.flatten()

ops.concat_and_cache_mla(
    local_k_c_normed[:cache_tokens],
    local_k_pe[:cache_tokens].squeeze(1),
    kv_cache,
    local_slot_mapping[:cache_tokens],
    kv_cache_dtype=self.kv_cache_dtype,
    scale=layer._k_scale,
)
```

```python
if self.pcp_world_size > 1:
    k_c_normed, k_pe = pcp_kv_allgather_and_restore(...)
elif self.dycp_world_size > 1 and full_dycp_prefill:
    k_c_normed, k_pe = pcp_kv_allgather_and_restore(...)
```

**What changed**

- Local KV is always written to cache with local slot mapping first.
- Cross-rank all-gathered KV is used only for attention computation.
- Full-DyCP prefill can now all-gather KV across the DyCP group.
- Mixed batches can be split and executed in two phases:
  - DyCP collective path first
  - local DP path second
- Backend-specific metadata classes were updated from `dcp_tot_seq_lens` to `cp_tot_seq_lens`.

**Why it matters**

This is the execution-level implementation of the design idea:

- local rank owns local KV storage
- all-gather is a temporary compute-time reconstruction, not the resident cache layout

That distinction is necessary for memory correctness.

### 5.7 Profile Run and Temporary All-Gather Memory Are Split Apart

**Files**

- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/gpu_worker.py`

**Representative code**

```python
def get_dycp_allgather_reserve_bytes(self) -> int:
    if self.dycp_world_size <= 1:
        return 0
    num_cp_seqs = self.scheduler_config.num_cp_seqs
    if num_cp_seqs <= 0:
        return 0
    head_size = self.model_config.get_head_size()
    dtype_bytes = self.dtype.itemsize
    per_token_bytes = head_size * dtype_bytes
    pre_div_tokens = num_cp_seqs * self.max_model_len
    extra_tokens = max(0, pre_div_tokens - self.max_num_tokens)
    return extra_tokens * per_token_bytes
```

```python
def _get_profile_num_tokens(self) -> int:
    if self.dycp_world_size > 1:
        return self.max_num_tokens
    if self.pcp_world_size > 1:
        return cdiv(self.max_num_tokens, self.pcp_world_size)
    return self.max_num_tokens
```

**What changed**

- DyCP profile run now stays conservative and profiles with full local scheduler budget.
- Additional DyCP all-gather reserve bytes are deducted from available KV cache memory.
- PCP profile run still shrinks by `pcp_world_size`.

**Why it matters**

This is exactly the right conceptual split for DyCP:

- KV residency gets smaller under CP sharding.
- peak execution memory does not necessarily get smaller, because not every request is guaranteed to run split.

This part of the patch is consistent with the design note and is one of the more convincing pieces of the implementation.

### 5.8 Cross-DP KV Manager Is Adjusted to Match Worker-Side Padding Policy

**File**

- `vllm/v1/core/cross_dp_kv_cache_manager.py`

**Representative code**

```python
def _avg_distribute_tokens_to_ranks(
    self,
    world_size: int,
    seq_len: int,
    cp_kv_cache_interleave_size: int = 1,
) -> list[int]:
    if world_size <= 1:
        return [seq_len]

    num_padded_tokens = cdiv(seq_len, 2 * world_size) * (2 * world_size)
    local_seq_len = num_padded_tokens // world_size
    return [local_seq_len for _ in range(world_size)]
```

**What changed**

- CP rank token distribution is now aligned to the same DualChunkSwap padding rule as the worker path.

**Why it matters**

If the manager allocates blocks with one per-rank width model but the worker later computes slot mapping with another, block counts and actual writes can drift apart. This change is trying to remove that inconsistency.

### 5.9 Secondary Support Changes

**Files**

- `vllm/model_executor/layers/fused_moe/config.py`
- `vllm/model_executor/layers/fused_moe/layer.py`
- `vllm/benchmarks/datasets.py`
- `vllm/v1/engine/core.py`

**What changed**

- Fused MoE parallel config now exposes `dycp_size` / `dycp_rank`.
- Benchmark random dataset can read `(input_len, output_len)` pairs from a local JSON file.
- `EngineCoreProc` ignores `START_DP_WAVE` when it is not a DP-specific core.

**Why it matters**

- MoE changes are mostly state propagation and compatibility prep.
- benchmark change is a local experiment convenience feature.
- `START_DP_WAVE` handling avoids spurious crashes in shared control paths.

## 6. Untracked Files and Their Meaning

### 6.1 `docs/design/dynamic_cp_kv_cache_memory.md`

This file explains the intended architecture in plain language and is useful. It clearly states the difference between:

- KV residency memory
- profile run peak memory

This design note strengthens the patch because the code changes are not random; they are implementing a documented plan.

### 6.2 `tests/v1/test_dynamic_cp_memory.py`

This test covers two important invariants:

- full-attention KV budget uses all CP factors: `pcp * dcp * dycp`
- `MultiGroupBlockTable` request width also includes DyCP

The test coverage is small but meaningful. It validates two of the highest-value accounting changes.

### 6.3 `chenxiao/v2lite_.txt`

This is a runtime log from a local validation run. It shows the author was testing:

- DeepSeek-V2-Lite
- chunked prefill
- DyCP scheduling
- MLA debug logs

This file should not be committed as part of the patch itself.

## 7. Main Review Findings

### Finding 1: Hardcoded DyCP long-request threshold

**File**

- `vllm/v1/core/sched/cross_dp_scheduler.py`

**Issue**

`long_request_threshold` is hardcoded to `4 * 1024`.

**Why risky**

After this patch, KV memory is sized using local CP residency. That means the scheduler must switch requests into CP mode no later than the local KV upper bound. A fixed 4K threshold is not tied to:

- `max_model_len`
- `pcp_world_size`
- `dcp_world_size`
- `dycp_world_size`

Under many configurations this threshold can be wrong, causing requests to remain on the single-rank path after local KV capacity has already been reduced.

### Finding 2: Restore index mismatch is silently patched instead of failing fast

**File**

- `vllm/v1/attention/backends/utils.py`

**Issue**

`pcp_kv_allgather_and_restore()` truncates, pads with identity, and clamps restore indices when the length does not match the gathered token count.

**Why risky**

`pcp_allgather_restore_idx` is not harmless shape metadata. It is the actual permutation that reconstructs KV order. If it is wrong, the attention result can become silently wrong rather than crashing early.

### Finding 3: Mixed batches with any DyCP request are forced into prefill semantics

**File**

- `vllm/v1/attention/backends/mla/common.py`

**Issue**

If `dycp_world_size > 1 and num_dycp_reqs > 0`, the builder forces:

- `num_decodes = 0`
- `num_prefills = num_reqs`

**Why risky**

This avoids collective divergence, but it also changes runtime semantics:

- normal decode requests in the same batch are no longer treated as decode
- decode-specific kernels and LSE correction paths may be bypassed

This is a functional behavior change, not just a metadata trick.

## 8. Secondary Cleanup Issues

- `vllm/v1/attention/backends/mla/common.py` contains many `chenxiao--debug` `logger.info(...)` calls on hot paths.
- `vllm/benchmarks/datasets.py` contains direct `print(...)` calls.
- `git diff --check HEAD` reports trailing whitespace in `vllm/benchmarks/datasets.py`.

These do not define the architecture, but they should be cleaned before merge.

## 9. Overall Assessment

### 9.1 What Looks Good

- KV residency accounting is much more internally consistent than before.
- Block table and KV memory planning are now aligned on DyCP.
- Worker-side preprocessing has a real owner (`PCPManager`) instead of scattered ad hoc logic.
- Profile run versus resident KV separation is conceptually correct.
- MLA forward path now clearly distinguishes local cache write from gathered compute-only KV.

### 9.2 What Still Looks Fragile

- the scheduler threshold for entering CP mode
- restore-index error handling
- mixed-batch decode/prefill semantics under DyCP

### 9.3 Merge Readiness

If this branch is meant for experimentation or internal validation, it already captures the intended architecture well.

If it is meant for a production merge, the three main review findings should be resolved first, and the debug/logging leftovers should be cleaned up.

## 10. Validation Status

I attempted to run:

```powershell
python -m pytest tests\v1\test_dynamic_cp_memory.py -q
```

but the current environment does not have `pytest` installed:

```text
No module named pytest
```

So this report is based on code review, diff analysis, and consistency checking, not an executed test pass.
