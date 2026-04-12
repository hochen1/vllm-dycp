# GQA DyCP Prefill Design

## Overview

This document describes the GQA-side adaptation of DyCP prefill for the
disaggregated PD deployment path. The design intentionally mirrors the MLA
prefill DyCP implementation where possible, but keeps the scope narrow:

- PD-separated deployment only
- prefill-only batch on the prefill instance
- full DyCP prefill batch only
- no pre-existing local context for the DyCP requests

The implementation currently targets the V1 GQA attention backends:

- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/flashinfer.py`


## Problem Statement

Before this change, GQA had DyCP support for decode-oriented execution, but
prefill did not correctly handle the DyCP-local query layout produced by the
worker-side DualChunkSwap scheduling logic.

A naive "run local prefill and then all-reduce outputs" approach is not
sufficient for GQA prefill:

- DyCP prefill reorders each request into a local head/tail query shard.
- Each local query token must attend to a different prefix of the full KV
  sequence.
- The correct causal mask therefore depends on the DualChunkSplit layout, not
  just on the final local token count.

MLA already solves this by:

1. building query/KV head-tail indices,
2. gathering KV across the DyCP group,
3. running two causal attention passes,
4. restoring the local output order.

The GQA implementation follows the same high-level strategy.


## Goals

- Reuse the existing DyCP worker-side metadata prepared by `GPUModelRunner`
  and `PCPManager`.
- Support correct GQA prefill attention under the DyCP DualChunkSwap layout.
- Keep KV cache writes local.
- Limit the change to prefill-only PD use cases to reduce regression risk.


## Non-Goals

- Mixed DyCP + non-DyCP prefill batches.
- Prefill batches with both decode and prefill requests.
- Existing context / chunked-context DyCP prefill.
- Decode-path behavior changes.
- New scheduler policy or worker-side partitioning changes.


## Preconditions

The specialized DyCP GQA prefill branch is enabled only when all of the
following are true:

- `dycp_world_size > 1`
- the batch is prefill-only (`num_decodes == 0`)
- the whole prefill batch is DyCP (`num_prefills == num_dycp_reqs`)
- all prefill tokens belong to DyCP requests
  (`num_prefill_tokens == num_dycp_tokens`)
- there is no pre-existing local context for those requests
  (`seq_lens_cpu == query_lens_cpu` for the DyCP prefill prefix)
- `pcp_allgather_restore_idx` is available from worker-side metadata

If any condition is not satisfied, the code falls back to the existing backend
path.


## Worker-Side Inputs Reused

The design depends on worker-side DyCP prefill metadata that already exists
after the MLA work:

- local DyCP query layout in `query_start_loc`
- local slot mapping for cache writes
- `num_dycp_reqs`
- `num_dycp_tokens`
- `pcp_allgather_restore_idx`

No worker-side scheduling logic is changed here.


## Metadata Design

Both GQA backends add a small `DyCPPrefillMetadata` structure with:

- `query_head_indices`
- `query_tail_indices`
- `kv_head_indices`
- `kv_tail_indices`
- `output_restore_idx`

`FlashInfer` additionally stores CPU `cu_seqlens` tensors for the ragged
wrapper:

- `q_head_cu_seq_lens_cpu`
- `q_tail_cu_seq_lens_cpu`
- `kv_head_cu_seq_lens_cpu`
- `kv_tail_cu_seq_lens_cpu`

The parent backend metadata also stores:

- `num_dycp_tokens`
- `pcp_allgather_restore_idx`
- `dycp_prefill_metadata`

`FlashInferMetadata` also carries a reusable
`BatchPrefillWithRaggedKVCacheWrapper` handle for this path.


## Builder Flow

### FlashAttention

In `FlashAttentionMetadataBuilder.build()`:

1. Detect the pure DyCP prefill/no-context case.
2. Build head/tail query indices with `get_pcp_query_indices(...)`.
3. Build head/tail KV indices with `get_pcp_kv_indices(...)`.
4. Store the indices in `DyCPPrefillMetadata`.
5. Skip the normal DyCP scheduler-metadata path for this branch.

The backend then runs a dedicated prefill forward path instead of the generic
varlen paged-KV path.


### FlashInfer

In `FlashInferMetadataBuilder.build()`:

1. Detect the same pure DyCP prefill/no-context case.
2. Build the same DualChunk indices.
3. Build CPU `cu_seqlens` tensors for head/tail query and KV shards.
4. Skip the extra DyCP `seq_lens_cpu` repartitioning step for this branch.
5. Force `prefill_use_trtllm = False` because this path uses explicit
   dual-chunk ragged prefill, not TRTLLM prefill.

Skipping the second `get_cp_local_seq_lens(...)` pass is important. For the
specialized DyCP prefill branch, `seq_lens_cpu` is already local after the
worker-side token partition and must not be partitioned again.


## Forward Path

### Common Logic

Both backends use the same high-level flow:

1. Write local K/V into the cache with the local DyCP slot mapping.
2. Gather local prefill K/V tensors across the DyCP group with
   `pcp_kv_allgather_and_restore(...)`.
3. Run two causal attention passes:
   - head chunk
   - tail chunk
4. Concatenate the two outputs.
5. Restore the original local query order with `output_restore_idx`.

Cache update remains local. Cross-rank gathering is used only for attention
compute.


### FlashAttention Path

`FlashAttentionImpl` adds `_forward_with_dycp_prefill(...)`, which:

- gathers raw prefill K/V tensors across the DyCP group,
- runs `flash_attn_varlen_func(...)` on head and tail shards separately,
- restores the final local output layout.

This branch bypasses the normal paged-KV prefill path because the DyCP
DualChunk layout is expressed directly through the explicit index lists.


### FlashInfer Path

`FlashInferImpl` adds `_forward_with_dycp_prefill(...)`, which:

- gathers raw prefill K/V tensors across the DyCP group,
- uses `BatchPrefillWithRaggedKVCacheWrapper` for the head pass,
- replans the same wrapper for the tail pass,
- restores the final local output layout.

This avoids depending on paged-KV planning for the dual-chunk KV layout.


## Why DualChunk Is Required

Suppose one request is globally `[0, 1, 2, 3]` and DyCP world size is 2.

The local query shard is not a contiguous prefix:

- rank 0 may get query tokens like `[0, 3]`
- rank 1 may get query tokens like `[1, 2]`

Those local query tokens require different causal KV ranges:

- the head token attends to a short prefix
- the tail token attends to a much longer prefix

Therefore the backend must:

- gather the full logical KV ordering,
- slice the correct head and tail KV subsets,
- run two separate causal attention calls.

This is exactly the same correctness requirement that MLA already handles.


## Scope Limits and Fallbacks

The specialized GQA DyCP prefill path does **not** currently handle:

- mixed DyCP + DP prefill batches
- DyCP prefill with existing local context
- chunked-context DyCP prefill
- mixed prefill/decode batches

Those cases continue to use the existing code paths.


## Validation Strategy

Recommended validation for this feature:

1. Single-request DyCP prefill correctness against non-DyCP baseline.
2. Multi-request pure DyCP prefill correctness with variable prompt lengths.
3. FlashAttention and FlashInfer backend parity for the supported scope.
4. PD-separated end-to-end run with DyCP-enabled prefill worker.
5. Regression checks for:
   - fp16 / bf16 model dtype
   - different block sizes
   - prompt lengths with odd/even local shard lengths

At the time of this change, only Python-level compile validation was run on the
modified files. Distributed correctness and performance validation still need to
be executed on GPU hardware.


## Future Work

- Extend GQA DyCP prefill to mixed DyCP + non-DyCP prefill batches.
- Support DyCP prefill with pre-existing context.
- Add dedicated distributed tests for the DualChunk GQA prefill path.
- Evaluate whether common DualChunk metadata should be shared across MLA and
  GQA backends to reduce duplication.
