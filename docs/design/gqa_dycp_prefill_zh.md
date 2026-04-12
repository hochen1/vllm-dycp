# GQA DyCP Prefill 设计说明

## 背景

这份文档说明 GQA 路径下 DyCP prefill 的适配设计。整体思路尽量复用之前 MLA 的 DyCP prefill 方案，但本次实现故意把范围收窄，只覆盖当前最需要的场景：

- 仅考虑 PD 分离部署
- 仅考虑 prefill 实例上的 prefill-only batch
- 仅考虑整批都是 DyCP prefill 请求
- 仅考虑这些 DyCP 请求在本地没有已有 context 的情况

当前实现覆盖的 v1 GQA backend 为：

- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/flashinfer.py`


## 问题定义

在这次改动之前，GQA 对 DyCP 的支持主要偏向 decode 路径，而 prefill 路径没有正确适配 worker 侧已经生成好的 DyCP DualChunkSwap 本地 query 布局。

GQA 的 DyCP prefill 不能简单理解成：

1. 每个 rank 用本地 query 跑 attention
2. 最后把输出做一次 all-reduce

这样做是不对的，原因在于：

- DyCP prefill 下，每个请求的 query 会被切成 local head/tail 两段
- 本地 query token 的顺序不是简单连续前缀
- 每个本地 query token 应该看到的 causal KV 前缀长度不同
- 所以 attention 的因果掩码依赖 DualChunk 的切分方式，而不是只依赖“本地 token 数量”

MLA 的 DyCP prefill 已经解决了这个问题，核心步骤是：

1. 构造 query 的 head/tail 索引
2. 构造 KV 的 head/tail 索引
3. 跨 DyCP group gather 并 restore KV
4. 分别对 head/tail 跑两次 causal attention
5. 最后 restore 输出顺序

这次 GQA 的适配沿用了同样的整体思路。


## 目标

- 复用 `GPUModelRunner` 和 `PCPManager` 已经准备好的 DyCP prefill 元数据
- 让 GQA 在 DyCP DualChunkSwap 布局下能够正确执行 prefill attention
- 保持 KV cache 写入仍然是本地行为
- 把改动限制在 prefill-only 的 PD 分离场景内，降低回归风险


## 非目标

本次不覆盖以下场景：

- 混合 DyCP + 非 DyCP 的 prefill batch
- 同时包含 decode 和 prefill 的混合 batch
- 已经带有本地 context 的 DyCP prefill
- chunked-context 的 DyCP prefill
- decode 路径逻辑调整
- 调度器策略变更
- worker 侧 token 划分逻辑变更


## 触发条件

GQA 的 DyCP prefill 专用分支只会在以下条件全部满足时启用：

- `dycp_world_size > 1`
- batch 是 prefill-only，也就是 `num_decodes == 0`
- 整个 prefill batch 都是 DyCP 请求，即 `num_prefills == num_dycp_reqs`
- 全部 prefill token 都属于 DyCP 请求，即 `num_prefill_tokens == num_dycp_tokens`
- 这些 DyCP 请求在本地没有已有 context，即 `seq_lens_cpu == query_lens_cpu`
- worker 侧已经提供了 `pcp_allgather_restore_idx`

只要有任意一个条件不满足，就回退到原有 backend 路径，不走这次新增的 DualChunk 分支。


## 复用的 Worker 侧输入

这次设计依赖于 MLA 那次 DyCP prefill 改动后已经存在的 worker 侧能力，包括：

- DyCP 本地 query 布局对应的 `query_start_loc`
- 本地 KV cache 写入用的 `slot_mapping`
- `num_dycp_reqs`
- `num_dycp_tokens`
- `pcp_allgather_restore_idx`

本次没有修改 worker 侧调度和 token 划分策略。


## 元数据设计

两个 GQA backend 都新增了一份小的 `DyCPPrefillMetadata`，包含：

- `query_head_indices`
- `query_tail_indices`
- `kv_head_indices`
- `kv_tail_indices`
- `output_restore_idx`

其中：

- `query_head_indices` / `query_tail_indices` 用于表示本地 query 的 DualChunk 切分
- `kv_head_indices` / `kv_tail_indices` 用于表示 gather 完整 KV 后，每段 query 对应可见的 causal KV 范围
- `output_restore_idx` 用于把两段 attention 输出拼回原本本地 query 顺序

此外，父级 backend metadata 里还补充了：

- `num_dycp_tokens`
- `pcp_allgather_restore_idx`
- `dycp_prefill_metadata`

`FlashInfer` 额外保存了 ragged wrapper 所需的 CPU 侧 `cu_seqlens`：

- `q_head_cu_seq_lens_cpu`
- `q_tail_cu_seq_lens_cpu`
- `kv_head_cu_seq_lens_cpu`
- `kv_tail_cu_seq_lens_cpu`

以及一个可复用的：

- `dycp_prefill_wrapper`


## Builder 设计

### FlashAttention

在 `FlashAttentionMetadataBuilder.build()` 中：

1. 检测是否满足“pure DyCP prefill + no-context”条件
2. 用 `get_pcp_query_indices(...)` 生成 query 的 head/tail 索引
3. 用 `get_pcp_kv_indices(...)` 生成 KV 的 head/tail 索引
4. 把这些索引保存到 `DyCPPrefillMetadata`
5. 对这个专用分支跳过原本 DyCP 的 scheduler metadata 路径

这样 forward 阶段可以直接走专门的 DualChunk prefill 分支，而不是继续复用普通的 varlen paged-KV prefill 路径。


### FlashInfer

在 `FlashInferMetadataBuilder.build()` 中：

1. 检测同样的“pure DyCP prefill + no-context”条件
2. 构造 query/KV 的 head/tail 索引
3. 为 ragged wrapper 额外构造 head/tail 的 CPU `cu_seqlens`
4. 对这个专用分支跳过额外一次 `get_cp_local_seq_lens(...)`
5. 显式把 `prefill_use_trtllm` 置为 `False`

第 4 点非常重要。因为在这个专用分支里，worker 侧的 token 已经完成 DyCP-local 划分，如果再做一次 `get_cp_local_seq_lens(...)`，就会把已经切好的本地长度再次切分一遍，导致 page planning 错误。

第 5 点也很重要。这个路径实际使用的是显式的 dual-chunk ragged prefill，而不是 TRTLLM prefill，所以 metadata 不能继续标记成 TRTLLM prefill。


## Forward 设计

### 通用流程

两个 backend 的专用 DyCP prefill 流程都是：

1. 先按本地 `slot_mapping` 把本地 K/V 写入 cache
2. 对当前 prefill 的本地 K/V 做 `all_gather + restore`
3. 分别运行两次 causal attention：
   - head chunk
   - tail chunk
4. 把两段输出拼接起来
5. 用 `output_restore_idx` 恢复为本地 query 原始顺序

注意：

- KV cache 更新仍然是本地写入
- 跨 rank gather 只服务于 attention 计算，不改变 cache update 语义


### FlashAttention 路径

`FlashAttentionImpl` 中新增 `_forward_with_dycp_prefill(...)`，主要逻辑是：

- 对原始 prefill K/V 调用 `pcp_kv_allgather_and_restore(...)`
- 基于 head/tail 索引分别调用两次 `flash_attn_varlen_func(...)`
- 把两段输出 restore 回本地 query 顺序

这个分支绕过了普通的 paged-KV prefill 路径，因为 DualChunk 的 causal 语义已经通过显式索引表达出来了。


### FlashInfer 路径

`FlashInferImpl` 中新增 `_forward_with_dycp_prefill(...)`，主要逻辑是：

- 对原始 prefill K/V 调用 `pcp_kv_allgather_and_restore(...)`
- 使用 `BatchPrefillWithRaggedKVCacheWrapper` 执行 head pass
- 复用同一个 wrapper 重新 plan 后执行 tail pass
- 最后 restore 输出顺序

这个实现避免继续依赖 paged-KV planner 去表达 DualChunk 的 KV 布局。


## 为什么一定要 DualChunk

以一个全局请求 `[0, 1, 2, 3]`、DyCP world size = 2 为例。

本地 query 分布不是简单连续前缀，例如：

- rank 0 可能拿到 `[0, 3]`
- rank 1 可能拿到 `[1, 2]`

这些 token 在 causal attention 下对应的可见 KV 范围是不同的：

- head token 只能看到较短前缀
- tail token 需要看到更长前缀

所以 backend 必须：

- 先恢复完整逻辑 KV 顺序
- 再按照 head/tail 规则切出对应 KV 子集
- 分两次运行 causal attention

这也是为什么不能简单做一个 output all-reduce。


## 当前限制

当前这条 GQA DyCP prefill 路径明确不支持：

- 混合 DyCP + 非 DyCP prefill batch
- 带已有 context 的 DyCP prefill
- chunked-context DyCP prefill
- mixed prefill/decode batch

这些情况仍然回退到现有路径。


## Review 结果

在本次实现后的 review 中，发现并修复了一个真实问题：

- `FlashInfer` 在 pure DyCP prefill 分支下，原本仍然会继续执行一次
  `get_cp_local_seq_lens(...)`
- 这会把已经 DyCP-local 的长度再次切分，导致 paged KV planning 出错

修复方式是：

- 专用 DyCP prefill 分支跳过这一步
- 并强制 `prefill_use_trtllm = False`

除此之外，在当前支持范围内，没有再发现明显的 correctness 问题。


## 验证建议

建议后续至少补以下验证：

1. 单请求 DyCP prefill 对齐非 DyCP baseline
2. 多请求 pure DyCP prefill，覆盖不同 prompt 长度
3. `FlashAttention` 与 `FlashInfer` 两条 backend 结果对齐
4. PD 分离端到端联调，确认 prefill worker 的 DyCP 路径正确
5. 覆盖 odd/even 长度请求，确保 head/tail 分段边界正确

本次我实际完成的校验只有：

- 修改文件的 Python 级别语法检查

还没有做真实 GPU 分布式 correctness 和性能测试。


## 后续工作

- 扩展到 mixed DyCP + DP prefill batch
- 支持带已有 context 的 DyCP prefill
- 增加专门的分布式测试
- 评估是否把 MLA/GQA 的 DualChunk 元数据进一步抽象成共享逻辑，减少重复实现
