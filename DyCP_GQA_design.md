# GQA DyCP Prefill Design

## 1. 目标和约束

本次修改严格遵循 `DyCP_GQA.md` 的约束，并参考以下两个 commit 的实现思路：

- `233e389b00c363db1ac445d2df0c4e5fdf93319c`
- `4d0479178b5e0c03ee0b1fa057ad7f583b42331b`

目标是为 GQA 格式 KV cache 补齐 DyCP prefill 支持，且满足以下边界：

- 仅面向 prefill 场景，不以 decode 请求为主要目标。
- 支持 PD 分离、以及单机按 batch 粒度做 PD 分开计算的部署。
- 一个 batch 内允许 DyCP 请求和普通 DP 请求混合。
- DyCP 请求必须位于 batch 前 `num_dycp_reqs` 个位置，这一点继续复用现有 MLA 路径已经依赖的 batch reorder 约束。
- 支持 chunk prefill。
- 暂不支持 prefix cache；该组合现在显式报错，不做静默降级。
- 不允许把 DyCP 请求偷偷退化成普通 DP 路径来“规避问题”。

## 2. 总体方案

MLA 的关键做法不是“把 DyCP 信息塞进同一个 metadata 里”，而是：

1. 对 mixed DyCP/DP prefill batch 预先切出两个独立子 batch。
2. 为 DyCP 子 batch 和 DP 子 batch 分别构造 metadata。
3. forward 时按 token 连续区间分别执行两次 attention。

这次 GQA 侧沿用同样的思路：

- `flash_attn`：增加 mixed DyCP/DP prefill 的 split metadata，并在 forward 中分两次调用 attention kernel。
- `flashinfer`：同样增加 split metadata 和 split forward，但由于它内部有可复用 wrapper / paged-kv 规划状态，不能直接递归复用同一个 builder，因此额外实现了独立的 split-prefill metadata 构造逻辑，保证 DyCP/DP 两部分不会互相覆盖 plan 结果。

## 3. 文件级修改说明

### 3.1 `vllm/v1/attention/backends/utils.py`

新增 `slice_common_attn_metadata(...)`。

作用：

- 从一个 `CommonAttentionMetadata` 中切出 request/token 子区间。
- 保留 DyCP 相关字段：
  - `num_dycp_reqs`
  - `num_dycp_tokens`
  - `cp_local_seq_lens`
  - `dcp_local_seq_lens`
  - `dycp_local_seq_lens`
- 显式 clone `seq_lens` 和 CPU 侧镜像，避免 DyCP 子 metadata 在本地化 `seq_lens` 时回写污染原 batch metadata。

这样做的原因：

- GQA 的两个 backend 都需要把 mixed batch 切成 DyCP 子 batch 和 DP 子 batch。
- 如果每个 backend 自己手写切片逻辑，很容易把 `seq_lens` 或 DyCP 字段切错，或者在 builder 内部原地修改时污染原 metadata。

### 3.2 `vllm/v1/attention/backends/flash_attn.py`

修改点分两部分。

#### A. `FlashAttentionMetadata`

新增两个隐藏字段：

- `_dycp_split`
- `_dp_split`

作用：

- 当 batch 是 mixed DyCP/DP prefill 时，不再依赖一个混合 metadata 直接跑完整 batch。
- 改为把两份已经构造好的子 metadata 挂在主 metadata 上，由 forward 决定分两次执行。

#### B. `FlashAttentionMetadataBuilder.build`

新增 mixed DyCP/DP prefill 判定：

- 仅在 `dycp_world_size > 1`
- `num_decodes == 0`
- `0 < prefill_num_dycp_reqs < num_prefills`

满足时：

1. 用 `slice_common_attn_metadata` 切出 DyCP 子 batch 和 DP 子 batch。
2. 分别递归构造两份子 metadata。
3. 返回一个只作为“容器”的主 metadata，把两份 split metadata 挂进去。

同时新增前缀缓存保护：

- 若 `DyCP prefill + prefix cache` 组合出现，直接抛 `NotImplementedError`。
- 这是显式限制，不是静默 fallback。

额外处理：

- 对递归构造出来的 `scheduler_metadata` 做 `.clone()`，避免两个 split metadata 共享 builder 内部的同一块 scheduler buffer，被后一次 build 覆盖。

#### C. `FlashAttentionImpl.forward`

如果 metadata 含有 `_dycp_split/_dp_split`：

1. 先对 DyCP token 前缀执行一次 forward。
2. 再对 DP token 后缀执行一次 forward。
3. 输出按原 token 顺序写回。

这样保证：

- DyCP 请求继续走带 collective 的路径。
- DP 请求继续走纯本地路径。
- 两者不会在同一个 attention call 中共享调度信息或计算路径。

### 3.3 `vllm/v1/attention/backends/flashinfer.py`

这是本次实现里最需要小心的部分。

#### A. `FlashInferMetadata`

新增：

- `_dycp_split`
- `_dp_split`

以及一组内部引用字段：

- `_paged_kv_indptr_cpu_ref`
- `_paged_kv_indptr_gpu_ref`
- `_paged_kv_indices_ref`
- `_paged_kv_last_page_len_cpu_ref`

原因：

- mixed split 子 metadata 需要拥有各自独立的 paged-kv 规划结果。
- `flashinfer` wrapper 的 plan 依赖这些张量。
- 如果只在局部作用域里创建它们，不保留引用，存在被 Python 提前回收的风险。

#### B. `FlashInferMetadataBuilder._build_split_prefill_metadata`

新增专门 helper，为 mixed batch 的单侧子 batch 构造纯 prefill metadata。

之所以单独写这个 helper，而不是像 `flash_attn` 那样直接递归复用 `self.build(...)`，原因是：

- `flashinfer` builder 内部复用了 singleton wrapper。
- 同一个 builder 连续 plan 两次，会让 DP 子 batch 的 plan 覆盖 DyCP 子 batch 的 plan。
- 同时 `paged_kv_indptr / paged_kv_indices / paged_kv_last_page_len` 也会被后一次覆盖。

这个 helper 的做法是：

1. 基于子 batch 独立计算 `seq_lens_cpu`、`paged_kv_indptr`、`paged_kv_indices`、`last_page_len`。
2. 为该子 batch 单独创建 wrapper，并单独执行 `plan(...)`。
3. 返回一份纯 prefill metadata，并把相关张量引用保存在 metadata 里。

这样可以保证：

- DyCP 子 batch 和 DP 子 batch 拥有完全独立的 flashinfer plan 结果。
- 两次 forward 串行执行时共享 workspace buffer，但不共享 plan 状态。

#### C. `FlashInferMetadataBuilder.build`

与 `flash_attn` 一样，新增 mixed DyCP/DP prefill 识别和 split metadata 构造。

但真正构造子 metadata 时调用的是：

- `_build_split_prefill_metadata(dycp_common_attn_metadata)`
- `_build_split_prefill_metadata(dp_common_attn_metadata)`

而不是递归复用原 build。

同样增加：

- `DyCP prefill + prefix cache` 的显式报错。

#### D. `FlashInferImpl.forward`

如果 metadata 含有 `_dycp_split/_dp_split`：

1. 先跑 DyCP token 前缀。
2. 再跑 DP token 后缀。
3. 保持输出顺序不变。

这样 mixed batch 下 GQA flashinfer 的执行方式就和 MLA 一致了：DyCP 和 DP 分别 forward。

### 3.4 `tests/v1/attention/test_attention_splitting.py`

新增两个 CPU 可跑单测：

- `test_slice_common_attn_metadata_preserves_dycp_fields`
- `test_slice_common_attn_metadata_clones_seq_lens`

覆盖点：

- 切片后 DyCP/CP 相关字段是否保留正确。
- 切片后的 `seq_lens` 修改是否会污染原 metadata。

这两项是本次 mixed split 正确性的基础保障。

### 3.5 `tests/v1/attention/test_dycp_gqa_attention.py`

新增 backend 级别的 attn 修正单测，各覆盖一个核心断言：

- `flash_attn`：mixed DyCP/DP prefill metadata 的 forward 结果，应等于
  “DyCP 子 metadata forward + DP 子 metadata forward” 的拼接结果。
- `flashinfer`：同样验证 mixed split forward 与子 batch forward 拼接等价。

测试策略不是跑真实 kernel，而是：

1. 保留真实的 split 逻辑。
2. 用轻量 mock 替换底层 kernel / wrapper。
3. 让 DyCP 子 batch 和 DP 子 batch 返回不同的伪输出偏移量。

这样如果：

- token 边界切错，
- 子 batch 顺序拼错，
- mixed batch 没有真正按 split metadata 分开执行，

测试都会直接失败。

## 4. 精度与性能考虑

### 4.1 精度

本次没有通过“关闭某条路径”来规避问题。

实际行为是：

- DyCP 子 batch 仍然使用 DyCP 本地 KV + collective 合并的原始逻辑。
- DP 子 batch 仍然使用 GQA backend 原有的本地 prefill 逻辑。
- 只是把 mixed batch 从“一次混合 forward”改成了“两次独立 forward”。

因此不会把 DyCP 请求错误地下沉成普通 DP，也不会让 DP 请求错误地携带 DyCP 语义。

### 4.2 性能

这次 split 主要避免两类性能问题：

- DyCP 和 DP 共用一次计划/调度时，DyCP 的局部 KV 布局和 DP 的普通布局彼此掺杂。
- `flashinfer` mixed batch 共用单个 wrapper plan 时，会让一个子 batch 的计划覆盖另一个子 batch，既有正确性风险，也会破坏稳定性能。

新的实现中：

- `flash_attn` 为 DyCP / DP 各跑一次，避免混合调度。
- `flashinfer` 为 DyCP / DP 分别持有独立 plan，但仍共享 workspace，避免不必要的全局退化。

## 5. 明确不支持的组合

以下组合本次显式不支持：

- `GQA DyCP prefill + prefix cache`

处理方式：

- 在 builder 阶段直接抛出 `NotImplementedError`。
- 这样可以避免静默给出错误结果，也符合“暂时不考虑支持 prefix cache”的范围定义。

## 6. 验证计划

已执行：

- `python -m py_compile vllm/v1/attention/backends/utils.py vllm/v1/attention/backends/flash_attn.py vllm/v1/attention/backends/flashinfer.py tests/v1/attention/test_attention_splitting.py`
- `python -m py_compile tests/v1/attention/test_dycp_gqa_attention.py`

尝试执行但当前环境缺失：

- `python -m pytest tests/v1/attention/test_attention_splitting.py -q`
  - 当前环境报错：`No module named pytest`

建议在具备测试依赖和 GPU/flashinfer 环境后补跑：

- attention splitting 单测
- GQA flash_attn mixed DyCP/DP prefill 回归
- GQA flashinfer mixed DyCP/DP prefill 回归
- chunk prefill 场景回归
