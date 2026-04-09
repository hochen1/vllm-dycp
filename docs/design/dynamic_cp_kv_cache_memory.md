# Dynamic CP KV Cache Memory Plan

## 场景和目标

目标场景是 P/D 分离下的 Prefill worker：开启 dynamic CP（DyCP），Prefill 开启 chunked prefill，长请求在到达 `max_model_len` 前一定会被调度器划入 CP 路径。这里需要把两类显存分开处理：

- KV cache 显存：按“一个请求最终在本 rank 常驻多少 KV token”分配。
- profile run 显存：按“一个调度 iteration 在本 rank 可能产生多大的 activation / workspace 峰值”预留。

这两者不能用同一个除法。KV cache 是跨 chunk 累积的常驻状态；profile run 是启动期用 dummy batch 测出来的非 KV 峰值。

## KV Cache 分配策略

Full attention KV cache 使用本 rank 的本地最大序列长度：

```text
effective_cp_world_size =
    max(1, pcp_world_size)
  * max(1, dcp_world_size)
  * max(1, dycp_world_size)

local_max_model_len = ceil(max_model_len / effective_cp_world_size)
num_pages_per_request = ceil(local_max_model_len / block_size)
```

因此 `kv_cache_interface.py` 中在 `pcp*dcp*dycp > 1` 时把 `max_model_len` 除掉是正确方向。原因是 dynamic CP 的长请求阈值由我们控制，能保证一个真正走到 `max_model_len` 的请求早已进入 CP 路径；本 rank 不应该再按完整 `max_model_len` 常驻 KV。

这个策略有一个必须满足的运行期不变量：不会走 CP 的请求，其最大 prompt / chunk resident 长度必须小于等于 `local_max_model_len`。换句话说，dynamic CP 的长请求阈值要不晚于本地 KV 上界，否则一个“尚未切 CP”的中长请求会先要求单 rank 存下超过本地上界的 KV。

chunked prefill 不改变这个结论。它只限制单步 prefill 计算量，不能把 full attention 的最终 KV residency 压到 `max_num_batched_tokens`；一个长请求会跨多个 chunk 累积 KV，最终本 rank 的上界仍是 `ceil(max_model_len / effective_cp_world_size)`。

## Profile Run 显存策略

profile run 测的是非 KV 峰值，不应该直接套 KV cache 的 `max_model_len / cp_world_size` 公式。动态 CP 和静态 PCP 的差别是：

- 静态 PCP：prefill 请求整体按 PCP rank 切分，profile run 可以用 `ceil(max_num_batched_tokens / pcp_world_size)`。
- 动态 CP：只有被调度器判定为 CP 的长请求会切分；短请求和普通 DP 请求仍可能在单个 rank 上以非 CP 形态运行。

启动期 `profile_run()` 没有真实的 `SchedulerOutput`，也没有真实 request 的 `num_cp_request`，因此不能按某个 batch 区分“是否有 CP 请求”。它只能基于拓扑和我们选择的 synthetic profile case 做上界估计。

因此当前执行策略采用保守上界：DyCP 开启时，profile run 仍按本 rank 的完整 `max_num_batched_tokens` 进行，以覆盖短请求/非 CP 请求的 activation 峰值；不把它盲目除以 `dycp_world_size`。这样会比“所有请求都切 CP”的理想模型多预留一些非 KV 显存，但避免启动时多分 KV、运行时在短请求 batch 上 OOM。

后续如果把 CP 请求阈值显式配置化，并在调度器中维护每 rank 的 token 预算，可以进一步把 DyCP profile run 收紧为：

```text
profile_tokens = max(
  ceil(cp_chunk_tokens / dycp_world_size),
  non_cp_short_path_peak_tokens,
  mixed_cp_and_non_cp_peak_tokens,
)
```

在这之前，保守 profile 是更稳的落地方案。

## 已执行的代码计划

1. 把 full-attention KV 本地长度计算集中到 `get_cp_kv_cache_world_size()` / `get_cp_kv_cache_model_len()`，避免散落的 `pcp*dcp*dycp` 公式。
2. `FullAttentionSpec.max_memory_usage_bytes()` 使用本地 CP KV 长度。
3. KV cache 日志中的 token 数恢复展示乘回 `pcp*dcp*dycp` 后的全局等效容量，并把 dyCP 纳入日志。
4. worker 侧 `MultiGroupBlockTable` 的 `max_num_blocks_per_req` 使用 `pcp*dcp*dycp`，避免 KV tensor 已按 DyCP 缩小但 block table 仍按 `pcp*dcp` 计算。
5. scheduler 的 CP hash block size 纳入 `dp_per_domain`，与 KV cache residency 口径对齐。
6. profile run 增加独立 token 预算 helper：静态 PCP 使用 `ceil`，动态 CP 保留完整本地 profile 预算。
7. 增加 CPU 单测覆盖 full-attention KV 预算和 block table 的 dyCP request width。
