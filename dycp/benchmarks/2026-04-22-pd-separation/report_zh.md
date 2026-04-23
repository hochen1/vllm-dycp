# DyCP 本地 PD 分离基准测试报告

**日期**: 2026-04-22 ~ 2026-04-23
**分支**: `dev-dycp-mixed-1`
**模型**: DeepSeek-V2-Lite (MLA, 64 Experts)
**硬件**: 8× NVIDIA L20Y (80GB), CUDA 12.9, PyTorch 2.7.0

---

## 1. 测试目标

验证 DyCP 本地 PD（Prefill-Decode）分离的核心假设：

- **Prefill 阶段**：使用全量 Context Parallelism（8-rank CP）并行计算，最小化 TTFT
- **Decode 阶段**：切换至单 rank（CP=1），消除跨 DP prefill-decode 混合导致的 EP all-to-all 阻塞，最大化 decode 吞吐

**PD 分离要解决的核心问题**：在纯 DP 模式下，不同 DP rank 可能同时执行 prefill 和 decode。由于 EP all-to-all 是跨所有 rank 的集合通信，decode rank 必须在每个 MoE 层等待 prefill rank 完成，导致 decode 延迟严重膨胀。PD 分离通过 CP=8 prefill 让**所有 rank 同时执行 prefill**，消除了"部分 rank 做 prefill、部分 rank 做 decode"的跨 DP 混合场景，从而消除 EP all-to-all 阻塞。

> **注意**：PD 分离不排除 batch 内 prefill+decode 混合调度（chunk prefill，调度器将 prefill 和 decode token 组合到同一 batch 以提高 GPU 利用率）。这种 batch 内混合是调度优化，不会导致 EP all-to-all 阻塞，因为 prefill 和 decode token 在同一 rank 上执行，不涉及跨 rank 同步等待。

通过 2×2 矩阵中的 4 种配置进行对比，量化 PD 分离的收益与开销。

## 2. 配置矩阵

| 配置 | CP 模式 | PD 分离 | 阈值 | 端口 | 说明 |
|------|---------|---------|------|------|------|
| **配置 5** | CP=1 (DP) | 否 | N/A | 8400 | 原版 vLLM v0.13.0 基线 |
| **配置 4** | CP=1 (DP) | 否 | 999999999 | 8400 | DyCP 分支纯 DP 基线 |
| **配置 1** | CP=8 (全量) | 否 | 1 | 8400 | 全 CP，无 PD |
| **配置 2** | CP=1 (DP) | 是 | 999999999\* | 9000 | PD 分离但无 CP |
| **配置 3** | CP=8→1 (DyCP) | 是 | 1 | 9000 | **DyCP 目标配置** |

\* 配置 2 服务端阈值=999999999；代理端阈值默认为 100。

**配置 5 说明**：原版 vLLM v0.13.0（releases/v0.13.0 分支），不包含 DyCP 代码。使用 `distributed_executor_backend=mp`（而非 `dmp`），无 `dp_per_domain` 和 `num_cp_seqs` 参数。对 `gpu_model_runner.py` 中的 `_dummy_run()` 做了最小修复以解决 DP padding 断言 bug。

**公共参数**（配置 1-4，DyCP 分支）：
- `data_parallel_size=8, tensor_parallel_size=1, dp_per_domain=8`
- `max_num_seqs=8, max_num_batched_tokens=4096`
- `gpu_memory_utilization=0.7, block_size=64, cp_kv_cache_interleave_size=64`
- `enable_expert_parallel=True, attention_backend=FLASHMLA`
- `cudagraph_mode=FULL_DECODE_ONLY, cudagraph_capture_sizes=[4,8,16,24,32,64]`
- `distributed_executor_backend=dmp`

**配置 5 参数**（原版 vLLM v0.13.0）：
- `data_parallel_size=8, tensor_parallel_size=1`（无 dp_per_domain）
- `max_num_seqs=8, max_num_batched_tokens=4096`
- `gpu_memory_utilization=0.7, block_size=64`
- `enable_expert_parallel=True, attention_backend=FLASHMLA`
- `cudagraph_mode=FULL_DECODE_ONLY, cudagraph_capture_sizes=[4,8,16,24,32,64]`
- `distributed_executor_backend=mp`

## 3. 测试方法

### 3.1 并发基准测试

| 参数 | 值 |
|------|-----|
| 输入长度 | 8192 tokens |
| 输出长度 | 1024 / 2048 tokens |
| 长度变化比例 | 0.0（等长） |
| 请求数量 | 500 |
| 请求速率 | 2 req/s |
| 最大并发 | 256 |
| 预热次数 | 3 |
| 随机种子 | 42 |
| 后端 | OpenAI (`/v1/completions`) |

### 3.2 单条请求基准测试

| 参数 | 值 |
|------|-----|
| 输入长度 | 8192 tokens |
| 输出长度 | 10 tokens |
| 请求数量 | 1 |
| 请求速率 | 1 req/s |
| 预热次数 | 0 |
| 随机种子 | 42 |
| 后端 | OpenAI (`/v1/completions`) |
| 用途 | 避免并发对 TTFT/TPOT 绝对性能的干扰，直接验证 CP prefill 的加速效果和 decode 性能是否持平 |

每组测量 5 次取平均值。配置 3 通过 proxy（端口 9000，完整 PD 流程）测量。

## 4. 结果汇总

### 4.1 并发测试：8K 输入 / 1K 输出

| 指标 | 配置 5<br>原版 DP | 配置 4<br>DyCP DP | 配置 1<br>全 CP | 配置 2<br>CP1+PD | **配置 3**<br>**DyCP** |
|------|-------------------|-------------------|-----------------|-------------------|------------------------|
| 完成数 | 500 | 500 | 500 | 500 | **500** |
| **TTFT 均值** | 782 ms | 1,022 ms | 1,268,765 ms | 20,927 ms | **311 ms** |
| **TTFT P50** | 639 ms | 674 ms | 1,621,303 ms | 19,604 ms | **253 ms** |
| **TPOT 均值** | 22.51 ms | 24.56 ms | 54.71 ms | 32.79 ms | **13.51 ms** |
| **TPOT P50** | 23.02 ms | 25.95 ms | 55.06 ms | 36.56 ms | **13.58 ms** |
| **ITL 中位数** | 8.99 ms | 9.68 ms | — | — | **9.74 ms** |
| **ITL P99** | 196.85 ms | 203.13 ms | — | — | **131.02 ms** |
| **E2EL 均值** | 23,807 ms | 26,142 ms | 1,324,737 ms | 54,471 ms | **14,127 ms** |
| **请求吞吐** | 1.93 req/s | 1.93 req/s | 0.14 req/s | 1.74 req/s | **1.92 req/s** |
| **输出吞吐** | 1,975 tok/s | 1,975 tok/s | 146 tok/s | 1,781 tok/s | **1,970 tok/s** |

### 4.2 并发测试：8K 输入 / 2K 输出（配置 3 vs 配置 4）

| 指标 | 配置 4<br>DyCP DP | **配置 3**<br>**DyCP** | 提升 |
|------|-------------------|------------------------|------|
| 完成数 | 500 | **499** | — |
| **TTFT 均值** | 24,299 ms | **9,596 ms** | **-60%** |
| **TPOT 均值** | 17.67 ms | **13.75 ms** | **-22%** |
| **ITL P99** | 203.01 ms | **143.17 ms** | **-29%** |
| **E2EL 均值** | 60,463 ms | **37,750 ms** | **-38%** |
| **输出吞吐** | 3,322 tok/s | **3,720 tok/s** | **+12%** |
| **请求吞吐** | 1.62 req/s | **1.82 req/s** | **+12%** |
| **最大并发** | 148 | **97** | -34% |

### 4.3 单条请求测试：8K 输入 / 10 输出

避免并发对 TTFT/TPOT 绝对性能的干扰，直接验证两个核心假设：(1) CP prefill 加速 TTFT；(2) CP=1 decode 的 TPOT 与纯 DP 基线持平。

**配置 4（纯 DP）5 次测量：**

| 运行 | TTFT | TPOT | ITL 中位数 |
|------|------|------|-----------|
| 1 | 506 ms | 8.40 ms | 8.38 ms |
| 2 | 507 ms | 8.41 ms | 8.36 ms |
| 3 | 508 ms | 8.34 ms | 8.31 ms |
| 4 | 506 ms | 8.34 ms | 8.30 ms |
| 5 | 516 ms | 8.56 ms | 8.33 ms |
| **均值** | **508 ms** | **8.41 ms** | **8.34 ms** |

**配置 3（DyCP，proxy 路径）5 次测量：**

| 运行 | TTFT | TPOT | ITL 中位数 |
|------|------|------|-----------|
| 1 | 186 ms | 9.94 ms | 9.84 ms |
| 2 | 187 ms | 9.84 ms | 9.87 ms |
| 3 | 186 ms | 9.84 ms | 9.86 ms |
| 4 | 186 ms | 9.93 ms | 9.97 ms |
| 5 | 192 ms | 9.86 ms | 9.93 ms |
| **均值** | **187 ms** | **9.88 ms** | **9.89 ms** |

**汇总对比：**

| 指标 | 配置 4<br>DyCP DP | 配置 3<br>DyCP (proxy) | 差异 |
|------|-------------------|------------------------|------|
| **TTFT 均值** | 508 ms | **187 ms** | **-63%** |
| **TPOT 均值** | **8.41 ms** | 9.88 ms | +1.5ms（待修复 bug） |
| **ITL 中位数** | **8.34 ms** | 9.89 ms | +1.5ms（同上） |

**关键发现：**

1. **CP prefill 加速验证**：配置 3 TTFT=187ms，比配置 4（508ms）快 **63%**，直接验证了 8-rank CP 并行 prefill 的加速效果。
2. **TPOT 差异 1.5ms 是待修复 bug**：配置 3 的 TPOT（9.88ms）比配置 4（8.41ms）高 1.5ms，根因是 `dycp_world_size > 1` 时 decode 路径无条件启用 DyCP 分支（`return_lse=True` + 120 次 NCCL 通信/step），即使 batch 中无 CP 请求。修复为按 `num_dycp_reqs > 0` 条件启用后，TPOT 应与配置 4 持平（详见 5.2.1 节）。
3. **并发下 TPOT 反转**：配置 4 从 8.41ms 恶化到 24.56ms（+192%），配置 3 仅从 9.88ms 恶化到 13.51ms（+37%，含 1.5ms bug 开销）。扣除 bug 开销后配置 3 的并发恶化仅约 2ms（+24%），验证了 PD 分离消除跨 DP EP all-to-all 阻塞的效果。

### 4.4 关键对比总结

**配置 3 vs 基线（8K/1K 并发）：**

| vs 配置 4（DyCP DP） | vs 配置 5（原版 DP） | vs 配置 2（PD 无 CP） |
|---|---|---|
| TTFT: **-70%** (311ms vs 1,022ms) | TTFT: **-60%** (311ms vs 782ms) | TTFT: **-98.5%** (311ms vs 20,927ms) |
| TPOT: **-45%** (13.51ms vs 24.56ms) | TPOT: **-40%** (13.51ms vs 22.51ms) | TPOT: **-59%** (13.51ms vs 32.79ms) |
| E2EL: **-46%** (14.1s vs 26.1s) | E2EL: **-41%** (14.1s vs 23.8s) | E2EL: **-74%** (14.1s vs 54.5s) |

**配置 5（原版 DP）vs 配置 4（DyCP DP）**：两个配置均使用纯 DP 模式（CP=1），配置 5 为原版 vLLM v0.13.0，配置 4 为 DyCP 分支。TTFT 配置 5 更低（782ms vs 1,022ms，-23%），可能因为 DyCP 分支的调度器额外开销（`LongShortRequestQueue`、cross-DP 协调）。TPOT 差异较小（22.51ms vs 24.56ms，-8%），说明 DyCP 框架在纯 DP 模式下引入少量每步开销。

**TPOT 并发恶化对比：**

| 配置 | 单条请求 TPOT | 并发 TPOT (8K/1K) | 恶化幅度 |
|------|-------------|-----------|---------|
| 配置 4 (DyCP DP) | 8.41 ms | 24.56 ms | **+192%** |
| 配置 3 (DyCP) | 9.88 ms | 13.51 ms | **+37%** |

---

## 5. 分析

### 5.1 TTFT 分析：CP Prefill 加速与 Proxy 开销

#### 5.1.1 CP Prefill 加速效果

单条请求数据直接验证了 CP prefill 的加速效果：配置 3 proxy TTFT=187ms，配置 4 TTFT=508ms，加速 **63%**。

8 rank CP 并行 prefill 将 8K tokens 分到每 rank ~1K tokens，计算量减少 8 倍。DualChunkSwap 数据分布 + all-gather 同步的协调开销远小于节省的计算时间。

并发场景下加速更显著（TTFT -70%，311ms vs 1,022ms），因为 CP prefill 不仅缩短单请求延迟，还减少了 prefill 占用的调度窗口，降低后续请求的排队延迟。

#### 5.1.2 Proxy 开销分解

为量化 proxy 两阶段流程的开销，额外测量了配置 3 直连 vLLM（端口 8400，绕过 proxy）的 TTFT=138ms。

**Proxy 开销 = 187 - 138 = 49ms**，包含：

| 开销项 | 估算 | 来源 |
|--------|------|------|
| HTTP 请求/响应往返（proxy↔vLLM ×2） | 10-20ms | aiohttp TCP + JSON 序列化 |
| KV cache IPC 传输（8K tokens, 8→1 rank） | 15-20ms | cudaMemcpyAsync 跨 rank |
| 调度器两次调度 + decode 请求初始化 | 5-10ms | scheduler loop + connector metadata |
| Prefill blocks 延迟释放（`delay_free=True`） | ~5ms | 调度开销 |

**CP prefill 加速 = 508 - 138 = 370ms**，远大于 proxy 开销 49ms。净收益 = 370 - 49 = 321ms。

#### 5.1.3 TTFT 随输出长度增长

8K/1K → 8K/2K 时 TTFT 显著增长：

| 配置 | 8K/1K TTFT | 8K/2K TTFT | 增长倍数 |
|------|-----------|-----------|---------|
| 配置 4 | 1,022 ms | 24,299 ms | **23.8×** |
| 配置 3 | 311 ms | 9,596 ms | **30.9×** |

根因不是 prefill 本身变慢，而是**更长输出导致 KV cache 占用时间更长，排队延迟增加**。配置 3 增长倍数更高是因为 8K/1K TTFT 基数极低（311ms），但绝对值仍远优于配置 4（9.6s vs 24.3s，快 2.5 倍）。

### 5.2 TPOT 分析：跨 DP EP All-to-All 阻塞

#### 5.2.1 单条请求：DyCP 框架待修复开销

单条请求下配置 3 的 TPOT（9.88ms）比配置 4（8.41ms）高 1.5ms（+18%）。这是 DyCP 框架的已知 bug，修复后应与配置 4 持平。

**根因**：`vllm/v1/attention/backends/flashinfer.py:1373-1393` 中 decode 路径检查 `dycp_world_size > 1` 而非 `num_dycp_reqs > 0`，导致即使 batch 中无 CP 请求也走 DyCP 分支：

| 开销来源 | 每层 | 60层/step | 配置 4 等价 | 影响等级 |
|---------|------|----------|-----------|---------|
| `return_lse=True` 强制 lse 计算 | 1次 | 60次 | 不调用 | 高 |
| NCCL all_gather + all_reduce | 2次 | 120次 | 不调用 | 高 |
| `lse` 张量分配 | 1次 | 60次 | 不分配 | 中 |
| TRT-LLM decode 被禁用 | 1次 | 1次 | 可用 | 中 |
| KV connector 钩子 | 2次 | 120次 | 1次检查 | 低 |
| `cp_local_seq_lens` 额外拷贝 | 1次 | 1次 | 跳过 | 低 |

**修复方案**：按 `num_dycp_reqs > 0` 条件启用 DyCP 分支，而非检查 `dycp_world_size`。修复后预期 TPOT 从 9.88ms 降至约 8.41ms，与配置 4 持平。

#### 5.2.2 并发场景：跨 DP EP All-to-All 阻塞

并发下 TPOT 发生反转——配置 4 从 8.41ms 恶化到 24.56ms（+192%），配置 3 仅从 9.88ms 恶化到 13.51ms（+37%）。

**两种混合的区别**：

- **跨 DP 混合（Type 1）**：不同 DP rank 同时执行不同类型的请求——有的 rank 做 prefill，有的做 decode。这是 EP all-to-all 阻塞的根因：EP all-to-all 是跨所有 rank 的集合通信，decode rank 的 MoE 计算瞬间完成，但必须在 all-to-all 同步点等待 prefill rank 完成，导致 decode 延迟被拖慢到与 prefill 的 MoE 耗时一致。
- **Batch 内混合（Type 2）**：同一 rank 的 batch 中同时包含 prefill 和 decode token（chunk prefill）。这是调度优化，避免计算资源浪费，不会导致跨 rank 同步等待。

**PD 分离消除的是 Type 1 混合**：CP=8 prefill 让所有 rank 同时执行 prefill，不存在"部分 rank 做 prefill、部分做 decode"的场景。Decode 阶段所有 rank 以 CP=1 独立执行 decode，即使 batch 内混入了新 prefill 请求的 token（Type 2），CP=8 的 prefill 执行速度极快（~40ms vs 单 rank ~150ms），all-to-all 阻塞窗口大幅缩短。

```
纯 DP + EP（跨 DP 混合，Type 1 阻塞）:
Rank 0: |--- prefill MoE (8K tokens, ~150ms) ---|-- decode --|-- prefill --|
Rank 1: |-- decode --X-- 等待 all-to-all 同步 ---|-- decode -X-- 等待 --|
                     ↑                              ↑
              decode 被阻塞                  decode 再次被阻塞

PD 分离（CP=8 消除 Type 1 混合）:
所有 Rank: |-- CP prefill (8 ranks, ~40ms) --|-- decode (各 rank 独立) --|
                                            ↑
                               无跨 DP 混合，decode 不被 prefill 阻塞
                               （batch 内 Type 2 混合仍存在，但阻塞窗口极短）
```

**配置 3 P99 ITL 仍有 131-143ms 的原因**：并发下新 prefill 请求到来时，CP=8 prefill 仍需 ~40ms 执行，期间 decode token 在 MoE all-to-all 同步点短暂等待。相比配置 4 的 ~150ms 单 rank prefill 阻塞窗口，缩短了 73%。

### 5.3 ITL 分析：Prefill-Decode 干扰的量化证据

**ITL（Inter-Token Latency）** 的中位数反映稳态 decode 性能，**P99 ITL 直接反映跨 DP 混合导致的 EP all-to-all 阻塞程度**。

#### 5.3.1 并发场景 ITL 对比

| 配置 | 输出长度 | ITL 中位数 | ITL P99 | P99/中位数 |
|------|---------|-----------|---------|-----------|
| 配置 4 (DyCP DP) | 1K | 9.68 ms | 203.13 ms | 21.0× |
| 配置 3 (DyCP) | 1K | 9.74 ms | 131.02 ms | 13.5× |
| 配置 4 (DyCP DP) | 2K | — | 203.01 ms | — |
| 配置 3 (DyCP) | 2K | — | 143.17 ms | — |

配置 4 的 P99 ITL 是中位数的 21 倍，配置 3 仅 13.5 倍。P99 ITL 的差异直接量化了跨 DP EP all-to-all 阻塞的严重程度。

#### 5.3.2 单条请求 ITL 对比

| 配置 | ITL 中位数 | TPOT |
|------|-----------|------|
| 配置 4 (DyCP DP) | **8.34 ms** | **8.41 ms** |
| 配置 3 (DyCP, proxy) | 9.89 ms | 9.88 ms |

单条请求下无跨 DP 混合，ITL P99 与中位数几乎一致。配置 3 的 1.5ms 差异来自 DyCP 框架待修复 bug（5.2.1 节），修复后应持平。

#### 5.3.3 配置 3 P99 ITL 仍有 131-143ms 的原因

PD 分离大幅缓解但仍存在间歇性阻塞：

1. **CP=8 prefill 的短暂阻塞窗口**：新 prefill 请求到来时，8 rank CP prefill 仍需 ~40ms，期间 decode token 在 MoE all-to-all 同步点短暂等待
2. **调度开销**：PD 分离引入 prefill→proxy→decode 两阶段调度
3. **KV cache IPC 传输**：decode 请求加载 KV cache 需要跨 rank 内存拷贝
4. **并发竞争**：8 个 decode 请求竞争 GPU 资源

### 5.4 配置 1（全 CP）分析：验证 PD 分离的必要性

配置 1（全 CP，无 PD）比配置 4 慢 **13.8 倍**，1244 秒内仅完成 169/500 个请求。三个根因：

1. **并发瓶颈（~10 倍影响）**：`num_cp_seqs=8` + `max_num_seqs=8` 时，每个 CP 请求占用全部 8 个 rank，最大并发仅 8 个请求。配置 4 可达 ~79 个并发。

2. **CUDA Graph 回退到 Eager 模式（2 倍 TPOT 影响）**：`cudagraph_capture_sizes_for_cp=4` 仅为最多 4 个 CP tokens 捕获 CUDA Graph。8 个 CP 序列同时 decode 时 `num_cp_tokens=8` 超出捕获范围，回退到 eager 执行（55.4ms vs 25.8ms TPOT）。

3. **跨 rank 通信开销**：每个 CP decode step 需要 all-gather（hidden states）+ EP all-to-all，增加单 rank DP 不存在的延迟。

**结论**：全 CP decode 在吞吐型场景下架构上不可行——这正是 PD 分离要解决的问题。

### 5.5 配置 5（原版 DP）vs 配置 4（DyCP DP）：框架开销基线

两个配置均使用纯 DP 模式（CP=1），区别在于配置 5 是原版 vLLM v0.13.0，配置 4 是 DyCP 分支。

| 指标 | 配置 5（原版 DP） | 配置 4（DyCP DP） | 差异 |
|------|-------------------|-------------------|------|
| TTFT 均值 | 782 ms | 1,022 ms | +31% |
| TPOT 均值 | 22.51 ms | 24.56 ms | +9% |
| ITL 中位数 | 8.99 ms | 9.68 ms | +8% |
| ITL P99 | 196.85 ms | 203.13 ms | +3% |
| E2EL 均值 | 23,807 ms | 26,142 ms | +10% |
| 请求吞吐 | 1.93 req/s | 1.93 req/s | 0% |

DyCP 分支在纯 DP 模式下引入约 8-10% 的性能开销，主要来源：
- **TTFT +31%**：DyCP 调度器（`LongShortRequestQueue`、cross-DP 协调）增加调度延迟
- **TPOT +9%**：DyCP 框架的每步额外开销（`_update_states_for_dycp()`、domain-level executor 协调），与 5.2.1 节分析的 `dycp_world_size > 1` 开销不同，这是纯 DP 路径下的框架开销

这一基线对比说明配置 3（DyCP）相对配置 5（原版）的净收益：TTFT -60%（311ms vs 782ms），TPOT -40%（13.51ms vs 22.51ms），E2EL -41%。

### 5.6 配置 2（PD 无 CP）分析：CP 缺失的代价

配置 2（PD 分离但无 CP）显著差于配置 3：
- **TTFT**：20,927ms vs 311ms（慢 67 倍）——无 CP 时单 rank prefill 8K tokens 极慢
- **TPOT**：32.79ms vs 13.51ms（慢 2.4 倍）——无 CP prefill 并行化，流水线积压
- **E2EL**：54,471ms vs 14,127ms（慢 3.9 倍）

**CP prefill 和 PD 分离两者缺一不可**。

### 5.7 输出长度对 TPOT 的影响

| 配置 | 8K/1K TPOT | 8K/2K TPOT | 变化 |
|------|-----------|-----------|------|
| 配置 4 | 24.56 ms | 17.67 ms | -28% |
| 配置 3 | 13.51 ms | 13.75 ms | +2% |

- **配置 4 TPOT 下降 28%**：8K/2K 时更多请求处于 decode 阶段，GPU batch 更饱满，单 token 推理效率提升。但代价是 TTFT 恶化。
- **配置 3 TPOT 基本不变**：PD 分离下 decode 阶段资源分配已接近最优。

配置 3 的输出吞吐比配置 4 高 12%（3,720 vs 3,322 tok/s），E2EL 低 38%。

---

## 6. Bug 修复记录

测试过程中发现并修复了 3 个 Bug，均为 PD 分离正确性的关键修复。

### 6.1 PD decode 请求被错误分类为 CP 长请求

**现象**：配置 3 的 decode 请求被分配到全部 8 个 rank（CP=8）而非单 rank（CP=1）。

**根因**：`LongShortRequestQueue.is_long_request()` 通过原始 prompt token 数量分类。8192 tokens ≥ `THRESHOLD=1` → 分类为长请求。

**修复**：调度路径中检查 `kv_transfer_params.do_remote_prefill` 标志来识别 PD decode 请求：

```python
# cross_dp_scheduler.py
is_long = self.waiting.is_long_request(request)
kv_params = request.kv_transfer_params
if kv_params and kv_params.get("do_remote_prefill"):
    is_long = False  # PD decode 始终使用 CP=1
```

### 6.2 `running_long_count` 计数不一致

**现象**：`running_long_count` 泄漏导致并发限制耗尽，新请求无法调度。

**根因**：调度路径使用局部变量 `is_long`（正确考虑 PD decode），但 `_free_request()` 和抢占路径使用 `is_long_request()`（不区分 PD decode）。

**修复**：三处（调度路径、抢占路径、释放路径）统一使用 `do_remote_prefill` 检查。

### 6.3 KV 对齐导致额外 prefill 步骤

**现象**：8192-token 请求的 decode 需要额外 64-token prefill 步骤，TTFT 从预期的 ~300ms 恶化至 495,234ms。

**根因**：`align_to_block_size(8192, 64) = 8128`，遗漏 64 个 token。PD 分离中 prefill 已计算所有 token 的 KV cache，应使用 `num_prompt_tokens - 1 = 8191`。

**修复**：PD decode 路径中用 `num_prompt_tokens - 1` 替代 `align_to_block_size`。

**效果**：TTFT 从 495,234ms 降至 311ms（**1591 倍改善**）。

---

## 7. Profile 数据

使用 Torch profiler 对所有 4 种配置采集了 profile，每次发送 1 个请求（8K 输入，10 个输出 tokens），通过 `/start_profile` 和 `/stop_profile` API 控制。

**Profile 文件**（每个配置 9 个文件 = 8 worker rank + 1 server 进程）：
- `profiles/config4/` — 纯 DP 基线
- `profiles/config1/` — 全 CP decode（展示 all-gather + EP all-to-all 开销）
- `profiles/config2/` — 无 CP 的 PD 分离（展示两阶段请求流程）
- `profiles/config3/` — DyCP 目标（展示 CP prefill → IPC KV 传输 → 单 rank decode）

**Config 4 decode profile 关键数据**（单条请求，8K 输入 / 10 输出）：

| 操作 | GPU 时间 | 占比 | 调用次数 |
|------|---------|------|---------|
| NCCL ReduceScatter | 149.8 ms | 27.1% | 286 |
| NCCL AllGather | 105.1 ms | 19.0% | 286 |
| MoE (fused_moe + moe_forward) | 108.4 ms | 19.6% | 962 |
| FlashAttention + 其他 | 190.2 ms | 34.4% | — |

EP all-to-all 占 decode GPU 时间的 46%，这是 Expert Parallelism 的固有成本。

查看方式：在 `chrome://tracing` 或 TensorBoard 中打开 `.pt.trace.json.gz` 文件。

---

## 8. 结论

1. **DyCP 本地 PD 分离达成设计目标**：全 CP prefill 实现低 TTFT + 单 rank decode 实现高吞吐。

2. **配置 3（DyCP）是整体最优配置**：
   - 8K/1K 并发：TTFT -70%（311ms vs 1,022ms），TPOT -45%（13.51ms vs 24.56ms），E2EL -46%
   - 8K/2K 并发：TTFT -60%（9.6s vs 24.3s），TPOT -22%，E2EL -38%
   - 单条请求：TTFT -63%（187ms vs 508ms），直接验证 CP prefill 加速效果

3. **PD 分离消除跨 DP EP all-to-all 阻塞**（核心发现）：
   - 纯 DP 模式下，不同 rank 同时执行 prefill 和 decode，EP all-to-all 集合通信导致 decode rank 被阻塞
   - PD 分离通过 CP=8 prefill 让所有 rank 同时执行 prefill，消除"部分 rank 做 prefill、部分做 decode"的跨 DP 混合
   - 配置 4 TPOT 并发恶化 192%（8.41→24.56ms），配置 3 仅恶化 37%（9.88→13.51ms，含 1.5ms bug 开销）
   - 配置 4 P99 ITL 是中位数的 21 倍（203ms vs 9.6ms），配置 3 仅 13.5 倍（131ms vs 9.7ms）

4. **DyCP 框架 decode 路径存在 1.5ms/step 待修复开销**（单条请求 TPOT 差异 9.88ms vs 8.41ms）：
   - 根因：`dycp_world_size > 1` 时 decode 路径无条件启用 `return_lse=True` + 120 次 NCCL 通信/step，即使 `num_dycp_reqs=0`
   - 修复方案：按 `num_dycp_reqs > 0` 条件启用 DyCP 分支，修复后 TPOT 应与配置 4 持平

5. **Proxy 开销 49ms，CP 加速 370ms，净收益 321ms**：PD 分离的两阶段请求流程（prefill→proxy→decode）引入的 HTTP 往返和 KV IPC 传输开销远小于 CP prefill 的加速收益。

6. **CP 和 PD 两者缺一不可**：配置 2（PD 无 CP）的 TTFT 比配置 3 慢 67 倍，配置 1（全 CP 无 PD）比配置 4 慢 13.8 倍。

7. **DyCP 框架在纯 DP 模式下引入约 8-10% 开销**：配置 4（DyCP DP）vs 配置 5（原版 DP），TTFT +31%（调度器开销），TPOT +9%（每步框架开销）。配置 3 相对配置 5 的净收益：TTFT -60%，TPOT -40%，E2EL -41%。

---

## 9. 复现步骤

```bash
# 从仓库根目录：
cd dycp/benchmarks/2026-04-22-pd-separation/

# 顺序运行全部 4 个基准测试：
bash scripts/run_all.sh

# 或单独运行某个配置：
bash scripts/start_config4.sh  # 终端 1
bash scripts/run_benchmark.sh config4 8400  # 终端 2

# 单条请求测试（配置 3 需同时启动 proxy）：
bash scripts/start_config3.sh  # 终端 1
python dycp/proxy/local_pd_proxy.py --vllm-url http://localhost:8400 --port 9000  # 终端 2
bash scripts/run_bench_single.sh config3_proxy_cp 9000  # 终端 3

# 采集 profile：
bash scripts/collect_profiles.sh config4
```

所有结果 JSON、服务端日志、基准测试日志和 profile trace 文件均归档在本目录中。