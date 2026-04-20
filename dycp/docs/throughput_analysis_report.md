# CP 高并发吞吐劣化分析报告

> 日期: 2026-04-20
> 配置: 8 × NVIDIA L20Y, DeepSeek-V2-Lite, dp_per_domain=8, enable_expert_parallel
> 场景: 8K input / 200 output tokens, c=32, rate=2

---

## 1. 问题描述

在 8K 输入场景下，PD 分离 + 动态 CP 相比 PD 分离无 CP：
- **TTFT 大幅改善**: 196ms vs 565ms (**-65.3%**) ✅
- **但高并发吞吐下降**: c=32 下 12158 vs 13958 tok/s (**-12.9%**) ❌

本报告通过实测数据定位吞吐劣化的根因。

## 2. 实测数据

### 2.1 基础性能参数

| 参数 | CP 模式 (threshold=100) | No-CP 模式 (threshold=128K) |
|------|------------------------|---------------------------|
| 单请求 8K prefill | **113ms** (8 rank CP) | **472ms** (1 rank) |
| Prefill 加速比 | 4.18x | 1x (baseline) |
| Decode per-token | 10.19ms | 9.91ms |
| Pure short decode 吞吐 | 906 tok/s | 772 tok/s |

### 2.2 MoE All-to-All 干扰实测

测试方法: 8 个并发请求，混合 1 个 8K 长请求 + 7 个短请求。

| 指标 | CP 模式 | No-CP 模式 |
|------|---------|-----------|
| 纯短请求 decode 吞吐 | 771 tok/s | 772 tok/s |
| 混合后 短请求 decode 吞吐 | 560 tok/s | 574 tok/s |
| **干扰导致 decode 下降** | **-27%** | **-26%** |

**结论**: 两种模式下 MoE all-to-all 同步干扰程度相当（~27%）。CP 模式并未显著减少干扰（因为 direct 请求不走 batch 分离）。

### 2.3 高并发吞吐对比（Direct 模式，无 Proxy）

| 并发 | CP tok/s | No-CP tok/s | Delta |
|------|---------|-------------|-------|
| c=10 | 11569 | 23750 | **-51.3%** |
| c=32 | 12158 | 13958 | **-12.9%** |

注：c=10 时差异更大（-51.3%），因为 10 个 CP prefill 完全串行化（max_cp_seqs=4），
而 no-CP 的 10 个 prefill 可以在 8 rank 上并行。c=32 时差异缩小，因为 decode 阶段
占比增大（200 output tokens），decode 阶段两种模式效率相当。

## 3. 根因分析

### 3.1 排除 Batch 分离因素

Batch 分离逻辑（`_determine_batch_mode`）只对 PD 请求（带 `kv_transfer_params`）生效。Direct 请求（无 `kv_transfer_params`）不受 batch 分离影响。**因此 direct 模式的 12.9% 差异与 batch 分离无关。**

代码证据: `cross_dp_scheduler.py` line 781:
```python
is_pd_decode = self._is_pd_decode_request(request)  
# → checks kv_transfer_params.get("do_remote_prefill")
# Direct requests: kv_transfer_params is None → always False
```

### 3.2 排除 idle ranks 因素

Decode 请求通过 `RequestManager.select_dp()` 分配到不同 rank，c=32 时所有 8 rank 都有 decode 请求。**不存在 idle rank。**

日志证据:
```
short req, selected_dp: [0], request id: decode-pd-xxx
short req, selected_dp: [1], request id: decode-pd-yyy
short req, selected_dp: [2], request id: decode-pd-zzz
```

### 3.3 根因: CP Prefill 的资源独占效应

**CP prefill 是批量处理的**（每 batch 最多 `num_cp_seqs=4` 个 CP 请求），不是串行。
但关键问题是 **CP batch 期间全部 8 rank 被锁定，decode 完全停滞**。

#### CP batch 处理模式

每个 8K CP 请求的 per-rank tokens = 952（DualChunkSwap 后）。
max_batch_tokens = 4096/rank → **一个 batch 最多 4 个 CP prefill**（4×952=3808 < 4096）。

```
CP 模式（32 请求）:
  Batch 1: 4 CP prefill on 8 ranks (~452ms) → 0 decode progress
  Batch 2: 4 CP prefill on 8 ranks (~452ms) → 0 decode progress
  ... (8 batches)
  Batch 9+: decode on 8 ranks → prefill 完全停滞
```

```
No-CP 模式（32 请求）:
  Batch 1: 8 prefill on 8 ranks (~472ms) + 可混合 decode
  ... (4 batches)
  During each batch: 空闲 rank 做 decode → prefill 和 decode 并行
```

#### 量化对比

| | CP | No-CP |
|--|------|--------|
| 每 batch prefill 数 | 4 | 8 |
| 32 请求需要 batch 数 | 8 | 4 |
| 总 prefill wall time | **3.6s** | **1.9s** |
| Prefill 期间可做 decode | **否** | **是** |

#### GPU-Seconds 效率

```
No-CP: 每请求 prefill = 472ms × 1 rank = 472 GPU-ms
CP:    每请求 prefill = 452ms/4 × 8 rank = 904 GPU-ms  ← 1.92x
```

CP 虽然 4 个请求 batch 处理（per-request time = 452/4 = 113ms），但 8 rank 全部被占用：
- 单请求 latency: 113ms (4.18x 加速) ✅
- 单请求 GPU cost: 904 GPU-ms (1.92x 消耗) ❌

**并行效率 = 4.18x 加速 / 8x 资源 = 52.2%**

### 3.4 32 请求的总 GPU-Seconds 对比

| 组件 | No-CP | CP | 比值 |
|------|-------|-----|------|
| Prefill GPU-seconds | 15.1 | 28.9 | **1.92x** |
| Decode GPU-seconds | 64.0 | 64.0 | 1.0x |
| **总 GPU-seconds** | **79.1** | **92.9** | **1.17x** |
| 理论 wall time (8 GPU) | 9.9s | 11.6s | **+17.5%** |

**理论吞吐损失 17.5%，实测 12.9%**。实测更好的原因: no-CP 模式下 MoE all-to-all 干扰导致 decode 实际吞吐低于理论值。

## 4. 为什么 CP 延迟大幅改善但吞吐下降

| 维度 | CP 效果 | 原因 |
|------|---------|------|
| **单请求延迟** | **-50.6%** ✅ | Prefill 加速 4.18x, 隔离干扰 |
| **高并发吞吐** | **-12.9%** ❌ | Prefill GPU-seconds 增加 1.92x |

**本质**: CP 是用 **GPU 资源** 换 **延迟**。8 个 rank 并行处理 1 个 prefill，每个 rank 只处理 1/8 的 tokens，总 GPU 资源消耗反而更多（因为并行效率 < 100%）。

这类似于 Amdahl's law 的变体：**并行加速有效但并行效率 < 1，导致总资源消耗增加**。

## 5. 优化方向

要同时实现 **延迟改善 + 吞吐不降**，需要提升 CP prefill 的并行效率：

| 优化 | 预期效果 | 复杂度 |
|------|---------|--------|
| 减少 CP rank 数（如 4 rank 而非 8） | 效率从 52% 提升到 ~70% | 低 |
| 优化 all-gather 为 P2P 或 reduce-scatter | 减少通信开销 ~5% | 中 |
| 减少 DualChunkSwap padding | 减少无效计算 ~3% | 中 |
| 提升 CP all-gather overlap (与 decode 重叠) | 隐藏通信延迟 | 高 |

**最简方案**: 降低 `dp_per_domain` 从 8 到 4，CP prefill 使用 4 rank（而非 8），并行效率从 52% 提升到 ~70%。延迟从 113ms 增加到 ~200ms（仍比 472ms 快 2.4x），但 GPU-seconds 从 904 降到 ~800 GPU-ms。

## 6. 结论

CP 在 8K 场景下是 **延迟优化方案**，不是 **吞吐优化方案**：
- ✅ TTFT 改善 65%（延迟敏感场景的核心价值）
- ❌ 高并发吞吐下降 13%（GPU 资源效率 52%）
- 根因是 CP 的并行效率 < 1（4.18x 加速 / 8x 资源 = 52%）
- 优化方向：提升并行效率（减少 rank 数、优化通信）
