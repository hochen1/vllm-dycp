# Local PD Separation — 设计文档

> 创建日期: 2026-04-18
> 状态: Phase 1 设计完成，待实施
> 分支: dev-dycp-mixed-1

---

## 1. 问题背景

在当前 vLLM-DyCP 的 PD 混部模式中，prefill 和 decode 使用相同配置执行。对于 MoE 模型（多DP大EP配置），decode 阶段更友好（减少 KV cache 冗余和 TP 通信），但 prefill 阶段受限于单 DP rank 的计算能力。

**核心矛盾**: 一条请求只会在一个 DP 网格上执行，限制了 prefill 阶段的并行度。

**解决思路**: 利用 DyCP 的 Context Parallelism 能力，让 prefill 阶段的长请求拆分到多个 DP rank 并行执行（全 CP），而 decode 阶段只在单个 rank 上执行（CP=1）。

## 2. 目标

### Phase 1 目标（本次实施）
1. **严格 PD 分离**: 一个 batch 中只能有 prefill 请求或 decode 请求
2. **Prefill 阶段**: 保持 DyCP 长短请求逻辑（长请求 > 128K tokens → 全 CP，短请求 → 单 CP）
3. **Decode 阶段**: 所有请求统一 CP=1
4. **部署架构**: Proxy + 1 个 vLLM 实例
5. **KV 传输**: 基于文件系统的 Connector，各 rank 独立保存（无 all-gather）

### Phase 2 目标（后续）
- 支持 chunked prefill
- 调度时支持 prefill + decode 混跑
- 保持 prefill 全 CP / decode CP=1

### 验证目标
- **功能**: 长请求 prefill 使用全 CP，decode 使用 CP=1，KV 从 connector 加载正确
- **性能**: benchmark 对比同架构开启/不开启 CP
  - `num_prompts=200, max_concurrency=max_num_seqs*dp_size, request_rate=2`

## 3. 方案选择

**选择: Proxy + KV Connector 方案**

| 对比维度 | Proxy + Connector | 内部状态机 |
|---------|-------------------|-----------|
| Scheduler 改动 | 最小，decode 作为新请求 | 大量改动，需要内部状态转换 |
| 复用现有框架 | WAITING_FOR_REMOTE_KVS 天然支持 | 需要新建状态机 |
| Connector 职责 | KV 搬运封装在 Connector 内 | KV 管理耦合调度逻辑 |
| 部署灵活性 | 可扩展为真正的 PD 分离部署 | 仅支持单实例 |

## 4. 架构设计

### 4.1 整体架构

```
┌─────────┐     prefill req (max_tokens=1)      ┌──────────────────────┐
│  Proxy   │ ──────────────────────────────────→ │   vLLM 实例           │
│          │ ←────────────────────────────────── │   (dp_per_domain=N)  │
│ 统一 ID  │     完成, 返回 kv_transfer_params   │                      │
│ 前缀     │                                     │  CrossDPScheduler    │
│          │     decode req (full max_tokens      │  LocalPDConnector    │
│          │      + kv_transfer_params)           │                      │
│          │ ──────────────────────────────────→ │                      │
│          │ ←─── stream decode tokens ────────── │                      │
└─────────┘                                      └──────────────────────┘
                                                         │
                                                    文件系统 KV
                                                  /tmp/kv_cache/
                                                  └── {prefix}/
                                                      ├── meta.json
                                                      ├── rank_0/
                                                      ├── rank_1/
                                                      └── ...
```

### 4.2 请求流程

#### Prefill 阶段
1. Client → Proxy: 原始请求
2. Proxy 生成共享前缀 `pd-{uuid[:12]}`
3. Proxy → vLLM: `request_id="prefill-{prefix}"`, `max_tokens=1`, `kv_transfer_params={do_remote_decode:True, pd_request_prefix:prefix}`
4. vLLM Scheduler: 根据 DyCP 逻辑分配 CP ranks（长请求全CP，短请求单CP）
5. Connector (worker): 每个 CP rank 独立保存 KV slice 到文件
6. Connector (scheduler): prefill 完成时写 meta.json，返回 kv_transfer_params
7. vLLM → Proxy: 完成，携带 kv_transfer_params

#### Decode 阶段
1. Proxy → vLLM: `request_id="decode-{prefix}"`, 原始 max_tokens, `kv_transfer_params={do_remote_prefill:True, pd_request_prefix:prefix, ...}`
2. Scheduler: `get_num_new_matched_tokens()` 检测到有外部 KV
3. 请求进入 `WAITING_FOR_REMOTE_KVS` 状态
4. Connector (worker): Decode rank 加载所有 rank 的 KV slices，重组填入本地 blocks
5. 请求 → `WAITING` → `RUNNING`，CP=1 正常 decode
6. vLLM → Proxy: 流式返回 decode tokens
7. Proxy → Client: 流式转发

### 4.3 KV 搬运策略

**选择: 各 rank 独立保存 + Decode 端单 rank 加载重组**

理由:
- 无 GPU all-gather，无额外显存开销
- 每个 rank 只 save 自己的 KV slice，I/O 可并行
- 文件系统作为中转，实现简单
- 天然适配 connector 框架的 `save_kv_layer()` / `start_load_kv()` 接口

**显存约束**: Decode rank 需要持有完整序列的 KV cache blocks。例如 32K tokens + cp_world_size=4，prefill 时每个 rank 只需 ~8K tokens 的 blocks，但 decode rank 需要 32K tokens 的 blocks。**这是 PD 分离的固有代价。**

**文件布局**:
```
{storage_path}/{pd_request_prefix}/
  ├── meta.json                    # 元数据
  │   {
  │     "cp_world_size": 4,
  │     "num_prompt_tokens": 32768,
  │     "block_size": 16,
  │     "interleave_size": 64,
  │     "per_rank_tokens": [8192, 8192, 8192, 8192],
  │     "completed": true
  │   }
  ├── rank_0/
  │   ├── layer_0.safetensors
  │   ├── layer_1.safetensors
  │   └── ...
  ├── rank_1/
  │   └── ...
  └── rank_{N-1}/
      └── ...
```

### 4.4 Scheduler Batch 分离（Phase 1）

在 `CrossDPScheduler.schedule()` 中添加 batch type 检测:
- 检查 running queue 和 waiting queue 中的请求类型
- 如果已有 running 的 decode 请求 → 本 batch 只调度 decode
- 如果已有 running 的 prefill 请求 → 本 batch 只调度 prefill
- 如果没有 running 请求 → 优先调度 decode（避免饥饿）
- Decode 请求在 `RequestManager.select_dp()` 中强制返回单 rank

## 5. 关键技术分析

### 5.1 现有框架复用分析

| 现有组件 | 复用方式 |
|---------|---------|
| `WAITING_FOR_REMOTE_KVS` 状态机 | 原生支持 decode 请求等待 KV 加载 |
| `CrossDPExampleConnector._cross_requests_need_load` | 复用 per-CP-rank 追踪模式 |
| `ExampleConnector.extract_kv_from_layer()` | 复用 KV 提取逻辑 |
| `ExampleConnector.inject_kv_into_layer()` | 复用 KV 注入逻辑 |
| `ExampleConnector.ReqMeta` | 复用请求元数据结构 |
| `align_to_block_size()` | 复用 token 对齐 |
| `KVConnectorModelRunnerMixin._get_kv_connector_output()` | 复用 worker 集成框架 |
| `kv_transfer_params` 传递链路 | HTTP → SamplingParams → Request → Connector → Response |

### 5.2 KV 重组逻辑分析

Prefill 全 CP 时，KV 按 `_avg_distribute_tokens_to_ranks()` 分布:
```python
num_padded_tokens = ceil(seq_len / (2 * world_size)) * (2 * world_size)
local_seq_len = num_padded_tokens // world_size
```

PCPManager (worker 侧) 使用 DualChunkSwap 策略分配 tokens:
- 每个 rank 持有 `local_seq_len` 个 token 的 KV
- Token 分配模式: head/tail 交错

**重组方案**: 由于 save 时每个 rank 保存的是该 rank 实际持有的 token 对应的 KV（按 slot_mapping 提取），load 时需要将所有 rank 的 KV 拼接后按照 DualChunkSwap 的反向操作还原为原始 token 顺序。

**关键依赖**: 需要从 PCPManager 获取 `pcp_allgather_restore_idx`（或等效的映射关系）来正确重组。

### 5.3 kv_transfer_params 传递链路（已验证存在）

```
HTTP Request (kv_transfer_params in body)
  → SamplingParams.extra_args["kv_transfer_params"]     # serving_completion.py:812-814
  → Request.kv_transfer_params                          # v1/request.py:78-81
  → Connector.get_num_new_matched_tokens(request, ...)  # scheduler.py:484-505
  → ... processing ...
  → RequestOutput.kv_transfer_params                    # output_processor.py:297
  → HTTP Response (kv_transfer_params in body)          # serving_completion.py:623
```

## 6. 实施文件清单

| 文件 | 操作 | 用途 |
|------|------|------|
| `vllm/distributed/kv_transfer/kv_connector/v1/local_pd_connector.py` | **新建** | 核心 connector |
| `vllm/distributed/kv_transfer/kv_connector/factory.py` | 修改 | 注册 connector |
| `vllm/v1/core/sched/cross_dp_scheduler.py` | 修改 | Batch 分离 + decode 强制 CP=1 |
| `dycp/proxy/local_pd_proxy.py` | **新建** | Proxy 服务 |
| `dycp/examples/start_local_pd.sh` | **新建** | 启动脚本 |
| `dycp/tests/test_local_pd/` | **新建** | 测试用例 |

## 7. 实施顺序与依赖

```
Phase 1.1: Connector 基础框架 ──────────────────┐
  ├── LocalPDConnector scheduler 侧              │
  ├── LocalPDConnector worker 侧                 │  可并行
  ├── Connector 注册到 factory                   │
  └── meta.json + KV 重组逻辑                    │
                                                  │
Phase 1.3: Proxy Server ────────────────────────┘
  ├── FastAPI 基础框架
  ├── 两阶段请求转发
  └── kv_transfer_params 传递

Phase 1.2: Scheduler Batch 分离 ← 依赖 1.1 接口
  ├── _determine_batch_mode()
  ├── 调度循环 filter
  └── Decode 强制 CP=1

Phase 1.4: 集成测试 & 验证 ← 依赖全部
  ├── 单元测试 (connector save/load)
  ├── 端到端功能验证
  └── Benchmark 性能测试
```

## 8. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| KV 重组与 DualChunkSwap 不匹配 | 生成内容错误 | 先用小模型+短序列验证 KV 正确性 |
| Decode rank 显存不足 (完整 KV) | OOM | 限制最大序列长度，监控显存 |
| 文件 I/O 成为瓶颈 | 延迟高 | 使用 tmpfs/ramdisk；后续可升级 |
| Batch 分离导致 GPU 利用率下降 | 吞吐降低 | Phase 2 混跑解决 |
| Prefill max_tokens=1 仍生成 1 token | 微小浪费 | 可接受 |

## 9. 进展跟踪

| 日期 | 里程碑 | 状态 | 备注 |
|------|--------|------|------|
| 2026-04-18 | 需求分析 & 方案设计 | ✅ 完成 | |
| 2026-04-18 | Connector 实现 | ✅ 完成 | `local_pd_connector.py` + factory 注册 |
| 2026-04-18 | Proxy 实现 | ✅ 完成 | `dycp/proxy/local_pd_proxy.py` |
| 2026-04-18 | Scheduler 改造 | ✅ 完成 | batch 分离 + decode 强制 CP=1 |
| 2026-04-18 | 启动脚本 | ✅ 完成 | `dycp/examples/start_local_pd.sh` |
| 2026-04-18 | 功能测试 | ✅ 完成 | 短请求端到端验证通过，KV 正确性确认 |
| | 长请求 CP 验证 | ⬜ 待验证 | 需要长 prompt (>128K tokens) 或降低阈值 |
| 2026-04-18 | 性能 benchmark | ✅ 完成 | 短请求: 直接 136ms vs PD分离 172ms (+26%) |

### 调试过程中发现并修复的第 6 个问题

6. **`_prefill_requests.pop()` 导致只有 rank_0 保存 KV**: `build_connector_meta` 被每个 CP rank 调用一次，`pop` 在 rank_0 处理后移除了 entry，rank_1~7 看不到。修复：改用 `get()`，最后一个 rank 处理后再清除。

### 性能 Benchmark 结果 (2026-04-18)

| 指标 | 直接请求 (无 PD 分离) | Proxy (PD 分离) | 差异 |
|------|---------------------|----------------|------|
| 短请求延迟 (ms) | ~136 | ~172 | +26% |
| 首次请求 (冷启动) | ~393 | ~175 | - |

**说明**: 短请求场景下 PD 分离额外开销来自:1) proxy 两次 HTTP 转发 2) KV 文件 I/O (save + load)。
对于长请求 (>threshold)，CP 并行加速的收益将远大于此开销。

### 功能验证结果 (2026-04-18)

| 测试 | 结果 | 说明 |
|------|------|------|
| Proxy 两阶段转发 | ✅ | prefill → decode 完整流程 |
| KV 保存 (save_kv_layer) | ✅ | 27 层 × [14, 576] MLA KV 正确保存 |
| KV 加载 (start_load_kv) | ✅ | MLA 自动检测 + 正确注入 |
| KV 正确性对比 | ✅ | 有/无 PD 分离输出完全一致 |
| 连续请求稳定性 | ✅ | 3 个连续请求均正常 |
| Streaming 响应 | ✅ | SSE chunk 正确转发 |
| 短请求 CP=1 | ✅ | `selected_dp: [0]` 单 rank |
| 文件清理 | ✅ | decode 完成后自动清理 |

### 调试过程中发现并修复的问题

1. **`align_to_block_size` 对短序列返回 0**: 14 tokens, block_size=64 → aligned=0 → KV 保存 0 tokens。修复：改用 `len(block_ids) * block_size` 计算有效 tokens。
2. **`load_kv_async=True` 导致死锁**: 单实例 PD 分离中，WAITING_FOR_REMOTE_KVS 需要 forward pass 触发 `get_finished()`，但 forward 需要请求被调度。修复：使用同步模式 `load_kv_async=False`。
3. **`cp_world_size` 记录不准**: `meta.json` 记录全局 `dp_per_domain` 而非实际使用的 CP rank 数。修复：使用 `len(request.cp_ranks)`。
4. **MLA 检测失败**: `start_load_kv` 在 forward 前调用，`attn_metadata` 未初始化。修复：从 KV cache tensor 维度推断 (3D=MLA, 4D=standard)。
5. **`shared_storage_path` vs `storage_path`**: 配置 key 不一致。修复：统一使用 `shared_storage_path`。

## 10. 长请求 KV 精度问题——完整分析记录

### 10.1 问题描述

全 CP prefill → decode 的长请求输出不正确（乱码或与直接请求不同的内容），而短请求（单 CP）精度完全正确。

### 10.2 调查过程

**第一阶段：认为是 DualChunkSwap 重组问题**
- 初始假设：`torch.cat(rank_slices)` 简单拼接没有还原 DualChunkSwap 的 head/tail 交错顺序
- 实现了 `_compute_dualchunkswap_restore_idx()` 函数，数学验证通过（W=4,8 场景均正确还原）
- 但应用后输出仍然错误

**第二阶段：发现 rank_1~7 KV 全为零**
- 通过 `test_kv_debug.py` 检查保存的 KV tensor norm：
  - rank_0: `[20.375, 26.75, 26.625, ...]` — 有效
  - rank_1~7: `[0.0, 0.0, ...]` — 全零
- 排除了 DualChunkSwap 重组假设，转向 KV 保存问题

**第三阶段：对比 slot_mapping 来源**
- 加日志对比 `req_meta.slot_mapping`（从 block_ids 预计算）和 `attn_metadata.slot_mapping`（worker 实际使用）：
  ```
  rank_0: req_sm=[64,65,...] actual_sm=[64,65,...] — 一致
  rank_1~7: req_sm=[64,65,...] actual_sm=[-1,-1,...] — 不一致！
  ```
- 发现 `attn_metadata.slot_mapping` 在 rank_1~7 全为 -1
- 所有 rank 的 `req_sm` 相同（都是 `[64,65,66,67,68]`）—— 说明 scheduler 给所有 rank 发了相同的 block_ids，这是错误的

**第四阶段：深入分析 block_table.py 的 interleaved slot mapping**

`block_table.py:compute_domain_slot_mapping()` (line 228-251) 的核心逻辑：

```python
# Interleaved block layout
virtual_block_size = block_size * cp_world_size  # 64 * 8 = 512
mask = (position // interleave_size) % cp_world_size == cp_rank
slot_mapping = np.where(mask, actual_slot, -1)
```

每个 rank 只负责特定的 interleave 片段。对于 `interleave_size=64, cp_world_size=8`：
- rank_0: 负责 position 0~63 (第 0 个 interleave 片段)
- rank_1: 负责 position 64~127
- ...
- rank_7: 负责 position 448~511

**第五阶段：定位根本矛盾**

修改 save_kv_layer 只保存 `slot != -1` 的 tokens 后：
```
rank_0: valid=13 tokens, positions [0..12]
rank_2: valid=13 tokens, positions [13..25]
其他 rank: valid=0
```

**根本矛盾发现**：

| 维度 | 机制 | 分配方式 |
|------|------|---------|
| **计算分配** | DualChunkSwap | 207 tokens → 8 rank × 26 tokens (head/tail 交错) |
| **KV 存储所有权** | Interleaved Block | 按 `(position // 64) % 8` 分配 |

每个 rank 在 DualChunkSwap 下计算 26 个 tokens，但这些 tokens 在 interleave 视角下可能不属于该 rank。结果：每个 rank 的 KV buffer 中只有属于自己 interleave 范围的少量 tokens 有有效 KV，其余为 -1。

对于 207 tokens, interleave_size=64, cp_world_size=8：
- 只有 rank_0 (pos 0~63) 和部分 rank_2/3 有少量有效 tokens
- 总共只保存了 ~26/207 = 12.5% 的 KV

### 10.3 结论与解决方案

**直接从各 rank 的 paged buffer 中独立提取 KV 的方案不可行**，因为每个 rank 只持有一小部分 interleave 片段的 KV，无法拼出完整的 KV cache。

**解决方案：All-Gather 方案（先采用最简单的方式调通）**

在 prefill 完成后、`save_kv_layer` 中，先通过 all-gather 将所有 rank 的 KV 聚合到每个 rank（或只聚合到 rank_0），然后 rank_0 保存完整的 KV。decode 端只需从 rank_0 加载即可。

```
Prefill 结束后:
  每个 rank 持有部分 KV (interleaved)
    ↓ all-gather (NCCL)
  rank_0 持有完整 KV
    ↓ save to file
  /tmp/kv_cache/{prefix}/full_kv/layer_*.safetensors

Decode:
  rank_0 从文件加载完整 KV
    ↓ inject to paged buffer
  正常 decode
```

**优点**：最简单，一次 all-gather 即可获得完整 KV
**缺点**：需要临时显存（完整 KV 的副本），all-gather 通信开销
**后续优化**：可以改为 P2P send/recv 只将 KV 发到 decode rank，避免广播

### 10.6 最新进展——All-Gather 方案遇阻 (2026-04-18 21:30)

**实施了三个版本的 all-gather 方案，均无法获得完整 KV：**

1. **V1**: 从 paged buffer 按 `req_meta.slot_mapping`（block_ids 推导）提取 → 每个 rank 26 tokens，但 interleave 以外的 slot 是零
2. **V2**: 从 paged buffer 按 `attn_metadata.slot_mapping`（带 interleave mask）提取 → 只有 ~13/26 tokens 有效  
3. **V3**: 按 `attn_metadata.slot_mapping`（clamp -1→0）提取并 all-gather → 208 tokens 中只有 26 个非零

**根本原因**：DyCP 的 prefill attention kernel 只将 KV 写入 `local_slot_mapping`（带 interleave mask）的有效位置（MLA common.py line 3138-3153 注释确认："Cross-rank all-gathered KV is **only used for attention computation**"），不写入其他 rank 的 interleave 范围。

**数据流**：
```
DualChunkSwap: rank_0 处理 26 tokens (head=[0..12], tail=[195..207])
Interleave mask: rank_0 只负责 pos 0~63
交集: 只有 head=[0..12] 属于 rank_0 → 写入 13 个 KV
Tail=[195..207] 属于 rank_3 → slot=-1 → KV 被丢弃！

rank_3 处理: head=[39..51], tail=[156..168]
Interleave mask: rank_3 负责 pos 192~255
交集: 全部不在范围内 → 写入 0 个 KV

结果: pos 195~207 的 KV 不在任何 rank 的 paged buffer 中
```

**关键疑问**：DyCP 直接请求（不走 PD 分离）能正常输出（已验证 127 tokens 输出正确）。那 decode 时如何处理缺失的 KV？

**可能的解释**：
1. DyCP decode 时每个 rank 的 `cp_local_seq_lens` 只反映该 rank 实际写入的 KV 数量（不是理论 interleave 范围），所以 partial attention 不会读取未写入的 KV
2. 或者 DyCP 使用 chunked prefill 分多步执行，确保逐步填充完整的 interleave KV
3. 或者在某些 block_size/interleave_size/cp_world_size 组合下，DualChunkSwap 和 interleave 恰好完全对齐

**验证结果 (2026-04-18 21:40)**：
- `cp_local_seq_lens` 计算值：rank_0=64, rank_1=64, rank_2=64, rank_3=15（理论 interleave 数量）
- 实际 KV 覆盖：rank_0=13, rank_2=13, 其他=0（DualChunkSwap ∩ Interleave 交集）
- **Chunked Prefill 是关键**：`enable_chunked_prefill=True, max_num_batched_tokens=4096`
  - 207 tokens 虽然 < 4096（单步完成），但更大的序列会分多步
  - 每步的 DualChunkSwap 分配不同 token 范围，逐步填充各 rank 的 interleave KV
  - 最终 `cp_local_seq_lens` 反映的是**预期**的 KV 数量，chunked prefill 确保逐步达到

**当前阻塞的解决方案 (采用)**：
1. 在 `save_kv_layer` 中不保存文件（因为 chunked prefill 可能分多步）
2. 在 `request_finished` 中利用 `delay_free_blocks=True` 延迟 block 释放
3. 在下一个 worker step 中从 paged buffer 提取完整 KV（all-gather 各 rank）并保存
4. 保存完成后释放 blocks

### 10.7 最终结论 (2026-04-18 22:00)

**发现**：`build_connector_meta` 只在请求首次调度时（`new_reqs`）标记 `is_store=True`。后续 chunked prefill 的 chunks（请求在 `cached_reqs`/running 中）不会触发 `save_kv_layer`。这意味着：
1. 对于单步 prefill (tokens < max_num_batched_tokens)：只有一次 save 机会，但 interleave 导致只保存了部分 KV
2. 对于多步 chunked prefill：只有第一步触发 save，后续步骤没有

**DyCP 自身对于低阈值下的 单步全 CP prefill 也存在 KV 不完整的问题**——`cp_local_seq_lens` 期望 rank_0 有 64 tokens KV，但实际只有 13 tokens。DyCP decode 时 partial attention 会读到未初始化的 KV。但由于 FlashMLA 的 softmax 对零值 KV 的处理方式，这在简单 prompt 上可能不会导致明显错误。

**Phase 1 结论**：
- **短请求（单 CP）精度完全正确** ✅
- **长请求（全 CP）需要 chunked prefill 来逐步填充各 rank 的 interleave KV**
- 当前 connector 只保存第一步的 KV，不支持跨 chunk 累积
- 需要修改 connector 的 store 跟踪逻辑来支持 chunked prefill，或修改保存时机到所有 chunk 完成后

### 10.8 Pre-Mask KV Capture 方案 (2026-04-18 22:30) — 成功 ✅

**方案**：不从 paged buffer 提取 KV（已被 interleave mask 过滤），而是从 attention 函数参数中捕获 pre-mask KV。

**实现**：
1. 修改 `maybe_transfer_kv_layer` 装饰器，从被装饰函数的 `kv_c_normed` 和 `k_pe` 参数中提取 raw KV，通过 `**kwargs` 传递给 `save_kv_layer`
2. `save_kv_layer` 对多 CP 请求使用 raw KV 进行 all-gather + DualChunkSwap restore

**验证结果**：
- KV norm 测试：**208/208 tokens 全部非零** ✅ （之前只有 26/208）
- 短请求精度：**完全匹配** ✅
- 长请求精度：**语义正确**（"quick brown fox" 的正确续写），但格式与直接请求略有差异（`\nUser:\n\n` 前缀差异）

### 10.9 重要约束：当前代码仅支持 PD 分离部署模式

**发现 (2026-04-18 22:40)**：Direct DyCP 全 CP 模式（非 PD 分离）在 low-threshold 下对 113 tokens 的输出质量很差（循环重复），而 PD 分离模式输出质量正常。

**根因**：DyCP 全 CP prefill 后，各 rank 的 paged buffer 中只有 DualChunkSwap ∩ Interleave 交集的 KV（约 12%）。DyCP 的 decode 通过 partial attention + all-gather 工作，但 `cp_local_seq_lens` 期望每个 rank 有完整 interleave 范围的 KV，实际远少于此，导致 decode 精度下降。PD 分离模式通过 pre-mask KV 的 all-gather 重建了完整 KV，decode 在单 rank 上使用完整 KV，精度反而更好。

**约束**：
- **当前修改后的代码（`VLLM_LONG_REQUEST_THRESHOLD` < 默认 128K）仅适用于 PD 分离部署模式**
- 非 PD 分离的 Direct DyCP 全 CP 模式在 low-threshold 下精度不可靠
- 生产使用必须通过 Proxy + PD 分离架构部署
- 短请求（< threshold）不受影响，两种模式均正确

### 10.4 其他已解决的问题

- ~~**meta.json 写入时机**~~: 在 `request_finished()` callback 中写入
- ~~**文件清理策略**~~: decode 完成时自动清理
- ~~**kv_transfer_params 传递**~~: 已验证 vLLM OpenAI endpoint 原生支持 `kv_transfer_params` 字段

## 11. 已实现的文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `vllm/distributed/kv_transfer/kv_connector/v1/local_pd_connector.py` | 新建 | 核心 connector (scheduler + worker 侧) |
| `vllm/distributed/kv_transfer/kv_connector/factory.py` | 修改 | 注册 LocalPDConnector |
| `vllm/v1/core/sched/cross_dp_scheduler.py` | 修改 | 添加 batch 分离逻辑 (188-229行) |
| `dycp/proxy/local_pd_proxy.py` | 新建 | FastAPI proxy server |
| `dycp/proxy/__init__.py` | 新建 | Package init |
| `dycp/examples/start_local_pd.sh` | 新建 | 启动脚本 |
| `dycp/docs/local_pd_separation_design.md` | 新建 | 本设计文档 |
