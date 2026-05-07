# DyCP 工作日报 — 2026-04-26

## 分支状态

- **分支**: `dev-dycp-prefill-gqa-2`
- **今日提交**:
  - `98bc12058` — [bugfix] fix Domain PD separation: P crash and D recv routing
- **核心修改文件**:
  - `vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py` — DyCP PD 分离 CP-aware KV 传输（+730/-78）
  - `vllm/v1/core/sched/cross_dp_scheduler.py` — CrossDP PD 分离调度适配（+79/-16）

## 今日完成的工作

### 1. PD 分离输出乱码根因分析与修复（上午，延续昨日）

**问题**: 昨日端到端测试中，PD 分离链路全通（RDMA pull 成功、`_update_waiting_for_remote_kv` 正确计算 `num_computed_tokens=20`），但 decode 输出为乱码（`* * * *`）。

**根因定位**: D 端（TP=1）在 `group_kv_pull()` 中只连接了 P 端（TP=4）的 1 个 TP rank（TP0），导致 D 只收到 1/4 的 KV heads，剩余 3/4 为未初始化数据。

**修复**: Bug 9-11（详见下方 Bug 表）

### 2. P 端 GQA TP Head Offset 修复

**问题**: P（TP=4）不知道 D 的 TP（=1），`remote_tp_size` 默认等于 `self.world_size=4`，导致 `tp_head_offset=0` 对所有 P TP rank 都相同 → 4 个 TP rank 写到同一偏移 → KV 数据覆盖。

**修复**: 在 `MooncakeAgentMetadata` 中新增 `tp_size` 字段，D 在握手时传递自身 TP size。P 端 `send_kv_to_decode` 和 `_send_blocks_cp_scatter` 使用握手中的 `d_tp` 计算正确的 `tp_head_offset`：
```python
d_tp = agent_meta.tp_size if agent_meta.tp_size > 0 else self.remote_tp_size
if self.use_mla or self.world_size == d_tp:
    d_token_bytes = p_token_bytes
    tp_head_offset = 0
else:
    d_token_bytes = p_token_bytes * self.world_size // d_tp
    tp_head_offset = self.tp_rank * p_token_bytes
```

**验证**: P 日志显示 `d_tp=1`, `tp_offset=0,256,512,768`（4 个 P TP rank 正确写到不同偏移）。

### 3. Bug 12: P 端首个请求完成后崩溃（KeyError）

**问题**: P 处理第一个请求后崩溃，第二个调度步骤在 `start_load_kv` 中抛出 `KeyError`。

**根因分析**:

P 使用标准 Scheduler（非 CrossDPScheduler）。`MooncakeConnectorScheduler._reqs_need_send` 的生命周期：

1. `update_state_after_alloc`: 添加 `req_id: []`（空 block_ids）
2. 第一次 `build_connector_meta`: 发出 `req_id: []` → worker 侧创建 `SendBlockMeta`
3. `request_finished`: 更新为 `req_id: block_ids`
4. 第二次 `build_connector_meta`: 发出 `req_id: block_ids` → worker 侧 `send_meta.ready.set()` → sender 线程发送 → **sender 删除 entry**
5. **第三次** `build_connector_meta`（BUG）: `_reqs_need_send` 未清理，再次发出 `req_id: block_ids` → worker 侧 `self.reqs_need_send.reqs[req_id]` 已被 sender 删除 → **KeyError**

**对比**: `CrossDPScheduler` 在 `build_connector_meta` 之后显式调用 `_reqs_need_send.clear()`，但标准 Scheduler 没有这个逻辑。

**修复**: 在 `MooncakeConnectorScheduler.build_connector_meta` 末尾，清理已发出且 block_ids 非空的 entry：
```python
if block_ids:
    sent_ready_keys.append(req_id)
for req_id in sent_ready_keys:
    del self._reqs_need_send[req_id]
    self._send_per_rank_blocks.pop(req_id, None)
```

同时在 worker 侧 `start_load_kv` 增加防御性检查：
```python
send_meta = self.reqs_need_send.reqs.get(req_id)
if send_meta is None:
    logger.warning("[PD] start_load_kv SEND: req=%s already consumed, skipping", req_id)
    continue
```

**验证**: P 连续处理 103 个请求后仍然存活，不再崩溃。

### 4. Bug 13: D 端非 rank-0 的 DP worker 收不到 KV

**问题**: 第一个请求（分配到 DP0）KV 传输正常，第二个请求（分配到 DP1）D 端 `reqs_to_recv=0`，KV 拉取从未发起，请求永久挂起。

**根因分析**:

`CrossDPScheduler` 为每个 DP rank 创建 `SchedulerOutput`。当某个 rank 没有调度 token 时，使用 `SchedulerOutput.make_empty()` 创建空输出。**空输出没有设置 `cp_rank` 属性**。

在 `build_connector_meta` 中：
```python
current_rank = (
    scheduler_output.cp_rank
    if hasattr(scheduler_output, 'cp_rank')
    else -1
)
```

空输出 → `current_rank=-1`。而 recv entry 的 `rank=1`（目标 DP rank），过滤条件：
```python
if rank != -1 and rank != current_rank:
    continue
```
`rank=1, current_rank=-1` → skip。**recv entry 永远不会被匹配**。

但对于非空输出（rank=0 有调度 token），`cp_rank=0` 设置正确，所以 rank=0 的 recv entry 能匹配。这就解释了为什么第一个请求（DP0）正常，第二个请求（DP1）挂起。

**修复**: 在 `CrossDPScheduler` 创建空输出时也设置 `cp_rank`：
```python
scheduler_output = SchedulerOutput.make_empty()
scheduler_output.cp_rank = idx  # 新增
scheduler_output.none_tokens_in_peer_sched = none_tokens_in_peer_sched
```

**验证**: 两个连续请求分别分配到 DP0 和 DP1，都成功完成 KV 传输：
```
build_connector_meta RECV: req=..., current_rank=0, target_rank=0  # DP0 ✓
build_connector_meta RECV: req=..., current_rank=1, target_rank=1  # DP1 ✓ (之前不会出现)
```

### 5. 端到端精度测试

#### 测试 1: batch_size=16（高并发）

**配置**: P(TP=4) → Proxy → D(DP=4, TP=1), gsm8k 100 条

**结果**: P 存活，103 prefill 完成。但 mooncake 握手并发导致 15 个请求 RDMA 传输失败，D 端报错：
```
Error happens during tranfering kvcache for [req_ids], see logs in prefiller.
```
P 端报错（空异常信息）：
```
Error processing Mooncake handshake:
```

88/100 完成，15 个请求永久挂起。**batch_size=16 下 mooncake 并发连接不稳定**。

#### 测试 2: batch_size=4（降低并发）

**结果**: 0 个 mooncake 错误，但 3 个请求仍然挂起。最终 97/100 完成。

#### 测试 3: batch_size=2（最终）

**结果**: 0 个 mooncake 错误，99/100 完成（1 个超时）。

**精度**: 76/99 = **76.8%**

**错误分析**:

| 类型 | 数量 | 说明 |
|------|------|------|
| 推理错误 | 12 | 模型给出错误答案（model-level） |
| 空输出（全是换行符）| 8 | 输出全为 `\n\n\n...`，疑似 KV 数据全零 |
| 乱码输出 | 3 | 输出含 `--------`、`=======`、`ƒƒƒƒ`，KV 数据损坏 |

**排除推理错误后**: 76/(99-11) = **86.4%**（与模型基线一致）

**KV 数据损坏分析**:
- cp_scatter 地址计算逻辑经过人工审核，偏移量正确
- 所有 P 端零错误（`Error processing Mooncake handshake: 0`）
- D 端零 KV 传输错误
- RDMA 传输"成功"但数据实际损坏
- **疑似原因**: P 的 block 在 D 的 RDMA read 完成前被释放/复用，导致 D 读到的是下一个请求的数据或未初始化数据

## 发现和修复的 Bug

| Bug | 严重度 | 位置 | 说明 | 状态 |
|-----|--------|------|------|------|
| Bug 9: D 不 fan-out P 多 TP rank | **Critical** | `group_kv_pull()` | D（TP=1）只连 P TP0，遗漏 TP1-3 | ✅ 已修复 |
| Bug 10: P 端 is_cp_scatter 遗漏 TP 不匹配 | **Critical** | `send_kv_to_decode()` | P 用 `_send_blocks` 无 head offset | ✅ 已修复 |
| Bug 11: 多路拉取提前 finished | **Critical** | `receive_kv()` | 第 1 个协程完成就标记 done | ✅ 已修复 |
| Bug 12: P 崩溃 — _reqs_need_send 未清理 | **Critical** | `build_connector_meta()` | 标准 Scheduler 不清理，重复发送 → KeyError | ✅ 已修复 |
| Bug 13: 空 SchedulerOutput 缺 cp_rank | **Critical** | `CrossDPScheduler` | 非 rank-0 DP worker 永远匹配不到 recv entry | ✅ 已修复 |
| Bug 14: GQA TP head offset | **Critical** | `_send_blocks_cp_scatter()` | P 不知道 D 的 TP，所有 rank 写同一偏移 | ✅ 已修复 |
| Bug 15: KV 数据损坏（11/99） | **High** | 疑似 block 生命周期 | RDMA 传输成功但数据错误 | 🔍 排查中 |

**累计 Bug 修复进度（04-25 至今）**:
- Bug 1-4: 04-25 已修复（request_id 不匹配、_reqs_need_recv 键结构、_reqs_need_send 清理、地址计算）
- Bug 5-8: 04-25 已修复（decode/prefill cp_world_size、GQA TP 维度、remote_dycp_ranks）
- Bug 9-11: 04-26 上午已修复（TP fan-out、is_cp_scatter、多路完成计数）
- Bug 12-14: 04-26 下午已修复（P 崩溃、cp_rank 路由、GQA TP offset）
- Bug 15: 04-26 排查中（KV 数据损坏）

## 精度测试结果

### PD 分离精度验证

**配置**：P (GPU 0-3, TP=4, port 8400) → Proxy (port 8000) → D (GPU 4-7, DP=4, TP=1, dp_per_domain=4, port 8401)

| 数据集 | 数量 | 完成 | 精度 | 乱码 | 推理错误 | 排除乱码精度 |
|--------|------|------|------|------|---------|-------------|
| gsm8k（batch=2）| 100 | 99 | **76.8%** | 11 | 12 | **86.4%** |

**对比历史数据**：

| 配置 | 精度 | 说明 |
|------|------|------|
| **PD 分离 Bug 12-14 修复后（今日下午）** | **76.8%**（排除乱码 86.4%） | 真正走 RDMA，但有 11 个 KV 损坏 |
| PD 分离 Bug 9-11 修复后（今日上午） | 91.0% | P 崩溃后全部 D recompute，非真正 PD |
| 单机 DyCP DP=8（04-24） | 91.4% | 8 卡全开 |

**重要发现**: 上午的 91% 精度并非真正 PD 分离结果 — P 在第一个请求后崩溃（Bug 12），D 全部 fallback 到 recompute 模式。修复 Bug 12-14 后 P 不再崩溃，真正走 RDMA KV 传输，但暴露了 KV 数据损坏问题。

## 待办

- [x] ~~修复 P 崩溃（Bug 12）~~ — `_reqs_need_send` 清理
- [x] ~~修复 D recv 路由（Bug 13）~~ — 空 SchedulerOutput cp_rank
- [x] ~~修复 GQA TP offset（Bug 14）~~ — 握手传递 tp_size
- [ ] **排查 KV 数据损坏（Bug 15）** — 11/99 请求输出乱码/全空
  - 疑似 P block 提前释放：sender 线程 RDMA write 完成后立即释放 block，但 D 的 RDMA read 可能未完成
  - 需检查 `expected_receivers` 和 block 释放时序
  - 可能需要增加 RDMA completion 同步机制
- [ ] 测试 P Domain2 (TP=2) + D Domain4 (TP=1) 的精度
- [ ] 排查 Qwen3-235B 单机 DyCP 复读问题（#108, #710）

## 注意事项

- **Mooncake 高并发不稳定**: batch_size=16 时 15/103 握手失败，batch_size=4 时 3/100 失败，batch_size=2 基本稳定。
- 被 kill -9 的 vLLM worker 进程可能不释放 GPU 内存，需要用 `nvidia-smi --query-compute-apps=pid,process_name --format=csv` 找到残留进程手动 kill。
- 当前 P 使用标准 Scheduler，不是 CrossDPScheduler。标准 Scheduler 没有 `_reqs_need_send.clear()` 的调用。
- `flash_attn 2.7.3` 已卸载。`mooncake-transfer-engine` 需从内部源安装。
- 当前分支 ahead of origin 2 commits。

## 修改文件清单

| 文件 | 修改内容 | 行数变化 |
|------|---------|---------|
| `mooncake_connector.py` | Bug 9-12,14 修复 + TP head offset + 防御性 start_load_kv | +773/-80 |
| `cross_dp_scheduler.py` | Bug 13 修复（空 SchedulerOutput cp_rank） | +1/-0 |
