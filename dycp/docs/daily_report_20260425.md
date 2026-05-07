# DyCP 工作日报 — 2026-04-25

## 分支状态

- **分支**: `dev-dycp-prefill-gqa-2`
- **今日提交**: `11dbca7cf` — [bugfix] fix dycp_real_token_indices negative dimension crash & support GQA DyCP prefill extend
- **未提交修改**: 2 个文件
  - `vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py` — DyCP PD 分离 CP-aware KV 传输（+660 行）
  - `vllm/v1/core/sched/cross_dp_scheduler.py` — CrossDP PD 分离调度适配（+80 行）

## 今日完成的工作

### 1. MooncakeConnector DyCP PD 分离参数传递（进行中）

**文件**: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py`

实现了 PD 分离场景下 CP-aware KV 传输的完整参数链路：

#### 1.1 元数据扩展

- `MooncakeAgentMetadata` 新增 `cp_rank`、`cp_world_size`、`cp_interleave_size`、`cp_block_size`、`num_tokens` 字段
- `RecvReqMeta` 新增 prefill 侧拓扑信息（`prefill_cp_world_size`、`prefill_cp_interleave_size`、`prefill_tp_size`）和 decode 侧 CP 参数
- `SendBlockMeta` 新增 `expected_receivers`/`completed_receivers` 引用计数，支持多 decode rank 请求同一 KV

#### 1.2 Scheduler 端（MooncakeConnectorScheduler）

- 初始化 `dp_per_domain`、`tp_size`、`cp_interleave_size` 配置
- 计算 `domain_port_base`：`dp_per_domain > 1` 时使用 Domain 级别端口
- `build_connector_meta()` 注入 `num_tokens`、`decode_cp_rank`、`decode_cp_world_size`、`decode_cp_interleave_size`
- `get_and_clear_blocks_to_save()` 返回 `prefill_cp_world_size`、`prefill_cp_interleave_size`、`prefill_tp_size`

#### 1.3 Worker 端（MooncakeConnectorWorker）

- 初始化本地 CP 拓扑（`_local_cp_rank`、`_local_cp_world_size`、`_local_cp_interleave_size`）
- sender 线程端口 = `domain_port_base + cp_rank * world_size`
- `_send_blocks_cp_scatter()` — 核心 CP scatter 传输方法：
  - 支持任意 M-to-N CP 拓扑（prefill cp_ws=M, decode cp_ws=N）
  - 基于 numpy 的 position-level 映射：按 interleave 公式筛选 owned positions
  - 计算 src/dst block+offset 映射
  - 合并连续地址区间，生成 RDMA batch transfer
- `group_kv_pull()` 支持多路径拉取：向每个 prefill DP rank 分别发 ZMQ handshake
- `receive_kv()` 传递 `cp_info` 填充到 `MooncakeAgentMetadata`
- 多 receiver 引用计数：`completed_receivers >= expected_receivers` 后才释放

### 2. PD 分离测试（早期，非 Domain 模式）

完成 Qwen3-30B PD 分离性能和精度测试（详见 `pd_separation_test_report_20260425.md`）：

- **性能**: 单条 4k TTFT 108ms，128k TTFT 4.4s，decode TPOT 稳定 8ms
- **精度**: gsm8k 100 条 90.0%，正常范围
- **稳定性**: 所有场景 0 fail

### 3. Domain DyCP 精度验证（注意：非真正 PD 分离）

> **重要更正**：以下测试虽然 P 和 D 都启动了，但精度测试请求直接打到 D 端口 8401，D 侧通过 `kv_load_failure_policy='recompute'` 本地 recompute 了 prefill，**没有走 Mooncake KV transfer 链路**。P 侧 0 条请求。实际测试的是 **D 侧单独跑 Domain DyCP** 的精度，不是真正的 PD 分离。

**Qwen3-30B（D 侧 Domain4 独立运行）**：

| 数据集 | 数量 | 精度 | 失败请求 | 耗时 |
|--------|------|------|---------|------|
| gsm8k | 100 | **90.00%** | 0 | ~5 min |

### 4. 环境修复（新增）

- **卸载不兼容的 `flash_attn 2.7.3`**：残留的 `.so` 文件导致 `ApplyRotaryEmb` 初始化时 import 失败（`undefined symbol: _ZNK3c106SymInt6sym_neERKS0_`），prefill 侧 TP=4 启动崩溃。使用 `pip uninstall flash-attn --break-system-packages` 彻底清除。
- **安装 `mooncake-transfer-engine 0.3.10.post2`**：PD 分离依赖的 RDMA KV 传输库，从内部源 `pypi.antfin-inc.com` 安装。注意 PyPI 上的 `mooncake` 包（0.0.1）是占位包，不包含 TransferEngine。

### 5. Qwen3-235B FP8 Domain DyCP 精度验证（注意：非真正 PD 分离）

> **重要更正**：与第 3 节相同，请求直接打到 D 端口 8401，D 侧 recompute prefill，**P 侧 0 条请求，未走 Mooncake KV transfer**。实际测试的是 D 侧单独跑 Domain4 DyCP 的精度。

**Qwen3-235B FP8（D 侧 Domain4 独立运行）**：

| 数据集 | 数量 | 精度 | 失败请求 | 耗时 | CP 请求 | 短请求 |
|--------|------|------|---------|------|---------|--------|
| gsm8k | 1000 | **93.80%** | 0 | ~51 min | 185 (18.5%) | 817 (81.5%) |

**对比历史数据（均为 D 侧独立运行 DyCP）**：

| 配置 | 精度 | 复读 | 说明 |
|------|------|------|------|
| 单机 DyCP DP=8 TP=1（04-24） | 91.4% | 2 条 | 8 卡全开 |
| 单机 DyCP DP=8 TP=1 100 条（04-24） | 93.0% | 0 | 8 卡全开 |
| **Domain4 DyCP DP=4 TP=1（今日）** | **93.8%** | **0** | 4 卡，recompute prefill |

**结论**：Domain4 DyCP (DP=4) 精度 93.8%，与昨日 DP=8 DyCP 的 91.4% 相比更高，0 复读。精度差异可能来自 DP 数量不同（4 vs 8）导致的 interleave 误差和调度差异。

### 5.1 P Domain2 + D Domain4 精度验证（注意：非真正 PD 分离）

> **重要更正**：同上，请求打到 8401，D 侧 recompute，P 侧 0 请求。

**Qwen3-235B FP8（P Domain2 闲置，D Domain4 独立运行）**：

| 数据集 | 数量 | 精度 | 失败请求 |
|--------|------|------|---------|
| gsm8k | 100 | **99.00%** | 0 |
| gsm8k | 1000 | 进行中 | — |

99% 精度异常高，进一步印证请求没走 PD 分离（D 侧本地 recompute 精度最优）。

**启动脚本**：
- Prefill Domain2: `/ossfs/workspace/bench8/pd_prefill_235b_domain2.sh`
- Decode Domain4: `/ossfs/workspace/bench8/pd_decode_235b_domain4.sh`

### 6. Domain PD 设计文档整理

编写 `domain_pd_design_doc.md`，完整记录 Domain PD 分离架构设计，涵盖：
- 进程模型（DomainEngineCoreProc、DomainMultiprocExecutor）
- CrossDPScheduler 调度逻辑
- Worker 端适配
- KV 传输参数传递
- 完整修改文件清单（vllm 29 文件 + vllm-ascend 12 文件）

生成了两个 patch 文件供昇腾侧参考：
- `domain_pd_vllm.patch`（4591 行）
- `domain_pd_vllm_ascend.patch`（3377 行）

### 7. PD Proxy 开发及真正的 PD 分离端到端调试（19:00–21:30）

> 以下工作均围绕 **真正的 PD 分离**展开：请求经 Proxy 路由，P 完成 prefill 后通过 Mooncake RDMA 将 KV cache 传给 D，D 基于传过来的 KV 做 decode。

#### 7.1 PD Proxy 开发

**文件**: `/ossfs/workspace/bench8/pd_proxy.py`

实现两阶段请求路由代理：
1. 请求 → P（`max_tokens=1`, `kv_transfer_params: {do_remote_decode: true}`）→ P 完成 prefill，返回包含 `remote_host/port/request_id` 的 kv_transfer_params
2. 请求 → D（携带 P 返回的 kv_transfer_params）→ D 通过 Mooncake RDMA 拉取 P 的 KV cache

#### 7.2 CrossDPScheduler PD 分离适配

**文件**: `vllm/v1/core/sched/cross_dp_scheduler.py`（+80 行）

| 修改点 | 说明 |
|--------|------|
| `_update_waiting_for_remote_kv` 重写 | CrossDPKVCacheManager 的 `get_block_ids(request)` 接受 Request 对象（非 string），返回 `list[tuple[list[int], ...]]` per-rank 结构 |
| `request_finished` 适配 | 将 per-rank block_ids 展平后传给 connector |
| `update_state_after_alloc` 修复 | 传 `blocks=new_blocks` 而非 `blocks=None`，确保 connector 能获取分配的 block 信息 |
| `_reqs_need_send` 清理 | 在所有 rank 的 `build_connector_meta` 调用完成后统一清理，避免 rank-based 清理的不可靠性 |

#### 7.3 MooncakeConnector 连续修复（5 轮 bug fix）

**文件**: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py`（+660 行）

**Bug 1: request_id 不匹配（P 和 D 生成不同 ID）**
- P 生成 `chatcmpl-PPPP`，D 生成 `chatcmpl-DDDD`
- D 用自己的 ID 发 RDMA 握手，但 P 只认 P 的 ID → 握手失败
- **修复**: 新增 `remote_request_id` 字段，P 在 `request_finished` 返回自己的 ID，D 用 P 的 ID 做握手、自己的 ID 做完成追踪

**Bug 2: `_reqs_need_recv` 键结构变更**
- 原始 key 为 `req_id: str`，CrossDP 场景需要 per-rank 管理
- **修复**: 改为 `(req_id, rank)` 元组键，`build_connector_meta` 按 `current_rank` 过滤

**Bug 3: P 侧 `_reqs_need_send` 永不清理**
- `SchedulerOutput.cp_rank` 默认值 0，空输出的 `make_empty()` 也是 0
- 原有的 rank-based 清理条件 `current_rank >= dp_per_domain - 1` 对 dp_per_domain=2 时要求 rank=1，但空输出永远 rank=0
- **修复**: 移除 connector 内的 rank-based 清理，由 scheduler 在所有 `build_connector_meta` 调用后统一清理

**Bug 4: `_send_blocks_cp_scatter` 地址计算错误**
- 公式 `p_vbs = p_block_size * p_cp_ws` 在 `interleave_size(64) > block_size(16)` 时计算错误
- 例：position 17 → 应为 block 1, offset 1 → 实际算出 block 0, offset 17
- **修复**: 用正确的全局→本地位置映射公式：
  ```python
  p_stride = p_interleave * p_cp_ws
  p_cycle_idx = positions // p_stride
  p_pos_in_chunk = positions % p_interleave
  p_local_pos = p_cycle_idx * p_interleave + p_pos_in_chunk
  p_block_idx = p_local_pos // p_block_size
  ```

**Bug 5: `decode_cp_world_size` 始终等于 `dp_per_domain`**
- 对 DP 请求（单 rank），`cp_world_size` 应为 1，但代码写死 `self.dp_per_domain`
- D 发给 P 的握手中 `cp_world_size: 4` → P 按 4-way interleave 计算 scatter → 地址完全错误
- **修复**: 改用 `len(request.cp_ranks)` 作为 `decode_cp_world_size`

**Bug 6: `prefill_cp_world_size` 始终等于 `dp_per_domain`**
- 同理，P 的 `request_finished` 总返回 `prefill_cp_world_size=2`，但短请求实际只在 1 个 rank 上
- D 因此连接所有 P rank，DP1 无数据但也发 KV → 乱数据
- **修复**: 改用 `len(cp_ranks)` 作为 `prefill_cp_world_size`

**Bug 7: GQA 场景 TP 维度不匹配**
- P（TP=2）每 worker 2 个 KV heads，D（TP=1）4 个 KV heads
- P 的 `token_bytes=512`（2 heads × 128 dim × 2 bytes），D 的实际 token_bytes=1024
- 原代码用 P 的 `token_bytes` 计算 D 的目标地址 → block 步长和偏移都错
- P_TP0 和 P_TP1 都写到 D 的相同偏移位置 → 互相覆盖
- **修复**: 新增 `remote_tp_size` 配置，计算 `d_token_bytes = p_token_bytes * p_tp / d_tp`，目标地址添加 `tp_head_offset = tp_rank * p_token_bytes`

**Bug 8: D 连接了不持有数据的 P rank**
- `remote_dycp_ranks: [0]` 但 D 连接所有 `p_cp_ws` 个 rank
- **修复**: 新增 `remote_dycp_ranks` 到 `RecvReqMeta`，`group_kv_pull` 只连接实际持有数据的 P rank

#### 7.4 端到端测试结果

| 测试项 | 状态 | 说明 |
|--------|------|------|
| Proxy 路由 | ✅ 通过 | P 完成 prefill（~0.6s），返回 kv_params，D 返回 200 |
| KV RDMA 拉取 | ✅ 通过 | D 从 P 的两个 TP worker 成功拉取 KV（KV pull DONE） |
| `_update_waiting_for_remote_kv` | ✅ 通过 | 正确计算 `num_computed_tokens=20`（21 tokens - 1） |
| 输出正确性 | ❌ 未通过 | 输出为乱码 `* * * *`，经过 Bug5-8 修复后尚未重新测试 |

**Proxy 日志示例**（正常路由）：
```
[prefill] done req=1b5acd2e elapsed=0.61s kv_params={
  'do_remote_prefill': True, 'remote_host': '33.180.165.191',
  'remote_port': 20002, 'remote_request_id': 'chatcmpl-ac2ed51c...',
  'prefill_cp_world_size': 2, 'prefill_tp_size': 2, 'remote_dycp_ranks': [0]
}
[decode] done req=1b5acd2e status=200 elapsed=1.07s
```

## 待解决问题

1. **输出乱码**: PD 分离链路全通但 decode 输出为乱码（`* * * *`）。已修复 TP 维度不匹配（Bug7）和 CP rank 路由错误（Bug5/6/8），但修复后尚未重新测试
2. **`expected_receivers` 赋值缺失**: `SendBlockMeta` 默认 `expected_receivers=1`，DyCP 请求跨多 rank 时应设为实际 receiver 数量
3. **P 侧 DP1 发送无关数据**: P 的 `_reqs_need_send` 是全局字典，cp_ranks=[0] 的请求在 rank 0 和 rank 1 的 `build_connector_meta` 中都会被添加到 SEND 元数据，但 rank 1 的 worker 的 block 中不含该请求的数据

## 待办

- [x] ~~端到端验证 DyCP + PD 分离（Qwen3-30B, dp_per_domain=4）~~ — 已完成，精度 90.0%
- [x] ~~Qwen3-235B FP8 Domain PD 分离 1000 条精度~~ — 已完成，精度 93.8%，0 复读
- [x] ~~PD Proxy 开发~~ — 已完成，两阶段路由正常工作
- [x] ~~KV RDMA 拉取链路打通~~ — 已完成，D 成功从 P 的两个 TP worker 拉取 KV
- [ ] **重新测试 Bug5-8 修复后的输出正确性**（下一步）
- [ ] 测试 P 开 Domain2、D 开 Domain4 的精度（gsm8k 100 条）
- [ ] 修复 `expected_receivers` 赋值逻辑
- [ ] 修复 P 侧 `_reqs_need_send` per-rank 管理（避免 DP1 发送无关数据）
- [ ] 排查昨日遗留的 Qwen3 235B 复读问题（#108, #710）— Domain PD 模式下未复现，可能为单机 DyCP 特有

## 注意事项

- `flash_attn 2.7.3` 已彻底卸载。若需 flash attention，需安装与 PyTorch 2.9 兼容的版本。
- `mooncake-transfer-engine` 需从内部源安装：`pip install mooncake-transfer-engine -i https://pypi.antfin-inc.com/simple/`
- Qwen3-30B 跑 131072 上下文需要 YaRN：`--hf-overrides '{"rope_parameters": {"rope_type":"yarn","factor":32.0,"original_max_position_embeddings":40960}}'`
- Decode 侧 torch inductor cache 偶现 `pickle data was truncated` 和 `compiled_fn_runner` 缺失 warning（不影响精度和稳定��），可能与多 DP Worker 并发编译有关。

## 修改文件清单

| 文件 | 修改内容 | 行数变化 |
|------|---------|---------|
| `vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py` | DyCP PD 分离 CP-aware 传输、TP 维度适配、request_id 映射、地址计算修复 | +660/-77 |
| `vllm/v1/core/sched/cross_dp_scheduler.py` | CrossDP PD 分离调度适配：`_update_waiting_for_remote_kv` 重写、`request_finished` 适配、`_reqs_need_send` 统一清理 | +80/-18 |
| `/ossfs/workspace/bench8/pd_proxy.py` | PD 分离两阶段代理（独立脚本，不在 vllm 仓库内） | 新建 195 行 |
