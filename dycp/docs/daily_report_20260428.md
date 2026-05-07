# DyCP 工作日报 — 2026-04-28

## 分支状态

- **分支**: `dev-dycp-prefill-gqa-2`
- **最新提交**: `98bc12058` [bugfix] fix Domain PD separation: P crash and D recv routing (04-26)
- **未提交修改**: 7 文件，+261 行（净增）

## 今日完成的工作

### 1. Bug 16 修复: aggregate_domain() cp_size 乘数缺失

**文件**: `vllm/distributed/kv_transfer/kv_connector/utils.py` (+62 行)

**问题**: `aggregate_domain()` 对 finished_recving 的计数逻辑错误。DyCP 请求被分配到所有 4 个 DP worker，每个 worker 都会 vote "收到了"，但 aggregate_domain 的 `default_count` 默认为 1，只收到 1 个 worker 的 vote 就认为完成了。实际需要等 4 个 worker 全部 vote 完才算完成。

**修复**:
- 在 `KVConnectorOutput` 中新增 `req_id_to_cp_size` 字段，从 worker 层传递每个请求的 CP world size
- `aggregate_domain()` 使用 `cp_size` 作为 `finished_recving` 的完成阈值（DyCP 请求需要 4 个 vote，普通请求需要 1 个）
- Worker 端通过 `kv_connector_model_runner_mixin.py` 传递 `req_id_to_cp_size`

**涉及文件**:
- `vllm/distributed/kv_transfer/kv_connector/utils.py` — aggregate_domain 聚合逻辑
- `vllm/v1/outputs.py` — KVConnectorOutput 新增 req_id_to_cp_size 字段
- `vllm/v1/worker/kv_connector_model_runner_mixin.py` — Worker 层传递 cp_size

### 2. Bug 17 修复: 非 DyCP 请求 default_count 错误

**问题**: Bug 16 修复后引入的回归 — 非 DyCP 请求（cp_size=1）的 `default_count` 仍然使用了错误的值，导致普通请求也需要多次 vote 才能完成。

**修复**: 非 DyCP 请求的 default_count 固定为 1。

### 3. Bug 18 修复: Scheduler 二次计数导致 DyCP 请求永久卡住

**文件**: `vllm/v1/core/sched/cross_dp_scheduler.py` (+47 行，-20 行)

**根因分析（核心发现）**:

KV 完成上报分三层：
1. **Worker 层**: 各 worker 独立 vote `receive_kv finished`
2. **Executor 层**: `aggregate_domain()` 聚合所有 worker 的 vote，只有全部 worker 都 vote 了才放入 `finished_recving`
3. **Scheduler 层**: `_update_from_kv_xfer_finished()` 处理 finished_recving

**Bug**: Scheduler 原本做了**第二层计数** — 对每个 DyCP 请求维护 `_kv_recv_remaining` 计数器（初始值=cp_world_size=4），每次收到 `finished_recving` 事件减 1，减到 0 才标记完成。但 `aggregate_domain()` 已经完成了跨 worker 聚合，产出的 `finished_recving` 是最终结果，不需要再计数。

**更严重的是**: `step_domain()` 流程中，`execute_model()` 调用 `aggregate_domain()` 后，如果 `None in model_outputs`，会再调用 `sample_tokens()` → 第二次 `aggregate_domain()`。第二次调用**覆盖**了第一次的结果，导致 `finished_recving` 信号丢失。Scheduler 只收到 1 次事件（预期 4 次），`remaining` 永远停在 3/4。

**证据**: 日志显示请求 `chatcmpl-9629d1ee9df347c5`（DyCP, cp_ranks=4）在 aggregate_domain 中 4→0 完成聚合，但 scheduler 只收到 1 次事件后 remaining=3，永久卡住。

**修复**: 移除 scheduler 中的 `_kv_recv_remaining` 计数器，直接将 `finished_recving` 视为最终完成信号，用 `_kv_recv_completed` 集合做去重。

### 4. PD 分离 KV 数据完整性校验

**文件**: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py` (+159 行)

- P 端 RDMA write 前验证：检查源 KV block 数据非零（abs_sum、abs_max、nonzero 统计）
- D 端 KV 接收后全量验证：检查所有 block 的 K/V 数据、per-head nonzero 统计
- D 端 RDMA 完成后 CUDA synchronize，确保 GPU 读取到最新数据
- 大量 `[PD]` 前缀日志用于追踪 start_load_kv、get_finished、receive_kv vote 等状态

### 5. 精度验证（100 条 GSM8K）

**测试环境**: Qwen3-30B-A3B, PD 分离模式 (P: TP=4, D: DP=4, TP=1, dp_per_domain=4)

**结果**: 100 条全部发出，proxy 日志确认 100 条全部完成响应，FAIL=0。ais_bench 显示 FIN=99/100（可能因两个 ais_bench 实例同时运行导致最后一条统计延迟）。

**关键验证**:
- Bug 18 修复后无 DyCP 请求卡住
- DyCP 请求（num_cp_ranks=4）和普通请求（num_cp_ranks=1）都成功完成 KV 接收
- aggregate_domain 聚合正常工作，scheduler 正确标记完成

### 6. 昇腾 Patch 参考

阅读了 `dycp/docs/domain_pd_vllm_ascend.patch`，对比了昇腾在 PD 分离场景下的实现方式。昇腾在 worker 层（NPUModelRunner）填充 `req_id_to_cp_size`，与我们的实现路径一致。

## Bug 状态

| Bug | 严重度 | 状态 | 说明 |
|-----|--------|------|------|
| Bug 15a: NaN*0 in allreduce | High | ✅ 已修复 | triton kernel NaN sanitize |
| Bug 15b: CUDA graph + DyCP 乱码 | Critical | ⏸️ 暂停 | 今日优先修 PD 分离卡住问题 |
| Bug 16: aggregate_domain cp_size | High | ✅ 已修复 | cp_size 作为完成阈值 |
| Bug 17: 非 DyCP default_count | Medium | ✅ 已修复 | 回归修复 |
| Bug 18: Scheduler 二次计数 | **Critical** | ✅ 已修复 | 移除冗余计数，信任 aggregate 结果 |

**累计 Bug 修复进度（04-25 至今）**:
- Bug 1-15a: 已修复
- Bug 15b: 排查中（CUDA graph + DyCP 乱码，暂停）
- Bug 16-18: 04-28 已修复（PD 分离 DyCP 卡住的三个关联 bug）

## 未提交修改清单

| 文件 | 修改内容 | 行数变化 |
|------|---------|---------|
| `vllm/attention/ops/common.py` | CP kernel NaN 清零 | +2 |
| `vllm/distributed/kv_transfer/kv_connector/utils.py` | aggregate_domain cp_size 聚合逻辑 + 日志 | +62 |
| `vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py` | P/D 端 KV 数据校验 + 状态日志 | +159 |
| `vllm/v1/attention/backends/mla/common.py` | 移除旧的诊断日志 | -6 |
| `vllm/v1/core/sched/cross_dp_scheduler.py` | Bug 18 修复：移除二次计数 | +47/-20 |
| `vllm/v1/outputs.py` | KVConnectorOutput 新增 req_id_to_cp_size | +3 |
| `vllm/v1/worker/kv_connector_model_runner_mixin.py` | Worker 层传递 cp_size | +2 |

## 待办（后续工作）

### 高优先级
- [ ] **单独重跑一次 100 条精度测试**：之前两个 ais_bench 实例同时运行导致结果可能不准，需要独立运行一次确认精度分数
- [ ] **清理诊断日志和 KV 校验代码**：mooncake_connector.py 中的 `.item()` 调用和 KV_VERIFY 日志影响性能，需要在确认修复后移除或降级
- [ ] **修复 `print()` 调用**：`request_queue.py:270` 使用了 print 而非 logger
- [ ] **减少 aggregate_domain 日志量**：当前每个 step 都打印，即使没有 recv/send 活动也打印，需要加条件过滤

### 中优先级
- [ ] **继续排查 Bug 15b**: CUDA graph + DyCP 乱码问题
  - 重点：logits_indices 在 CUDA graph replay 时是否正确更新
  - 重点：DyCP batch reorder 在 graph capture/replay 的一致性
- [ ] **提交代码**：7 个文件 +261 行未提交，建议按功能分 commit

### 低优先级
- [ ] 测试 P Domain2 (TP=2) + D Domain4 (TP=1) 精度
- [ ] 排查 Qwen3-235B 单机 DyCP 复读问题
- [ ] 跑完精度测试后考虑跑 benchmark 性能测试

## 注意事项

- 清除 triton 缓存后重启服务才能使 kernel 修改生效：`rm -rf ~/.triton/cache/* /tmp/torchinductor_root/*`
- 不要同时启动两个 ais_bench 实例对同一个端口发请求，会导致统计混乱
- 诊断日志使用 `logger.info`/`logger.warning` 级别，确认问题后应降级或移除
- Mooncake batch_size=2 时约 1/100 概率有请求永久挂起
- 当前 7 个文件 +261 行未提交，建议尽快 commit
- enforce-eager 模式约慢 3x，但可排除 CUDA graph 相关问题
