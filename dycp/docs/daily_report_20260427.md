# DyCP 工作日报 — 2026-04-27

## 分支状态

- **分支**: `dev-dycp-prefill-gqa-2`
- **今日修改文件**:
  - `vllm/attention/ops/common.py` — NaN*0 修复 in `_correct_attn_cp_out_kernel`
  - `vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py` — D 端 KV 验证日志（延续昨日）

## 今日完成的工作

### 1. Bug 15 根因缩小：RDMA 数据验证通过

**结论**: D 端 KV_VERIFY 日志确认 RDMA 传输数据 100% 正确。

DyCP 请求 (160 tokens, cp_world_size=4) 的 D 端 KV 验证：
- rank 0 (block 26): K_sum=59351.77, K_nz=32759/32768 ✓
- rank 1 (block 3): K_sum=60764.48, K_nz=32762/32768 ✓
- rank 2 (block 8): K_sum=63070.96, K_nz=32765/32768 ✓
- rank 3 (block 21): K_sum=0.00, K_nz=0/32768 （预期：160 tokens < 192，rank 3 分到 0 tokens）✓

**排除**: RDMA 传输数据损坏、scatter 地址计算错误、block_table slot mapping 不匹配。

### 2. NaN*0 修复

**问题**: 在 `_correct_attn_cp_out_kernel` 中，当 rank 3 的 seqused_k=0 时，flash_attn 可能返回 NaN 的 output。triton kernel 正确计算 factor=exp(-inf)=0，但 IEEE 754 中 NaN*0=NaN，NaN 通过 allreduce 传播到所有 rank。

**修复**: 在 triton kernel 中添加 NaN sanitize：
```python
output = output * factor
# NaN * 0 = NaN in IEEE 754; when seqused_k=0 flash_attn returns NaN output
output = tl.where(output != output, 0.0, output)
```

**验证结果**:

| 模式 | 乱码数 | 精度 | 说明 |
|------|--------|------|------|
| CUDA graph（修复前，旧 triton 缓存）| 14/99 | 73.7% | 与昨日 11/99 相当，NaN fix 未生效 |
| enforce-eager（修复后，triton 缓存清除）| 0/23 | 87.0% | DyCP 请求正确解码 ✓ |
| CUDA graph（修复后，triton 缓存清除）| 14/99 | 73.7% | 仍有乱码！ |

**关键发现**: enforce-eager 模式 0 乱码，CUDA graph 模式 14 乱码。说明 NaN fix 在 eager 模式下有效，但 CUDA graph 模式存在额外问题。

### 3. 乱码请求特征分析

14 个乱码请求的特征：
- prompt_chars 全部在 531-766（较长），正常请求在 300-422（较短）
- 较长 prompt → 更可能被调度为 DyCP (cp_world_size=4)
- 所有乱码输出从第一个 token 开始就是 `\n\n\n...` 或重复垃圾字符
- 非 NaN/inf 导致的采样错误？logits 可能全 0 或被错误地 softmax

### 4. DyCP Decode 诊断日志（gpu_model_runner）

**文件**: `vllm/v1/worker/gpu_model_runner.py`（+29 行）

为排查 Bug 15b（CUDA graph + DyCP 乱码），在 `execute_model()` 的 logits 计算之后添加诊断日志：

**触发条件**: `dycp_world_size > 1` && `num_cp_request > 0` && CUDAGraph 模式

**输出信息**:
- DyCP 请求 vs DP 请求的 hidden_states norm 对比
- DyCP 和 DP 的 logits max/min 值
- DyCP 和 DP 的 top token IDs

**用途**: 对比 DyCP 请求和 DP 请求的输出分布差异，判断是 attention 阶段数据错误还是采样阶段 indices 错误。

### 5. 当前排查方向

**采样阶段排查**（用户提示）:
- 全是 `\n` 的输出更像采样阶段错误，而非 attention 数据损坏
- 可能的原因：CUDA graph replay 时 DyCP 请求的 logits_indices 错误，导致取到了错误 rank 的 logits
- DyCP 在 `gpu_model_runner.py` 中使用 `logits_indices allgather` 来确保只在正确的 rank 上采样
- 如果 CUDA graph 没有正确 capture logits_indices 的动态值，可能导致采样出 `\n`

**CUDA graph 排查**:
- `cudagraph_capture_sizes_for_cp=2` 已设置，allreduce 路径应该被正确 capture
- 但 logits_indices、batch ordering 等动态数据在 CUDA graph replay 时是否正确更新？
- 需要对比 eager 模式和 CUDA graph 模式下 DyCP 请求的 logits 输出

## 进行中的测试

- [ ] enforce-eager 完整 100 条精度测试（当前在跑，预计 ~60 分钟）
  - 目的：确认 eager 模式下精度完全正确

## Bug 状态

| Bug | 严重度 | 状态 | 说明 |
|-----|--------|------|------|
| Bug 15a: NaN*0 in allreduce | High | ✅ 已修复 | triton kernel NaN sanitize |
| Bug 15b: CUDA graph + DyCP 乱码 | **Critical** | 🔍 排查中 | eager 正确，CUDA graph 乱码 |

**累计 Bug 修复进度（04-25 至今）**:
- Bug 1-14: 已修复
- Bug 15a: 04-27 已修复（NaN*0 sanitize）
- Bug 15b: 04-27 排查中（CUDA graph + DyCP 交互问题）

## 待办

- [ ] **排查 Bug 15b**: CUDA graph 模式下 DyCP 输出乱码
  - 利用新增的 gpu_model_runner 诊断日志，对比 eager vs CUDA graph 的 hidden_states 和 logits
  - 排查采样阶段：logits_indices 在 CUDA graph replay 时是否正确
  - 排查 DyCP 的 batch reorder 和 logits allgather 是否被正确 capture
- [ ] 确认 enforce-eager 100 条精度（测试中）
- [ ] 测试 P Domain2 (TP=2) + D Domain4 (TP=1) 的精度
- [ ] 排查 Qwen3-235B 单机 DyCP 复读问题
- [ ] 确认 Bug 15a/15b 修复后清理诊断日志代码
- [ ] 提交今日修改（3 文件 +106 行）

## 未提交修改清单

| 文件 | 修改内容 | 行数变化 |
|------|---------|---------|
| `vllm/attention/ops/common.py` | Bug 15a: CP kernel NaN 清零 | +2 |
| `vllm/v1/worker/gpu_model_runner.py` | DyCP decode 诊断日志（Bug 15b 排查） | +29 |
| `vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py` | PD 分离 KV 数据两端校验日志 | +75 |

## 注意事项

- 清除 triton 缓存后重启服务才能使 kernel 修改生效：`rm -rf ~/.triton/cache/* /tmp/torchinductor_root/*`
- enforce-eager 模式约慢 3x，但可排除 CUDA graph 问题
- Mooncake batch_size=2 时约 1/100 概率有请求永久挂起
- **诊断日志使用 logger.warning/info**: 确认问题修复后应降级或移除，避免线上性能影响
- **当前分支 ahead of origin 2 commits**，今日修改（3 文件 +106 行）尚未 commit
