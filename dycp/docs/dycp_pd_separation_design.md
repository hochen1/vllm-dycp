# DyCP PD 分离方案

## Context

当前 DyCP 只支持单机 decode-only 模式。PD 分离（MooncakeConnector）可以工作但 prefill 侧不支持 DyCP。目标是支持 **DyCP + PD 分离**：decode 侧使用 DyCP（dp_per_domain=M，KV cache interleaved），prefill 侧用 TP 计算完整 KV 后通过 Mooncake RDMA 传输到 decode 侧的 interleaved cache。

核心挑战：**prefill 的 KV 是连续存储的，decode 的 KV 是 interleaved 分散在 M 个 rank 上的**。需要在传输时做 scatter。

## 方案概览

**策略：Prefill 侧 scatter 写入。** Prefill 有完整 KV，由 prefill sender 根据 decode 侧的 CP 拓扑信息，直接 RDMA 写到各 decode rank 的正确 block+offset。

不新建 connector 类，而是在现有 MooncakeConnector 中扩展 CP 感知能力。

## 数据流

```
Request 到达 decode → CrossDPScheduler 判定为 long →
分配 interleaved blocks 到 M 个 rank → 进入 WAITING_FOR_REMOTE_KVS →
M 个 decode rank 各自发 ZMQ handshake（含 cp_rank, block_ids）到 prefill →
Prefill sender 收到 M 个 handshake，计算 scatter 映射 →
RDMA 写入各 rank 的 interleaved blocks →
M 个 rank 各自报告完成 → 聚合后 request 进入 WAITING → 正常 DyCP decode
```

## 具体修改

### 1. MooncakeConnector 扩展 CP 元数据

**文件**: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py`

```python
class MooncakeAgentMetadata(msgspec.Struct, ...):
    # 新增字段
    cp_rank: int = 0
    cp_world_size: int = 1
    cp_interleave_size: int = 1
    decode_block_size: int = 16
```

### 2. Prefill sender 实现 CP scatter

**文件**: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py`

在 `MooncakeConnectorWorker._send_blocks` 中，当 `agent_meta.cp_world_size > 1` 时，调用新方法 `_send_blocks_cp_scatter`：

- 遍历 request 的所有 token position
- 根据 interleave 公式 `(pos // interleave_size) % cp_world_size == cp_rank` 筛选属于该 decode rank 的 position
- 计算 src block+offset（prefill 侧连续布局）→ dst block+offset（decode 侧 interleaved 布局）
- 合并连续区间，生成 `(src_ptr, dst_ptr, length)` RDMA 传输列表

Interleaved offset 公式（与 `block_table.py:283-288` 一致）：
```python
virtual_block_offset = pos % (block_size * cp_world_size)
local_block_offset = (
    virtual_block_offset // (cp_world_size * interleave_size) * interleave_size
    + virtual_block_offset % interleave_size
)
```

### 3. SendBlockMeta 引用计数

**文件**: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py`

```python
@dataclass
class SendBlockMeta:
    local_block_ids: list[int]
    ready: threading.Event
    expire_time: float = float("inf")
    expected_receivers: int = 1    # M for CP requests
    completed_receivers: int = 0
```

同一个 request 的 KV 会被 M 个 decode rank 分别请求。Prefill 只需读取一次 src blocks，但写 M 次（到不同 dst）。当 `completed_receivers == expected_receivers` 时才释放。

### 4. CrossDPScheduler 传递 block 分配信息

**文件**: `vllm/v1/core/sched/cross_dp_scheduler.py`

在 `schedule()` 中分配 WAITING_FOR_REMOTE_KVS 请求的 blocks 后，将 per-rank block IDs 传给 connector：

```python
# ~line 1018, 现在 blocks=None 需要改为传实际 blocks
if self.connector is not None:
    self.connector.update_state_after_alloc(
        request=request,
        blocks=new_blocks,
        num_external_tokens=num_external_computed_tokens,
    )
```

### 5. CrossDPKVCacheManager 暴露 per-rank block IDs

**文件**: `vllm/v1/core/cross_dp_kv_cache_manager.py`

新增方法：
```python
def get_per_rank_block_ids(self, request: Request) -> dict[int, list[int]]:
    """返回 {cp_rank: [block_id, ...]} 的映射"""
```

connector 用这个信息告诉 prefill 各 decode rank 的目标 blocks。

### 6. Decode 侧 receive 路径

**文件**: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py`

`MooncakeConnectorWorker.start_load_kv` 中，当检测到 CP 模式时：
- 在 ZMQ handshake 中附加 `cp_rank`, `cp_world_size`, `cp_interleave_size`
- 每个 decode rank 独立发送 handshake，只包含自己 rank 的 block_ids
- Prefill 收到后按 rank 做 scatter

### 7. Prefill 侧 `request_finished` 传递 CP 信息

**文件**: `vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py`

`kv_transfer_params` 返回值增加：
```python
{
    "do_remote_prefill": True,
    "cp_world_size": M,
    "cp_interleave_size": interleave_size,
}
```

decode 侧收到后知道这是 CP request，分配 interleaved blocks。

### 8. 配置与接入

**文件**: `vllm/config/kv_transfer.py`, `vllm/v1/engine/core.py`

- `kv_connector_extra_config` 中增加 `decode_cp_world_size` 和 `decode_cp_interleave_size`
- 当 `dp_per_domain > 1` 且有 `kv_transfer_config` 时自动启用 CP scatter 逻辑

## 关键文件清单

| 文件 | 修改内容 |
|------|---------|
| `mooncake_connector.py` | CP 元数据、scatter 传输、引用计数 |
| `cross_dp_scheduler.py` | 传递 block 分配给 connector |
| `cross_dp_kv_cache_manager.py` | 暴露 per-rank block IDs |
| `block_table.py` | 参考 interleave 公式（不需要修改） |
| `kv_transfer.py` | 配置验证 |
| `core.py` | 启用条件判断 |

## 验证方案

1. **单元测试**: 写 `test_cp_scatter_mapping.py` 验证 interleave 映射正确性
2. **集成测试**: Qwen3-30B PD 分离 + DyCP，发送 1 条 32k 请求，验证 decode 能正确读取 interleaved KV
3. **精度测试**: gsm8k 100 条，对比 PD+DyCP vs 纯 DP 精度
4. **性能测试**: 对比 PD+DyCP vs 纯 PD 的 TTFT 和 throughput

## 风险

1. **Block size 不一致**: Prefill block_size=16, Decode block_size=64*CP。scatter 需要 token 级别映射，不能做 block 级别 copy
2. **TP 到 DP 的 KV head 映射**: Prefill TP=4 每个 rank 只有 1/4 的 KV heads。Decode TP=1 需要所有 heads。需要配合 `TpKVTopology` 确保每个 decode rank 从正确的 prefill TP rank 获取数据
3. **M 个 handshake 的并发**: Prefill sender 需要处理同一 request 的 M 个并发请求，不能在第一个完成后就释放
