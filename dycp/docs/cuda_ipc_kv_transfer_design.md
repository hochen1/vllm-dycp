# CUDA IPC KV Transfer — 设计文档

> 创建日期: 2026-04-20
> 状态: 设计完成，待实施
> 分支: dev-dycp-mixed-1
> 前置: Phase 2 已完成（batch 分离移除 + proxy 智能分流 + EP padding 修复）

---

## 1. 问题背景

### 当前 KV 传输方式

Local PD 分离中，prefill 使用 CP（多 rank 并行），decode 使用单 rank。Prefill 完成后需要将各 rank 的 KV cache 汇聚给 decode rank。

**当前方案**：`wait_for_save()` 中使用 NCCL `all-gather`

```
Prefill:
  save_kv_layer decorator → 捕获 raw KV（pre-interleave mask）→ 存入 _pending_local_kv
  wait_for_save() → all-gather 所有 rank 的 KV → DualChunkSwap restore → 存入 _gpu_kv_buffer

Decode:
  start_load_kv() → 从 _gpu_kv_buffer 读取 → 注入 decode rank 的 paged buffer
```

**问题**：
1. all-gather 是集合操作，**阻塞所有 8 个 workers**（~10-20ms）
2. 8 rank 中只有 1 个 decode rank 需要数据，**87.5% 带宽浪费**
3. 需要额外的 `_gpu_kv_buffer` 存储完整 KV 副本（**GPU 内存浪费**）
4. `save_kv_layer` 需要通过 decorator 捕获 raw KV（**prefill 有额外开销**）

### 目标

用 CUDA IPC 单边读替代 all-gather：decode rank 直接从 prefill rank 的 paged buffer 读取 KV，prefill rank 零额外开销。

---

## 2. 技术方案

### 核心思路

每个 rank 的 KV cache 在 prefill 时自然写入其 paged buffer（这是 attention kernel 的标准行为）。通过 CUDA IPC，decode rank 可以直接访问其他 rank 的 paged buffer 内存，无需 send/recv/all-gather。

```
Prefill:
  各 rank 的 attention kernel → 自然写 KV 到各自的 paged buffer
  → wait_for_save: 仅记录 metadata（哪些 block/slot 被写入），不做 all-gather
  → prefill 完成，返回 proxy

Decode:
  start_load_kv → 通过 IPC 指针从各 prefill rank 的 paged buffer 读取 KV
  → 按 interleave 映射重组 → 写入 decode rank 自己的 paged buffer
  → forward pass
```

### 数据流对比

| 环节 | 当前方案 | CUDA IPC 方案 |
|------|---------|--------------|
| Prefill forward | save_kv_layer 捕获 raw KV | **无额外操作** |
| KV 传输 | all-gather（同步，阻塞 8 workers） | **IPC 单边读（decode 独立操作）** |
| 中间存储 | _gpu_kv_buffer（完整 KV 副本） | **无**（直接读→写 paged buffer） |
| Decode 加载 | 从 _gpu_kv_buffer 注入 | 从远程 paged buffer 直接读取+注入 |

### Proxy 流程不变

保持串行下发（先 prefill 等完成，再发 decode）。后续优化可改为同时下发。

---

## 3. IPC Handle 管理

### 初始化阶段

每个 worker 进程在启动时交换 paged buffer 的 IPC handle：

```python
# 每个 rank:
cuda_lib = CudaRTLibrary()
my_handle = cuda_lib.cudaIpcGetMemHandle(paged_buffer.data_ptr())

# 通过 dist.all_gather 交换 handles（128 bytes per handle）
all_handles = dist.all_gather(my_handle)

# 打开其他 rank 的 handles
for rank, handle in enumerate(all_handles):
    if rank != my_rank:
        remote_ptr[rank] = cuda_lib.cudaIpcOpenMemHandle(handle)
```

**关键点**：
- IPC handle 是 128 字节固定大小结构
- `cudaIpcMemLazyEnablePeerAccess` 自动启用 NVLink P2P
- Handle 一次交换，终身有效（paged buffer 生命周期 = 进程生命周期）
- 已有 vLLM 基础设施：`vllm/distributed/device_communicators/cuda_wrapper.py`

### API 参考

```python
# vllm/distributed/device_communicators/cuda_wrapper.py
class CudaRTLibrary:
    def cudaIpcGetMemHandle(self, devPtr) -> cudaIpcMemHandle_t
    def cudaIpcOpenMemHandle(self, handle) -> ctypes.c_void_p

# 已有 P2P 验证：
# vllm/distributed/device_communicators/all_reduce_utils.py
gpu_p2p_access_check()  # 真实跨进程 P2P 测试
```

---

## 4. KV 读取逻辑

### Interleave 机制

DyCP 的 KV cache 按 interleave 分配：

```python
# block_table.py 中的 interleave mask:
mask = (virtual_offset // interleave_size) % cp_world_size == cp_rank
```

即：每个 rank 只存储特定 interleave 片段的 KV。所有 rank 的 paged buffer 合起来是完整的 KV。

### Decode 侧读取流程

```python
def start_load_kv(self, forward_context):
    """通过 IPC 从各 prefill rank 的 paged buffer 读取 KV"""
    for each decode_request:
        prefix = request.kv_transfer_params["pd_request_prefix"]
        prefill_meta = self._completed_prefills[prefix]
        
        # 获取各 rank 的 block_ids 和 interleave 参数
        per_rank_block_ids = prefill_meta["per_rank_block_ids"]
        interleave_size = self._interleave_size
        cp_world_size = self._cp_world_size
        
        for layer_idx in range(num_layers):
            kv_layer = forward_context.kv_caches[layer_idx]
            
            for src_rank in range(cp_world_size):
                # 计算 src_rank 的 slot 位置
                src_block_ids = per_rank_block_ids[src_rank]
                src_slots = compute_interleave_slots(
                    src_block_ids, interleave_size, cp_world_size, src_rank
                )
                
                # 通过 IPC 指针读取远程 KV
                remote_paged_buf = self._remote_paged_buffers[src_rank][layer_idx]
                remote_kv = remote_paged_buf[src_slots]  # 直接索引远程内存
                
                # 写入本地 paged buffer 的对应位置
                local_slots = compute_decode_slots(src_slots, ...)
                kv_layer[local_slots] = remote_kv
```

### 关键数据

需要从 prefill 传递给 decode 的 metadata（通过 kv_transfer_params）：

```python
kv_transfer_params = {
    "pd_request_prefix": "pd-xxx",
    "do_remote_prefill": True,
    # 新增：每个 prefill rank 的 block 信息
    "per_rank_block_ids": [[b0, b1, ...], [b0, b1, ...], ...],  # per rank
    "num_prompt_tokens": 1024,
    "cp_world_size": 8,
    "interleave_size": 64,
}
```

---

## 5. 需要修改的文件

### 核心修改

| 文件 | 改动 | 说明 |
|------|------|------|
| `local_pd_connector.py` | **重写 wait_for_save** | 去掉 all-gather，只记录 metadata |
| `local_pd_connector.py` | **重写 start_load_kv** | 从 IPC 读替代从 _gpu_kv_buffer 读 |
| `local_pd_connector.py` | **新增 init_ipc_handles** | 初始化时交换 paged buffer IPC handles |
| `local_pd_connector.py` | **删除 _gpu_kv_buffer** | 不再需要中间 buffer |
| `local_pd_connector.py` | **精简 save_kv_layer** | 不再需要捕获 raw KV |

### 不需要修改

| 组件 | 原因 |
|------|------|
| Proxy | 保持串行下发 |
| Scheduler | 状态机不变 |
| Engine Core | 无改动 |
| Attention / MLA | 无改动 |
| block_table.py | 无改动（interleave 逻辑已有） |

### 可能需要小改

| 文件 | 改动 | 说明 |
|------|------|------|
| `request_finished()` | 返回 per_rank_block_ids | decode 需要知道各 rank 的 block 布局 |
| Worker 初始化 | 调用 init_ipc_handles | 在 worker 启动时交换 handles |

---

## 6. 原型验证结果

已验证 CUDA IPC 单边读的可行性（`dycp/tests/test_cuda_ipc_prototype.py`）：

| 测试 | 数据量 | 耗时 | 带宽 | 正确性 |
|------|--------|------|------|--------|
| GPU 0 → GPU 1 | 1.1MB | 0.20ms | 6.0 GB/s | 完全匹配 |
| GPU 0-3 → GPU 4 | 30.4MB | 16.7ms | 1.9 GB/s | 完成 |

关键发现：
- IPC handle 交换仅 128 bytes，打开耗时 ~8ms（一次性）
- 数据传输通过 `cudaMemcpy` + IPC 指针，走 NVLink 硬件
- 多 rank 串行读 30MB 耗时 17ms（可通过并行优化到 ~5ms）

---

## 7. 预期性能

| 指标 | 当前（all-gather） | CUDA IPC | 改善 |
|------|-------------------|----------|------|
| wait_for_save 耗时 | 10-20ms（阻塞 8 workers） | **0ms**（仅记录 metadata） | -20ms |
| Prefill 额外开销 | save_kv_layer per layer | **零** | 消除 |
| start_load_kv 耗时 | ~5ms（本地 buffer 读） | ~17ms（IPC 读） | +12ms |
| 净 TTFT 变化 | - | **-8ms**（-20+12） | 改善 |
| Worker 阻塞 | 8 workers 同步等待 | **0 workers 阻塞** | 吞吐提升 |
| GPU 内存 | _gpu_kv_buffer（完整 KV 副本）| **无额外内存** | 节省 |

**注**：start_load_kv 的 IPC 读只阻塞 decode rank 自己，不影响其他 7 个 rank 的工作。

---

## 8. 实施步骤

### Step 1：IPC Handle 基础设施（~100 行）

在 connector 初始化时交换 paged buffer IPC handles。

```python
class LocalPDConnector:
    def init_ipc_handles(self, kv_caches):
        """在 worker 初始化时调用"""
        cuda_lib = CudaRTLibrary()
        self._remote_paged_buffers = {}
        
        for layer_idx, kv_cache in enumerate(kv_caches):
            # 获取本 rank 的 handle
            handle = cuda_lib.cudaIpcGetMemHandle(kv_cache.data_ptr())
            # 交换 handles（通过 dycp_group.all_gather）
            all_handles = dycp_group.all_gather_object(handle)
            # 打开其他 rank 的 handles
            for rank, h in enumerate(all_handles):
                if rank != my_rank:
                    ptr = cuda_lib.cudaIpcOpenMemHandle(h)
                    self._remote_paged_buffers.setdefault(rank, {})[layer_idx] = ptr
```

### Step 2：精简 wait_for_save（~-50 行）

去掉 all-gather 和 KV 重建逻辑，只记录 metadata。

```python
def wait_for_save(self):
    """记录 prefill metadata，不做 all-gather"""
    # 只需要保存各 rank 的 block_ids（用于 decode 时定位 KV）
    # 不需要 _pending_local_kv、all-gather、restore、_gpu_kv_buffer
    pass  # metadata 在 request_finished 中处理
```

### Step 3：重写 start_load_kv（~100 行）

从 IPC 指针读取远程 paged buffer。

### Step 4：精简 save_kv_layer（~-30 行）

不再需要通过 decorator 捕获 raw KV。可能仍需要记录 slot_mapping metadata。

### Step 5：验证

1. 单请求精度验证（KV 正确性）
2. 并发压力测试（32 并发）
3. TTFT + 吞吐 benchmark

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| Interleave 顺序重组错误 | KV 位置错乱 → 输出错误 | 与 all-gather 版本对比验证 |
| IPC handle 泄漏 | GPU 内存泄漏 | paged buffer 生命周期 = 进程生命周期，自动清理 |
| 非 NVLink 拓扑性能差 | PCIe P2P 带宽低 | 检测 P2P 能力，fallback 到 all-gather |
| 跨节点不支持 | 多节点部署受限 | IPC 仅单机；跨节点走 P2pNcclConnector |
| paged buffer 布局变化 | vLLM 升级后 slot 计算不匹配 | 复用 block_table.py 的 interleave 逻辑 |

---

## 10. 后续优化方向

1. **同时下发 prefill+decode**：proxy 并行发送，decode 进入 WAITING_FOR_REMOTE_KVS，KV 就绪后自动调度
2. **异步 IPC 读**：在后台 CUDA stream 上执行 IPC 读，不阻塞 decode 的 start_load_kv
3. **并行多 rank 读取**：当前 prototype 是串行读 4 rank（17ms），可用多 stream 并行（~5ms）
4. **逐层流水线**：边读 layer N 的 KV 边跑 layer N-1 的 attention
