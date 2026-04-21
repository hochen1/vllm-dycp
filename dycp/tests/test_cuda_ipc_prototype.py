"""
Minimal prototype: CUDA IPC one-sided read across GPU processes.
Uses vLLM's CudaRTLibrary for IPC handle management.
"""
import os
import sys
import time
import ctypes
import torch
import torch.multiprocessing as mp

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def _get_cuda_lib():
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
    return CudaRTLibrary()


def writer_fn(rank, shape, dtype_str, handle_queue, done_event):
    """Prefill rank: write data to GPU, share IPC handle."""
    torch.cuda.set_device(rank)
    dtype = getattr(torch, dtype_str)

    tensor = torch.arange(
        shape[0] * shape[1], dtype=dtype, device=f"cuda:{rank}"
    ).reshape(shape)
    print(f"[Writer GPU:{rank}] tensor {tensor.shape}, "
          f"first5={tensor.flatten()[:5].tolist()}", flush=True)

    # Get IPC handle
    cuda_lib = _get_cuda_lib()
    handle = cuda_lib.cudaIpcGetMemHandle(
        ctypes.c_void_p(tensor.data_ptr())
    )
    # handle is cudaIpcMemHandle_t (128 bytes)
    handle_bytes = bytes(handle)

    handle_queue.put({
        "handle": handle_bytes,
        "shape": list(shape),
        "nbytes": tensor.nelement() * tensor.element_size(),
    })
    print(f"[Writer GPU:{rank}] IPC handle shared ({len(handle_bytes)} bytes)", flush=True)

    done_event.wait(timeout=60)
    print(f"[Writer GPU:{rank}] Done", flush=True)


def reader_fn(rank, dtype_str, handle_queue, done_event, writer_gpu):
    """Decode rank: open IPC handle, read data. Pure one-sided."""
    torch.cuda.set_device(rank)
    dtype = getattr(torch, dtype_str)

    info = handle_queue.get(timeout=30)
    handle_bytes = info["handle"]
    shape = info["shape"]
    nbytes = info["nbytes"]
    print(f"[Reader GPU:{rank}] Got handle for shape={shape}, {nbytes} bytes", flush=True)

    # Reconstruct handle
    from vllm.distributed.device_communicators.cuda_wrapper import (
        cudaIpcMemHandle_t,
    )
    handle = cudaIpcMemHandle_t()
    ctypes.memmove(ctypes.byref(handle), handle_bytes, len(handle_bytes))

    # Open handle — gives us a pointer to writer's GPU memory
    cuda_lib = _get_cuda_lib()
    t0 = time.time()
    remote_ptr = cuda_lib.cudaIpcOpenMemHandle(handle)
    t1 = time.time()
    print(f"[Reader GPU:{rank}] Opened handle in {(t1-t0)*1000:.1f}ms", flush=True)

    # Allocate local buffer and copy from remote (one-sided read via NVLink)
    local = torch.empty(shape, dtype=dtype, device=f"cuda:{rank}")

    t2 = time.time()
    # cudaMemcpy(dst, src, size, kind=cudaMemcpyDefault=4)
    cuda_lib.CUDART_CHECK(
        cuda_lib.funcs["cudaMemcpy"](
            ctypes.c_void_p(local.data_ptr()),
            remote_ptr,
            ctypes.c_size_t(nbytes),
            ctypes.c_int(4),  # cudaMemcpyDefault
        )
    )
    torch.cuda.synchronize()
    t3 = time.time()

    # Verify
    expected = torch.arange(
        shape[0] * shape[1], dtype=dtype, device=f"cuda:{rank}"
    ).reshape(shape)
    match = torch.allclose(local, expected)
    copy_ms = (t3 - t2) * 1000
    bw = nbytes / (t3 - t2) / 1e9 if (t3 - t2) > 0 else 0

    print(f"[Reader GPU:{rank}] Copy {nbytes/1024:.0f}KB in {copy_ms:.2f}ms "
          f"({bw:.1f} GB/s)", flush=True)
    print(f"[Reader GPU:{rank}] Data match: {match}", flush=True)
    print(f"[Reader GPU:{rank}] First 5: {local.flatten()[:5].tolist()}", flush=True)

    # Cleanup
        # Note: cudaIpcCloseMemHandle not in vLLM wrapper, skip cleanup
    done_event.set()


def prefill_fn(rank, tokens, kv_dim, dtype_str, handle_queue, done_event):
    """Prefill rank for multi-rank test."""
    torch.cuda.set_device(rank)
    dtype = getattr(torch, dtype_str)

    kv = torch.randn(tokens, kv_dim, dtype=dtype, device=f"cuda:{rank}")

    cuda_lib = _get_cuda_lib()
    handle = cuda_lib.cudaIpcGetMemHandle(ctypes.c_void_p(kv.data_ptr()))
    handle_queue.put({
        "rank": rank,
        "handle": bytes(handle),
        "shape": list(kv.shape),
        "nbytes": kv.nelement() * kv.element_size(),
    })
    done_event.wait(timeout=60)


def decode_fn(rank, num_sources, dtype_str, handle_queue, done_event):
    """Decode rank: read from multiple prefill ranks."""
    torch.cuda.set_device(rank)
    dtype = getattr(torch, dtype_str)

    from vllm.distributed.device_communicators.cuda_wrapper import (
        cudaIpcMemHandle_t,
    )
    cuda_lib = _get_cuda_lib()

    # Collect all handles
    sources = {}
    for _ in range(num_sources):
        info = handle_queue.get(timeout=30)
        sources[info["rank"]] = info

    print(f"[Decode GPU:{rank}] Got {len(sources)} handles", flush=True)

    # Read from each prefill rank (one-sided)
    t_start = time.time()
    all_kv = []
    total_bytes = 0

    for src_rank in sorted(sources.keys()):
        info = sources[src_rank]
        handle = cudaIpcMemHandle_t()
        ctypes.memmove(ctypes.byref(handle), info["handle"], len(info["handle"]))

        remote_ptr = cuda_lib.cudaIpcOpenMemHandle(handle)
        local = torch.empty(info["shape"], dtype=dtype, device=f"cuda:{rank}")

        cuda_lib.CUDART_CHECK(
            cuda_lib.funcs["cudaMemcpy"](
                ctypes.c_void_p(local.data_ptr()),
                remote_ptr,
                ctypes.c_size_t(info["nbytes"]),
                ctypes.c_int(4),
            )
        )
        all_kv.append(local)
        total_bytes += info["nbytes"]
        # skip cudaIpcCloseMemHandle (not in vLLM wrapper)

    torch.cuda.synchronize()
    t_end = time.time()

    full_kv = torch.cat(all_kv, dim=0)
    elapsed_ms = (t_end - t_start) * 1000
    bw = total_bytes / (t_end - t_start) / 1e9 if (t_end - t_start) > 0 else 0

    print(f"[Decode GPU:{rank}] Read {num_sources} ranks: "
          f"{full_kv.shape} = {total_bytes/1024/1024:.1f}MB "
          f"in {elapsed_ms:.1f}ms ({bw:.1f} GB/s)", flush=True)

    done_event.set()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    n_gpus = torch.cuda.device_count()
    print(f"GPUs: {n_gpus}")

    if n_gpus < 2:
        print("Need >= 2 GPUs")
        sys.exit(1)

    # Test 1: Basic IPC (GPU 0 → GPU 1)
    print("\n" + "=" * 60)
    print("Test 1: Basic CUDA IPC (GPU 0 → GPU 1)")
    print("=" * 60)

    q = mp.Queue()
    ev = mp.Event()
    shape = (1024, 576)  # 1024 tokens × 576 KV dim (MLA)

    w = mp.Process(target=writer_fn, args=(0, shape, "bfloat16", q, ev))
    r = mp.Process(target=reader_fn, args=(1, "bfloat16", q, ev, 0))
    w.start(); r.start()
    w.join(30); r.join(30)

    # Test 2: Multi-rank (GPU 0-3 → GPU 4)
    if n_gpus >= 5:
        print("\n" + "=" * 60)
        print("Test 2: Multi-rank IPC (GPU 0,1,2,3 → GPU 4)")
        print("  Simulating CP=4, 27 layers × 256 tokens × 576 dim")
        print("=" * 60)

        q2 = mp.Queue()
        ev2 = mp.Event()
        tokens = 27 * 256  # 27 layers × 256 tokens
        kv_dim = 576

        procs = []
        for gpu in range(4):
            p = mp.Process(target=prefill_fn, args=(gpu, tokens, kv_dim, "bfloat16", q2, ev2))
            p.start()
            procs.append(p)

        d = mp.Process(target=decode_fn, args=(4, 4, "bfloat16", q2, ev2))
        d.start()
        procs.append(d)

        for p in procs:
            p.join(60)

    print("\n" + "=" * 60)
    print("PROTOTYPE COMPLETE")
    print("=" * 60)
