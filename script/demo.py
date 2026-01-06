import torch
import torch_npu
import math
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("hccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_distributed_attention(rank, world_size):
    print(f"Running on rank {rank}.")
    setup(rank, world_size)

    torch.npu.set_device(rank)
    torch.manual_seed(42)
    q = torch.randn(1, 8, 1, 128, dtype=torch.float16).npu()
    k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    
    scale = 1 / math.sqrt(128.0)
    actseqlen = [1]
    
    if rank == 0:
        k_local = k[:, :, :512, :]
        v_local = v[:, :, :512, :]
        actseqlenkv_local = [512]
    else:
        k_local = k[:, :, 512:, :]
        v_local = v[:, :, 512:, :]
        actseqlenkv_local = [512]

    out_local, lse_local = torch_npu.npu_fused_infer_attention_score(
        q, k_local, v_local, 
        actual_seq_lengths=actseqlen, 
        actual_seq_lengths_kv=actseqlenkv_local,
        num_heads=8, 
        input_layout="BNSD", 
        scale=scale, 
        pre_tokens=65535, 
        next_tokens=65535, 
        softmax_lse_flag=True
    )
    
    denominator_local = torch.exp(lse_local)
    numerator_local = out_local * denominator_local

    dist.all_reduce(numerator_local, op=dist.ReduceOp.SUM)
    
    dist.all_reduce(denominator_local, op=dist.ReduceOp.SUM)

    out_all_reduced = numerator_local / denominator_local
    lse_all_reduced = torch.log(denominator_local)

    if rank == 0:
        print("\n--- Verification on Rank 0 ---")
        
        out_full, lse_full = torch_npu.npu_fused_infer_attention_score(
            q, k, v, 
            actual_seq_lengths=actseqlen, 
            actual_seq_lengths_kv=[1024],
            num_heads=8, 
            input_layout="BNSD", 
            scale=scale, 
            pre_tokens=65535, 
            next_tokens=65535, 
            softmax_lse_flag=True
        )
        
        print("Shape of full output:", out_full.shape)
        print("Shape of reduced output:", out_all_reduced.shape)
        print("\nLSE from full calculation:\n", lse_full)
        print("LSE from reduced calculation:\n", lse_all_reduced.to(torch.float16))
        
        print("\nOutput from full calculation (first 5 elements):\n", out_full.flatten()[:5])
        print("Output from reduced calculation (first 5 elements):\n", out_all_reduced.flatten()[:5])
        
        out_all_reduced = out_all_reduced.to(torch.float16)
        are_they_close = torch.allclose(out_full, out_all_reduced, rtol=1e-2, atol=1e-4)
        print(f"\nAre the full output and reduced output close? -> {are_they_close}")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    try:
        mp.spawn(run_distributed_attention,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
    except Exception as e:
        print(f"Error: Could not spawn processes. Make sure you have {world_size} NPUs available.")
        print(f"Original error: {e}")

