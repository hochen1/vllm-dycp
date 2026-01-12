import argparse
import sys
import torch
from typing import Any

from vllm.platforms import current_platform

try:
    from vllm.vllm_flash_attn import (
        fa_version_unsupported_reason,
        flash_attn_varlen_func,
        is_fa_version_supported,
    )
except ImportError:
    raise

# 测试配置
QUICK_CASES = [
    dict(
        seq_lens=[(1, 4096)],  # q_len 1, kv_len 4k
        num_heads=(8, 8),
        head_size=128,
        dtype=torch.bfloat16,
        block_size=128,
        num_blocks=32,
        fa_version=3,
    )
]

def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
    x_flat = x.to(torch.float64).flatten()
    y_flat = y.to(torch.float64).flatten()
    
    cos_sim = (x_flat @ y_flat) / (x_flat.norm() * y_flat.norm()).clamp_min(1e-12)
    cos_diff = 1 - cos_sim.item()
    
    print(f"[{name}] Cosine Difference: {cos_diff:.6e}")
    assert cos_diff < 1e-4, f"{name} difference too large! Check implementation."

def test_dycp_flashattn(case: dict[str, Any]) -> None:
    seq_lens = case["seq_lens"]
    num_heads = case["num_heads"]
    head_size = case["head_size"]
    dtype = case["dtype"]
    block_size = case["block_size"]
    num_blocks = case["num_blocks"]
    fa_version = case["fa_version"]

    torch.set_default_device("cuda")
    current_platform.seed_everything(42)

    if not is_fa_version_supported(fa_version):
        print(f"Skip FA{fa_version}: {fa_version_unsupported_reason(fa_version)}")
        return

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    global_key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype)
    global_value_cache = torch.randn_like(global_key_cache)
    
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)
    
    max_num_blocks_per_seq = (max(kv_lens) + block_size - 1) // block_size
    global_block_table = torch.arange(num_blocks, dtype=torch.int32).view(num_seqs, max_num_blocks_per_seq)

    print("Running Reference (Full KV)...")
    out_ref, lse_ref = flash_attn_varlen_func(
        q=query, 
        k=global_key_cache, 
        v=global_value_cache,
        cu_seqlens_q=cu_query_lens, 
        seqused_k=kv_lens_t,
        max_seqlen_q=max(query_lens), 
        max_seqlen_k=max(kv_lens),
        softmax_scale=scale, 
        causal=False, 
        block_table=global_block_table,
        fa_version=fa_version, 
        return_softmax_lse=True,
    )

    world_size = 2
    rank_outs = []
    rank_lses = []

    blocks_per_rank = num_blocks // world_size
    print(f"Distributed split: {world_size} ranks, {blocks_per_rank} blocks per rank.")

    for rank in range(world_size):
        start_idx = rank * blocks_per_rank
        end_idx = (rank + 1) * blocks_per_rank
        
        local_k_cache = global_key_cache[start_idx:end_idx].clone().contiguous()
        local_v_cache = global_value_cache[start_idx:end_idx].clone().contiguous()
        
        local_block_table = torch.arange(blocks_per_rank, dtype=torch.int32).view(num_seqs, -1)
        local_kv_lens = torch.full((num_seqs,), blocks_per_rank * block_size, dtype=torch.int32)

        print(f" - Rank {rank}: k_cache_shape={local_k_cache.shape}, first_block_id={local_block_table[0,0].item()}")

        out_local, lse_local = flash_attn_varlen_func(
            q=query,
            k=local_k_cache,
            v=local_v_cache,
            cu_seqlens_q=cu_query_lens,
            seqused_k=local_kv_lens,
            max_seqlen_q=max(query_lens),
            max_seqlen_k=local_kv_lens.max().item(),
            softmax_scale=scale,
            causal=False,
            block_table=local_block_table,
            fa_version=fa_version,
            return_softmax_lse=True,
        )
        rank_outs.append(out_local)
        rank_lses.append(lse_local)

    lse_stack = torch.stack(rank_lses, dim=0).permute(0, 2, 1).unsqueeze(-1)
    out_stack = torch.stack(rank_outs, dim=0)

    lse_max = torch.max(lse_stack, dim=0)[0]
    exp_lse = torch.exp(lse_stack - lse_max)
    
    weighted_out_sum = torch.sum(out_stack * exp_lse, dim=0)
    sum_exp = torch.sum(exp_lse, dim=0)
    
    out_dist = (weighted_out_sum / (sum_exp + 1e-10)).to(dtype)
    print(out_ref)
    print(out_dist)

    max_diff = (out_ref - out_dist).abs().max().item()
    print(f"\nMax Difference: {max_diff:.6e}")
    
    torch.testing.assert_close(out_ref, out_dist, atol=1e-3, rtol=1e-3)
    cal_diff(out_ref, out_dist, "attn output")
    print("GQA distribute kv cache successful!")

if __name__ == "__main__":
    test_dycp_flashattn(QUICK_CASES[0])
