import math
import torch
from typing import Any

from vllm.attention.ops.flashmla import (
    flash_mla_with_kvcache,
    get_mla_metadata,
    is_flashmla_dense_supported,
)
from vllm.triton_utils import triton

def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> None:

    x_flat = x.to(torch.float64).flatten()
    y_flat = y.to(torch.float64).flatten()
    
    cos_sim = (x_flat @ y_flat) / (x_flat.norm() * y_flat.norm()).clamp_min(1e-12)
    cos_diff = 1 - cos_sim.item()
    
    print(f"[{name}] Cosine Difference: {cos_diff:.6e}")
    assert cos_diff < 1e-4, f"{name} difference too large! Check implementation."

def test_dycp_flashmla():
    supported, reason = is_flashmla_dense_supported()
    if not supported:
        print(f"Skipping: {reason}")
        return

    b = 1
    s_q = 1
    mean_sk = 2048
    h_q = 32
    h_kv = 1
    d = 576
    dv = 512
    block_size = 64
    dtype = torch.bfloat16
    device = torch.device("cuda:0")

    torch.set_default_device(device)
    torch.manual_seed(42)

    # global
    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
    num_total_blocks = b * max_seqlen_pad // block_size

    q = torch.randn(b, s_q, h_q, d, dtype=dtype)
    global_blocked_k = torch.randn(num_total_blocks, block_size, h_kv, d, dtype=dtype)
    global_block_table = torch.arange(num_total_blocks, dtype=torch.int32).view(b, -1)

    tile_metadata_ref, num_splits_ref = get_mla_metadata(
        cache_seqlens, s_q * h_q // h_kv, h_kv
    )
    
    out_ref, lse_ref = flash_mla_with_kvcache(
        q, global_blocked_k, global_block_table, cache_seqlens, dv,
        tile_metadata_ref, num_splits_ref, causal=False
    )

    # split kv cache
    world_size = 2
    rank_outs = []
    rank_lses = []

    blocks_per_seq_full = max_seqlen_pad // block_size
    blocks_per_seq_local = blocks_per_seq_full // world_size
    
    print(f"Total blocks per seq: {blocks_per_seq_full}, Rank Local: {blocks_per_seq_local}")

    for rank in range(world_size):
        start_col = rank * blocks_per_seq_local
        end_col = (rank + 1) * blocks_per_seq_local
        
        rank_block_indices = global_block_table[:, start_col:end_col].contiguous()
        
        local_blocked_k = global_blocked_k[rank_block_indices.view(-1)].clone()
        
        local_block_table = torch.arange(
            b * blocks_per_seq_local, dtype=torch.int32
        ).view(b, blocks_per_seq_local)
        
        local_cache_seqlens = torch.full((b,), blocks_per_seq_local * block_size, dtype=torch.int32)
        local_tile_metadata, local_num_splits = get_mla_metadata(
            local_cache_seqlens, s_q * h_q // h_kv, h_kv
        )

        out_local, lse_local = flash_mla_with_kvcache(
            q, local_blocked_k, local_block_table, local_cache_seqlens, dv,
            local_tile_metadata, local_num_splits, causal=False
        )
        
        rank_outs.append(out_local)
        rank_lses.append(lse_local)

    # lse_stack: [Rank, B, H, S_Q] -> [Rank, B, S_Q, H, 1]
    lse_stack = torch.stack(rank_lses, dim=0).permute(0, 1, 3, 2).unsqueeze(-1)
    out_stack = torch.stack(rank_outs, dim=0) # [Rank, B, S_Q, H, DV]

    lse_max = torch.max(lse_stack, dim=0)[0]
    exp_lse = torch.exp(lse_stack - lse_max)
    
    weighted_out_sum = torch.sum(out_stack * exp_lse, dim=0)
    sum_exp = torch.sum(exp_lse, dim=0)
    
    out_dist = (weighted_out_sum / (sum_exp + 1e-10)).to(dtype)

    cal_diff(out_ref, out_dist, name="mla attn output")


if __name__ == "__main__":
    test_dycp_flashmla()
