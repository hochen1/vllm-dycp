import math
import torch
import torch.nn.functional as F
from vllm.vllm_flash_attn import flash_attn_varlen_func

def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
    x_flat = x.to(torch.float64).flatten()
    y_flat = y.to(torch.float64).flatten()
    
    cos_sim = (x_flat @ y_flat) / (x_flat.norm() * y_flat.norm()).clamp_min(1e-12)
    cos_diff = 1 - cos_sim.item()
    
    print(f"[{name}] Cosine Difference: {cos_diff:.6e}")
    assert cos_diff < 1e-4, f"{name} difference too large! Check implementation."

def test_mla_flash_attn_distributed():
    b = 1   # batch size
    s_q = 1 # query sequence length
    s_k = 2048  # key sequence length
    h_q = 16
    d_pe = 64
    d_nope = 512
    block_size = 64
    dtype = torch.bfloat16
    device = "cuda"

    softmax_scale = 1.0 / math.sqrt(d_pe + d_nope)

    q_pe = torch.randn(b * s_q, h_q, d_pe, dtype=dtype, device=device)
    q_nope = torch.randn(b * s_q, h_q, d_nope, dtype=dtype, device=device)
    
    num_blocks = s_k // block_size
    k_pe_cache = torch.randn(num_blocks, block_size, 1, d_pe, dtype=dtype, device=device)
    kv_c_cache = torch.randn(num_blocks, block_size, 1, d_nope, dtype=dtype, device=device)
    
    cu_seqlens_q = torch.tensor([0, s_q], dtype=torch.int32, device=device)
    print(f"cu_seqlens_q:{cu_seqlens_q}")
    seqused_k = torch.tensor([s_k], dtype=torch.int32, device=device)
    print(f"seqused_k:{seqused_k}")
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(b, -1)
    print(f"block_table:{block_table}")

    out_ref, lse_ref = flash_attn_varlen_func(
        q=q_pe,
        k=k_pe_cache,
        v=kv_c_cache,
        q_v=q_nope,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=s_q,
        max_seqlen_k=s_k,
        block_table=block_table,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        causal=False,
        return_softmax_lse=True,
        fa_version=3,
    )

    world_size = 2
    rank_outs = []
    rank_lses = []

    blocks_per_rank = num_blocks // world_size

    for rank in range(world_size):
        start_b = rank * blocks_per_rank
        end_b = (rank + 1) * blocks_per_rank
        
        local_k_pe = k_pe_cache[start_b:end_b].clone().contiguous()
        local_kv_c = kv_c_cache[start_b:end_b].clone().contiguous()
        
        local_block_table = torch.arange(blocks_per_rank, dtype=torch.int32, device=device).view(b, -1)
        print(f"local_block_table:{local_block_table}")
        local_seqused_k = torch.tensor([blocks_per_rank * block_size], dtype=torch.int32, device=device)
        print(f"local_seqused_k:{local_seqused_k}")
        out_rank, lse_rank = flash_attn_varlen_func(
            q=q_pe,
            k=local_k_pe,
            v=local_kv_c,
            q_v=q_nope,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=s_q,
            max_seqlen_k=local_seqused_k.item(),
            block_table=local_block_table,
            seqused_k=local_seqused_k,
            softmax_scale=softmax_scale,
            causal=False,
            return_softmax_lse=True,
            fa_version=3,
        )
        rank_outs.append(out_rank)
        rank_lses.append(lse_rank)

    lse_stack = torch.stack(rank_lses, dim=0).permute(0, 2, 1).unsqueeze(-1)
    out_stack = torch.stack(rank_outs, dim=0) # [Rank, total_q, heads, d_nope]

    lse_max = torch.max(lse_stack, dim=0)[0]
    exp_lse = torch.exp(lse_stack - lse_max)
    
    weighted_out_sum = torch.sum(out_stack * exp_lse, dim=0)
    sum_exp = torch.sum(exp_lse, dim=0)
    
    out_dist = (weighted_out_sum / (sum_exp + 1e-10)).to(dtype)

    cal_diff(out_ref, out_dist, "flash attn mla output")

    print("MLA Distributed KV Cache Test Passed!")

if __name__ == "__main__":
    test_mla_flash_attn_distributed()
