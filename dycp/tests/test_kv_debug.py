"""Compare KV cache values between direct prefill and PD-separated prefill."""
import json
import os
import requests
import safetensors.torch
import time
import torch
import numpy as np

VLLM = "http://localhost:8400"
prompt = " ".join(["The quick brown fox jumps over the lazy dog."] * 20)

# Step 1: PD prefill (saves KV to files)
os.system("rm -rf /tmp/local_pd_kv/*")
r = requests.post(f"{VLLM}/v1/chat/completions", json={
    "model": "auto", "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 1, "stream": False,
    "kv_transfer_params": {"do_remote_prefill": False, "do_remote_decode": True,
                           "pd_request_prefix": "pd-kv-debug"}
}, timeout=120)
print(f"Prefill: {r.json().get('kv_transfer_params')}")
time.sleep(2)

# Check saved KV shapes
meta = json.load(open("/tmp/local_pd_kv/pd-kv-debug/meta.json"))
print(f"meta: {meta}")

layer = "model.layers.0.self_attn.attn.safetensors"
for rank in range(meta["cp_world_size"]):
    kv = safetensors.torch.load_file(
        f"/tmp/local_pd_kv/pd-kv-debug/rank_{rank}/{layer}")["kv_cache"]
    print(f"  rank_{rank}: shape={kv.shape}, first_5_norms={torch.norm(kv[:5], dim=-1).tolist()}")

# Compute expected tokens_per_rank
W = meta["cp_world_size"]
N = meta["num_prompt_tokens"]
padded = int(np.ceil(N / (2 * W)) * (2 * W))
tpr = padded // W
print(f"\nExpected: N={N}, padded={padded}, tpr={tpr}, W={W}")

# Compute restore index
cs = tpr // 2
all_pos = []
for rank in range(W):
    hs = rank * cs
    ts = (2*W-1-rank) * cs
    for i in range(cs): all_pos.append(hs + i)
    for i in range(cs): all_pos.append(ts + i)
restore = np.argsort(all_pos)

# Load all rank KV and apply restore
all_kv = []
for rank in range(W):
    kv = safetensors.torch.load_file(
        f"/tmp/local_pd_kv/pd-kv-debug/rank_{rank}/{layer}")["kv_cache"]
    all_kv.append(kv)

concat = torch.cat(all_kv, dim=0)
restore_t = torch.from_numpy(restore.copy()).clamp(0, concat.shape[0]-1)
restored = concat[restore_t][:N]

print(f"\nConcat shape: {concat.shape}")
print(f"Restored shape: {restored.shape}")
print(f"Restored first 5 norms: {torch.norm(restored[:5], dim=-1).tolist()}")

# Check: are rank KV shapes == tpr?
rank0_kv = all_kv[0]
print(f"\nRank 0 actual shape: {rank0_kv.shape} (expected tpr={tpr})")
if rank0_kv.shape[0] != tpr:
    print(f"MISMATCH! Rank saves {rank0_kv.shape[0]} tokens, expected {tpr}")
