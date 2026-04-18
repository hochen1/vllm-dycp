"""Compare KV norms between PD-saved and expected values."""
import os
import requests
import safetensors.torch
import time
import torch

VLLM = "http://localhost:8400"
prompt = " ".join(["The quick brown fox jumps over the lazy dog."] * 20)

# Prefill via PD path (saves KV with all-gather)
os.system("rm -rf /tmp/local_pd_kv/*")
r = requests.post(f"{VLLM}/v1/chat/completions", json={
    "model": "auto", "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 1, "stream": False,
    "kv_transfer_params": {"do_remote_prefill": False, "do_remote_decode": True,
                           "pd_request_prefix": "pd-norm-check"}
}, timeout=120)
print(f"Prefill: {r.json().get('kv_transfer_params')}")
time.sleep(2)

# Check saved KV
layer = "model.layers.0.self_attn.attn.safetensors"
kv_path = f"/tmp/local_pd_kv/pd-norm-check/rank_0/{layer}"
if os.path.exists(kv_path):
    kv = safetensors.torch.load_file(kv_path)["kv_cache"]
    print(f"Saved KV shape: {kv.shape}")
    print(f"First 10 norms: {torch.norm(kv[:10], dim=-1).tolist()}")
    print(f"Last 10 norms:  {torch.norm(kv[-10:], dim=-1).tolist()}")

    # Check for zeros (missing KV)
    norms = torch.norm(kv, dim=-1)
    num_zero = (norms < 1e-6).sum().item()
    num_nonzero = kv.shape[0] - num_zero
    print(f"Zero-norm tokens: {num_zero} / {kv.shape[0]}")
    print(f"Non-zero tokens: {num_nonzero}")

    # Show which positions have valid data
    valid_positions = (norms > 1e-6).nonzero(as_tuple=True)[0]
    print(f"Valid positions: {valid_positions.tolist()}")
else:
    print(f"KV file not found: {kv_path}")
