"""Precise TTFT breakdown: separate each overhead component."""
import requests
import time
import statistics

VLLM = "http://localhost:8400"
PROXY = "http://localhost:9000"

prompt_8k = "人工智能是一种模拟人类智能的技术，它通过计算机程序和算法来实现各种智能功能。" * 400

# Warmup
for _ in range(3):
    requests.post(f"{PROXY}/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1, "stream": False}, timeout=30)

N = 5
print(f"=== TTFT Breakdown ({N} runs) ===")
print()

# A. Direct (no PD): baseline TTFT
direct_t = []
for _ in range(N):
    t0 = time.monotonic()
    requests.post(f"{VLLM}/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": prompt_8k}],
        "max_tokens": 1, "stream": False, "temperature": 0,
    }, timeout=180)
    direct_t.append((time.monotonic() - t0) * 1000)

# B. Prefill-only (direct, with KV save)
prefill_t = []
for i in range(N):
    prefix = f"pd-ttft-b-{i}"
    t0 = time.monotonic()
    requests.post(f"{VLLM}/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": prompt_8k}],
        "max_tokens": 1, "stream": False, "temperature": 0,
        "kv_transfer_params": {
            "do_remote_prefill": False, "do_remote_decode": True,
            "pd_request_prefix": prefix,
        }
    }, timeout=180)
    prefill_t.append((time.monotonic() - t0) * 1000)

# C. Decode-only (direct, with KV load) — uses last prefill's prefix
decode_t = []
last_prefix = f"pd-ttft-b-{N-1}"
r = requests.post(f"{VLLM}/v1/chat/completions", json={
    "model": "auto", "messages": [{"role": "user", "content": prompt_8k}],
    "max_tokens": 1, "stream": False, "temperature": 0,
    "kv_transfer_params": {
        "do_remote_prefill": False, "do_remote_decode": True,
        "pd_request_prefix": "pd-ttft-c-fresh",
    }
}, timeout=180)
kv_p = r.json().get("kv_transfer_params", {})

for i in range(N):
    # Fresh prefill for each decode
    prefix = f"pd-ttft-c-{i}"
    requests.post(f"{VLLM}/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": prompt_8k}],
        "max_tokens": 1, "stream": False, "temperature": 0,
        "kv_transfer_params": {
            "do_remote_prefill": False, "do_remote_decode": True,
            "pd_request_prefix": prefix,
        }
    }, timeout=180)
    # Decode using this prefix
    t0 = time.monotonic()
    r2 = requests.post(f"{VLLM}/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": prompt_8k}],
        "max_tokens": 1, "stream": False, "temperature": 0,
        "kv_transfer_params": {
            "do_remote_prefill": True, "do_remote_decode": False,
            "pd_request_prefix": prefix,
            **kv_p,
        }
    }, timeout=180)
    d2 = r2.json()
    if "error" in d2:
        print(f"  Decode-only error: {d2['error']['message'][:80]}")
        decode_t.append(float('inf'))
    else:
        decode_t.append((time.monotonic() - t0) * 1000)

# D. Full PD via proxy
proxy_t = []
for _ in range(N):
    t0 = time.monotonic()
    requests.post(f"{PROXY}/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": prompt_8k}],
        "max_tokens": 1, "stream": False, "temperature": 0,
    }, timeout=180)
    proxy_t.append((time.monotonic() - t0) * 1000)

# Report
a = statistics.median(direct_t)
b = statistics.median(prefill_t)
valid_decode = [t for t in decode_t if t < 100000]
c = statistics.median(valid_decode) if valid_decode else float('nan')
d = statistics.median(proxy_t)

print(f"A. Direct (baseline):        {a:.0f}ms  (CP prefill + 1 decode tok)")
print(f"B. Prefill-only (w/ KV save):{b:.0f}ms  (CP prefill + KV all-gather + 1 tok)")
print(f"C. Decode-only (w/ KV load): {c:.0f}ms  (KV inject + scheduling + 1 tok)")
print(f"D. Proxy total (PD):         {d:.0f}ms  (B via proxy + C via proxy)")
print()
print(f"--- Fixed Overhead Breakdown ---")
print(f"KV save overhead (B - A):       {b - a:>+.0f}ms")
print(f"Serial B+C:                     {b + c:.0f}ms")
print(f"Proxy overhead (D - B - C):     {d - b - c:>+.0f}ms  (2x HTTP + proxy processing)")
print(f"Decode scheduling (C - decode_baseline): ~{c - 10:.0f}ms  (batch switch + KV inject)")
print(f"Total PD overhead (D - A):      {d - a:>+.0f}ms")
