"""Simple TTFT breakdown: just proxy vs direct, with timing instrumentation."""
import requests
import time
import statistics

VLLM = "http://localhost:8400"
PROXY = "http://localhost:9000"

prompt_8k = "人工智能是一种模拟人类智能的技术，它通过计算机程序和算法来实现各种智能功能。" * 400

for _ in range(2):
    requests.post(f"{PROXY}/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1, "stream": False}, timeout=30)

N = 5
direct_t, proxy_t, prefill_only_t = [], [], []

for i in range(N):
    # Direct (no PD)
    t0 = time.monotonic()
    requests.post(f"{VLLM}/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": prompt_8k}],
        "max_tokens": 1, "stream": False, "temperature": 0,
    }, timeout=180)
    direct_t.append((time.monotonic() - t0) * 1000)

    # Prefill only (to vLLM, no decode)
    t0 = time.monotonic()
    requests.post(f"{VLLM}/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": prompt_8k}],
        "max_tokens": 1, "stream": False, "temperature": 0,
        "kv_transfer_params": {
            "do_remote_prefill": False, "do_remote_decode": True,
            "pd_request_prefix": f"pd-simple-{i}",
        }
    }, timeout=180)
    prefill_only_t.append((time.monotonic() - t0) * 1000)

    # Full PD via proxy
    t0 = time.monotonic()
    requests.post(f"{PROXY}/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": prompt_8k}],
        "max_tokens": 1, "stream": False, "temperature": 0,
    }, timeout=180)
    proxy_t.append((time.monotonic() - t0) * 1000)

a = statistics.median(direct_t)
b = statistics.median(prefill_only_t)
d = statistics.median(proxy_t)

print(f"=== 8K TTFT Breakdown ({N} runs) ===")
print(f"A. Direct (baseline):   {a:.0f}ms")
print(f"B. Prefill-only:        {b:.0f}ms")
print(f"C. Proxy total:         {d:.0f}ms")
print()
print(f"--- Overhead Breakdown ---")
print(f"KV all-gather+save:     {b - a:>+.0f}ms  (B - A)")
print(f"Proxy+decode overhead:  {d - b:>+.0f}ms  (C - B = HTTP + KV inject + batch switch + 1 decode)")
print(f"Total PD overhead:      {d - a:>+.0f}ms  (C - A)")
