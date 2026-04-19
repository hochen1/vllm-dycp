"""Measure 8K input baseline: Direct vs PD no-CP vs PD with-CP."""
import requests
import time

VLLM = "http://localhost:8400"
PROXY = "http://localhost:9000"

# ~8K tokens
prompt = "人工智能是一种模拟人类智能的技术，它通过计算机程序和算法来实现各种智能功能。" * 400

print("=== 8K Input Baseline ===")

# 1. Direct (no PD, uses current threshold config)
t0 = time.monotonic()
r = requests.post(f"{VLLM}/v1/chat/completions", json={
    "model": "auto", "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 5, "stream": False, "temperature": 0,
}, timeout=120)
t1 = time.monotonic()
d = r.json()
if "error" in d:
    print(f"Direct: ERROR - {d['error']['message'][:100]}")
else:
    print(f"Direct: {(t1-t0)*1000:.0f}ms, prompt_tokens={d['usage']['prompt_tokens']}")

# 2. PD via proxy
t0 = time.monotonic()
r = requests.post(f"{PROXY}/v1/chat/completions", json={
    "model": "auto", "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 5, "stream": False, "temperature": 0,
}, timeout=120)
t1 = time.monotonic()
d = r.json()
if "error" in d:
    print(f"PD:     ERROR - {d['error']['message'][:100]}")
else:
    print(f"PD:     {(t1-t0)*1000:.0f}ms, prompt_tokens={d['usage']['prompt_tokens']}")

# 3. Prefill only (measure pure prefill time)
t0 = time.monotonic()
r = requests.post(f"{VLLM}/v1/chat/completions", json={
    "model": "auto", "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 1, "stream": False, "temperature": 0,
}, timeout=120)
t1 = time.monotonic()
d = r.json()
if "error" in d:
    print(f"Prefill only: ERROR - {d['error']['message'][:100]}")
else:
    print(f"Prefill only: {(t1-t0)*1000:.0f}ms, prompt_tokens={d['usage']['prompt_tokens']}")
