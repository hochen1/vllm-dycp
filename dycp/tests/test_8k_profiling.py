"""Profile 8K input scenario: measure prefill time breakdown."""
import requests
import time
import statistics

VLLM = "http://localhost:8400"

# ~8K tokens
prompt = "人工智能是一种模拟人类智能的技术，它通过计算机程序和算法来实现各种智能功能。" * 400

print("=== 8K Input Profiling ===")

# Warmup
for _ in range(2):
    requests.post(f"{VLLM}/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1, "stream": False,
    }, timeout=30)

# Measure prefill-only (max_tokens=1) - 5 runs
prefill_times = []
for i in range(5):
    t0 = time.monotonic()
    r = requests.post(f"{VLLM}/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1, "stream": False, "temperature": 0,
    }, timeout=120)
    t1 = time.monotonic()
    d = r.json()
    tokens = d["usage"]["prompt_tokens"]
    prefill_times.append((t1 - t0) * 1000)
    if i == 0:
        print(f"Prompt tokens: {tokens}")

print(f"Prefill (5 runs): median={statistics.median(prefill_times):.0f}ms "
      f"min={min(prefill_times):.0f}ms max={max(prefill_times):.0f}ms")

# Measure full request (max_tokens=100) - 5 runs
full_times = []
for i in range(5):
    t0 = time.monotonic()
    r = requests.post(f"{VLLM}/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100, "stream": False, "temperature": 0,
    }, timeout=120)
    t1 = time.monotonic()
    full_times.append((t1 - t0) * 1000)

print(f"Full (prefill+100 decode): median={statistics.median(full_times):.0f}ms")
print(f"Decode portion (100 tok): ~{statistics.median(full_times) - statistics.median(prefill_times):.0f}ms")
print(f"Decode per token: ~{(statistics.median(full_times) - statistics.median(prefill_times)) / 100:.1f}ms/tok")

# Estimate: if CP could reduce prefill by 7/8
prefill_ms = statistics.median(prefill_times)
print(f"\n=== CP Benefit Estimate (8K, 8 ranks) ===")
print(f"Current prefill: {prefill_ms:.0f}ms (single rank)")
print(f"CP ideal prefill: {prefill_ms / 8:.0f}ms (8 ranks, no overhead)")
print(f"CP realistic prefill: {prefill_ms / 8 + 25:.0f}ms (8 ranks, +25ms overhead)")
print(f"Potential TTFT savings: {prefill_ms - prefill_ms / 8 - 25:.0f}ms")
