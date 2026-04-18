"""Compare DyCP direct vs PD separation with Chinese prompt."""
import requests
import json

VLLM = "http://localhost:8400"
PROXY = "http://localhost:9000"

prompt = "请详细解释什么是人工智能，它的发展历史是怎样的，目前有哪些主要的应用领域，未来的发展趋势是什么？请从技术原理、社会影响、伦理问题三个角度来分析。"

params = {
    "model": "auto",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 50,
    "stream": False,
    "temperature": 0,
    "seed": 42,
}

print("=" * 60)
print("INPUT (same for both)")
print("=" * 60)
print(f"prompt: {prompt}")
print(f"max_tokens: 50, temperature: 0, seed: 42")

# 1. Direct (DyCP, no PD separation)
print()
print("=" * 60)
print("OUTPUT 1: Direct DyCP (no PD separation)")
print("=" * 60)
r1 = requests.post(f"{VLLM}/v1/chat/completions", json=params, timeout=60)
d1 = r1.json()
c1 = d1["choices"][0]["message"]["content"]
print(f"prompt_tokens: {d1['usage']['prompt_tokens']}")
print(f"completion_tokens: {d1['usage']['completion_tokens']}")
print(f"content: {c1}")

# 2. Via Proxy (PD separation: prefill full CP -> decode CP=1)
print()
print("=" * 60)
print("OUTPUT 2: PD Separation (via Proxy)")
print("=" * 60)
r2 = requests.post(f"{PROXY}/v1/chat/completions", json=params, timeout=120)
d2 = r2.json()
if "error" in d2:
    print(f"ERROR: {d2['error']}")
    c2 = ""
else:
    c2 = d2["choices"][0]["message"]["content"]
    print(f"prompt_tokens: {d2['usage']['prompt_tokens']}")
    print(f"completion_tokens: {d2['usage']['completion_tokens']}")
    print(f"content: {c2}")

print()
print("=" * 60)
print(f"MATCH: {c1 == c2}")
print("=" * 60)
