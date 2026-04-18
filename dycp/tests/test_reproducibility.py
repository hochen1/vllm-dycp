"""Test output reproducibility between direct and PD-separated requests."""
import requests

VLLM = "http://localhost:8400"
PROXY = "http://localhost:9000"

prompt = " ".join(["The quick brown fox jumps over the lazy dog."] * 20)
params = {"model": "auto", "messages": [{"role": "user", "content": prompt}],
          "max_tokens": 30, "stream": False, "temperature": 0, "seed": 42}

# Direct (no PD) x2
r1 = requests.post(f"{VLLM}/v1/chat/completions", json=params, timeout=60)
c1 = r1.json()["choices"][0]["message"]["content"]

r2 = requests.post(f"{VLLM}/v1/chat/completions", json=params, timeout=60)
c2 = r2.json()["choices"][0]["message"]["content"]

# Via proxy (PD separation)
r3 = requests.post(f"{PROXY}/v1/chat/completions", json=params, timeout=120)
d3 = r3.json()
c3 = d3["choices"][0]["message"]["content"] if "choices" in d3 else f"ERROR: {d3}"

print(f"Direct #1: {c1[:80]!r}")
print(f"Direct #2: {c2[:80]!r}")
print(f"PD proxy:  {c3[:80]!r}")
print(f"Direct reproducible: {c1 == c2}")
print(f"PD matches direct: {c1 == c3}")
