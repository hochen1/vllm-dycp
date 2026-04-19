"""Test 8K PD no-CP with detailed error handling."""
import requests

PROXY = "http://localhost:9000"
# ~8K tokens: each Chinese sentence ≈ 19 tokens
prompt = "人工智能是一种模拟人类智能的技术，它通过计算机程序和算法来实现各种智能功能。" * 400

print(f"Sending 8K request to proxy...")
try:
    r = requests.post(f"{PROXY}/v1/chat/completions", json={
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "stream": False,
        "temperature": 0,
    }, timeout=300)
    d = r.json()
    if "error" in d:
        print(f"ERROR: {d['error']}")
    else:
        pt = d["usage"]["prompt_tokens"]
        print(f"OK: prompt_tokens={pt}")
except Exception as e:
    print(f"EXCEPTION: {e}")
