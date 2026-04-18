"""Compare: no-CP baseline vs DyCP full-CP vs PD separation, with long Chinese prompt."""
import requests

VLLM = "http://localhost:8400"
PROXY = "http://localhost:9000"

# Long Chinese prompt >100 tokens to trigger full CP
prompt = (
    "请你扮演一个资深的人工智能专家，从以下几个方面详细介绍人工智能的发展："
    "第一，人工智能的定义和基本概念是什么？"
    "第二，从1950年代图灵测试到现在，人工智能经历了哪些重要的发展阶段？"
    "第三，深度学习、强化学习、自然语言处理等关键技术的原理是什么？"
    "第四，人工智能在医疗、金融、教育、交通等领域有哪些具体的应用案例？"
    "第五，未来十年人工智能的发展趋势和面临的主要挑战是什么？"
    "请尽量全面、深入地回答以上问题。"
)

common = {
    "model": "auto",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 100,
    "stream": False,
    "temperature": 0,
    "seed": 42,
}

print("=" * 70)
print("INPUT")
print("=" * 70)
print(f"prompt ({len(prompt)} chars): {prompt[:80]}...")
print(f"max_tokens=100, temperature=0, seed=42")

# 1. Direct DyCP (>100 tokens → full CP, 8 ranks)
print()
print("=" * 70)
print("OUTPUT 1: Direct DyCP (full CP, 8 ranks, no PD separation)")
print("=" * 70)
r1 = requests.post(f"{VLLM}/v1/chat/completions", json=common, timeout=60)
d1 = r1.json()
c1 = d1["choices"][0]["message"]["content"]
t1 = d1["usage"]["prompt_tokens"]
print(f"prompt_tokens: {t1}")
print(f"completion_tokens: {d1['usage']['completion_tokens']}")
print(f"content:\n{c1}")

# 2. PD Separation via Proxy (prefill full CP → decode CP=1)
print()
print("=" * 70)
print("OUTPUT 2: PD Separation (prefill full CP → decode CP=1)")
print("=" * 70)
r2 = requests.post(f"{PROXY}/v1/chat/completions", json=common, timeout=120)
d2 = r2.json()
if "error" in d2:
    print(f"ERROR: {d2['error']}")
    c2 = ""
else:
    c2 = d2["choices"][0]["message"]["content"]
    print(f"prompt_tokens: {d2['usage']['prompt_tokens']}")
    print(f"completion_tokens: {d2['usage']['completion_tokens']}")
    print(f"content:\n{c2}")

# 3. Short prompt baseline (single CP = no CP effect)
short_prompt = "请简要介绍人工智能。"
short_params = {**common, "messages": [{"role": "user", "content": short_prompt}]}
print()
print("=" * 70)
print("OUTPUT 3: Short prompt baseline (single CP = no DyCP effect)")
print("=" * 70)
r3 = requests.post(f"{VLLM}/v1/chat/completions", json=short_params, timeout=60)
d3 = r3.json()
c3 = d3["choices"][0]["message"]["content"]
print(f"prompt_tokens: {d3['usage']['prompt_tokens']}")
print(f"completion_tokens: {d3['usage']['completion_tokens']}")
print(f"content:\n{c3}")

print()
print("=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"Output 1 vs 2 match: {c1 == c2}")
print(f"Output 1 tokens: {t1} ({'full CP' if t1 > 100 else 'single CP'})")
