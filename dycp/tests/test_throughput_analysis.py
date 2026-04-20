"""
Comprehensive throughput analysis: why CP loses 12.9% throughput at c=32.
Collects real data to support each hypothesis.
"""
import requests
import time
import statistics
import concurrent.futures

VLLM = "http://localhost:8400"
prompt_8k = "人工智能是一种模拟人类智能的技术，它通过计算机程序和算法来实现各种智能功能。" * 400
prompt_short = "你好"

def warmup():
    for _ in range(3):
        requests.post(f"{VLLM}/v1/chat/completions", json={
            "model": "auto", "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1, "stream": False}, timeout=30)

def burst(prompt, max_tokens, n, concurrency, rate=0):
    """Send n requests with given concurrency and rate, return per-request results."""
    results = []
    t_start = time.monotonic()
    def send(i):
        t0 = time.monotonic()
        r = requests.post(f"{VLLM}/v1/chat/completions", json={
            "model": "auto", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": False, "temperature": 0,
        }, timeout=300)
        d = r.json()
        if "error" in d:
            return {"error": True, "latency": (time.monotonic()-t0)*1000}
        return {
            "latency": (time.monotonic()-t0)*1000,
            "ptok": d["usage"]["prompt_tokens"],
            "ctok": d["usage"]["completion_tokens"],
        }
    with concurrent.futures.ThreadPoolExecutor(concurrency) as ex:
        futs = []
        for i in range(n):
            futs.append(ex.submit(send, i))
            if rate > 0 and i < n-1:
                time.sleep(1.0/rate)
        results = [f.result() for f in futs]
    total_time = time.monotonic() - t_start
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {"n": n, "ok": 0, "err": len(results)}
    ptok = sum(r["ptok"] for r in ok)
    ctok = sum(r["ctok"] for r in ok)
    lats = [r["latency"] for r in ok]
    return {
        "n": n, "ok": len(ok), "err": len(results)-len(ok),
        "total_s": total_time,
        "tok_s": (ptok+ctok)/total_time,
        "ptok_s": ptok/total_time,
        "ctok_s": ctok/total_time,
        "avg_lat": statistics.mean(lats),
        "p50_lat": statistics.median(lats),
        "p99_lat": sorted(lats)[int(len(lats)*0.99)],
        "avg_ptok": ptok/len(ok),
    }

warmup()
SEP = "=" * 70

####################################################################
print(SEP)
print("EXPERIMENT 1: Scheduler Step Overhead")
print("Measure time per scheduler step with different batch compositions")
print(SEP)

# 1a: Pure decode (8 short requests, 100 tokens each)
print("\n1a. Pure decode (8 short × 100 tokens):")
r = burst(prompt_short, 100, 8, 8)
print(f"    {r['total_s']:.2f}s, {r['tok_s']:.0f} tok/s, avg_lat={r['avg_lat']:.0f}ms")
decode_only_time = r['total_s']

# 1b. Single 8K prefill only (1 request, max_tokens=1)
print("1b. Single 8K prefill (1 req, max_tokens=1):")
t0 = time.monotonic()
requests.post(f"{VLLM}/v1/chat/completions", json={
    "model": "auto", "messages": [{"role": "user", "content": prompt_8k}],
    "max_tokens": 1, "stream": False, "temperature": 0,
}, timeout=120)
prefill_time = (time.monotonic()-t0)*1000
print(f"    {prefill_time:.0f}ms")

# 1c. 8 sequential 8K prefills (serial, measures per-prefill time)
print("1c. 8 sequential 8K prefills:")
times = []
for i in range(8):
    t0 = time.monotonic()
    requests.post(f"{VLLM}/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": prompt_8k}],
        "max_tokens": 1, "stream": False, "temperature": 0,
    }, timeout=120)
    times.append((time.monotonic()-t0)*1000)
print(f"    Per-prefill: {statistics.median(times):.0f}ms, total: {sum(times):.0f}ms")

####################################################################
print(f"\n{SEP}")
print("EXPERIMENT 2: MoE All-to-All Sync Interference")
print("How much does a heavy prefill slow down light decode on other ranks?")
print(SEP)

# 2a: 8 pure short (decode baseline)
print("\n2a. 8 pure short (decode baseline):")
r_pure = burst(prompt_short, 200, 8, 8)
print(f"    {r_pure['total_s']:.2f}s, avg_lat={r_pure['avg_lat']:.0f}ms, {r_pure['ctok_s']:.0f} decode_tok/s")

# 2b: 1 long + 7 short (interference test)
print("2b. 1 long 8K + 7 short (interference):")
t_start = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(8) as ex:
    futs = []
    futs.append(ex.submit(lambda: requests.post(f"{VLLM}/v1/chat/completions", json={
        "model":"auto","messages":[{"role":"user","content":prompt_8k}],
        "max_tokens":200,"stream":False,"temperature":0}, timeout=300).json()))
    for _ in range(7):
        futs.append(ex.submit(lambda: requests.post(f"{VLLM}/v1/chat/completions", json={
            "model":"auto","messages":[{"role":"user","content":prompt_short}],
            "max_tokens":200,"stream":False,"temperature":0}, timeout=300).json()))
    results_mix = [f.result() for f in futs]
t_mix = time.monotonic() - t_start
short_lats = [r["usage"]["completion_tokens"] for r in results_mix if r["usage"]["prompt_tokens"] < 100]
long_lat = [r for r in results_mix if r["usage"]["prompt_tokens"] > 100]
ctok_short = sum(r["usage"]["completion_tokens"] for r in results_mix if r["usage"]["prompt_tokens"] < 100)
ctok_long = sum(r["usage"]["completion_tokens"] for r in results_mix if r["usage"]["prompt_tokens"] > 100)
print(f"    Total: {t_mix:.2f}s")
print(f"    Short decode tokens: {ctok_short}, Long decode tokens: {ctok_long}")
print(f"    Decode tok/s (short only): {ctok_short/t_mix:.0f}")
print(f"    Interference factor: {r_pure['total_s']/t_mix:.2f}x slowdown")

# 2c: 4 long + 4 short
print("2c. 4 long 8K + 4 short:")
t_start = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(8) as ex:
    futs = []
    for _ in range(4):
        futs.append(ex.submit(lambda: requests.post(f"{VLLM}/v1/chat/completions", json={
            "model":"auto","messages":[{"role":"user","content":prompt_8k}],
            "max_tokens":200,"stream":False,"temperature":0}, timeout=300).json()))
    for _ in range(4):
        futs.append(ex.submit(lambda: requests.post(f"{VLLM}/v1/chat/completions", json={
            "model":"auto","messages":[{"role":"user","content":prompt_short}],
            "max_tokens":200,"stream":False,"temperature":0}, timeout=300).json()))
    results_44 = [f.result() for f in futs]
t_44 = time.monotonic() - t_start
ctok_44 = sum(r["usage"]["total_tokens"] for r in results_44)
print(f"    Total: {t_44:.2f}s, {ctok_44/t_44:.0f} tok/s")

####################################################################
print(f"\n{SEP}")
print("EXPERIMENT 3: Batch Separation Gap Measurement")
print("How much time is lost in prefill↔decode batch transitions?")
print(SEP)

# Run 10 8K requests as fast as possible, measure total time
# Each request does prefill→decode. With batch sep, transitions happen.
print("\n3a. 10 serial 8K requests (prefill+200 decode each):")
times_serial = []
for _ in range(10):
    t0 = time.monotonic()
    requests.post(f"{VLLM}/v1/chat/completions", json={
        "model":"auto","messages":[{"role":"user","content":prompt_8k}],
        "max_tokens":200,"stream":False,"temperature":0}, timeout=300)
    times_serial.append((time.monotonic()-t0)*1000)
serial_median = statistics.median(times_serial)
print(f"    Per-request: {serial_median:.0f}ms")

# 3b. 10 concurrent 8K (measures batch separation impact)
print("3b. 10 concurrent 8K requests:")
r_10 = burst(prompt_8k, 200, 10, 10, rate=10)
print(f"    Total: {r_10['total_s']:.2f}s, {r_10['tok_s']:.0f} tok/s")
print(f"    Avg lat: {r_10['avg_lat']:.0f}ms, P50: {r_10['p50_lat']:.0f}ms")

####################################################################
print(f"\n{SEP}")
print("EXPERIMENT 4: CP All-Gather Overhead at Scale")
print("Isolate all-gather cost from batch separation cost")
print(SEP)

# 4a. Prefill throughput: how many 8K prefills per second?
print("\n4a. 10 concurrent 8K prefills (max_tokens=1, prefill only):")
r_pf = burst(prompt_8k, 1, 10, 10, rate=10)
print(f"    Total: {r_pf['total_s']:.2f}s, prefill_tok/s={r_pf['ptok_s']:.0f}")
print(f"    Avg lat: {r_pf['avg_lat']:.0f}ms")

# 4b. Pure decode throughput: 10 short requests, 200 tokens
print("4b. 10 concurrent pure decode (short × 200):")
r_dc = burst(prompt_short, 200, 10, 10, rate=10)
print(f"    Total: {r_dc['total_s']:.2f}s, decode_tok/s={r_dc['ctok_s']:.0f}")
print(f"    Avg lat: {r_dc['avg_lat']:.0f}ms")

####################################################################
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)
print()
print("Key measurements (current config: CP enabled, threshold=100):")
print(f"  Single 8K prefill time:    {prefill_time:.0f}ms")
print(f"  Pure decode throughput:     {r_pure['ctok_s']:.0f} tok/s")
print(f"  Mixed interference factor:  {t_mix/r_pure['total_s']:.2f}x")
print(f"  10-concurrent throughput:   {r_10['tok_s']:.0f} tok/s")
