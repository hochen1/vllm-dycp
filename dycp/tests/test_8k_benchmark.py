"""8K input, 1K output benchmark: PD+CP vs PD no-CP vs Direct."""
import requests
import time
import statistics
import concurrent.futures

VLLM = "http://localhost:8400"
PROXY = "http://localhost:9000"

# ~8K tokens
PROMPT_8K = "人工智能是一种模拟人类智能的技术，它通过计算机程序和算法来实现各种智能功能。" * 400

def measure(url, prompt, max_tokens, n=3):
    times = []
    for _ in range(n):
        t0 = time.monotonic()
        r = requests.post(f"{url}/v1/chat/completions", json={
            "model": "auto", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": False, "temperature": 0,
        }, timeout=300)
        d = r.json()
        if "error" in d:
            return {"error": d["error"]["message"][:80]}
        times.append((time.monotonic() - t0) * 1000)
    return {
        "median": statistics.median(times),
        "min": min(times),
        "tokens": d["usage"]["prompt_tokens"],
        "comp": d["usage"]["completion_tokens"],
    }

def throughput_test(url, prompt, max_tokens, n, concurrency, rate):
    results = []
    t_start = time.monotonic()
    def send(i):
        t0 = time.monotonic()
        try:
            r = requests.post(f"{url}/v1/chat/completions", json={
                "model": "auto", "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "stream": False, "temperature": 0,
            }, timeout=300)
            d = r.json()
            if "error" in d:
                return {"error": True}
            return {"latency": (time.monotonic()-t0)*1000,
                    "ptok": d["usage"]["prompt_tokens"],
                    "ctok": d["usage"]["completion_tokens"]}
        except:
            return {"error": True}

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = []
        for i in range(n):
            futs.append(ex.submit(send, i))
            if rate > 0 and i < n-1:
                time.sleep(1.0/rate)
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())

    total = time.monotonic() - t_start
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {"errors": len(results)}
    ptok = sum(r["ptok"] for r in ok)
    ctok = sum(r["ctok"] for r in ok)
    return {
        "ok": len(ok), "err": len(results)-len(ok),
        "time": total,
        "avg_lat": statistics.mean([r["latency"] for r in ok]),
        "ptok_s": ptok/total, "ctok_s": ctok/total,
        "total_s": (ptok+ctok)/total,
    }

print("=== 8K Input Benchmark ===")
print()

# Single-request latency
print("--- Single-Request Latency (TTFT proxy) ---")

# Warmup
for _ in range(2):
    requests.post(f"{PROXY}/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1, "stream": False}, timeout=30)

# PD with CP (current config: threshold=100)
r_pd_cp = measure(PROXY, PROMPT_8K, 100, n=3)
print(f"PD+CP:    {r_pd_cp}")

# Direct (no PD)
r_direct = measure(VLLM, PROMPT_8K, 100, n=3)
print(f"Direct:   {r_direct}")

if "error" not in r_pd_cp and "error" not in r_direct:
    savings = r_direct["median"] - r_pd_cp["median"]
    print(f"TTFT savings: {savings:.0f}ms ({savings/r_direct['median']*100:.1f}%)")

# Throughput (low concurrency first)
print()
print("--- Throughput (c=2, 8K input, 100 output) ---")
t_pd = throughput_test(PROXY, PROMPT_8K, 100, 5, 2, 0.5)
t_direct = throughput_test(VLLM, PROMPT_8K, 100, 5, 2, 0.5)
print(f"PD+CP:  {t_pd}")
print(f"Direct: {t_direct}")
