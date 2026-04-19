"""
Final comprehensive test: accuracy + performance comparison.

Compares:
  - Direct DyCP (no PD separation, baseline)
  - PD Separation via Proxy (optimized: GPU buffer + batch AG + conn pool)

Tests accuracy on both short and long prompts (Chinese),
then measures single-request latency and throughput benchmark.
"""
import json
import requests
import statistics
import time
import concurrent.futures

VLLM = "http://localhost:8400"
PROXY = "http://localhost:9000"

SEPARATOR = "=" * 70


def warmup():
    for _ in range(3):
        requests.post(f"{PROXY}/v1/chat/completions", json={
            "model": "auto", "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1, "stream": False,
        }, timeout=30)
    for _ in range(3):
        requests.post(f"{VLLM}/v1/chat/completions", json={
            "model": "auto", "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1, "stream": False,
        }, timeout=30)


# ==============================
# Part 1: Accuracy Tests
# ==============================

def test_accuracy():
    print(SEPARATOR)
    print("PART 1: ACCURACY")
    print(SEPARATOR)

    cases = [
        ("Short (CN, ~14 tok)", "请简要介绍人工智能。"),
        ("Short (EN, ~12 tok)", "What is artificial intelligence?"),
        ("Long (CN, ~120 tok)", "人工智能是一种模拟人类智能的技术，它通过计算机程序和算法来实现。" * 7),
        ("Long (EN, ~200 tok)", " ".join(["The quick brown fox jumps over the lazy dog."] * 20)),
    ]

    results = []
    for label, prompt in cases:
        params = {
            "model": "auto",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "stream": False,
            "temperature": 0,
            "seed": 42,
        }

        r1 = requests.post(f"{VLLM}/v1/chat/completions", json=params, timeout=60)
        d1 = r1.json()
        c1 = d1["choices"][0]["message"]["content"]
        t1 = d1["usage"]["prompt_tokens"]

        r2 = requests.post(f"{PROXY}/v1/chat/completions", json=params, timeout=120)
        d2 = r2.json()
        if "error" in d2:
            c2 = f"ERROR: {d2['error']}"
            match = False
        else:
            c2 = d2["choices"][0]["message"]["content"]
            match = c1 == c2

        status = "MATCH" if match else "DIFF"
        results.append((label, t1, status))

        print(f"\n{label} ({t1} tokens):")
        print(f"  Direct: {c1[:70]}{'...' if len(c1) > 70 else ''}")
        print(f"  PD:     {c2[:70]}{'...' if len(c2) > 70 else ''}")
        print(f"  Status: {status}")

    print(f"\n{'Label':<25} {'Tokens':<10} {'Result':<10}")
    print("-" * 45)
    for label, tokens, status in results:
        print(f"{label:<25} {tokens:<10} {status:<10}")

    return results


# ==============================
# Part 2: Single-Request Latency
# ==============================

def test_latency():
    print(f"\n{SEPARATOR}")
    print("PART 2: SINGLE-REQUEST LATENCY (5 runs each)")
    print(SEPARATOR)

    prompts = {
        "Short (~15 tok)": "请简要介绍人工智能。",
        "Medium (~120 tok)": "人工智能是一种模拟人类智能的技术，它通过计算机程序和算法来实现。" * 7,
        "Long (~400 tok)": "人工智能是一种模拟人类智能的技术，它通过计算机程序和算法来实现。" * 25,
    }

    print(f"\n{'Prompt':<20} {'Direct(ms)':<15} {'PD(ms)':<15} {'Overhead':<15}")
    print("-" * 65)

    latency_results = []
    for label, prompt in prompts.items():
        direct_t, pd_t = [], []
        for _ in range(5):
            t0 = time.monotonic()
            requests.post(f"{VLLM}/v1/chat/completions", json={
                "model": "auto", "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 20, "stream": False, "temperature": 0,
            }, timeout=30)
            direct_t.append((time.monotonic() - t0) * 1000)

            t0 = time.monotonic()
            requests.post(f"{PROXY}/v1/chat/completions", json={
                "model": "auto", "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 20, "stream": False, "temperature": 0,
            }, timeout=60)
            pd_t.append((time.monotonic() - t0) * 1000)

        d = statistics.median(direct_t)
        p = statistics.median(pd_t)
        overhead = p - d
        pct = overhead / d * 100
        print(f"{label:<20} {d:<15.0f} {p:<15.0f} {overhead:>+.0f}ms ({pct:>+.1f}%)")
        latency_results.append((label, d, p, overhead, pct))

    return latency_results


# ==============================
# Part 3: Throughput Benchmark
# ==============================

def run_throughput(url, num_prompts, prompt, max_tokens, concurrency, rate):
    results = []
    start = time.monotonic()

    def send(i):
        t0 = time.monotonic()
        try:
            r = requests.post(f"{url}/v1/chat/completions", json={
                "model": "auto",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "stream": False, "temperature": 0,
            }, timeout=300)
            d = r.json()
            if "error" in d:
                return {"error": True, "latency": (time.monotonic() - t0) * 1000}
            return {
                "latency": (time.monotonic() - t0) * 1000,
                "prompt_tokens": d["usage"]["prompt_tokens"],
                "completion_tokens": d["usage"]["completion_tokens"],
            }
        except Exception as e:
            return {"error": True, "latency": (time.monotonic() - t0) * 1000}

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = []
        for i in range(num_prompts):
            futures.append(ex.submit(send, i))
            if rate > 0 and i < num_prompts - 1:
                time.sleep(1.0 / rate)
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    total_time = time.monotonic() - start
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {"errors": len(results), "successful": 0}

    latencies = [r["latency"] for r in ok]
    prompt_toks = sum(r["prompt_tokens"] for r in ok)
    comp_toks = sum(r["completion_tokens"] for r in ok)
    return {
        "successful": len(ok),
        "errors": len(results) - len(ok),
        "total_time_s": round(total_time, 2),
        "avg_latency_ms": round(statistics.mean(latencies), 0),
        "p50_latency_ms": round(statistics.median(latencies), 0),
        "prompt_tok_s": round(prompt_toks / total_time, 1),
        "decode_tok_s": round(comp_toks / total_time, 1),
        "total_tok_s": round((prompt_toks + comp_toks) / total_time, 1),
    }


def test_throughput():
    print(f"\n{SEPARATOR}")
    print("PART 3: THROUGHPUT BENCHMARK")
    print(SEPARATOR)

    prompt_medium = "人工智能是一种模拟人类智能的技术，它通过计算机程序和算法来实现。" * 7

    configs = [
        ("c=2, ~120 tok", 10, prompt_medium, 50, 2, 1),
        ("c=4, ~120 tok", 10, prompt_medium, 50, 4, 2),
    ]

    print(f"\n{'Config':<20} {'Mode':<10} {'OK/Err':<10} {'AvgLat(ms)':<12} {'PromptT/s':<12} {'DecodeT/s':<12} {'TotalT/s'}")
    print("-" * 96)

    for label, n, prompt, max_tok, conc, rate in configs:
        # Direct
        m1 = run_throughput(VLLM, n, prompt, max_tok, conc, rate)
        # PD
        m2 = run_throughput(PROXY, n, prompt, max_tok, conc, rate)

        for mode, m in [("Direct", m1), ("PD", m2)]:
            if "avg_latency_ms" in m:
                print(f"{label:<20} {mode:<10} {m['successful']}/{m['errors']:<7} "
                      f"{m['avg_latency_ms']:<12.0f} {m['prompt_tok_s']:<12.1f} "
                      f"{m['decode_tok_s']:<12.1f} {m['total_tok_s']}")
            else:
                print(f"{label:<20} {mode:<10} 0/{m['errors']:<7} {'FAILED':<12}")


if __name__ == "__main__":
    print("Warming up...")
    warmup()

    test_accuracy()
    test_latency()
    test_throughput()

    print(f"\n{SEPARATOR}")
    print("TEST COMPLETE")
    print(SEPARATOR)
