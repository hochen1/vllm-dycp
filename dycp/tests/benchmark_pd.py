"""
Benchmark: PD Separation with CP vs without CP.

Both use Proxy + vLLM (PD separation architecture).
- Config A: VLLM_LONG_REQUEST_THRESHOLD=100 → long prompts use full CP
- Config B: VLLM_LONG_REQUEST_THRESHOLD=999999 → all prompts use single CP (no CP effect)

Compares: TTFT, decode throughput, total throughput.
"""
import argparse
import json
import time
import statistics
import requests
import concurrent.futures

PROXY_URL = "http://localhost:9000"


def generate_prompts(n, input_len_tokens_approx):
    """Generate n prompts of approximately input_len_tokens_approx tokens."""
    # Each Chinese char ≈ 1-2 tokens. Use repeated sentence.
    base = "人工智能是一种模拟人类智能的技术，它通过计算机程序和算法来实现。"
    # base is ~30 tokens
    repeats = max(1, input_len_tokens_approx // 30)
    prompt = base * repeats
    return [prompt] * n


def send_request(prompt, max_tokens, request_id):
    """Send one request through proxy and measure timing."""
    start = time.monotonic()
    try:
        r = requests.post(
            f"{PROXY_URL}/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": False,
                "temperature": 0,
            },
            timeout=300,
        )
        end = time.monotonic()
        d = r.json()
        if "error" in d:
            return {"error": str(d["error"]), "latency": end - start}
        return {
            "latency": end - start,
            "prompt_tokens": d["usage"]["prompt_tokens"],
            "completion_tokens": d["usage"]["completion_tokens"],
            "total_tokens": d["usage"]["total_tokens"],
        }
    except Exception as e:
        return {"error": str(e), "latency": time.monotonic() - start}


def run_benchmark(num_prompts, input_len, output_len, concurrency, request_rate):
    """Run benchmark and return metrics."""
    prompts = generate_prompts(num_prompts, input_len)
    results = []
    errors = 0

    start_time = time.monotonic()

    if concurrency == 1:
        # Sequential with rate limiting
        for i, prompt in enumerate(prompts):
            result = send_request(prompt, output_len, i)
            results.append(result)
            if "error" in result:
                errors += 1
            if request_rate > 0 and i < len(prompts) - 1:
                time.sleep(1.0 / request_rate)
    else:
        # Concurrent
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = []
            for i, prompt in enumerate(prompts):
                f = ex.submit(send_request, prompt, output_len, i)
                futures.append(f)
                if request_rate > 0 and i < len(prompts) - 1:
                    time.sleep(1.0 / request_rate)
            for f in concurrent.futures.as_completed(futures):
                result = f.result()
                results.append(result)
                if "error" in result:
                    errors += 1

    total_time = time.monotonic() - start_time

    # Compute metrics
    successful = [r for r in results if "error" not in r]
    if not successful:
        return {"error": f"All {num_prompts} requests failed", "errors": errors}

    latencies = [r["latency"] for r in successful]
    prompt_toks = sum(r["prompt_tokens"] for r in successful)
    comp_toks = sum(r["completion_tokens"] for r in successful)

    return {
        "num_prompts": num_prompts,
        "successful": len(successful),
        "errors": errors,
        "total_time_s": round(total_time, 2),
        "avg_latency_ms": round(statistics.mean(latencies) * 1000, 1),
        "p50_latency_ms": round(statistics.median(latencies) * 1000, 1),
        "p99_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.99)] * 1000, 1),
        "prompt_throughput_tok_s": round(prompt_toks / total_time, 1),
        "decode_throughput_tok_s": round(comp_toks / total_time, 1),
        "total_throughput_tok_s": round((prompt_toks + comp_toks) / total_time, 1),
        "avg_prompt_tokens": round(prompt_toks / len(successful), 1),
        "avg_completion_tokens": round(comp_toks / len(successful), 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--input-len", type=int, default=200)
    parser.add_argument("--output-len", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-rate", type=float, default=2.0)
    parser.add_argument("--label", type=str, default="benchmark")
    args = parser.parse_args()

    print(f"=== {args.label} ===")
    print(f"Config: {args.num_prompts} prompts, ~{args.input_len} input tokens, "
          f"{args.output_len} output tokens, concurrency={args.concurrency}, "
          f"rate={args.request_rate}")
    print()

    # Warmup
    print("Warmup...")
    send_request("Hello", 5, -1)

    print("Running benchmark...")
    metrics = run_benchmark(
        args.num_prompts, args.input_len, args.output_len,
        args.concurrency, args.request_rate,
    )

    print()
    for k, v in metrics.items():
        print(f"  {k}: {v}")
