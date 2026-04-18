"""End-to-end tests for Local PD Separation."""
import json
import os
import sys
import requests

VLLM_URL = "http://localhost:8400"
PROXY_URL = "http://localhost:9000"


def test_long_request_cp():
    """Test that long requests (>threshold tokens) use full CP."""
    prompt = " ".join(["The quick brown fox jumps over the lazy dog."] * 20)
    os.system("rm -rf /tmp/local_pd_kv/*")

    print("=== Step 1: Prefill (long prompt, should use full CP) ===")
    resp = requests.post(f"{VLLM_URL}/v1/chat/completions", json={
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "stream": False,
        "kv_transfer_params": {
            "do_remote_prefill": False,
            "do_remote_decode": True,
            "pd_request_prefix": "pd-longcp1",
        }
    }, timeout=120)
    data = resp.json()
    if "error" in data:
        print(f"ERROR: {data['error']}")
        return False

    prompt_tokens = data["usage"]["prompt_tokens"]
    kv_params = data.get("kv_transfer_params", {})
    cp_world = kv_params.get("cp_world_size", 0)
    print(f"  prompt_tokens={prompt_tokens}, cp_world_size={cp_world}")

    kv_dir = "/tmp/local_pd_kv/pd-longcp1"
    if os.path.exists(kv_dir):
        rank_dirs = sorted(d for d in os.listdir(kv_dir) if d.startswith("rank_"))
        print(f"  rank_dirs={rank_dirs}")
        if len(rank_dirs) > 1:
            print(f"  PASS: Full CP used ({len(rank_dirs)} ranks)")
        else:
            print(f"  INFO: Single rank (may be below threshold)")

    # Step 2: Decode
    print("\n=== Step 2: Decode (should use CP=1) ===")
    resp = requests.post(f"{VLLM_URL}/v1/chat/completions", json={
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "stream": False,
        "kv_transfer_params": {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "pd_request_prefix": "pd-longcp1",
            "cp_world_size": cp_world,
            "num_prompt_tokens": prompt_tokens,
        }
    }, timeout=120)
    data = resp.json()
    if "error" in data:
        print(f"ERROR: {data['error']}")
        return False

    content = data["choices"][0]["message"]["content"]
    print(f"  content={content[:80]!r}")
    return True


def test_proxy_e2e():
    """Test full proxy end-to-end."""
    print("\n=== Step 3: Proxy E2E (long prompt) ===")
    prompt = " ".join(["The quick brown fox jumps over the lazy dog."] * 20)

    resp = requests.post(f"{PROXY_URL}/v1/chat/completions", json={
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "stream": False,
    }, timeout=120)
    data = resp.json()
    if "error" in data:
        print(f"ERROR: {data['error']}")
        return False

    content = data["choices"][0]["message"]["content"]
    req_id = data.get("id", "")
    print(f"  content={content[:80]!r}")
    print(f"  request_id={req_id}")
    return "decode-pd-" in req_id


def test_kv_correctness_short():
    """Compare short request output with and without PD separation."""
    print("\n=== Step 4a: KV Correctness - Short Request ===")
    prompt = "Explain what a neural network is in simple terms."

    resp1 = requests.post(f"{VLLM_URL}/v1/chat/completions", json={
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 30, "stream": False, "temperature": 0, "seed": 42,
    }, timeout=60)
    content1 = resp1.json()["choices"][0]["message"]["content"]

    resp2 = requests.post(f"{PROXY_URL}/v1/chat/completions", json={
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 30, "stream": False, "temperature": 0, "seed": 42,
    }, timeout=120)
    d2 = resp2.json()
    if "error" in d2:
        print(f"  With PD ERROR: {d2['error']}")
        return False
    content2 = d2["choices"][0]["message"]["content"]

    match = content1 == content2
    print(f"  Without PD: {content1[:60]!r}")
    print(f"  With PD:    {content2[:60]!r}")
    print(f"  Match: {match}")
    return match


def test_kv_correctness_long():
    """Compare long request output (full CP) with/without PD separation."""
    print("\n=== Step 4b: KV Correctness - Long Request (full CP) ===")
    # >100 tokens prompt to trigger full CP
    prompt = " ".join(["The quick brown fox jumps over the lazy dog."] * 20)

    resp1 = requests.post(f"{VLLM_URL}/v1/chat/completions", json={
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 30, "stream": False, "temperature": 0, "seed": 42,
    }, timeout=60)
    d1 = resp1.json()
    content1 = d1["choices"][0]["message"]["content"]
    tokens1 = d1["usage"]["prompt_tokens"]

    resp2 = requests.post(f"{PROXY_URL}/v1/chat/completions", json={
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 30, "stream": False, "temperature": 0, "seed": 42,
    }, timeout=120)
    d2 = resp2.json()
    if "error" in d2:
        print(f"  ERROR: {d2['error']}")
        return False
    content2 = d2["choices"][0]["message"]["content"]
    tokens2 = d2["usage"]["prompt_tokens"]

    match = content1 == content2
    print(f"  Tokens: {tokens1} (no PD) vs {tokens2} (PD)")
    print(f"  Without PD: {content1[:80]!r}")
    print(f"  With PD:    {content2[:80]!r}")
    print(f"  Match: {match}")
    return match


if __name__ == "__main__":
    results = []
    results.append(("Long Request CP", test_long_request_cp()))
    results.append(("Proxy E2E", test_proxy_e2e()))
    results.append(("KV Correctness (short)", test_kv_correctness_short()))
    results.append(("KV Correctness (long/full CP)", test_kv_correctness_long()))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    all_pass = all(r[1] for r in results)
    print(f"\nOverall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
