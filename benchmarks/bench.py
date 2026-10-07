"""
Benchmark client: hits a running InferFlow API and measures real
latency/throughput under configurable concurrency. No numbers here are
invented - everything is computed from actual HTTP round-trips.

Usage:
    python benchmarks/bench.py --concurrency 1 --requests 20
    python benchmarks/bench.py --concurrency 10 --requests 20
"""
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests

PROMPTS = [
    "Hello, my name is",
    "The weather today is",
    "In the future, technology will",
    "My favorite food is",
    "The best way to learn programming is",
]


def one_request(base_url: str, max_tokens: int, idx: int) -> dict:
    prompt = PROMPTS[idx % len(PROMPTS)]
    start = time.perf_counter()
    resp = requests.post(
        f"{base_url}/generate",
        json={"prompt": prompt, "max_tokens": max_tokens},
        timeout=60,
    )
    wall_ms = (time.perf_counter() - start) * 1000
    resp.raise_for_status()
    data = resp.json()
    data["wall_ms"] = wall_ms
    return data


def percentile(values, pct):
    values = sorted(values)
    idx = int(len(values) * pct / 100)
    idx = min(idx, len(values) - 1)
    return values[idx]


def run_benchmark(base_url: str, concurrency: int, num_requests: int, max_tokens: int):
    results = []
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(one_request, base_url, max_tokens, i)
            for i in range(num_requests)
        ]
        for f in futures:
            results.append(f.result())

    total_wall_s = time.perf_counter() - start

    wall_latencies = [r["wall_ms"] for r in results]
    ttfts = [r["ttft_ms"] for r in results]
    tokens_per_sec_list = [r["tokens_per_sec"] for r in results]
    total_tokens = sum(r["completion_tokens"] for r in results)

    summary = {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "max_tokens": max_tokens,
        "total_wall_time_s": round(total_wall_s, 3),
        "throughput_req_per_sec": round(num_requests / total_wall_s, 3),
        "throughput_tokens_per_sec": round(total_tokens / total_wall_s, 3),
        "latency_ms": {
            "p50": round(percentile(wall_latencies, 50), 2),
            "p95": round(percentile(wall_latencies, 95), 2),
            "p99": round(percentile(wall_latencies, 99), 2),
            "min": round(min(wall_latencies), 2),
            "max": round(max(wall_latencies), 2),
            "mean": round(statistics.mean(wall_latencies), 2),
        },
        "ttft_ms": {
            "p50": round(percentile(ttfts, 50), 2),
            "p95": round(percentile(ttfts, 95), 2),
            "mean": round(statistics.mean(ttfts), 2),
        },
        "per_request_tokens_per_sec_mean": round(statistics.mean(tokens_per_sec_list), 2),
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--out", default=None, help="optional path to save JSON result")
    args = parser.parse_args()

    summary = run_benchmark(args.base_url, args.concurrency, args.requests, args.max_tokens)
    print(json.dumps(summary, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved to {args.out}")
