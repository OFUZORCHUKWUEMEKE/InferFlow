# Benchmark Results

All numbers below are from actually running `benchmarks/bench.py` against
`src/api.py` serving `distilgpt2` on CPU (4 cores, Linux, no GPU).
Model: distilgpt2 (82M params). `max_tokens=20` per request unless noted.
Raw JSON in [`benchmarks/results/`](benchmarks/results/).

## Baseline (KV-cache on, concurrency=1)

20 sequential requests, one at a time.

| Metric | Value |
|---|---|
| Throughput | 1.75 req/sec, 34.99 tokens/sec |
| Latency p50 | 568.0 ms |
| Latency p95 | 611.0 ms |
| Latency p99 | 611.0 ms |
| TTFT p50 | 50.9 ms |
| TTFT p95 | 72.8 ms |

## Concurrency=10

Same 20 requests, 10 in flight at once.

| Metric | Value |
|---|---|
| Throughput | 1.70 req/sec, 33.9 tokens/sec |
| Latency p50 | 5856.6 ms |
| Latency p95 | 6196.3 ms |
| TTFT p50 | 503.2 ms |

**The finding that matters:** throughput did **not** improve with 10x
concurrency (1.70 vs 1.75 req/sec - within noise, essentially flat).
Latency got roughly **10x worse** (p50 568ms -> 5857ms). This is the
textbook signature of **no batching, no true parallel compute**: FastAPI's
threadpool lets 10 requests be "in flight," but each one still waits its
turn for the same single CPU-bound model forward pass. Ten threads queueing
for one resource looks like concurrency but behaves like a line at one
checkout counter - the counter doesn't get faster because more people are
waiting. See [`README.md`](README.md#whats-real-vs-simplified-here) for
what a real batching engine (vLLM/SGLang) does differently.

## Optimization comparisons

One request, prompt = "The future of artificial intelligence is", 30 tokens generated.

| Config | TTFT (ms) | Tokens/sec | vs. baseline |
|---|---|---|---|
| Baseline (KV-cache on) | 51.4 | 35.11 | - |
| KV-cache off | 51.1 | 13.70 | **2.6x slower** |
| Int8 dynamic quantization | 35.3 | 45.08 | **1.28x faster** |

**KV-cache:** TTFT is unaffected (first token always does a full forward
pass regardless of caching), but steady-state generation is 2.6x slower
without it - every step recomputes attention over the *entire* sequence
so far instead of reusing cached K/V from prior steps. This gap grows with
sequence length; at `max_tokens=20` it's already 2.6x, and it would be
worse for longer generations.

**Int8 dynamic quantization:** modest but real win on CPU (+28% tokens/sec,
~31% lower TTFT) from `torch.quantization.quantize_dynamic` on Linear
layers. This is a CPU-specific technique - production GPU serving usually
reaches for GPTQ/AWQ/FP8 instead (see README limitations).
