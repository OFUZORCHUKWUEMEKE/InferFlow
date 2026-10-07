# InferFlow

A from-scratch exploration of AI inference serving: loading a Hugging Face
model, serving it over HTTP, benchmarking it honestly, and optimizing it
one measured step at a time.

Environment this was built and benchmarked on: **CPU only** (4 cores,
Linux), no NVIDIA/AMD GPU. That constraint shapes several decisions below.

## Architecture

```
client -> FastAPI (src/api.py) -> InferenceEngine (src/inference.py) -> distilgpt2 (CPU)
                |
                v
          /metrics (Prometheus) -> Grafana
```

- `src/inference.py` - model loading + autoregressive generation loop with
  per-token timing (TTFT, inter-token latency). No web framework
  dependency, so it's testable standalone.
- `src/api.py` - FastAPI wrapper. Loads the model once at startup
  (lifespan), exposes `/generate`, `/health`, `/metrics`.
- `src/optimizations.py` - alternate engine configurations (KV-cache
  off, int8 dynamic quantization) for comparison against baseline.
- `benchmarks/bench.py` - concurrent HTTP load generator that computes
  p50/p95/p99 latency, TTFT, tokens/sec, and throughput from real
  requests against a running server. No numbers are hardcoded.
- `docker/` - Dockerfile + docker-compose (app + Prometheus + Grafana).
- `k8s/` - Deployment/Service/HPA manifests (written to be correct,
  not cluster-tested - see Known Limitations).

## Running it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# terminal 1
uvicorn src.api:app --host 0.0.0.0 --port 8000

# terminal 2
python benchmarks/bench.py --concurrency 1 --requests 20
python benchmarks/bench.py --concurrency 10 --requests 20
```

## Benchmark results

See [`benchmarks/results/`](benchmarks/results/) for raw JSON. Summary
filled in after running - see `BENCHMARKS.md`.

## What's real vs. simplified here

| Area | What we did | What production systems do differently |
|---|---|---|
| Batching | None by default; FastAPI handles requests one at a time through a single model instance | vLLM/SGLang use continuous batching - new requests join an in-flight batch mid-generation |
| KV-cache | Standard HF `past_key_values`, benchmarked on vs off | PagedAttention (vLLM) allocates KV-cache in non-contiguous pages to avoid fragmentation |
| Quantization | `torch.quantization.quantize_dynamic` (int8, CPU) | Production usually uses GPU-targeted quantization (GPTQ, AWQ, bitsandbytes) or FP8 on H100s |
| Concurrency | Python threads via FastAPI's threadpool for sync endpoints | Real servers use async model runners + request queues + scheduling policies |
| Monitoring | Prometheus counters/histograms for latency, TTFT, token count, error rate | Production adds GPU utilization/memory, queue depth, per-tenant rate limits |

## Known limitations

- **No SGLang benchmark.** SGLang's performance advantages (RadixAttention,
  continuous batching, prefix caching at scale) are GPU-oriented and
  largely moot on CPU-only hardware. Implementing it here would exercise
  the install/config path but not demonstrate the actual value proposition
  (which is GPU throughput under concurrency). Documented as a gap rather
  than faked.
- **K8s manifests are unverified against a real cluster** - no cluster was
  available in this environment. They're structurally valid YAML following
  standard patterns, not something we watched roll out.
- **Single model, single process.** No multi-model routing, no model
  warm-pool, no autoscaling actually exercised (HPA manifest exists but
  was never triggered against real load).
- **CPU-only benchmarks.** Every number in this repo reflects CPU
  inference on a 4-core machine. They do not transfer to GPU serving
  numbers - don't quote these figures as if they represent GPU performance.

## Lessons learned

See `LESSONS.md`.
