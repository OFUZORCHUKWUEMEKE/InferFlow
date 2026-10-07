# Lessons Learned

Written after running the actual benchmarks in [`BENCHMARKS.md`](BENCHMARKS.md) -
not drafted in advance, since the point of this project is to not claim
numbers or conclusions we haven't measured.

## What surprised us (or confirmed theory concretely)

- **"Concurrent" requests ≠ parallel compute.** Running 10 requests through
  FastAPI's threadpool against a single synchronous model instance produced
  *zero* throughput gain (1.75 -> 1.70 req/sec) and ~10x worse latency.
  The requests were concurrent in the sense of "in flight simultaneously,"
  but the CPU-bound forward pass is still one resource being timesliced
  across 10 waiters. This is the concrete version of the "scheduling"
  bottleneck discussed in Phase 1 - without batching or a multi-worker
  pool, adding concurrency only adds queueing delay, not capacity.
- **KV-cache's TTFT-vs-steady-state split is real and measurable.** TTFT
  was identical with cache on/off (51.4ms vs 51.1ms) because the first
  token always requires a full forward pass over the prompt either way.
  The 2.6x slowdown only shows up in the tokens/sec for the full
  generation, i.e. in steps 2 through N. This matches the theory from
  Phase 1 but it's a different thing to see it in a number.
- **CPU-side quantization is a legitimate, if modest, lever.** +28%
  tokens/sec from `torch.quantization.quantize_dynamic` is a real, free
  win with no retraining - but it's far from the multi-x gains quantization
  gets on GPU (GPTQ/AWQ), because CPU int8 kernels aren't as aggressively
  optimized as GPU ones.

## Failure modes observed

- No crashes or OOM - `distilgpt2` is small enough (82M params) that CPU
  memory was never under real pressure on this machine. We did not
  observe an actual OOM failure mode in this project; that remains a gap
  (noted in README "Known Limitations") rather than something we can
  claim to have reproduced and fixed.

## What we'd do differently with a GPU

- Benchmark SGLang/vLLM directly against this baseline - meaningless on
  CPU since their main value (continuous batching, PagedAttention) is
  about maximizing GPU utilization under concurrency, which we don't have
  here.
- Re-run the concurrency=10 test and expect it to actually show a
  throughput benefit, since a GPU can genuinely execute a batched forward
  pass in parallel rather than timeslicing one CPU core.
- Compare quantization schemes meant for GPU (AWQ/GPTQ/FP8) instead of
  CPU-only dynamic int8.
