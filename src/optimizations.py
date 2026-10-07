"""
Optimization experiments, each benchmarked against the baseline InferenceEngine.

1. no_cache_engine: KV-cache disabled - shows the cost of recomputing
   attention over the whole sequence every step.
2. quantized_engine: dynamic int8 quantization of Linear layers - CPU-only
   optimization (no GPU needed), trades a little accuracy for speed/memory.

Each function returns a ready-to-use InferenceEngine-like object so
benchmarks/bench.py's logic can be reused unmodified against them.
"""
import torch

from inference import InferenceEngine, MODEL_NAME


def no_cache_engine() -> InferenceEngine:
    return InferenceEngine(model_name=MODEL_NAME, use_cache=False)


def quantized_engine() -> InferenceEngine:
    engine = InferenceEngine(model_name=MODEL_NAME, use_cache=True)
    engine.model = torch.quantization.quantize_dynamic(
        engine.model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return engine


if __name__ == "__main__":
    import time

    prompt = "The future of artificial intelligence is"

    for name, factory in [
        ("baseline (kv-cache on)", lambda: InferenceEngine()),
        ("no kv-cache", no_cache_engine),
        ("int8 dynamic quantization", quantized_engine),
    ]:
        engine = factory()
        start = time.perf_counter()
        result = engine.generate(prompt, max_tokens=30)
        wall = (time.perf_counter() - start) * 1000
        print(f"\n--- {name} ---")
        print(f"  total_ms (incl. model setup inside run): {wall:.1f}ms")
        print(f"  generate() total_ms: {result.total_ms:.1f}ms")
        print(f"  tokens/sec: {result.tokens_per_sec:.2f}")
        print(f"  ttft_ms: {result.ttft_ms:.1f}ms")
