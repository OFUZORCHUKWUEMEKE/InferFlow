"""Minimal smoke test - not a full suite. Verifies the generation loop
produces the right shapes/types and that KV-cache on/off both work."""
from inference import InferenceEngine


def test_generate_basic():
    engine = InferenceEngine(use_cache=True)
    result = engine.generate("Hello, my name is", max_tokens=5)
    assert result.completion_tokens <= 5
    assert result.completion_tokens >= 1
    assert isinstance(result.text, str)
    assert result.ttft_ms > 0
    assert result.total_ms >= result.ttft_ms


def test_generate_no_cache():
    engine = InferenceEngine(use_cache=False)
    result = engine.generate("Hello, my name is", max_tokens=5)
    assert result.completion_tokens >= 1


if __name__ == "__main__":
    test_generate_basic()
    test_generate_no_cache()
    print("All smoke tests passed.")
