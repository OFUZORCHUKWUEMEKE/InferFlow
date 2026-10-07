"""
Core inference wrapper: loads a Hugging Face causal LM and runs
autoregressive generation with per-token timing instrumentation.

This is the thing every other layer (API, benchmark, optimizations)
wraps around. Kept dependency-free of FastAPI/Prometheus so it can be
unit-tested and reused standalone.
"""
import time
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float                 # time to first token
    inter_token_latencies_ms: list = field(default_factory=list)
    total_ms: float = 0.0

    @property
    def tokens_per_sec(self) -> float:
        if self.total_ms == 0:
            return 0.0
        return self.completion_tokens / (self.total_ms / 1000)


class InferenceEngine:
    """Loads once, reused across requests. Loading a model per-request
    would dominate latency (model load >> inference time)."""

    def __init__(self, model_name: str = MODEL_NAME, use_cache: bool = True):
        self.model_name = model_name
        self.use_cache = use_cache
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()  # disables dropout etc. - inference mode, not training

    @torch.inference_mode()  # no autograd bookkeeping at all (cheaper than no_grad)
    def generate(self, prompt: str, max_tokens: int = 20) -> GenerationResult:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        prompt_tokens = input_ids.shape[1]

        past_key_values = None
        generated_ids = []
        inter_token_latencies_ms = []

        start = time.perf_counter()
        first_token_time = None

        current_input = input_ids
        for _ in range(max_tokens):
            step_start = time.perf_counter()

            if self.use_cache and past_key_values is not None:
                # Only feed the newest token; KV-cache supplies the rest of the context.
                model_input = current_input[:, -1:]
            else:
                model_input = current_input

            output = self.model(
                model_input,
                past_key_values=past_key_values,
                use_cache=self.use_cache,
            )
            past_key_values = output.past_key_values if self.use_cache else None

            next_token_logits = output.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            generated_ids.append(next_token.item())
            current_input = torch.cat([current_input, next_token], dim=1)

            step_elapsed = time.perf_counter() - step_start
            if first_token_time is None:
                first_token_time = step_elapsed
            else:
                inter_token_latencies_ms.append(step_elapsed * 1000)

        total_elapsed = time.perf_counter() - start
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(generated_ids),
            ttft_ms=(first_token_time or 0.0) * 1000,
            inter_token_latencies_ms=inter_token_latencies_ms,
            total_ms=total_elapsed * 1000,
        )


if __name__ == "__main__":
    engine = InferenceEngine()
    result = engine.generate("Hello, my name is", max_tokens=20)
    print(f"Text: {result.text!r}")
    print(f"Prompt tokens: {result.prompt_tokens}")
    print(f"Completion tokens: {result.completion_tokens}")
    print(f"TTFT: {result.ttft_ms:.2f}ms")
    print(f"Total: {result.total_ms:.2f}ms")
    print(f"Tokens/sec: {result.tokens_per_sec:.2f}")
