"""
FastAPI serving layer around InferenceEngine.

Model loads once at startup (lifespan), not per-request - loading
distilgpt2 takes ~1-2s, which would dominate every request's latency
if done per-call. Requests run through a single shared engine instance;
FastAPI's async event loop lets multiple requests be in-flight, but the
actual forward pass is synchronous CPU work, so true parallel compute
doesn't happen without a thread/process pool (see README "Known Limitations").
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

from inference import InferenceEngine

engine: InferenceEngine | None = None

REQUEST_COUNT = Counter("inferflow_requests_total", "Total requests", ["status"])
REQUEST_LATENCY = Histogram("inferflow_request_latency_seconds", "End-to-end request latency")
TTFT_HISTOGRAM = Histogram("inferflow_ttft_seconds", "Time to first token")
TOKENS_GENERATED = Counter("inferflow_tokens_generated_total", "Total tokens generated")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = InferenceEngine()
    yield
    engine = None


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 20


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_ms: float
    tokens_per_sec: float


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    start = time.perf_counter()
    try:
        result = engine.generate(req.prompt, max_tokens=req.max_tokens)
    except Exception:
        REQUEST_COUNT.labels(status="error").inc()
        raise

    REQUEST_COUNT.labels(status="ok").inc()
    REQUEST_LATENCY.observe(time.perf_counter() - start)
    TTFT_HISTOGRAM.observe(result.ttft_ms / 1000)
    TOKENS_GENERATED.inc(result.completion_tokens)

    return GenerateResponse(
        text=result.text,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        ttft_ms=result.ttft_ms,
        total_ms=result.total_ms,
        tokens_per_sec=result.tokens_per_sec,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": engine is not None}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
