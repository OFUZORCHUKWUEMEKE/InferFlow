FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Pre-download model weights at build time so the first request
# doesn't pay the HF download cost.
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('distilgpt2'); \
    AutoModelForCausalLM.from_pretrained('distilgpt2')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
