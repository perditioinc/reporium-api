FROM python:3.12-slim

WORKDIR /app

# Install CPU-only PyTorch before the rest of requirements.
# sentence-transformers brings in full CUDA PyTorch (~2 GB) by default;
# Cloud Run has no GPU so we only need the CPU wheel (~200 MB).
# Separating this into its own layer also lets Docker re-use the cached
# layer across builds where only app code or other deps change.
RUN pip install --no-cache-dir \
      torch \
      --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the sentence-transformers model into the image at build time.
# Without this, every cold-start downloads ~90 MB from HuggingFace which:
#   1. Takes 30–300s depending on HuggingFace rate limits and network
#   2. Frequently exceeds Cloud Run's startup probe timeout (default 240s)
#   3. Gets rate-limited (429) when multiple instances start simultaneously
# Baking the model weights into the layer eliminates all three problems.
# The model is ~90 MB and is cached at /root/.cache/huggingface/hub/.
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/hub
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('Model downloaded OK')"

EXPOSE 8080

CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
