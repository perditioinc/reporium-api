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

EXPOSE 8080

CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
