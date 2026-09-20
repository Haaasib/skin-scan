FROM python:3.11-slim

WORKDIR /app

ENV OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN pip install --no-cache-dir uv && \
    uv pip install --system -e . && \
    uv pip uninstall --system opencv-python opencv-contrib-python || true && \
    uv pip install --system --reinstall "opencv-python-headless" "mediapipe==0.10.21" "onnxruntime>=1.16.0" && \
    uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch && \
    uv pip install --system "transformers>=4.38.0" "huggingface_hub>=0.20.0" "safetensors>=0.4.0"

COPY . .

RUN python -m src.ml.download_models && \
    python -c "from src.ml.vit_panel import get_vit_panel; p=get_vit_panel(); print('vit', p.available)"

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
