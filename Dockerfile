FROM python:3.11-slim

WORKDIR /app

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
    uv pip install --system --reinstall "opencv-python-headless" "mediapipe==0.10.21" "onnxruntime>=1.16.0"

COPY . .

RUN python -m src.ml.download_models

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
