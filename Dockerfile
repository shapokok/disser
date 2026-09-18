# Crop Disease Detection API + frontend
#
#   docker build -t crop-disease .
#   docker run -p 5001:5001 -v $(pwd)/models:/app/models crop-disease
#
# Model weights (models/*.pth) are NOT baked into the image: mount the models/
# directory (see docker-compose.yml).

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CROP_HOST=0.0.0.0 \
    CROP_PORT=5001 \
    CROP_DEVICE=cpu

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch: the default PyPI wheel pulls several GB of CUDA libraries.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu $(grep -E "^(torch|torchvision)==" requirements.txt | cut -d";" -f1) \
    && grep -v -E "^(torch|torchvision|nvidia-|triton)" requirements.txt > /tmp/requirements-cpu.txt \
    && pip install -r /tmp/requirements-cpu.txt

COPY backend ./backend
COPY frontend ./frontend
COPY models/class_names.json models/model_metrics.json ./models/
COPY results/metrics ./results/metrics
COPY data/sample_images ./data/sample_images

RUN mkdir -p data/uploads results/heatmaps

EXPOSE 5001
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5001/api/health').status==200 else 1)"

CMD ["gunicorn", "--chdir", "backend", "--bind", "0.0.0.0:5001", "--workers", "1", "--threads", "4", "--timeout", "300", "app:app"]
