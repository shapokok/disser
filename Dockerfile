# Multi-stage Docker build for Crop Disease Detection System
# Stage 1: Base image with Python and dependencies
FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backend/requirements.txt /app/backend/

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Stage 2: Application
FROM base as app

# Copy application code
COPY backend /app/backend
COPY frontend /app/frontend
COPY models /app/models
COPY data /app/data

# Create necessary directories
RUN mkdir -p /app/data/uploads \
    /app/data/test_images \
    /app/models \
    /app/results/heatmaps

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=/app/backend/app.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/')" || exit 1

# Set working directory to backend
WORKDIR /app/backend

# Run the application
CMD ["python", "app.py"]
