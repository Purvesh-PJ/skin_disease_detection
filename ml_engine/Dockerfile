# Dockerfile for Skin Disease Detection ML Training Engine
# Base image with Python 3.12 (matching project standard)
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    PYTHONPATH=/app

# Install minimal OS dependencies needed for OpenCV, system utils, and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container matching repository layout
WORKDIR /app

# Install ML requirements first for optimal Docker layer caching
COPY ml_engine/requirements.txt /app/ml_engine/requirements.txt
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r /app/ml_engine/requirements.txt

# Copy ML engine code into the container
COPY ml_engine/ /app/ml_engine/

# Create placeholder for trained_models directory inside container
RUN mkdir -p /app/trained_models

# Default entrypoint is an interactive shell (prevents accidental model training)
CMD ["bash"]
