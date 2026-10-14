# Multi-stage production Dockerfile for Skin Disease Detection API
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/trained_models \
    PORT=5000 \
    GUNICORN_TIMEOUT=300 \
    GUNICORN_WORKERS=1 \
    GUNICORN_THREADS=4

# Install minimal OS dependencies required for system networking & healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Ensure model directory and uploads directory exist
RUN mkdir -p /app/trained_models /app/uploads

# Install Python dependencies first for optimal Docker layer caching
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Pre-download AI model weights directly during Docker build from Hugging Face Model Hub
# This eliminates runtime download delay and prevents Render/Cloud cold-start timeouts
RUN python -c "\
import urllib.request; \
import os; \
models = { \
    'resnet101.h5': 'https://huggingface.co/purvesh-pj/skin-disease-ensemble/resolve/main/resnet101.h5', \
    'densenet121.h5': 'https://huggingface.co/purvesh-pj/skin-disease-ensemble/resolve/main/densenet121.h5', \
    'efficientnetb3.h5': 'https://huggingface.co/purvesh-pj/skin-disease-ensemble/resolve/main/efficientnetb3.h5' \
}; \
headers = {'User-Agent': 'Mozilla/5.0'}; \
for fname, url in models.items(): \
    target = os.path.join('/app/trained_models', fname); \
    print(f'Pre-caching model weight: {fname}...'); \
    req = urllib.request.Request(url, headers=headers); \
    with urllib.request.urlopen(req) as resp, open(target, 'wb') as f: \
        f.write(resp.read()); \
print('All model weights successfully pre-baked into image.'); \
"

# Copy backend application source code into container
COPY backend/ /app/

# Expose port
EXPOSE 5000

# Container healthcheck against / endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/ || exit 1

# Launch production Gunicorn with threaded workers to prevent HTTP blocking during TensorFlow inference
CMD ["sh", "-c", "gunicorn main:app --bind 0.0.0.0:${PORT:-5000} --workers ${GUNICORN_WORKERS:-1} --threads ${GUNICORN_THREADS:-4} --timeout ${GUNICORN_TIMEOUT:-300} --access-logfile - --error-logfile -"]
