# Dockerfile for Skin Disease Detection ML Training Engine
# Base image with Python 3.12 (matching project standard)
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    PYTHONPATH=/app

# Step 1: Install minimal OS dependencies needed for OpenCV, system utils, and healthchecks
RUN echo "=== [Step 1/5] Installing OS System Libraries (OpenCV dependencies) ===" && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container matching repository layout
WORKDIR /app

# Upgrade pip and packaging tools with visible progress
RUN echo "=== [Step 2/5] Upgrading Pip & Setup Utilities ===" && \
    pip install --no-cache-dir --upgrade pip setuptools wheel

# Step 3: Install Deep Learning Framework (TensorFlow & NumPy) in dedicated cached layer
RUN echo "=== [Step 3/5] Installing Deep Learning Framework (TensorFlow & NumPy) ===" && \
    pip install --no-cache-dir \
    "tensorflow>=2.18.0" \
    "numpy>=1.26.4" \
    "scipy>=1.11.4"

# Step 4: Install Computer Vision & Machine Learning Tools (Scikit-Learn, OpenCV, Albumentations)
RUN echo "=== [Step 4/5] Installing Machine Learning & Vision Libraries (Scikit-Learn, OpenCV) ===" && \
    pip install --no-cache-dir \
    "scikit-learn>=1.4.0" \
    "joblib>=1.4.0" \
    "pandas>=2.2.2" \
    "opencv-python-headless>=4.10.0.84" \
    "albumentations>=1.4.0"

# Step 5: Install Data Science Visualization & Notebook Research Tools
COPY ml_engine/requirements.txt /app/ml_engine/requirements.txt
RUN echo "=== [Step 5/5] Installing Research, EDA & Notebook Packages (JupyterLab, Matplotlib, Seaborn) ===" && \
    pip install --no-cache-dir -r /app/ml_engine/requirements.txt

# Copy ML engine code into the container
COPY ml_engine/ /app/ml_engine/

# Create placeholder for trained_models directory inside container
RUN mkdir -p /app/trained_models

# Default entrypoint is an interactive shell (prevents accidental model training)
CMD ["bash"]
