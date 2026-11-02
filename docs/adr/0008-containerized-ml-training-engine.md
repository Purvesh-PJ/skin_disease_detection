# ADR 0008: Containerized ML Training Engine Environment via Docker

- **Status**: Accepted
- **Date**: 2026-11-02
- **Context**: Issue #44 (Containerized Docker environment for ML engine training and research workflow)

## Context & Problem Statement

Following the modularization of the training pipeline (ADR 0007), the machine learning research engine (`ml_engine/`) could run headlessly via `python ml_engine/run_pipeline.py`. However, the repository lacked a containerized environment for ML training.

While the web tier (Flask backend and React frontend) was fully dockerized in `docker-compose.yml`, contributors and researchers attempting to train models locally had to manually install Python, compile C-extensions, and manage heavy machine learning dependencies (TensorFlow, OpenCV, Scikit-learn, Albumentations) directly on their host machines. This introduced OS-specific setup friction across Windows, macOS, and Linux.

## Decision

We containerized the ML training engine alongside the existing application services using a dedicated, decoupled architecture:

### 1. Dedicated `ml_engine/Dockerfile`
We created a separate `Dockerfile` based on `python:3.12-slim`:
- Pre-installs system libraries required for OpenCV (`libgl1`, `libglib2.0-0`, `curl`, `ca-certificates`).
- Pre-installs `ml_engine/requirements.txt` with layer caching.
- Sets `TF_CPP_MIN_LOG_LEVEL=2` and `TF_ENABLE_ONEDNN_OPTS=0` for clean execution.
- Sets the default entrypoint to display pipeline runner options (`python ml_engine/run_pipeline.py --help`).

### 2. Architecture Decoupling (Serving vs. Training)
We deliberately maintain two distinct Docker environments rather than a single monolithic container:
- **`backend/Dockerfile`**: Optimized for low-memory, high-concurrency production inference (~300MB image, Flask, LiteRT/TFLite runtime, Gunicorn).
- **`ml_engine/Dockerfile`**: Optimized for deep learning model training and feature extraction (~3GB image, full TensorFlow, Scikit-learn, ImageDataGenerators).

This separation prevents backend image bloat and ensures the live microservice can deploy to free-tier cloud environments (e.g. Render 512MB RAM limit) without incurring the multi-gigabyte memory and storage penalties of a training runtime.

### 3. Unified Docker Compose Service (`ml-engine`)
Integrated the `ml-engine` service into root `docker-compose.yml`:
- **Volume Bind Mounts**: Maps host `./ml_engine` to container `/app/ml_engine` and `./trained_models` to container `/app/trained_models`.
- Ensures datasets stored in `ml_engine/data/` and trained `.h5` / `.tflite` model checkpoints are immediately synchronized with the host machine and directly accessible by the backend inference container.

## Developer Workflows

Developers can train models without installing Python or TensorFlow on their host OS:

```bash
# 1. Build the ML engine image:
docker compose build ml-engine

# 2. View pipeline CLI help:
docker compose run --rm ml-engine

# 3. Train a specific model architecture (e.g., ResNet-101):
docker compose run --rm ml-engine python ml_engine/run_pipeline.py --model resnet101 --epochs 20

# 4. Train full 3-model stacking ensemble and export quantized TFLite:
docker compose run --rm ml-engine python ml_engine/run_pipeline.py --model ensemble --epochs 15
```

## Consequences

- **Positive**: Zero-friction contributor onboarding — no local Python or CUDA driver setup required to execute model training.
- **Positive**: Strict decoupling between lightweight production serving (~300MB) and heavy training dependencies.
- **Positive**: Host volume persistence guarantees training checkpoints (`.tflite`, `.h5`) are retained across container lifecycles.
- **Positive**: Consistent, reproducible Linux-based training environment across all developer operating systems.
