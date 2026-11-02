# ADR 0008: Containerized ML Training Engine Environment via Docker

- **Status**: Accepted
- **Date**: 2026-11-02
- **Context**: Issue #44 (Containerized Docker environment for ML engine training and research workflow)

## Context & Problem Statement

Following the modularization of the training pipeline (ADR 0007), the machine learning research engine (`ml_engine/`) could run headlessly via `python ml_engine/run_pipeline.py`. However, the repository lacked a containerized environment for ML training.

While the web tier (Flask backend and React frontend) was fully dockerized in `docker-compose.yml`, contributors and researchers attempting to train models locally had to manually install Python, compile C-extensions, and manage heavy machine learning dependencies (TensorFlow, OpenCV, Scikit-learn, Albumentations) directly on their host machines. This introduced OS-specific setup friction across Windows, macOS, and Linux.

Furthermore, a containerized ML environment must guard against **accidental automated model training** — heavy GPU/CPU training workloads must never execute automatically upon container startup or during general `docker compose up` commands.

## Decision

We containerized the ML training engine alongside the existing application services using a dedicated, decoupled, and safe architecture:

### 1. Dedicated `ml_engine/Dockerfile`
We created a separate `Dockerfile` based on `python:3.12-slim`:
- Pre-installs system libraries required for OpenCV (`libgl1`, `libglib2.0-0`, `curl`, `ca-certificates`).
- Pre-installs `ml_engine/requirements.txt` with layer caching.
- Sets `PYTHONPATH=/app`, `TF_CPP_MIN_LOG_LEVEL=2`, and `TF_ENABLE_ONEDNN_OPTS=0`.
- Sets working directory strictly to `/app` to ensure relative repository paths (`/app/ml_engine/data`, `/app/trained_models`) resolve identically inside the container as they do on the host machine.
- Sets the default `CMD` to an interactive shell (`bash`), explicitly preventing unintended model training.

### 2. Architecture Decoupling (Serving vs. Training)
We deliberately maintain two distinct Docker environments rather than a single monolithic container:
- **`backend/Dockerfile`**: Optimized for low-memory, high-concurrency production inference (~300MB image, Flask, LiteRT/TFLite runtime, Gunicorn).
- **`ml_engine/Dockerfile`**: Optimized for deep learning model training and feature extraction (~3GB image, full TensorFlow, Scikit-learn, ImageDataGenerators).

This separation prevents backend image bloat and ensures the live microservice can deploy to free-tier cloud environments (e.g. Render 512MB RAM limit) without incurring the multi-gigabyte memory and storage penalties of a training runtime.

### 3. Unified Docker Compose Service (`ml-engine`) with Training Safeguards
Integrated the `ml-engine` service into root `docker-compose.yml`:
- **Volume Bind Mounts**: Maps host `./ml_engine` to container `/app/ml_engine` and `./trained_models` to container `/app/trained_models`.
- **Training Safeguard (`restart: "no"`)**: Ensures the container never auto-restarts into a training loop.
- **Idle Readiness Command**: Container stays in an idle standby state (`tail -f /dev/null`) when running with `docker compose up`, only executing training when explicitly commanded by the developer.

## Developer Workflows

Developers can inspect the environment, verify datasets, and train models without installing Python or TensorFlow on their host OS:

```bash
# 1. Build the ML engine image (targeted, skips web services):
docker compose build ml-engine

# 2. Inspect ML framework and dataset readiness (0% training, safe check):
docker compose run --rm ml-engine python ml_engine/scripts/verify_dataset_readiness.py

# 3. Enter interactive container shell (safe exploration):
docker compose run --rm ml-engine bash

# 4. View pipeline CLI help:
docker compose run --rm ml-engine python ml_engine/run_pipeline.py --help

# 5. Explicitly trigger training for a specific architecture (e.g., ResNet-101):
docker compose run --rm ml-engine python ml_engine/run_pipeline.py --model resnet101 --epochs 20

# 6. Explicitly trigger full 3-model stacking ensemble and auto-export quantized TFLite:
docker compose run --rm ml-engine python ml_engine/run_pipeline.py --model ensemble --epochs 15
```

## Consequences

- **Positive**: Zero-friction contributor onboarding — no local Python or CUDA driver setup required.
- **Positive**: Accidental training protection — container start never initiates heavy model fitting automatically.
- **Positive**: Complete path consistency — `/app` workspace ensures all relative paths (`REPO_ROOT / 'ml_engine' / 'data'`) function seamlessly.
- **Positive**: Host volume persistence guarantees training checkpoints (`.tflite`, `.h5`) are retained across container lifecycles.
