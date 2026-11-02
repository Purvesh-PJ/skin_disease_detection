# ADR 0008: Containerized ML Training Engine Environment via Docker

- **Status**: Accepted
- **Date**: 2026-11-02
- **Context**: Issue #44 (Containerized Docker environment for ML engine training and research workflow)

## Context & Problem Statement

The machine learning research engine (`ml_engine/`) required contributors to manually install heavy dependencies (TensorFlow, OpenCV, Scikit-learn) directly on host systems, introducing cross-platform setup friction. Additionally, running ML in Docker requires strict safeguards against **accidental automated model training** during container boot or `docker compose up`.

## Decision

We containerized the ML training engine with a decoupled, safe, and performance-optimized architecture:

1. **Dedicated `ml_engine/Dockerfile`**:
   - Python 3.12-slim base with system OpenCV libraries (`libgl1`, `libglib2.0-0`) and cached ML dependencies.
   - Decoupled from lightweight backend image to keep production serving footprint small (~300MB vs ~3GB).
   - Default `CMD` set to interactive `bash`, explicitly preventing unintended training loops.

2. **Compose Service with Safety Safeguards (`ml-engine`)**:
   - Volume mounts for `./ml_engine` and `./trained_models` for host persistence.
   - `restart: "no"` and idle standby command (`tail -f /dev/null`) so the service only trains when explicitly commanded.

3. **I/O & Compute Workload Separation**:
   - **Dataset Preparation (10,015 files)**: Executed on the native host (`npm run ml:prepare`) in ~30s to avoid WSL2/Docker 9p cross-filesystem bind-mount I/O latency.
   - **Model Training**: Executed containerized (`docker compose run --rm ml-engine ...`) for dependency isolation.

## Essential Commands

```bash
# Verify environment & dataset readiness (0% training, safe check):
npm run ml:verify
# Or via Docker: docker compose run --rm ml-engine python ml_engine/scripts/verify_dataset_readiness.py

# Train single model (ResNet-101):
docker compose run --rm ml-engine python ml_engine/run_pipeline.py --model resnet101 --epochs 20

# Train full 3-model stacking ensemble & export quantized TFLite:
docker compose run --rm ml-engine python ml_engine/run_pipeline.py --model ensemble --epochs 15
```

## Consequences

- **Positive**: Zero local Python/CUDA driver setup needed for model training.
- **Positive**: Strict protection against accidental heavy training on container startup.
- **Positive**: Host volume persistence guarantees `.h5` and `.tflite` model checkpoints are retained.

