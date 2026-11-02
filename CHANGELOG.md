# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-11-03

### Added
- **Containerized ML Training Engine**: Dedicated `ml_engine/Dockerfile` (Python 3.12-slim, OpenCV system libraries, layer-cached dependencies) and decoupled `ml-engine` service in `docker-compose.yml` for isolated model training without polluting the host environment.
- **Accidental Training Safeguards**: Configured `restart: "no"` and an idle container standby state (`tail -f /dev/null`) to guarantee that heavy neural network training never triggers automatically on container boot or `docker compose up`.
- **Modular Machine Learning Pipeline (`ml_engine/pipeline/`)**: Decoupled the training lifecycle into single-responsibility modules:
  - `data_generators.py`: Keras batch image generators with data augmentation.
  - `class_weights.py`: Balanced class weighting for minority diagnostic conditions.
  - `model_trainer.py`: Callback management (`EarlyStopping`, `ReduceLROnPlateau`) and model fitting.
  - `ensemble_training.py`: Multi-model base training and Logistic Regression meta-learner stacking.
  - `model_evaluation.py`: Test dataset evaluation, classification report, and confusion matrix.
  - `training_pipeline.py`: Master pipeline runner coordinating the end-to-end training lifecycle.
- **Headless Master Training CLI**: Added `python ml_engine/run_pipeline.py --model [resnet101|densenet121|efficientnetb3|ensemble]` for automated training and evaluation.
- **Dynamic Range Quantization (`ml_engine/export/tflite.py`)**: Automated conversion of heavy Keras `.h5` models into deployable, optimized `.tflite` flatbuffers.
- **Comprehensive Automated Testing Suite (63 Tests)**:
  - Backend unit and integration tests (`pytest`, `pytest-cov`, `pytest-mock`) covering endpoints (`/health`, `/predict`), preprocessing, ensemble consensus, and cloud model downloader with ~90%+ coverage.
  - Frontend component and service tests (`Jest` + React Testing Library) covering `ResultsCard`, `HistoryList`, `ImageUploadCard`, and `predictionService`.
  - ML Engine automation tests (`ml_engine/tests/`) validating dataset verification, archive extraction, and TFLite compilation.
- **Unified Test Runners**: Root commands `npm test`, `npm run test:backend`, `npm run test:frontend`, `npm run test:ml`, and full-stack coverage `npm run test:cov`.
- **Automated GitHub Actions CI/CD (`.github/workflows/ci.yml`)**: Cloud CI pipeline running tests and build checks on every push and PR to ensure zero regressions.
- **High-Speed Native Dataset Preprocessor (`npm run ml:prepare`)**: Direct disk-to-disk file copier and partitioner that splits 10,015 images into balanced train/val/test folders across 7 classes in ~30 seconds, bypassing WSL2/Docker volume bind-mount latency.
- **Live Streaming Progress Feedback**: Real-time archive extraction counters and live copying progress logs (every 500 images) with `PYTHONUNBUFFERED=1` so terminal status is continuously visible.
- **Safe Dataset Readiness Verifier (`npm run ml:verify`)**: Self-contained `verify_dataset_readiness.py` script that safely checks core dependencies and dataset integrity with 0% risk of starting training.
- **Smart Virtual Environment Runner (`scripts/run_ml_command.py`)**: Automatically detects host Python virtual environments, preventing missing `pandas` errors.
- **Architecture Decision Records**:
  - [ADR 0006](docs/adr/0006-testing-infrastructure-and-ci-cd.md): Testing infrastructure and CI/CD quality gate decisions.
  - [ADR 0007](docs/adr/0007-automated-local-ml-training-and-quantization.md): Modular ML training architecture and TFLite export.
  - [ADR 0008](docs/adr/0008-containerized-ml-training-engine.md): Containerized ML training engine and file I/O separation.

### Changed
- Streamlined HAM10000 dataset onboarding into a direct 2-step browser download workflow (`ml_engine/data/raw/`).
- Centralized dataset configuration paths (`DEFAULT_DATASET_DIR`, `DEFAULT_RAW_DIR`, `SPLITS`) into `ml_engine/constants.py`.
- Upgraded frontend Dockerfile to `node:20-alpine` and tuned CI build flags to prevent memory issues.

### Removed
- Removed legacy `backend/convert_to_tflite.py` and redundant `ml_engine/scripts/download_dataset.py`.

## [2.0.0] - 2026-11-01

### Added
- Direct 1-click access to the Diagnostic Studio (`/dashboard`) from the landing page with zero login friction.
- Browser `localStorage` diagnostic scan history persistence for instant, offline-resilient saving.
- Architecture Decision Record [0005](docs/adr/0005-removed-mongodb.md) documenting decoupling of MongoDB.
- Automated TFLite model pre-caching during Docker build via `download_models.py` from Hugging Face Hub.

### Changed
- Refactored Python backend into a dedicated, stateless AI inference microservice (<200MB RAM footprint).
- Streamlined Header and Navigation components to eliminate auth tokens, user menus, and session state.
- Updated `docker-compose.yml` and `.env.example` to remove database containers, credentials, and MongoDB dependencies.

### Removed
- Removed MongoDB database connection pool, user models, repositories, and server-side history routes.
- Removed user authentication layers (`Flask-JWT-Extended`, `Flask-Bcrypt`, `AuthContext`, login/signup pages).
- Removed `pymongo` dependency from `backend/requirements.txt`.

## [1.0.0] - 2026-10-15

### Added
- Three-model deep learning ensemble (ResNet-101, DenseNet-121, EfficientNet-B3) trained on the HAM10000 dataset.
- Soft-voting consensus engine averaging 7-class softmax probability vectors.
- React 18 single-page application with clinical scan dashboard and curated test sample picker.
- Flask REST API supporting JWT authentication, bcrypt password hashing, and image inference endpoints.
- MongoDB Atlas cloud integration for persistent user accounts and scan audit history.
- Docker Compose configuration for multi-container local execution.
- Architecture Decision Records (ADRs) documenting technical choices and system constraints.

### Changed
- Converted full Keras `.h5` model graphs to quantized TensorFlow Lite flatbuffers, reducing peak inference memory from ~650 MB to ~120 MB.
- Structured inference execution to run sequentially with explicit garbage collection (`gc.collect()`).
- Offloaded heavy model checkpoints (~64 MB) to Hugging Face Model Hub with automated on-demand streaming.
- Overhauled documentation and architecture diagrams to clear industry standards.

### Fixed
- Fixed Gunicorn worker `SIGKILL 137` OOM crash on Render free tier (512 MB cgroup memory ceiling).
- Fixed excessive log noise caused by automated platform health check pings.
