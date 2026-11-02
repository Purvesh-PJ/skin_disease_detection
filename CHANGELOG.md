# Changelog

All notable changes to this project will be documented in this file.

## [2.1.0] - 2026-11-03

### 🧪 1. Full-Stack Automated Testing & CI/CD Pipeline
- **Comprehensive Test Suite (63 Tests)**:
  - **Backend (pytest)**: 30 unit & integration tests covering image preprocessing, ensemble voting, cloud downloader, health checks, and predict routes with ~90%+ coverage.
  - **Frontend (Jest + RTL)**: 33 component and unit tests covering `ResultsCard`, `HistoryList`, `ImageUploadCard`, and `predictionService`.
  - **ML Automation Tests**: 8 tests covering dataset verification, archive unpacking, stratification, and TFLite compilation.
- **Unified Test Runners**: Root shortcuts `npm test` (runs all 63 tests), `npm run test:backend`, `npm run test:frontend`, `npm run test:ml`, and full-stack coverage `npm run test:cov`.
- **GitHub Actions CI/CD (`.github/workflows/ci.yml`)**: Automated CI pipeline that runs tests and build checks on every push and pull request to ensure zero regressions.
- **Testing Architecture Record**: Added [ADR 0006](docs/adr/0006-testing-infrastructure-and-ci-cd.md) documenting testing frameworks and quality gates.

### 🔬 2. Modular ML Research & Training Engine
- **Independent Architecture Modules (`ml_engine/architectures/`)**:
  - `resnet.py`: ResNet-101 fine-tuned for dermatoscopic border and texture feature extraction.
  - `densenet.py`: DenseNet-121 with dense feature reuse for lesion classification.
  - `efficientnet.py`: EfficientNet-B3 with compound scaling for fine-grained diagnosis.
- **Modular Pipeline (`ml_engine/pipeline/`)**: Decoupled the training process into single-responsibility modules:
  - `data_generators.py`: Keras batch generators with data augmentation.
  - `class_weights.py`: Balanced class weighting for minority diagnostic classes.
  - `model_trainer.py`: Callback management (`EarlyStopping`, `ReduceLROnPlateau`) and model fitting.
  - `ensemble_training.py`: Multi-model base training and Logistic Regression meta-learner stacking.
  - `model_evaluation.py`: Test dataset evaluation, classification report, and confusion matrix.
  - `training_pipeline.py`: Master pipeline runner coordinating the end-to-end training lifecycle.
- **Single-Command Training CLI**: Train any model or the full ensemble headlessly:
  `python ml_engine/run_pipeline.py --model [resnet101|densenet121|efficientnetb3|ensemble] --epochs 20`
- **TFLite Model Quantization (`ml_engine/export/tflite.py`)**: Automated dynamic range quantization converting heavy `.h5` models into lightweight `.tflite` flatbuffers.
- **ML Training Architecture Record**: Added [ADR 0007](docs/adr/0007-automated-local-ml-training-and-quantization.md).

### 🐳 3. Containerized Docker ML Environment & Safeguards
- **Dedicated Docker ML Environment**: Added `ml_engine/Dockerfile` (Python 3.12-slim, OpenCV system libraries, layer-cached TensorFlow/Scikit-learn) and a decoupled `ml-engine` service in `docker-compose.yml`, isolating training from the lightweight backend microservice.
- **Accidental Training Safeguards**: Configured `restart: "no"` and an idle container standby state (`tail -f /dev/null`) to guarantee that heavy training never triggers automatically on container boot or `docker compose up`.
- **Docker ML Architecture Record**: Added [ADR 0008](docs/adr/0008-containerized-ml-training-engine.md).

### 📦 4. High-Performance Dataset Tooling & Developer Experience
- **Simplified Dataset Onboarding**: Replaced Kaggle API token setup with a straightforward 2-step browser download into `ml_engine/data/raw/`.
- **Fast Native Preprocessor (`npm run ml:prepare`)**: Direct disk-to-disk file copier and partitioner that splits 10,015 images into balanced train/val/test folders across 7 classes in ~30 seconds, bypassing slow WSL2/Docker volume bind-mount lag.
- **Live Terminal Progress**: Real-time archive extraction counters and live copy progress logs (every 500 images) with `PYTHONUNBUFFERED=1` so the terminal never feels stuck.
- **Safe Dataset Readiness Verifier (`npm run ml:verify`)**: Self-contained `verify_dataset_readiness.py` script that safely checks core dependencies and dataset integrity with 0% risk of starting training.
- **Smart Virtual Environment Runner (`scripts/run_ml_command.py`)**: Automatically detects host Python virtual environments, preventing missing `pandas` errors.

### 🧹 5. Cleanup & Maintenance
- Removed unused legacy scripts (`backend/convert_to_tflite.py` and `ml_engine/scripts/download_dataset.py`).
- Centralized dataset paths in `ml_engine/constants.py`.
- Upgraded frontend Dockerfile to `node:20-alpine` to fix webpack memory limits.

## [2.0.0] - 2026-11-01

### 🚀 What's New
- **Direct 1-Click Access**: Open the Diagnostic Studio (`/dashboard`) directly from the landing page without logging in.
- **Local History**: Scan history now saves directly in your browser (`localStorage`), so it works instantly and offline.
- **Hugging Face Model Caching**: Models download automatically on first launch from Hugging Face Hub.

### ⚡ Improvements
- **Lightweight Backend**: Converted the backend into a fast, stateless microservice using under 200MB of RAM (perfect for free cloud hosting like Render).
- **Simplified Navigation**: Cleaned up the header and navigation bar by removing login/signup buttons and user menus.

### 🧹 Cleanup
- Removed MongoDB database, user accounts, and JWT authentication dependencies.

## [1.0.0] - 2026-10-15

### 🚀 Initial Release
- Multi-model deep learning ensemble (ResNet-101, DenseNet-121, EfficientNet-B3) trained on HAM10000.
- Soft-voting consensus engine for skin disease classification.
- React 18 single-page application with scan dashboard and sample picker.
- Flask REST API with quantized TensorFlow Lite inference.
