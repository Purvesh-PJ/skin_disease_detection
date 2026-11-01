# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-11-01

### Added
- Automated full-stack CI/CD workflow via GitHub Actions (`.github/workflows/ci.yml`) with automated test execution and coverage verification.
- Comprehensive automated test infrastructure with 63 unit and integration tests across backend (`pytest`) and frontend (`Jest` + RTL).
- Unified single-command root test runner (`npm test`) and full-stack coverage reporter (`npm run test:cov`).
- Direct 1-click access to the Diagnostic Studio (`/dashboard`) from the landing page with zero login friction.
- Browser `localStorage` diagnostic scan history persistence for instant, offline-resilient saving.
- Architecture Decision Record [0007](docs/adr/0007-automated-local-ml-training-and-quantization.md) documenting local ML training automation, dataset verification, and quantized TFLite export.
- Automated ML training pipeline (
un_pipeline.py), dataset integrity verifier (download_dataset.py), and TFLite model quantizer (pipeline/export_tflite.py).
- Added automated ML engine testing suite (ml_engine/tests/) with root command 
pm run test:ml.
- Architecture Decision Record [0006](docs/adr/0006-testing-infrastructure-and-ci-cd.md) documenting testing frameworks selection (pytest, Jest) and CI/CD quality gates.
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
