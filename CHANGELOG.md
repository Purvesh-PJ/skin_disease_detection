# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
