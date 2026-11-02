# Changelog

All notable changes to this project will be documented in this file.

## [2.1.0] - 2026-11-03

### 🚀 What's New
- **Docker ML Training Environment**: You can now train models in an isolated Docker container (`ml-engine`) without installing TensorFlow, OpenCV, or CUDA on your host machine.
- **Accidental Training Safety**: Docker containers now stay idle on boot and will never start heavy model training unless you explicitly run a training command.
- **Automated Testing Suite (63 Tests)**: Added full test coverage across backend (`pytest`) and frontend (`Jest`), with automated GitHub Actions CI running on every push.
- **Single-Command Training CLI**: Train any model or the full ensemble with a simple command:
  `python ml_engine/run_pipeline.py --model resnet101 --epochs 20`
- **One-Command Dataset Tooling**:
  - `npm run ml:prepare`: Automatically unpacks and splits all 10,015 images into train/val/test folders in ~30 seconds.
  - `npm run ml:verify`: Instantly verifies dataset integrity and libraries with zero risk of starting training.

### ⚡ Improvements
- **Simplified Dataset Onboarding**: Replaced Kaggle API token setup with a straightforward 2-step browser download into `ml_engine/data/raw/`.
- **Live Terminal Progress**: Added real-time progress counters (updating every 500 images) and unbuffered logs (`PYTHONUNBUFFERED=1`) so the terminal never feels stuck or frozen.
- **Smart Python Runner**: Scripts now automatically locate your local virtual environment, preventing `ModuleNotFoundError: No module named 'pandas'`.
- **Fast Host File Copying**: Prioritized native preparation to avoid slow WSL2/Docker volume bind-mount lag when handling 10,000+ files.

### 🧹 Cleanup
- Removed unused legacy scripts (`backend/convert_to_tflite.py` and `ml_engine/scripts/download_dataset.py`).
- Centralized dataset paths in `ml_engine/constants.py`.
- Documented architectural decisions in ADR 0006 (Testing), ADR 0007 (ML Pipeline), and ADR 0008 (Docker ML Engine).

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
