# 0007. Automated Local Machine Learning Training, Dataset Management, and Quantized TFLite Export

## Status
Accepted

## Context
Following the transition to a decoupled microservice architecture (ADR 0005) and establishment of full-stack testing (ADR 0006), the machine learning research and training engine (`ml_engine/`) required modernization:
- Retraining or upgrading CNN backbones required manual, fragmented script executions.
- The repository previously lacked a standardized converter to transform heavy Keras `.h5` model graphs into low-memory, quantized `.tflite` flatbuffers required by the backend inference microservice.
- Cloud-hosted training platforms (e.g., Google Colab) introduce runtime disconnects, ephemeral storage limits, and cost barriers.
- Developers and data scientists required a unified, 1-click local training workflow that bridges both interactive Jupyter Notebook research and headless automated CLI pipelines.

## Decision

### 1. Dual-Track Architecture: Interactive Research & Automated CLI
We adopted a dual-track ML engineering standard:
- **Track 1 (Interactive Research via Jupyter Notebooks)**: `01_exploratory_data_analysis.ipynb` and `02_model_training_and_evaluation.ipynb` for visual demographic analysis, class imbalance inspection, augmentation tuning, and confusion matrix evaluation. Added a dedicated Section 5 for direct in-notebook `.tflite` export.
- **Track 2 (Headless Automated CLI Pipeline)**: `ml_engine/run_pipeline.py` providing a lightweight CLI driver supported by a modular orchestration layer (`ml_engine/pipeline/orchestrator.py` and `ml_engine/pipeline/data_loader.py`) to train individual architectures (`resnet101`, `densenet121`, `efficientnetb3`) or the full soft-voting ensemble with balanced class weighting.

### 2. Automated Dynamic Range Quantization (`ml_engine/pipeline/export_tflite.py`)
Implemented `export_tflite.py` utilizing `tf.lite.TFLiteConverter` with `Optimize.DEFAULT`:
- Automatically converts `.h5` weight graphs into `.tflite` flatbuffers.
- Achieves ~70% binary size reduction while preserving multi-class diagnostic accuracy.
- Guarantees peak memory consumption remains below ~120 MB on Render's 512 MB cgroup limit.

### 3. Operational Utilities & Data Segregation (`ml_engine/scripts/`, `ml_engine/data/`)
- `ml_engine/data/`: Strictly dedicated to housing actual data assets (`skin_disease_dataset/base_dir`).
- `ml_engine/scripts/download_dataset.py`: Validates the existence and distribution of all 10,015 HAM10000 images across `train_dir`, `val_dir`, and `test_dir` for all 7 classes (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`), providing 1-click Kaggle CLI automated download instructions if local dataset is missing.
- `ml_engine/scripts/prepare_dataset.py`: Automated stratified data splitter.

### 4. Machine Learning Automation Test Suite (`ml_engine/tests/`)
Created an automated test suite (`test_ml_engine_automation.py`) and integrated it into the root test runner (`npm run test:ml`):
- Verifies real dataset split integrity across 7 classes.
- Constructs minimal Keras neural networks and executes end-to-end `.h5` to quantized `.tflite` conversion.
- Verifies input/output tensor shapes `[1, 224, 224, 3]` and `[1, 7]` via the TFLite Interpreter.

## Consequences
- **Positive**: 1-click zero-friction local training and quantization without cloud dependency or subscription fees.
- **Positive**: Both notebook researchers and backend engineers share the exact same conversion utilities.
- **Positive**: Automated unit tests safeguard model conversion and tensor compatibility.
- **Positive**: Clean separation between heavy training dependencies (`ml_engine/requirements.txt`) and lightweight backend serving dependencies (`backend/requirements.txt`).

