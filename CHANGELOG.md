# 📋 Changelog

All notable changes to the **Skin Disease Detection Platform** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-08-24

### 🚀 Added
* **Tri-Model Deep Learning Ensemble**:
  * Integrated **EfficientNetB3**, **ResNet101**, and **DenseNet121** CNN architectures trained on HAM10000 dataset (10,015 images).
  * Soft-voting ensemble mechanism for averaging softmax probability vectors across 7 skin disease classes (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).
  * Automated cloud model downloader (`cloud_downloader.py`) to fetch missing `.h5` model checkpoints on demand.
* **Backend API & Microservices (Python 3.12 / Flask)**:
  * Application Factory pattern with modular blueprints (`/auth`, `/predict`, `/history`, `/health`).
  * JWT-based authentication with `flask-jwt-extended` and salted Blowfish password hashing via `flask-bcrypt`.
  * Persistent MongoDB Atlas integration with `users` and `prediction_history` collections.
  * Instant **1-Click Recruiter Demo Access** (`/auth/demo-login`) with automatic demo user provisioning.
  * Centralized domain exception handlers and 16 MB upload ceiling with `secure_filename` sanitization.
* **Modern Frontend Web Application (React 18)**:
  * Responsive clinical workspace with drag-and-drop image dropzone and live dermoscopy preview.
  * Interactive HAM10000 sample images picker for instant 1-click evaluation without file searching.
  * Diagnostic result card displaying predicted disease, risk category, and percentage confidence gauge.
  * Dark/Light mode theme switching with instant CSS custom property tokens and `localStorage` persistence.
  * Optimistic UI history synchronization with fallback local caching for zero-downtime offline review.
* **DevOps & Infrastructure**:
  * Multi-stage `Dockerfile` configurations for both Flask backend and React SPA.
  * `docker-compose.yml` orchestrating MongoDB, backend, and frontend bridge networks.
  * Automated Windows batch execution launchers (`run_backend.bat`, `run_frontend.bat`).
  * Production deployment compatibility for Vercel, Render, Railway, and MongoDB Atlas.
* **Documentation**:
  * Created complete, categorized documentation portal in `docs/` (`getting-started/`, `architecture/`, `machine-learning/`, `api-reference/`, `devops/`, `security/`, `troubleshooting/`).

---

## [0.1.0] - Pre-Release
* Initial exploratory data analysis and Jupyter notebook prototyping (`dataset_inspection.ipynb`, `model_training.ipynb`).
