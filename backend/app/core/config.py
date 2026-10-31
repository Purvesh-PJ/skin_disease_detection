"""
Core Application Settings & Environment Configuration
-----------------------------------------------------
Centralized configuration manager reading environment variables.
"""

import os
from dotenv import load_dotenv, find_dotenv

# Load root .env
dotenv_path = find_dotenv(usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

class AppConfig:
    """Lightweight application configuration container."""
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev_default_secret_key_123")
    CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")]

    MODEL_DIR_SETTING = os.getenv("MODEL_DIR", "trained_models")
    if os.path.isabs(MODEL_DIR_SETTING):
        MODEL_DIR = MODEL_DIR_SETTING
    else:
        # Check repo root first (standard monorepo layout: /trained_models)
        root_path = os.path.abspath(os.path.join(REPO_ROOT, MODEL_DIR_SETTING))
        backend_path = os.path.abspath(os.path.join(BASE_DIR, MODEL_DIR_SETTING))
        MODEL_DIR = root_path if os.path.exists(root_path) else backend_path

    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    HUGGINGFACE_REPO_BASE = "https://huggingface.co/purvesh-pj/skin-disease-ensemble/resolve/main"
    MODEL_ZIP_URL = os.getenv("MODEL_ZIP_URL", "")
    MODEL_CLOUD_URLS = {
        "resnet": os.getenv("MODEL_URL_RESNET", f"{HUGGINGFACE_REPO_BASE}/resnet101.tflite"),
        "densenet": os.getenv("MODEL_URL_DENSENET", f"{HUGGINGFACE_REPO_BASE}/densenet121.tflite"),
        "efficientnet": os.getenv("MODEL_URL_EFFICIENTNET", f"{HUGGINGFACE_REPO_BASE}/efficientnetb3.tflite")
    }

config = AppConfig()
