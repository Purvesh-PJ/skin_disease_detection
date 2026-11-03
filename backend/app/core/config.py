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


def _resolve_app_version() -> str:
    """Resolve application version from env, VERSION file, or package.json (SSOT)."""
    env_version = os.getenv("APP_VERSION")
    if env_version:
        return env_version.strip()

    candidate_paths = (
        os.path.join(REPO_ROOT, "VERSION"),
        os.path.join(BASE_DIR, "VERSION"),
        os.path.join(REPO_ROOT, "package.json"),
        os.path.join(BASE_DIR, "package.json"),
    )
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    if path.endswith(".json"):
                        import json
                        val = json.load(f).get("version")
                        if val:
                            return str(val).strip()
                    else:
                        val = f.read().strip()
                        if val:
                            return val
            except Exception:
                pass
    return "2.1.0"


class AppConfig:
    """Lightweight application configuration container."""
    VERSION = _resolve_app_version()
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
