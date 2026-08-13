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

class AppConfig:
    """Enterprise application configuration container."""
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev_default_secret_key_123")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev_jwt_secret_key_456")
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/skin_disease_db")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "skin_disease_db")
    CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")]

    MODEL_DIR_SETTING = os.getenv("MODEL_DIR", "trained_models")
    if os.path.isabs(MODEL_DIR_SETTING):
        MODEL_DIR = MODEL_DIR_SETTING
    else:
        MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, MODEL_DIR_SETTING))

    MODEL_ZIP_URL = os.getenv("MODEL_ZIP_URL", "")
    MODEL_CLOUD_URLS = {
        "resnet": os.getenv("MODEL_URL_RESNET", ""),
        "densenet": os.getenv("MODEL_URL_DENSENET", ""),
        "efficientnet": os.getenv("MODEL_URL_EFFICIENTNET", "")
    }

config = AppConfig()
