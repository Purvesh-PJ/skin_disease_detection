"""
Prediction Service
------------------
Business service layer for skin disease classification:
- Lazy loading AI models
- Triggering cloud downloads if missing
- Preprocessing skin lesion images
- Multi-model ensemble inference
"""

import os
import logging
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input
from app.core.config import config
from app.core.constants import IDX2CLASS, USER_FRIENDLY_MAPPING, TARGET_IMAGE_SIZE
from app.core.exceptions import ModelNotFoundError, InvalidImageError
from app.ml.cloud_downloader import download_and_extract_zip, download_file_from_cloud

logger = logging.getLogger(__name__)

_loaded_models = {}

def get_models():
    """Lazy loads AI model ensemble dictionary."""
    global _loaded_models
    if not _loaded_models:
        os.makedirs(config.MODEL_DIR, exist_ok=True)

        model_files = {
            "resnet": os.path.join(config.MODEL_DIR, "resnet101.h5"),
            "densenet": os.path.join(config.MODEL_DIR, "densenet121.h5"),
            "efficientnet": os.path.join(config.MODEL_DIR, "efficientnetb3.h5")
        }

        missing_models = [name for name, path in model_files.items() if not os.path.exists(path)]

        if missing_models and config.MODEL_ZIP_URL:
            logger.info(f"Models missing {missing_models}. Triggering cloud ZIP download...")
            download_and_extract_zip(config.MODEL_ZIP_URL, config.MODEL_DIR)

        for name, path in model_files.items():
            if not os.path.exists(path) and config.MODEL_CLOUD_URLS.get(name):
                logger.info(f"Attempting individual download for '{name}'...")
                download_file_from_cloud(config.MODEL_CLOUD_URLS[name], path)

            if os.path.exists(path):
                try:
                    _loaded_models[name] = tf.keras.models.load_model(path, compile=False)
                    logger.info(f"Loaded model '{name}' from {path}")
                except Exception as e:
                    logger.error(f"Failed to load model '{name}' from {path}: {e}")
            else:
                logger.warning(f"Model file not found for '{name}' at path: {path}")

    return _loaded_models

def load_and_preprocess_image(image_path: str, target_size=TARGET_IMAGE_SIZE):
    """Loads, resizes, and preprocesses input image for inference."""
    image = cv2.imread(image_path)
    if image is None:
        raise InvalidImageError(f"Image not found or unreadable: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

def predict_skin_disease(image_path: str) -> dict:
    """Executes ensemble prediction on input skin lesion image."""
    active_models = get_models()
    if not active_models:
        raise ModelNotFoundError(
            f"AI models unavailable. Ensure model files exist in {config.MODEL_DIR} or set MODEL_ZIP_URL."
        )

    preprocessed_image = load_and_preprocess_image(image_path)

    predictions = {}
    for name, model in active_models.items():
        pred = model.predict(preprocessed_image)[0]
        predictions[name] = pred

    avg_prediction = np.mean(list(predictions.values()), axis=0)
    predicted_index = int(np.argmax(avg_prediction))
    predicted_label = IDX2CLASS.get(predicted_index, str(predicted_index))
    friendly_info = USER_FRIENDLY_MAPPING.get(predicted_label, {})

    max_confidence = max(pred[predicted_index] for pred in predictions.values())
    confidence_percent = f"{int(round(max_confidence * 100))}"

    return {
        "predicted_disease": predicted_label,
        "confidence": confidence_percent,
        "disease_details": friendly_info
    }
