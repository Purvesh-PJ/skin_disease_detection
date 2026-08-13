from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import urllib.request
import zipfile
import shutil
import numpy as np
import tensorflow as tf
import cv2
import logging
from tensorflow.keras.applications.efficientnet import preprocess_input

logger = logging.getLogger(__name__)

# Base directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_DIR_SETTING = os.getenv("MODEL_DIR", "trained_models")

if os.path.isabs(MODEL_DIR_SETTING):
    MODEL_DIR = MODEL_DIR_SETTING
else:
    MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, MODEL_DIR_SETTING))

# Global registry for loaded models
_loaded_models = {}

# Cloud storage URLs for automatic download if models are missing locally
MODEL_CLOUD_URLS = {
    "resnet": os.getenv("MODEL_URL_RESNET", ""),
    "densenet": os.getenv("MODEL_URL_DENSENET", ""),
    "efficientnet": os.getenv("MODEL_URL_EFFICIENTNET", "")
}
# Optional single ZIP download URL containing all 3 .h5 files
MODEL_ZIP_URL = os.getenv("MODEL_ZIP_URL", "")

def transform_google_drive_url(url):
    """Transforms Google Drive view/share links into direct raw content download URLs."""
    if "drive.google.com" in url or "drive.usercontent.google.com" in url:
        file_id = None
        if "/file/d/" in url:
            parts = url.split("/file/d/")
            if len(parts) > 1:
                file_id = parts[1].split("/")[0].split("?")[0]
        elif "id=" in url:
            file_id = url.split("id=")[1].split("&")[0]

        if file_id:
            return f"https://drive.usercontent.google.com/download?id={file_id}&confirm=t"
    return url

def download_file_from_cloud(url, dest_path):
    """Downloads a file from cloud storage URL to local disk."""
    try:
        download_url = transform_google_drive_url(url)
        logger.info(f"Downloading model from cloud: {download_url} -> {dest_path}")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        req = urllib.request.Request(
            download_url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        logger.info(f"Successfully downloaded model to {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download model from {url}: {e}")
        return False

def download_and_extract_zip(zip_url, target_dir):
    """Downloads and extracts a zip file containing models."""
    zip_path = os.path.join(target_dir, "models_download.zip")
    if download_file_from_cloud(zip_url, zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            logger.info(f"Extracted model zip into {target_dir}")
            os.remove(zip_path)
            return True
        except Exception as e:
            logger.error(f"Failed to extract zip {zip_path}: {e}")
    return False

def get_models():
    """
    Lazy loads and returns AI models. Returns a dict of loaded models.
    If model files are missing locally, attempts automatic download from cloud URLs.
    """
    global _loaded_models
    if not _loaded_models:
        os.makedirs(MODEL_DIR, exist_ok=True)

        model_files = {
            "resnet": os.path.join(MODEL_DIR, "resnet101.h5"),
            "densenet": os.path.join(MODEL_DIR, "densenet121.h5"),
            "efficientnet": os.path.join(MODEL_DIR, "efficientnetb3.h5")
        }

        # Check if any model is missing
        missing_models = [name for name, path in model_files.items() if not os.path.exists(path)]

        # If models are missing and a ZIP URL is provided, try zip download
        if missing_models and MODEL_ZIP_URL:
            logger.info(f"Models missing {missing_models}. Attempting cloud ZIP download...")
            download_and_extract_zip(MODEL_ZIP_URL, MODEL_DIR)

        for name, path in model_files.items():
            # If still missing, check individual file URL
            if not os.path.exists(path) and MODEL_CLOUD_URLS.get(name):
                logger.info(f"Attempting individual cloud download for '{name}'...")
                download_file_from_cloud(MODEL_CLOUD_URLS[name], path)

            if os.path.exists(path):
                try:
                    _loaded_models[name] = tf.keras.models.load_model(path)
                    logger.info(f"Loaded model '{name}' from {path}")
                except Exception as e:
                    logger.error(f"Failed to load model '{name}' from {path}: {e}")
            else:
                logger.warning(f"Model file not found for '{name}' at path: {path}")

    return _loaded_models

# Mapping from class names to indices
class_indices = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6
}
idx2class = {v: k for k, v in class_indices.items()}

# User-friendly display mapping
USER_FRIENDLY_MAPPING = {
    "akiec": {
        "name": "Actinic Keratoses",
        "description": (
            "Actinic keratoses are rough, scaly patches on the skin caused by years of sun exposure. "
            "They can sometimes develop into skin cancer and should be monitored by a dermatologist."
        )
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "description": (
            "Basal cell carcinoma is the most common type of skin cancer. It is usually slow-growing "
            "and rarely metastasizes, but professional evaluation is recommended."
        )
    },
    "bkl": {
        "name": "Benign Keratosis",
        "description": (
            "Benign keratoses are non-cancerous skin growths. They are typically harmless, though any changes "
            "should be evaluated by a healthcare provider."
        )
    },
    "df": {
        "name": "Dermatofibroma",
        "description": (
            "Dermatofibromas are benign skin nodules that generally do not require treatment unless they "
            "cause discomfort or cosmetic concerns."
        )
    },
    "mel": {
        "name": "Melanoma",
        "description": (
            "Melanoma is a serious form of skin cancer that can be life-threatening if not detected early. "
            "Immediate consultation with a dermatologist is crucial."
        )
    },
    "nv": {
        "name": "Melanocytic Nevus",
        "description": (
            "Melanocytic nevi (moles) are usually benign. However, any noticeable changes in size, shape, "
            "or color should be examined by a professional."
        )
    },
    "vasc": {
        "name": "Vascular Lesion",
        "description": (
            "Vascular lesions are abnormalities of the blood vessels. While often benign, they may require "
            "treatment if symptomatic or for cosmetic reasons."
        )
    }
}

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Loads, resizes, and preprocesses input skin image for AI model inference."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or cannot be read: " + image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

def setup_prediction_routes(app: Flask):

    @app.route('/predict', methods=['POST'])
    def prediction():
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        active_models = get_models()
        if not active_models:
            return jsonify({
                "error": "Model files not available on server.",
                "message": f"Please ensure trained .h5 model files exist in {MODEL_DIR} or set MODEL_ZIP_URL in .env"
            }), 503

        temp_dir = os.path.join(BASE_DIR, 'backend', 'uploads')
        os.makedirs(temp_dir, exist_ok=True)
        
        filename = secure_filename(image_file.filename)
        file_path = os.path.join(temp_dir, filename)

        try:
            image_file.save(file_path)
            preprocessed_image = load_and_preprocess_image(file_path, target_size=(224, 224))

            predictions = {}
            for name, model in active_models.items():
                pred = model.predict(preprocessed_image)[0]
                predictions[name] = pred

            avg_prediction = np.mean(list(predictions.values()), axis=0)
            predicted_index = int(np.argmax(avg_prediction))
            predicted_label = idx2class.get(predicted_index, str(predicted_index))
            friendly_info = USER_FRIENDLY_MAPPING.get(predicted_label, {})

            max_confidence = max(pred[predicted_index] for pred in predictions.values())
            confidence_percent = f"{int(round(max_confidence * 100))}"

            return jsonify({
                "message": "Image processed successfully",
                "filename": filename,
                "predicted_disease": predicted_label,
                "confidence": confidence_percent,
                "disease_details": friendly_info
            })
        except Exception as e:
            logger.exception("Prediction processing error")
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
