"""
Skin Lesion Prediction API Controller
------------------------------------
Public API endpoint for processing skin image uploads and returning ensemble predictions.
Zero authentication friction: Open diagnostic endpoint for clinical AI evaluation.
"""

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import logging
from app.core.config import config
from app.services.prediction_service import process_prediction
from app.core.exceptions import BaseDomainException

logger = logging.getLogger(__name__)

predict_router = Blueprint('predict', __name__)

@predict_router.route('/predict', methods=['POST'])
def handle_prediction():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(image_file.filename)
    file_path = os.path.join(config.UPLOAD_FOLDER, filename)

    try:
        image_file.save(file_path)
        result = process_prediction(file_path, filename)
        return jsonify(result), 200
    except BaseDomainException as e:
        logger.warning(f"Domain Exception: {e.message}")
        return jsonify({"error": e.message}), e.status_code
    except Exception as e:
        logger.exception("Unexpected error in prediction handler")
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
