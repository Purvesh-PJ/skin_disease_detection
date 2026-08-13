"""
Prediction Routes (Controller)
------------------------------
Flask route controller for image upload and skin disease prediction endpoint.
"""

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import logging
from app.services.prediction_service import predict_skin_disease

logger = logging.getLogger(__name__)

prediction_blueprint = Blueprint('prediction', __name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

@prediction_blueprint.route('/predict', methods=['POST'])
def prediction():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_dir = os.path.join(BASE_DIR, 'backend', 'uploads')
    os.makedirs(temp_dir, exist_ok=True)

    filename = secure_filename(image_file.filename)
    file_path = os.path.join(temp_dir, filename)

    try:
        image_file.save(file_path)
        result = predict_skin_disease(file_path)
        result["message"] = "Image processed successfully"
        result["filename"] = filename
        return jsonify(result), 200
    except RuntimeError as e:
        logger.warning(f"Prediction service unavailable: {e}")
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.exception("Prediction processing error")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
