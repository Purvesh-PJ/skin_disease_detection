"""
Skin Lesion Prediction API Controller
------------------------------------
API endpoint for processing skin image uploads, returning ensemble predictions,
and persisting scan history to MongoDB for authenticated users.
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
import os
import logging
from app.services.prediction_service import predict_skin_disease
from app.models.history import save_prediction
from app.core.exceptions import BaseDomainException

logger = logging.getLogger(__name__)

predict_router = Blueprint('predict', __name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

@predict_router.route('/predict', methods=['POST'])
@jwt_required(optional=True)
def handle_prediction():
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

        # Persist prediction in MongoDB if user is authenticated
        current_user_email = get_jwt_identity()
        if current_user_email:
            try:
                saved_doc = save_prediction(current_user_email, result)
                if saved_doc:
                    result["history_id"] = saved_doc.get("_id")
                    result["saved_to_db"] = True
            except Exception as db_err:
                logger.error(f"Failed to persist prediction history: {db_err}")
                result["saved_to_db"] = False
        else:
            result["saved_to_db"] = False

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

