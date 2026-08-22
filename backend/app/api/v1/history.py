"""
Prediction History API Controller
---------------------------------
API endpoints for managing skin disease scan history stored in MongoDB.
"""

from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.models.history import get_user_history, delete_history_item, clear_user_history

history_router = Blueprint('history', __name__)

@history_router.route('', methods=['GET'])
@history_router.route('/', methods=['GET'])
@jwt_required()
def fetch_history():
    """Returns list of past scans for the authenticated user from MongoDB."""
    user_email = get_jwt_identity()
    records = get_user_history(user_email, limit=50)
    return jsonify({
        "status": "success",
        "count": len(records),
        "history": records
    }), 200

@history_router.route('/<history_id>', methods=['DELETE'])
@jwt_required()
def remove_history_item(history_id):
    """Deletes a single prediction record by ID."""
    user_email = get_jwt_identity()
    success = delete_history_item(history_id, user_email)
    if success:
        return jsonify({"message": "Scan record deleted successfully", "id": history_id}), 200
    return jsonify({"error": "Record not found or unauthorized"}), 404

@history_router.route('/clear', methods=['DELETE'])
@jwt_required()
def remove_all_history():
    """Clears all scan history for the authenticated user."""
    user_email = get_jwt_identity()
    deleted_count = clear_user_history(user_email)
    return jsonify({
        "message": f"Successfully cleared {deleted_count} scan records",
        "deleted_count": deleted_count
    }), 200
