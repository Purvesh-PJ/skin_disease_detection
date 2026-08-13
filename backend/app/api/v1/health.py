"""
Health Check API Controller
---------------------------
Base endpoint for application health and status checks.
"""

from flask import Blueprint, jsonify

health_router = Blueprint('health', __name__)

@health_router.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online",
        "service": "Skin Disease Detection API",
        "version": "1.0.0"
    }), 200
