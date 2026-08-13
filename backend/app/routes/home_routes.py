"""
Home Routes (Controller)
------------------------
Flask route controller for base status endpoint.
"""

from flask import Blueprint, jsonify

home_blueprint = Blueprint('home', __name__)

@home_blueprint.route('/')
def home():
    return jsonify({
        "status": "online",
        "message": "Welcome to the Skin Disease Detection API!",
        "version": "1.0.0"
    }), 200