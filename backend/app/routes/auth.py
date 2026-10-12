"""
Authentication API Controller
-----------------------------
API endpoints for user registration, login, 1-click recruiter demo access,
token validation, and profile settings stored in MongoDB.
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.services.auth_service import AuthService

auth_router = Blueprint('auth', __name__)

@auth_router.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or 'email' not in data or 'password' not in data:
        return jsonify({"error": "Email and password are required"}), 400

    result = AuthService.authenticate(data["email"], data["password"])
    if result:
        return jsonify({
            "token": result["token"],
            "message": "Login successful",
            "user": result["user"]
        }), 200

    return jsonify({"error": "Invalid credentials. Please check your email/password."}), 401


@auth_router.route('/demo-login', methods=['POST'])
def demo_login():
    """1-Click Recruiter/Demo login endpoint with auto-provisioning."""
    result = AuthService.login_demo()
    return jsonify({
        "token": result["token"],
        "message": "Welcome Demo",
        "user": result["user"],
        "is_demo": True
    }), 200


@auth_router.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or not all(k in data for k in ("username", "email", "password")):
        return jsonify({"error": "Username, email, and password are required"}), 400

    try:
        result = AuthService.register(data["username"], data["email"], data["password"])
        return jsonify({
            "message": "User registered successfully!",
            "token": result["token"],
            "user": result["user"]
        }), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 409


@auth_router.route('/verify-token', methods=['GET'])
@jwt_required()
def verify_token():
    current_user_email = get_jwt_identity()
    user_profile = AuthService.get_profile(current_user_email)
    if not user_profile:
        return jsonify({"error": "User session not found"}), 404

    return jsonify({
        'message': 'Token is valid',
        'user': user_profile
    }), 200


@auth_router.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    current_user_email = get_jwt_identity()
    user_profile = AuthService.get_profile(current_user_email)
    if not user_profile:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"user": user_profile}), 200


@auth_router.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    current_user_email = get_jwt_identity()
    data = request.get_json()
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid profile payload"}), 400

    user_profile = AuthService.update_profile(current_user_email, data)
    if not user_profile:
        return jsonify({"error": "Failed to update profile settings"}), 400

    return jsonify({
        "message": "Profile settings saved successfully",
        "user": user_profile
    }), 200

