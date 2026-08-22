"""
Authentication API Controller
-----------------------------
API endpoints for user registration, login, 1-click recruiter demo access,
token validation, and profile settings stored in MongoDB.
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from flask_bcrypt import Bcrypt
from app.models.user import (
    find_user_by_email,
    create_user,
    is_email_taken,
    seed_demo_user,
    update_user_settings,
    get_clean_user_profile
)

bcrypt = Bcrypt()
auth_router = Blueprint('auth', __name__)

@auth_router.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    if not data or 'email' not in data or 'password' not in data:
        return jsonify({"error": "Email and password are required"}), 400

    email = data["email"].strip().lower()
    user = find_user_by_email(email)
    
    if user and bcrypt.check_password_hash(user["password"], data["password"]):
        access_token = create_access_token(identity=user["email"])
        return jsonify({
            "token": access_token,
            "message": "Login successful",
            "user": get_clean_user_profile(user)
        }), 200

    return jsonify({"error": "Invalid credentials. Please check your email/password."}), 401


@auth_router.route('/demo-login', methods=['POST'])
def demo_login():
    """1-Click Recruiter/Demo login endpoint with auto-provisioning."""
    demo_user = seed_demo_user()
    access_token = create_access_token(identity=demo_user["email"])
    
    return jsonify({
        "token": access_token,
        "message": "Welcome, Recruiter / Guest Evaluator!",
        "user": get_clean_user_profile(demo_user),
        "is_demo": True
    }), 200


@auth_router.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    if not data or not all(k in data for k in ("username", "email", "password")):
        return jsonify({"error": "Username, email, and password are required"}), 400

    email = data["email"].strip().lower()
    password = data["password"]
    username = data["username"].strip()

    if is_email_taken(email):
        return jsonify({"error": "An account with this email already exists"}), 409

    new_user = create_user(username, email, password)
    access_token = create_access_token(identity=new_user["email"])
    
    return jsonify({
        "message": "User registered successfully!",
        "token": access_token,
        "user": get_clean_user_profile(new_user)
    }), 201


@auth_router.route('/verify-token', methods=['GET'])
@jwt_required()
def verify_token():
    current_user_email = get_jwt_identity()
    user = find_user_by_email(current_user_email)
    
    if not user:
        return jsonify({"error": "User session not found"}), 404
        
    return jsonify({
        'message': 'Token is valid',
        'user': get_clean_user_profile(user)
    }), 200


@auth_router.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    current_user_email = get_jwt_identity()
    user = find_user_by_email(current_user_email)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"user": get_clean_user_profile(user)}), 200


@auth_router.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    current_user_email = get_jwt_identity()
    data = request.get_json()
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid profile payload"}), 400
    
    updated_user = update_user_settings(current_user_email, data)
    if not updated_user:
        return jsonify({"error": "Failed to update profile settings"}), 400
        
    return jsonify({
        "message": "Profile settings saved successfully",
        "user": get_clean_user_profile(updated_user)
    }), 200

