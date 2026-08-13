"""
Authentication API Controller
-----------------------------
API endpoints for user registration, login, and token validation.
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from flask_bcrypt import Bcrypt
from app.models.user import find_user_by_email, create_user, is_email_taken

bcrypt = Bcrypt()
auth_router = Blueprint('auth', __name__)

@auth_router.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    if not data or 'email' not in data or 'password' not in data:
        return jsonify({"error": "Invalid input"}), 400

    user = find_user_by_email(data["email"])
    if user and bcrypt.check_password_hash(user["password"], data["password"]):
        access_token = create_access_token(identity=user["email"])
        return jsonify({
            "token": access_token,
            "message": "Login successful",
            "user": {
                "username": user.get("username"),
                "email": user.get("email"),
                "roles": user.get("roles", ["user"])
            }
        }), 200

    return jsonify({"error": "Invalid credentials"}), 401


@auth_router.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    if not data or not all(k in data for k in ("username", "email", "password")):
        return jsonify({"error": "Invalid input"}), 400

    email = data["email"]
    password = data["password"]
    username = data["username"]

    if is_email_taken(email):
        return jsonify({"error": "Email already registered"}), 409

    create_user(username, email, password)
    return jsonify({"message": "User registered successfully!"}), 201


@auth_router.route('/verify-token', methods=['GET'])
@jwt_required()
def verify_token():
    current_user_email = get_jwt_identity()
    user = find_user_by_email(current_user_email)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
        
    return jsonify({
        'message': 'Token is valid',
        'user': {
            'username': user.get('username'),
            'email': user.get('email'),
            'roles': user.get('roles', ['user'])
        }
    }), 200
