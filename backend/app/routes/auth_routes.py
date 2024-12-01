# backend/app/auth/auth_routes.py
from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token
from app.db_models.user_model import create_user, find_user_by_email, is_email_taken
from flask_bcrypt import Bcrypt

auth_blueprint = Blueprint('auth', __name__)
bcrypt = Bcrypt()

@auth_blueprint.route('/login', methods=['POST'])
def login():
    data = request.json
    user = find_user_by_email(data["email"])

    if user and bcrypt.check_password_hash(user["password"], data["password"]):
        token = create_access_token(identity=user["email"])
        return jsonify({"token": token, "message": "Login successful"})
    return jsonify({"error": "Invalid credentials"}), 401


@auth_blueprint.route('/register', methods=['POST'])
def register():
    data = request.json
    if is_email_taken(data["email"]):
        return jsonify({"error": "Email already registered"}), 409

    new_user = create_user(data["username"], data["email"], data["password"])
    return jsonify({"message": "User registered successfully!"})
