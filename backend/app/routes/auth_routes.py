from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token
from app.db_models.user_model import find_user_by_email
from flask_bcrypt import Bcrypt

auth_blueprint = Blueprint('auth', __name__)
bcrypt = Bcrypt()

@auth_blueprint.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    if not data or 'email' not in data or 'password' not in data:
        return jsonify({"error": "Invalid input"}), 400

    user = find_user_by_email(data["email"])
    if user and bcrypt.check_password_hash(user["password"], data["password"]):
        token = create_access_token(identity=user["email"])
        
        # Debug: Log the token to ensure it's being created
        print("Token created:", token)

        return jsonify({"token": token, "message": "Login successful"}), 200

    return jsonify({"error": "Invalid credentials"}), 401




@auth_blueprint.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    # Validate input
    if not data or not all(k in data for k in ("username", "email", "password")):
        return jsonify({"error": "Invalid input"}), 400

    email = data["email"]
    password = data["password"]
    username = data["username"]

    # Check if email is already registered
    if is_email_taken(email):
        return jsonify({"error": "Email already registered"}), 409

    # Create the user
    create_user(username, email, password)
    return jsonify({"message": "User registered successfully!"}), 201
