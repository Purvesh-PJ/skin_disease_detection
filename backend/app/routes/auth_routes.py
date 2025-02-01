from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required
from flask_bcrypt import Bcrypt
from app.db_models.user_model import find_user_by_email, create_user
from functools import wraps
import jwt

# Initialize Bcrypt
bcrypt = Bcrypt()
# Initialize Blueprint
auth_blueprint = Blueprint('auth', __name__)
# Secret key for JWT decoding
auth_secret_key = "d9574c5c06e96b0e2ef7bbfeb3e3cfae5920ad5d3f1b1a9a6f2b60c08a1e5dbf"

# Login endpoint
@auth_blueprint.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    if not data or 'email' not in data or 'password' not in data:
        return jsonify({"error": "Invalid input"}), 400

    user = find_user_by_email(data["email"])
    if user and bcrypt.check_password_hash(user["password"], data["password"]):
        # Adjusted to extract roles
        token = jwt.encode({"email": user["email"], "username": user["username"], "roles": user["roles"]}, auth_secret_key, algorithm="HS256")
        return jsonify({"token": token, "message": "Login successful"}), 200
    return jsonify({"error": "Invalid credentials"}), 401

# Registration endpoint
def is_email_taken(email):
    user = find_user_by_email(email)
    return user is not None

# def create_user(username, email, password):
#     hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
#     create_user(username, email, hashed_password)

@auth_blueprint.route('/register', methods=['POST'])
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

# Token verification logic
def token_required(f):
    """Custom decorator to verify JWT tokens."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')

        if auth_header and " " in auth_header:
            token = auth_header.split(" ")[1]
        else:
            return jsonify({'error': 'Token is missing'}), 401

        try:
            payload = jwt.decode(token, auth_secret_key, algorithms=["HS256"])
            request.user = payload  # Attach payload to request for access in the route
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401

        return f(*args, **kwargs)
    return decorated

# Verify token endpoint
@auth_blueprint.route('/verify-token', methods=['GET'])
@token_required
def verify_token():
    # Access user information from the token
    user_info = request.user
    return jsonify({
        'message': 'Token is valid',
        'user': user_info
    }), 200