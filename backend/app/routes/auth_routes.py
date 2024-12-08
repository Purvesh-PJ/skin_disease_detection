from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required
from flask_bcrypt import Bcrypt
from app.db_models.user_model import find_user_by_email
from functools import wraps
import jwt

# Initialize Bcrypt
bcrypt = Bcrypt()

# Initialize Blueprint
auth_blueprint = Blueprint('auth', __name__)


# Login endpoint
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


# Registration endpoint
def is_email_taken(email):
    # Dummy email-check logic (you'll need to implement this)
    return False


def create_user(username, email, password):
    # Dummy user creation logic (you'll need to implement this)
    print(f"Created user {username}, {email}")


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


# Token verification logic
auth_secret_key = "d9574c5c06e96b0e2ef7bbfeb3e3cfae5920ad5d3f1b1a9a6f2b60c08a1e5dbf"


def token_required(f):
    """Custom decorator to verify JWT tokens."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        print("Authorization header received:", auth_header)

        if auth_header and " " in auth_header:
            token = auth_header.split(" ")[1]
        else:
            print("No valid token found!")
            return jsonify({'error': 'Token is missing'}), 401

        try:
            payload = jwt.decode(token, auth_secret_key, algorithms=["HS256"])
            print("Decoded token payload:", payload)
        except jwt.ExpiredSignatureError:
            print("Token expired!")
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError as e:
            print(f"Invalid token: {str(e)}")
            return jsonify({'error': 'Invalid token'}), 401

        return f(*args, **kwargs)
    return decorated


# Verify token endpoint
@auth_blueprint.route('/verify-token', methods=['GET'])
@token_required
def verify_token():
    print("Token verification successful!")
    return jsonify({'message': 'Token is valid'}), 200

