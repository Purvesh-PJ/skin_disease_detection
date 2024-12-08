from flask import Blueprint, request, jsonify
import jwt
from functools import wraps

auth_blueprint = Blueprint('auth', __name__)
app.config['JWT_SECRET_KEY'] = "d9574c5c06e96b0e2ef7bbfeb3e3cfae5920ad5d3f1b1a9a6f2b60c08a1e5dbf"

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        print("Authorization header received:", auth_header)  # Log the Authorization header

        if auth_header and " " in auth_header:
            token = auth_header.split(" ")[1]
        else:
            print("No valid Authorization header found!")  # Log missing/invalid header
            return jsonify({'error': 'Token is missing'}), 401

        print("Extracted token:", token)  # Log the extracted token

        try:
            payload = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=["HS256"])
            print("Decoded token payload:", payload)  # Log the decoded payload
        except jwt.ExpiredSignatureError:
            print("Token has expired!")  # Log token expiration
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError as e:
            print(f"Invalid token! Error: {str(e)}")  # Log invalid token error details
            return jsonify({'error': 'Invalid token'}), 401
        return f(*args, **kwargs)
    return decorated


@auth_blueprint.route('/verify-token', methods=['GET'])
@token_required
def verify_token():
    print("Token verification successful!")  # Log successful verification
    return jsonify({'message': 'Token is valid'}), 200
