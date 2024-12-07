from flask import request, jsonify
import jwt
from functools import wraps

app.config['JWT_SECRET_KEY'] = "d9574c5c06e96b0e2ef7bbfeb3e3cfae5920ad5d3f1b1a9a6f2b60c08a1e5dbf"

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization').split(" ")[1] if request.headers.get('Authorization') else None
        print(token)
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        try:
            jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/verify-token', methods=['GET'])

@token_required
def verify_token():
    return jsonify({'message': 'Token is valid'}), 200
