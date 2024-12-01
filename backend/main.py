from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required
from flask_cors import CORS
from app.routes.home_routes import setup_home_routes
from app.routes.prediction_routes import setup_prediction_routes
from app.routes.auth_routes import auth_blueprint  # Import the auth blueprint

app = Flask(__name__)  # Instance of the Flask app
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# Set the secret keys for Flask and JWT
app.config['SECRET_KEY'] = 'FlaskSecretKey12345!'  # Flask's secret key for general app security
app.config['JWT_SECRET_KEY'] = 'JwtSecretKey45678@'  # Secret key for JWT encoding/decoding

# Initialize JWTManager
jwt = JWTManager(app)

# Register routes
setup_home_routes(app)  # Home route (for a simple home page)
setup_prediction_routes(app)  # Prediction route (for handling predictions)
app.register_blueprint(auth_blueprint, url_prefix='/auth')  # Register auth routes with the /auth prefix

@app.route('/verify-token', methods=['GET'])
@jwt_required()  # Automatically validates the token
def verify_token():
    return jsonify({'message': 'Token is valid'}), 200

if __name__ == '__main__':
    app.run(debug=True)
