from flask import Flask, jsonify
from flask_jwt_extended import JWTManager, jwt_required
from flask_cors import CORS
from app.routes.home_routes import setup_home_routes
from app.routes.prediction_routes import setup_prediction_routes
from app.routes.auth_routes import auth_blueprint  # Import the auth blueprint
# import logging

# Create a Flask app instance
app = Flask(__name__)  

# Configure Flask and JWT secret keys
app.config['SECRET_KEY'] = 'FlaskSecretKey12345!'  # Flask's general secret key
app.config['JWT_SECRET_KEY'] = 'd9574c5c06e96b0e2ef7bbfeb3e3cfae5920ad5d3f1b1a9a6f2b60c08a1e5dbf'  # Secret key for JWT
app.register_blueprint(auth_blueprint, url_prefix='/auth')  # Routes for authentication
# logging.basicConfig(level=logging.DEBUG)
# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})  

# Initialize JWT Manager
jwt = JWTManager(app)

# Register app routes
setup_home_routes(app)  # Routes for the home page
setup_prediction_routes(app)  # Routes for handling predictions


# Route to verify token validity
# @app.route('/verify-token', methods=['GET'])

# @jwt_required()  # Requires a valid JWT
# def verify_token():
#     return jsonify({'message': 'Token is valid'}), 200

# Entry point for running the app
if __name__ == '__main__':
    app.run(debug=True)  # Start the app in debug mode
