"""
Skin Disease Detection API
--------------------------
Main application entry point with improved organization, error handling,
configuration management, and security practices.
"""

import os
import logging
from datetime import timedelta
from flask import Flask, jsonify, request
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

# Load environment variables from root or current directory
try:
    from dotenv import load_dotenv, find_dotenv
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
        print(f"Environment variables loaded from: {dotenv_path}")
    else:
        load_dotenv()
except ImportError:
    print("python-dotenv package not found. Using system environment variables.")

# Import routes (Blueprints)
from app.routes.home_routes import home_blueprint
from app.routes.prediction_routes import prediction_blueprint
from app.routes.auth_routes import auth_blueprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_app(testing=False):
    """
    Application factory function to create and configure the Flask app
    
    Args:
        testing (bool): Flag to indicate if the app is being created for testing
        
    Returns:
        Flask: Configured Flask application
    """
    app = Flask(__name__)
    
    configure_app(app, testing)
    setup_cors(app)
    jwt = setup_jwt(app)
    register_blueprints(app)
    register_error_handlers(app)
    register_request_handlers(app)
    
    return app

def configure_app(app, testing=False):
    """Configure the Flask application with appropriate settings"""
    app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'FlaskSecretKey12345!')
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'd9574c5c06e96b0e2ef7bbfeb3e3cfae5920ad5d3f1b1a9a6f2b60c08a1e5dbf')
    
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
    app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)
    
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    if testing:
        app.config['TESTING'] = True

def setup_cors(app):
    """Configure Cross-Origin Resource Sharing"""
    cors_origins = os.getenv('CORS_ORIGINS', '*')
    if cors_origins != '*':
        origins = [origin.strip() for origin in cors_origins.split(',')]
    else:
        origins = '*'
    CORS(app, resources={r"/*": {"origins": origins}})
    logger.info(f"CORS configured with origins: {origins}")

def setup_jwt(app):
    """Initialize and configure JWT manager"""
    jwt = JWTManager(app)
    
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return jsonify({
            'status': 'error',
            'message': 'The token has expired',
            'error': 'token_expired'
        }), 401
    
    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        return jsonify({
            'status': 'error',
            'message': 'Signature verification failed',
            'error': 'invalid_token'
        }), 401
    
    @jwt.unauthorized_loader
    def missing_token_callback(error):
        return jsonify({
            'status': 'error',
            'message': 'Request does not contain an access token',
            'error': 'authorization_required'
        }), 401
    
    logger.info("JWT Manager configured")
    return jwt

def register_blueprints(app):
    """Register all blueprints and routes"""
    app.register_blueprint(auth_blueprint, url_prefix='/auth')
    app.register_blueprint(home_blueprint)
    app.register_blueprint(prediction_blueprint)
    logger.info("All blueprints and routes registered")

def register_error_handlers(app):
    """Register error handlers for the application"""
    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        response = {
            'status': 'error',
            'message': error.description,
            'error': error.name
        }
        logger.error(f"HTTP Error: {error.code} - {error.name}")
        return jsonify(response), error.code
    
    @app.errorhandler(Exception)
    def handle_generic_exception(error):
        response = {
            'status': 'error',
            'message': 'An unexpected error occurred',
            'error': 'internal_server_error'
        }
        logger.exception("Unhandled exception occurred")
        return jsonify(response), 500
    
    logger.info("Error handlers registered")

def register_request_handlers(app):
    """Register request handlers for logging"""
    @app.before_request
    def log_request_info():
        logger.debug(f"Request: {request.method} {request.path} - {request.remote_addr}")
    
    @app.after_request
    def log_response_info(response):
        logger.debug(f"Response: {response.status}")
        return response

# Create WSGI application object for Gunicorn / uWSGI / Render / Railway
app = create_app()

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'True').lower() in ('true', '1', 't')
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    
    logger.info(f"Starting application on {host}:{port} (Debug: {debug_mode})")
    app.run(debug=debug_mode, host=host, port=port)
