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

# Try to import dotenv, but continue if not available
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file if it exists
    load_dotenv()
    print("Environment variables loaded from .env file")
except ImportError:
    print("python-dotenv package not found. Using default environment variables.")
    # You can install it with: pip install python-dotenv

# Import routes
from app.routes.home_routes import setup_home_routes
from app.routes.prediction_routes import setup_prediction_routes
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
    # Create Flask app instance
    app = Flask(__name__)
    
    # Load configuration based on environment
    configure_app(app, testing)
    
    # Setup CORS
    setup_cors(app)
    
    # Initialize JWT
    jwt = setup_jwt(app)
    
    # Register blueprints and routes
    register_blueprints(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register request handlers for logging
    register_request_handlers(app)
    
    return app

def configure_app(app, testing=False):
    """Configure the Flask application with appropriate settings"""
    # Basic configuration
    app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'FlaskSecretKey12345!')
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'd9574c5c06e96b0e2ef7bbfeb3e3cfae5920ad5d3f1b1a9a6f2b60c08a1e5dbf')
    
    # JWT configuration
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
    app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)
    
    # Upload configuration
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Testing configuration
    if testing:
        app.config['TESTING'] = True
        # Add any testing-specific configuration here

def setup_cors(app):
    """Configure Cross-Origin Resource Sharing"""
    # In production, you should restrict origins to your frontend domain
    origins = os.getenv('CORS_ORIGINS', '*')
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
    # Register authentication routes
    app.register_blueprint(auth_blueprint, url_prefix='/auth')
    
    # Register other routes
    setup_home_routes(app)
    setup_prediction_routes(app)
    
    logger.info("All blueprints and routes registered")

def register_error_handlers(app):
    """Register error handlers for the application"""
    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        """Handle all HTTP exceptions"""
        response = {
            'status': 'error',
            'message': error.description,
            'error': error.name
        }
        logger.error(f"HTTP Error: {error.code} - {error.name}")
        return jsonify(response), error.code
    
    @app.errorhandler(Exception)
    def handle_generic_exception(error):
        """Handle all unhandled exceptions"""
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
        """Log information about each request"""
        logger.debug(f"Request: {request.method} {request.path} - {request.remote_addr}")
    
    @app.after_request
    def log_response_info(response):
        """Log information about each response"""
        logger.debug(f"Response: {response.status}")
        return response

# Entry point for running the app
if __name__ == '__main__':
    app = create_app()
    debug_mode = os.getenv('FLASK_DEBUG', 'True').lower() in ('true', '1', 't')
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    
    logger.info(f"Starting application on {host}:{port} (Debug: {debug_mode})")
    app.run(debug=debug_mode, host=host, port=port)
