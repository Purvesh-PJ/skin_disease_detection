"""
Skin Disease Detection API
--------------------------
Lightweight AI inference service with modular router registration,
centralized settings, and structured error handling.
"""

import os
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from app.core.config import config
from app.core.exceptions import BaseDomainException
from app.routes import register_routes

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
    """
    app = Flask(__name__)
    
    configure_app(app, testing)
    setup_cors(app)
    register_routes(app)
    register_error_handlers(app)
    register_request_handlers(app)

    # Start non-blocking model warmup (Hugging Face sync & load) in background
    if not testing:
        from app.services.prediction_service import start_model_warmup
        start_model_warmup()
    
    return app

def configure_app(app, testing=False):
    """Configure the Flask application with settings from AppConfig"""
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
    app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
    
    if testing:
        app.config['TESTING'] = True

def setup_cors(app):
    """Configure Cross-Origin Resource Sharing with full wildcard and header support"""
    CORS(
        app,
        resources={r"/*": {"origins": "*"}},
        supports_credentials=False,
        allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept"],
        methods=["GET", "POST", "OPTIONS"]
    )
    logger.info("CORS configured with open cross-origin access for web clients")

def register_error_handlers(app):
    """Register domain and HTTP error handlers"""
    @app.errorhandler(BaseDomainException)
    def handle_domain_exception(error):
        logger.warning(f"Domain Error [{error.status_code}]: {error.message}")
        return jsonify({'status': 'error', 'message': error.message}), error.status_code

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

# Create WSGI application object for Gunicorn / Render
app = create_app()

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    
    logger.info(f"Starting AI inference service on {host}:{port} (Debug: {debug_mode})")
    app.run(debug=debug_mode, host=host, port=port)
