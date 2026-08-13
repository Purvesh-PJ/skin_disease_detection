"""
Centralized Router Registry
---------------------------
Registers all versioned API routers and blueprints with the Flask application.
"""

from app.api.v1.health import health_router
from app.api.v1.predict import predict_router
from app.api.v1.auth import auth_router

def register_routes(app):
    """Registers API routers with Flask app."""
    app.register_blueprint(health_router)
    app.register_blueprint(predict_router)
    app.register_blueprint(auth_router, url_prefix='/auth')
