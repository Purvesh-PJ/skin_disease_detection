"""
Routes Package Initialization & Registry
----------------------------------------
Central entrypoint for registering public AI inference and health check routes.
"""

from app.routes.health import health_router
from app.routes.predict import predict_router

def register_routes(app):
    """Registers all blueprints with the Flask application."""
    app.register_blueprint(health_router)
    app.register_blueprint(predict_router)
