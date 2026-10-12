"""
Routes Package Initialization & Registry
----------------------------------------
Central entrypoint for registering all modular blueprints with the Flask app.
"""

from app.routes.health import health_router
from app.routes.predict import predict_router
from app.routes.auth import auth_router
from app.routes.history import history_router

def register_routes(app):
    """Registers all blueprints with the Flask application."""
    app.register_blueprint(health_router)
    app.register_blueprint(predict_router)
    app.register_blueprint(auth_router, url_prefix='/auth')
    app.register_blueprint(history_router, url_prefix='/history')
