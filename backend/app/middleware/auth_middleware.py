"""
Authentication Middleware
-------------------------
Role-based authorization decorator for protecting API endpoints with JWT tokens.
"""

from functools import wraps
from flask import jsonify
from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity
from app.models.user import find_user_by_email

def authorize(roles=None):
    """
    Decorator for role-based endpoint access control.
    """
    if roles is None:
        roles = []

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                verify_jwt_in_request()
                identity = get_jwt_identity()
                user = find_user_by_email(identity)
                
                if user and (not roles or set(roles) & set(user.get("roles", []))):
                    return fn(*args, **kwargs)
                
                return jsonify({"error": "Forbidden: Insufficient permissions"}), 403
            except Exception as e:
                return jsonify({"error": "Unauthorized access", "details": str(e)}), 401
        return wrapper
    return decorator
