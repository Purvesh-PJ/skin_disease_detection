from flask_jwt_extended import JWTManager, verify_jwt_in_request, get_jwt_identity
from functools import wraps

jwt = JWTManager()

def authorize(roles=[]):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            verify_jwt_in_request()
            user = users_collection.find_one({"email": get_jwt_identity()})
            if user and (not roles or set(roles) & set(user["roles"])):
                return fn(*args, **kwargs)
            return {"error": "Unauthorized access"}, 403
        return wrapper
    return decorator
