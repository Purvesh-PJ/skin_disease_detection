"""
Domain Exceptions
-----------------
Custom exception hierarchy for clean error handling across services.
"""

class BaseDomainException(Exception):
    """Base exception for application domain errors."""
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

class ModelNotFoundError(BaseDomainException):
    """Raised when an AI model file or cloud download is unavailable."""
    def __init__(self, message="AI model not found or unavailable."):
        super().__init__(message, status_code=530)

class InvalidImageError(BaseDomainException):
    """Raised when an uploaded image is corrupted or invalid."""
    def __init__(self, message="Invalid or unreadable image file."):
        super().__init__(message, status_code=400)

class AuthenticationError(BaseDomainException):
    """Raised when authentication credentials or tokens are invalid."""
    def __init__(self, message="Invalid credentials or authorization token."):
        super().__init__(message, status_code=401)
