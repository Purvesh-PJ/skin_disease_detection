"""
Authentication Business Service
-------------------------------
Handles user credential authentication, registration, demo auto-provisioning,
JWT token generation, and profile sanitation.
"""

from flask_jwt_extended import create_access_token
from flask_bcrypt import Bcrypt
from app.models.user import (
    find_user_by_email,
    create_user,
    is_email_taken,
    seed_demo_user,
    update_user_settings,
    get_clean_user_profile
)

bcrypt = Bcrypt()

class AuthService:
    """Service encapsulating user authentication and profile business logic."""

    @staticmethod
    def authenticate(email: str, password: str) -> dict:
        """
        Validates credentials and returns JWT token and cleaned user profile.
        Returns None if invalid credentials.
        """
        email = email.strip().lower()

        # Auto-seed demo account if demo credentials are used
        if email in ("demo@skindisease.ai", "recruiter.demo@skindisease.ai"):
            seed_demo_user()

        user = find_user_by_email(email)
        if user and bcrypt.check_password_hash(user["password"], password):
            token = create_access_token(identity=user["email"])
            return {
                "token": token,
                "user": get_clean_user_profile(user)
            }
        return None

    @staticmethod
    def login_demo() -> dict:
        """Provisions/seeds and logs in demo recruiter account."""
        demo_user = seed_demo_user()
        token = create_access_token(identity=demo_user["email"])
        return {
            "token": token,
            "user": get_clean_user_profile(demo_user),
            "is_demo": True
        }

    @staticmethod
    def register(username: str, email: str, password: str) -> dict:
        """
        Registers a new user and returns JWT token.
        Raises ValueError if email is already taken.
        """
        email = email.strip().lower()
        username = username.strip()

        if is_email_taken(email):
            raise ValueError("An account with this email already exists")

        new_user = create_user(username, email, password)
        token = create_access_token(identity=new_user["email"])
        return {
            "token": token,
            "user": get_clean_user_profile(new_user)
        }

    @staticmethod
    def get_profile(email: str) -> dict:
        """Retrieves sanitized user profile by email."""
        user = find_user_by_email(email)
        if not user:
            return None
        return get_clean_user_profile(user)

    @staticmethod
    def update_profile(email: str, settings: dict) -> dict:
        """Updates user profile settings and returns updated profile."""
        updated_user = update_user_settings(email, settings)
        if not updated_user:
            return None
        return get_clean_user_profile(updated_user)
