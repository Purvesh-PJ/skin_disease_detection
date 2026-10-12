"""
Models Package Initialization
-----------------------------
Exports typed Domain Entities (User, UserSettings, PredictionHistory).
"""

from app.models.user import User
from app.models.user_settings import UserSettings
from app.models.history import PredictionHistory

__all__ = ["User", "UserSettings", "PredictionHistory"]
