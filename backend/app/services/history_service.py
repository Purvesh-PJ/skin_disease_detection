"""
Prediction History Business Service
-----------------------------------
Handles retrieval, deletion, and cleanup of user prediction history records.
"""

from app.models.history import get_user_history, delete_history_item, clear_user_history

class HistoryService:
    """Service encapsulating user scan history business logic."""

    @staticmethod
    def list_history(user_email: str, limit: int = 50) -> list:
        """Retrieves history records for authenticated user."""
        return get_user_history(user_email, limit=limit)

    @staticmethod
    def delete_record(history_id: str, user_email: str) -> bool:
        """Deletes a specific history record belonging to user."""
        return delete_history_item(history_id, user_email)

    @staticmethod
    def clear_all(user_email: str) -> int:
        """Clears all scan history records for user and returns deleted count."""
        return clear_user_history(user_email)
