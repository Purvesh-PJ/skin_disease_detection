"""
Repositories Module Initialization
----------------------------------
Exposes user and history database access operations.
"""
from app.repositories import user_repository, history_repository

__all__ = ["user_repository", "history_repository"]
