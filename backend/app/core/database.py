"""
Centralized MongoDB Database Connector & Connection Pool
-------------------------------------------------------
Singleton client and database reference to prevent multiple redundant connections.
"""

import logging
from pymongo import MongoClient
from app.core.config import config

logger = logging.getLogger(__name__)

_client = None
_db = None

def get_db():
    """Returns singleton MongoDB database instance with pooled connections."""
    global _client, _db
    if _db is None:
        try:
            _client = MongoClient(config.MONGO_DB_URI, serverSelectionTimeoutMS=5000)
            try:
                _db = _client.get_default_database()
                if _db is None:
                    _db = _client[config.MONGO_DB_NAME]
            except Exception:
                _db = _client[config.MONGO_DB_NAME]
            logger.info(f"MongoDB connection initialized successfully to '{config.MONGO_DB_NAME}'")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    return _db

def get_client():
    """Returns raw singleton MongoClient instance if needed for health checks."""
    global _client
    if _client is None:
        get_db()
    return _client
