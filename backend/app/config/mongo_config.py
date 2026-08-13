from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
import logging

logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/skin_disease_db")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "skin_disease_db")

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

try:
    # Attempt to get default database from URI, or fallback to MONGO_DB_NAME
    db = client.get_default_database()
    if db is None:
        db = client.get_database(MONGO_DB_NAME)
except Exception:
    db = client.get_database(MONGO_DB_NAME)

users_collection = db.get_collection("users")
