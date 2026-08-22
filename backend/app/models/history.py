"""
Prediction History Database Model & Data Access Operations
----------------------------------------------------------
Interacts with MongoDB 'prediction_history' collection for storing,
retrieving, and deleting skin lesion prediction records per user.
"""

from datetime import datetime
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
from app.core.config import config

mongo_client = MongoClient(config.MONGO_DB_URI)
try:
    db = mongo_client.get_default_database()
    if db is None:
        db = mongo_client[config.MONGO_DB_NAME]
except Exception:
    db = mongo_client[config.MONGO_DB_NAME]

history_collection = db["prediction_history"]

def save_prediction(user_email: str, prediction_data: dict) -> dict:
    """Stores an image prediction result in MongoDB linked to user email."""
    if not user_email or not prediction_data:
        return None

    created_iso = datetime.utcnow().isoformat()
    record = {
        "user_email": user_email.strip().lower(),
        "predicted_disease": prediction_data.get("predicted_disease"),
        "confidence": prediction_data.get("confidence", "0"),
        "disease_details": prediction_data.get("disease_details", {}),
        "filename": prediction_data.get("filename", "unknown_lesion.jpg"),
        "created_at": created_iso
    }

    result = history_collection.insert_one(record)
    record["_id"] = str(result.inserted_id)
    return record

def get_user_history(user_email: str, limit: int = 50) -> list:
    """Retrieves chronological past prediction scans for a given user."""
    if not user_email:
        return []

    cursor = history_collection.find(
        {"user_email": user_email.strip().lower()}
    ).sort("created_at", DESCENDING).limit(limit)

    records = []
    for item in cursor:
        item["_id"] = str(item["_id"])
        records.append(item)
    return records

def delete_history_item(history_id: str, user_email: str) -> bool:
    """Deletes a specific prediction history record for a user."""
    if not history_id or not user_email:
        return False
    try:
        res = history_collection.delete_one({
            "_id": ObjectId(history_id),
            "user_email": user_email.strip().lower()
        })
        return res.deleted_count > 0
    except Exception:
        return False

def clear_user_history(user_email: str) -> int:
    """Deletes all history records for a given user."""
    if not user_email:
        return 0
    res = history_collection.delete_many({
        "user_email": user_email.strip().lower()
    })
    return res.deleted_count
