"""
User Database Model & Data Access Operations
--------------------------------------------
Interacts with MongoDB users collection for account management.
"""

from datetime import datetime
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
from app.core.config import config

mongo_client = MongoClient(config.MONGO_DB_URI)
try:
    db = mongo_client.get_default_database()
    if db is None:
        db = mongo_client[config.MONGO_DB_NAME]
except Exception:
    db = mongo_client[config.MONGO_DB_NAME]

users_collection = db["users"]
bcrypt = Bcrypt()

def get_clean_user_profile(user_doc):
    """Sanitizes user document to safe JSON dictionary without sensitive hashes."""
    if not user_doc:
        return None
    return {
        "username": user_doc.get("username", "User"),
        "email": user_doc.get("email"),
        "roles": user_doc.get("roles", ["user"]),
        "settings": user_doc.get("settings", {
            "full_name": user_doc.get("username", "User"),
            "role_title": "Healthcare Evaluator",
            "specialization": "Dermoscopy Analysis",
            "theme": "dark",
            "email_notifications": True
        }),
        "created_at": user_doc.get("created_at", datetime.utcnow().isoformat())
    }

def create_user(username, email, password):
    """Creates a new user with a hashed password and stores it in MongoDB."""
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    user = {
        "username": username,
        "email": email.strip().lower(),
        "password": hashed_password,
        "roles": ["user"],
        "settings": {
            "full_name": username,
            "role_title": "Clinical Evaluator",
            "specialization": "Dermatology AI",
            "theme": "dark",
            "email_notifications": True
        },
        "created_at": datetime.utcnow().isoformat()
    }
    users_collection.insert_one(user)
    return user

def find_user_by_email(email):
    """Finds a user in MongoDB by email."""
    if not email:
        return None
    return users_collection.find_one({"email": email.strip().lower()})

def is_email_taken(email):
    """Checks if email already exists in MongoDB."""
    if not email:
        return False
    return users_collection.find_one({"email": email.strip().lower()}) is not None

def update_user_settings(email, new_settings):
    """Updates user profile settings in MongoDB."""
    if not email or not isinstance(new_settings, dict):
        return None
    
    current_user = find_user_by_email(email)
    if not current_user:
        return None
        
    merged_settings = current_user.get("settings", {})
    merged_settings.update(new_settings)
    
    updates = {"settings": merged_settings}
    if "full_name" in new_settings and new_settings["full_name"]:
        updates["username"] = new_settings["full_name"]

    users_collection.update_one(
        {"email": email.strip().lower()},
        {"$set": updates}
    )
    return find_user_by_email(email)

def seed_demo_user():
    """Ensures a pre-configured Recruiter Demo Account exists in MongoDB."""
    demo_email = "recruiter.demo@skindisease.ai"
    user = find_user_by_email(demo_email)
    
    if not user:
        hashed_password = bcrypt.generate_password_hash("DemoRecruiter@2026").decode('utf-8')
        demo_user = {
            "username": "Recruiter Guest",
            "email": demo_email,
            "password": hashed_password,
            "roles": ["demo", "user"],
            "settings": {
                "full_name": "Tech Recruiter (Demo)",
                "role_title": "Senior AI/ML Technical Recruiter",
                "specialization": "Clinical AI Project Evaluation",
                "theme": "dark",
                "email_notifications": True
            },
            "created_at": datetime.utcnow().isoformat()
        }
        users_collection.insert_one(demo_user)
        return demo_user
    return user

