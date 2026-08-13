"""
User Database Model & Data Access Operations
--------------------------------------------
Interacts with MongoDB users collection for account management.
"""

from pymongo import MongoClient
from flask_bcrypt import Bcrypt
from app.core.config import config

mongo_client = MongoClient(config.MONGO_DB_URI)
db = mongo_client[config.MONGO_DB_NAME]
users_collection = db["users"]

bcrypt = Bcrypt()

def create_user(username, email, password):
    """Creates a new user with a hashed password and stores it in MongoDB."""
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    user = {
        "username": username,
        "email": email,
        "password": hashed_password,
        "roles": ["user"]
    }
    users_collection.insert_one(user)
    return user

def find_user_by_email(email):
    """Finds a user in MongoDB by email."""
    return users_collection.find_one({"email": email})

def is_email_taken(email):
    """Checks if email already exists in MongoDB."""
    return users_collection.find_one({"email": email}) is not None
