# backend/app/db_model/user_collection.py
from app.config.mongo_config import users_collection
from flask_bcrypt import Bcrypt

bcrypt = Bcrypt()

def create_user(username, email, password):
    """
    Creates a new user with a hashed password and stores it in the database.
    """
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    user = {
        "username": username,
        "email": email,
        "password": hashed_password,
        "roles": ["user"]  # Default roles, can be extended
    }
    # Insert the new user into the MongoDB collection
    users_collection.insert_one(user)
    return user

def find_user_by_email(email):
    """
    Finds a user in the database by email.
    """
    return users_collection.find_one({"email": email})

def is_email_taken(email):
    """
    Checks if the email is already taken by an existing user.
    """
    return users_collection.find_one({"email": email}) is not None


