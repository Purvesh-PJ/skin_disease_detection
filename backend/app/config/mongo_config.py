from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/skin_disease_db")
client = MongoClient(MONGO_URI)
db = client.get_database("skin_disease_db")
users_collection = db.get_collection("users")
