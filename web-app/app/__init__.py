from flask import Flask
from pymongo import MongoClient
import os

# Initialize the Flask application
app = Flask(__name__)

# MongoDB connection setup
mongo_uri = os.environ.get('DATABASE_URL', 'mongodb://localhost:27017/')
client = MongoClient(mongo_uri)
db = client['your_database_name']  # Replace with your actual database name

# Import the routes after the Flask app is created
from app import routes
