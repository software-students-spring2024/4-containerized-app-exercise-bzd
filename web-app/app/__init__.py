"""Initialize Flask application and its configurations."""

import os
from flask import Flask
from pymongo import MongoClient


def create_app():
    """factory function."""
    app = Flask(__name__)

    # MongoDB connection setup
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongo_uri)
    db = client["image_classification"]

    from app import routes  # pylint: disable=import-outside-toplevel

    routes.init_app(app, db)

    return app
