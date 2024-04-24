from flask import Flask, render_template, request, redirect, url_for, jsonify
from pymongo import MongoClient
import os
import cv2
from PIL import Image
import io
import base64
import bson
import logging
app = Flask(__name__)

# Initialize MongoDB connection
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client["image_classification"]
collection = db["predictions"]

@app.route("/capture", methods=['POST'])
def capture_and_store():
    logging.info("Attempting to store uploaded image...")
    try:
        image_data = request.json.get('image')
        if not image_data:
            logging.warning("No image data provided.")
            return jsonify({'status': 'fail', 'message': 'No image data provided.'}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(image_data.split(',')[1])
        collection.insert_one({"image": bson.binary.Binary(image_data), "prediction": None})
        logging.info("Image stored successfully.")
        return jsonify({'status': 'success', 'message': 'Image stored successfully.'}), 200
    except Exception as e:
        logging.error("Failed to store image", exc_info=True)
        return jsonify({'status': 'error', 'message': 'Failed to store image.'}), 500

@app.route("/")
def index():
    """Render index page with results from the database."""
    try:
        cursor = db["predictions"].find().limit(50)
        decoded_results = [{
            "image": base64.b64encode(doc["image"]).decode('utf-8'),
            "prediction": doc["prediction"]
        } for doc in cursor if doc.get("image")]
        return render_template("index.html", results=decoded_results)
    except Exception as e:
        app.logger.error("Failed to fetch results from database", exc_info=e)
        return "Error fetching data", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
