from flask import Flask, render_template, request, redirect, url_for, jsonify
from pymongo import MongoClient
import os
import cv2
from PIL import Image
import io
import base64
import bson

app = Flask(__name__)

# Initialize MongoDB connection
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client["image_classification"]
collection = db["predictions"]

@app.route("/capture", methods=['POST'])
def capture_and_store():
    app.logger.info("Attempting to store uploaded image...")
    try:
        # Get the image data from the POST request
        image_data = request.json.get('image')
        # The image is sent as a base64 string, we need to decode it
        if image_data:
            image_data = base64.b64decode(image_data.split(',')[1])
            collection.insert_one({
                "image": bson.binary.Binary(image_data),
                "prediction": None
            })
            app.logger.info("Image stored successfully.")
            return jsonify({'status': 'success', 'message': 'Image stored successfully.'}), 200
        else:
            app.logger.warning("No image data provided.")
            return jsonify({'status': 'fail', 'message': 'No image data provided.'}), 400
    except Exception as e:
        app.logger.error("Failed to store image", exc_info=e)
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
