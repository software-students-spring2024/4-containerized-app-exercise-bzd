from flask import Flask, render_template, request, redirect, url_for
from pymongo import MongoClient
import os
import cv2
from PIL import Image
import io
import base64
import logging

app = Flask(__name__)

# Initialize MongoDB connection
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client["image_classification"]
collection = db["predictions"]

def capture_image():
    """Capture image using webcam and return it as a binary string."""
    cap = cv2.VideoCapture(0)
    try:
        success, image = cap.read()
        if success:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='JPEG')
            return img_byte_arr.getvalue()
    finally:
        cap.release()  # Ensure the resource is released in any case
    return None

@app.route("/capture", methods=['GET'])
def capture_and_store():
    app.logger.info("Attempting to capture image...")
    try:
        image_data = capture_image()
        if image_data:
            db["predictions"].insert_one({
                "image": image_data,
                "prediction": None
            })
            app.logger.info("Image captured and stored successfully.")
        else:
            app.logger.warning("No image data captured.")
    except Exception as e:
        app.logger.error("Failed to capture or store image", exc_info=e)
    return redirect(url_for('index'))

@app.route("/")
def index():
    """Render index page with results from the database."""
    try:
        cursor = db["predictions"].find().limit(50)  # Limit to 50 records for performance
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
