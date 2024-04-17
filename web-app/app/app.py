from flask import Flask, render_template, request, redirect, url_for
from pymongo import MongoClient
import os
import cv2
import base64
import io
from PIL import Image

# Setup Flask app
app = Flask(__name__)

# Setup MongoDB connection
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client["image_classification"]
collection = db["predictions"]

def capture_image():
    """Capture image using webcam and return it as a binary string."""
    cap = cv2.VideoCapture(0)  # '0' for default camera
    success, image = cap.read()
    if success:
        cv2.imwrite("captured.jpg", image)  # Save frame as JPEG file
        cap.release()  # Release the capture

        # Convert image to binary format
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr
    cap.release()
    return None

@app.route("/capture", methods=['GET'])
def capture_and_store():
    """Capture an image, store it in MongoDB, and redirect to index."""
    image_data = capture_image()
    if image_data:
        db["predictions"].insert_one({
            "image": image_data,
            "prediction": None  # Placeholder for prediction result
        })
    return redirect(url_for('index'))

@app.route("/")
def index():
    """Render index page with results from the database."""
    try:
        results = db["predictions"].find()
        # Decode images for display
        decoded_results = []
        for result in results:
            if result.get("image"):
                image = base64.b64encode(result["image"]).decode('utf-8')
                decoded_results.append({"image": image, "prediction": result["prediction"]})
        return render_template("index.html", results=decoded_results)
    except Exception as e:
        app.logger.error("Failed to fetch results from database", exc_info=e)
        return "Error fetching data", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
