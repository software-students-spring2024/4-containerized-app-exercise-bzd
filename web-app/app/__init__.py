from flask import Flask, render_template
from pymongo import MongoClient
import os

# Setup Flask app
app = Flask(__name__)

# Setup MongoDB connection
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client["image_classification"]  # Ensure you specify your actual database name


def insert_sample_document():
    """Insert a sample document into the MongoDB collection."""
    image = "web-app/ad4c4c52-b21a-41d6-ba9a-cd79b0dc6db4.jpg"
    
    sample_document = {
        "prediction": "cat",
        "image": image,  # This should be binary data in a real scenario
    }
    db["prediction"].insert_one(sample_document)
    print("Sample document inserted.")

def init_app(app, db):
    """Pass app and db into routes."""

    @app.route("/")
    def index():
        """Render index page with results from the database."""
        try:
            results = db["prediction"].find()
            # Ensure that results are cast to list if necessary, or handled appropriately
            return render_template("index.html", results=list(results))
        except Exception as e:
            app.logger.error("Failed to fetch results from database", exc_info=e)
            # Handle errors appropriately, perhaps render an error page or a message
            return "Error fetching data", 500

# Initialize the app with routes
init_app(app, db)
insert_sample_document

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

