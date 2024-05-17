import io
import json
import os
from PIL import Image
import torch
from torchvision import transforms, models
from pymongo import MongoClient
import logging
import time

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MongoDB connection setup
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client["image_classification"]
collection = db["predictions"]

def fetch_image_from_db():
    """Fetch the latest image from MongoDB that hasn't been processed."""
    try:
        # Attempt to find the first document where 'processed' field is False
        document = collection.find_one({"processed": False})
        if document:
            # If a document is found, return the image and its document ID
            logging.info("Image fetched for processing.")
            logging.info(f"Fetched document: {document}")
            return document['image'], document['_id']
        else:
            # If no unprocessed image is found, log this event and return None
            logging.info("No unprocessed images found in the database.")
            return None, None
    except Exception as e:
        # Log any exceptions that occur during the fetch
        logging.error("Failed to fetch image from database: %s", e)
        return None, None

def preprocess_image(image_bytes, target_size=(224, 224)):
    """Preprocess the image to fit the model requirements."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        logging.error("Failed to preprocess image: %s", e)
        return None

# Load the pre-trained ResNet-18 model and set it to evaluation mode
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

def load_labels():
    """Load labels for image classification."""
    try:
        with open("imagenet_classes.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error("Failed to load labels: %s", e)
        return None

labels = load_labels()

def predict(model, image):
    """Perform prediction using the preloaded model and image."""
    try:
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return labels[predicted.item()]
    except Exception as e:
        logging.error("Failed to perform prediction: %s", e)
        return None

def update_prediction_in_db(doc_id, prediction):
    """Update the MongoDB document with the prediction result."""
    try:
        result = collection.update_one({"_id": doc_id}, {"$set": {"prediction": prediction, "processed": True}})
        logging.info(f"Update result: {result.modified_count} document(s) updated.")
    except Exception as e:
        logging.error("Failed to update the database: %s", e)

def main():
    try:
        while True:
            image_bytes, doc_id = fetch_image_from_db()
            if image_bytes and doc_id:
                preprocessed_image = preprocess_image(image_bytes)
                if preprocessed_image is not None:
                    predicted_label = predict(model, preprocessed_image)
                    if predicted_label is not None:
                        update_prediction_in_db(doc_id, predicted_label)
                        logging.info(f"Predicted Label: {predicted_label}")
                    else:
                        logging.error("Prediction failed")
                else:
                    logging.error("Image preprocessing failed")
            else:
                logging.info("No unprocessed images found in the database. Waiting for new images.")
                time.sleep(10)
    except KeyboardInterrupt:
        logging.info("Shutdown requested, exiting...")
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)
        raise

if __name__ == "__main__":
    main()
