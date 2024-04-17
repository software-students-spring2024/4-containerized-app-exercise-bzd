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
        document = collection.find_one({"processed": {"$exists": False}})
        if document:
            image_bytes = document['image']
            return image_bytes, document['_id']
        return None, None
    except Exception as e:
        logging.error("Failed to fetch image from database: %s", e)
        return None, None

def preprocess_image(image_bytes, target_size=(224, 224)):
    """Preprocess the image to fit the model requirements."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image)
        image = image.unsqueeze(0)
        return image
    except Exception as e:
        logging.error("Failed to preprocess image: %s", e)
        return None

def load_model():
    """Load a pre-trained ResNet18 model with updated parameter usage."""
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.eval()
        return model
    except Exception as e:
        logging.error("Failed to load model: %s", e)
        return None

def load_labels():
    """Load labels for image classification."""
    try:
        with open("machine-learning-client/imagenet_classes.json", "r", encoding="utf-8") as f:
            labels = json.load(f)
        return labels
    except Exception as e:
        logging.error("Failed to load labels: %s", e)
        return None

def predict(model, image, labels):
    try:
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_id = predicted.item()  # Ensure this is an integer
        return labels[class_id]  # Access list directly with integer index
    except Exception as e:
        logging.error("Failed to perform prediction: %s", e)
        return None

def update_prediction_in_db(doc_id, prediction):
    """Update the MongoDB document with the prediction result."""
    try:
        collection.update_one(
            {"_id": doc_id},
            {"$set": {"prediction": prediction, "processed": True}}
        )
    except Exception as e:
        logging.error("Failed to update the database: %s", e)

import time

def main():
    """Run a continuous loop checking for images to process."""
    while True:
        image_bytes, doc_id = fetch_image_from_db()
        if image_bytes and doc_id:
            preprocessed_image = preprocess_image(image_bytes)
            if preprocessed_image is not None:
                model = load_model()
                if model is not None:
                    labels = load_labels()
                    if labels is not None:
                        predicted_label = predict(model, preprocessed_image, labels)
                        if predicted_label is not None:
                            update_prediction_in_db(doc_id, predicted_label)
                            logging.info(f"Predicted Label: {predicted_label}")
                        else:
                            logging.error("Prediction failed")
                    else:
                        logging.error("Label loading failed")
                else:
                    logging.error("Model loading failed")
            else:
                logging.error("Image preprocessing failed")
        else:
            logging.info("No unprocessed images found in the database. Waiting for new images.")
            time.sleep(10)  # Wait for 10 seconds before checking again

if __name__ == "__main__":
    main()