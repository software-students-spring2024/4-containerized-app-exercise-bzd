import io
import json
import os
from PIL import Image
import cv2
import torch
from torchvision import transforms, models
from pymongo import MongoClient
from torchvision.models import resnet18, ResNet18_Weights

# MongoDB connection setup
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client["image_classification"]
collection = db["predictions"]

def fetch_image_from_db():
    """Fetch the latest image from MongoDB that hasn't been processed."""
    document = collection.find_one({"processed": {"$exists": False}})
    if document:
        image_bytes = document['image']
        return image_bytes, document['_id']
    return None, None

def preprocess_image(image_bytes, target_size=(224, 224)):
    """Preprocess the image to fit the model requirements."""
    image = Image.open(io.BytesIO(image_bytes))
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def load_model():
    """Load a pre-trained ResNet18 model with updated parameter usage."""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    return model

def load_labels():
    """Load labels for image classification."""
    with open("machine-learning-client/imagenet_classes.json", "r", encoding="utf-8") as f:
        labels = json.load(f)
    return labels

def predict(model, image, labels):
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    class_id = predicted.item()  # Ensure this is an integer
    return labels[class_id]  # Access list directly with integer index

def update_prediction_in_db(doc_id, prediction):
    """Update the MongoDB document with the prediction result."""
    collection.update_one(
        {"_id": doc_id},
        {"$set": {"prediction": prediction, "processed": True}}
    )

def main():
    """Fetch an image from the DB, process it, predict using a model, and update the prediction."""
    image_bytes, doc_id = fetch_image_from_db()
    if image_bytes and doc_id:
        preprocessed_image = preprocess_image(image_bytes)
        model = load_model()
        labels = load_labels()
        predicted_label = predict(model, preprocessed_image, labels)
        update_prediction_in_db(doc_id, predicted_label)
        print(f"Predicted Label: {predicted_label}")
    else:
        print("No unprocessed images found in the database.")

if __name__ == "__main__":
    main()
