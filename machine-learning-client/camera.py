"""This module is used for capturing, processing, and predicting images using a pre-trained model and storing results in MongoDB."""

import io
import json
from PIL import Image

import cv2
import torch
from torchvision import transforms, models
from pymongo import MongoClient
from torchvision.models import resnet18, ResNet18_Weights

import os

# MongoDB connection setup
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client["image_classification"]
collection = db["predictions"]


def capture_image_from_camera():
    """Capture an image from the camera when the spacebar is pressed or exit on 'q'."""
    cap = cv2.VideoCapture(0)
    print("Press the spacebar to capture an image, or 'q' to quit.")
    while True:
        ret, image = cap.read()
        cv2.imshow("Camera Stream", image)
        key = cv2.waitKey(1)
        if key == ord(" "):
            cap.release()
            cv2.destroyAllWindows()
            return image
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return None


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the image to fit the model requirements."""
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
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



def serialize_image(image):
    """Convert an OpenCV image to a binary format for storage."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image)
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def save_prediction(image, prediction):
    """Save the prediction and image to MongoDB."""
    serialized_image = serialize_image(image)
    document = {"prediction": prediction, "image": serialized_image}
    collection.insert_one(document)

def insert_sample_document():
    """Insert a sample document into the MongoDB collection."""
    image = serialize_image("machine-learning-client/ad4c4c52-b21a-41d6-ba9a-cd79b0dc6db4.jpg")
    sample_document = {
        "prediction": "cat",
        "image": image # This should be binary data in a real scenario
    }
    collection.insert_one(sample_document)
    print("Sample document inserted.")


def main():
    """Capture an image, process it, predict using a model, and save the prediction."""
    insert_sample_document
    camera_image = capture_image_from_camera()

    if camera_image is not None:
        preprocessed_image = preprocess_image(camera_image)
        model = load_model()
        labels = load_labels()
        predicted_label = predict(model, preprocessed_image, labels)
        save_prediction(camera_image, predicted_label)
        print(f"Predicted Label: {predicted_label}")
    else:
        print("Image capture was canceled.")


if __name__ == "__main__":
    main()
