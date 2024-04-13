import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import json

def capture_image_from_camera():
    cap = cv2.VideoCapture(0)  
    print("Press the spacebar to capture an image, or 'q' to quit.")
    while True:
        ret, image = cap.read()  
        cv2.imshow('Camera Stream', image)  
        key = cv2.waitKey(1)
        if key == ord(' '):  
            cap.release()  
            cv2.destroyAllWindows()  
            return image
        elif key == ord('q'):  
            break
    cap.release()  
    cv2.destroyAllWindows()  
    return None

def preprocess_image(image, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    image = image.unsqueeze(0)  
    return image

def load_model():
    model = models.resnet18(pretrained=True)
    model.eval()  
    return model

def load_labels():
    with open("imagenet_classes.json", "r") as f:
        labels = json.load(f)
    return labels

def predict(model, image, labels):
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    class_id = predicted.item()
    return labels[str(class_id)]  

def main():
    camera_image = capture_image_from_camera()
    if camera_image is not None:
        preprocessed_image = preprocess_image(camera_image)
        model = load_model()
        labels = load_labels()
        predicted_label = predict(model, preprocessed_image, labels)
        print(f'Predicted Label: {predicted_label}')
    else:
        print("Image capture was canceled.")

if __name__ == '__main__':
    main()
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import json

def capture_image_from_camera():
    cap = cv2.VideoCapture(0)  
    print("Press the spacebar to capture an image, or 'q' to quit.")
    while True:
        ret, image = cap.read()  
        cv2.imshow('Camera Stream', image)  
        key = cv2.waitKey(1)
        if key == ord(' '):  
            cap.release()  
            cv2.destroyAllWindows()  
            return image
        elif key == ord('q'):  
            break
    cap.release()  
    cv2.destroyAllWindows()  
    return None

def preprocess_image(image, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    image = image.unsqueeze(0)  
    return image

def load_model():
    model = models.resnet18(pretrained=True)
    model.eval()  
    return model

def load_labels():
    with open("imagenet_classes.json", "r") as f:
        labels = json.load(f)
    return labels

def predict(model, image, labels):
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    class_id = predicted.item()
    return labels[str(class_id)]  

def main():
    camera_image = capture_image_from_camera()
    if camera_image is not None:
        preprocessed_image = preprocess_image(camera_image)
        model = load_model()
        labels = load_labels()
        predicted_label = predict(model, preprocessed_image, labels)
        print(f'Predicted Label: {predicted_label}')
    else:
        print("Image capture was canceled.")

if __name__ == '__main__':
    main()
