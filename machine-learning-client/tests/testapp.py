import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from camera import capture_image_from_camera, preprocess_image, load_model, predict, save_prediction, serialize_image
import torch

# Mocks
@pytest.fixture
def mock_image():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.return_value = MagicMock(spec=torch.nn.Module)
    return model

@pytest.fixture
def mock_labels():
    return {"0": "cat", "1": "dog"}

# Test camera capture function
def test_capture_image_from_camera(mock_image):
    with patch('cv2.VideoCapture') as mock_cap:
        mock_cap_instance = mock_cap.return_value
        mock_cap_instance.read.return_value = (True, mock_image)
        with patch('cv2.waitKey', side_effect=[ord(' '), 27]):  # Simulate space and escape keys
            captured_image = capture_image_from_camera()
            assert captured_image is not None
            assert captured_image.shape == (480, 640, 3)

# Test image preprocessing
def test_preprocess_image(mock_image):
    processed_image = preprocess_image(mock_image)
    assert processed_image.shape == (1, 3, 224, 224)  # Check the shape of the tensor

# Test model loading
def test_load_model(mock_model):
    with patch('torchvision.models.resnet18', mock_model):
        model = load_model()
        assert isinstance(model, torch.nn.Module)

# Test prediction function
def test_predict(mock_model, mock_labels):
    model = mock_model()
    model.eval.return_value = MagicMock()
    
    # Mock the outputs of the model to be a tensor, as expected by torch.max
    outputs = torch.tensor([[10.0, 1.0]])
    model.forward.return_value = outputs  # Use .forward to represent the model's prediction
    
    image_tensor = torch.rand((1, 3, 224, 224))
    
    with patch('torch.max', return_value=(torch.tensor([1]), torch.tensor([0]))):
        prediction = predict(model, image_tensor, mock_labels)
        assert prediction == "cat"


# Test image serialization
def test_serialize_image(mock_image):
    binary_image = serialize_image(mock_image)
    assert isinstance(binary_image, bytes)

# Test database interaction
def test_save_prediction(mock_image, mock_labels):
    with patch('pymongo.collection.Collection.insert_one') as mock_insert:
        save_prediction(mock_image, "cat")
        assert mock_insert.called

