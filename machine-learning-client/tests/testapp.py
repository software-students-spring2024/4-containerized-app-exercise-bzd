"""
This module tests the functionalities of the camera module including image capture,
preprocessing, model predictions, and interaction with the database.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from camera import capture_image_from_camera, preprocess_image, load_model, predict, save_prediction, serialize_image

@pytest.fixture
def mock_image():
    """Fixture to provide a mock image array for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def mock_model():
    """Fixture to provide a mock model for testing."""
    model = MagicMock()
    model.return_value = MagicMock(spec=torch.nn.Module)
    return model

@pytest.fixture
def mock_labels():
    """Fixture to provide mock labels for testing."""
    return {"0": "cat", "1": "dog"}

def test_capture_image_from_camera(mock_image):
    """Test image capture functionality by simulating camera input and keypress."""
    with patch('cv2.VideoCapture') as mock_cap:
        mock_cap_instance = mock_cap.return_value
        mock_cap_instance.read.return_value = (True, mock_image)
        with patch('cv2.waitKey', side_effect=[ord(' '), 27]):  # Simulate space and escape keys
            captured_image = capture_image_from_camera()
            assert captured_image is not None
            assert captured_image.shape == (480, 640, 3)

def test_preprocess_image(mock_image):
    """Test image preprocessing to ensure it returns the correct tensor shape."""
    processed_image = preprocess_image(mock_image)
    assert processed_image.shape == (1, 3, 224, 224)

def test_load_model(mock_model):
    """Test model loading to ensure it returns a model instance."""
    with patch('torchvision.models.resnet18', mock_model):
        model = load_model()
        assert isinstance(model, torch.nn.Module)

def test_predict(mock_model, mock_labels):
    """Test the prediction function to ensure it correctly predicts using a mock model."""
    model = mock_model()
    model.eval.return_value = MagicMock()
    
    # Mock the outputs of the model to be a tensor, as expected by torch.max
    outputs = torch.tensor([[10.0, 1.0]])
    model.forward.return_value = outputs  # Use .forward to represent the model's prediction
    
    image_tensor = torch.rand((1, 3, 224, 224))
    
    with patch('torch.max', return_value=(torch.tensor([1]), torch.tensor([0]))):
        prediction = predict(model, image_tensor, mock_labels)
        assert prediction == "cat"

def test_serialize_image(mock_image):
    """Test image serialization to ensure it returns bytes."""
    binary_image = serialize_image(mock_image)
    assert isinstance(binary_image, bytes)

def test_save_prediction(mock_image, mock_labels):
    """Test database interaction to ensure the save operation is called."""
    with patch('pymongo.collection.Collection.insert_one') as mock_insert:
        save_prediction(mock_image, "cat")
        assert mock_insert.called
