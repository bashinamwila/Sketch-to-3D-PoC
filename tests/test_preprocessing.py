import cv2
import numpy as np
import pytest
import os
from src.preprocessing import ImagePreprocessor

@pytest.fixture
def synthetic_sketch(tmp_path):
    """
    Creates a synthetic sketch image (white box on black background) for testing.
    """
    img_path = tmp_path / "test_sketch.png"
    # Create a 800x600 black image
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    # Draw a white rectangle (the "paper")
    pts = np.array([[100, 100], [700, 150], [650, 500], [150, 450]], np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))
    # Draw some "sketch lines" inside
    cv2.line(img, (200, 200), (600, 200), (0, 0, 0), 2)
    cv2.imwrite(str(img_path), img)
    return str(img_path)

def test_preprocessing_pipeline(synthetic_sketch):
    preprocessor = ImagePreprocessor(canonical_width=400) # smaller for faster test
    processed_img = preprocessor.process(synthetic_sketch)
    
    assert processed_img is not None
    assert processed_img.shape[1] == 400
    assert len(processed_img.shape) == 2 # binary image
    
    # Check if we have some edges detected
    assert np.any(processed_img > 0)

def test_auto_canny():
    preprocessor = ImagePreprocessor()
    # Create a simple gradient image
    img = np.linspace(0, 255, 10000).reshape(100, 100).astype(np.uint8)
    edges = preprocessor.auto_canny(img)
    assert edges.shape == (100, 100)

def test_resize():
    preprocessor = ImagePreprocessor(canonical_width=200)
    img = np.zeros((100, 400), dtype=np.uint8)
    resized = preprocessor.resize_to_canonical(img)
    assert resized.shape == (50, 200)
