import numpy as np
import cv2
from src.renderer import Renderer

def test_lineart_extraction():
    renderer = Renderer()
    # Create a dummy white image with a black line
    img = np.full((900, 1200, 3), 255, dtype=np.uint8)
    cv2.line(img, (100, 100), (1100, 100), (0, 0, 0), 5)
    
    lineart = renderer.get_lineart(img)
    assert lineart.shape == (900, 1200)
    # Check if we have some black pixels (the line)
    # Inverted lineart: 0 is black, 255 is white
    assert np.any(lineart < 255)
