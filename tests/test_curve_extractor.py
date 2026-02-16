import numpy as np
import cv2
from src.curve_extractor import CurveExtractor

def test_curve_extraction():
    # Create an image with a circle
    img = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(img, (200, 200), 50, 255, 1)
    
    # Empty lines list
    lines = []
    
    extractor = CurveExtractor()
    curves = extractor.extract_curves(img, lines)
    
    assert len(curves) >= 1
    assert curves[0]['type'] == 'ellipse'
    
    # Check center
    center = curves[0]['center']
    assert abs(center[0] - 200) < 5
    assert abs(center[1] - 200) < 5

def test_bspline_fit():
    # Create a wavy line
    img = np.zeros((400, 400), dtype=np.uint8)
    for x in range(100, 300):
        y = int(200 + 20 * np.sin(x / 10.0))
        img[y, x] = 255
        
    extractor = CurveExtractor()
    curves = extractor.extract_curves(img, [])
    
    assert len(curves) >= 1
    assert curves[0]['type'] in ['bspline', 'polyline']
