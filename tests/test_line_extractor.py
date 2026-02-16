import numpy as np
import cv2
from src.line_extractor import LineExtractor

def test_line_extraction():
    # Create a 400x400 image with two perpendicular lines
    img = np.zeros((400, 400), dtype=np.uint8)
    cv2.line(img, (50, 200), (350, 200), 255, 1) # Horizontal
    cv2.line(img, (200, 50), (200, 350), 255, 1) # Vertical
    
    extractor = LineExtractor(threshold=30, min_line_length=100)
    lines, vertices, _ = extractor.extract_lines(img)
    
    assert len(lines) >= 2
    assert len(vertices) >= 1
    
    # Check if we found the intersection near (200, 200)
    found_intersection = False
    for vx, vy in vertices:
        if abs(vx - 200) < 10 and abs(vy - 200) < 10:
            found_intersection = True
            break
    assert found_intersection

def test_angle_snapping():
    extractor = LineExtractor()
    # A line at 2 degrees should snap to 0
    line = (0, 0, 100, 3) # ~1.7 degrees
    snapped = extractor.snap_line(line)
    # Check if horizontal
    assert snapped[1] == snapped[3]
