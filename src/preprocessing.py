import cv2
import numpy as np
from loguru import logger

class ImagePreprocessor:
    def __init__(self, canonical_width=1200):
        self.canonical_width = canonical_width

    def process(self, image_path):
        """
        Full preprocessing pipeline for Stage 1.
        """
        logger.info(f"Preprocessing image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")

        # 1. Greyscale conversion
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Perspective Correction (Placeholder for now, returns original if corners not found)
        # In a real PoC, we'd implement findContours to detect the paper.
        warped = self.correct_perspective(grey)

        # 3. Adaptive Histogram Equalisation (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalised = clahe.apply(warped)

        # 4. Noise Reduction
        blurred = cv2.GaussianBlur(equalised, (5, 5), 0)

        # 5. Edge Detection (Canny with auto-tuning via Otsu)
        edges = self.auto_canny(blurred)

        # 6. Morphological Closing
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 7. Resize to Canonical Width
        final_img = self.resize_to_canonical(closed)

        logger.info("Preprocessing complete.")
        return final_img

    def correct_perspective(self, grey):
        """
        Detects paper corners and warps to flat rectangular image.
        For PoC, if detection fails, return as-is.
        """
        # Simplified corner detection for PoC
        blur = cv2.GaussianBlur(grey, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return grey

        # Find largest contour (presumably the paper)
        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

        if len(approx) == 4:
            # Reorder points: TL, TR, BR, BL
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            (tl, tr, br, bl) = rect
            width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            max_width = max(int(width_a), int(width_b))

            height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_height = max(int(height_a), int(height_b))

            dst = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            return cv2.warpPerspective(grey, M, (max_width, max_height))
        
        return grey

    def auto_canny(self, image, sigma=0.33):
        """
        Canny edge detection with auto-tuned thresholds based on image median.
        """
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(image, lower, upper)

    def resize_to_canonical(self, image):
        """
        Resize image to canonical width while preserving aspect ratio.
        """
        (h, w) = image.shape[:2]
        r = self.canonical_width / float(w)
        dim = (self.canonical_width, int(h * r))
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
