import cv2
import numpy as np
from loguru import logger

class Renderer:
    def __init__(self, output_size=(1200, 900)):
        self.output_size = output_size

    def render_snapshot(self, model, directions):
        """
        Render a snapshot of the model aligned to sketch perspective.
        """
        logger.info("Calculating camera alignment from vanishing points.")
        
        vp_l = directions.get('vp_left')
        vp_r = directions.get('vp_right')
        
        azimuth = 45 # Default 3/4 view
        if vp_l and vp_r:
            # Estimate azimuth based on VP balance
            # If VP_L is further from center than VP_R, we see more of the right face
            center_x = self.output_size[0] / 2
            dist_l = abs(vp_l[0] - center_x)
            dist_r = abs(vp_r[0] - center_x)
            
            # Simple heuristic for azimuth adjustment
            ratio = dist_l / (dist_l + dist_r + 1e-6)
            azimuth = 10 + ratio * 70 # Range 10 to 80 degrees
            logger.info(f"Calculated camera azimuth: {azimuth:.1f} degrees.")

        # For PoC, we still use a placeholder image but now with alignment metadata
        dummy_render = np.full((self.output_size[1], self.output_size[0], 3), 255, dtype=np.uint8)
        cv2.putText(dummy_render, f"3D Render (Azimuth: {azimuth:.1f} deg)", (200, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2)
        
        return dummy_render

    def get_lineart(self, render_img):
        """
        Convert render to CAD-style lineart (Canny + Thinning).
        """
        logger.info("Extracting lineart from render.")
        grey = cv2.cvtColor(render_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grey, 50, 150)
        
        # Optional: Thinning
        # thinned = cv2.ximgproc.thinning(edges) 
        # (needs opencv-contrib-python)
        
        # Invert: white background, black lines
        lineart = cv2.bitwise_not(edges)
        return lineart
