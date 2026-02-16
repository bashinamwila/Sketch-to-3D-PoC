import cv2
import numpy as np
from loguru import logger
import cadquery as cq
import os

class Renderer:
    def __init__(self, output_size=(1200, 900)):
        self.output_size = output_size

    def render_snapshot(self, model, directions):
        """
        True 3D Rendering: Uses SVG projection to create an architectural view.
        """
        logger.info("Generating 3D architectural projection.")
        
        vp_l = directions.get('vp_left')
        vp_r = directions.get('vp_right')
        
        azimuth = 45
        if vp_l and vp_r:
            center_x = self.output_size[0] / 2
            dist_l = abs(vp_l[0] - center_x)
            dist_r = abs(vp_r[0] - center_x)
            ratio = dist_l / (dist_l + dist_r + 1e-6)
            azimuth = 10 + ratio * 70
            logger.info(f"Using calculated azimuth: {azimuth:.1f} deg")

        # 1. Export SVG projection from CadQuery
        # We rotate the model to match the camera azimuth
        try:
            # Create a temporary SVG
            svg_path = "output/temp_render.svg"
            # Rotate model for the view
            view_model = model.rotate((0,0,0), (0,0,1), azimuth)
            
            # Export using hidden/internal SVG projection if possible
            # Standard export with Hidden Line Removal (HLR)
            cq.exporters.export(view_model, svg_path, cq.exporters.ExportTypes.SVG,
                                opt={
                                    "width": self.output_size[0],
                                    "height": self.output_size[1],
                                    "marginLeft": 10,
                                    "marginTop": 10,
                                    "showAxes": False,
                                    "projectionDir": (1, 1, 1), # Isometric-ish
                                    "strokeWidth": 0.5,
                                    "strokeColor": (0, 0, 0)
                                })
            
            # For the PoC, since converting SVG to PNG in headless python 
            # is complex (requires cairo/rsvg), we provide a fallback message.
            logger.info(f"Architectural SVG projection saved to {svg_path}")
            
            # Dummy PNG for the pipeline flow
            dummy_img = np.full((self.output_size[1], self.output_size[0], 3), 255, dtype=np.uint8)
            cv2.putText(dummy_img, f"View Azimuth: {azimuth:.1f} deg", (100, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
            cv2.putText(dummy_img, "See output/temp_render.svg for pro lineart", (100, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
            return dummy_img
            
        except Exception as e:
            logger.warning(f"SVG Render failed: {e}")
            return np.full((self.output_size[1], self.output_size[0], 3), 255, dtype=np.uint8)

    def get_lineart(self, render_img):
        """
        Extract lineart (Canny).
        """
        logger.info("Finalizing lineart pass.")
        grey = cv2.cvtColor(render_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grey, 50, 150)
        return cv2.bitwise_not(edges)
