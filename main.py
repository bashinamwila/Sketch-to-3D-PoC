import sys
import os
from loguru import logger
from src.preprocessing import ImagePreprocessor
from src.line_extractor import LineExtractor
from src.curve_extractor import CurveExtractor
from src.topology import TopologyReconstructor
from src.model_builder import ModelBuilder
from src.renderer import Renderer

def run_pipeline(image_path, output_dir="output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("Starting Sketch-to-3D Pipeline")
    
    # Stage 1: Preprocessing
    preprocessor = ImagePreprocessor()
    edge_img = preprocessor.process(image_path)
    
    # Stage 2: Line Extraction
    line_extractor = LineExtractor()
    lines, vertices, vps = line_extractor.extract_lines(edge_img)
    
    # Stage 3: Curve Extraction
    curve_extractor = CurveExtractor()
    curves = curve_extractor.extract_curves(edge_img, lines)
    
    # Stage 4: Topology Reconstruction
    topology_recon = TopologyReconstructor()
    topology = topology_recon.reconstruct(lines, vertices, curves, edge_img.shape)
    
    # Stage 5: CadQuery Model Generation
    builder = ModelBuilder()
    model = builder.build(topology)
    
    if model:
        builder.export(model, os.path.join(output_dir, "building_model.step"))
        
        # Stage 6: Camera-Aligned Snapshot
        renderer = Renderer()
        render_img = renderer.render_snapshot(model, vps)
        lineart = renderer.get_lineart(render_img)
        
        import cv2
        cv2.imwrite(os.path.join(output_dir, "render.png"), render_img)
        cv2.imwrite(os.path.join(output_dir, "lineart.png"), lineart)
        
    logger.info("Pipeline execution finished.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
    else:
        run_pipeline(sys.argv[1])
