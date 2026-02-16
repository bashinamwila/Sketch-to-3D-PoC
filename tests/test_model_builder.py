from src.model_builder import ModelBuilder
from shapely.geometry import Polygon
import os

def test_model_building(tmp_path):
    # Create a simple square topology
    poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    topology = {'footprint': poly}
    
    builder = ModelBuilder(wall_height=3.0, wall_thickness=0.2)
    model = builder.build(topology)
    
    assert model is not None
    
    # Export test
    export_path = str(tmp_path / "test_model.step")
    builder.export(model, export_path)
    assert os.path.exists(export_path)
    assert os.path.getsize(export_path) > 0
