from src.topology import TopologyReconstructor
from shapely.geometry import Polygon

def test_topology_reconstruction():
    # Create a simple square of lines
    # (100, 100) to (200, 100) to (200, 200) to (100, 200) and back
    lines = [
        (100, 100, 200, 100),
        (200, 100, 200, 200),
        (200, 200, 100, 200),
        (100, 200, 100, 100)
    ]
    
    reconstructor = TopologyReconstructor(pixels_per_metre=100)
    topology = reconstructor.reconstruct(lines, [], [], (400, 400))
    
    assert 'footprint' in topology
    footprint = topology['footprint']
    
    # Area should be (100/100) * (100/100) = 1.0 m2
    assert abs(footprint.area - 1.0) < 0.01
    
    # Origin was (100, 200). 
    # (100, 200) in image -> (0, 0) in world
    # (200, 100) in image -> (1, 1) in world
    bounds = footprint.bounds # minx, miny, maxx, maxy
    assert bounds[0] == 0.0
    assert bounds[1] == 0.0
    assert bounds[2] == 1.0
    assert bounds[3] == 1.0
