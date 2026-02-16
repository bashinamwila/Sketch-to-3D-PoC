import cadquery as cq
from loguru import logger

class ModelBuilder:
    def __init__(self, wall_height=3.0, wall_thickness=0.2):
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness

    def build(self, topology):
        """
        Generate CadQuery model from topology.
        """
        logger.info("Building 3D model in CadQuery.")
        
        if not topology or 'footprint' not in topology:
            logger.warning("Empty topology, cannot build model.")
            return None

        footprint = topology['footprint']
        exterior_coords = list(footprint.exterior.coords)

        # 1. Extrude Footprint
        # CadQuery uses (x, y) for the workplane
        model = (
            cq.Workplane("XY")
            .polyline(exterior_coords)
            .close()
            .extrude(self.wall_height)
        )

        # 2. Shell to create walls
        # Negative thickness shells inwards
        model = model.shell(-self.wall_thickness)

        # 3. Add interior walls
        for room in topology.get('rooms', []):
            # This is simplified: in reality we'd extrude the shared edges
            pass

        # 4. Add Roof (Simplified Gable)
        model = self.add_gable_roof(model, footprint)

        logger.info("3D model generation complete.")
        return model

    def add_gable_roof(self, model, footprint):
        """
        Add a simple gable roof on top of the model.
        """
        minx, miny, maxx, maxy = footprint.bounds
        midx = (minx + maxx) / 2
        ridge_height = self.wall_height + 1.5
        
        # Create a triangular prism for the roof
        # We'll define it on the XZ plane and extrude along Y
        roof = (
            cq.Workplane("XZ")
            .workplane(offset=miny)
            .polyline([(minx, self.wall_height), (midx, ridge_height), (maxx, self.wall_height)])
            .close()
            .extrude(maxy - miny)
        )
        
        return model.union(roof)

    def export(self, model, filename):
        """
        Export model to STEP/STL.
        """
        if model:
            cq.exporters.export(model, filename)
            logger.info(f"Model exported to {filename}")
