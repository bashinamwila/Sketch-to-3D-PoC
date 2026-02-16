import cadquery as cq
from loguru import logger
import numpy as np

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
        
        # Use height from topology if in perspective mode
        wall_h = topology.get('height', self.wall_height)

        # 1. Extrude Footprint
        model = (
            cq.Workplane("XY")
            .polyline(exterior_coords)
            .close()
            .extrude(wall_h)
        )

        # 2. Shell to create walls
        model = model.shell(-self.wall_thickness)

        # 4. Add Roof
        sub_footprints = topology.get('sub_footprints', [])
        
        if len(sub_footprints) > 1:
            logger.info(f"Composite footprint detected ({len(sub_footprints)} parts).")
            for part in sub_footprints:
                poly = part['poly']
                rtype = part['roof_type']
                if rtype == 'gable':
                    model = self.add_gable_roof(model, poly, wall_h)
                else:
                    model = self.add_medial_hip_roof(model, poly, wall_h)
        else:
            # Simple shape
            part = sub_footprints[0] if sub_footprints else {'poly': footprint, 'roof_type': 'hip'}
            if part.get('roof_type') == 'gable':
                model = self.add_gable_roof(model, part['poly'], wall_h)
            else:
                model = self.add_medial_hip_roof(model, part['poly'], wall_h)

        logger.info("3D model generation complete.")
        return model

    def add_medial_hip_roof(self, model, footprint, wall_h):
        """
        Create a sharp hip roof by lofting to a ridge line.
        """
        minx, miny, maxx, maxy = footprint.bounds
        width = maxx - minx
        depth = maxy - miny
        
        short_dim = min(width, depth)
        long_dim = max(width, depth)
        ridge_h = wall_h + (short_dim * 0.4) 
        
        ridge_len = max(0.01, long_dim - short_dim)
        mid_x = (minx + maxx) / 2
        mid_y = (miny + maxy) / 2
        
        # Base wire
        base_wire = (
            cq.Workplane("XY")
            .workplane(offset=wall_h)
            .polyline(list(footprint.exterior.coords))
            .close()
            .wire()
            .val()
        )

        # Define Ridge Points
        if width > depth:
            p1, p2 = (mid_x - ridge_len/2, mid_y), (mid_x + ridge_len/2, mid_y)
        else:
            p1, p2 = (mid_x, mid_y - ridge_len/2), (mid_x, mid_y + ridge_len/2)

        try:
            # Create a ridge wire
            ridge_wire = (
                cq.Workplane("XY")
                .workplane(offset=ridge_h)
                .polyline([p1, p2])
                .wire()
                .val()
            )
            roof = cq.Workplane("XY").add(base_wire).add(ridge_wire).toPending().loft()
            return model.union(roof)
        except Exception:
            # Fallback to tiny rectangle
            peak_wire = (
                cq.Workplane("XY")
                .workplane(offset=ridge_h)
                .center(mid_x, mid_y)
                .rect(0.01, 0.01)
                .wire()
                .val()
            )
            roof = cq.Workplane("XY").add(base_wire).add(peak_wire).toPending().loft()
            return model.union(roof)

    def add_gable_roof(self, model, footprint, wall_h):
        """
        Add a gable roof (triangular end). 
        Extrudes a triangle along the length of the rectangle.
        """
        minx, miny, maxx, maxy = footprint.bounds
        width = maxx - minx
        depth = maxy - miny
        
        # Decide orientation based on which side is longer
        if width > depth:
            # Gable ends are on Left/Right, ridge is along X
            mid_y = (miny + maxy) / 2
            ridge_h = wall_h + (depth * 0.5)
            # Create triangle on YZ plane and extrude along X
            roof = (
                cq.Workplane("YZ")
                .workplane(offset=minx)
                .polyline([(miny, wall_h), (mid_y, ridge_h), (maxy, wall_h)])
                .close()
                .extrude(width)
            )
        else:
            # Gable ends are on Front/Back, ridge is along Y
            mid_x = (minx + maxx) / 2
            ridge_h = wall_h + (width * 0.5)
            # Create triangle on XZ plane and extrude along Y
            roof = (
                cq.Workplane("XZ")
                .workplane(offset=miny)
                .polyline([(minx, wall_h), (mid_x, ridge_h), (maxx, wall_h)])
                .close()
                .extrude(depth)
            )
        
        return model.union(roof)

    def export(self, model, filename):
        """
        Export model to STEP/STL.
        """
        if model:
            cq.exporters.export(model, filename)
            logger.info(f"Model exported to {filename}")
