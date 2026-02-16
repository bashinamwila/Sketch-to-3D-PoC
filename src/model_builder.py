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
        wall_h = topology.get('height', self.wall_height)
        
        # Calculate a global ridge height for consistency across parts
        # 40% of the main block's shortest dimension
        main_poly = next((p['poly'] for p in topology.get('sub_footprints', []) if p['label'] == 'main'), footprint)
        minx, miny, maxx, maxy = main_poly.bounds
        global_ridge_h = wall_h + (min(maxx-minx, maxy-miny) * 0.4)

        # 1. Extrude Footprint
        model = (
            cq.Workplane("XY")
            .polyline(list(footprint.exterior.coords))
            .close()
            .extrude(wall_h)
        )

        # 2. Shell to create walls
        model = model.shell(-self.wall_thickness)

        # 4. Add Roof Parts
        sub_footprints = topology.get('sub_footprints', [])
        
        if len(sub_footprints) > 1:
            logger.info(f"Composite footprint detected ({len(sub_footprints)} parts).")
            for part in sub_footprints:
                poly = part['poly']
                rtype = part['roof_type']
                if rtype == 'gable':
                    model = self.add_gable_roof(model, poly, wall_h, global_ridge_h)
                else:
                    model = self.add_medial_hip_roof(model, poly, wall_h, global_ridge_h)
        else:
            model = self.add_medial_hip_roof(model, footprint, wall_h, global_ridge_h)

        logger.info("3D model generation complete.")
        return model

    def add_medial_hip_roof(self, model, footprint, wall_h, ridge_h):
        """
        Create a sharp hip roof by lofting to a ridge line.
        """
        minx, miny, maxx, maxy = footprint.bounds
        width, depth = maxx - minx, maxy - miny
        mid_x, mid_y = (minx + maxx) / 2, (miny + maxy) / 2
        
        # Ridge length for a hip roof is (Long Dim - Short Dim)
        ridge_len = max(0.1, abs(width - depth))
        
        if width > depth:
            p1, p2 = (mid_x - ridge_len/2, mid_y), (mid_x + ridge_len/2, mid_y)
        else:
            p1, p2 = (mid_x, mid_y - ridge_len/2), (mid_x, mid_y + ridge_len/2)
            
        base_wire = cq.Workplane("XY", origin=(0,0,wall_h)).polyline(list(footprint.exterior.coords)).close().wire().val()
        
        try:
            ridge_wire = cq.Workplane("XY", origin=(0,0,ridge_h)).polyline([p1, p2]).wire().val()
            roof = cq.Workplane("XY").add(base_wire).add(ridge_wire).toPending().loft()
            return model.union(roof)
        except Exception:
            peak_wire = cq.Workplane("XY", origin=(mid_x, mid_y, ridge_h)).rect(0.01, 0.01).wire().val()
            roof = cq.Workplane("XY").add(base_wire).add(peak_wire).toPending().loft()
            return model.union(roof)

    def add_gable_roof(self, model, footprint, wall_h, ridge_h):
        """
        Add a gable roof by extruding a triangle.
        Ensures alignment by using explicit workplane origins.
        """
        minx, miny, maxx, maxy = footprint.bounds
        width, depth = maxx - minx, maxy - miny
        mid_x, mid_y = (minx + maxx) / 2, (miny + maxy) / 2
        
        try:
            if width > depth:
                # Ridge along X, Triangle on YZ
                # Origin at minx (shared wall), Triangle covers depth
                roof = (
                    cq.Workplane("YZ", origin=(minx, mid_y, wall_h))
                    .polyline([(-depth/2, 0), (0, ridge_h - wall_h), (depth/2, 0)])
                    .close()
                    .extrude(width)
                )
            else:
                # Ridge along Y, Triangle on XZ
                roof = (
                    cq.Workplane("XZ", origin=(mid_x, miny, wall_h))
                    .polyline([(-width/2, 0), (0, ridge_h - wall_h), (width/2, 0)])
                    .close()
                    .extrude(depth)
                )
            return model.union(roof)
        except Exception as e:
            logger.warning(f"Gable roof failed: {e}. Falling back to hip.")
            return self.add_medial_hip_roof(model, footprint, wall_h, ridge_h)

    def export(self, model, filename):
        if model:
            cq.exporters.export(model, filename)
            logger.info(f"Model exported to {filename}")
