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
        pitch = topology.get('pitch', 0.4)

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
            # Calculate a global ridge height reference for composite parts
            main_part = next((p for p in sub_footprints if p['label'] == 'main'), sub_footprints[0])
            mbx = main_part['poly'].bounds
            global_ridge_h = wall_h + (min(mbx[2]-mbx[0], mbx[3]-mbx[1]) * pitch)

            for part in sub_footprints:
                poly = part['poly']
                rtype = part['roof_type']
                if rtype == 'gable':
                    model = self.add_gable_roof(model, poly, wall_h, global_ridge_h)
                else:
                    model = self.add_medial_hip_roof(model, poly, wall_h, pitch, global_ridge_h)
        else:
            # Simple shape
            part = sub_footprints[0] if sub_footprints else {'poly': footprint, 'roof_type': 'hip'}
            mbx = part['poly'].bounds
            ridge_h = wall_h + (min(mbx[2]-mbx[0], mbx[3]-mbx[1]) * pitch)
            if part.get('roof_type') == 'gable':
                model = self.add_gable_roof(model, part['poly'], wall_h, ridge_h)
            else:
                model = self.add_medial_hip_roof(model, part['poly'], wall_h, pitch, ridge_h)

        logger.info("3D model generation complete.")
        return model

    def add_medial_hip_roof(self, model, footprint, wall_h, pitch, ridge_h):
        """
        Create a sharp hip roof by lofting to a ridge line (Medial Axis).
        """
        minx, miny, maxx, maxy = footprint.bounds
        width, depth = maxx - minx, maxy - miny
        mid_x, mid_y = (minx + maxx) / 2, (miny + maxy) / 2
        
        # Mathematical ridge length: max(0, long - short)
        ridge_len = max(0.01, abs(width - depth))
        
        # Base wire
        base_wire = (
            cq.Workplane("XY")
            .workplane(offset=wall_h)
            .polyline(list(footprint.exterior.coords))
            .close()
            .wire()
            .val()
        )

        try:
            # ROBUST RIDGE: Loft to a very thin rectangle instead of a line
            # This is mathematically equivalent to a ridge but more stable in OCCT
            if width > depth:
                # Ridge along X
                ridge_wire = (
                    cq.Workplane("XY")
                    .workplane(offset=ridge_h)
                    .center(mid_x, mid_y)
                    .rect(ridge_len, 0.01) # Thin X-aligned ridge
                    .wire()
                    .val()
                )
            else:
                # Ridge along Y
                ridge_wire = (
                    cq.Workplane("XY")
                    .workplane(offset=ridge_h)
                    .center(mid_x, mid_y)
                    .rect(0.01, ridge_len) # Thin Y-aligned ridge
                    .wire()
                    .val()
                )
            
            roof = cq.Workplane("XY").add(base_wire).add(ridge_wire).toPending().loft()
            return model.union(roof)
        except Exception as e:
            logger.warning(f"Medial ridge loft failed: {e}. Falling back to point.")
            peak_wire = cq.Workplane("XY", origin=(mid_x, mid_y, ridge_h)).rect(0.01, 0.01).wire().val()
            roof = cq.Workplane("XY").add(base_wire).add(peak_wire).toPending().loft()
            return model.union(roof)

    def add_gable_roof(self, model, footprint, wall_h, ridge_h):
        """
        Add a gable roof by extruding a triangle.
        """
        minx, miny, maxx, maxy = footprint.bounds
        width, depth = maxx - minx, maxy - miny
        mid_x, mid_y = (minx + maxx) / 2, (miny + maxy) / 2
        
        try:
            if width > depth:
                roof = (
                    cq.Workplane("YZ", origin=(minx, mid_y, wall_h))
                    .polyline([(-depth/2, 0), (0, ridge_h - wall_h), (depth/2, 0)])
                    .close()
                    .extrude(width)
                )
            else:
                roof = (
                    cq.Workplane("XZ", origin=(mid_x, miny, wall_h))
                    .polyline([(-width/2, 0), (0, ridge_h - wall_h), (width/2, 0)])
                    .close()
                    .extrude(depth)
                )
            return model.union(roof)
        except Exception as e:
            logger.warning(f"Gable roof failed: {e}")
            return model

    def export(self, model, filename):
        if model:
            cq.exporters.export(model, filename)
            logger.info(f"Model exported to {filename}")
