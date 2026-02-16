import cadquery as cq
from loguru import logger
import numpy as np

class ModelBuilder:
    def __init__(self, wall_height=3.0, wall_thickness=0.2):
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness

    def build(self, topology):
        """
        Build 3D model using composite volumes for robust geometry.
        """
        logger.info("Building Composite 3D Model.")
        
        if not topology or 'footprint' not in topology:
            return None

        footprint = topology['footprint']
        wall_h = topology.get('height', self.wall_height)
        pitch = topology.get('pitch', 0.4)
        
        # 1. Extrude Footprint
        model = (
            cq.Workplane("XY")
            .polyline(list(footprint.exterior.coords))
            .close()
            .extrude(wall_h)
        )

        # 2. Shell
        model = model.shell(-self.wall_thickness)

        # 3. INTELLIGENCE: Inject Openings (Windows/Doors)
        total_w = topology.get('width', 10.0)
        for op in topology.get('openings', []):
            try:
                # Map relative X to world X (Front wall projection)
                world_x = op['rel_x'] * total_w
                void = (
                    cq.Workplane("XZ", origin=(world_x, 0, op['z_level'] + op['h']/2))
                    .rect(op['w'], op['h'])
                    .extrude(self.wall_thickness * 3, both=True)
                )
                model = model.cut(void)
            except Exception as e:
                logger.warning(f"Failed to cut opening: {e}")

        # 4. Add Roof
        sub_footprints = topology.get('sub_footprints', [])
        
        if len(sub_footprints) > 1:
            logger.info(f"Composite parts: {len(sub_footprints)}")
            # Calculate global ridge height from main block
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
            mbx = footprint.bounds
            ridge_h = wall_h + (min(mbx[2]-mbx[0], mbx[3]-mbx[1]) * pitch)
            model = self.add_medial_hip_roof(model, footprint, wall_h, pitch, ridge_h)

        logger.info("3D model generation complete.")
        return model

    def add_medial_hip_roof(self, model, footprint, wall_h, pitch, ridge_h):
        minx, miny, maxx, maxy = footprint.bounds
        width, depth = maxx - minx, maxy - miny
        mid_x, mid_y = (minx + maxx) / 2, (miny + maxy) / 2
        ridge_len = max(0.01, abs(width - depth))
        
        base_wire = cq.Workplane("XY", origin=(0, 0, wall_h)).polyline(list(footprint.exterior.coords)).close().wire().val()
        
        try:
            # Use thin rectangle for robust ridge
            if width > depth:
                ridge_wire = cq.Workplane("XY", origin=(mid_x, mid_y, ridge_h)).rect(ridge_len, 0.01).wire().val()
            else:
                ridge_wire = cq.Workplane("XY", origin=(mid_x, mid_y, ridge_h)).rect(0.01, ridge_len).wire().val()
            
            roof = cq.Workplane("XY").add(base_wire).add(ridge_wire).toPending().loft()
            return model.union(roof)
        except:
            peak = cq.Workplane("XY", origin=(mid_x, mid_y, ridge_h)).rect(0.01, 0.01).wire().val()
            return model.union(cq.Workplane("XY").add(base_wire).add(peak).toPending().loft())

    def add_gable_roof(self, model, footprint, wall_h, ridge_h):
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
        except:
            return model

    def export(self, model, filename):
        if model:
            cq.exporters.export(model, filename)
            logger.info(f"Model exported to {filename}")
