import cadquery as cq
from loguru import logger
import numpy as np

class ModelBuilder:
    def __init__(self, wall_height=3.0, wall_thickness=0.2, eave_overhang=0.5):
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness
        self.eave_overhang = eave_overhang

    def build(self, topology):
        """
        Build Detailed 3D Model with Overhangs and Openings.
        """
        logger.info("Building Detailed Architectural Model.")
        
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

        # 3. INTELLIGENCE: Inject Openings
        total_w = topology.get('width', 10.0)
        for op in topology.get('openings', []):
            try:
                world_x = op['rel_x'] * total_w
                void = (
                    cq.Workplane("XZ", origin=(world_x, 0, op['z_level'] + op['h']/2))
                    .rect(op['w'], op['h'])
                    .extrude(self.wall_thickness * 3, both=True)
                )
                model = model.cut(void)
            except Exception as e:
                logger.warning(f"Failed to cut opening: {e}")

        # 4. Add Roof with Eaves
        sub_footprints = topology.get('sub_footprints', [])
        
        if len(sub_footprints) > 1:
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

        logger.info("Detailed 3D model generation complete.")
        return model

    def add_medial_hip_roof(self, model, footprint, wall_h, pitch, ridge_h):
        minx, miny, maxx, maxy = footprint.bounds
        width, depth = maxx - minx, maxy - miny
        mid_x, mid_y = (minx + maxx) / 2, (miny + maxy) / 2
        ridge_len = max(0.01, abs(width - depth))
        
        # Eave Wire: Offset outwards from footprint
        eave_wire = (
            cq.Workplane("XY", origin=(0, 0, wall_h))
            .polyline(list(footprint.exterior.coords))
            .close()
            .offset2D(self.eave_overhang)
            .wire()
            .val()
        )
        
        try:
            if width > depth:
                ridge_wire = cq.Workplane("XY", origin=(mid_x, mid_y, ridge_h)).rect(ridge_len, 0.01).wire().val()
            else:
                ridge_wire = cq.Workplane("XY", origin=(mid_x, mid_y, ridge_h)).rect(0.01, ridge_len).wire().val()
            
            roof = cq.Workplane("XY").add(eave_wire).add(ridge_wire).toPending().loft()
            return model.union(roof)
        except:
            peak = cq.Workplane("XY", origin=(mid_x, mid_y, ridge_h)).rect(0.01, 0.01).wire().val()
            return model.union(cq.Workplane("XY").add(eave_wire).add(peak).toPending().loft())

    def add_gable_roof(self, model, footprint, wall_h, ridge_h):
        minx, miny, maxx, maxy = footprint.bounds
        width, depth = maxx - minx, maxy - miny
        mid_x, mid_y = (minx + maxx) / 2, (miny + maxy) / 2
        
        # Gable Overhang adjustment
        oh = self.eave_overhang
        
        try:
            if width > depth:
                # Ridge along X, Triangle on YZ
                roof = (
                    cq.Workplane("YZ", origin=(minx - oh, mid_y, wall_h))
                    .polyline([(-depth/2 - oh, 0), (0, ridge_h - wall_h), (depth/2 + oh, 0)])
                    .close()
                    .extrude(width + 2*oh)
                )
            else:
                # Ridge along Y, Triangle on XZ
                roof = (
                    cq.Workplane("XZ", origin=(mid_x, miny - oh, wall_h))
                    .polyline([(-width/2 - oh, 0), (0, ridge_h - wall_h), (width/2 + oh, 0)])
                    .close()
                    .extrude(depth + 2*oh)
                )
            return model.union(roof)
        except:
            return model

    def export(self, model, filename):
        if model:
            cq.exporters.export(model, filename)
            logger.info(f"Model exported to {filename}")
