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
        Intelligence Layer: Construct an integrated architectural solid.
        """
        logger.info("Building Integrated 3D Model.")
        
        if not topology or 'footprint' not in topology:
            return None

        footprint = topology['footprint']
        wall_h = topology.get('height', self.wall_height)
        pitch = topology.get('pitch', 0.4)
        
        # 1. Unified Wall Shell
        # We extrude the entire L-shape at once for a watertight union
        model = (
            cq.Workplane("XY")
            .polyline(list(footprint.exterior.coords))
            .close()
            .extrude(wall_h)
            .shell(-self.wall_thickness)
        )

        # 2. INTELLIGENCE: Precision Opening Injection
        sub_footprints = topology.get('sub_footprints', [])
        px_min, px_max = topology.get('global_px_bounds', (0, 1000))
        px_width = px_max - px_min
        px_per_m = topology.get('scale', 100.0)

        for op in topology.get('openings', []):
            try:
                parent_label = op.get('parent_label', 'main')
                part = next((p for p in sub_footprints if p['label'] == parent_label), sub_footprints[0])
                mbx = part['poly'].bounds
                
                # Precise World X Mapping
                world_x = (op['rel_x'] * px_width) / px_per_m
                front_y = mbx[1]
                
                void = (
                    cq.Workplane("XZ", origin=(world_x, front_y, op['z_level'] + op['h']/2))
                    .rect(op['w'], op['h'])
                    .extrude(self.wall_thickness * 4, both=True)
                )
                model = model.cut(void)

                # INTELLIGENCE: Mullion Injection
                if op.get('type') == 'double_window':
                    mullion = (
                        cq.Workplane("XZ", origin=(world_x, front_y, op['z_level'] + op['h']/2))
                        .rect(0.05, op['h'])
                        .extrude(self.wall_thickness, both=True)
                    )
                    model = model.union(mullion)
            except:
                continue

        # 3. INTELLIGENCE: Integrated Roof System
        # We generate the roof parts but ensure they use a shared global ridge height
        # and perfectly align with the wall tops.
        main_part = next((p for p in sub_footprints if p['label'] == 'main'), sub_footprints[0])
        mbx = main_part['poly'].bounds
        global_ridge_h = wall_h + (min(mbx[2]-mbx[0], mbx[3]-mbx[1]) * pitch)

        for part in sub_footprints:
            poly = part['poly']
            rtype = part['roof_type']
            face_pref = part.get('gable_face', 'auto')
            
            if rtype == 'gable':
                model = self.add_gable_roof(model, poly, wall_h, global_ridge_h, face_pref)
            else:
                model = self.add_medial_hip_roof(model, poly, wall_h, global_ridge_h)

        logger.info("Integrated architectural generation complete.")
        return model

    def add_medial_hip_roof(self, model, footprint, wall_h, ridge_h):
        """
        Sharp Hip Roof with zero-width ridge logic.
        """
        minx, miny, maxx, maxy = footprint.bounds
        width, depth = maxx - minx, maxy - miny
        mid_x, mid_y = (minx + maxx) / 2, (miny + maxy) / 2
        
        # Skeleton Ridge Length
        ridge_len = max(0.001, abs(width - depth))
        
        base_wire = (
            cq.Workplane("XY", origin=(0, 0, wall_h))
            .polyline(list(footprint.exterior.coords))
            .close()
            .offset2D(self.eave_overhang)
            .wire().val()
        )
        
        try:
            # Create ridge as a nearly-zero-width rectangle for OCCT stability
            # but scaled to ridge_len for mathematical correctness
            if width > depth:
                ridge_wire = cq.Workplane("XY", origin=(mid_x, mid_y, ridge_h)).rect(ridge_len, 0.001).wire().val()
            else:
                ridge_wire = cq.Workplane("XY", origin=(mid_x, mid_y, ridge_h)).rect(0.001, ridge_len).wire().val()
            
            roof = cq.Workplane("XY").add(base_wire).add(ridge_wire).toPending().loft()
            return model.union(roof)
        except:
            peak = cq.Workplane("XY", origin=(mid_x, mid_y, ridge_h)).rect(0.001, 0.001).wire().val()
            return model.union(cq.Workplane("XY").add(base_wire).add(peak).toPending().loft())

    def add_gable_roof(self, model, footprint, wall_h, ridge_h, face_pref):
        """
        Gable Roof with sharp ridge and correct protrusion alignment.
        """
        minx, miny, maxx, maxy = footprint.bounds
        width, depth = maxx - minx, maxy - miny
        mid_x, mid_y = (minx + maxx) / 2, (miny + maxy) / 2
        oh = self.eave_overhang
        
        try:
            if face_pref == 'front':
                # Front-facing triangle (perpendicular to Y axis)
                roof = (
                    cq.Workplane("XZ", origin=(mid_x, miny - oh, wall_h))
                    .polyline([(-width/2 - oh, 0), (0, ridge_h - wall_h), (width/2 + oh, 0)])
                    .close()
                    .extrude(depth + oh) # Extrude back into the house
                )
            else:
                # Standard side-facing
                roof = (
                    cq.Workplane("YZ", origin=(minx - oh, mid_y, wall_h))
                    .polyline([(-depth/2 - oh, 0), (0, ridge_h - wall_h), (depth/2 + oh, 0)])
                    .close()
                    .extrude(width + 2*oh)
                )
            return model.union(roof)
        except:
            return model

    def export(self, model, filename):
        if model:
            cq.exporters.export(model, filename)
            logger.info(f"Model exported to {filename}")
