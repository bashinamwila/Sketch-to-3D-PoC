import cadquery as cq
from loguru import logger
import numpy as np

class ModelBuilder:
    def __init__(self, wall_height=3.0, wall_thickness=0.2, eave_overhang=0.5, roof_thickness=0.2):
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness
        self.eave_overhang = eave_overhang
        self.roof_thickness = roof_thickness

    def build(self, topology):
        """
        Build a high-fidelity architectural model with robust solid geometry.
        """
        logger.info("Building Intelligent Architectural Model.")
        
        if not topology or 'footprint' not in topology:
            return None

        footprint = topology['footprint']
        wall_h = topology.get('height', self.wall_height)
        pitch = topology.get('pitch', 0.4)
        
        # 1. Solid Walls
        model = (
            cq.Workplane("XY")
            .polyline(list(footprint.exterior.coords))
            .close()
            .extrude(wall_h)
            .shell(-self.wall_thickness)
        )

        # 2. INTELLIGENCE: Inject Openings (Multi-Face)
        sub_footprints = topology.get('sub_footprints', [])
        for op in topology.get('openings', []):
            try:
                parent_label = op.get('parent_label', 'main')
                part = next((p for p in sub_footprints if p['label'] == parent_label), sub_footprints[0])
                mbx = part['poly'].bounds
                
                # Calculate world position on the front face of this specific part
                world_x = mbx[0] + (op['rel_x'] * (mbx[2] - mbx[0]))
                front_y = mbx[1]
                
                void = (
                    cq.Workplane("XZ", origin=(world_x, front_y, op['z_level'] + op['h']/2))
                    .rect(op['w'], op['h'])
                    .extrude(self.wall_thickness * 3, both=True)
                )
                model = model.cut(void)
            except Exception as e:
                logger.warning(f"Failed to cut opening: {e}")

        # 3. Add Roof Parts
        
        # Consistent global ridge height
        main_part = next((p for p in sub_footprints if p['label'] == 'main'), sub_footprints[0])
        mbx = main_part['poly'].bounds
        global_ridge_h = wall_h + (min(mbx[2]-mbx[0], mbx[3]-mbx[1]) * pitch)

        for part in sub_footprints:
            poly = part['poly']
            rtype = part['roof_type']
            face_pref = part.get('gable_face', 'auto')
            
            if rtype == 'gable':
                model = self.add_solid_gable_roof(model, poly, wall_h, global_ridge_h, face_pref)
            else:
                model = self.add_solid_hip_roof(model, poly, wall_h, pitch, global_ridge_h)

        logger.info("Detailed architectural generation complete.")
        return model

    def add_solid_hip_roof(self, model, footprint, wall_h, pitch, ridge_h):
        """
        Sharp hip roof created as a solid volume (bottom face + top ridge).
        """
        minx, miny, maxx, maxy = footprint.bounds
        width, depth = maxx - minx, maxy - miny
        mid_x, mid_y = (minx + maxx) / 2, (miny + maxy) / 2
        ridge_len = max(0.01, abs(width - depth))
        
        # 1. Base Wire (at wall top)
        base_wire = (
            cq.Workplane("XY", origin=(0, 0, wall_h))
            .polyline(list(footprint.exterior.coords))
            .close()
            .offset2D(self.eave_overhang)
            .wire().val()
        )
        
        # 2. Ridge Wire (at peak)
        if width > depth:
            ridge_wire = cq.Workplane("XY", origin=(mid_x, mid_y, ridge_h)).rect(ridge_len, 0.01).wire().val()
        else:
            ridge_wire = cq.Workplane("XY", origin=(mid_x, mid_y, ridge_h)).rect(0.01, ridge_len).wire().val()
            
        try:
            # Loft base to ridge to create a solid mass
            roof_solid = cq.Workplane("XY").add(base_wire).add(ridge_wire).toPending().loft()
            return model.union(roof_solid)
        except Exception as e:
            logger.warning(f"Solid hip loft failed: {e}")
            return model

    def add_solid_gable_roof(self, model, footprint, wall_h, ridge_h, face_pref='auto'):
        """
        Detailed solid Gable roof.
        """
        minx, miny, maxx, maxy = footprint.bounds
        width, depth = maxx - minx, maxy - miny
        mid_x, mid_y = (minx + maxx) / 2, (miny + maxy) / 2
        oh = self.eave_overhang
        
        try:
            if face_pref == 'front' or (face_pref == 'auto' and width > depth):
                # Front-facing gable end
                # Triangle on XZ plane, extruded along Depth
                roof = (
                    cq.Workplane("XZ", origin=(mid_x, miny - oh, wall_h))
                    .polyline([(-width/2 - oh, 0), (0, ridge_h - wall_h), (width/2 + oh, 0)])
                    .close()
                    .extrude(depth + 2*oh)
                )
            else:
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
