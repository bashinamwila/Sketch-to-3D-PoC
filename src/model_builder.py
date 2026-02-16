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
        # Check for composite decomposition (L-shape)
        sub_footprints = topology.get('sub_footprints', [])
        
        if len(sub_footprints) > 1:
            logger.info(f"Composite footprint detected ({len(sub_footprints)} parts). Using Composite Hip Roof.")
            model = self.add_composite_hip_roof(model, sub_footprints, wall_h)
        else:
            # Simple shape
            try:
                model = self.add_tapered_hip_roof(model, footprint, wall_h)
            except Exception as e:
                logger.warning(f"Tapered roof failed: {e}. Using Flat Roof fallback.")
                model = self.add_flat_roof_parapet(model, footprint, wall_h)

        logger.info("3D model generation complete.")
        return model

    def add_composite_hip_roof(self, model, sub_footprints, wall_h):
        """
        Add a hip roof by unioning hip roofs for each sub-rectangle.
        """
        for rect in sub_footprints:
            try:
                # Reuse the tapered roof logic for each part
                # Note: add_tapered_hip_roof performs a Union. 
                # So calling it sequentially adds each roof part to the model.
                model = self.add_tapered_hip_roof(model, rect, wall_h)
            except Exception as e:
                logger.warning(f"Failed to add roof part: {e}")
        return model

    def add_tapered_hip_roof(self, model, footprint, wall_h):
        """
        Create a hip roof using tapered extrusion.
        """
        # Create a workplane at wall height
        roof_plane = cq.Workplane("XY").workplane(offset=wall_h)
        
        # Extrude the footprint with a taper (angle in degrees)
        # 45 degrees is a standard steep roof.
        # We extrude by a height proportional to the building size
        minx, miny, maxx, maxy = footprint.bounds
        roof_h = min(maxx - minx, maxy - miny) * 0.4
        
        roof = (
            roof_plane
            .polyline(list(footprint.exterior.coords))
            .close()
            .extrude(roof_h, taper=45) # 45 deg taper creates hip effect
        )
        
        return model.union(roof)

    def add_flat_roof_parapet(self, model, footprint, wall_h):
        """
        Add a flat roof with a parapet wall for complex shapes.
        """
        # 1. Create the roof slab
        roof_slab = (
            cq.Workplane("XY")
            .workplane(offset=wall_h - 0.2)
            .polyline(list(footprint.exterior.coords))
            .close()
            .extrude(0.2)
        )
        
        # 2. Create the parapet (rim)
        # Extrude a thin shell upwards
        parapet = (
            cq.Workplane("XY")
            .workplane(offset=wall_h)
            .polyline(list(footprint.exterior.coords))
            .close()
            .extrude(0.5) # 0.5m tall parapet
        )
        # Hollow it out to make it a rim
        parapet = parapet.faces(">Z").shell(-0.2)
        
        return model.union(roof_slab).union(parapet)

    def add_hip_roof(self, model, footprint, wall_h):
        """
        Add a hip roof (slopes on all sides).
        """
        minx, miny, maxx, maxy = footprint.bounds
        width = maxx - minx
        depth = maxy - miny
        
        # Determine ridge length and position
        ridge_h = wall_h + min(width, depth) * 0.4 # 40% pitch
        inset = min(width, depth) / 2.0
        
        # Ridge line points
        ridge_y_start = miny + inset
        ridge_y_end = maxy - inset
        
        # Create the roof solid by lofting
        # Bottom wire (eaves)
        eaves = cq.Workplane("XY").polyline(list(footprint.exterior.coords)).close().wire().val()
        eaves = eaves.translate((0, 0, wall_h))
        
        # Ridge (top wire/edge)
        # Simplify: create a small rectangle or line at the top
        ridge_len = max(0.1, depth - width) if depth > width else max(0.1, width - depth)
        
        # Build a pyramid-like shape using loft
        # Note: CadQuery loft can be tricky. Fallback to simple pyramid if needed.
        try:
            # Create WIRES for lofting
            # Top small rectangle (ridge)
            top_wire = (
                cq.Workplane("XY")
                .workplane(offset=ridge_h)
                .rect(max(0.1, width-depth if width>depth else 0.1), 
                      max(0.1, depth-width if depth>width else 0.1))
                .wire()
                .val()
            )
            
            # Base rectangle (eaves)
            base_wire = (
                cq.Workplane("XY")
                .workplane(offset=wall_h)
                .polyline(list(footprint.exterior.coords))
                .close()
                .wire()
                .val()
            )
            
            # Loft between the two wires using a fresh workplane
            roof = cq.Workplane("XY").add(base_wire).add(top_wire).toPending().loft()
            return model.union(roof)
        except Exception as e:
            logger.warning(f"Hip roof failed: {e}. Fallback to Gable.")
            return self.add_gable_roof(model, footprint, wall_h)

    def add_gable_roof(self, model, footprint, wall_h):
        """
        Add a simple gable roof on top of the model.
        """
        minx, miny, maxx, maxy = footprint.bounds
        midx = (minx + maxx) / 2
        ridge_height = wall_h + 1.5
        
        # Create a triangular prism for the roof
        # We'll define it on the XZ plane and extrude along Y
        roof = (
            cq.Workplane("XZ")
            .workplane(offset=miny)
            .polyline([(minx, wall_h), (midx, ridge_height), (maxx, wall_h)])
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
