from shapely.ops import unary_union, polygonize
from shapely.geometry import LineString, Polygon
from loguru import logger
import numpy as np

class TopologyReconstructor:
    def __init__(self, pixels_per_metre=100):
        self.pixels_per_metre = pixels_per_metre

    def reconstruct(self, lines, vertices, curves, img_shape, directions=None):
        """
        Full topology reconstruction for Stage 4.
        """
        logger.info("Reconstructing topology from lines and curves.")
        
        # Check if we should use Perspective Mode
        if directions and len(directions.get('vertical', [])) > 2:
            return self.reconstruct_perspective(directions)

        # Fallback to Plan Mode
        return self.reconstruct_plan(lines, vertices, curves)

    def reconstruct_perspective(self, directions):
        """
        Estimate 3D box dimensions from perspective lines using the 12m rule.
        """
        # 1. Height from verticals (pixels)
        heights = [np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2) for l in directions['vertical']]
        avg_h_px = np.mean(heights) if heights else 300
        
        # 2. Width/Depth from perspective lines (pixels)
        # Use total span of segments in each direction
        def get_span(lines):
            if not lines: return 0
            all_pts = []
            for l in lines: all_pts.extend([(l[0], l[1]), (l[2], l[3])])
            all_pts = np.array(all_pts)
            return np.max(np.linalg.norm(all_pts[:, None] - all_pts, axis=2))

        w_px = get_span(directions['left_vp'])
        d_px = get_span(directions['right_vp'])
        
        # 3. Apply 12m Rule (PRD 4.4.4)
        # Longest horizontal dimension = 12.0m
        max_dim_px = max(w_px, d_px)
        if max_dim_px == 0: max_dim_px = 500
        
        px_per_m = max_dim_px / 12.0
        
        w_m = w_px / px_per_m
        d_m = d_px / px_per_m
        h_m = avg_h_px / px_per_m

        # Cap height to realistic ratio if it's too tall, or boost if too short
        max_dim_m = max(w_m, d_m)
        
        # 4. Height Sanity Check
        # If height is < 2.5m for a building > 5m wide, likely under-estimated due to perspective
        if max_dim_m > 5.0 and h_m < 2.5:
            logger.warning(f"Calculated height {h_m:.1f}m too low. Boosting to standard 3.0m storey.")
            h_m = 3.0
        
        # 5. Complexity Check (Internal Verticals) -> L-Shape Generation
        verticals = directions['vertical']
        w_px_min = min(l[0] for l in verticals) if verticals else 0
        w_px_max = max(l[0] for l in verticals) if verticals else 1
        
        internal_verticals = [l for l in verticals 
                              if (l[0] > w_px_min + (w_px_max - w_px_min) * 0.2) and 
                                 (l[0] < w_px_max - (w_px_max - w_px_min) * 0.2)]
        
        footprint = None
        
        if len(internal_verticals) > 1:
            logger.warning(f"Detected {len(internal_verticals)} internal vertical lines. Generating L-shaped footprint.")
            
            # Calculate Split X position (relative to width)
            avg_split_px = np.mean([l[0] for l in internal_verticals])
            split_ratio = (avg_split_px - w_px_min) / (w_px_max - w_px_min)
            
            # Construct L-Shape
            # Assume "Main Mass" is the larger side, "Wing" is the smaller side
            # Wing depth = 60% of total depth
            
            split_m = w_m * split_ratio
            wing_depth_m = d_m * 0.6
            
            if split_ratio < 0.5:
                # Wing is on the Left (smaller section), Main on Right
                # This creates a "b" shape or inverted L
                # Points: (0,0) -> (split, 0) -> (split, full_d) -> (w, full_d) -> (w, 0) ... wait
                # Let's simplify: Union of two rects.
                # Rect 1 (Wing): 0 to split, 0 to wing_depth
                # Rect 2 (Main): split to w, 0 to d
                # ... actually, looking at sketch, let's assume "Front" is y=0.
                
                # Polygon coordinates (Counter-Clockwise)
                coords = [
                    (0, 0), (split_m, 0), (split_m, d_m - wing_depth_m), 
                    (w_m, d_m - wing_depth_m), (w_m, d_m), (0, d_m)
                ]
                # Wait, that shape is weird. Let's do Union.
                p1 = Polygon([(0, 0), (split_m, 0), (split_m, wing_depth_m), (0, wing_depth_m)])
                p2 = Polygon([(split_m, 0), (w_m, 0), (w_m, d_m), (split_m, d_m)])
                footprint = unary_union([p1, p2])
            else:
                # Wing is on the Right
                # Rect 1 (Main): 0 to split, 0 to d
                # Rect 2 (Wing): split to w, 0 to wing_depth
                p1 = Polygon([(0, 0), (split_m, 0), (split_m, d_m), (0, d_m)])
                p2 = Polygon([(split_m, 0), (w_m, 0), (w_m, wing_depth_m), (split_m, wing_depth_m)])
                footprint = unary_union([p1, p2])
        else:
            # Simple Box
            footprint = Polygon([(0,0), (w_m, 0), (w_m, d_m), (0, d_m)])
        
        topology = {
            'view_type': 'perspective',
            'footprint': footprint,
            'height': h_m,
            'width': w_m,
            'depth': d_m,
            'scale': px_per_m
        }
        logger.info(f"Perspective Mode (12m rule): Estimated {w_m:.1f}m x {d_m:.1f}m x {h_m:.1f}m box.")
        return topology

    def reconstruct_plan(self, lines, vertices, curves):
        # 1. Snap line endpoints to nearest vertices to close gaps
        snapped_lines = self.snap_endpoints_to_vertices(lines, vertices, threshold=30)
        extended_lines = self.extend_lines(snapped_lines, extension=5)

        # 2. Floor Plan Polygon Detection
        shapely_lines = [LineString([(l[0], l[1]), (l[2], l[3])]) for l in extended_lines]
        merged = unary_union(shapely_lines)
        polygons = list(polygonize(merged))
        
        if not polygons:
            logger.warning("No polygons detected by polygonize. Falling back to convex hull of vertices.")
            all_pts = []
            for l in extended_lines:
                all_pts.extend([(l[0], l[1]), (l[2], l[3])])
            if len(all_pts) < 3:
                return {}
            footprint = Polygon(all_pts).convex_hull
            interior_rooms = []
        else:
            footprint = max(polygons, key=lambda p: p.area)
            interior_rooms = [p for p in polygons if p != footprint]

        minx, miny, maxx, maxy = footprint.bounds
        origin = (minx, maxy) 

        normalized_footprint = self.normalize_polygon(footprint, origin)
        normalized_rooms = [self.normalize_polygon(p, origin) for p in interior_rooms]

        return {
            'view_type': 'plan',
            'footprint': normalized_footprint,
            'rooms': normalized_rooms,
            'curves': curves,
            'origin_px': origin,
            'scale': self.pixels_per_metre
        }

    def snap_endpoints_to_vertices(self, lines, vertices, threshold=20):
        if not vertices: return lines
        v_array = np.array(vertices)
        new_lines = []
        for x1, y1, x2, y2 in lines:
            p1, p2 = np.array([x1, y1]), np.array([x2, y2])
            d1 = np.linalg.norm(v_array - p1, axis=1)
            if np.min(d1) < threshold: p1 = v_array[np.argmin(d1)]
            d2 = np.linalg.norm(v_array - p2, axis=1)
            if np.min(d2) < threshold: p2 = v_array[np.argmin(d2)]
            new_lines.append((p1[0], p1[1], p2[0], p2[1]))
        return new_lines

    def extend_lines(self, lines, extension=5):
        extended = []
        for x1, y1, x2, y2 in lines:
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length == 0: continue
            ux, uy = dx / length, dy / length
            extended.append((x1 - ux * extension, y1 - uy * extension, x2 + ux * extension, y2 + uy * extension))
        return extended

    def normalize_polygon(self, poly, origin):
        ox, oy = origin
        coords = [( (x - ox) / self.pixels_per_metre, (oy - y) / self.pixels_per_metre ) for x, y in poly.exterior.coords]
        return Polygon(coords)
